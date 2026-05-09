# Design decisions

Why each non-obvious choice was made, and what was tried that didn't work.
For *what* the system looks like, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Subprocess-per-swap (vs in-process model swap)

**Decision:** every swap is `engine.shutdown()` (kills the EngineCore
subprocess) followed by `from_vllm_config()` (spawns a fresh one). The
ModelManager only ever owns at most one engine.

**Why:** vLLM V1 single-process mode leaks ~0.47 GB per swap of internal
CUDA library workspaces (NCCL init, Flash Attention scratch, cuBLAS
handles) — there is no Python reference path to free them, the leak comes
from inside CUDA libraries through PyTorch's allocator. Over 50+ swaps
you run out of headroom. Killing the subprocess and letting the OS reclaim
its address space drops drift to **~0 MiB across 50+ swaps**.

**Cost:** ~10–17 s of subprocess imports per swap. Hidden behind the
previous model's serving by the pipelined-acquire flow (see
ARCHITECTURE.md, "Swap lifecycle").

**Alternative tried:** elaborate cleanup chasing the leak — clearing
torch/dynamo/inductor caches, cuDNN/cuBLAS workspaces, vLLM's
WorkspaceManager, ForwardContext, UBatchContext, aggressive
`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:...`,
`CUBLAS_WORKSPACE_CONFIG=:0:0`. None of it brought drift below ~0.5 GB
per swap. The leak is in CUDA-library-allocated memory that has no
Python reference graph to walk.

---

## 80 GB hugetlbfs pool (vs cudaHostAlloc, vs filesystem page cache, vs runai_streamer)

**Decision:** parent mmaps a 80 GB hugetlbfs file (`/mnt/hugetlbfs/blitz_pool`,
backed by 80 × 1 GB hugepages reserved at boot), reads safetensors shards
into it via parallel `preadv()`, and the EngineCore subprocess re-mmaps the
same file and `cudaHostRegister`s it in its own CUDA context. Weight load
then becomes a `cudaMemcpy(host→device)` from registered pinned memory —
PCIe 5.0 wire speed, ~44–48 GB/s.

**Why hugetlbfs (vs cudaHostAlloc):**
| Approach | Register speed (80 GB) |
|---|---|
| `cudaHostAlloc` (anonymous pinned) | ~30 s |
| `cudaHostRegister` on hugetlbfs file | ~1.4 s |

`cudaHostAlloc` walks 4 KB pages one at a time through the kernel's pin
machinery. `cudaHostRegister` on a 1 GB-page-backed file pins one
hugepage at a time — 250,000× fewer syscalls.

**Why a shared file (vs the parent doing the register):** if the parent
`cudaHostRegister`s the 80 GB region itself, that 80 GB shows up in the
parent's UVA accounting and *eats GPU virtual address space*. Subsequent
EngineCore subprocesses then see only ~12 GB of "free" GPU memory
according to `cudaMemGetInfo`, and refuse to start. So: parent only
mmaps; subprocess does the register in its own CUDA context.

**Why 80 GB and not 95 GB:** the biggest model in the registry is
~75 GB on disk (Qwen3.5-122B-A10B-NVFP4, gpt-oss-120b, Qwen3-Coder-Next).
80 GB has slight headroom; bigger pools push us over the 124 GB system-RAM
budget once you add the EngineCore RSS, FastAPI RSS, OS page cache for
the HF cache, and CUDA's own shared-memory allocations during model init
(which can take 17 GB on a 65 GB model).

**Why power-of-2 pool size:** PyTorch rounds pinned allocations to the
next power of 2 (PyTorch issue #150517). 5 × 16 GB chunks → 0% overhead;
non-power-of-2 chunks pay 60% shmem overhead.

**Alternative tried — page cache warming:** read shards once to warm OS
page cache, let vLLM's stock loader read from there. Worked, but capped
at ~7 GB/s (small reads, no DMA from pinned pages) vs 44 GB/s through
the registered hugetlbfs pool. Kept around as a fallback for hardware
without 1 GB hugepages.

**Alternative tried — runai_streamer / fastsafetensors:** vLLM ships
support for both. They optimize disk → GPU streaming but don't help
across swaps because each load re-pays the per-shard pinned-buffer
setup. The hugetlbfs pool is the only path that amortizes the registration
across all future swaps.

---

## Drain-then-switch (vs eager-switch)

**Decision:** when a request for model B arrives while model A is active,
the dispatcher does NOT switch immediately. It commits the next-model
decision (B), but keeps draining `queues[A]` until the active queue is
empty AND `in_flight == 0`. New A requests arriving *after* B's request
still get served on A.

**Why:** without this, eager-switch causes thrashing. Concrete
counter-example without drain-then-switch:

```
A1, A2 in flight on A
B1 arrives → swap to B
A3 arrives during swap → queues, can't be served
swap completes, B1 runs
A3 wakes up, sees active=B → triggers swap back to A
2 swaps for what could have been 1
```

Verified with `tests/queue_stress` (now in git history): A2 fired *after*
B1 was queued, but A2 finished in 1.2 s on the still-active model A
engine, before any model A teardown happened.

**Cost:** B can starve indefinitely if A keeps getting traffic. Acceptable
trade for our use case (interactive agent workloads, not high-throughput
serving). If starvation becomes an issue, add a queue-age cap — drain
until `now - oldest_B_request > N seconds`.

**Why per-model FIFOs (vs single global FIFO):** with a single FIFO
ordered by arrival time, the dispatcher couldn't tell B and C apart from
A's own backlog. Per-model FIFOs let the dispatcher say "oldest waiter
in any non-active queue is the next model to load," which is the right
fairness signal across model groups.

---

## Pipelined acquire (start spawn before drain, not after)

**Decision:** the moment the dispatcher commits to a next model, it starts
both `pool_task` (parent disk read) and `spawn_task` (subprocess imports
vLLM, hits GPU barrier, waits) — *before* the active queue has drained.

**Why:** subprocess Python imports take ~10–17 s even with the
torch.compile cache warm. If we wait until the active queue drains before
forking, that 10–17 s is fully visible to the user as "swap latency." If
we fork immediately, the subprocess is parked at the GPU barrier with
weights ready in pool by the time the active model finishes — only the
weight transfer (1 s) + KV cache profiling (4 s) + CUDA graph capture
(5 s) is left as visible work.

Concretely: ~35 s of work hidden inside a 50 K-token decode on the
previous model, vs an extra 35 s tacked on after.

**Why it's safe to start the next model's pool load while the active model
is still using its GPU weights:** the active model loaded its weights
pool→GPU at startup and never reads pool again. Pool memory is dead to
it. Overwriting pool with the next model's bytes is invisible to the
active engine.

**Why not also overlap with another swap:** the dispatcher's
`if swap_task is None` gate ensures only one swap at a time. Two
overlapping swaps would risk two subprocesses both wanting the GPU, and
debugging that race is not worth the small additional savings.

---

## GPU barrier in the worker (vs holding `from_vllm_config` until GPU is free)

**Decision:** patch `vllm/v1/worker/gpu_worker.py` to wait on
`BLITZ_GPU_GO_FILE` before its first CUDA call. The parent only touches
that file after `_unload()` confirms `nvidia-smi` shows ≤ 2 GiB used.

**Why this and not "just delay the spawn":** if we delayed the spawn
itself, we'd lose the pipelining — the subprocess wouldn't be importing
vLLM during the active model's serving. The barrier lets imports +
parallel_state + tokenizer + config parsing all happen in parallel with
the old model's GPU usage, but blocks the FIRST CUDA call (and therefore
all GPU memory allocation) until VRAM is verifiably free.

**Why poll `nvidia-smi` (vs trust `engine.shutdown()`):** subprocess exit
is async. After `engine.shutdown()` returns in the parent, the child
process may still be unwinding for hundreds of milliseconds, and the
NVIDIA driver may take longer still to reclaim memory. A fixed
`asyncio.sleep(2.5)` was the original implementation — it failed
intermittently on big models. `nvidia-smi` polling is the only robust
"VRAM actually freed" signal we have without using NVML directly (which
would be a dep we don't currently want).

**Why a file barrier (vs ZMQ / pipe / shared mem):** simplicity. The
subprocess inherits env at fork; `BLITZ_GPU_GO_FILE` is a per-swap unique
path; touching the file is one syscall in the parent; checking is one
`os.path.exists` in the child. Total complexity: ~10 lines patched into
vLLM. ZMQ would mean a vLLM-side listener and serialization round-trip.

---

## MM-warmup skip (monkey-patch in `server.py`)

**Decision:** at module load, replace
`vllm.renderers.base.BaseRenderer._warmup_mm_processor` with a no-op.

**Why:** read of `vllm/renderers/base.py`: the warmup runs
`processor.apply(dummy_inputs)` on a fake image (or audio), then
**explicitly clears the resulting cache** with `clear_mm_cache` /
`_clear_processor_cache`. So 5–12 s of dummy preprocessing per cold load
is throwaway compute. The only retained state is Python module-level lazy
imports (PIL, torchvision transforms, encoder-kernel JIT) — which would
also be triggered by the first real MM request anyway.

**Cost:** first MM request after a cold load pays ~1–2 s for those lazy
imports. Subsequent MM requests are unaffected. Verified end-to-end with
single-image, multi-image, and video input across 5 multimodal models.

**Why monkey-patch (vs config flag):** vLLM has no flag to disable MM
warmup. The renderer's `warmup` method is invoked unconditionally from
`OpenAIServingChat.__init__` via `init_app_state`. A 1-line
class-attribute reassignment in `server.py` runs before any
`OpenAIServingChat` is constructed and is the smallest possible
intervention.

---

## Profile cache (`blitzinfer/loader/profile_cache.py`)

**Decision:** persist `num_gpu_blocks` per
`(model, max_model_len, gpu_util, max_num_seqs, vllm_version, driver_version, gpu_name, extra_args_tuple)`
to `~/.cache/blitzinfer/profile/<sha16>.json`. On subsequent loads, pass
`--num-gpu-blocks-override <cached>` to skip vLLM's profile forward pass.

**Why:** vLLM's profile pass takes ~4 s on cold load (forward through the
model with `max_num_batched_tokens` to measure peak activation memory,
from which it derives the available KV cache size). For our use case the
answer is deterministic across loads — same model, same context length,
same util target → same num_gpu_blocks. Caching it saves 4 s per swap on
cache hit.

**Why these specific cache key components:** any one of them changing can
shift the answer. vLLM version changes activation layouts; driver version
shifts internal allocator behavior; GPU name (in case of hardware swap)
changes the available memory; the model + max_model_len + util are the
direct inputs to the calculation; `extra_args_tuple` covers
`--hf-overrides` (e.g. rope_scaling) which changes model shape.

**Cost:** ~4 s slower on first load of a new (model, config) combo.
Self-healing: after one load, the cache is warm forever (until any of
the keys changes).

---

## Per-model env var leak (known issue, not yet fixed)

**Observation:** vLLM 0.20.1 includes the full `os.environ` in
`cache_key_factors.json` for the torch.compile cache. `_spawn_engine`
sets per-model env vars (e.g. `VLLM_MXFP4_USE_MARLIN=1` for gpt-oss-120b)
via `os.environ[k] = v` in the parent — and never unsets them. Once
gpt-oss has been loaded once, every subsequent model's compile cache key
is contaminated by `VLLM_MXFP4_USE_MARLIN=1`, and depending on load order
across server restarts, the cache hash flips between two values for the
same model. When it flips, you eat a 25–30 s recompile.

**Fix (not yet done):** scope per-model env to the subprocess only — pass
via `subprocess.Popen(env=…)` instead of mutating the parent's
`os.environ`. Requires a deeper change to vLLM's `AsyncLLM.from_vllm_config`
spawn path (it currently inherits parent env). Workaround for now: pin
the load order so the cache stays consistent.

---

## What was tried and reverted

### `fork` worker multiproc method

**Hypothesis:** `VLLM_WORKER_MULTIPROC_METHOD=fork` skips the 10–17 s
import gap because the child inherits the parent's already-imported vLLM.

**Result:** worked for small models (qwen2.5-7b: −5.2 s; kimi-vl: −10.0 s)
but regressed two MoE models by +12–13 s (qwen3.6-35b-a3b, qwen3.6-27b).
Hypothesis for the regression: copy-on-write overhead — the qwen3_5_moe
init writes to many pages during construction (256K context, 42–66 shards,
many CUDA graph capture sizes), each first write forces a page-table
update and per-page memcpy from the shared parent copy. When that
exceeds the imports-saved cost, fork loses.

**Decision:** reverted to `spawn`. Net runtime across all 8 models was
identical (~927 s); spawn is more predictable.

**Future direction:** a *dedicated spawn pool* — pre-spawn 1–2 idle
Python processes that have already imported vLLM at module level, sitting
on a stdin pipe waiting for `(model_repo, args)`. On swap, parent picks
a warm worker and sends the config. Unlike fork, no CoW issues; unlike
on-demand spawn, the import cost is paid once ahead-of-time. Cost:
~3–5 GB RAM per warm worker. Estimated saving: 10–15 s on every swap.
Not yet implemented.

### `forkserver` multiproc method

vLLM 0.20.1's `envs.py` validator rejects it (`spawn` and `fork` are the
only allowed values), even though `api_server.py` has a code path for it.

### Cleaning up the per-swap drift

See "Subprocess-per-swap" above. Hours of investigation; nothing worked.

---

## Things explicitly NOT done

- **Tensor parallelism / pipeline parallelism.** Single GPU. The whole
  point of this project is rapid switching on one card.
- **Continuous batching across models.** Impossible — different models
  have different KV cache shapes, tokenizers, attention kernels.
- **Speculative pre-loading of likely-next models.** The user's rule:
  "no predictive prefetch, only facts based work." The pipelining only
  starts after a real request for a different model arrives.
- **Two models resident on the GPU simultaneously.** RTX PRO 6000 has
  96 GB, several models in the registry are 60–80 GB. Two-resident
  doesn't fit. Even if it did, vLLM's KV cache profiling assumes
  exclusive use.
- **Cross-process tensor sharing via CUDA IPC.** Considered for the pool;
  rejected because the subprocess `cudaHostRegister` approach is simpler
  and fast enough.

---

## Things that would be nice to add

- **Warm spawn pool** (see fork-mode note above). Biggest single win
  available. Estimated 10–15 s/swap.
- **Persistent CUDA graphs.** `Capturing CUDA graphs (decode, FULL)` runs
  for 5–8 s on every cold load and is fully model+config-deterministic.
  Could be saved per `(model, kv_blocks, capture_sizes)` and replayed.
  Likely a vLLM-side change.
- **Per-subprocess env (fix the env-leak issue).** Documented above.
- **NCCL persistence across swaps.** Currently each subprocess re-inits
  NCCL (1–2 s, world_size=1 so it's mostly context creation). Hard to
  share across processes.
- **Tokenizer pre-load via shared memory.** Each subprocess re-tokenizes
  config at startup; small (~1 s) but adds up.
