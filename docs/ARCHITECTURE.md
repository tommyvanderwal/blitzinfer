# Architecture

This is the technical map of what's in the box. For *why* each piece is shaped
this way, see [DESIGN.md](DESIGN.md).

## High level

```
   ┌──────────────────────────────────────────────────────────────┐
   │   Clients (any OpenAI-compatible)                             │
   │   curl, agents, OpenCode, Claude Code, hermes, ...            │
   └───────────────────────────────┬───────────────────────────────┘
                                   │ HTTPS /v1/chat/completions
                                   │ /v1/models, /v1/admin/*
                                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │   FastAPI process (uvicorn, :8000)                            │
   │  ┌─────────────────────────────────────────────────────┐     │
   │  │ ModelManager                                         │     │
   │  │   _queues: dict[str, deque]      ← per-model FIFO    │     │
   │  │   engine, active_name             ← live engine      │     │
   │  │   in_flight, _swapping            ← state            │     │
   │  │                                                      │     │
   │  │   acquire()  ─►  enqueue + await future               │     │
   │  │   release()  ─►  in_flight-- + notify                 │     │
   │  └──────────────────┬──────────────────────────────────┘     │
   │                     │ notify_all()                            │
   │                     ▼                                         │
   │  ┌─────────────────────────────────────────────────────┐     │
   │  │ dispatcher (single coroutine)                        │     │
   │  │   1. drain queues[active] onto live engine            │     │
   │  │   2. reap finished swap_task                          │     │
   │  │   3. if no swap & other queue non-empty:              │     │
   │  │        pick next = oldest queued; start swap_task     │     │
   │  │   4. cv.wait()                                        │     │
   │  └──────────────────┬──────────────────────────────────┘     │
   │                     │ from_vllm_config(...)                   │
   │                     ▼                                         │
   │  ┌─────────────────────────────────────────────────────┐     │
   │  │ AsyncLLM (vLLM 0.20.1 client)                        │     │
   │  └──────────────────┬──────────────────────────────────┘     │
   │                     │ ZMQ                                     │
   └─────────────────────┼─────────────────────────────────────────┘
                         │
                         ▼
   ┌──────────────────────────────────────────────────────────────┐
   │   EngineCore subprocess  (one model at a time, killed on swap)│
   │   • imports vLLM, opens tokenizer, parses config              │
   │   • blocks at BLITZ_GPU_GO_FILE barrier (patched gpu_worker)  │
   │   • when released: set_device → load weights from pool → KV   │
   │     cache → CUDA graphs → handshake "ready"                   │
   └──────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │   GPU — one model fully loaded                                │
   │   Active model owns 60–80 GB of VRAM. Subprocess termination  │
   │   on swap is what reclaims it (OS clears process pages).      │
   └──────────────────────────────────────────────────────────────┘
```

The two side channels:

```
   parent process                          subprocess (EngineCore)

   shared hugetlbfs file                   re-mmaps + cudaHostRegister
   /mnt/hugetlbfs/blitz_pool   ─────►       in its own CUDA context
   (80 × 1 GB hugepages,                    → reads weights pool→GPU
   parent mmap'd, NOT registered)           at PCIe 5 wire speed

   touch(go_file) on stage D    ─────►     blocks on
                                            while not exists(go_file): sleep(50ms)
                                           (patch in vllm/v1/worker/gpu_worker.py)
```

## Request lifecycle (active model)

```
client                       FastAPI          ModelManager       dispatcher           live engine
  │                             │                 │                  │                     │
  ├─POST /v1/chat/completions──►│                 │                  │                     │
  │                             ├─acquire(model)─►│                  │                     │
  │                             │                 ├─enqueue+notify──►│                     │
  │                             │                 │                  ├─dispatch (active=)──┤
  │                             │                 │◄─resolve future──┤                     │
  │                             │◄─handler────────┤                  │                     │
  │                             ├─create_chat_completion──────────────────────────────────►│
  │◄────────────SSE stream──────┤◄───────────────token chunks──────────────────────────────┤
  │                             ├─wrapped().finally:                                        │
  │                             │  release()─────►│                  │                     │
  │                             │                 ├─in_flight--+notify◄────────────────────┤
```

Every request goes through the queue — there is no fast path. The dispatcher
sees `queues[active_name]` non-empty and immediately resolves the future with
the live `openai_serving_chat` handler. Latency overhead is one async wakeup
(~µs).

## Swap lifecycle (the pipelined acquire)

The swap is the interesting part. It runs in a separate `swap_task`
coroutine spawned by the dispatcher; the dispatcher itself stays free to
keep dispatching new arrivals on `queues[active]`.

```
                          ▼ B1 arrives, dispatcher commits to model B
          ┌──────────────────────────────────────────────────────────┐
 stage A  │ start spawn_task + pool_task  (T+0)                       │
          │   spawn_task:  fork EngineCore subprocess                 │
          │                imports vLLM (~10–17 s)                    │
          │                hits BLITZ_GPU_GO_FILE barrier ──┐         │
          │                                                  │         │
          │   pool_task:   parent reads B's safetensors      │         │
          │                shards into hugetlbfs pool        │         │
          │                (~3–10 s, 4 shards × 16 chunks    │         │
          │                parallel preadv ≈ 10 GB/s)        │         │
          ├──────────────────────────────────────────────────┼─────────┤
 ...      │  (active model A keeps serving in_flight        │         │
 active   │   requests; dispatcher keeps draining           │         │
 serves   │   queues[A] for new arrivals)                   │         │
          ├──────────────────────────────────────────────────┼─────────┤
 stage B  │ wait until queues[A] empty AND in_flight == 0   │         │
          │                                                  │         │
 stage C  │ _unload():                                       │         │
          │   engine.shutdown()                              │         │
          │   poll nvidia-smi --query-gpu=memory.used        │         │
          │   until ≤ 2 GiB                       (≈3 s)     │         │
          │   ◄── proves driver actually reclaimed VRAM      │         │
          ├──────────────────────────────────────────────────┼─────────┤
 stage D  │ await pool_task  (already done)                 │         │
          │ touch(go_file)  ────────────────────────────────►┘         │
          │                          ◄── subprocess crosses barrier   │
          ├──────────────────────────────────────────────────────────┤
 stage E  │ subprocess: set_device → cudaHostRegister(pool, 80 GB) →  │
          │             multi_thread_safetensors_iterator reads pool, │
          │             vLLM copies pool→GPU at ~44 GB/s →            │
          │             KV cache, CUDA graphs, FlashInfer autotune →  │
          │             handshake "ready"                             │
          │ await spawn_task  ── from_vllm_config returns             │
          │ engine.reset_mm_cache()                                    │
          │ init_app_state() ── installs new openai_serving_chat       │
          │ self.engine = engine; self.active_name = name              │
          ├──────────────────────────────────────────────────────────┤
          │ finally: _swapping=False; notify_all                      │
          │   ◄── dispatcher wakes, dispatches queues[B]              │
          └──────────────────────────────────────────────────────────┘
```

The "wait" in stage B is what makes the pipeline work: vLLM imports + pool
disk read happen *behind* the active model's serving, so when the user
finally finishes their long decode on model A, the subprocess for B is
already parked at the barrier with weights ready in RAM. Only the GPU-touching
work (set_device → weight transfer pool→GPU → KV/CUDA-graph init) remains
visible after model A finishes.

Concretely, on a 75 GB NVFP4 MoE swap from gpt-oss-120b:
- Sequential cold load (`vllm serve`): ~75 s after the previous request finishes.
- Pipelined: ~40 s after the previous request finishes; the other ~35 s ran
  during the 50 K-token decode on the previous model.

## Dispatcher state diagram

```
              ┌───────────────────────────┐
              │   idle                    │
              │   no swap_task            │
              │   no model active OR      │
              │   active queue empty +    │
              │   all other queues empty  │
              └───────┬───────────────────┘
                      │ request enqueued
                      ▼
              ┌───────────────────────────┐         active queue
              │   serving                 │◄──┐    has work
              │   active engine alive     │   │
              │   queues[active] draining │   │
              │   no swap_task            │   │
              └───────┬───────────────────┘   │
                      │ active queue empty +  │
                      │ another queue has work│
                      ▼                       │
              ┌───────────────────────────┐   │
              │   spawning                │   │
              │   swap_task running:      │   │
              │     stage A → B → C → D   │   │
              │   active engine still     │───┘ active queue refills
              │   alive, can dispatch     │     (request for active arrives)
              └───────┬───────────────────┘
                      │ stage B done
                      ▼
              ┌───────────────────────────┐
              │   unloading               │
              │   engine = None           │
              │   waiting nvidia-smi ≤2GiB│
              │   queues[active_name]     │
              │   stops dispatching       │
              └───────┬───────────────────┘
                      │ stage E done
                      ▼
              ┌───────────────────────────┐
              │   installing              │
              │   from_vllm_config OK     │
              │   init_app_state runs     │
              └───────┬───────────────────┘
                      │ self.engine assigned;
                      │ finally: notify_all
                      ▼
                 (back to serving, with new active_name)
```

## Memory map

```
   GPU VRAM (96 GB)
   ┌──────────────────────────────────────────────────┐
   │ active model weights        60–80 GB              │
   │ KV cache + CUDA graphs      8–25 GB               │
   │ activations, scratch        ~4 GB                 │
   │ free margin                 ~1–4 GB               │
   └──────────────────────────────────────────────────┘
       ⚠ at most ONE model resident at any time

   System RAM (128 GB)
   ┌──────────────────────────────────────────────────┐
   │ /mnt/hugetlbfs/blitz_pool   80 GB (1 GB hugepages)│
   │   parent mmap, NOT cudaHostRegister'd             │
   │   subprocess re-mmaps + registers in its own ctx  │
   ├──────────────────────────────────────────────────┤
   │ EngineCore subprocess RSS   3–5 GB                │
   │   (vLLM + torch + tokenizer + flashinfer state)   │
   ├──────────────────────────────────────────────────┤
   │ FastAPI parent RSS          1–2 GB                │
   ├──────────────────────────────────────────────────┤
   │ OS + page cache for HF model files     ~30 GB     │
   │   (helps cold disk reads when pool is overwritten │
   │    by a different model)                          │
   └──────────────────────────────────────────────────┘
```

## Two invariants (worth reading the code to verify)

1. **The 80 GB pool is never used twice.** `SharedPool.load_shards` overwrites
   in place; once a subprocess has copied weights pool→GPU it never reads
   pool again. There is no scenario where two different models occupy pool
   memory simultaneously — `if self._loaded_model == model_name: return`
   short-circuits same-model reloads, and any other case `pwrite()`s on top
   of the existing bytes.

2. **Never two subprocess loads in parallel.** The dispatcher's
   `if swap_task is None` gate ensures a second swap can't begin until the
   current one's `from_vllm_config` has returned (= subprocess is fully
   loaded). Pipelining overlaps loading with *serving*, never with another
   loading. A request for a third model arriving mid-swap-to-B sees the gate
   fail and `cv.wait()`s; only after B completes does C's swap start.

## vLLM patches (in `patches/`)

| Patch | What | Why |
|---|---|---|
| `0001-blitz-gpu-barrier.patch` | Insert a file-existence wait in `vllm/v1/worker/gpu_worker.py` just before the worker's first CUDA call | Lets the subprocess start importing vLLM while the previous model is still on the GPU; parks until the parent confirms VRAM is released. No-op when `BLITZ_GPU_GO_FILE` is unset. |
| `0002-pinned-loader-hook.patch` | Have `vllm/model_executor/model_loader/default_loader.py` import `multi_thread_safetensors_weights_iterator` from `blitzinfer.loader.pinned_loader` | Routes weight loading through the shared hugetlbfs pool when `BLITZ_SHARED_POOL` + `BLITZ_SHARDS_LAYOUT` are set; falls through to vLLM's stock disk read otherwise. |

Both patches are designed so that without the gateway's env vars, vLLM
behaves identically to upstream — they're additive. `scripts/apply_vllm_patches.sh`
is idempotent (it dry-run-reverses the patch first to detect already-applied
hunks).

## Files at a glance

```
blitzinfer/
├── api/
│   ├── __init__.py
│   └── server.py              ← gateway. ModelManager, dispatcher, routes,
│                                pipelined swap, admin endpoints
└── loader/
    ├── __init__.py
    ├── shared_pool.py         ← parent-side hugetlbfs mmap, parallel preadv
    │                            into the pool, layout-JSON writer
    ├── pinned_loader.py       ← subprocess-side: cudaHostRegister + safetensors
    │                            parsing as views into pool memory
    └── profile_cache.py       ← persistent cache of num_gpu_blocks per
                                  (model, ctx, util, vllm/driver/gpu) so
                                  vLLM's profile pass can be skipped via
                                  --num-gpu-blocks-override on subsequent loads
```

That's all of it. ~1500 lines of Python total.
