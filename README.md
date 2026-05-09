# BlitzInfer

A multi-model LLM serving gateway built on [vLLM](https://github.com/vllm-project/vllm), with smart per-model queueing, pipelined model swaps, and a cross-process shared hugetlbfs pool for fast cold-loads.

One vLLM EngineCore at a time, full GPU per model, OpenAI-compatible REST API, ~0 MiB VRAM drift across rotations.

> **Heads up — this is a one-machine project.** It is hand-tuned for the specific box below: an RTX PRO 6000 (96 GB) on a Ryzen 7 7800X3D with 128 GB DDR5 in a PCIe 5.0 x16 slot. Tuning, registry, page sizes, pool size, and several timings all assume that hardware. Public and free to try if you have something similar — but **YMMV** on anything else, and "anything else" includes other Blackwells with less VRAM, slower CPUs, fewer PCIe lanes, or less host RAM.
>
> Built specifically to keep one big GPU saturated by rotating through whatever model the next request asks for, as fast as possible. Not a general-purpose multi-GPU serving stack.

## Target machine

| Component | This box |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB GDDR7 (SM120) |
| CPU | AMD Ryzen 7 7800X3D |
| GPU slot | PCIe 5.0 ×16 |
| System RAM | 128 GB DDR5 |
| Storage | NVMe (PCIe 4.0+) for the HF model cache |
| OS | Ubuntu 24.04 LTS, kernel ≥ 6.8, with `hugepagesz=1G hugepages=80` on the kernel cmdline |
| Software | vLLM 0.20.1, PyTorch 2.11+cu130, FlashInfer 0.6.8.post1, Python 3.13 |

If your hardware is meaningfully different — less VRAM, no SM120, slow PCIe, less than ~110 GB usable RAM — expect to need to retune or skip parts of the design (the 80 GB pinned pool is the most likely thing to fight you).

## What it does

You point it at a registry of models. Requests come in over `/v1/chat/completions`. The gateway:

- Serves concurrent requests for the active model on the live engine (vLLM batches them internally).
- Queues requests for non-active models per-model (FIFO).
- When the active queue empties, picks the next model — *longest-waiting queued request wins* — and swaps to it.
- **Drains before switching**: same-model requests that arrive *after* a cross-model request still get served on the active engine. No thrashing.
- **Pipelines the swap**: the moment a next-model decision is made, it spawns a fresh EngineCore subprocess (vLLM imports + tokenizer + GPU barrier wait) and reads the next model's weights into an 80 GB hugetlbfs pool — all *while the current engine keeps serving*. Only when the current model finishes does it kill the old subprocess, wait for `nvidia-smi` to report VRAM idle, and release the new subprocess past its barrier.
- **Subprocess teardown per swap** = OS reclaims VRAM; ~0 MiB drift over 50+ swaps.

## Why

Compared to running `vllm serve` per model: you don't pay the 60–120 s cold-load cost on every cross-model request — most of it is hidden behind the previous model's serving.

Compared to LM Studio / Ollama / GGUF stacks: you keep vLLM's full performance (CUDA graphs, FlashAttention, FP8/MXFP4/NVFP4 kernels, prefix caching, 32-way concurrency per model).

## Install

```bash
git clone https://github.com/tommyvanderwal/blitzinfer.git
cd blitzinfer
python3.13 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Apply the two vLLM patches that the gateway expects.
# (gpu_worker.py barrier + default_loader.py shared-pool hook.)
bash scripts/apply_vllm_patches.sh
```

You also need a hugetlbfs mount and 80 × 1 GB hugepages reserved at boot. On Ubuntu:

```bash
# /etc/default/grub:
GRUB_CMDLINE_LINUX_DEFAULT="... hugepagesz=1G hugepages=80"

# /etc/fstab:
hugetlbfs /mnt/hugetlbfs hugetlbfs pagesize=1G,size=80G,mode=0777 0 0

sudo update-grub && sudo reboot
```

## Run

```bash
# Optional — preload a model on startup:
export BLITZ_DEFAULT_MODEL=qwen3-coder-next

python -m blitzinfer.api.server
```

The server listens on `0.0.0.0:8000`. Hit it like any OpenAI endpoint:

```bash
export BLITZINFER_HOST=192.168.1.42   # your gateway box

curl -s "http://${BLITZINFER_HOST}:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-122b-a10b",
    "messages": [{"role": "user", "content": "What is in this image?"}],
    "max_tokens": 200
  }'
```

`/v1/models` lists the registry. `/v1/admin/status` shows live queue depths, `swapping` flag, in-flight count, and shared-pool state. `/v1/admin/load` and `/v1/admin/unload` are explicit admin hooks.

## Models in the default registry

| Served name | Repo | Quant / size |
|---|---|---|
| `qwen3.5-122b-a10b` | `RedHatAI/Qwen3.5-122B-A10B-NVFP4` | NVFP4 / 75 GB, image+video |
| `qwen3.6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B-FP8` | FP8 MoE / 35 GB, image |
| `qwen3.6-27b` | `Qwen/Qwen3.6-27B-FP8` | FP8 / 29 GB, image |
| `gemma-4-31b` | `google/gemma-4-31b-it` | BF16 / 59 GB, image |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | MXFP4 MoE / 64 GB, Harmony tools |
| `qwen3-coder-next` | `Qwen/Qwen3-Coder-Next-FP8` | FP8 MoE / 75 GB |
| `qwen3-32b` | `Qwen/Qwen3-32B-FP8` | FP8 / 32 GB |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | BF16 / 14 GB |
| `kimi-vl` | `moonshotai/Kimi-VL-A3B-Instruct` | BF16 / 6 GB, image |

Edit `REGISTRY` in `blitzinfer/api/server.py` to change the lineup. Tool/reasoning parsers, context length, and model-specific env (e.g. `VLLM_MXFP4_USE_MARLIN=1` for gpt-oss) are configured per model.

## Performance

Reference benchmark (warm caches, 9 models, 3 tests each):

| Model | T1 cold load | Note |
|---|---|---|
| qwen2.5-7b | 17.7 s | |
| qwen3-32b | 28.7 s | |
| qwen3-coder-next | 37.6 s | 75 GB FP8 MoE |
| qwen3.6-35b-a3b | 36.5 s | |
| qwen3.6-27b | 46.3 s | |
| qwen3.5-122b-a10b | ~75 s warm | 75 GB NVFP4 |
| gemma-4-31b | 55.0 s | |
| gpt-oss-120b | 30.8 s | |
| kimi-vl | 29.8 s | |

Pipelining proof — 50 K-token decode on model N runs concurrently with a request on model N+1:

- `qwen3-coder-next || qwen3.6-35b-a3b`: only **27.1 s of swap visible after model N finished** (vs ~45 s sequential).
- `gpt-oss-120b || qwen3.5-122b-a10b`: 40.8 s post-N swap for a 75 GB NVFP4 MoE (vs ~75 s sequential).

## Architecture

```
HTTP /v1/chat/completions
    ↓
ModelManager.acquire(model)            ← every request goes through the queue
    ↓ (per-model deque)
single dispatcher coroutine
    ↓                            ↘
drain queues[active] (FIFO)        if active queue empty + other queue non-empty:
    ↓                                _do_pipelined_swap(next_model)
new request future ── set_result(handler)    (oldest queued wins)
    ↓                                ↓
chat_completion runs concurrently   spawn EngineCore subprocess +
on the live engine                  parent loads next weights into 80 GB
    ↓                                hugetlbfs pool, IN PARALLEL with
release() → in_flight--             dispatch above
                                    ↓
                                    drain wait → unload → wait nvidia-smi
                                    ≤ 2 GiB → touch BLITZ_GPU_GO_FILE →
                                    subprocess crosses barrier in
                                    gpu_worker.py → loads weights from pool
                                    → init_app_state installs handler →
                                    notify_all → dispatcher serves new queue
```

Two invariants:
- The 80 GB pool is never used twice. `SharedPool.load_shards` overwrites in place; once a subprocess has copied weights pool → GPU it never reads pool again.
- Never two subprocess loads in parallel. The dispatcher's `if swap_task is None` gate ensures a second swap can't begin until the current one's subprocess is fully loaded. Pipelining overlaps loading with *serving*, not with another loading.

`CLAUDE.md` has the long version for contributors.

## License & status

MIT — see [LICENSE](LICENSE). vLLM is Apache 2.0 and stays a pip dependency; the two `.patch` files in `patches/` are MIT-licensed (authored diffs). When applied to your installed vLLM, the resulting modified files on disk remain Apache 2.0 — you can't relicense vLLM's code, only your own contribution.

No support, no warranty, no roadmap commitments — see the LICENSE for the legalese version. This is one person's tuning effort for one specific box, made public on the off chance it's useful to someone with the same hardware. Issues / PRs are welcome but I make no promise to act on them.
