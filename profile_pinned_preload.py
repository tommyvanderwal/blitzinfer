#!/usr/bin/env python3
"""Focused profile: preload + reload timing breakdown.

Single test: qwen3-32b -> llama-70b with pinned preload.
Measures every phase precisely.
"""

import gc
import glob as _glob
import logging
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("profile_pinned")


def create_engine(model_path, **extra_kwargs):
    from sglang.srt.entrypoints.engine import Engine
    kwargs = dict(
        model_path=model_path,
        mem_fraction_static=0.85,
        max_running_requests=4,
        disable_cuda_graph=True,
        fp8_gemm_runner_backend="triton",
        log_level="info",
    )
    kwargs.update(extra_kwargs)
    t0 = time.perf_counter()
    engine = Engine(**kwargs)
    logger.info(f"Engine created in {time.perf_counter() - t0:.1f}s")
    return engine


def drop_page_cache(model_path):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_path.replace("/", "--")
    model_cache = os.path.join(cache_dir, model_dir_name)
    refs_main = os.path.join(model_cache, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            snap = f.read().strip()
        model_dir = os.path.join(model_cache, "snapshots", snap)
    else:
        return
    st_files = sorted(_glob.glob(os.path.join(model_dir, "*.safetensors")))
    total = 0
    for fpath in st_files:
        real = os.path.realpath(fpath)
        sz = os.path.getsize(real)
        fd = os.open(real, os.O_RDONLY)
        os.posix_fadvise(fd, 0, sz, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        total += sz
    logger.info(f"Dropped page cache: {total/1024**3:.1f}GB")


def main():
    INITIAL = "Qwen/Qwen3-32B-FP8"
    TARGET = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    TARGET_OVERRIDES = {"quantization": "awq"}

    print("=" * 70)
    print("DETAILED PROFILING: Pinned Preload + Reload")
    print(f"  {INITIAL} -> {TARGET}")
    print("=" * 70)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {gpu.name}, {total/1024**3:.0f}GB total, {free/1024**3:.0f}GB free")

    # Step 1: Create engine with initial model
    engine = create_engine(INITIAL)

    # Step 2: Drop page cache for target
    drop_page_cache(TARGET)

    # Step 3: Start preloading
    logger.info("=== Starting preload ===")
    t_pre = time.perf_counter()
    success, msg = engine.preload_for_reload(TARGET)
    logger.info(f"preload_for_reload returned in {time.perf_counter() - t_pre:.2f}s: {success}")

    # Step 4: Wait for preload (37GB at ~5 GB/s from cold disk + pin_memory)
    logger.info("Waiting 60s for preload to complete...")
    time.sleep(60)

    # Step 5: Reload with preloaded weights
    logger.info("=== Starting reload ===")
    t0 = time.perf_counter()
    success, msg = engine.reload_model(
        model_path=TARGET,
        server_args_overrides=TARGET_OVERRIDES,
    )
    total = time.perf_counter() - t0
    logger.info(f"Engine.reload_model total: {total:.3f}s, success={success}")

    if success:
        # Quick generation test
        params = {"temperature": 0, "max_new_tokens": 20}
        try:
            result = engine.generate(prompt="Hello! How are you?", sampling_params=params)
            logger.info(f"Generation: {result['text'][:60]!r}")
        except Exception as e:
            logger.error(f"Generation failed: {e}")

    engine.shutdown()

    print("\n" + "=" * 70)
    print(f"Total Engine.reload_model: {total:.3f}s")
    print("Check PROFILE logs above for per-phase breakdown")
    print("=" * 70)


if __name__ == "__main__":
    main()
