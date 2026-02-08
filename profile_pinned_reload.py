#!/usr/bin/env python3
"""Profile reload_model() with pre-loaded pinned memory weights.

Compares:
1. Default: safetensors from page cache (~7 GB/s)
2. Pinned: pre-loaded weights in pinned CPU memory (~44 GB/s target)

Run on remote RTX PRO 6000:
    cd ~/pythonprojects/blitzinfer
    venv/bin/python profile_pinned_reload.py
"""

import gc
import glob
import os
import sys
import time
import logging

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sglang", "python"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("profile")

MODELS = {
    "qwen2.5-7b": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "overrides": {},
        "weight_gb": 14.2,
    },
    "llama-3.1-70b": {
        "model_path": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "overrides": {"quantization": "awq", "context_length": 131072},
        "weight_gb": 37.0,
    },
    "qwen3-32b": {
        "model_path": "Qwen/Qwen3-32B",
        "overrides": {},
        "weight_gb": 61.0,
    },
}

COMMON_KWARGS = {
    "mem_fraction_static": 0.94,
    "trust_remote_code": True,
    "disable_cuda_graph": True,
    "attention_backend": "triton",
    "log_level": "info",
}


def get_gpu_mem():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3, free / 1024**3, total / 1024**3


def get_model_dir(model_path: str) -> str:
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_path.replace("/", "--")
    model_cache = os.path.join(cache_dir, model_dir_name)
    refs_main = os.path.join(model_cache, "refs", "main")
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            snap = f.read().strip()
        return os.path.join(model_cache, "snapshots", snap)
    snaps = sorted(os.listdir(os.path.join(model_cache, "snapshots")))
    return os.path.join(model_cache, "snapshots", snaps[-1])


def preload_to_pinned(model_path: str) -> dict:
    """Load all safetensors weights into pinned CPU memory.

    Returns dict of {name: pinned_tensor}.
    """
    from safetensors.torch import load_file

    model_dir = get_model_dir(model_path)
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

    log.info(f"  Preloading {len(st_files)} safetensors files from {model_dir}")

    result = {}
    total_bytes = 0
    t0 = time.perf_counter()

    for i, f in enumerate(st_files):
        real_path = os.path.realpath(f)
        t_shard = time.perf_counter()

        # Load to CPU (non-pinned first)
        tensors = load_file(real_path, device="cpu")

        shard_bytes = 0
        for name, tensor in tensors.items():
            # Pin each tensor for fast GPU DMA transfer
            pinned = tensor.pin_memory()
            result[name] = pinned
            shard_bytes += tensor.numel() * tensor.element_size()

        total_bytes += shard_bytes
        shard_gb = shard_bytes / 1024**3
        shard_time = time.perf_counter() - t_shard
        log.info(
            f"    Shard {i+1}/{len(st_files)}: {shard_gb:.1f}GB in {shard_time:.2f}s "
            f"({shard_gb/shard_time:.1f} GB/s)"
        )

    elapsed = time.perf_counter() - t0
    total_gb = total_bytes / 1024**3
    log.info(
        f"  Preload complete: {len(result)} tensors, {total_gb:.1f}GB pinned "
        f"in {elapsed:.1f}s ({total_gb/elapsed:.1f} GB/s)"
    )
    return result


def warm_page_cache(model_path: str):
    """Read all safetensor files into OS page cache."""
    model_dir = get_model_dir(model_path)
    st_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    total_bytes = 0
    t0 = time.perf_counter()
    for f in st_files:
        real_path = os.path.realpath(f)
        total_bytes += os.path.getsize(real_path)
        with open(real_path, 'rb') as fp:
            while True:
                chunk = fp.read(64 * 1024 * 1024)
                if not chunk:
                    break
    elapsed = time.perf_counter() - t0
    gb = total_bytes / 1024**3
    log.info(f"  Page cache warm: {gb:.1f}GB in {elapsed:.1f}s ({gb/elapsed:.1f} GB/s)")


def run_profile():
    from sglang.srt.entrypoints.engine import Engine as SglEngine

    log.info("=" * 70)
    log.info("PROFILING: Pinned Memory vs Page Cache reload_model()")
    log.info("=" * 70)

    used, free, total = get_gpu_mem()
    log.info(f"GPU: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")

    # Test target: llama-3.1-70b (37GB, fits in RAM with room for pinning)
    # qwen3-32b is 61GB - pinning would use ~122GB which is tight with 124GB RAM
    target = "llama-3.1-70b"
    target_cfg = MODELS[target]

    # Phase 0: Preload target weights into pinned memory (background-simulatable)
    log.info(f"\n--- PHASE 0: Preload {target} weights into pinned memory ---")
    pinned_weights = preload_to_pinned(target_cfg["model_path"])

    # Also warm page cache for comparison
    log.info(f"\n--- PHASE 0b: Warm page cache for {target} ---")
    warm_page_cache(target_cfg["model_path"])

    # Phase 1: Load initial model
    initial = "qwen2.5-7b"
    cfg = MODELS[initial]
    log.info(f"\n--- PHASE 1: Load initial model {initial} ---")
    t0 = time.perf_counter()
    engine = SglEngine(model_path=cfg["model_path"], **COMMON_KWARGS)
    log.info(f"Initial Engine: {time.perf_counter() - t0:.1f}s")

    out = engine.generate(
        prompt="What is 2+2? Answer briefly.",
        sampling_params={"max_new_tokens": 10, "temperature": 0.0},
    )
    log.info(f"[{initial}] OK: {out['text'][:60]}")

    # Phase 2: Reload with page cache (baseline)
    log.info(f"\n{'='*70}")
    log.info(f"BASELINE: Page cache reload -> {target}")
    log.info(f"{'='*70}")

    t0 = time.perf_counter()
    success, msg = engine.reload_model(
        model_path=target_cfg["model_path"],
        server_args_overrides=target_cfg.get("overrides", {}),
        flush_cache=True,
    )
    t_baseline = time.perf_counter() - t0
    assert success, f"Baseline failed: {msg}"

    out = engine.generate(
        prompt="What is 2+2? Answer briefly.",
        sampling_params={"max_new_tokens": 10, "temperature": 0.0},
    )
    log.info(f"[BASELINE {target}] {t_baseline:.2f}s - Output: {out['text'][:60]}")

    # Return to initial
    engine.reload_model(
        model_path=MODELS[initial]["model_path"],
        server_args_overrides=MODELS[initial].get("overrides", {}),
        flush_cache=True,
    )

    # Phase 3: Reload with pinned memory
    log.info(f"\n{'='*70}")
    log.info(f"PINNED: Preloaded pinned memory reload -> {target}")
    log.info(f"{'='*70}")

    t0 = time.perf_counter()
    success, msg = engine.reload_model(
        model_path=target_cfg["model_path"],
        server_args_overrides=target_cfg.get("overrides", {}),
        flush_cache=True,
        preloaded_weights=pinned_weights,
    )
    t_pinned = time.perf_counter() - t0
    assert success, f"Pinned failed: {msg}"

    out = engine.generate(
        prompt="What is 2+2? Answer briefly.",
        sampling_params={"max_new_tokens": 10, "temperature": 0.0},
    )
    log.info(f"[PINNED {target}] {t_pinned:.2f}s - Output: {out['text'][:60]}")

    # Return to initial
    engine.reload_model(
        model_path=MODELS[initial]["model_path"],
        server_args_overrides=MODELS[initial].get("overrides", {}),
        flush_cache=True,
    )

    # Phase 4: Second pinned run (to verify consistency)
    log.info(f"\n{'='*70}")
    log.info(f"PINNED (2nd run): Preloaded pinned memory reload -> {target}")
    log.info(f"{'='*70}")

    t0 = time.perf_counter()
    success, msg = engine.reload_model(
        model_path=target_cfg["model_path"],
        server_args_overrides=target_cfg.get("overrides", {}),
        flush_cache=True,
        preloaded_weights=pinned_weights,
    )
    t_pinned2 = time.perf_counter() - t0
    assert success, f"Pinned 2nd failed: {msg}"

    out = engine.generate(
        prompt="What is 2+2? Answer briefly.",
        sampling_params={"max_new_tokens": 10, "temperature": 0.0},
    )
    log.info(f"[PINNED2 {target}] {t_pinned2:.2f}s - Output: {out['text'][:60]}")

    # Cleanup
    engine.shutdown()
    del pinned_weights
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    weight_gb = target_cfg["weight_gb"]
    log.info("\n" + "=" * 70)
    log.info(f"SUMMARY: {target} ({weight_gb}GB)")
    log.info("=" * 70)
    log.info(f"{'Method':<25} {'Total':>8} {'Rate':>10}")
    log.info("-" * 45)
    log.info(f"{'Page cache (baseline)':<25} {t_baseline:>6.2f}s {weight_gb/t_baseline:>8.1f} GB/s")
    log.info(f"{'Pinned memory':<25} {t_pinned:>6.2f}s {weight_gb/t_pinned:>8.1f} GB/s")
    log.info(f"{'Pinned memory (2nd)':<25} {t_pinned2:>6.2f}s {weight_gb/t_pinned2:>8.1f} GB/s")
    log.info("-" * 45)
    speedup = t_baseline / min(t_pinned, t_pinned2)
    log.info(f"Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    run_profile()
