#!/usr/bin/env python3
"""Focused profiling: measure model.load_weights with truly warm page cache.

Only warms the TARGET model's page cache (not all models), ensuring it stays
in RAM. Tests:
1. Cold SSD read speed (first reload, no page cache)
2. Warm page cache speed (second reload, page cache hot)

Run on remote RTX PRO 6000:
    cd ~/pythonprojects/blitzinfer
    venv/bin/python profile_reload_focused.py
"""

import gc
import os
import sys
import time
import glob
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


def drop_page_cache_for_model(model_path: str):
    """Drop page cache for model files using posix_fadvise."""
    import ctypes
    import ctypes.util
    model_dir = get_model_dir(model_path)
    st_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    POSIX_FADV_DONTNEED = 4
    total = 0
    for f in st_files:
        real_path = os.path.realpath(f)
        sz = os.path.getsize(real_path)
        fd = os.open(real_path, os.O_RDONLY)
        libc.posix_fadvise(fd, 0, sz, POSIX_FADV_DONTNEED)
        os.close(fd)
        total += sz
    log.info(f"  Dropped page cache for {model_path}: {total/1024**3:.1f}GB ({len(st_files)} files)")


def warm_page_cache(model_path: str) -> float:
    """Read all safetensor files into OS page cache."""
    model_dir = get_model_dir(model_path)
    st_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    total_bytes = 0
    t0 = time.perf_counter()
    for f in st_files:
        real_path = os.path.realpath(f)
        sz = os.path.getsize(real_path)
        total_bytes += sz
        with open(real_path, 'rb') as fp:
            while True:
                chunk = fp.read(64 * 1024 * 1024)
                if not chunk:
                    break
    elapsed = time.perf_counter() - t0
    gb = total_bytes / 1024**3
    bw = gb / elapsed if elapsed > 0 else 0
    log.info(f"  Page cache warm: {model_path} - {gb:.1f}GB in {elapsed:.1f}s ({bw:.1f} GB/s)")
    return gb


def run_profile():
    from sglang.srt.entrypoints.engine import Engine as SglEngine

    log.info("=" * 70)
    log.info("FOCUSED PROFILING: Cold vs Warm Page Cache reload_model()")
    log.info("=" * 70)

    used, free, total = get_gpu_mem()
    log.info(f"GPU: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")

    # Initial model: qwen2.5-7b (smallest)
    initial = "qwen2.5-7b"
    cfg = MODELS[initial]
    log.info(f"\n--- Load initial model: {initial} ---")
    t0 = time.perf_counter()
    engine = SglEngine(model_path=cfg["model_path"], **COMMON_KWARGS)
    log.info(f"Initial Engine: {time.perf_counter() - t0:.1f}s")

    out = engine.generate(
        prompt="What is 2+2? Answer briefly.",
        sampling_params={"max_new_tokens": 10, "temperature": 0.0},
    )
    log.info(f"[{initial}] OK: {out['text'][:60]}")

    # Test each target model: cold then warm
    test_sequence = [
        ("llama-3.1-70b", "qwen2.5-7b"),  # Test llama, return to qwen
        ("qwen3-32b", "qwen2.5-7b"),       # Test qwen3, return to qwen
    ]

    results = []
    for target, return_to in test_sequence:
        target_cfg = MODELS[target]
        return_cfg = MODELS[return_to]

        # ─── COLD run: drop page cache, then reload ───
        log.info(f"\n{'='*70}")
        log.info(f"COLD RELOAD: -> {target} (page cache dropped)")
        log.info(f"{'='*70}")
        drop_page_cache_for_model(target_cfg["model_path"])

        t0 = time.perf_counter()
        success, msg = engine.reload_model(
            model_path=target_cfg["model_path"],
            server_args_overrides=target_cfg.get("overrides", {}),
            flush_cache=True,
        )
        t_cold = time.perf_counter() - t0
        assert success, f"Cold reload failed: {msg}"

        out = engine.generate(
            prompt="What is 2+2? Answer briefly.",
            sampling_params={"max_new_tokens": 10, "temperature": 0.0},
        )
        log.info(f"[COLD {target}] {t_cold:.2f}s - Output: {out['text'][:60]}")

        # Return to initial
        engine.reload_model(
            model_path=return_cfg["model_path"],
            server_args_overrides=return_cfg.get("overrides", {}),
            flush_cache=True,
        )

        # ─── WARM run: pre-warm page cache, then reload ───
        log.info(f"\n{'='*70}")
        log.info(f"WARM RELOAD: -> {target} (page cache pre-warmed)")
        log.info(f"{'='*70}")
        warm_page_cache(target_cfg["model_path"])

        t0 = time.perf_counter()
        success, msg = engine.reload_model(
            model_path=target_cfg["model_path"],
            server_args_overrides=target_cfg.get("overrides", {}),
            flush_cache=True,
        )
        t_warm = time.perf_counter() - t0
        assert success, f"Warm reload failed: {msg}"

        out = engine.generate(
            prompt="What is 2+2? Answer briefly.",
            sampling_params={"max_new_tokens": 10, "temperature": 0.0},
        )
        log.info(f"[WARM {target}] {t_warm:.2f}s - Output: {out['text'][:60]}")

        speedup = t_cold / t_warm if t_warm > 0 else 0
        rate_cold = target_cfg["weight_gb"] / t_cold if t_cold > 0 else 0
        rate_warm = target_cfg["weight_gb"] / t_warm if t_warm > 0 else 0
        results.append({
            "model": target,
            "weight_gb": target_cfg["weight_gb"],
            "cold": t_cold,
            "warm": t_warm,
            "speedup": speedup,
            "rate_cold": rate_cold,
            "rate_warm": rate_warm,
        })

        # Return to initial for next test
        engine.reload_model(
            model_path=return_cfg["model_path"],
            server_args_overrides=return_cfg.get("overrides", {}),
            flush_cache=True,
        )

    # Shutdown
    engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY: Cold vs Warm Page Cache reload_model()")
    log.info("=" * 70)
    log.info(f"{'Model':<20} {'Weights':>8} {'Cold':>8} {'Warm':>8} {'Speedup':>8} {'Cold GB/s':>10} {'Warm GB/s':>10}")
    log.info("-" * 78)
    for r in results:
        log.info(
            f"{r['model']:<20} {r['weight_gb']:>6.1f}GB "
            f"{r['cold']:>6.2f}s {r['warm']:>6.2f}s "
            f"{r['speedup']:>6.1f}x "
            f"{r['rate_cold']:>8.1f} {r['rate_warm']:>8.1f}"
        )

    log.info("\nKey: 'Cold' = page cache dropped, 'Warm' = page cache pre-loaded")
    log.info("If Warm GB/s >> Cold GB/s, disk I/O is the bottleneck.")
    log.info("If Warm GB/s ~ Cold GB/s, CPU/copy overhead is the bottleneck.")


if __name__ == "__main__":
    run_profile()
