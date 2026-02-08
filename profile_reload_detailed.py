#!/usr/bin/env python3
"""Detailed profiling of reload_model() for all 4 working models.

Instruments every sub-step of reload_model to find exact bottlenecks.
Warms page cache first so disk I/O is not a factor.

Run on remote RTX PRO 6000:
    cd ~/pythonprojects/blitzinfer
    venv/bin/python profile_reload_detailed.py
"""

import gc
import os
import sys
import time
import subprocess
import logging

import torch

# Add sglang source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sglang", "python"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("profile")

# ── Model definitions ──────────────────────────────────────────────────
MODELS = {
    "qwen2.5-7b": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "overrides": {},
    },
    "llama-3.1-70b": {
        "model_path": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "overrides": {"quantization": "awq", "context_length": 131072},
    },
    "qwen3-32b": {
        "model_path": "Qwen/Qwen3-32B",
        "overrides": {},
    },
    "kimi-vl": {
        "model_path": "moonshotai/Kimi-VL-A3B-Instruct",
        "overrides": {},
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


def warm_page_cache(model_path: str):
    """Read all safetensor files into OS page cache."""
    from huggingface_hub import scan_cache_dir, HfFileSystemResolvedPath
    import glob

    # Find the model directory in HF cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_path.replace("/", "--")
    model_cache = os.path.join(cache_dir, model_dir_name)

    if not os.path.isdir(model_cache):
        log.warning(f"Model cache not found: {model_cache}")
        return 0.0

    # Find latest snapshot
    refs_dir = os.path.join(model_cache, "refs")
    snapshots_dir = os.path.join(model_cache, "snapshots")
    snapshot_hash = None
    if os.path.isdir(refs_dir):
        main_ref = os.path.join(refs_dir, "main")
        if os.path.isfile(main_ref):
            with open(main_ref) as f:
                snapshot_hash = f.read().strip()

    if snapshot_hash:
        model_dir = os.path.join(snapshots_dir, snapshot_hash)
    else:
        # Use latest snapshot
        snaps = sorted(os.listdir(snapshots_dir))
        model_dir = os.path.join(snapshots_dir, snaps[-1]) if snaps else model_cache

    # Find safetensor files
    st_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not st_files:
        # Check for symlinks pointing to blobs
        st_files = glob.glob(os.path.join(model_dir, "**/*.safetensors"), recursive=True)

    total_bytes = 0
    t0 = time.perf_counter()
    for f in st_files:
        # Resolve symlinks
        real_path = os.path.realpath(f)
        sz = os.path.getsize(real_path)
        total_bytes += sz
        # Read in 64MB chunks to warm page cache
        with open(real_path, 'rb') as fp:
            while True:
                chunk = fp.read(64 * 1024 * 1024)
                if not chunk:
                    break
    elapsed = time.perf_counter() - t0
    gb = total_bytes / 1024**3
    bw = gb / elapsed if elapsed > 0 else 0
    log.info(f"  Page cache warm: {model_path} - {gb:.1f}GB in {elapsed:.1f}s ({bw:.1f} GB/s, {len(st_files)} files)")
    return gb


def run_profile():
    from sglang.srt.entrypoints.engine import Engine as SglEngine

    log.info("=" * 70)
    log.info("DETAILED PROFILING: reload_model() Sub-Step Timing")
    log.info("=" * 70)

    used, free, total = get_gpu_mem()
    log.info(f"GPU: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")

    # Phase 0: Warm page cache for ALL models
    log.info("\n--- PHASE 0: Warm page cache for all models ---")
    for name, cfg in MODELS.items():
        warm_page_cache(cfg["model_path"])

    # Phase 1: Create engine with initial model (qwen2.5-7b = smallest)
    initial = "qwen2.5-7b"
    cfg = MODELS[initial]
    log.info(f"\n--- PHASE 1: Initial load of {initial} ---")

    t0 = time.perf_counter()
    engine = SglEngine(model_path=cfg["model_path"], **COMMON_KWARGS)
    t_init = time.perf_counter() - t0
    log.info(f"Initial Engine creation: {t_init:.1f}s")

    # Verify initial model works
    out = engine.generate(
        prompt="What is 2+2? Answer briefly.",
        sampling_params={"max_new_tokens": 10, "temperature": 0.0},
    )
    log.info(f"[{initial}] Output: {out['text'][:80]}")

    # Phase 2: Profile each reload
    # Test sequence: small→large→medium→vision→small (covers all architectures)
    switch_sequence = [
        ("qwen2.5-7b", "qwen3-32b"),
        ("qwen3-32b", "llama-3.1-70b"),
        ("llama-3.1-70b", "kimi-vl"),
        ("kimi-vl", "qwen2.5-7b"),
        # Second round to test warm reloads
        ("qwen2.5-7b", "llama-3.1-70b"),
        ("llama-3.1-70b", "qwen3-32b"),
    ]

    results = []
    for idx, (from_model, to_model) in enumerate(switch_sequence, 1):
        log.info(f"\n{'='*70}")
        log.info(f"RELOAD {idx}/{len(switch_sequence)}: {from_model} -> {to_model}")
        log.info(f"{'='*70}")

        used, free, _ = get_gpu_mem()
        log.info(f"Before: GPU {used:.1f}GB used, {free:.1f}GB free")

        cfg = MODELS[to_model]
        t_reload_start = time.perf_counter()

        success, message = engine.reload_model(
            model_path=cfg["model_path"],
            server_args_overrides=cfg.get("overrides", {}),
            flush_cache=True,
        )

        t_reload = time.perf_counter() - t_reload_start

        if not success:
            log.error(f"RELOAD FAILED: {message}")
            results.append({
                "idx": idx,
                "switch": f"{from_model} -> {to_model}",
                "total": t_reload,
                "success": False,
            })
            break

        log.info(f"Reload total: {t_reload:.2f}s")

        # Verify generation
        if to_model == "kimi-vl":
            prompt = "Describe what you see: a red car."
        else:
            prompt = "What is 2+2? Answer with just the number."

        t_gen = time.perf_counter()
        out = engine.generate(
            prompt=prompt,
            sampling_params={"max_new_tokens": 20, "temperature": 0.0},
        )
        t_gen = time.perf_counter() - t_gen
        text = out['text'][:100]
        log.info(f"[{to_model}] Generated in {t_gen:.1f}s: {text}")

        used, _, _ = get_gpu_mem()
        log.info(f"After: GPU {used:.1f}GB used")

        results.append({
            "idx": idx,
            "switch": f"{from_model} -> {to_model}",
            "total": t_reload,
            "gen_time": t_gen,
            "success": True,
            "output": text,
        })

    # Shutdown
    log.info("\n--- SHUTDOWN ---")
    engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY: reload_model() Detailed Timing")
    log.info("=" * 70)
    log.info(f"{'#':<3} {'Switch':<30} {'Total':>8} {'Gen':>6} {'Status'}")
    log.info("-" * 60)
    for r in results:
        if r["success"]:
            log.info(
                f"{r['idx']:<3} {r['switch']:<30} {r['total']:>6.2f}s {r.get('gen_time', 0):>4.1f}s  OK"
            )
        else:
            log.info(f"{r['idx']:<3} {r['switch']:<30} {r['total']:>6.2f}s          FAIL")

    successful = [r for r in results if r["success"]]
    if successful:
        total_reload = sum(r["total"] for r in successful)
        avg = total_reload / len(successful)
        log.info("-" * 60)
        log.info(f"    {'TOTAL':<30} {total_reload:>6.1f}s")
        log.info(f"    {'AVERAGE':<30} {avg:>6.2f}s")

    log.info("\nLook for 'PROFILE:' lines above for per-step breakdown.")
    log.info("Key phases: free_params, free_kv, gc, _initialize_model,")
    log.info("  model.load_weights, process_weights_after_loading,")
    log.info("  init_memory_pool, init_attention_backend")


if __name__ == "__main__":
    run_profile()
