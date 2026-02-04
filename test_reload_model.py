#!/usr/bin/env python3
"""Test SGLang Engine reload_model() for fast cross-architecture switching.

Tests the new reload_model() method that swaps model architecture within
the existing Engine process, avoiding the 4-5s subprocess restart overhead.

Run on remote RTX PRO 6000:
    cd ~/pythonprojects/blitzinfer
    venv/bin/python test_reload_model.py
"""

import gc
import os
import sys
import time
import torch
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sglang", "python"))
from sglang.srt.entrypoints.engine import Engine as SglEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("test_reload")


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
    "gpt-oss-120b": {
        "model_path": "openai/gpt-oss-120b",
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


def test_reload():
    log.info("=" * 70)
    log.info("TEST: SGLang Engine reload_model() - Cross-Architecture Switching")
    log.info("=" * 70)

    used, free, total = get_gpu_mem()
    log.info(f"GPU: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")

    # Phase 1: Create engine with initial model
    initial = "qwen2.5-7b"
    cfg = MODELS[initial]
    log.info(f"\n--- PHASE 1: Initial load of {initial} ---")

    t0 = time.perf_counter()
    engine = SglEngine(model_path=cfg["model_path"], **COMMON_KWARGS)
    t_init = time.perf_counter() - t0
    log.info(f"Initial Engine creation: {t_init:.1f}s")

    out = engine.generate(
        prompt="What is 2+2? Answer with just the number.",
        sampling_params={"max_new_tokens": 10, "temperature": 0.0},
    )
    log.info(f"[{initial}] Output: {out['text'][:100]}")

    # Phase 2: Switch through all text models (3 different architectures)
    # Note: gpt-oss-120b excluded - flashinfer MXFP4 assertion error (K%4)
    switch_sequence = [
        ("qwen2.5-7b", "llama-3.1-70b"),
        ("llama-3.1-70b", "qwen3-32b"),
        ("qwen3-32b", "qwen2.5-7b"),
        ("qwen2.5-7b", "qwen3-32b"),   # Second round to verify stability
        ("qwen3-32b", "llama-3.1-70b"),
    ]

    results = []
    for from_model, to_model in switch_sequence:
        log.info(f"\n{'='*60}")
        log.info(f"RELOAD: {from_model} -> {to_model}")
        log.info(f"{'='*60}")
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
                "switch": f"{from_model} -> {to_model}",
                "time": t_reload,
                "success": False,
            })
            break

        log.info(f"Reload time: {t_reload:.1f}s")

        # Verify generation
        t_gen = time.perf_counter()
        out = engine.generate(
            prompt="What is 2+2? Answer with just the number.",
            sampling_params={"max_new_tokens": 20, "temperature": 0.0},
        )
        t_gen = time.perf_counter() - t_gen
        text = out['text'][:100]
        log.info(f"[{to_model}] Generated in {t_gen:.1f}s: {text}")

        used, _, _ = get_gpu_mem()
        log.info(f"GPU: {used:.1f}GB used")

        results.append({
            "switch": f"{from_model} -> {to_model}",
            "time": t_reload,
            "gen_time": t_gen,
            "success": True,
            "output": text,
        })

    # Shutdown
    log.info("\n--- SHUTDOWN ---")
    engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()
    used, _, _ = get_gpu_mem()
    log.info(f"After shutdown: GPU {used:.1f}GB used")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY: reload_model() Results")
    log.info("=" * 70)

    # Baseline (Engine restart) times from profiling
    baseline = {
        "gpt-oss-120b -> qwen2.5-7b": 14.2,
        "qwen2.5-7b -> llama-3.1-70b": 23.5,
        "llama-3.1-70b -> qwen3-32b": 26.0,
        "qwen3-32b -> gpt-oss-120b": 31.5,
    }

    log.info(f"{'Switch':<35} {'reload_model':>12} {'Engine restart':>14} {'Speedup':>8}")
    log.info("-" * 70)
    total_reload = 0
    total_baseline = 0
    for r in results:
        base = baseline.get(r["switch"], 0)
        speedup = base / r["time"] if r["time"] > 0 and base > 0 else 0
        status = "OK" if r["success"] else "FAIL"
        log.info(
            f"{r['switch']:<35} {r['time']:>10.1f}s {base:>12.1f}s {speedup:>7.1f}x  {status}"
        )
        if r["success"]:
            total_reload += r["time"]
            total_baseline += base
    if total_reload > 0:
        log.info("-" * 70)
        log.info(
            f"{'TOTAL':<35} {total_reload:>10.1f}s {total_baseline:>12.1f}s "
            f"{total_baseline/total_reload:>7.1f}x"
        )


if __name__ == "__main__":
    test_reload()
