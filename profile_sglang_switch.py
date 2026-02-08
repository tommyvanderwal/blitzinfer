#!/usr/bin/env python3
"""Profile SGLang Engine initialization stages to identify switching bottlenecks.

Measures each sub-stage of Engine() construction:
- Process spawning
- Tokenizer loading
- Model construction (nn.Module)
- Weight loading (disk I/O + parsing)
- KV cache profiling
- Engine warmup

Run on remote RTX PRO 6000:
    ssh tommy@192.168.2.90
    cd ~/pythonprojects/blitzinfer
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python profile_sglang_switch.py
"""

import gc
import os
import sys
import time
import json
import torch
import logging

# SGLang Engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sglang", "python"))
from sglang.srt.entrypoints.engine import Engine as SglEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("profile")


# Models to profile (same as server.py)
MODELS = {
    "qwen2.5-7b": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "mem_fraction_static": 0.94,
    },
    "llama-3.1-70b": {
        "model_path": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "mem_fraction_static": 0.94,
        "quantization": "awq",
    },
    "qwen3-32b": {
        "model_path": "Qwen/Qwen3-32B",
        "mem_fraction_static": 0.94,
    },
    "kimi-vl": {
        "model_path": "moonshotai/Kimi-VL-A3B-Thinking-2506",
        "mem_fraction_static": 0.94,
        "mm_attention_backend": "sdpa",
    },
    "gpt-oss-120b": {
        "model_path": "openai/gpt-oss-120b",
        "mem_fraction_static": 0.94,
        "context_length": 131072,
        "tool_call_parser": "harmony",
        "moe_runner_backend": "triton_kernel",
    },
}


def get_gpu_mem():
    """Return (used_gb, free_gb, total_gb)"""
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024**3
    free_gb = free / 1024**3
    total_gb = total / 1024**3
    return used, free_gb, total_gb


def cleanup_gpu():
    """Aggressively release GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()


def create_engine(model_id: str, model_cfg: dict) -> tuple:
    """Create an Engine and return (engine, timing_dict)."""
    timings = {}

    # Baseline GPU
    used_before, _, _ = get_gpu_mem()
    timings["gpu_before_gb"] = used_before

    t_total_start = time.perf_counter()

    # Build kwargs
    engine_kwargs = {
        "model_path": model_cfg["model_path"],
        "mem_fraction_static": model_cfg.get("mem_fraction_static", 0.94),
        "trust_remote_code": True,
        "disable_cuda_graph": True,
        "attention_backend": "triton",
        "log_level": "info",
    }
    for k, v in model_cfg.items():
        if k not in ("model_path", "mem_fraction_static"):
            engine_kwargs[k] = v

    log.info(f"[{model_id}] Creating SglEngine with kwargs: {json.dumps({k: str(v) for k, v in engine_kwargs.items()}, indent=2)}")

    t_engine_start = time.perf_counter()
    engine = SglEngine(**engine_kwargs)
    t_engine_end = time.perf_counter()

    timings["engine_init_s"] = t_engine_end - t_engine_start

    # Post-init GPU
    used_after, free_after, total = get_gpu_mem()
    timings["gpu_after_gb"] = used_after
    timings["gpu_delta_gb"] = used_after - used_before

    # Quick generation test to ensure engine works
    t_gen_start = time.perf_counter()
    out = engine.generate(prompt="Say hello", sampling_params={"max_new_tokens": 10, "temperature": 0.0})
    t_gen_end = time.perf_counter()
    timings["first_gen_s"] = t_gen_end - t_gen_start
    timings["gen_text"] = out["text"][:100]

    t_total_end = time.perf_counter()
    timings["total_s"] = t_total_end - t_total_start

    return engine, timings


def shutdown_engine(engine, model_id: str) -> dict:
    """Shutdown engine and return timing dict."""
    timings = {}
    used_before, _, _ = get_gpu_mem()
    timings["gpu_before_shutdown_gb"] = used_before

    t_start = time.perf_counter()
    engine.shutdown()
    t_shutdown = time.perf_counter()
    timings["shutdown_s"] = t_shutdown - t_start

    # Cleanup
    del engine
    cleanup_gpu()
    t_cleanup = time.perf_counter()
    timings["cleanup_s"] = t_cleanup - t_shutdown

    used_after, _, _ = get_gpu_mem()
    timings["gpu_after_shutdown_gb"] = used_after
    timings["gpu_freed_gb"] = used_before - used_after
    timings["total_shutdown_s"] = t_cleanup - t_start

    return timings


def profile_switch(from_model: str, to_model: str, from_engine) -> dict:
    """Profile a full model switch: shutdown from_model, start to_model."""
    result = {
        "from": from_model,
        "to": to_model,
    }

    log.info(f"\n{'='*60}")
    log.info(f"SWITCH: {from_model} -> {to_model}")
    log.info(f"{'='*60}")

    t_switch_start = time.perf_counter()

    # Phase 1: Shutdown
    shutdown_timings = shutdown_engine(from_engine, from_model)
    result["shutdown"] = shutdown_timings
    log.info(f"  Shutdown: {shutdown_timings['total_shutdown_s']:.1f}s "
             f"(engine={shutdown_timings['shutdown_s']:.1f}s, "
             f"cleanup={shutdown_timings['cleanup_s']:.1f}s, "
             f"freed={shutdown_timings['gpu_freed_gb']:.1f}GB)")

    # Phase 2: Load new model
    to_cfg = MODELS[to_model]
    new_engine, load_timings = create_engine(to_model, to_cfg)
    result["load"] = load_timings
    log.info(f"  Load: {load_timings['engine_init_s']:.1f}s "
             f"(GPU: {load_timings['gpu_delta_gb']:.1f}GB)")

    t_switch_end = time.perf_counter()
    result["total_switch_s"] = t_switch_end - t_switch_start

    log.info(f"  TOTAL SWITCH: {result['total_switch_s']:.1f}s")

    return new_engine, result


def main():
    log.info("=" * 80)
    log.info("SGLANG ENGINE SWITCHING PROFILER")
    log.info("=" * 80)

    used, free, total = get_gpu_mem()
    log.info(f"GPU: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")

    results = []

    # Test sequence: cycle through models to measure cold and warm switches
    # Start with smallest model, then cycle through larger ones
    switch_sequence = [
        "qwen2.5-7b",    # Small, fast baseline
        "llama-3.1-70b",  # Large AWQ
        "qwen3-32b",     # Large bf16
        "kimi-vl",       # Vision model
        "gpt-oss-120b",  # Largest, MoE
        "qwen2.5-7b",    # Back to small (second load = warm?)
    ]

    # Load initial model
    initial_model = switch_sequence[0]
    log.info(f"\n--- INITIAL LOAD: {initial_model} ---")
    engine, init_timings = create_engine(initial_model, MODELS[initial_model])
    results.append({
        "type": "initial_load",
        "model": initial_model,
        "timings": init_timings,
    })
    log.info(f"Initial load of {initial_model}: {init_timings['engine_init_s']:.1f}s "
             f"(GPU: {init_timings['gpu_delta_gb']:.1f}GB)")

    # Perform switches
    current_model = initial_model
    for next_model in switch_sequence[1:]:
        engine, switch_result = profile_switch(current_model, next_model, engine)
        results.append({
            "type": "switch",
            "result": switch_result,
        })
        current_model = next_model

    # Final shutdown
    log.info(f"\n--- FINAL SHUTDOWN: {current_model} ---")
    final_shutdown = shutdown_engine(engine, current_model)
    results.append({
        "type": "final_shutdown",
        "model": current_model,
        "timings": final_shutdown,
    })

    # Print summary
    log.info("\n" + "=" * 80)
    log.info("PROFILING SUMMARY")
    log.info("=" * 80)

    log.info(f"\n{'From':<20} {'To':<20} {'Shutdown':>10} {'Load':>10} {'Total':>10}")
    log.info("-" * 72)

    for r in results:
        if r["type"] == "initial_load":
            log.info(f"{'(cold)':<20} {r['model']:<20} {'N/A':>10} "
                     f"{r['timings']['engine_init_s']:>9.1f}s {'N/A':>10}")
        elif r["type"] == "switch":
            sr = r["result"]
            log.info(f"{sr['from']:<20} {sr['to']:<20} "
                     f"{sr['shutdown']['total_shutdown_s']:>9.1f}s "
                     f"{sr['load']['engine_init_s']:>9.1f}s "
                     f"{sr['total_switch_s']:>9.1f}s")

    # Save detailed results
    output_file = "/tmp/sglang_switch_profile.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()
