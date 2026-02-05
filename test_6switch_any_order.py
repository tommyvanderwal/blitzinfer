#!/usr/bin/env python3
"""Test 6+ model switches in random order to verify all architectures work.

This tests the critical requirement: switching from ANY model to ANY other model
must work reliably without memory leaks.
"""
import os
import gc
import time
import random

# Single-process mode for faster switching
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from vllm import LLM, SamplingParams

from blitzinfer.engine.cleanup import full_cleanup, get_gpu_memory_info, log_gpu_memory

# Test models with different architectures/quantizations
MODELS = {
    "gpt-oss-120b": {
        "path": "openai/gpt-oss-120b",
        "dtype": "auto",  # MXFP4 quantized
        "gpu_util": 0.90,
        "test_prompt": "What is 2+2? Answer briefly.",
    },
    "qwen3-32b": {
        "path": "Qwen/Qwen3-32B",
        "dtype": "bfloat16",
        "gpu_util": 0.90,
        "test_prompt": "What is 2+2? Answer briefly.",
    },
    "qwen2.5-7b": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": "bfloat16",
        "gpu_util": 0.90,
        "test_prompt": "What is 2+2? Answer briefly.",
    },
}


def load_and_test_model(model_name: str) -> tuple[LLM, bool]:
    """Load a model and verify it generates sensible output."""
    config = MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"Loading: {model_name}")
    print(f"{'='*60}")

    log_gpu_memory("before load")
    start = time.perf_counter()

    llm = LLM(
        model=config["path"],
        dtype=config["dtype"],
        gpu_memory_utilization=config["gpu_util"],
        trust_remote_code=True,
        max_model_len=4096,  # Short context for testing
        enforce_eager=True,
    )

    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.1f}s")
    log_gpu_memory("after load")

    # Test generation
    print(f"Testing generation...")
    sampling = SamplingParams(max_tokens=50, temperature=0.7)
    outputs = llm.generate([config["test_prompt"]], sampling)

    text = outputs[0].outputs[0].text.strip()
    print(f"Output: {text[:100]}...")

    # Simple validation - should mention "4" for 2+2
    is_valid = "4" in text or "four" in text.lower()
    print(f"Validation: {'PASS' if is_valid else 'FAIL'}")

    return llm, is_valid


def cleanup_model(llm) -> float:
    """Clean up model and return freed memory."""
    print(f"\nCleaning up...")
    log_gpu_memory("before cleanup")

    start = time.perf_counter()
    freed = full_cleanup(llm, nuclear=True, force_free=True)
    cleanup_time = time.perf_counter() - start

    print(f"Cleanup time: {cleanup_time:.1f}s")
    log_gpu_memory("after cleanup")

    return freed


def main():
    print("="*70)
    print("Model Switching Stress Test - 6+ switches in random order")
    print("="*70)

    # Track memory across switches
    initial_mem = get_gpu_memory_info()
    print(f"\nInitial GPU state: {initial_mem['used_gb']:.1f}GB used, {initial_mem['free_gb']:.1f}GB free")

    # Create random switch sequence (6+ switches)
    model_names = list(MODELS.keys())
    switch_sequence = []

    # Start with gpt-oss-120b (the MXFP4 model that had issues)
    current = "gpt-oss-120b"
    switch_sequence.append(current)

    # Add 11 more random switches (total 12) to thoroughly test
    for _ in range(11):
        # Pick a different model
        candidates = [m for m in model_names if m != current]
        next_model = random.choice(candidates)
        switch_sequence.append(next_model)
        current = next_model

    print(f"\nSwitch sequence: {' -> '.join(switch_sequence)}")
    print(f"Total switches: {len(switch_sequence)}")

    # Run the switches
    results = []
    llm = None

    for i, model_name in enumerate(switch_sequence):
        print(f"\n{'#'*70}")
        print(f"# SWITCH {i+1}/{len(switch_sequence)}: {model_name}")
        print(f"{'#'*70}")

        # Cleanup previous model
        if llm is not None:
            freed = cleanup_model(llm)
            llm = None
            gc.collect()
            torch.cuda.empty_cache()

        # Load and test new model
        try:
            llm, is_valid = load_and_test_model(model_name)
            mem = get_gpu_memory_info()
            results.append({
                "switch": i + 1,
                "model": model_name,
                "success": True,
                "valid": is_valid,
                "used_gb": mem["used_gb"],
                "free_gb": mem["free_gb"],
            })
        except Exception as e:
            print(f"\n!!! LOAD FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "switch": i + 1,
                "model": model_name,
                "success": False,
                "valid": False,
                "error": str(e),
            })
            # Try to recover
            llm = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Final cleanup
    if llm is not None:
        cleanup_model(llm)
        llm = None

    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    final_mem = get_gpu_memory_info()
    total_drift = final_mem["used_gb"] - initial_mem["used_gb"]

    print(f"\n{'Switch':<8} {'Model':<20} {'Success':<10} {'Valid':<8} {'Used GB':<10} {'Free GB':<10}")
    print("-" * 70)

    for r in results:
        if r["success"]:
            print(f"{r['switch']:<8} {r['model']:<20} {'YES':<10} {'YES' if r['valid'] else 'NO':<8} {r['used_gb']:<10.1f} {r['free_gb']:<10.1f}")
        else:
            print(f"{r['switch']:<8} {r['model']:<20} {'FAIL':<10} {'-':<8} {'-':<10} {'-':<10}")

    print("-" * 70)
    print(f"\nFinal GPU: {final_mem['used_gb']:.1f}GB used, {final_mem['free_gb']:.1f}GB free")
    print(f"Memory drift: {total_drift:+.1f}GB")

    # Determine pass/fail
    all_success = all(r["success"] for r in results)
    all_valid = all(r.get("valid", False) for r in results if r["success"])
    memory_ok = final_mem["free_gb"] >= 90.0

    print(f"\n{'='*70}")
    if all_success and all_valid and memory_ok:
        print("OVERALL: PASS - All switches successful, all validations passed, memory OK")
    else:
        issues = []
        if not all_success:
            issues.append("some switches failed")
        if not all_valid:
            issues.append("some validations failed")
        if not memory_ok:
            issues.append(f"memory leak (only {final_mem['free_gb']:.1f}GB free, need 90+GB)")
        print(f"OVERALL: FAIL - {', '.join(issues)}")
    print("="*70)


if __name__ == "__main__":
    main()
