#!/usr/bin/env python3
"""Test second startup timing with optimized config."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import sys
import types

# Patch torchvision._meta_registrations
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta

sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import time
import gc


def create_llm(run_number):
    """Create LLM with optimized settings."""
    from vllm import LLM, SamplingParams

    kv_bytes = 4 * 1024 * 1024 * 1024

    start = time.perf_counter()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=kv_bytes,
        enforce_eager=True,
    )
    init_time = time.perf_counter() - start
    print(f"  Run {run_number} init: {init_time:.2f}s")

    # First inference
    start = time.perf_counter()
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    inf_time = time.perf_counter() - start
    print(f"  Run {run_number} inference: {inf_time:.2f}s")

    return llm, init_time, inf_time


def main():
    print("\n" + "="*70)
    print("SECOND STARTUP TEST (Optimized Config)")
    print("Testing if Python import caching helps")
    print("="*70)

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Pre-import vllm to cache
    from vllm import LLM, SamplingParams
    print("vLLM imported and cached")

    results = []

    # Run 1
    print("\n--- Run 1 (first load) ---")
    torch.cuda.empty_cache()
    gc.collect()
    llm1, init1, inf1 = create_llm(1)
    del llm1
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    results.append(('Run 1', init1, inf1))

    # Run 2
    print("\n--- Run 2 (second load, imports cached) ---")
    torch.cuda.empty_cache()
    gc.collect()
    llm2, init2, inf2 = create_llm(2)
    del llm2
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    results.append(('Run 2', init2, inf2))

    # Run 3
    print("\n--- Run 3 (third load) ---")
    llm3, init3, inf3 = create_llm(3)
    del llm3
    gc.collect()
    torch.cuda.empty_cache()
    results.append(('Run 3', init3, inf3))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Run':<10} {'Init':>10} {'Inference':>10} {'Total':>10}")
    print("-"*70)
    for run, init_t, inf_t in results:
        print(f"{run:<10} {init_t:>10.2f}s {inf_t:>10.2f}s {init_t+inf_t:>10.2f}s")
    print("="*70)

    improvement = results[0][1] - results[1][1]
    pct = (improvement / results[0][1]) * 100 if results[0][1] > 0 else 0
    print(f"\nRun 2 vs Run 1: {improvement:+.2f}s ({pct:.1f}% improvement)")


if __name__ == '__main__':
    main()
