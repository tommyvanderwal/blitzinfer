#!/usr/bin/env python3
"""
Test vLLM multiprocessing mode with optimized settings.
Skip memory profiling by specifying kv_cache_memory_bytes.
"""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

import sys
import types
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import gc
import time
import torch


def get_memory():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def test_optimized():
    """Test vLLM multiprocessing with optimizations."""
    print("="*70)
    print("TEST: Optimized vLLM Multiprocessing")
    print("="*70)

    initial = get_memory()
    print(f"\nInitial free memory: {initial:.2f} GB")

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    # Common config for fast loading
    # Note: 96GB reported but only 63GB usable (iGPU unified memory)
    config = dict(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.60,  # ~57GB of 96GB reported
        max_model_len=4096,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=40 * 1024**3,  # 40GB - skip profiling!
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )

    results = []

    # First model
    print("\n>>> Loading first model...")
    start = time.perf_counter()
    llm1 = LLM(**config)
    load1 = time.perf_counter() - start
    print(f"Loaded in {load1:.2f}s")
    results.append(("First load", load1))

    # Generate
    out = llm1.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text}")

    after_load1 = get_memory()
    print(f"Free: {after_load1:.2f} GB (used {initial - after_load1:.2f} GB)")

    # Cleanup first
    print("\n>>> Cleanup...")
    cleanup_start = time.perf_counter()
    del llm1
    gc.collect()
    try:
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    cleanup = time.perf_counter() - cleanup_start
    print(f"Cleanup: {cleanup:.2f}s")
    results.append(("Cleanup", cleanup))

    after_cleanup = get_memory()
    print(f"Free: {after_cleanup:.2f} GB")

    time.sleep(0.5)

    # Second model
    print("\n>>> Loading second model...")
    start = time.perf_counter()
    llm2 = LLM(**config)
    load2 = time.perf_counter() - start
    print(f"Loaded in {load2:.2f}s")
    results.append(("Second load", load2))

    # Generate
    out = llm2.generate(["Joke"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text}")

    after_load2 = get_memory()
    print(f"Free: {after_load2:.2f} GB")

    del llm2
    gc.collect()

    # Summary
    switch_time = cleanup + load2
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, t in results:
        print(f"  {name}: {t:.2f}s")
    print(f"\n  Total switch time: {switch_time:.2f}s")
    print(f"  VRAM used: {initial - after_load1:.2f} GB")
    print("="*70)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    test_optimized()
