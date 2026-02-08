#!/usr/bin/env python3
"""Test fastest possible startup configuration."""

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


def test_fastest_startup():
    """Test with all optimizations: small batch + kv_cache_memory_bytes."""
    print("\n" + "="*70)
    print("FAST STARTUP TEST")
    print("  - max_num_batched_tokens = 1024")
    print("  - kv_cache_memory_bytes = 4GB")
    print("  - enforce_eager = True (skip CUDA graph)")
    print("="*70)

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    from vllm import LLM, SamplingParams

    # 4GB KV cache
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
    print(f"\n  LLM init time: {init_time:.2f}s")

    # First inference
    start = time.perf_counter()
    out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    first_inf = time.perf_counter() - start
    print(f"  First inference: {first_inf:.2f}s")
    print(f"  Output: {out[0].outputs[0].text[:50]}...")

    # Second inference
    start = time.perf_counter()
    out = llm.generate(["Test"], SamplingParams(max_tokens=10))
    second_inf = time.perf_counter() - start
    print(f"  Second inference: {second_inf:.2f}s")

    # Third inference
    start = time.perf_counter()
    out = llm.generate(["Quick"], SamplingParams(max_tokens=10))
    third_inf = time.perf_counter() - start
    print(f"  Third inference: {third_inf:.2f}s")

    print(f"\n  TOTAL (init + 3 inferences): {init_time + first_inf + second_inf + third_inf:.2f}s")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return init_time, first_inf, second_inf, third_inf


def test_minimal_batch():
    """Test with minimal batch size."""
    print("\n" + "="*70)
    print("MINIMAL BATCH TEST")
    print("  - max_num_batched_tokens = 512")
    print("  - kv_cache_memory_bytes = 4GB")
    print("  - enforce_eager = True")
    print("="*70)

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    from vllm import LLM, SamplingParams

    kv_bytes = 4 * 1024 * 1024 * 1024

    start = time.perf_counter()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        max_model_len=1024,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=kv_bytes,
        enforce_eager=True,
    )
    init_time = time.perf_counter() - start
    print(f"\n  LLM init time: {init_time:.2f}s")

    start = time.perf_counter()
    out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    first_inf = time.perf_counter() - start
    print(f"  First inference: {first_inf:.2f}s")
    print(f"  Output: {out[0].outputs[0].text[:50]}...")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return init_time, first_inf


def test_num_gpu_blocks():
    """Test with num_gpu_blocks_override (alternative to kv_cache_memory_bytes)."""
    print("\n" + "="*70)
    print("NUM_GPU_BLOCKS_OVERRIDE TEST")
    print("  - max_num_batched_tokens = 1024")
    print("  - num_gpu_blocks_override = 1000")
    print("="*70)

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    from vllm import LLM, SamplingParams

    start = time.perf_counter()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        num_gpu_blocks_override=1000,
        enforce_eager=True,
    )
    init_time = time.perf_counter() - start
    print(f"\n  LLM init time: {init_time:.2f}s")

    start = time.perf_counter()
    out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    first_inf = time.perf_counter() - start
    print(f"  First inference: {first_inf:.2f}s")
    print(f"  Output: {out[0].outputs[0].text[:50]}...")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return init_time, first_inf


if __name__ == '__main__':
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    results = {}

    if len(sys.argv) > 1:
        test = sys.argv[1]
        if test == "fast":
            results['fast'] = test_fastest_startup()
        elif test == "minimal":
            results['minimal'] = test_minimal_batch()
        elif test == "blocks":
            results['blocks'] = test_num_gpu_blocks()
    else:
        print("\nUsage: python3.12 test_fast_startup.py [fast|minimal|blocks|all]")
        print("Running 'fast' test by default...")
        results['fast'] = test_fastest_startup()

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for test_name, times in results.items():
        if len(times) >= 2:
            print(f"  {test_name}: init={times[0]:.2f}s, first_inf={times[1]:.2f}s")
