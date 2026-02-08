#!/usr/bin/env python3
"""Test startup optimizations with ROCm 7.2 + Python 3.12."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# os.environ['AMD_SERIALIZE_KERNEL'] = '1'  # Disabled for speed

import sys
import types

# Patch torchvision._meta_registrations to avoid nms operator issue
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta

sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import time
import gc


def test_standard_startup():
    """Standard startup with memory profiling."""
    print("\n[Test 1] Standard startup")

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    from vllm import LLM, SamplingParams

    start = time.time()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        max_model_len=1024,
        enforce_eager=True,
    )
    startup_time = time.time() - start
    print(f"  Startup: {startup_time:.2f}s")

    # First inference
    start = time.time()
    out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    first_gen = time.time() - start
    print(f"  First gen: {first_gen:.2f}s")
    print(f"  Output: {out[0].outputs[0].text}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return startup_time, first_gen


def test_kv_cache_bytes():
    """Startup with kv_cache_memory_bytes (skip profiling)."""
    print("\n[Test 2] With kv_cache_memory_bytes")

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    # 4GB KV cache
    kv_bytes = 4 * 1024 * 1024 * 1024

    from vllm import LLM, SamplingParams

    start = time.time()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        kv_cache_memory_bytes=kv_bytes,
        max_model_len=1024,
        enforce_eager=True,
    )
    startup_time = time.time() - start
    print(f"  Startup: {startup_time:.2f}s")

    start = time.time()
    out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    first_gen = time.time() - start
    print(f"  First gen: {first_gen:.2f}s")
    print(f"  Output: {out[0].outputs[0].text}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return startup_time, first_gen


def test_num_blocks_override():
    """Startup with num_gpu_blocks_override."""
    print("\n[Test 3] With num_gpu_blocks_override")

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    from vllm import LLM, SamplingParams

    start = time.time()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        num_gpu_blocks_override=500,
        max_model_len=1024,
        enforce_eager=True,
    )
    startup_time = time.time() - start
    print(f"  Startup: {startup_time:.2f}s")

    start = time.time()
    out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    first_gen = time.time() - start
    print(f"  First gen: {first_gen:.2f}s")
    print(f"  Output: {out[0].outputs[0].text}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return startup_time, first_gen


if __name__ == '__main__':
    print("=" * 60)
    print("Startup Optimization Test - ROCm 7.2 + Python 3.12")
    print("=" * 60)

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    if len(sys.argv) > 1:
        test = sys.argv[1]
        if test == "1":
            test_standard_startup()
        elif test == "2":
            test_kv_cache_bytes()
        elif test == "3":
            test_num_blocks_override()
    else:
        print("\nUsage: python3.12 test_startup_opt.py [1|2|3]")
        print("  1 = Standard startup")
        print("  2 = kv_cache_memory_bytes (skip profiling)")
        print("  3 = num_gpu_blocks_override")
        print("\nRunning test 1...")
        test_standard_startup()
