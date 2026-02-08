#!/usr/bin/env python3
"""Test if kv_cache_memory_bytes skips profiling and speeds up startup."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import sys
# vllm repo is at blitzinfer/vllm, package is at blitzinfer/vllm/vllm
# Need to add the repo root to path so 'import vllm' finds blitzinfer/vllm/vllm
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import time
import gc


def test_standard():
    """Standard startup with memory profiling."""
    print("\n[Test] Standard startup (with profiling)")

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    from vllm import LLM, SamplingParams

    start = time.time()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        gpu_memory_utilization=0.20,
        max_model_len=1024,
        enforce_eager=True,
    )
    elapsed = time.time() - start
    print(f"  Startup time: {elapsed:.2f}s")

    # Do one inference
    start = time.time()
    output = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    first_gen = time.time() - start
    print(f"  First generation: {first_gen:.2f}s")
    print(f"  Output: {output[0].outputs[0].text}")

    del llm
    gc.collect()

    return elapsed, first_gen


def test_kv_cache_bytes():
    """Startup with kv_cache_memory_bytes set (skips profiling)."""
    print("\n[Test] Startup with kv_cache_memory_bytes")

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    # 4GB KV cache
    kv_bytes = 4 * 1024 * 1024 * 1024

    from vllm import LLM, SamplingParams

    start = time.time()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        kv_cache_memory_bytes=kv_bytes,
        max_model_len=1024,
        enforce_eager=True,
    )
    elapsed = time.time() - start
    print(f"  Startup time: {elapsed:.2f}s")

    start = time.time()
    output = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    first_gen = time.time() - start
    print(f"  First generation: {first_gen:.2f}s")
    print(f"  Output: {output[0].outputs[0].text}")

    del llm
    gc.collect()

    return elapsed, first_gen


def test_num_blocks_override():
    """Startup with num_gpu_blocks_override set."""
    print("\n[Test] Startup with num_gpu_blocks_override")

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    from vllm import LLM, SamplingParams

    start = time.time()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        num_gpu_blocks_override=1000,
        max_model_len=1024,
        enforce_eager=True,
    )
    elapsed = time.time() - start
    print(f"  Startup time: {elapsed:.2f}s")

    start = time.time()
    output = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    first_gen = time.time() - start
    print(f"  First generation: {first_gen:.2f}s")
    print(f"  Output: {output[0].outputs[0].text}")

    del llm
    gc.collect()

    return elapsed, first_gen


if __name__ == '__main__':
    print("=" * 60)
    print("KV Cache Config Test")
    print("=" * 60)
    print("Testing different startup configs...")
    print("(Each test needs separate process for accurate timing)")

    if len(sys.argv) > 1:
        test = sys.argv[1]
        if test == "1":
            test_standard()
        elif test == "2":
            test_kv_cache_bytes()
        elif test == "3":
            test_num_blocks_override()
    else:
        print("\nUsage: python test_kv_skip.py [1|2|3]")
        print("  1 = Standard (with profiling)")
        print("  2 = kv_cache_memory_bytes (skip profiling)")
        print("  3 = num_gpu_blocks_override")
        print("\nRunning test 1 (standard)...")
        test_standard()
