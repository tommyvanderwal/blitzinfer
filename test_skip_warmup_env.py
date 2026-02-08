#!/usr/bin/env python3
"""Test VLLM_SKIP_WARMUP environment variable."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_SKIP_WARMUP'] = '1'  # Skip warmup!
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

import sys
import types

# Patch torchvision._meta_registrations
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta

sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import time
import gc


def test_skip_warmup():
    """Test startup with VLLM_SKIP_WARMUP=1."""
    print("\n" + "="*70)
    print("TEST: VLLM_SKIP_WARMUP=1")
    print("="*70)

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    torch.cuda.empty_cache()
    gc.collect()

    from vllm import LLM, SamplingParams

    kv_bytes = 4 * 1024 * 1024 * 1024

    print(f"\n[{time.strftime('%H:%M:%S')}] Starting LLM init (VLLM_SKIP_WARMUP=1)...")
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
    print(f"[{time.strftime('%H:%M:%S')}] LLM init complete: {init_time:.2f}s")

    # First inference - this IS the warmup now
    print(f"\n[{time.strftime('%H:%M:%S')}] First inference (this IS the warmup)...")
    start = time.perf_counter()
    try:
        out = llm.generate(["Hello, how are you?"], SamplingParams(max_tokens=20))
        first_inf = time.perf_counter() - start
        print(f"[{time.strftime('%H:%M:%S')}] First inference: {first_inf:.2f}s")
        print(f"  Output: {out[0].outputs[0].text}")
    except Exception as e:
        first_inf = time.perf_counter() - start
        print(f"[{time.strftime('%H:%M:%S')}] First inference FAILED after {first_inf:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return init_time, -1, -1

    # Second inference
    print(f"\n[{time.strftime('%H:%M:%S')}] Second inference...")
    start = time.perf_counter()
    try:
        out = llm.generate(["Tell me a joke"], SamplingParams(max_tokens=20))
        second_inf = time.perf_counter() - start
        print(f"[{time.strftime('%H:%M:%S')}] Second inference: {second_inf:.2f}s")
        print(f"  Output: {out[0].outputs[0].text}")
    except Exception as e:
        second_inf = -1
        print(f"Second inference FAILED: {e}")

    # Third inference
    print(f"\n[{time.strftime('%H:%M:%S')}] Third inference...")
    start = time.perf_counter()
    try:
        out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=10))
        third_inf = time.perf_counter() - start
        print(f"[{time.strftime('%H:%M:%S')}] Third inference: {third_inf:.2f}s")
        print(f"  Output: {out[0].outputs[0].text}")
    except Exception as e:
        third_inf = -1
        print(f"Third inference FAILED: {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (with VLLM_SKIP_WARMUP=1)")
    print("="*70)
    print(f"  Init time:        {init_time:.2f}s")
    print(f"  First inference:  {first_inf:.2f}s (this was the warmup)")
    print(f"  Second inference: {second_inf:.2f}s")
    if third_inf > 0:
        print(f"  Third inference:  {third_inf:.2f}s")
    print(f"  Time to first response: {init_time + first_inf:.2f}s")
    print("="*70)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return init_time, first_inf, second_inf


if __name__ == '__main__':
    test_skip_warmup()
