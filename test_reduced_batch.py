#!/usr/bin/env python3
"""Test startup with reduced max_num_batched_tokens."""

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


def test_startup(max_batched_tokens, description):
    """Test startup with specified max_num_batched_tokens."""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"  max_num_batched_tokens = {max_batched_tokens}")
    print(f"{'='*70}")

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
        max_num_batched_tokens=max_batched_tokens,
        enforce_eager=True,
    )
    init_time = time.perf_counter() - start
    print(f"  LLM init time: {init_time:.2f}s")

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

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return init_time, first_inf, second_inf


if __name__ == '__main__':
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    tests = [
        (1024, "Small batch (1024 tokens)"),
        (2048, "Medium batch (2048 tokens)"),
        # (16384, "Default batch (16384 tokens)"),  # Too slow, skip
    ]

    results = {}
    for batch_size, desc in tests:
        times = test_startup(batch_size, desc)
        results[batch_size] = times

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Batch Size':<15} {'Init':>10} {'1st Inf':>10} {'2nd Inf':>10}")
    print("-"*70)
    for batch_size, (init_t, first_t, second_t) in results.items():
        print(f"{batch_size:<15} {init_t:>10.2f}s {first_t:>10.2f}s {second_t:>10.2f}s")
    print("="*70)
