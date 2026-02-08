#!/usr/bin/env python3
"""
Baseline vLLM switching - NO optimizations.

This measures the actual baseline performance without any patching.
"""

import gc
import os
import sys
import time
import types
from typing import Any, Dict, List, Tuple

os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch


def get_mem() -> Tuple[float, float]:
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**3)
    return used, free / (1024**3)


def cleanup_model(llm):
    if llm is not None:
        del llm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def main():
    from vllm import LLM, SamplingParams

    print("="*70)
    print("BASELINE vLLM SWITCHING (NO OPTIMIZATIONS)")
    print("="*70)

    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    used, free = get_mem()
    print(f"\nInitial GPU: {used:.1f}GB used, {free:.1f}GB free")

    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    qwen = "Qwen/Qwen2.5-7B-Instruct"
    mistral = "mistralai/Mistral-7B-Instruct-v0.3"

    results = []

    # Initialize with Qwen
    print("\n" + "-"*70)
    print("PHASE 1: Initialize Qwen")
    print("-"*70)

    t0 = time.perf_counter()
    llm = LLM(model=qwen, **config)
    init_time = (time.perf_counter() - t0) * 1000

    outputs = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=20, temperature=0.7))
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "4" in text
    print(f"Time: {init_time:.0f}ms - {'PASS' if valid else 'FAIL'}")
    results.append(("Init Qwen", init_time, None, valid))

    # Switch to Mistral
    print("\n" + "-"*70)
    print("PHASE 2: Switch Qwen → Mistral")
    print("-"*70)

    t_total = time.perf_counter()

    t0 = time.perf_counter()
    cleanup_model(llm)
    llm = None
    cleanup_time = (time.perf_counter() - t0) * 1000

    used, free = get_mem()
    print(f"After cleanup: {used:.1f}GB used (cleanup: {cleanup_time:.0f}ms)")

    t0 = time.perf_counter()
    llm = LLM(model=mistral, **config)
    load_time = (time.perf_counter() - t0) * 1000

    total_time = (time.perf_counter() - t_total) * 1000

    outputs = llm.generate(["Capital of France?"], SamplingParams(max_tokens=20, temperature=0.7))
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "paris" in text.lower()
    print(f"Time: {total_time:.0f}ms (cleanup: {cleanup_time:.0f}ms, load: {load_time:.0f}ms) - {'PASS' if valid else 'FAIL'}")
    results.append(("Qwen→Mistral", total_time, {"cleanup": cleanup_time, "load": load_time}, valid))

    # Switch back to Qwen
    print("\n" + "-"*70)
    print("PHASE 3: Switch Mistral → Qwen")
    print("-"*70)

    t_total = time.perf_counter()

    t0 = time.perf_counter()
    cleanup_model(llm)
    llm = None
    cleanup_time = (time.perf_counter() - t0) * 1000

    used, free = get_mem()
    print(f"After cleanup: {used:.1f}GB used (cleanup: {cleanup_time:.0f}ms)")

    t0 = time.perf_counter()
    llm = LLM(model=qwen, **config)
    load_time = (time.perf_counter() - t0) * 1000

    total_time = (time.perf_counter() - t_total) * 1000

    outputs = llm.generate(["Largest planet?"], SamplingParams(max_tokens=20, temperature=0.7))
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "jupiter" in text.lower()
    print(f"Time: {total_time:.0f}ms (cleanup: {cleanup_time:.0f}ms, load: {load_time:.0f}ms) - {'PASS' if valid else 'FAIL'}")
    results.append(("Mistral→Qwen", total_time, {"cleanup": cleanup_time, "load": load_time}, valid))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Phase':<20} {'Total':<12} {'Cleanup':<12} {'Load':<12} {'Status'}")
    print("-"*60)
    for name, total, timings, valid in results:
        if timings:
            print(f"{name:<20} {total:<12.0f} {timings['cleanup']:<12.0f} {timings['load']:<12.0f} {'PASS' if valid else 'FAIL'}")
        else:
            print(f"{name:<20} {total:<12.0f} {'-':<12} {'-':<12} {'PASS' if valid else 'FAIL'}")

    switch_results = [r for r in results if r[2] is not None]
    if switch_results:
        avg = sum(r[1] for r in switch_results) / len(switch_results)
        print(f"\nAverage switch time: {avg:.0f}ms")

    cleanup_model(llm)
    used, free = get_mem()
    print(f"\nFinal GPU: {used:.1f}GB used, {free:.1f}GB free")
    print("="*70)


if __name__ == '__main__':
    main()
