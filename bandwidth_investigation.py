#!/usr/bin/env python3
"""
Investigate the bandwidth discrepancy:
- 20 GB/s for single large tensor
- 2.5 GB/s for many small tensors (safetensors)
"""

import os
import gc
import time
from pathlib import Path

os.environ['HIP_VISIBLE_DEVICES'] = '0'

import torch
from safetensors.torch import load_file


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def test_single_vs_many():
    print("=" * 70)
    print("BANDWIDTH: SINGLE LARGE vs MANY SMALL TENSORS")
    print("=" * 70)

    total_gb = 7.0  # 7GB total
    total_elements = int(total_gb * 1024**3 / 2)  # float16

    print(f"Total size: {total_gb} GB")

    # Test 1: Single large tensor
    print("\n>>> Test 1: Single contiguous tensor")
    cpu_tensor = torch.randn(total_elements, dtype=torch.float16)
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    gpu_tensor = cpu_tensor.to('cuda')
    torch.cuda.synchronize()
    single_time = (time.perf_counter() - t0) * 1000
    bw = total_gb / (single_time / 1000)
    print(f"  Time: {single_time:.0f}ms ({bw:.2f} GB/s)")

    del gpu_tensor
    gc.collect()
    torch.cuda.empty_cache()

    # Test 2: Many small tensors (same total size)
    print("\n>>> Test 2: 339 small tensors (like model weights)")
    num_tensors = 339
    elements_per = total_elements // num_tensors
    cpu_tensors = [torch.randn(elements_per, dtype=torch.float16) for _ in range(num_tensors)]
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    gpu_tensors = [t.to('cuda') for t in cpu_tensors]
    torch.cuda.synchronize()
    many_time = (time.perf_counter() - t0) * 1000
    bw = total_gb / (many_time / 1000)
    print(f"  Time: {many_time:.0f}ms ({bw:.2f} GB/s)")

    del gpu_tensors
    gc.collect()
    torch.cuda.empty_cache()

    # Test 3: Many small tensors with non_blocking
    print("\n>>> Test 3: 339 tensors with non_blocking=True")
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    gpu_tensors = [t.to('cuda', non_blocking=True) for t in cpu_tensors]
    torch.cuda.synchronize()
    async_time = (time.perf_counter() - t0) * 1000
    bw = total_gb / (async_time / 1000)
    print(f"  Time: {async_time:.0f}ms ({bw:.2f} GB/s)")

    del gpu_tensors
    gc.collect()
    torch.cuda.empty_cache()

    # Test 4: Batch transfer using torch.cat
    print("\n>>> Test 4: Concatenate then transfer (single DMA)")
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    big_tensor = torch.cat(cpu_tensors)
    gpu_big = big_tensor.to('cuda')
    torch.cuda.synchronize()
    cat_time = (time.perf_counter() - t0) * 1000
    bw = total_gb / (cat_time / 1000)
    print(f"  Time: {cat_time:.0f}ms ({bw:.2f} GB/s)")

    del big_tensor, gpu_big
    gc.collect()
    torch.cuda.empty_cache()

    # Test 5: Pinned memory for small tensors
    print("\n>>> Test 5: Pinned memory + small tensors")
    pinned_tensors = [torch.empty_like(t, pin_memory=True).copy_(t) for t in cpu_tensors]
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    gpu_tensors = [t.to('cuda', non_blocking=True) for t in pinned_tensors]
    torch.cuda.synchronize()
    pinned_time = (time.perf_counter() - t0) * 1000
    bw = total_gb / (pinned_time / 1000)
    print(f"  Time: {pinned_time:.0f}ms ({bw:.2f} GB/s)")

    del gpu_tensors, pinned_tensors
    gc.collect()
    torch.cuda.empty_cache()

    # Test 6: Pre-allocate GPU memory, then copy
    print("\n>>> Test 6: Pre-allocate GPU + copy")
    gc.collect()
    torch.cuda.empty_cache()

    # Pre-allocate
    gpu_tensors = [torch.empty_like(t, device='cuda') for t in cpu_tensors]
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for gpu_t, cpu_t in zip(gpu_tensors, cpu_tensors):
        gpu_t.copy_(cpu_t, non_blocking=True)
    torch.cuda.synchronize()
    prealloc_time = (time.perf_counter() - t0) * 1000
    bw = total_gb / (prealloc_time / 1000)
    print(f"  Time: {prealloc_time:.0f}ms ({bw:.2f} GB/s)")

    del gpu_tensors
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Single large tensor:      {single_time:.0f}ms ({total_gb / (single_time / 1000):.1f} GB/s)")
    print(f"339 small tensors:        {many_time:.0f}ms ({total_gb / (many_time / 1000):.1f} GB/s)")
    print(f"339 non_blocking:         {async_time:.0f}ms ({total_gb / (async_time / 1000):.1f} GB/s)")
    print(f"Concat then transfer:     {cat_time:.0f}ms ({total_gb / (cat_time / 1000):.1f} GB/s)")
    print(f"Pinned + non_blocking:    {pinned_time:.0f}ms ({total_gb / (pinned_time / 1000):.1f} GB/s)")
    print(f"Pre-alloc + copy:         {prealloc_time:.0f}ms ({total_gb / (prealloc_time / 1000):.1f} GB/s)")

    print(f"\nOverhead from fragmentation: {many_time / single_time:.1f}x slower")

    del cpu_tensor, cpu_tensors


if __name__ == '__main__':
    test_single_vs_many()
