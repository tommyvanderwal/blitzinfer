#!/usr/bin/env python3
"""
Test pinned memory transfer speeds on APU unified memory.
The 780M has unified memory - CPU and GPU share DDR5.
Proper memory mapping should be near-instantaneous (just page table updates).
"""

import os
import sys
import gc
import time
import types

os.environ['HIP_VISIBLE_DEVICES'] = '0'

import torch


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def test_transfer_methods():
    print("=" * 70)
    print("UNIFIED MEMORY TRANSFER TEST")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {get_mem():.2f} GB free")

    # Test size: 14 GB (model size)
    size_gb = 14
    size_elements = int(size_gb * 1024**3 / 4)  # float32

    print(f"\nTest size: {size_gb} GB ({size_elements} elements)")

    # Method 1: Normal CPU → GPU (unpinned)
    print("\n>>> Method 1: Normal CPU tensor → GPU (unpinned)")
    cpu_tensor = torch.randn(size_elements, dtype=torch.float32)
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    gpu_tensor = cpu_tensor.to('cuda')
    torch.cuda.synchronize()
    normal_time = (time.perf_counter() - t0) * 1000
    bw = size_gb / (normal_time / 1000)
    print(f"  Time: {normal_time:.0f}ms ({bw:.2f} GB/s)")

    del gpu_tensor
    gc.collect()
    torch.cuda.empty_cache()

    # Method 2: Pinned CPU → GPU
    print("\n>>> Method 2: Pinned CPU tensor → GPU")
    pinned_tensor = torch.empty(size_elements, dtype=torch.float32, pin_memory=True)
    pinned_tensor.copy_(cpu_tensor)

    t0 = time.perf_counter()
    gpu_tensor = pinned_tensor.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    pinned_time = (time.perf_counter() - t0) * 1000
    bw = size_gb / (pinned_time / 1000)
    print(f"  Time: {pinned_time:.0f}ms ({bw:.2f} GB/s)")

    del gpu_tensor, pinned_tensor
    gc.collect()
    torch.cuda.empty_cache()

    # Method 3: cuda.HalfTensor directly
    print("\n>>> Method 3: Direct GPU allocation + copy")
    t0 = time.perf_counter()
    gpu_tensor = torch.empty(size_elements, dtype=torch.float32, device='cuda')
    gpu_tensor.copy_(cpu_tensor)
    torch.cuda.synchronize()
    direct_time = (time.perf_counter() - t0) * 1000
    bw = size_gb / (direct_time / 1000)
    print(f"  Time: {direct_time:.0f}ms ({bw:.2f} GB/s)")

    del gpu_tensor
    gc.collect()
    torch.cuda.empty_cache()

    # Method 4: Check if unified memory is being used
    print("\n>>> Method 4: Check unified memory (HIP managed)")
    try:
        # On APUs, managed memory should allow zero-copy access
        t0 = time.perf_counter()
        # Try to use the CPU tensor directly on GPU
        # This might work on unified memory systems
        result = cpu_tensor.cuda()
        torch.cuda.synchronize()
        managed_time = (time.perf_counter() - t0) * 1000
        bw = size_gb / (managed_time / 1000)
        print(f"  Time: {managed_time:.0f}ms ({bw:.2f} GB/s)")
        del result
    except Exception as e:
        print(f"  Not supported: {e}")

    gc.collect()
    torch.cuda.empty_cache()

    # Method 5: Float16 (half the data)
    print("\n>>> Method 5: Float16 transfer (half the data)")
    cpu_f16 = cpu_tensor[:size_elements//2].to(torch.float16)
    print(f"  Size: {size_gb/2:.1f} GB")

    t0 = time.perf_counter()
    gpu_f16 = cpu_f16.to('cuda')
    torch.cuda.synchronize()
    f16_time = (time.perf_counter() - t0) * 1000
    bw = (size_gb/2) / (f16_time / 1000)
    print(f"  Time: {f16_time:.0f}ms ({bw:.2f} GB/s)")

    del gpu_f16, cpu_f16
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Normal (unpinned): {normal_time:.0f}ms")
    print(f"Pinned memory:     {pinned_time:.0f}ms")
    print(f"Direct copy:       {direct_time:.0f}ms")
    print(f"Float16 (7GB):     {f16_time:.0f}ms")

    if pinned_time < normal_time:
        print(f"\nPinned speedup: {normal_time / pinned_time:.2f}x")

    del cpu_tensor


def test_zero_copy():
    """Test if zero-copy is possible on unified memory."""
    print("\n" + "=" * 70)
    print("ZERO-COPY TEST (Unified Memory)")
    print("=" * 70)

    size_elements = 256 * 1024 * 1024  # 1GB as float32
    size_gb = size_elements * 4 / (1024**3)

    print(f"Test size: {size_gb:.2f} GB")

    # Allocate on GPU
    print("\n>>> GPU allocation")
    t0 = time.perf_counter()
    gpu_tensor = torch.randn(size_elements, dtype=torch.float32, device='cuda')
    torch.cuda.synchronize()
    alloc_time = (time.perf_counter() - t0) * 1000
    print(f"  Allocation: {alloc_time:.0f}ms")

    # Access from CPU (should be zero-copy on unified memory)
    print("\n>>> CPU access of GPU tensor (zero-copy test)")
    t0 = time.perf_counter()
    cpu_view = gpu_tensor.cpu()
    access_time = (time.perf_counter() - t0) * 1000
    bw = size_gb / (access_time / 1000) if access_time > 0 else float('inf')
    print(f"  Access: {access_time:.0f}ms ({bw:.2f} GB/s)")

    # Compute on GPU
    print("\n>>> GPU compute")
    t0 = time.perf_counter()
    result = gpu_tensor * 2 + 1
    torch.cuda.synchronize()
    compute_time = (time.perf_counter() - t0) * 1000
    print(f"  Compute: {compute_time:.0f}ms")

    del gpu_tensor, cpu_view, result
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_transfer_methods()
    test_zero_copy()
