#!/usr/bin/env python3
"""Test pinned memory transfer speed with 1GB chunks.

This is a quick test to verify transfer speed before running full model tests.
"""

import torch
import gc
import time


def get_mem():
    with open('/proc/meminfo', 'r') as f:
        mem = {}
        for line in f:
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                val = int(parts[1].strip().split()[0]) / 1024 / 1024  # GB
                mem[key] = val
    return mem


def main():
    print("=" * 60)
    print("PINNED MEMORY TRANSFER SPEED TEST")
    print("=" * 60)

    m0 = get_mem()
    print(f"Start: MemAvail={m0['MemAvailable']:.1f}GB, Shmem={m0['Shmem']:.1f}GB\n")

    # Allocate 40GB in 1GB chunks (safe size)
    print("Allocating 40GB pinned memory (40x 1GB chunks)...")
    t0 = time.perf_counter()
    chunks = []
    for i in range(40):
        chunk = torch.empty(1 * 1024**3, dtype=torch.uint8, device='cpu', pin_memory=True)
        # Fill with some data
        chunk.fill_(i % 256)
        chunks.append(chunk)
    alloc_time = time.perf_counter() - t0
    print(f"  Allocation: {alloc_time:.1f}s")

    m1 = get_mem()
    print(f"  Shmem delta: {m1['Shmem'] - m0['Shmem']:.1f}GB (expected 40GB)")
    print(f"  MemAvail: {m1['MemAvailable']:.1f}GB\n")

    # Warm up GPU
    _ = torch.empty(1024, device='cuda')
    torch.cuda.synchronize()

    # Test single chunk transfer
    print("Single chunk transfer (1GB)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu_chunk = chunks[0].to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    single_time = time.perf_counter() - t0
    print(f"  Time: {single_time:.3f}s ({1.0 / single_time:.1f} GB/s)")
    del gpu_chunk
    torch.cuda.empty_cache()

    # Test bulk transfer (simulating 35GB model)
    print("\nBulk transfer (35GB = 35 chunks)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu_tensors = []
    for i in range(35):
        gpu_t = chunks[i].to('cuda', non_blocking=True)
        gpu_tensors.append(gpu_t)
    torch.cuda.synchronize()
    bulk_time = time.perf_counter() - t0
    print(f"  Time: {bulk_time:.2f}s ({35.0 / bulk_time:.1f} GB/s)")

    # Verify data integrity
    print("\nVerifying data integrity...")
    all_correct = True
    for i, gpu_t in enumerate(gpu_tensors):
        expected = i % 256
        actual = gpu_t[0].item()
        if actual != expected:
            print(f"  ERROR: chunk {i} expected {expected}, got {actual}")
            all_correct = False
    if all_correct:
        print("  All data correct!")

    # Cleanup
    del gpu_tensors
    del chunks
    gc.collect()
    torch.cuda.empty_cache()

    m2 = get_mem()
    print(f"\nAfter cleanup: MemAvail={m2['MemAvailable']:.1f}GB, Shmem={m2['Shmem']:.1f}GB")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Allocation: {alloc_time:.1f}s for 40GB")
    print(f"  Memory overhead: {(m1['Shmem'] - m0['Shmem']) - 40:.1f}GB")
    print(f"  Transfer speed: {35.0 / bulk_time:.1f} GB/s (35GB bulk)")
    print("=" * 60)


if __name__ == "__main__":
    main()
