#!/usr/bin/env python3
"""Test 16GB chunks (power of 2) for pinned memory."""

import torch
import gc
import time


def get_shmem_gb():
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('Shmem:'):
                return int(line.split()[1]) / 1024 / 1024
    return 0


def main():
    print("=" * 60)
    print("16GB CHUNK TEST (Power of 2)")
    print("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)

    initial = get_shmem_gb()
    print(f"Initial shmem: {initial:.2f}GB\n")

    # Test 1: Single 16GB allocation
    print("[Test 1] Single 16GB pinned allocation")
    print("-" * 40)

    shmem_before = get_shmem_gb()
    t0 = time.perf_counter()
    chunk = torch.empty(16 * 1024**3, dtype=torch.uint8, device='cpu', pin_memory=True)
    alloc_time = time.perf_counter() - t0

    time.sleep(0.5)
    shmem_after = get_shmem_gb()
    delta = shmem_after - shmem_before
    overhead = ((delta - 16) / 16 * 100) if delta > 0 else 0

    print(f"  Requested: 16GB")
    print(f"  Shmem used: {delta:.2f}GB")
    print(f"  Overhead: {overhead:.1f}%")
    print(f"  Alloc time: {alloc_time:.1f}s")
    print(f"  Is pinned: {chunk.is_pinned()}")

    del chunk
    gc.collect()
    time.sleep(1)

    # Test 2: 4x 16GB = 64GB (for 65GB model with slight overshoot)
    print(f"\n[Test 2] 4x 16GB = 64GB chunked allocation")
    print("-" * 40)

    gc.collect()
    shmem_before = get_shmem_gb()
    chunks = []

    t0 = time.perf_counter()
    for i in range(4):
        chunk = torch.empty(16 * 1024**3, dtype=torch.uint8, device='cpu', pin_memory=True)
        chunks.append(chunk)
        curr_shmem = get_shmem_gb() - shmem_before
        print(f"  Chunk {i+1}/4: shmem = {curr_shmem:.1f}GB (expected {(i+1)*16}GB)")

    alloc_time = time.perf_counter() - t0
    final_shmem = get_shmem_gb() - shmem_before
    overhead = ((final_shmem - 64) / 64 * 100)

    print(f"\n  Total: {final_shmem:.1f}GB shmem for 64GB")
    print(f"  Overhead: {overhead:.1f}%")
    print(f"  Time: {alloc_time:.1f}s")

    # Test GPU transfer speed
    print(f"\n[Test 3] GPU transfer speed")
    print("-" * 40)

    _ = torch.empty(1024, device='cuda')  # Warm up
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu_chunk = chunks[0].to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    transfer_time = time.perf_counter() - t0

    print(f"  16GB transfer: {transfer_time:.2f}s ({16/transfer_time:.1f} GB/s)")

    del gpu_chunk
    torch.cuda.empty_cache()

    # Cleanup
    del chunks
    gc.collect()
    time.sleep(1)
    final = get_shmem_gb()
    print(f"\nFinal shmem: {final:.2f}GB (should be ~{initial:.2f}GB)")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if abs(overhead) < 5:
        print("✓ 16GB chunks have 0% overhead - RECOMMENDED")
        print("  Use 5x 16GB = 80GB for models up to 65GB")
    else:
        print(f"✗ 16GB chunks have {overhead:.1f}% overhead")
    print("=" * 60)


if __name__ == "__main__":
    main()
