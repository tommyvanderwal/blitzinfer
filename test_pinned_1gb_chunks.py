#!/usr/bin/env python3
"""Test if 70x 1GB pinned allocations avoid the overhead.

Hypothesis: CUDA uses 1GB huge pages for large allocations, causing overhead.
If we allocate in 1GB chunks, we should see NO overhead.
"""

import torch
import gc
import time

def get_mem():
    """Get memory stats from /proc/meminfo."""
    with open('/proc/meminfo', 'r') as f:
        mem = {}
        for line in f:
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                val = int(parts[1].strip().split()[0])  # kB
                mem[key] = val
    return mem

def show_mem(label):
    m = get_mem()
    print(f"[{label:25s}] Shmem: {m['Shmem']/1024/1024:.2f}GB, "
          f"MemAvail: {m['MemAvailable']/1024/1024:.2f}GB", flush=True)
    return m

def main():
    print("=" * 70)
    print("TESTING 70x 1GB PINNED CHUNKS (HUGE PAGE HYPOTHESIS)")
    print("=" * 70)
    print()
    print("Hypothesis: CUDA uses 1GB huge pages for large allocations.")
    print("If we allocate exactly 1GB chunks, we should see NO overhead.")
    print()

    gc.collect()
    torch.cuda.empty_cache()
    m0 = show_mem("START")

    # Test 1: Single 70GB allocation (for comparison - will likely fail)
    print("\n--- Test 1: Single 70GB pinned allocation ---")
    print("(Skipping - known to cause 60% overhead and potential OOM)")

    # Test 2: 70x 1GB allocations
    print("\n--- Test 2: 70x 1GB pinned allocations ---")
    chunks = []
    t0 = time.perf_counter()

    for i in range(70):
        chunk = torch.empty(1 * 1024**3, dtype=torch.uint8, device="cpu", pin_memory=True)
        chunks.append(chunk)

        if (i + 1) % 10 == 0:
            m = get_mem()
            delta_shmem = (m['Shmem'] - m0['Shmem']) / 1024 / 1024
            expected = i + 1
            overhead = (delta_shmem - expected) / expected * 100 if expected > 0 else 0
            print(f"  Chunk {i+1:2d}/70: Shmem delta = {delta_shmem:.1f}GB "
                  f"(expected {expected}GB, overhead {overhead:.1f}%)")

    alloc_time = time.perf_counter() - t0
    m1 = show_mem("AFTER 70x 1GB PINNED")

    total_shmem = (m1['Shmem'] - m0['Shmem']) / 1024 / 1024
    expected_shmem = 70.0
    overhead_pct = (total_shmem - expected_shmem) / expected_shmem * 100

    print(f"\nAllocation time: {alloc_time:.1f}s")
    print(f"Total Shmem used: {total_shmem:.1f}GB (expected 70GB)")
    print(f"Overhead: {overhead_pct:.1f}%")

    # Verify all chunks are pinned
    all_pinned = all(c.is_pinned() for c in chunks)
    print(f"All chunks pinned: {all_pinned}")

    # Test 3: GPU transfer speed from these chunks
    print("\n--- Test 3: GPU transfer speed from 1GB chunks ---")

    # Warm up GPU
    _ = torch.empty(1024, device='cuda')
    torch.cuda.synchronize()

    # Transfer one chunk to GPU and measure speed
    chunk = chunks[0]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu_tensor = chunk.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    transfer_time = time.perf_counter() - t0

    bandwidth = 1.0 / transfer_time  # GB/s
    print(f"  1GB chunk transfer: {transfer_time:.3f}s ({bandwidth:.1f} GB/s)")

    del gpu_tensor
    torch.cuda.empty_cache()

    # Test 4: Transfer multiple chunks (simulate model load)
    print("\n--- Test 4: Transfer 35GB (35 chunks) to GPU ---")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    gpu_tensors = []
    for i in range(35):
        gpu_t = chunks[i].to('cuda', non_blocking=True)
        gpu_tensors.append(gpu_t)

    torch.cuda.synchronize()
    transfer_time = time.perf_counter() - t0

    bandwidth = 35.0 / transfer_time
    print(f"  35GB transfer: {transfer_time:.2f}s ({bandwidth:.1f} GB/s)")

    del gpu_tensors
    torch.cuda.empty_cache()

    # Cleanup
    print("\n--- Cleanup ---")
    del chunks
    gc.collect()
    torch.cuda.empty_cache()
    show_mem("AFTER CLEANUP")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if overhead_pct < 5:
        print(f"SUCCESS: 70x 1GB pinned allocations have {overhead_pct:.1f}% overhead")
        print("This confirms the huge page hypothesis!")
        print("\nRecommendation: Use 1GB chunks in PinnedMemoryArena")
    else:
        print(f"UNEXPECTED: 70x 1GB pinned allocations have {overhead_pct:.1f}% overhead")
        print("The huge page hypothesis may be incorrect.")
    print("=" * 70)


if __name__ == "__main__":
    main()
