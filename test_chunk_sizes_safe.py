#!/usr/bin/env python3
"""Test different pinned memory chunk sizes - safe version with smaller allocations."""

import torch
import gc
import time
import sys


def get_shmem_gb():
    """Get current shmem usage in GB."""
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('Shmem:'):
                return int(line.split()[1]) / 1024 / 1024  # kB -> GB
    return 0


def test_single_alloc(size_gb: float, label: str = ""):
    """Test a single pinned allocation."""
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)

    shmem_before = get_shmem_gb()
    size_bytes = int(size_gb * 1024**3)

    try:
        t0 = time.perf_counter()
        chunk = torch.empty(size_bytes, dtype=torch.uint8, device='cpu', pin_memory=True)
        alloc_time = time.perf_counter() - t0

        shmem_after = get_shmem_gb()
        shmem_delta = shmem_after - shmem_before
        overhead_pct = ((shmem_delta - size_gb) / size_gb * 100) if size_gb > 0 else 0

        label_str = f" {label}" if label else ""
        print(f"  {size_gb:6.3f}GB{label_str:20s}: shmem={shmem_delta:6.2f}GB, overhead={overhead_pct:6.1f}%", flush=True)

        # Cleanup immediately
        del chunk
        gc.collect()
        time.sleep(0.3)

        return overhead_pct
    except Exception as e:
        print(f"  {size_gb:6.3f}GB: FAILED - {e}", flush=True)
        return None


def test_chunked(total_gb: float, chunk_gb: float):
    """Test chunked allocation."""
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)

    shmem_before = get_shmem_gb()
    num_chunks = int(total_gb / chunk_gb)
    chunk_bytes = int(chunk_gb * 1024**3)

    chunks = []
    try:
        t0 = time.perf_counter()
        for _ in range(num_chunks):
            chunk = torch.empty(chunk_bytes, dtype=torch.uint8, device='cpu', pin_memory=True)
            chunks.append(chunk)
        alloc_time = time.perf_counter() - t0

        shmem_after = get_shmem_gb()
        shmem_delta = shmem_after - shmem_before
        actual_gb = num_chunks * chunk_gb
        overhead_pct = ((shmem_delta - actual_gb) / actual_gb * 100) if actual_gb > 0 else 0

        print(f"  {num_chunks:2d}x {chunk_gb:.1f}GB = {actual_gb:4.0f}GB: "
              f"shmem={shmem_delta:6.2f}GB, overhead={overhead_pct:6.1f}%, time={alloc_time:5.1f}s", flush=True)

        # Cleanup
        del chunks
        gc.collect()
        time.sleep(0.5)

        return overhead_pct
    except Exception as e:
        print(f"  {num_chunks}x {chunk_gb:.1f}GB: FAILED - {e}", flush=True)
        del chunks
        gc.collect()
        return None


def main():
    print("=" * 70, flush=True)
    print("PINNED MEMORY CHUNK SIZE INVESTIGATION", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    # Test 1: Single allocations
    print("[Test 1] Single allocation overhead by size", flush=True)
    print("-" * 50, flush=True)

    results = {}
    for size in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]:
        overhead = test_single_alloc(size)
        if overhead is not None:
            results[size] = overhead

    # Test 2: Boundary around 1GB
    print(flush=True)
    print("[Test 2] Boundary testing around 1GB", flush=True)
    print("-" * 50, flush=True)

    boundaries = [
        (1.0, "exact 1GB"),
        (1.001, "1GB + 1MB"),
        (1.01, "1GB + 10MB"),
        (1.1, "1GB + 100MB"),
        (1.5, "1.5GB"),
        (1.99, "~2GB"),
        (2.0, "exact 2GB"),
        (2.001, "2GB + 1MB"),
    ]

    for size, label in boundaries:
        test_single_alloc(size, label)

    # Test 3: Boundary around 2GB (potential huge page size)
    print(flush=True)
    print("[Test 3] Boundary testing around 2GB", flush=True)
    print("-" * 50, flush=True)

    for size, label in [(1.9, "1.9GB"), (1.99, "1.99GB"), (2.0, "2.0GB"), (2.01, "2.01GB"), (2.1, "2.1GB")]:
        test_single_alloc(size, label)

    # Test 4: Chunked allocations to 20GB (safer than 40GB)
    print(flush=True)
    print("[Test 4] Chunked allocations to 20GB", flush=True)
    print("-" * 50, flush=True)

    chunked_results = {}
    for chunk_size in [1.0, 2.0, 4.0, 5.0, 10.0]:
        overhead = test_chunked(20.0, chunk_size)
        if overhead is not None:
            chunked_results[chunk_size] = overhead

    # Summary
    print(flush=True)
    print("=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print("\nSingle allocation overhead:", flush=True)
    for size, overhead in sorted(results.items()):
        status = "OK" if overhead < 5 else "HIGH OVERHEAD"
        print(f"  {size:5.1f}GB: {overhead:5.1f}% {status}", flush=True)

    print("\n20GB chunked allocation:", flush=True)
    for chunk, overhead in sorted(chunked_results.items()):
        status = "OK" if overhead < 5 else "HIGH OVERHEAD"
        print(f"  {int(20/chunk):2d}x {chunk:.0f}GB: {overhead:5.1f}% {status}", flush=True)

    # Find threshold
    print("\nOVERHEAD THRESHOLD:", flush=True)
    threshold = None
    for size in sorted(results.keys()):
        if results[size] > 10:  # More than 10% overhead
            threshold = size
            break
    if threshold:
        print(f"  Overhead starts at {threshold}GB allocations", flush=True)
    else:
        print("  No significant overhead detected up to 10GB", flush=True)

    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
