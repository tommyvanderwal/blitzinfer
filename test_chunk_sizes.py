#!/usr/bin/env python3
"""Test different pinned memory chunk sizes to find optimal allocation.

Tests various chunk sizes to determine where the huge page overhead begins.
"""

import torch
import gc
import time


def get_shmem_gb():
    """Get current shmem usage in GB."""
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('Shmem:'):
                return int(line.split()[1]) / 1024 / 1024  # kB -> GB
    return 0


def test_chunk_size(size_gb: float, label: str = None):
    """Test a single chunk allocation and measure overhead."""
    gc.collect()
    torch.cuda.empty_cache()

    shmem_before = get_shmem_gb()
    size_bytes = int(size_gb * 1024**3)

    try:
        t0 = time.perf_counter()
        chunk = torch.empty(size_bytes, dtype=torch.uint8, device='cpu', pin_memory=True)
        alloc_time = time.perf_counter() - t0

        shmem_after = get_shmem_gb()
        shmem_delta = shmem_after - shmem_before
        overhead_pct = ((shmem_delta - size_gb) / size_gb * 100) if size_gb > 0 else 0

        is_pinned = chunk.is_pinned()

        del chunk
        gc.collect()

        label_str = f" ({label})" if label else ""
        print(f"  {size_gb:6.2f}GB{label_str:15s}: shmem={shmem_delta:5.2f}GB, "
              f"overhead={overhead_pct:5.1f}%, time={alloc_time:.2f}s, pinned={is_pinned}")

        return {
            'size_gb': size_gb,
            'shmem_gb': shmem_delta,
            'overhead_pct': overhead_pct,
            'time_s': alloc_time,
            'success': True
        }
    except Exception as e:
        print(f"  {size_gb:6.2f}GB: FAILED - {e}")
        return {'size_gb': size_gb, 'success': False, 'error': str(e)}


def test_chunked_allocation(total_gb: float, chunk_size_gb: float):
    """Test allocating total_gb using chunks of chunk_size_gb."""
    gc.collect()
    torch.cuda.empty_cache()

    shmem_before = get_shmem_gb()
    num_chunks = int(total_gb / chunk_size_gb)
    chunk_bytes = int(chunk_size_gb * 1024**3)

    chunks = []
    t0 = time.perf_counter()

    try:
        for i in range(num_chunks):
            chunk = torch.empty(chunk_bytes, dtype=torch.uint8, device='cpu', pin_memory=True)
            chunks.append(chunk)

        alloc_time = time.perf_counter() - t0
        shmem_after = get_shmem_gb()
        shmem_delta = shmem_after - shmem_before
        actual_gb = num_chunks * chunk_size_gb
        overhead_pct = ((shmem_delta - actual_gb) / actual_gb * 100) if actual_gb > 0 else 0

        print(f"  {num_chunks:3d}x {chunk_size_gb:.2f}GB = {actual_gb:.1f}GB: "
              f"shmem={shmem_delta:5.2f}GB, overhead={overhead_pct:5.1f}%, time={alloc_time:.1f}s")

        del chunks
        gc.collect()

        return {
            'total_gb': actual_gb,
            'chunk_size_gb': chunk_size_gb,
            'num_chunks': num_chunks,
            'shmem_gb': shmem_delta,
            'overhead_pct': overhead_pct,
            'time_s': alloc_time,
            'success': True
        }
    except Exception as e:
        print(f"  {num_chunks}x {chunk_size_gb:.2f}GB: FAILED - {e}")
        del chunks
        gc.collect()
        return {'success': False, 'error': str(e)}


def main():
    print("=" * 70)
    print("PINNED MEMORY CHUNK SIZE INVESTIGATION")
    print("=" * 70)
    print()

    # Test 1: Single allocations of various sizes
    print("[Test 1] Single allocation overhead by size")
    print("-" * 50)

    single_sizes = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    results = []

    for size in single_sizes:
        result = test_chunk_size(size)
        results.append(result)
        time.sleep(0.5)  # Let memory settle

    # Test 2: Boundary testing around 1GB
    print()
    print("[Test 2] Boundary testing around 1GB")
    print("-" * 50)

    # Test exact boundaries
    boundary_sizes = [
        (1.0, "1GB exact"),
        (1.0 + 1/1024, "1GB + 1MB"),
        (1.0 + 10/1024, "1GB + 10MB"),
        (1.0 + 100/1024, "1GB + 100MB"),
        (1.0 + 500/1024, "1GB + 500MB"),
        (2.0 - 1/1024, "2GB - 1MB"),
        (2.0, "2GB exact"),
    ]

    for size, label in boundary_sizes:
        test_chunk_size(size, label)
        time.sleep(0.5)

    # Test 3: Chunked allocations to reach 40GB
    print()
    print("[Test 3] Chunked allocations to reach 40GB total")
    print("-" * 50)

    chunk_configs = [
        (40.0, 1.0),   # 40x 1GB
        (40.0, 2.0),   # 20x 2GB
        (40.0, 4.0),   # 10x 4GB
        (40.0, 5.0),   # 8x 5GB
        (40.0, 8.0),   # 5x 8GB
        (40.0, 10.0),  # 4x 10GB
    ]

    chunked_results = []
    for total, chunk_size in chunk_configs:
        result = test_chunked_allocation(total, chunk_size)
        chunked_results.append(result)
        time.sleep(1)  # Let memory settle between tests

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nSingle allocation overhead:")
    print(f"{'Size':>8s} | {'Shmem':>8s} | {'Overhead':>10s}")
    print("-" * 32)
    for r in results:
        if r['success']:
            print(f"{r['size_gb']:7.1f}GB | {r['shmem_gb']:7.2f}GB | {r['overhead_pct']:9.1f}%")

    print("\nChunked allocation (40GB total):")
    print(f"{'Config':>15s} | {'Shmem':>8s} | {'Overhead':>10s} | {'Time':>8s}")
    print("-" * 50)
    for r in chunked_results:
        if r['success']:
            config = f"{r['num_chunks']}x{r['chunk_size_gb']:.0f}GB"
            print(f"{config:>15s} | {r['shmem_gb']:7.2f}GB | {r['overhead_pct']:9.1f}% | {r['time_s']:7.1f}s")

    # Find optimal
    print()
    optimal = min([r for r in chunked_results if r['success']],
                  key=lambda x: (x['overhead_pct'], -x['chunk_size_gb']))
    print(f"OPTIMAL: {optimal['num_chunks']}x {optimal['chunk_size_gb']:.0f}GB chunks "
          f"({optimal['overhead_pct']:.1f}% overhead)")

    print("=" * 70)


if __name__ == "__main__":
    main()
