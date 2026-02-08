#!/usr/bin/env python3
"""Final chunk size test with proper measurement."""

import torch
import gc
import time


def get_shmem_kb():
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('Shmem:'):
                return int(line.split()[1])
    return 0


def measure_single(size_gb: float):
    """Allocate, measure while held, then release."""
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)

    shmem_before = get_shmem_kb()

    chunk = torch.empty(int(size_gb * 1024**3), dtype=torch.uint8, device='cpu', pin_memory=True)

    # Measure while still holding
    time.sleep(0.3)
    shmem_during = get_shmem_kb()
    shmem_delta = (shmem_during - shmem_before) / 1024 / 1024  # GB

    # Keep reference to prevent GC
    is_pinned = chunk.is_pinned()

    # Now release
    del chunk
    gc.collect()
    time.sleep(0.3)

    shmem_after = get_shmem_kb()
    released = shmem_during - shmem_after

    return {
        'requested': size_gb,
        'shmem_used': shmem_delta,
        'overhead_pct': ((shmem_delta - size_gb) / size_gb * 100) if size_gb > 0 else 0,
        'released_kb': released,
        'pinned': is_pinned
    }


def measure_chunked(total_gb: float, chunk_gb: float):
    """Allocate chunks, measure, release."""
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)

    num = int(total_gb / chunk_gb)
    shmem_before = get_shmem_kb()

    chunks = []
    for _ in range(num):
        chunks.append(torch.empty(int(chunk_gb * 1024**3), dtype=torch.uint8, device='cpu', pin_memory=True))

    time.sleep(0.3)
    shmem_during = get_shmem_kb()
    shmem_delta = (shmem_during - shmem_before) / 1024 / 1024

    actual = num * chunk_gb
    del chunks
    gc.collect()

    return {
        'config': f"{num}x{chunk_gb:.0f}GB",
        'expected': actual,
        'shmem_used': shmem_delta,
        'overhead_pct': ((shmem_delta - actual) / actual * 100) if actual > 0 else 0
    }


def main():
    print("=" * 70)
    print("FINAL CHUNK SIZE ANALYSIS")
    print("=" * 70)

    # Clear state
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    print(f"Initial shmem: {get_shmem_kb()/1024/1024:.2f}GB\n")

    # Test single allocations
    print("SINGLE ALLOCATIONS")
    print("-" * 50)
    print(f"{'Size':>8} | {'Shmem':>8} | {'Overhead':>10} | Notes")
    print("-" * 50)

    test_sizes = [
        (0.5, ""),
        (1.0, "power of 2"),
        (1.5, ""),
        (2.0, "power of 2"),
        (2.5, ""),
        (3.0, ""),
        (4.0, "power of 2"),
        (5.0, ""),
        (6.0, ""),
        (8.0, "power of 2"),
    ]

    results = []
    for size, note in test_sizes:
        try:
            r = measure_single(size)
            results.append(r)
            status = "✓" if abs(r['overhead_pct']) < 5 else f"→{r['shmem_used']:.1f}GB"
            print(f"{size:7.1f}GB | {r['shmem_used']:7.2f}GB | {r['overhead_pct']:9.1f}% | {note} {status}")
        except Exception as e:
            print(f"{size:7.1f}GB | ERROR: {str(e)[:40]}")
        time.sleep(0.5)

    # Test chunked allocations to 40GB
    print()
    print("CHUNKED ALLOCATIONS (40GB total)")
    print("-" * 50)
    print(f"{'Config':>12} | {'Shmem':>8} | {'Overhead':>10}")
    print("-" * 50)

    chunk_sizes = [1.0, 2.0, 4.0, 5.0, 8.0, 10.0]
    chunked_results = []

    for cs in chunk_sizes:
        try:
            r = measure_chunked(40.0, cs)
            chunked_results.append(r)
            status = "✓" if abs(r['overhead_pct']) < 5 else "⚠"
            print(f"{r['config']:>12} | {r['shmem_used']:7.1f}GB | {r['overhead_pct']:9.1f}% {status}")
        except Exception as e:
            print(f"{int(40/cs)}x{cs:.0f}GB | ERROR: {str(e)[:40]}")
        time.sleep(0.5)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find sizes with no overhead
    no_overhead = [r for r in results if abs(r['overhead_pct']) < 5]
    print(f"\nSizes with ~0% overhead: {[r['requested'] for r in no_overhead]}GB")

    # Find sizes with overhead
    with_overhead = [(r['requested'], r['shmem_used']) for r in results if r['overhead_pct'] > 10]
    if with_overhead:
        print(f"Sizes with overhead:")
        for req, used in with_overhead:
            print(f"  {req}GB → {used:.1f}GB shmem")

    # Chunked recommendation
    good_chunks = [r for r in chunked_results if abs(r['overhead_pct']) < 5]
    if good_chunks:
        # Prefer larger chunks for faster allocation
        best = max(good_chunks, key=lambda x: float(x['config'].split('x')[1].replace('GB', '')))
        print(f"\nRECOMMENDED chunk size: {best['config']} (0% overhead, fewer allocations)")

    print("=" * 70)


if __name__ == "__main__":
    main()
