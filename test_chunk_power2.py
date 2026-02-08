#!/usr/bin/env python3
"""Test if power-of-2 chunk sizes avoid overhead."""

import torch
import gc
import time
import subprocess


def get_shmem_kb():
    """Get shmem in KB directly."""
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('Shmem:'):
                return int(line.split()[1])
    return 0


def clean_state():
    """Ensure clean state."""
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)


def test_allocation(size_gb: float, label: str = ""):
    """Test single allocation with fresh state."""
    clean_state()

    shmem_before = get_shmem_kb()
    size_bytes = int(size_gb * 1024**3)

    try:
        chunk = torch.empty(size_bytes, dtype=torch.uint8, device='cpu', pin_memory=True)

        # Force sync
        time.sleep(0.2)

        shmem_after = get_shmem_kb()
        shmem_delta_kb = shmem_after - shmem_before
        shmem_delta_gb = shmem_delta_kb / 1024 / 1024

        overhead_pct = ((shmem_delta_gb - size_gb) / size_gb * 100) if size_gb > 0 else 0

        # Keep chunk alive for measurement
        result = {
            'size_gb': size_gb,
            'shmem_gb': shmem_delta_gb,
            'overhead_pct': overhead_pct,
            'label': label
        }

        # Now cleanup
        del chunk
        gc.collect()
        time.sleep(0.5)

        return result

    except Exception as e:
        gc.collect()
        return {'size_gb': size_gb, 'error': str(e), 'label': label}


def main():
    print("=" * 70)
    print("POWER-OF-2 CHUNK SIZE TEST")
    print("=" * 70)
    print()

    # First, verify clean state
    clean_state()
    initial_shmem = get_shmem_kb() / 1024 / 1024
    print(f"Initial shmem: {initial_shmem:.2f}GB")
    print()

    # Test power-of-2 sizes
    print("[Test 1] Power-of-2 sizes (expect 0% overhead)")
    print("-" * 50)

    power2_sizes = [0.5, 1.0, 2.0, 4.0]  # Keep smaller to avoid OOM
    for size in power2_sizes:
        result = test_allocation(size, f"{size}GB (2^{int(size).bit_length()-1 if size >= 1 else 'x'})")
        if 'error' not in result:
            status = "OK" if abs(result['overhead_pct']) < 5 else "OVERHEAD"
            print(f"  {size:4.1f}GB: shmem={result['shmem_gb']:5.2f}GB, overhead={result['overhead_pct']:6.1f}% [{status}]")
        else:
            print(f"  {size:4.1f}GB: ERROR - {result['error']}")

    # Test non-power-of-2 sizes
    print()
    print("[Test 2] Non-power-of-2 sizes (may have overhead)")
    print("-" * 50)

    nonpower2_sizes = [1.5, 3.0, 5.0, 6.0]
    for size in nonpower2_sizes:
        result = test_allocation(size, f"{size}GB (not 2^n)")
        if 'error' not in result:
            status = "OK" if abs(result['overhead_pct']) < 5 else "OVERHEAD"
            print(f"  {size:4.1f}GB: shmem={result['shmem_gb']:5.2f}GB, overhead={result['overhead_pct']:6.1f}% [{status}]")
        else:
            print(f"  {size:4.1f}GB: ERROR - {result['error']}")

    # Test chunked allocations with different sizes to total 20GB
    print()
    print("[Test 3] Chunked to 20GB total")
    print("-" * 50)

    configs = [
        (20, 1.0, "20x 1GB (power of 2)"),
        (10, 2.0, "10x 2GB (power of 2)"),
        (5, 4.0, "5x 4GB (power of 2)"),
        (4, 5.0, "4x 5GB (NOT power of 2)"),
        (7, 3.0, "7x 3GB = 21GB (NOT power of 2)"),  # rounds to 21GB
    ]

    for num_chunks, chunk_gb, label in configs:
        clean_state()
        shmem_before = get_shmem_kb()
        chunk_bytes = int(chunk_gb * 1024**3)

        try:
            chunks = []
            for _ in range(num_chunks):
                chunks.append(torch.empty(chunk_bytes, dtype=torch.uint8, device='cpu', pin_memory=True))

            time.sleep(0.3)
            shmem_after = get_shmem_kb()
            shmem_delta_gb = (shmem_after - shmem_before) / 1024 / 1024
            expected_gb = num_chunks * chunk_gb
            overhead_pct = ((shmem_delta_gb - expected_gb) / expected_gb * 100)

            status = "OK" if abs(overhead_pct) < 5 else "OVERHEAD"
            print(f"  {label:30s}: shmem={shmem_delta_gb:5.1f}GB, expected={expected_gb:.0f}GB, "
                  f"overhead={overhead_pct:5.1f}% [{status}]")

            del chunks
            gc.collect()
        except Exception as e:
            print(f"  {label:30s}: ERROR - {e}")
            gc.collect()

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
If power-of-2 sizes (1GB, 2GB, 4GB) have 0% overhead but non-power-of-2
sizes (1.5GB, 3GB, 5GB) have overhead, then CUDA is rounding allocations
up to the next power of 2 for huge page alignment.

RECOMMENDATION: Use power-of-2 chunk sizes (1GB, 2GB, or 4GB)
    """)


if __name__ == "__main__":
    main()
