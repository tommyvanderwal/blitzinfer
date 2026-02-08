#!/usr/bin/env python3
"""
Verify pinned memory improvement is real, not warmup effect.

Run multiple iterations of both approaches to get fair comparison.
"""

import gc
import time
from pathlib import Path
from typing import List, Tuple

import torch
from safetensors.torch import safe_open

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def find_safetensors(model_name: str) -> List[str]:
    """Find safetensors files for a model in HF cache."""
    model_dir = model_name.replace("/", "--")

    for cache_dir in HF_CACHE.glob(f"models--{model_dir}*"):
        snapshots = cache_dir / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                st_files = list(snapshot.glob("model-*.safetensors"))
                if st_files:
                    return sorted([str(f) for f in st_files])

    raise FileNotFoundError(f"No safetensors found for {model_name}")


def run_normal_copy(st_files: List[str], gpu_buffer: torch.Tensor) -> float:
    """Single run of normal CPU→GPU copy."""
    t0 = time.perf_counter()

    for sf in st_files:
        with safe_open(sf, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)  # Non-pinned CPU tensor
                numel = tensor.numel()
                gpu_buffer[:numel].copy_(tensor.view(-1)[:numel], non_blocking=False)

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_pinned_copy(st_files: List[str], gpu_buffer: torch.Tensor, pinned_buffer: torch.Tensor) -> float:
    """Single run of pinned memory CPU→GPU copy."""
    t0 = time.perf_counter()

    for sf in st_files:
        with safe_open(sf, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)  # Non-pinned CPU tensor
                numel = tensor.numel()

                # Copy to pinned first, then to GPU
                pinned_buffer[:numel].copy_(tensor.view(-1)[:numel])
                gpu_buffer[:numel].copy_(pinned_buffer[:numel], non_blocking=True)

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    print("="*70)
    print("VERIFY PINNED MEMORY (Multiple Iterations)")
    print("="*70)

    # Initialize CUDA
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    st_files = find_safetensors(model_name)
    total_bytes = sum(Path(f).stat().st_size for f in st_files)
    gb = total_bytes / (1024**3)

    print(f"\nModel: {model_name}")
    print(f"Total: {gb:.2f} GB")
    print(f"Files: {len(st_files)}")

    # Allocate buffers
    max_elements = 200_000_000
    gpu_buffer = torch.empty(max_elements, dtype=torch.float16, device='cuda')
    pinned_buffer = torch.empty(max_elements, dtype=torch.float16, pin_memory=True)

    n_runs = 5

    print(f"\nRunning {n_runs} iterations of each method...")
    print("-"*70)

    # Warmup both methods first
    print("Warmup run (normal)...")
    run_normal_copy(st_files, gpu_buffer)
    print("Warmup run (pinned)...")
    run_pinned_copy(st_files, gpu_buffer, pinned_buffer)

    # Alternating runs to eliminate ordering bias
    normal_times = []
    pinned_times = []

    for i in range(n_runs):
        # Run normal
        gc.collect()
        torch.cuda.synchronize()
        elapsed = run_normal_copy(st_files, gpu_buffer)
        normal_times.append(elapsed)
        bw = gb / elapsed
        print(f"  Normal  run {i+1}: {elapsed*1000:.0f}ms ({bw:.1f} GB/s)")

        # Run pinned
        gc.collect()
        torch.cuda.synchronize()
        elapsed = run_pinned_copy(st_files, gpu_buffer, pinned_buffer)
        pinned_times.append(elapsed)
        bw = gb / elapsed
        print(f"  Pinned  run {i+1}: {elapsed*1000:.0f}ms ({bw:.1f} GB/s)")

        print()

    # Statistics
    print("="*70)
    print("RESULTS")
    print("="*70)

    def stats(times):
        avg = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)
        return avg, min_t, max_t

    n_avg, n_min, n_max = stats(normal_times)
    p_avg, p_min, p_max = stats(pinned_times)

    print(f"\nNormal copy:")
    print(f"  Avg: {n_avg*1000:.0f}ms ({gb/n_avg:.1f} GB/s)")
    print(f"  Min: {n_min*1000:.0f}ms ({gb/n_min:.1f} GB/s)")
    print(f"  Max: {n_max*1000:.0f}ms ({gb/n_max:.1f} GB/s)")

    print(f"\nPinned copy:")
    print(f"  Avg: {p_avg*1000:.0f}ms ({gb/p_avg:.1f} GB/s)")
    print(f"  Min: {p_min*1000:.0f}ms ({gb/p_min:.1f} GB/s)")
    print(f"  Max: {p_max*1000:.0f}ms ({gb/p_max:.1f} GB/s)")

    speedup = n_avg / p_avg
    savings = (n_avg - p_avg) * 1000

    print(f"\n{'='*70}")
    print(f"CONCLUSION")
    print(f"{'='*70}")
    print(f"  Speedup:     {speedup:.2f}x")
    print(f"  Time saved:  {savings:.0f}ms per load")

    if speedup > 1.5:
        print(f"\n  [CONFIRMED] Pinned memory is {speedup:.1f}x faster!")
    else:
        print(f"\n  [MARGINAL] Speedup is only {speedup:.2f}x")

    # Cleanup
    del gpu_buffer, pinned_buffer


if __name__ == '__main__':
    main()
