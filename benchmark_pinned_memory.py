#!/usr/bin/env python3
"""
Test pinned memory for faster CPU→GPU transfers.

On unified memory systems like AMD 780M, pinned memory may or may not help.
Let's measure.
"""

import gc
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
from safetensors.torch import safe_open

# Use the cached model files
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def find_safetensors(model_name: str) -> List[str]:
    """Find safetensors files for a model in HF cache."""
    model_dir = model_name.replace("/", "--")

    for cache_dir in HF_CACHE.glob(f"models--{model_dir}*"):
        snapshots = cache_dir / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                # Only get sharded files, not consolidated
                st_files = list(snapshot.glob("model-*.safetensors"))
                if st_files:
                    return sorted([str(f) for f in st_files])

    raise FileNotFoundError(f"No safetensors found for {model_name}")


def get_mem():
    """Get GPU memory (used, free) in GB."""
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**3)
    return used, free / (1024**3)


def benchmark_normal_copy(st_files: List[str]) -> Tuple[float, float]:
    """Benchmark normal CPU→GPU copy."""
    total_bytes = sum(Path(f).stat().st_size for f in st_files)

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    used, free = get_mem()
    print(f"  Before: {used:.1f}GB used, {free:.1f}GB free")

    # Allocate GPU buffer
    max_param = torch.empty(200_000_000, dtype=torch.float16, device='cuda')

    t0 = time.perf_counter()
    count = 0

    for sf in st_files:
        with safe_open(sf, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)  # CPU tensor (non-pinned)
                numel = tensor.numel()
                max_param[:numel].copy_(tensor.view(-1)[:numel], non_blocking=False)
                count += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del max_param
    gc.collect()
    torch.cuda.empty_cache()

    return elapsed, total_bytes


def benchmark_pinned_copy(st_files: List[str]) -> Tuple[float, float]:
    """Benchmark pinned memory CPU→GPU copy."""
    total_bytes = sum(Path(f).stat().st_size for f in st_files)

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    used, free = get_mem()
    print(f"  Before: {used:.1f}GB used, {free:.1f}GB free")

    # Allocate GPU buffer
    max_param = torch.empty(200_000_000, dtype=torch.float16, device='cuda')

    # Pre-allocate pinned CPU buffer
    pinned_buffer = torch.empty(200_000_000, dtype=torch.float16, pin_memory=True)

    t0 = time.perf_counter()
    count = 0

    for sf in st_files:
        with safe_open(sf, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)  # CPU tensor (non-pinned)
                numel = tensor.numel()

                # Copy to pinned buffer first, then to GPU
                pinned_buffer[:numel].copy_(tensor.view(-1)[:numel])
                max_param[:numel].copy_(pinned_buffer[:numel], non_blocking=True)
                count += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del max_param, pinned_buffer
    gc.collect()
    torch.cuda.empty_cache()

    return elapsed, total_bytes


def benchmark_async_copy(st_files: List[str]) -> Tuple[float, float]:
    """Benchmark async CPU→GPU copy."""
    total_bytes = sum(Path(f).stat().st_size for f in st_files)

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Allocate GPU buffer
    max_param = torch.empty(200_000_000, dtype=torch.float16, device='cuda')

    t0 = time.perf_counter()
    count = 0

    for sf in st_files:
        with safe_open(sf, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)  # CPU tensor
                numel = tensor.numel()
                max_param[:numel].copy_(tensor.view(-1)[:numel], non_blocking=True)
                count += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del max_param
    gc.collect()
    torch.cuda.empty_cache()

    return elapsed, total_bytes


def benchmark_bulk_then_copy(st_files: List[str]) -> Tuple[float, float]:
    """Load all weights to a dict first, then bulk copy to GPU."""
    total_bytes = sum(Path(f).stat().st_size for f in st_files)

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()

    # Phase 1: Load all to CPU dict
    t_load = time.perf_counter()
    all_tensors = []
    for sf in st_files:
        with safe_open(sf, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                all_tensors.append(tensor)
    load_time = (time.perf_counter() - t_load) * 1000

    # Phase 2: Bulk copy to GPU
    t_copy = time.perf_counter()
    gpu_tensors = []
    for t in all_tensors:
        gpu_tensors.append(t.to('cuda', non_blocking=True))

    torch.cuda.synchronize()
    copy_time = (time.perf_counter() - t_copy) * 1000

    elapsed = time.perf_counter() - t0

    print(f"    Load: {load_time:.0f}ms, Copy: {copy_time:.0f}ms")

    del all_tensors, gpu_tensors
    gc.collect()
    torch.cuda.empty_cache()

    return elapsed, total_bytes


def main():
    print("="*70)
    print("PINNED MEMORY BENCHMARK")
    print("="*70)

    # Initialize CUDA
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    try:
        st_files = find_safetensors(model_name)
        total_size = sum(Path(f).stat().st_size for f in st_files)
        print(f"\nModel: {model_name}")
        print(f"Files: {len(st_files)}")
        print(f"Total: {total_size/(1024**3):.2f} GB")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    results = []

    print("\n" + "-"*70)
    print("1. Normal sync copy (baseline)")
    print("-"*70)
    elapsed, total_bytes = benchmark_normal_copy(st_files)
    bw = (total_bytes / (1024**3)) / elapsed
    print(f"  Time: {elapsed*1000:.0f}ms, Bandwidth: {bw:.1f} GB/s")
    results.append(("Normal sync", elapsed*1000, bw))

    print("\n" + "-"*70)
    print("2. Async copy")
    print("-"*70)
    elapsed, total_bytes = benchmark_async_copy(st_files)
    bw = (total_bytes / (1024**3)) / elapsed
    print(f"  Time: {elapsed*1000:.0f}ms, Bandwidth: {bw:.1f} GB/s")
    results.append(("Async", elapsed*1000, bw))

    print("\n" + "-"*70)
    print("3. Pinned memory copy")
    print("-"*70)
    elapsed, total_bytes = benchmark_pinned_copy(st_files)
    bw = (total_bytes / (1024**3)) / elapsed
    print(f"  Time: {elapsed*1000:.0f}ms, Bandwidth: {bw:.1f} GB/s")
    results.append(("Pinned", elapsed*1000, bw))

    print("\n" + "-"*70)
    print("4. Bulk load then copy")
    print("-"*70)
    elapsed, total_bytes = benchmark_bulk_then_copy(st_files)
    bw = (total_bytes / (1024**3)) / elapsed
    print(f"  Time: {elapsed*1000:.0f}ms, Bandwidth: {bw:.1f} GB/s")
    results.append(("Bulk+copy", elapsed*1000, bw))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    baseline = results[0][1]
    for name, time_ms, bw in results:
        diff = (baseline - time_ms) / baseline * 100
        if diff > 0:
            print(f"  {name}: {time_ms:.0f}ms ({bw:.1f} GB/s) - {diff:.1f}% faster")
        else:
            print(f"  {name}: {time_ms:.0f}ms ({bw:.1f} GB/s)")

    print("="*70)


if __name__ == '__main__':
    main()
