#!/usr/bin/env python3
"""
Test different concurrency levels for pipelined weight loading.

Goal: Find optimal number of buffers/streams for 8-core Ryzen 9 8945HS
"""

import gc
import time
from pathlib import Path
from typing import List

import torch
from safetensors.torch import safe_open

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def find_safetensors(model_name: str) -> List[str]:
    """Find safetensors files for a model."""
    model_dir = model_name.replace("/", "--")
    for cache_dir in HF_CACHE.glob(f"models--{model_dir}*"):
        snapshots = cache_dir / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                st_files = list(snapshot.glob("model*.safetensors"))
                if st_files:
                    return sorted([str(f) for f in st_files])
    raise FileNotFoundError(f"No safetensors found for {model_name}")


def load_pipelined(st_files: List[str], gpu_buffer: torch.Tensor, n_buffers: int) -> float:
    """Pipelined: read into pinned buffer while copying previous to GPU."""
    t0 = time.perf_counter()

    # Pre-allocate pinned buffers
    max_size = 600_000_000
    pinned_buffers = [torch.empty(max_size, dtype=torch.float16, pin_memory=True) for _ in range(n_buffers)]
    streams = [torch.cuda.Stream() for _ in range(n_buffers)]

    buffer_idx = 0
    pending_copies = []

    for sf in st_files:
        with safe_open(sf, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                numel = tensor.numel()

                # Wait for previous copy on this buffer to complete
                if len(pending_copies) >= n_buffers:
                    old_stream, old_event = pending_copies.pop(0)
                    old_event.wait()

                # Copy to pinned buffer (handle dtype)
                pinned_buf = pinned_buffers[buffer_idx]
                flat = tensor.view(-1)
                if tensor.dtype == torch.float16:
                    pinned_buf[:numel].copy_(flat)
                elif tensor.dtype == torch.bfloat16:
                    pinned_buf[:numel].view(torch.bfloat16).copy_(flat)
                else:
                    # For other dtypes, just copy as bytes
                    pinned_buf[:numel].view(tensor.dtype).copy_(flat)

                # Async copy to GPU
                stream = streams[buffer_idx]
                with torch.cuda.stream(stream):
                    gpu_buffer[:numel].copy_(pinned_buf[:numel], non_blocking=True)

                event = torch.cuda.Event()
                event.record(stream)
                pending_copies.append((stream, event))

                buffer_idx = (buffer_idx + 1) % n_buffers

    # Wait for all pending copies
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def load_sequential(st_files: List[str], gpu_buffer: torch.Tensor) -> float:
    """Sequential loading - baseline."""
    t0 = time.perf_counter()

    for sf in st_files:
        with safe_open(sf, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                numel = tensor.numel()
                gpu_buffer[:numel].copy_(tensor.view(-1)[:numel])

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    print("="*70)
    print("CONCURRENCY SWEEP FOR PIPELINED LOADING")
    print("CPU: Ryzen 9 8945HS (8 cores)")
    print("="*70)

    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Test with both models
    models = [
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen (4 files)"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral (1 file)"),
    ]

    # Concurrency levels to test
    buffer_counts = [1, 2, 3, 4, 6, 8, 12, 16]

    # Allocate GPU buffer
    gpu_buffer = torch.empty(600_000_000, dtype=torch.float16, device='cuda')

    for model_name, label in models:
        print(f"\n{'='*70}")
        print(f"Model: {label}")
        print("="*70)

        try:
            st_files = find_safetensors(model_name)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        total_bytes = sum(Path(f).stat().st_size for f in st_files)
        gb = total_bytes / (1024**3)
        print(f"Total: {gb:.2f} GB in {len(st_files)} files")

        results = []

        # Baseline
        print("\n  Testing Sequential (baseline)...")
        gc.collect()
        torch.cuda.synchronize()
        elapsed = load_sequential(st_files, gpu_buffer)
        bw = gb / elapsed
        results.append(("Sequential", 0, elapsed * 1000, bw))
        print(f"    {elapsed*1000:.0f}ms ({bw:.1f} GB/s)")

        # Test each concurrency level
        for n in buffer_counts:
            print(f"  Testing {n} buffers...")
            gc.collect()
            torch.cuda.synchronize()

            # Run 3 times and take average
            times = []
            for _ in range(3):
                elapsed = load_pipelined(st_files, gpu_buffer, n_buffers=n)
                times.append(elapsed)

            avg_elapsed = sum(times) / len(times)
            bw = gb / avg_elapsed
            results.append((f"Pipelined", n, avg_elapsed * 1000, bw))
            print(f"    {avg_elapsed*1000:.0f}ms ({bw:.1f} GB/s)")

        # Summary for this model
        print(f"\n  {'Approach':<20} {'Buffers':<10} {'Time (ms)':<12} {'BW (GB/s)':<12} {'Speedup':<10}")
        print("  " + "-"*64)
        baseline = results[0][2]
        for name, n, time_ms, bw in results:
            speedup = baseline / time_ms
            if n == 0:
                print(f"  {name:<20} {'N/A':<10} {time_ms:<12.0f} {bw:<12.1f} {speedup:<10.2f}x")
            else:
                print(f"  {name:<20} {n:<10} {time_ms:<12.0f} {bw:<12.1f} {speedup:<10.2f}x")

        # Find optimal
        pipelined_results = [r for r in results if r[1] > 0]
        if pipelined_results:
            best = max(pipelined_results, key=lambda x: x[3])  # max bandwidth
            print(f"\n  OPTIMAL: {best[1]} buffers ({best[3]:.1f} GB/s, {baseline/best[2]:.2f}x speedup)")

    del gpu_buffer
    gc.collect()
    torch.cuda.empty_cache()
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
