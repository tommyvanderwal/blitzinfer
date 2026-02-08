#!/usr/bin/env python3
"""Benchmark raw pinned → GPU transfer speed to understand limits."""

import os
import gc
import time
import subprocess

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

import torch
from pathlib import Path
from huggingface_hub import snapshot_download

from blitzinfer.memory import (
    PinnedMemoryArena,
    load_model_to_arena,
    get_model_size,
    get_premerged_tensors_for_vllm,
)


def nvidia_smi_free_gb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def cleanup():
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    print("=" * 80)
    print("RAW TRANSFER SPEED BENCHMARK")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    _ = torch.randn(1000, device='cuda')
    cleanup()

    model_name = "Qwen/Qwen3-VL-32B-Thinking-FP8"
    model_path = Path(snapshot_download(model_name, local_files_only=True))
    model_size = get_model_size(str(model_path))
    model_gb = model_size / 1e9

    print(f"\nModel: {model_name}")
    print(f"Size: {model_gb:.1f} GB")

    # Load into arena
    arena = PinnedMemoryArena(model_gb + 5)
    load_model_to_arena(str(model_path), arena, model_name)
    pinned_tensors = arena.get_all_tensors(model_name)

    print(f"\nLoaded {len(pinned_tensors)} tensors into pinned arena")

    # Test 1: Raw batch transfer (all tensors, non-blocking)
    print("\n" + "=" * 60)
    print("TEST 1: Batch transfer with non_blocking=True")
    print("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    total_bytes = 0
    for t in pinned_tensors.values():
        total_bytes += t.numel() * t.element_size()

    print(f"Total bytes to transfer: {total_bytes / 1e9:.2f} GB")

    # Warmup
    for _ in range(2):
        gpu_tensors = {}
        for name, tensor in pinned_tensors.items():
            gpu_tensors[name] = tensor.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        del gpu_tensors
        gc.collect()
        torch.cuda.empty_cache()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    gpu_tensors = {}
    for name, tensor in pinned_tensors.items():
        gpu_tensors[name] = tensor.to('cuda', non_blocking=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bw = total_bytes / 1e9 / elapsed

    print(f"Time: {elapsed:.3f}s")
    print(f"Bandwidth: {bw:.1f} GB/s")

    del gpu_tensors
    cleanup()

    # Test 2: Batch transfer with explicit streams
    print("\n" + "=" * 60)
    print("TEST 2: Using multiple CUDA streams")
    print("=" * 60)

    # Create streams
    num_streams = 4
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    # Group tensors by stream
    tensor_list = list(pinned_tensors.items())
    chunk_size = len(tensor_list) // num_streams + 1

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    gpu_tensors = {}
    for i, stream in enumerate(streams):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(tensor_list))
        with torch.cuda.stream(stream):
            for name, tensor in tensor_list[start:end]:
                gpu_tensors[name] = tensor.to('cuda', non_blocking=True)

    # Wait for all streams
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bw = total_bytes / 1e9 / elapsed

    print(f"Time: {elapsed:.3f}s")
    print(f"Bandwidth: {bw:.1f} GB/s")

    del gpu_tensors
    del streams
    cleanup()

    # Test 3: Copy to pre-allocated GPU tensors
    print("\n" + "=" * 60)
    print("TEST 3: Copy to pre-allocated GPU tensors")
    print("=" * 60)

    # Pre-allocate GPU tensors
    print("Pre-allocating GPU tensors...")
    gpu_tensors = {}
    for name, tensor in pinned_tensors.items():
        gpu_tensors[name] = torch.empty_like(tensor, device='cuda')

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Copy data
    for name, tensor in pinned_tensors.items():
        gpu_tensors[name].copy_(tensor, non_blocking=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bw = total_bytes / 1e9 / elapsed

    print(f"Time: {elapsed:.3f}s")
    print(f"Bandwidth: {bw:.1f} GB/s")

    del gpu_tensors
    cleanup()

    # Test 4: Copy merged tensors only
    print("\n" + "=" * 60)
    print("TEST 4: Transfer pre-merged tensors")
    print("=" * 60)

    premerged = get_premerged_tensors_for_vllm(pinned_tensors)
    merged_bytes = 0
    for t in premerged.values():
        merged_bytes += t.numel() * t.element_size()

    print(f"Pre-merged tensors: {len(premerged)} ({merged_bytes / 1e9:.2f} GB)")

    # Filter to unique tensors (avoid double-counting normalized names)
    unique_tensors = {}
    seen_ids = set()
    for name, tensor in premerged.items():
        tid = id(tensor)
        if tid not in seen_ids:
            unique_tensors[name] = tensor
            seen_ids.add(tid)

    unique_bytes = sum(t.numel() * t.element_size() for t in unique_tensors.values())
    print(f"Unique tensors: {len(unique_tensors)} ({unique_bytes / 1e9:.2f} GB)")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    gpu_tensors = {}
    for name, tensor in unique_tensors.items():
        gpu_tensors[name] = tensor.to('cuda', non_blocking=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bw = unique_bytes / 1e9 / elapsed

    print(f"Time: {elapsed:.3f}s")
    print(f"Bandwidth: {bw:.1f} GB/s")

    del gpu_tensors
    del premerged
    cleanup()

    arena.clear()
    del arena
    cleanup()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
