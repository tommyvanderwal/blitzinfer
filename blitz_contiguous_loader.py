#!/usr/bin/env python3
"""
Fast model loader using contiguous memory for optimal GPU transfer.

Key insight: Safetensor mmap tensors are non-contiguous, causing slow GPU transfer (8 GB/s).
Cloning to contiguous memory before transfer gives 18.9 GB/s - a 2.4x speedup!

For full model: 65 GB at 18.9 GB/s = 3.4s (vs current 9s = 2.6x faster)
"""

import os
import sys
import time
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

import torch
from safetensors import safe_open


def get_model_path(model_name: str) -> Path:
    """Get local path for HuggingFace model"""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_dir = f"models--{model_name.replace('/', '--')}"
    model_path = cache_dir / model_dir / "snapshots"
    if model_path.exists():
        snapshots = list(model_path.iterdir())
        if snapshots:
            return snapshots[0]
    raise ValueError(f"Model not found: {model_name}")


def load_file_contiguous_gpu(file_path: str) -> dict:
    """Load safetensor file to GPU using contiguous memory for speed"""
    gpu_tensors = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            # Get mmap tensor
            cpu_t = f.get_tensor(key)
            # Clone to contiguous memory, then transfer
            gpu_tensors[key] = cpu_t.clone().to("cuda", non_blocking=True)
    return gpu_tensors


def load_model_contiguous(model_path: Path, parallel: int = 4) -> tuple:
    """Load model using contiguous memory optimization"""
    safetensor_files = sorted(glob.glob(str(model_path / "*.safetensors")))

    all_tensors = {}

    if parallel == 1:
        # Sequential
        for sf in safetensor_files:
            tensors = load_file_contiguous_gpu(sf)
            all_tensors.update(tensors)
    else:
        # Parallel file loading
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(load_file_contiguous_gpu, sf) for sf in safetensor_files]
            for future in futures:
                tensors = future.result()
                all_tensors.update(tensors)

    torch.cuda.synchronize()

    total_size = sum(t.numel() * t.element_size() for t in all_tensors.values())
    return all_tensors, total_size


def load_file_direct_gpu(file_path: str) -> dict:
    """Load safetensor file directly to GPU (baseline)"""
    gpu_tensors = {}
    with safe_open(file_path, framework="pt", device="cuda") as f:
        for key in f.keys():
            gpu_tensors[key] = f.get_tensor(key)
    return gpu_tensors


def load_model_direct(model_path: Path) -> tuple:
    """Load model directly to GPU (baseline)"""
    safetensor_files = sorted(glob.glob(str(model_path / "*.safetensors")))

    all_tensors = {}
    for sf in safetensor_files:
        tensors = load_file_direct_gpu(sf)
        all_tensors.update(tensors)

    torch.cuda.synchronize()

    total_size = sum(t.numel() * t.element_size() for t in all_tensors.values())
    return all_tensors, total_size


def benchmark_model(model_name: str):
    """Benchmark loading methods for a model"""
    print("="*60)
    print(f"Benchmarking: {model_name}")
    print("="*60)

    model_path = get_model_path(model_name)
    safetensor_files = sorted(glob.glob(str(model_path / "*.safetensors")))
    total_file_size = sum(os.path.getsize(f) for f in safetensor_files)

    print(f"Path: {model_path}")
    print(f"Files: {len(safetensor_files)}")
    print(f"Total size: {total_file_size/1e9:.2f} GB")

    # Check GPU memory
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: {free/1e9:.2f}/{total/1e9:.2f} GiB free")

    # Warm page cache
    print("\nWarming page cache...")
    for sf in safetensor_files:
        with open(sf, "rb") as f:
            _ = f.read()
    print("Cache warmed.")

    results = {}

    # Baseline: Direct GPU load
    print("\n--- Baseline: Direct GPU Load ---")
    torch.cuda.empty_cache()
    t0 = time.time()
    tensors, size = load_model_direct(model_path)
    t1 = time.time()
    speed = size / (t1 - t0) / 1e9
    print(f"Time: {t1-t0:.2f}s, Size: {size/1e9:.2f} GB, Speed: {speed:.1f} GB/s")
    results['direct'] = {'time': t1-t0, 'speed': speed}
    del tensors
    torch.cuda.empty_cache()

    # Contiguous: Sequential
    print("\n--- Contiguous: Sequential ---")
    torch.cuda.empty_cache()
    t0 = time.time()
    tensors, size = load_model_contiguous(model_path, parallel=1)
    t1 = time.time()
    speed = size / (t1 - t0) / 1e9
    print(f"Time: {t1-t0:.2f}s, Speed: {speed:.1f} GB/s")
    results['contiguous_seq'] = {'time': t1-t0, 'speed': speed}
    del tensors
    torch.cuda.empty_cache()

    # Contiguous: Parallel (2 threads)
    print("\n--- Contiguous: Parallel (2 threads) ---")
    torch.cuda.empty_cache()
    t0 = time.time()
    tensors, size = load_model_contiguous(model_path, parallel=2)
    t1 = time.time()
    speed = size / (t1 - t0) / 1e9
    print(f"Time: {t1-t0:.2f}s, Speed: {speed:.1f} GB/s")
    results['contiguous_p2'] = {'time': t1-t0, 'speed': speed}
    del tensors
    torch.cuda.empty_cache()

    # Contiguous: Parallel (4 threads)
    print("\n--- Contiguous: Parallel (4 threads) ---")
    torch.cuda.empty_cache()
    t0 = time.time()
    tensors, size = load_model_contiguous(model_path, parallel=4)
    t1 = time.time()
    speed = size / (t1 - t0) / 1e9
    print(f"Time: {t1-t0:.2f}s, Speed: {speed:.1f} GB/s")
    results['contiguous_p4'] = {'time': t1-t0, 'speed': speed}
    del tensors
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    baseline_time = results['direct']['time']
    for method, data in results.items():
        speedup = baseline_time / data['time']
        print(f"{method:20s}: {data['time']:.2f}s ({data['speed']:.1f} GB/s) - {speedup:.2f}x baseline")

    best = min(results.values(), key=lambda x: x['time'])
    print(f"\nBest: {best['time']:.2f}s ({best['speed']:.1f} GB/s)")
    print(f"Theoretical (18.9 GB/s): {size/1e9/18.9:.2f}s")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"

    if model == "gpt":
        model = "openai/gpt-oss-120b"
    elif model == "qwen":
        model = "Qwen/Qwen3-VL-32B-Instruct"

    benchmark_model(model)


if __name__ == "__main__":
    main()
