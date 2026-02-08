#!/usr/bin/env python3
"""
Fast model loader using pinned memory for near-hardware-speed GPU transfer.

Key insight: Regular CPU→GPU transfer is 8.6 GB/s but pinned memory is 45 GB/s!
This is a 5x speedup.

Strategy:
1. Pre-allocate pinned memory buffer
2. Load safetensors to pinned memory
3. Async transfer to GPU
4. Pipeline: while GPU transfers one shard, CPU loads next

Target: 65 GB at ~45 GB/s = ~1.5s (vs current 10s)
"""

import os
import sys
import time
import glob
from pathlib import Path
from typing import Dict, List, Optional
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


class FastTensorLoader:
    """Fast tensor loading using pinned memory and pipelining"""

    def __init__(self, device='cuda'):
        self.device = device
        self.pinned_buffers: Dict[str, torch.Tensor] = {}

    def load_safetensor_pinned(self, file_path: str) -> Dict[str, torch.Tensor]:
        """Load safetensor file to pinned CPU memory"""
        tensors = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                cpu_tensor = f.get_tensor(key)
                # Allocate pinned memory and copy
                pinned = torch.empty_like(cpu_tensor, pin_memory=True)
                pinned.copy_(cpu_tensor)
                tensors[key] = pinned
        return tensors

    def transfer_to_gpu(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Transfer tensors from pinned CPU memory to GPU"""
        gpu_tensors = {}
        for key, tensor in tensors.items():
            gpu_tensors[key] = tensor.to(self.device, non_blocking=True)
        return gpu_tensors

    def load_model_pipelined(self, model_path: Path) -> Dict[str, torch.Tensor]:
        """
        Load model with pipelining:
        - While GPU transfers shard N, CPU loads shard N+1 to pinned memory
        """
        safetensor_files = sorted(glob.glob(str(model_path / "*.safetensors")))
        if not safetensor_files:
            safetensor_files = sorted(glob.glob(str(model_path / "model*.safetensors")))

        all_tensors = {}
        total_size = 0

        # Pipelined loading
        cpu_load_future = None
        prev_pinned = None

        with ThreadPoolExecutor(max_workers=1) as executor:
            for i, sf in enumerate(safetensor_files):
                # Start loading next file to pinned memory (async)
                if i < len(safetensor_files):
                    cpu_load_future = executor.submit(self.load_safetensor_pinned, sf)

                # If we have previous data, transfer to GPU
                if prev_pinned is not None:
                    gpu_tensors = self.transfer_to_gpu(prev_pinned)
                    all_tensors.update(gpu_tensors)
                    total_size += sum(t.numel() * t.element_size() for t in gpu_tensors.values())

                # Wait for CPU load to complete
                if cpu_load_future is not None:
                    prev_pinned = cpu_load_future.result()

            # Transfer last batch
            if prev_pinned is not None:
                gpu_tensors = self.transfer_to_gpu(prev_pinned)
                all_tensors.update(gpu_tensors)
                total_size += sum(t.numel() * t.element_size() for t in gpu_tensors.values())

        torch.cuda.synchronize()
        return all_tensors, total_size

    def load_model_simple(self, model_path: Path) -> Dict[str, torch.Tensor]:
        """Simple pinned memory loading (no pipelining, for comparison)"""
        safetensor_files = sorted(glob.glob(str(model_path / "*.safetensors")))

        all_tensors = {}
        total_size = 0

        for sf in safetensor_files:
            # Load to pinned memory
            pinned = self.load_safetensor_pinned(sf)
            # Transfer to GPU
            gpu_tensors = self.transfer_to_gpu(pinned)
            all_tensors.update(gpu_tensors)
            total_size += sum(t.numel() * t.element_size() for t in gpu_tensors.values())
            # Free pinned memory
            del pinned

        torch.cuda.synchronize()
        return all_tensors, total_size


def benchmark_loading_methods(model_name: str):
    """Compare different loading methods"""
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

    loader = FastTensorLoader()
    results = {}

    # Method 1: Direct safetensor GPU load (baseline)
    print("\n--- Method 1: Direct GPU Load (baseline) ---")
    torch.cuda.empty_cache()
    t0 = time.time()
    all_tensors = {}
    for sf in safetensor_files:
        with safe_open(sf, framework="pt", device="cuda") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)
    torch.cuda.synchronize()
    t1 = time.time()
    tensor_size = sum(t.numel() * t.element_size() for t in all_tensors.values())
    speed = tensor_size / (t1 - t0) / 1e9
    print(f"Time: {t1-t0:.2f}s, Size: {tensor_size/1e9:.2f} GB, Speed: {speed:.2f} GB/s")
    results['direct'] = {'time': t1-t0, 'speed': speed}
    del all_tensors
    torch.cuda.empty_cache()

    # Method 2: Simple pinned memory load
    print("\n--- Method 2: Pinned Memory (simple) ---")
    torch.cuda.empty_cache()
    t0 = time.time()
    all_tensors, tensor_size = loader.load_model_simple(model_path)
    t1 = time.time()
    speed = tensor_size / (t1 - t0) / 1e9
    print(f"Time: {t1-t0:.2f}s, Size: {tensor_size/1e9:.2f} GB, Speed: {speed:.2f} GB/s")
    results['pinned_simple'] = {'time': t1-t0, 'speed': speed}
    del all_tensors
    torch.cuda.empty_cache()

    # Method 3: Pipelined pinned memory load
    print("\n--- Method 3: Pinned Memory (pipelined) ---")
    torch.cuda.empty_cache()
    t0 = time.time()
    all_tensors, tensor_size = loader.load_model_pipelined(model_path)
    t1 = time.time()
    speed = tensor_size / (t1 - t0) / 1e9
    print(f"Time: {t1-t0:.2f}s, Size: {tensor_size/1e9:.2f} GB, Speed: {speed:.2f} GB/s")
    results['pinned_pipelined'] = {'time': t1-t0, 'speed': speed}
    del all_tensors
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    baseline = results['direct']['time']
    for method, data in results.items():
        speedup = baseline / data['time']
        print(f"{method:20s}: {data['time']:.2f}s ({data['speed']:.1f} GB/s) - {speedup:.2f}x baseline")

    print(f"\nTheoretical: 65 GB @ 45 GB/s = 1.44s ({baseline/1.44:.1f}x possible speedup)")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"

    if model == "gpt":
        model = "openai/gpt-oss-120b"
    elif model == "qwen":
        model = "Qwen/Qwen3-VL-32B-Instruct"

    benchmark_loading_methods(model)


if __name__ == "__main__":
    main()
