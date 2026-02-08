#!/usr/bin/env python3
"""
Profile raw loading performance to find actual bottlenecks.

This bypasses vLLM to understand fundamental limits.
"""

import os
import sys
import time
import glob
from pathlib import Path

# Don't import torch/vllm yet - profile import time too


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


def profile_imports():
    """Profile import times"""
    print("\n=== Import Profiling ===")

    t0 = time.time()
    import torch
    t1 = time.time()
    print(f"import torch: {t1-t0:.2f}s")

    t0 = time.time()
    from safetensors import safe_open
    t1 = time.time()
    print(f"import safetensors: {t1-t0:.2f}s")

    return torch, safe_open


def profile_disk_io(file_path: str):
    """Profile raw disk read speed"""
    print(f"\n=== Disk I/O: {Path(file_path).name} ===")

    # Get file size
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size/1e9:.2f} GB")

    # Raw read (will use page cache if available)
    t0 = time.time()
    with open(file_path, 'rb') as f:
        data = f.read()
    t1 = time.time()
    speed = file_size / (t1 - t0) / 1e9
    print(f"Raw read: {t1-t0:.2f}s = {speed:.2f} GB/s")

    # Read again (should be from page cache)
    t0 = time.time()
    with open(file_path, 'rb') as f:
        data = f.read()
    t1 = time.time()
    speed = file_size / (t1 - t0) / 1e9
    print(f"Cached read: {t1-t0:.2f}s = {speed:.2f} GB/s")

    del data
    return file_size


def profile_safetensor_parsing(file_path: str, safe_open):
    """Profile safetensor parsing"""
    print(f"\n=== Safetensor Parsing: {Path(file_path).name} ===")

    # Parse to CPU
    t0 = time.time()
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}
    t1 = time.time()

    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    speed = total_size / (t1 - t0) / 1e9
    print(f"Parse to CPU: {t1-t0:.2f}s, {total_size/1e9:.2f} GB = {speed:.2f} GB/s")
    print(f"Tensors: {len(keys)}")

    return tensors, total_size


def profile_gpu_transfer(tensors: dict, torch, device='cuda'):
    """Profile CPU to GPU transfer"""
    print(f"\n=== GPU Transfer ===")

    total_size = sum(t.numel() * t.element_size() for t in tensors.values())

    # Warm up GPU
    torch.randn(1000, device=device)
    torch.cuda.synchronize()

    # Transfer all tensors
    t0 = time.time()
    gpu_tensors = {}
    for k, t in tensors.items():
        gpu_tensors[k] = t.to(device, non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.time()

    speed = total_size / (t1 - t0) / 1e9
    print(f"Transfer: {t1-t0:.2f}s, {total_size/1e9:.2f} GB = {speed:.2f} GB/s")

    return gpu_tensors


def profile_pinned_transfer(tensors: dict, torch, device='cuda'):
    """Profile transfer with pinned memory"""
    print(f"\n=== Pinned Memory Transfer ===")

    total_size = sum(t.numel() * t.element_size() for t in tensors.values())

    # Copy to pinned memory first
    t0 = time.time()
    pinned_tensors = {}
    for k, t in tensors.items():
        pinned = torch.empty_like(t, pin_memory=True)
        pinned.copy_(t)
        pinned_tensors[k] = pinned
    t1 = time.time()
    print(f"Copy to pinned: {t1-t0:.2f}s")

    # Transfer from pinned
    torch.cuda.synchronize()
    t0 = time.time()
    gpu_tensors = {}
    for k, t in pinned_tensors.items():
        gpu_tensors[k] = t.to(device, non_blocking=True)
    torch.cuda.synchronize()
    t1 = time.time()

    speed = total_size / (t1 - t0) / 1e9
    print(f"Pinned transfer: {t1-t0:.2f}s = {speed:.2f} GB/s")

    # Cleanup
    del pinned_tensors

    return gpu_tensors


def profile_direct_gpu_load(file_path: str, safe_open, torch, device='cuda'):
    """Profile direct loading to GPU"""
    print(f"\n=== Direct GPU Load: {Path(file_path).name} ===")

    t0 = time.time()
    with safe_open(file_path, framework="pt", device=device) as f:
        keys = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}
    torch.cuda.synchronize()
    t1 = time.time()

    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    speed = total_size / (t1 - t0) / 1e9
    print(f"Direct GPU load: {t1-t0:.2f}s, {total_size/1e9:.2f} GB = {speed:.2f} GB/s")

    return tensors


def profile_model(model_name: str):
    """Full profiling for a model"""
    print("\n" + "="*60)
    print(f"Profiling: {model_name}")
    print("="*60)

    # Get model path
    model_path = get_model_path(model_name)
    print(f"Path: {model_path}")

    # Find safetensor files
    safetensor_files = sorted(glob.glob(str(model_path / "*.safetensors")))
    print(f"Files: {len(safetensor_files)}")

    total_size = sum(os.path.getsize(f) for f in safetensor_files)
    print(f"Total size: {total_size/1e9:.2f} GB")

    # Import
    torch, safe_open = profile_imports()

    # Check GPU
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"\nGPU memory: {free/1e9:.2f}/{total/1e9:.2f} GiB free")

    # Profile first file (representative)
    test_file = safetensor_files[0]

    # Disk I/O
    profile_disk_io(test_file)

    # Safetensor parsing
    tensors, tensor_size = profile_safetensor_parsing(test_file, safe_open)

    if torch.cuda.is_available():
        # GPU transfer
        gpu_tensors = profile_gpu_transfer(tensors, torch)
        del gpu_tensors
        torch.cuda.empty_cache()

        # Pinned transfer
        gpu_tensors = profile_pinned_transfer(tensors, torch)
        del gpu_tensors
        torch.cuda.empty_cache()

        # Direct GPU load
        del tensors
        gpu_tensors = profile_direct_gpu_load(test_file, safe_open, torch)
        del gpu_tensors
        torch.cuda.empty_cache()

    # Full model load
    print(f"\n=== Full Model Load (all {len(safetensor_files)} files) ===")

    t0 = time.time()
    all_tensors = {}
    for sf in safetensor_files:
        with safe_open(sf, framework="pt", device="cuda" if torch.cuda.is_available() else "cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    total_tensor_size = sum(t.numel() * t.element_size() for t in all_tensors.values())
    speed = total_tensor_size / (t1 - t0) / 1e9
    print(f"Full load: {t1-t0:.2f}s, {total_tensor_size/1e9:.2f} GB = {speed:.2f} GB/s")
    print(f"Tensors: {len(all_tensors)}")

    # Cleanup
    del all_tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'model': model_name,
        'total_size_gb': total_size / 1e9,
        'load_time': t1 - t0,
        'speed_gbps': speed
    }


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"

    if model == "gpt":
        model = "openai/gpt-oss-120b"
    elif model == "qwen":
        model = "Qwen/Qwen3-VL-32B-Instruct"

    result = profile_model(model)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {result['model']}")
    print(f"Size: {result['total_size_gb']:.2f} GB")
    print(f"Load time: {result['load_time']:.2f}s")
    print(f"Effective speed: {result['speed_gbps']:.2f} GB/s")

    # Theoretical comparison
    print("\nTheoretical limits:")
    print(f"  NVMe SSD: ~7 GB/s -> {result['total_size_gb']/7:.1f}s")
    print(f"  PCIe 5.0 x16: ~64 GB/s -> {result['total_size_gb']/64:.1f}s")
    print(f"  DDR5 RAM: ~100 GB/s -> {result['total_size_gb']/100:.1f}s")


if __name__ == "__main__":
    main()
