#!/usr/bin/env python3
"""
Fast weight loading: Bypass safetensors overhead.
Test different loading strategies to find the fastest approach.
"""

import os
import sys
import gc
import time
import types
import mmap
from pathlib import Path

os.environ['HIP_VISIBLE_DEVICES'] = '0'

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def test_loading_methods():
    print("=" * 70)
    print("FAST WEIGHT LOADING TEST")
    print("=" * 70)

    from huggingface_hub import snapshot_download

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_path = snapshot_download(model_name)
    safetensor_files = list(Path(model_path).glob("*.safetensors"))

    # Use just one file for testing
    test_file = safetensor_files[0]
    print(f"Test file: {test_file.name}")

    initial = get_mem()
    print(f"Initial GPU: {initial:.2f} GB")

    # Method 1: Standard load_file to GPU
    print("\n>>> Method 1: load_file(device='cuda')")
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    weights1 = load_file(str(test_file), device="cuda")
    torch.cuda.synchronize()
    method1_time = (time.perf_counter() - t0) * 1000

    file_gb = sum(t.numel() * t.element_size() for t in weights1.values()) / (1024**3)
    bw1 = file_gb / (method1_time / 1000)
    print(f"  Time: {method1_time:.0f}ms for {file_gb:.2f} GB ({bw1:.2f} GB/s)")

    del weights1
    gc.collect()
    torch.cuda.empty_cache()

    # Method 2: Load to CPU then transfer
    print("\n>>> Method 2: load_file(device='cpu') + .to('cuda')")
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    weights2_cpu = load_file(str(test_file), device="cpu")
    cpu_load_time = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    weights2_gpu = {k: v.to('cuda', non_blocking=True) for k, v in weights2_cpu.items()}
    torch.cuda.synchronize()
    transfer_time = (time.perf_counter() - t1) * 1000

    method2_time = cpu_load_time + transfer_time
    bw2 = file_gb / (method2_time / 1000)
    print(f"  CPU load: {cpu_load_time:.0f}ms")
    print(f"  Transfer: {transfer_time:.0f}ms")
    print(f"  Total: {method2_time:.0f}ms ({bw2:.2f} GB/s)")

    del weights2_cpu, weights2_gpu
    gc.collect()
    torch.cuda.empty_cache()

    # Method 3: Memory-mapped loading with safe_open
    print("\n>>> Method 3: safe_open (memory mapped)")
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    weights3 = {}
    with safe_open(str(test_file), framework="pt", device="cuda") as f:
        for key in f.keys():
            weights3[key] = f.get_tensor(key)
    torch.cuda.synchronize()
    method3_time = (time.perf_counter() - t0) * 1000

    bw3 = file_gb / (method3_time / 1000)
    print(f"  Time: {method3_time:.0f}ms ({bw3:.2f} GB/s)")

    del weights3
    gc.collect()
    torch.cuda.empty_cache()

    # Method 4: Parallel tensor loading
    print("\n>>> Method 4: Parallel loading (ThreadPoolExecutor)")
    from concurrent.futures import ThreadPoolExecutor
    gc.collect()
    torch.cuda.empty_cache()

    def load_tensor(key_file):
        key, file_path = key_file
        with safe_open(str(file_path), framework="pt", device="cuda") as f:
            return key, f.get_tensor(key)

    t0 = time.perf_counter()
    weights4 = {}
    with safe_open(str(test_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for key in keys:
            futures.append(executor.submit(load_tensor, (key, test_file)))
        for future in futures:
            k, v = future.result()
            weights4[k] = v

    torch.cuda.synchronize()
    method4_time = (time.perf_counter() - t0) * 1000

    bw4 = file_gb / (method4_time / 1000)
    print(f"  Time: {method4_time:.0f}ms ({bw4:.2f} GB/s)")

    del weights4
    gc.collect()
    torch.cuda.empty_cache()

    # Method 5: Streaming load (tensor by tensor)
    print("\n>>> Method 5: Streaming load to pre-allocated GPU memory")
    gc.collect()
    torch.cuda.empty_cache()

    # First, get metadata to pre-allocate
    with safe_open(str(test_file), framework="pt", device="cpu") as f:
        metadata = [(key, f.get_tensor(key).shape, f.get_tensor(key).dtype) for key in f.keys()]

    # Pre-allocate GPU tensors
    t0 = time.perf_counter()
    weights5 = {}
    for key, shape, dtype in metadata:
        weights5[key] = torch.empty(shape, dtype=dtype, device='cuda')

    # Stream from CPU to GPU
    with safe_open(str(test_file), framework="pt", device="cpu") as f:
        for key in f.keys():
            cpu_tensor = f.get_tensor(key)
            weights5[key].copy_(cpu_tensor, non_blocking=True)

    torch.cuda.synchronize()
    method5_time = (time.perf_counter() - t0) * 1000

    bw5 = file_gb / (method5_time / 1000)
    print(f"  Time: {method5_time:.0f}ms ({bw5:.2f} GB/s)")

    del weights5
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"File size: {file_gb:.2f} GB")
    print(f"Method 1 (load_file GPU):   {method1_time:.0f}ms ({bw1:.2f} GB/s)")
    print(f"Method 2 (CPU + transfer):  {method2_time:.0f}ms ({bw2:.2f} GB/s)")
    print(f"Method 3 (safe_open mmap):  {method3_time:.0f}ms ({bw3:.2f} GB/s)")
    print(f"Method 4 (parallel):        {method4_time:.0f}ms ({bw4:.2f} GB/s)")
    print(f"Method 5 (streaming):       {method5_time:.0f}ms ({bw5:.2f} GB/s)")

    best_time = min(method1_time, method2_time, method3_time, method4_time, method5_time)
    best_bw = file_gb / (best_time / 1000)
    print(f"\nBest: {best_time:.0f}ms ({best_bw:.2f} GB/s)")

    theoretical = file_gb * 1000 / 20  # 20 GB/s max
    print(f"Theoretical (20 GB/s): {theoretical:.0f}ms")
    print(f"Gap: {best_time - theoretical:.0f}ms ({(best_time - theoretical) / best_time * 100:.0f}% overhead)")


def test_pytorch_binary_format():
    """Test if PyTorch's native binary format is faster."""
    print("\n" + "=" * 70)
    print("PYTORCH BINARY FORMAT TEST")
    print("=" * 70)

    from huggingface_hub import snapshot_download

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_path = snapshot_download(model_name)
    safetensor_files = list(Path(model_path).glob("*.safetensors"))
    test_file = safetensor_files[0]

    # Load from safetensors and save as .pt
    print(">>> Loading from safetensors...")
    weights = load_file(str(test_file), device="cpu")
    file_gb = sum(t.numel() * t.element_size() for t in weights.values()) / (1024**3)
    print(f"  Size: {file_gb:.2f} GB, {len(weights)} tensors")

    pt_file = Path("/tmp/test_weights.pt")
    print(">>> Saving as PyTorch binary...")
    torch.save(weights, pt_file)
    del weights
    gc.collect()

    # Test loading from .pt
    print(">>> Loading from .pt to GPU...")
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    weights_pt = torch.load(pt_file, map_location='cuda', weights_only=True)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - t0) * 1000

    bw_pt = file_gb / (pt_time / 1000)
    print(f"  Time: {pt_time:.0f}ms ({bw_pt:.2f} GB/s)")

    del weights_pt
    gc.collect()
    torch.cuda.empty_cache()

    # Compare to safetensors
    print(">>> Loading from safetensors to GPU...")
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    weights_st = load_file(str(test_file), device="cuda")
    torch.cuda.synchronize()
    st_time = (time.perf_counter() - t0) * 1000

    bw_st = file_gb / (st_time / 1000)
    print(f"  Time: {st_time:.0f}ms ({bw_st:.2f} GB/s)")

    print(f"\nPyTorch binary: {pt_time:.0f}ms")
    print(f"Safetensors:    {st_time:.0f}ms")

    if pt_time < st_time:
        print(f"PyTorch binary is {st_time / pt_time:.2f}x faster")
    else:
        print(f"Safetensors is {pt_time / st_time:.2f}x faster")

    # Cleanup
    pt_file.unlink()
    del weights_st
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_loading_methods()
    test_pytorch_binary_format()
