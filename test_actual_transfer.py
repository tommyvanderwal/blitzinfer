#!/usr/bin/env python3
"""Test actual data transfer time."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'

import time
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    model_path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
    snapshot_dir = list((model_path / "snapshots").iterdir())[0]
    safetensor_files = sorted(list(snapshot_dir.glob("*.safetensors")))

    print("=" * 60)
    print("Actual Data Transfer Test")
    print("=" * 60)

    # Test 1: Load safetensors (mmap - instant)
    print("\n[Test 1] Load safetensors (mmap)")
    start = time.time()
    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f, device="cpu"))
    mmap_time = time.time() - start
    print(f"  Time: {mmap_time:.3f}s (mmap, no actual read)")

    # Test 2: Force read all data into memory
    print("\n[Test 2] Force read all data into memory")
    start = time.time()
    total_bytes = 0
    for k, v in state_dict.items():
        # Accessing .data_ptr() forces the mmap to read
        _ = v.data_ptr()
        total_bytes += v.numel() * v.element_size()
    read_time = time.time() - start
    total_gb = total_bytes / (1024**3)
    print(f"  Time: {read_time:.2f}s")
    print(f"  Size: {total_gb:.2f} GB")
    print(f"  Speed: {total_gb / read_time:.2f} GB/s")

    # Test 3: Move to GPU
    print("\n[Test 3] Move to GPU (bfloat16)")
    start = time.time()
    gpu_dict = {}
    for k, v in state_dict.items():
        gpu_dict[k] = v.to("cuda", dtype=torch.bfloat16, non_blocking=True)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"  Time: {gpu_time:.2f}s")
    print(f"  Speed: {total_gb / gpu_time:.2f} GB/s")

    # Test 4: Pre-convert to bfloat16 on CPU, then move
    print("\n[Test 4] Pre-convert to bfloat16 on CPU")
    del gpu_dict
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    start = time.time()
    bf16_dict = {}
    for k, v in state_dict.items():
        bf16_dict[k] = v.to(dtype=torch.bfloat16)
    cpu_convert = time.time() - start
    print(f"  CPU convert time: {cpu_convert:.2f}s")

    bf16_bytes = sum(v.numel() * v.element_size() for v in bf16_dict.values())
    bf16_gb = bf16_bytes / (1024**3)

    start = time.time()
    gpu_dict = {}
    for k, v in bf16_dict.items():
        gpu_dict[k] = v.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    gpu_move_time = time.time() - start
    print(f"  GPU move time: {gpu_move_time:.2f}s")
    print(f"  BF16 size: {bf16_gb:.2f} GB")
    print(f"  Total: {cpu_convert + gpu_move_time:.2f}s")

    # Test 5: Direct load to GPU
    print("\n[Test 5] Direct load safetensors to GPU")
    del state_dict, bf16_dict, gpu_dict
    gc.collect()
    torch.cuda.empty_cache()

    start = time.time()
    gpu_dict = {}
    for f in safetensor_files:
        gpu_dict.update(load_file(f, device="cuda"))
    torch.cuda.synchronize()
    direct_time = time.time() - start
    print(f"  Time: {direct_time:.2f}s")
    print(f"  Speed: {total_gb / direct_time:.2f} GB/s")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Safetensors mmap: {mmap_time:.3f}s")
    print(f"  Force read to RAM: {read_time:.2f}s")
    print(f"  CPU->GPU transfer: {gpu_time:.2f}s")
    print(f"  Direct to GPU: {direct_time:.2f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
