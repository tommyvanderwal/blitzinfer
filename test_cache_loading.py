#!/usr/bin/env python3
"""Test cached model loading performance."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'

import time
import sys
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer')

import torch
from pathlib import Path


def test_safetensors_loading():
    """Test loading from safetensors files."""
    from safetensors.torch import load_file

    model_path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"

    # Find safetensors files
    snapshot_dir = list((model_path / "snapshots").iterdir())[0]
    safetensor_files = list(snapshot_dir.glob("*.safetensors"))

    print(f"Found {len(safetensor_files)} safetensors files")

    start = time.time()
    state_dict = {}
    for f in sorted(safetensor_files):
        print(f"  Loading {f.name}...")
        state_dict.update(load_file(f, device="cpu"))
    elapsed = time.time() - start

    total_params = sum(p.numel() for p in state_dict.values())
    total_bytes = sum(p.numel() * p.element_size() for p in state_dict.values())

    print(f"\nSafetensors loading:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Params: {total_params / 1e9:.2f}B")
    print(f"  Size: {total_bytes / (1024**3):.2f} GB")
    print(f"  Speed: {total_bytes / (1024**3) / elapsed:.2f} GB/s")

    return state_dict


def test_torch_cache_save(state_dict, cache_path: Path):
    """Save to torch format."""
    print(f"\nSaving to torch cache: {cache_path}")
    start = time.time()
    torch.save(state_dict, cache_path, pickle_protocol=4)
    elapsed = time.time() - start
    size_gb = cache_path.stat().st_size / (1024**3)
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Size: {size_gb:.2f} GB")
    print(f"  Speed: {size_gb / elapsed:.2f} GB/s")


def test_torch_cache_load(cache_path: Path, use_mmap: bool = True):
    """Load from torch cache."""
    print(f"\nLoading from torch cache (mmap={use_mmap}): {cache_path}")
    start = time.time()
    state_dict = torch.load(cache_path, map_location="cpu", mmap=use_mmap)
    elapsed = time.time() - start
    size_gb = cache_path.stat().st_size / (1024**3)
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Size: {size_gb:.2f} GB")
    print(f"  Speed: {size_gb / elapsed:.2f} GB/s")
    return state_dict


def test_move_to_gpu(state_dict):
    """Test moving weights to GPU."""
    print("\nMoving weights to GPU...")
    start = time.time()

    gpu_state_dict = {}
    for k, v in state_dict.items():
        gpu_state_dict[k] = v.to("cuda", non_blocking=True)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_bytes = sum(p.numel() * p.element_size() for p in gpu_state_dict.values())
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {total_bytes / (1024**3) / elapsed:.2f} GB/s")

    return gpu_state_dict


def main():
    cache_path = Path("/home/tommy/pythonprojects/blitzinfer/.cache/qwen7b.pt")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model Loading Performance Test")
    print("=" * 60)

    # Test 1: Load from safetensors
    print("\n[Test 1] Load from safetensors files")
    state_dict = test_safetensors_loading()

    # Test 2: Save to torch cache
    print("\n[Test 2] Save to torch cache")
    test_torch_cache_save(state_dict, cache_path)

    # Clear the state dict from memory
    del state_dict
    import gc
    gc.collect()

    # Test 3: Load from torch cache with mmap
    print("\n[Test 3] Load from torch cache (with mmap)")
    state_dict = test_torch_cache_load(cache_path, use_mmap=True)

    # Test 4: Move to GPU
    print("\n[Test 4] Move weights to GPU")
    gpu_state_dict = test_move_to_gpu(state_dict)

    # Cleanup
    del state_dict, gpu_state_dict
    gc.collect()
    torch.cuda.empty_cache()

    # Test 5: Load from cache again (should be hot in OS cache)
    print("\n[Test 5] Load from torch cache (second time - OS cached)")
    state_dict = test_torch_cache_load(cache_path, use_mmap=True)

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
