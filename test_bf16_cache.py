#!/usr/bin/env python3
"""Test bf16 cached loading - the fast path."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'

import time
from pathlib import Path
import gc

import torch
from safetensors.torch import load_file, save_file


CACHE_DIR = Path("/home/tommy/pythonprojects/blitzinfer/.cache")


def create_bf16_cache(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Create bf16 cached weights."""
    model_path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
    snapshot_dir = list((model_path / "snapshots").iterdir())[0]
    safetensor_files = sorted(list(snapshot_dir.glob("*.safetensors")))

    cache_file = CACHE_DIR / "qwen7b_bf16.safetensors"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        print(f"Cache exists: {cache_file}")
        return cache_file

    print("Creating bf16 cache...")
    start = time.time()

    # Load all weights
    state_dict = {}
    for f in safetensor_files:
        state_dict.update(load_file(f, device="cpu"))

    # Convert to bf16
    bf16_dict = {}
    for k, v in state_dict.items():
        bf16_dict[k] = v.to(dtype=torch.bfloat16).contiguous()

    del state_dict
    gc.collect()

    # Save as single safetensors file
    save_file(bf16_dict, cache_file)

    elapsed = time.time() - start
    size_gb = cache_file.stat().st_size / (1024**3)
    print(f"Created cache in {elapsed:.2f}s ({size_gb:.2f} GB)")

    return cache_file


def test_cached_load(cache_file: Path):
    """Test loading from bf16 cache."""
    print("\n" + "=" * 60)
    print("Testing bf16 cached loading")
    print("=" * 60)

    # Cold load (first time)
    print("\n[Test 1] Cold load (first access)")
    gc.collect()
    torch.cuda.empty_cache()

    start = time.time()
    state_dict = load_file(cache_file, device="cpu")
    load_time = time.time() - start
    print(f"  Load time: {load_time:.3f}s")

    # Move to GPU
    start = time.time()
    gpu_dict = {}
    for k, v in state_dict.items():
        gpu_dict[k] = v.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"  GPU move time: {gpu_time:.2f}s")
    print(f"  Total: {load_time + gpu_time:.2f}s")

    # Cleanup
    del state_dict, gpu_dict
    gc.collect()
    torch.cuda.empty_cache()

    # Warm load (file in OS cache)
    print("\n[Test 2] Warm load (OS page cache)")
    start = time.time()
    state_dict = load_file(cache_file, device="cpu")
    load_time = time.time() - start
    print(f"  Load time: {load_time:.3f}s")

    start = time.time()
    gpu_dict = {}
    for k, v in state_dict.items():
        gpu_dict[k] = v.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"  GPU move time: {gpu_time:.2f}s")
    print(f"  Total: {load_time + gpu_time:.2f}s")

    # Test 3: Direct to GPU
    del state_dict, gpu_dict
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[Test 3] Direct load to GPU")
    start = time.time()
    gpu_dict = load_file(cache_file, device="cuda")
    torch.cuda.synchronize()
    direct_time = time.time() - start
    print(f"  Time: {direct_time:.2f}s")

    print("\n" + "=" * 60)


def main():
    # Create cache if needed
    cache_file = create_bf16_cache()

    # Test loading
    test_cached_load(cache_file)


if __name__ == '__main__':
    main()
