#!/usr/bin/env python3
"""
Fast weight loader: Bypass safetensors per-tensor GPU allocation.
Uses bulk CPU→GPU transfer for maximum bandwidth.
"""

import os
import sys
import gc
import time
import types
from pathlib import Path

os.environ['HIP_VISIBLE_DEVICES'] = '0'

import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def load_weights_standard(safetensor_files):
    """Standard loading: safetensors direct to GPU."""
    all_weights = {}
    for sf in safetensor_files:
        weights = load_file(str(sf), device='cuda')
        torch.cuda.synchronize()
        all_weights.update(weights)
    return all_weights


def load_weights_fast(safetensor_files):
    """Fast loading: CPU load, bulk transfer, slice back."""
    all_weights = {}

    for sf in safetensor_files:
        # Load to CPU (very fast - memory mapped)
        weights_cpu = load_file(str(sf), device='cpu')

        # Get metadata for slicing
        metadata = []
        offset = 0
        for key, tensor in weights_cpu.items():
            flat = tensor.flatten()
            metadata.append((key, tensor.shape, tensor.dtype, offset, flat.numel()))
            offset += flat.numel()

        # Concatenate all tensors
        flat_list = [w.flatten() for w in weights_cpu.values()]
        big = torch.cat(flat_list)

        # Single bulk transfer
        big_gpu = big.to('cuda', non_blocking=True)
        torch.cuda.synchronize()

        # Slice back into individual tensors
        for key, shape, dtype, start, length in metadata:
            tensor = big_gpu[start:start+length].view(shape)
            all_weights[key] = tensor

        del weights_cpu, flat_list, big

    return all_weights


def test_loading():
    print("=" * 70)
    print("FAST WEIGHT LOADER TEST")
    print("=" * 70)

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    model_path = snapshot_download('Qwen/Qwen2.5-7B-Instruct')
    safetensor_files = sorted(Path(model_path).glob('*.safetensors'))

    print(f"Files: {len(safetensor_files)}")

    # Method 1: Standard
    print("\n>>> Standard loading (safetensors direct to GPU)...")
    gc.collect()
    torch.cuda.empty_cache()
    initial = get_mem()

    t0 = time.perf_counter()
    weights1 = load_weights_standard(safetensor_files)
    standard_time = (time.perf_counter() - t0) * 1000

    total_gb = sum(t.numel() * t.element_size() for t in weights1.values()) / (1024**3)
    print(f"Time: {standard_time:.0f}ms for {total_gb:.1f}GB ({total_gb/(standard_time/1000):.1f} GB/s)")

    del weights1
    gc.collect()
    torch.cuda.empty_cache()

    # Method 2: Fast
    print("\n>>> Fast loading (CPU + bulk transfer)...")
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    weights2 = load_weights_fast(safetensor_files)
    fast_time = (time.perf_counter() - t0) * 1000

    print(f"Time: {fast_time:.0f}ms for {total_gb:.1f}GB ({total_gb/(fast_time/1000):.1f} GB/s)")

    # Verify weights are correct
    print("\n>>> Verifying weights...")
    sample_key = list(weights2.keys())[0]
    print(f"Sample: {sample_key}, shape={weights2[sample_key].shape}, device={weights2[sample_key].device}")

    del weights2
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Standard: {standard_time:.0f}ms")
    print(f"Fast:     {fast_time:.0f}ms")
    print(f"Speedup:  {standard_time/fast_time:.2f}x")

    saved = standard_time - fast_time
    print(f"\nTime saved: {saved:.0f}ms ({saved/1000:.1f}s)")


if __name__ == '__main__':
    test_loading()
