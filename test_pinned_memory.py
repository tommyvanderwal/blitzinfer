#!/usr/bin/env python3
"""Test pinned memory for faster GPU access on iGPU."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'

import time
from pathlib import Path
import gc

import torch
from safetensors.torch import load_file


CACHE_FILE = Path("/home/tommy/pythonprojects/blitzinfer/.cache/qwen7b_bf16.safetensors")


def test_standard_transfer():
    """Standard CPU->GPU transfer."""
    print("\n[Standard transfer]")
    gc.collect()
    torch.cuda.empty_cache()

    state_dict = load_file(CACHE_FILE, device="cpu")

    start = time.time()
    gpu_dict = {}
    for k, v in state_dict.items():
        gpu_dict[k] = v.cuda(non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_gb = sum(v.numel() * v.element_size() for v in gpu_dict.values()) / (1024**3)
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {total_gb / elapsed:.2f} GB/s")

    return elapsed


def test_pinned_memory():
    """Use pinned memory for faster transfer."""
    print("\n[Pinned memory transfer]")
    gc.collect()
    torch.cuda.empty_cache()

    state_dict = load_file(CACHE_FILE, device="cpu")

    # Convert to pinned memory
    print("  Converting to pinned memory...")
    start = time.time()
    pinned_dict = {}
    for k, v in state_dict.items():
        pinned_dict[k] = v.pin_memory()
    pin_time = time.time() - start
    print(f"  Pin time: {pin_time:.2f}s")

    del state_dict
    gc.collect()

    # Transfer from pinned memory
    start = time.time()
    gpu_dict = {}
    for k, v in pinned_dict.items():
        gpu_dict[k] = v.cuda(non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_gb = sum(v.numel() * v.element_size() for v in gpu_dict.values()) / (1024**3)
    print(f"  Transfer time: {elapsed:.2f}s")
    print(f"  Speed: {total_gb / elapsed:.2f} GB/s")
    print(f"  Total (pin + transfer): {pin_time + elapsed:.2f}s")

    return pin_time + elapsed


def test_shared_memory():
    """Test if we can use shared memory (iGPU optimization)."""
    print("\n[Shared memory test]")
    gc.collect()
    torch.cuda.empty_cache()

    # Check if unified memory is being used
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA memory allocated before: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    state_dict = load_file(CACHE_FILE, device="cpu")

    # Try using share_memory_() for inter-process sharing
    # This may help with iGPU unified memory
    start = time.time()
    shared_dict = {}
    for k, v in state_dict.items():
        shared_dict[k] = v.share_memory_()
    share_time = time.time() - start
    print(f"  Share memory time: {share_time:.2f}s")

    # Now move to GPU
    start = time.time()
    gpu_dict = {}
    for k, v in shared_dict.items():
        gpu_dict[k] = v.cuda(non_blocking=True)
    torch.cuda.synchronize()
    transfer_time = time.time() - start
    print(f"  Transfer time: {transfer_time:.2f}s")

    print(f"  CUDA memory allocated after: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    print(f"  Total: {share_time + transfer_time:.2f}s")

    return share_time + transfer_time


def test_direct_gpu():
    """Load directly to GPU."""
    print("\n[Direct GPU load]")
    gc.collect()
    torch.cuda.empty_cache()

    start = time.time()
    gpu_dict = load_file(CACHE_FILE, device="cuda")
    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_gb = sum(v.numel() * v.element_size() for v in gpu_dict.values()) / (1024**3)
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {total_gb / elapsed:.2f} GB/s")

    return elapsed


def main():
    if not CACHE_FILE.exists():
        print(f"Cache file not found: {CACHE_FILE}")
        print("Run test_bf16_cache.py first to create it.")
        return

    print("=" * 60)
    print("Pinned Memory Test for iGPU")
    print("=" * 60)

    times = {}
    times['standard'] = test_standard_transfer()

    gc.collect()
    torch.cuda.empty_cache()

    times['pinned'] = test_pinned_memory()

    gc.collect()
    torch.cuda.empty_cache()

    times['shared'] = test_shared_memory()

    gc.collect()
    torch.cuda.empty_cache()

    times['direct'] = test_direct_gpu()

    print("\n" + "=" * 60)
    print("Summary:")
    for name, t in times.items():
        print(f"  {name:15s}: {t:.2f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
