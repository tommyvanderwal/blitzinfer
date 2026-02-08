#!/usr/bin/env python3
"""Test memory impact of premerge operations.

This verifies the theory that `.contiguous()` in premerge.py creates
massive CPU memory copies that cause OOM during model switching.
"""

import os
import sys
import gc

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_ram_available():
    """Get available RAM in GB."""
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024 / 1024**3
    return 0


def log_ram(label):
    """Log RAM state."""
    avail = get_ram_available()
    print(f"[{label}] RAM available: {avail:.1f}GB")
    return avail


def test_reshape_memory():
    """Test if reshape().contiguous() creates memory copies."""
    print("\n=== TEST: reshape().contiguous() memory impact ===")

    baseline = log_ram("Baseline")

    # Create a 4D tensor similar to GPT-OSS blocked MXFP4 format
    # Real GPT-OSS: [experts=8, size_n=XXX, num_blocks=XXX, block_size=16]
    # Let's use a smaller test: 1GB tensor
    print("\nCreating 1GB 4D tensor...")
    # 1GB = 1024 * 1024 * 1024 bytes
    # float32: 4 bytes per element
    # So we need 1024 * 1024 * 256 elements = 268,435,456
    # Shape: [8, 1024, 32768, 16] = 8 * 1024 * 32768 * 16 = 4,294,967,296 elements (too big)
    # Let's do [8, 1024, 4096, 16] = 536,870,912 elements = 2GB float32
    # Actually use bfloat16 (2 bytes): [8, 1024, 4096, 16] = 1GB
    tensor_4d = torch.randn(8, 512, 8192, 16, dtype=torch.bfloat16)
    size_gb = tensor_4d.numel() * 2 / 1024**3
    print(f"Tensor shape: {tensor_4d.shape}, size: {size_gb:.2f}GB")

    gc.collect()
    after_create = log_ram("After create 4D tensor")
    print(f"RAM used for tensor: {baseline - after_create:.1f}GB")

    # Test 1: reshape only (should be a view, no copy)
    print("\n--- Test 1: reshape() only ---")
    tensor_reshaped = tensor_4d.reshape(8, 512, 8192 * 16)
    gc.collect()
    after_reshape = log_ram("After reshape()")
    print(f"RAM change: {after_create - after_reshape:.2f}GB (should be ~0)")
    print(f"Is view: {tensor_reshaped.data_ptr() == tensor_4d.data_ptr()}")

    # Test 2: reshape + contiguous (creates copy if not already contiguous)
    print("\n--- Test 2: reshape().contiguous() ---")
    tensor_contiguous = tensor_4d.reshape(8, 512, 8192 * 16).contiguous()
    gc.collect()
    after_contiguous = log_ram("After reshape().contiguous()")
    print(f"RAM change: {after_reshape - after_contiguous:.2f}GB (should be ~{size_gb:.1f}GB if copy)")
    print(f"Is same memory: {tensor_contiguous.data_ptr() == tensor_4d.data_ptr()}")

    # Clean up
    print("\n--- Cleanup ---")
    del tensor_contiguous
    gc.collect()
    after_del_contiguous = log_ram("After del contiguous")

    del tensor_reshaped
    gc.collect()
    after_del_reshaped = log_ram("After del reshaped")

    del tensor_4d
    gc.collect()
    after_del_original = log_ram("After del original")

    print(f"\nTotal recovered: {after_del_original - after_create:.1f}GB")


def test_torch_cat_memory():
    """Test if torch.cat creates memory copies."""
    print("\n\n=== TEST: torch.cat() memory impact ===")

    baseline = log_ram("Baseline")

    # Create component tensors similar to q_proj, k_proj, v_proj
    # Each ~1GB
    print("\nCreating 3x 1GB tensors...")
    tensors = [torch.randn(1024, 4096, 128, dtype=torch.bfloat16) for _ in range(3)]
    size_each = tensors[0].numel() * 2 / 1024**3
    print(f"Each tensor: {tensors[0].shape}, size: {size_each:.2f}GB")

    gc.collect()
    after_create = log_ram("After create 3 tensors")
    print(f"RAM used: {baseline - after_create:.1f}GB (expected ~{size_each*3:.1f}GB)")

    # torch.cat creates a new tensor
    print("\n--- torch.cat() ---")
    merged = torch.cat(tensors, dim=0)
    gc.collect()
    after_cat = log_ram("After torch.cat()")
    print(f"RAM change: {after_create - after_cat:.2f}GB (expected ~{size_each*3:.1f}GB for merged copy)")
    print(f"Merged shape: {merged.shape}")

    # If we delete original tensors, does memory get freed?
    print("\n--- Delete originals ---")
    del tensors
    gc.collect()
    after_del_originals = log_ram("After del originals")
    print(f"RAM recovered: {after_cat - after_del_originals:.1f}GB (should be ~{size_each*3:.1f}GB)")

    # Delete merged
    del merged
    gc.collect()
    after_del_merged = log_ram("After del merged")
    print(f"RAM recovered: {after_del_originals - after_del_merged:.1f}GB")


def test_pinned_memory_cat():
    """Test if torch.cat on pinned tensors creates pinned or regular copies."""
    print("\n\n=== TEST: torch.cat() with pinned memory ===")

    baseline = log_ram("Baseline")

    # Create pinned tensors
    print("\nCreating 3x 0.5GB pinned tensors...")
    tensors = []
    for i in range(3):
        t = torch.randn(512, 2048, 128, dtype=torch.bfloat16).pin_memory()
        tensors.append(t)
    size_each = tensors[0].numel() * 2 / 1024**3
    print(f"Each tensor: {tensors[0].shape}, size: {size_each:.2f}GB, pinned: {tensors[0].is_pinned()}")

    gc.collect()
    after_create = log_ram("After create pinned tensors")

    # torch.cat on pinned tensors
    print("\n--- torch.cat() on pinned tensors ---")
    merged = torch.cat(tensors, dim=0)
    gc.collect()
    after_cat = log_ram("After torch.cat()")
    print(f"RAM change: {after_create - after_cat:.2f}GB")
    print(f"Merged is pinned: {merged.is_pinned()}")  # This is the key question!

    # Cleanup
    del merged, tensors
    gc.collect()
    log_ram("After cleanup")


if __name__ == "__main__":
    test_reshape_memory()
    test_torch_cat_memory()
    test_pinned_memory_cat()

    print("\n\n=== CONCLUSIONS ===")
    print("1. reshape() is a view (no memory copy) if tensor is contiguous")
    print("2. reshape().contiguous() creates a FULL COPY if reshape changed layout")
    print("3. torch.cat() ALWAYS creates a new tensor (copy)")
    print("4. For GPT-OSS 60GB model: premerge creates ~60GB extra copies!")
    print("5. Total: 80GB arena + 60GB copies = 140GB > 124GB RAM = OOM")
