#!/usr/bin/env python3
"""
Test pinned memory transfer speed vs safetensors.
Uses 10GB subset to fit in available memory.
"""

import gc
import os
import time
import subprocess
from pathlib import Path
from typing import List, Dict

import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download


def nvidia_smi_free_gb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def find_model_path(model_name: str) -> Path:
    return Path(snapshot_download(model_name, local_files_only=True))


def get_safetensor_files(model_path: Path) -> List[Path]:
    return sorted(model_path.glob("*.safetensors"))


def get_total_size(files: List[Path]) -> int:
    return sum(f.stat().st_size for f in files)


def cleanup():
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    print("=" * 80)
    print("PINNED vs SAFETENSORS TRANSFER TEST")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    # Initialize CUDA
    _ = torch.randn(1000, device='cuda')
    cleanup()

    model_name = "openai/gpt-oss-120b"
    model_path = find_model_path(model_name)
    files = get_safetensor_files(model_path)
    total_bytes = get_total_size(files)
    full_model_gb = total_bytes / 1e9

    print(f"\nModel: {model_name}")
    print(f"Full size: {full_model_gb:.1f} GB")

    # Use only first 2 files (~10GB)
    test_files = files[:2]
    test_bytes = get_total_size(test_files)
    test_gb = test_bytes / 1e9
    print(f"Test subset: {test_gb:.1f} GB ({len(test_files)} files)")

    # TEST 1: Pure pinned → GPU transfer
    print("\n" + "-" * 80)
    print("TEST 1: Pure Pinned → GPU (10GB synthetic)")
    print("-" * 80)

    cleanup()
    size_bytes = 10 * 1024**3  # 10GB

    print("  Allocating 10GB pinned...")
    pinned = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)

    # Warmup
    gpu = pinned[:1024*1024].to('cuda')
    torch.cuda.synchronize()
    del gpu
    torch.cuda.empty_cache()

    print("  Transferring 10GB pinned → GPU...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu = pinned.to('cuda', non_blocking=False)
    torch.cuda.synchronize()
    pinned_time = time.perf_counter() - t0
    pinned_bw = 10.0 / pinned_time

    print(f"  Time: {pinned_time*1000:.0f}ms, Bandwidth: {pinned_bw:.1f} GB/s")

    del gpu, pinned
    cleanup()

    # TEST 2: Safetensors direct GPU load
    print("\n" + "-" * 80)
    print(f"TEST 2: Safetensors Direct GPU ({test_gb:.1f}GB)")
    print("-" * 80)

    print(f"  Loading {len(test_files)} files via safetensors...")
    t0 = time.perf_counter()
    for f in test_files:
        state_dict = load_file(str(f), device='cuda')
        del state_dict
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    safetensor_time = time.perf_counter() - t0
    safetensor_bw = test_gb / safetensor_time

    print(f"  Time: {safetensor_time*1000:.0f}ms, Bandwidth: {safetensor_bw:.1f} GB/s")

    cleanup()

    # TEST 3: Read files to pinned, then transfer
    print("\n" + "-" * 80)
    print(f"TEST 3: File → Pinned → GPU ({test_gb:.1f}GB)")
    print("-" * 80)

    print(f"  Allocating {test_gb:.1f}GB pinned...")
    pinned = torch.empty(test_bytes, dtype=torch.uint8, pin_memory=True)

    print("  Reading files to pinned...")
    t0 = time.perf_counter()
    offset = 0
    for f in test_files:
        size = f.stat().st_size
        with open(f, 'rb') as fp:
            data = fp.read()
            pinned[offset:offset+size].copy_(torch.frombuffer(data, dtype=torch.uint8))
        offset += size
        del data
        gc.collect()
    read_time = time.perf_counter() - t0
    read_bw = test_gb / read_time
    print(f"  Read: {read_time*1000:.0f}ms ({read_bw:.1f} GB/s)")

    print("  Transferring to GPU...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu = pinned.to('cuda', non_blocking=False)
    torch.cuda.synchronize()
    transfer_time = time.perf_counter() - t0
    transfer_bw = test_gb / transfer_time
    print(f"  Transfer: {transfer_time*1000:.0f}ms ({transfer_bw:.1f} GB/s)")

    total_time = read_time + transfer_time
    print(f"  Total: {total_time*1000:.0f}ms")

    del gpu, pinned
    cleanup()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<30} {'Time':>10} {'Bandwidth':>12}")
    print("-" * 55)
    print(f"{'Pure pinned (10GB)':<30} {pinned_time*1000:>9.0f}ms {pinned_bw:>11.1f} GB/s")
    print(f"{'Safetensors':<30} {safetensor_time*1000:>9.0f}ms {safetensor_bw:>11.1f} GB/s")
    print(f"{'File→Pinned→GPU':<30} {total_time*1000:>9.0f}ms {test_gb/total_time:>11.1f} GB/s")
    print(f"{'  (Read only)':<30} {read_time*1000:>9.0f}ms {read_bw:>11.1f} GB/s")
    print(f"{'  (Transfer only)':<30} {transfer_time*1000:>9.0f}ms {transfer_bw:>11.1f} GB/s")

    # Extrapolate to full 65GB model
    print("\n" + "-" * 80)
    print(f"EXTRAPOLATION TO {full_model_gb:.0f}GB MODEL")
    print("-" * 80)

    est_safetensor = full_model_gb / safetensor_bw
    est_read = full_model_gb / read_bw
    est_transfer = full_model_gb / transfer_bw

    print(f"\n  Safetensors:     ~{est_safetensor:.1f}s")
    print(f"  Pinned approach: ~{est_read + est_transfer:.1f}s (read: {est_read:.1f}s + transfer: {est_transfer:.1f}s)")
    print(f"  With prefetch:   ~{est_transfer:.1f}s (read overlapped with inference)")

    kv_init = 5.0
    print(f"\n  + vLLM init: ~{kv_init:.0f}s")
    print(f"\n  ESTIMATED SWITCH TIME:")
    print(f"    Current (safetensors): ~{est_safetensor + kv_init:.0f}s")
    print(f"    With prefetch:         ~{est_transfer + kv_init:.0f}s")
    print(f"    Speedup:               {(est_safetensor + kv_init) / (est_transfer + kv_init):.1f}x")


if __name__ == '__main__':
    main()
