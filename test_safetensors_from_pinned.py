#!/usr/bin/env python3
"""
Test if safetensors can load directly from pinned memory bytes.

If this works, we can:
1. Pre-read safetensor files into pinned memory (background)
2. On switch, load directly from pinned bytes to GPU
3. Get close to 48 GB/s transfer speed
"""

import gc
import time
import subprocess
from pathlib import Path
from typing import List, Dict
from io import BytesIO

import torch
from safetensors import safe_open
from safetensors.torch import load_file, load
from huggingface_hub import snapshot_download


def nvidia_smi_free_gb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def find_model_path(model_name: str) -> Path:
    return Path(snapshot_download(model_name, local_files_only=True))


def get_safetensor_files(model_path: Path) -> List[Path]:
    return sorted(model_path.glob("*.safetensors"))


def get_total_size(files: List[Path]) -> int:
    return sum(f.stat().st_size for f in files)


def main():
    print("=" * 80)
    print("SAFETENSORS FROM PINNED MEMORY TEST")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    # Initialize CUDA
    _ = torch.randn(1000, device='cuda')
    cleanup()

    model_name = "openai/gpt-oss-120b"
    model_path = find_model_path(model_name)
    files = get_safetensor_files(model_path)

    # Use first 2 files for testing (~10GB)
    test_files = files[:2]
    total_bytes = get_total_size(test_files)
    total_gb = total_bytes / 1e9

    print(f"\nModel: {model_name}")
    print(f"Test files: {len(test_files)} ({total_gb:.1f} GB)")

    # TEST 1: Baseline - safetensors from disk
    print("\n" + "-" * 80)
    print("TEST 1: Safetensors from disk (baseline)")
    print("-" * 80)

    cleanup()
    t0 = time.perf_counter()
    for f in test_files:
        state_dict = load_file(str(f), device='cuda')
        del state_dict
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    disk_time = time.perf_counter() - t0
    disk_bw = total_gb / disk_time
    print(f"  Time: {disk_time:.2f}s ({disk_bw:.1f} GB/s)")

    # TEST 2: Pre-read to regular memory, then load
    print("\n" + "-" * 80)
    print("TEST 2: Pre-read to regular memory, then safetensors.load()")
    print("-" * 80)

    cleanup()

    # Pre-read files to memory
    print("  Reading files to memory...")
    t0 = time.perf_counter()
    file_bytes = []
    for f in test_files:
        with open(f, 'rb') as fp:
            file_bytes.append(fp.read())
    read_time = time.perf_counter() - t0
    print(f"  Read time: {read_time:.2f}s ({total_gb / read_time:.1f} GB/s)")

    # Load from bytes to GPU
    print("  Loading from bytes to GPU...")
    t0 = time.perf_counter()
    for data in file_bytes:
        state_dict = load(data)  # Load to CPU first
        # Then transfer to GPU
        for k, v in state_dict.items():
            state_dict[k] = v.to('cuda')
        del state_dict
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    load_time = time.perf_counter() - t0
    print(f"  Load+transfer time: {load_time:.2f}s ({total_gb / load_time:.1f} GB/s)")

    del file_bytes
    cleanup()

    # TEST 3: Pre-read to PINNED memory, then load
    print("\n" + "-" * 80)
    print("TEST 3: Pre-read to PINNED memory, then transfer")
    print("-" * 80)

    # Allocate pinned buffer
    print(f"  Allocating {total_gb:.1f}GB pinned buffer...")
    pinned_buffer = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)

    # Pre-read files to pinned memory
    print("  Reading files to pinned memory...")
    t0 = time.perf_counter()
    offset = 0
    file_offsets = []
    for f in test_files:
        size = f.stat().st_size
        with open(f, 'rb') as fp:
            data = fp.read()
            pinned_buffer[offset:offset+size].copy_(
                torch.frombuffer(bytearray(data), dtype=torch.uint8)
            )
        file_offsets.append((offset, size))
        offset += size
        del data
    read_time = time.perf_counter() - t0
    print(f"  Read time: {read_time:.2f}s ({total_gb / read_time:.1f} GB/s)")

    # Now try loading from pinned bytes
    print("  Loading safetensors from pinned bytes...")
    t0 = time.perf_counter()
    for start, size in file_offsets:
        # Get bytes view from pinned tensor
        bytes_data = pinned_buffer[start:start+size].numpy().tobytes()
        state_dict = load(bytes_data)
        # Transfer to GPU
        for k, v in state_dict.items():
            state_dict[k] = v.to('cuda')
        del state_dict
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    load_time = time.perf_counter() - t0
    print(f"  Load+transfer time: {load_time:.2f}s ({total_gb / load_time:.1f} GB/s)")

    del pinned_buffer
    cleanup()

    # TEST 4: Direct pinned tensor transfer (maximum speed reference)
    print("\n" + "-" * 80)
    print("TEST 4: Raw pinned tensor → GPU (maximum speed reference)")
    print("-" * 80)

    cleanup()
    pinned = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu = pinned.to('cuda', non_blocking=False)
    torch.cuda.synchronize()
    raw_time = time.perf_counter() - t0
    raw_bw = total_gb / raw_time
    print(f"  Time: {raw_time*1000:.0f}ms ({raw_bw:.1f} GB/s)")

    del gpu, pinned
    cleanup()

    # TEST 5: Load safetensors to CPU tensors, pin_memory(), then transfer
    print("\n" + "-" * 80)
    print("TEST 5: Safetensors→CPU→pin_memory()→GPU")
    print("-" * 80)

    cleanup()

    # Read files to memory first
    file_bytes = []
    for f in test_files:
        with open(f, 'rb') as fp:
            file_bytes.append(fp.read())

    # Load to CPU, pin, then transfer
    print("  Loading to CPU, pinning, transferring...")
    t0 = time.perf_counter()
    for data in file_bytes:
        state_dict = load(data)
        for k, v in state_dict.items():
            pinned = v.pin_memory()
            gpu = pinned.to('cuda', non_blocking=False)
            del pinned, gpu
        del state_dict
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    pinned_load_time = time.perf_counter() - t0
    print(f"  Time: {pinned_load_time:.2f}s ({total_gb / pinned_load_time:.1f} GB/s)")

    del file_bytes
    cleanup()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<45} {'Time':>10} {'Bandwidth':>12}")
    print("-" * 70)
    print(f"{'1. Safetensors from disk (baseline)':<45} {disk_time:>9.2f}s {disk_bw:>11.1f} GB/s")
    print(f"{'4. Raw pinned → GPU (max reference)':<45} {raw_time*1000:>8.0f}ms {raw_bw:>11.1f} GB/s")
    print(f"{'5. Safetensors→CPU→pin→GPU':<45} {pinned_load_time:>9.2f}s {total_gb/pinned_load_time:>11.1f} GB/s")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print(f"""
Raw pinned → GPU: {raw_bw:.0f} GB/s (hardware limit)
Safetensors from disk: {disk_bw:.1f} GB/s

The gap is due to:
1. Per-tensor overhead in safetensors parsing
2. Non-pinned memory transfers
3. CUDA kernel launch overhead per tensor

To get closer to hardware limit, we need to:
1. Pre-load all weights into a single contiguous pinned buffer
2. Transfer the entire buffer to GPU in one DMA operation
3. Then slice the GPU buffer into individual tensors

This avoids per-tensor overhead during the switch.
""")


if __name__ == '__main__':
    main()
