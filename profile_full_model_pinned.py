#!/usr/bin/env python3
"""
Test pinned arena load with the FULL 65GB model.
Memory is now clean so we can do this properly.
"""

import gc
import time
import subprocess
from pathlib import Path
from typing import List

import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download


def nvidia_smi_free_gb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def get_ram_free_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024 / 1024  # KB to GB


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
    print("FULL MODEL PINNED ARENA TRANSFER TEST")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")
    print(f"RAM free: {get_ram_free_gb():.1f} GB")

    # Initialize CUDA
    _ = torch.randn(1000, device='cuda')
    cleanup()

    model_name = "openai/gpt-oss-120b"
    model_path = find_model_path(model_name)
    files = get_safetensor_files(model_path)
    total_bytes = get_total_size(files)
    total_gb = total_bytes / 1e9

    print(f"\nModel: {model_name}")
    print(f"Files: {len(files)}")
    print(f"Total size: {total_gb:.1f} GB")

    # Check we have enough RAM
    if get_ram_free_gb() < total_gb + 10:
        print(f"\nWARNING: May not have enough RAM for full model ({get_ram_free_gb():.1f}GB free, need {total_gb + 10:.0f}GB)")

    # TEST 1: Safetensors baseline (streaming)
    print("\n" + "-" * 80)
    print(f"TEST 1: Safetensors Direct GPU (streaming, {total_gb:.1f}GB)")
    print("-" * 80)

    print("  Loading all files via safetensors...")
    t0 = time.perf_counter()
    tensor_count = 0
    for i, f in enumerate(files):
        state_dict = load_file(str(f), device='cuda')
        tensor_count += len(state_dict)
        del state_dict
        torch.cuda.empty_cache()
        print(f"    File {i+1}/{len(files)}: {f.name} ({f.stat().st_size/1e9:.1f}GB)")
    torch.cuda.synchronize()
    safetensor_time = time.perf_counter() - t0
    safetensor_bw = total_gb / safetensor_time

    print(f"\n  Total time: {safetensor_time:.2f}s")
    print(f"  Bandwidth:  {safetensor_bw:.1f} GB/s")
    print(f"  Tensors:    {tensor_count}")

    cleanup()

    # TEST 2: File read → Pinned → GPU (full model)
    print("\n" + "-" * 80)
    print(f"TEST 2: File → Pinned → GPU (full model, {total_gb:.1f}GB)")
    print("-" * 80)

    print(f"  Allocating {total_gb:.1f}GB pinned arena...")
    t0 = time.perf_counter()
    pinned = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    alloc_time = time.perf_counter() - t0
    print(f"  Allocation time: {alloc_time:.2f}s")
    print(f"  RAM free after alloc: {get_ram_free_gb():.1f}GB")

    print("\n  Reading files to pinned arena...")
    t0 = time.perf_counter()
    offset = 0
    for i, f in enumerate(files):
        size = f.stat().st_size
        with open(f, 'rb') as fp:
            data = fp.read()
            # Copy data into pinned buffer
            pinned[offset:offset+size].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        offset += size
        print(f"    File {i+1}/{len(files)}: {f.name}")
        del data
        gc.collect()
    read_time = time.perf_counter() - t0
    read_bw = total_gb / read_time
    print(f"\n  Read time: {read_time:.2f}s ({read_bw:.1f} GB/s)")

    print("\n  Transferring pinned → GPU...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu = pinned.to('cuda', non_blocking=False)
    torch.cuda.synchronize()
    transfer_time = time.perf_counter() - t0
    transfer_bw = total_gb / transfer_time
    print(f"  Transfer time: {transfer_time:.2f}s ({transfer_bw:.1f} GB/s)")

    total_pinned_time = read_time + transfer_time

    del gpu, pinned
    cleanup()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<35} {'Time':>10} {'Bandwidth':>12}")
    print("-" * 60)
    print(f"{'Safetensors (streaming)':<35} {safetensor_time:>9.1f}s {safetensor_bw:>11.1f} GB/s")
    print(f"{'Pinned arena (total)':<35} {total_pinned_time:>9.1f}s {total_gb/total_pinned_time:>11.1f} GB/s")
    print(f"{'  - File read':<35} {read_time:>9.1f}s {read_bw:>11.1f} GB/s")
    print(f"{'  - GPU transfer':<35} {transfer_time:>9.1f}s {transfer_bw:>11.1f} GB/s")

    kv_init = 5.0  # Estimated vLLM overhead

    print(f"\n{'WITH vLLM INIT (~5s)':<35}")
    print(f"{'  Safetensors + init':<35} {safetensor_time + kv_init:>9.1f}s")
    print(f"{'  Pinned (no prefetch)':<35} {total_pinned_time + kv_init:>9.1f}s")
    print(f"{'  Pinned (with prefetch)':<35} {transfer_time + kv_init:>9.1f}s")

    speedup_no_prefetch = safetensor_time / total_pinned_time
    speedup_with_prefetch = (safetensor_time + kv_init) / (transfer_time + kv_init)

    print(f"\n{'SPEEDUP':<35}")
    print(f"{'  Without prefetch':<35} {speedup_no_prefetch:>9.2f}x")
    print(f"{'  With prefetch':<35} {speedup_with_prefetch:>9.2f}x")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"""
CONFIRMED RESULTS:
- Safetensors: {safetensor_time:.1f}s ({safetensor_bw:.1f} GB/s)
- Pure GPU transfer: {transfer_time:.1f}s ({transfer_bw:.1f} GB/s)

THE BOTTLENECK IS FILE I/O ({read_bw:.1f} GB/s from page cache).

With prefetch (file read during inference):
- Switch time: {transfer_time:.1f}s GPU transfer + {kv_init:.0f}s init = {transfer_time + kv_init:.1f}s
- Speedup: {speedup_with_prefetch:.1f}x over current {safetensor_time + kv_init:.0f}s

IMPLEMENTATION:
1. Pre-read weights to pinned arena during inference ({read_time:.0f}s, hidden)
2. On switch: single DMA transfer ({transfer_time:.1f}s)
3. vLLM initialization ({kv_init:.0f}s)
""")


if __name__ == '__main__':
    main()
