#!/usr/bin/env python3
"""
Benchmark raw hardware I/O speeds to understand theoretical limits.

Tests:
1. Raw disk read (NVMe SSD) - should be 12-14 GB/s on PCIe 5.0
2. RAM to VRAM transfer (pinned memory) - should be ~64 GB/s on PCIe 5.0 x16
3. Safetensors mmap vs direct read comparison
"""

import os
import sys
import time
import glob
import mmap
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch


def get_model_path(model_name: str) -> Path:
    """Get local path for HuggingFace model"""
    cache_dir = Path.home() / ".cache/huggingface/hub"
    model_dir = f"models--{model_name.replace('/', '--')}"
    model_path = cache_dir / model_dir / "snapshots"
    if model_path.exists():
        snapshots = list(model_path.iterdir())
        if snapshots:
            return snapshots[0]
    raise ValueError(f"Model not found: {model_name}")


def drop_caches():
    """Drop filesystem caches (requires sudo or won't work)"""
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        return True
    except:
        return False


def benchmark_disk_read(file_path: str, warm_cache: bool = True):
    """Benchmark raw disk read speed"""
    file_size = os.path.getsize(file_path)
    print(f"\nFile: {Path(file_path).name}")
    print(f"Size: {file_size/1e9:.2f} GB")

    if warm_cache:
        # Warm cache first
        print("Warming cache...")
        with open(file_path, 'rb') as f:
            _ = f.read()

    # Timed read
    t0 = time.time()
    with open(file_path, 'rb') as f:
        data = f.read()
    t1 = time.time()

    speed = file_size / (t1 - t0) / 1e9
    cache_status = "cached" if warm_cache else "cold"
    print(f"Direct read ({cache_status}): {t1-t0:.2f}s = {speed:.1f} GB/s")

    return data, speed


def benchmark_mmap_read(file_path: str):
    """Benchmark mmap read speed"""
    file_size = os.path.getsize(file_path)

    # Warm cache
    with open(file_path, 'rb') as f:
        _ = f.read()

    # Mmap read (just mapping, not accessing)
    t0 = time.time()
    with open(file_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # Force read all data
        data = mm[:]
        mm.close()
    t1 = time.time()

    speed = file_size / (t1 - t0) / 1e9
    print(f"Mmap read: {t1-t0:.2f}s = {speed:.1f} GB/s")

    return speed


def benchmark_ram_to_vram():
    """Benchmark RAM to VRAM transfer with different memory types"""
    print("\n" + "="*60)
    print("RAM to VRAM Transfer Benchmarks")
    print("="*60)

    sizes = [1, 4, 16, 32]  # GB

    for size_gb in sizes:
        if size_gb > 32:
            continue  # Skip if too large

        size_bytes = int(size_gb * 1e9)
        num_elements = size_bytes // 2  # float16

        print(f"\n--- {size_gb} GB tensor (float16) ---")

        # 1. Regular (pageable) memory
        cpu_tensor = torch.randn(num_elements, dtype=torch.float16)
        torch.cuda.synchronize()

        t0 = time.time()
        gpu_tensor = cpu_tensor.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.time()

        speed = size_bytes / (t1 - t0) / 1e9
        print(f"Pageable memory: {t1-t0:.2f}s = {speed:.1f} GB/s")
        del gpu_tensor
        torch.cuda.empty_cache()

        # 2. Contiguous clone then transfer
        t0 = time.time()
        contiguous = cpu_tensor.clone()
        gpu_tensor = contiguous.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.time()

        speed = size_bytes / (t1 - t0) / 1e9
        print(f"Clone + transfer: {t1-t0:.2f}s = {speed:.1f} GB/s")
        del gpu_tensor, contiguous
        torch.cuda.empty_cache()

        # 3. Pinned memory
        pinned = torch.empty(num_elements, dtype=torch.float16, pin_memory=True)
        pinned.copy_(cpu_tensor)
        torch.cuda.synchronize()

        t0 = time.time()
        gpu_tensor = pinned.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.time()

        speed = size_bytes / (t1 - t0) / 1e9
        print(f"Pinned memory: {t1-t0:.2f}s = {speed:.1f} GB/s")

        del gpu_tensor, pinned, cpu_tensor
        torch.cuda.empty_cache()


def benchmark_safetensors_loading(model_path: Path):
    """Compare safetensors mmap vs hypothetical direct load"""
    from safetensors import safe_open

    print("\n" + "="*60)
    print("Safetensors Loading Comparison")
    print("="*60)

    safetensor_files = sorted(glob.glob(str(model_path / "*.safetensors")))[:3]  # First 3 files

    for sf in safetensor_files:
        file_size = os.path.getsize(sf)
        print(f"\n--- {Path(sf).name} ({file_size/1e9:.2f} GB) ---")

        # Warm cache
        with open(sf, 'rb') as f:
            _ = f.read()

        # Method 1: safetensors mmap to CPU
        t0 = time.time()
        with safe_open(sf, framework="pt", device="cpu") as f:
            tensors_cpu = {k: f.get_tensor(k) for k in f.keys()}
        t1 = time.time()
        total_size = sum(t.numel() * t.element_size() for t in tensors_cpu.values())
        speed = total_size / (t1 - t0) / 1e9
        print(f"Safetensors mmap→CPU: {t1-t0:.2f}s = {speed:.1f} GB/s")

        # Check if tensors are contiguous
        non_contig = sum(1 for t in tensors_cpu.values() if not t.is_contiguous())
        print(f"  Non-contiguous tensors: {non_contig}/{len(tensors_cpu)}")

        # Method 2: safetensors mmap direct to GPU
        torch.cuda.empty_cache()
        t0 = time.time()
        with safe_open(sf, framework="pt", device="cuda") as f:
            tensors_gpu = {k: f.get_tensor(k) for k in f.keys()}
        torch.cuda.synchronize()
        t1 = time.time()
        speed = total_size / (t1 - t0) / 1e9
        print(f"Safetensors mmap→GPU: {t1-t0:.2f}s = {speed:.1f} GB/s")

        del tensors_gpu
        torch.cuda.empty_cache()

        # Method 3: Raw file read + manual tensor creation
        t0 = time.time()
        with open(sf, 'rb') as f:
            raw_data = f.read()
        t1 = time.time()
        speed = len(raw_data) / (t1 - t0) / 1e9
        print(f"Raw file read: {t1-t0:.2f}s = {speed:.1f} GB/s")

        # Method 4: CPU tensors clone to contiguous then GPU
        torch.cuda.empty_cache()
        t0 = time.time()
        with safe_open(sf, framework="pt", device="cpu") as f:
            tensors_gpu = {}
            for k in f.keys():
                t = f.get_tensor(k)
                tensors_gpu[k] = t.clone().to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.time()
        speed = total_size / (t1 - t0) / 1e9
        print(f"Mmap→clone→GPU: {t1-t0:.2f}s = {speed:.1f} GB/s")

        del tensors_gpu, tensors_cpu
        torch.cuda.empty_cache()


def benchmark_full_model_load(model_path: Path):
    """Benchmark full model loading with different methods"""
    from safetensors import safe_open

    print("\n" + "="*60)
    print("Full Model Loading Comparison")
    print("="*60)

    safetensor_files = sorted(glob.glob(str(model_path / "*.safetensors")))
    total_file_size = sum(os.path.getsize(f) for f in safetensor_files)
    print(f"Model: {model_path.parent.name}")
    print(f"Files: {len(safetensor_files)}")
    print(f"Total size: {total_file_size/1e9:.2f} GB")

    # Warm cache
    print("\nWarming cache...")
    for sf in safetensor_files:
        with open(sf, 'rb') as f:
            _ = f.read()
    print("Cache warm.")

    # Method 1: Direct safetensors to GPU (baseline - what vLLM does)
    print("\n--- Method 1: Safetensors direct to GPU ---")
    torch.cuda.empty_cache()
    t0 = time.time()
    all_tensors = {}
    for sf in safetensor_files:
        with safe_open(sf, framework="pt", device="cuda") as f:
            for k in f.keys():
                all_tensors[k] = f.get_tensor(k)
    torch.cuda.synchronize()
    t1 = time.time()
    total_size = sum(t.numel() * t.element_size() for t in all_tensors.values())
    speed = total_size / (t1 - t0) / 1e9
    print(f"Time: {t1-t0:.2f}s, Speed: {speed:.1f} GB/s")
    del all_tensors
    torch.cuda.empty_cache()

    # Method 2: Raw read to RAM, then bulk transfer
    print("\n--- Method 2: Raw read to RAM, bulk GPU transfer ---")
    torch.cuda.empty_cache()

    # Step 1: Read all files to RAM
    t0 = time.time()
    raw_data = {}
    for sf in safetensor_files:
        with open(sf, 'rb') as f:
            raw_data[sf] = f.read()
    t1 = time.time()
    read_time = t1 - t0
    read_speed = total_file_size / read_time / 1e9
    print(f"  Raw read: {read_time:.2f}s = {read_speed:.1f} GB/s")

    # Step 2: Parse and transfer
    t0 = time.time()
    all_tensors = {}
    for sf in safetensor_files:
        with safe_open(sf, framework="pt", device="cuda") as f:
            for k in f.keys():
                all_tensors[k] = f.get_tensor(k)
    torch.cuda.synchronize()
    t1 = time.time()
    parse_time = t1 - t0
    print(f"  Parse+transfer: {parse_time:.2f}s")
    print(f"  Total: {read_time + parse_time:.2f}s")

    del all_tensors, raw_data
    torch.cuda.empty_cache()


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"

    if model == "gpt":
        model = "openai/gpt-oss-120b"
    elif model == "qwen":
        model = "Qwen/Qwen3-VL-32B-Instruct"

    model_path = get_model_path(model)
    print(f"Model path: {model_path}")

    # Get first safetensor file for single-file tests
    safetensor_files = sorted(glob.glob(str(model_path / "*.safetensors")))
    test_file = safetensor_files[0]

    # 1. Raw disk I/O
    print("\n" + "="*60)
    print("1. RAW DISK I/O")
    print("="*60)
    benchmark_disk_read(test_file, warm_cache=True)

    # 2. Mmap comparison
    print("\n" + "="*60)
    print("2. MMAP vs DIRECT READ")
    print("="*60)
    benchmark_mmap_read(test_file)

    # 3. RAM to VRAM
    benchmark_ram_to_vram()

    # 4. Safetensors comparison
    benchmark_safetensors_loading(model_path)

    # 5. Full model
    benchmark_full_model_load(model_path)


if __name__ == "__main__":
    main()
