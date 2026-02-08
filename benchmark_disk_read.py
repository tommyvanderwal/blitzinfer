#!/usr/bin/env python3
"""Benchmark raw disk read speed for model files.

This tests I/O speed without any vLLM interference to establish baseline.
"""
import os
import sys
import time
import glob
from pathlib import Path
from huggingface_hub import snapshot_download


def get_model_path(model_name: str) -> str:
    """Get local path for HuggingFace model."""
    return snapshot_download(model_name, local_files_only=True)


def benchmark_sequential_read(model_path: str, chunk_size: int = 64 * 1024 * 1024):
    """Benchmark sequential read speed."""
    safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))

    if not safetensor_files:
        print(f"No safetensor files found in {model_path}")
        return 0, 0

    total_size = sum(os.path.getsize(f) for f in safetensor_files)
    print(f"Total model size: {total_size / 1024**3:.2f} GB ({len(safetensor_files)} files)")

    # Drop caches first
    print("Dropping caches...")
    os.system('sync')
    os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1')
    time.sleep(1)

    # Read all files
    print(f"Reading with {chunk_size // 1024 // 1024}MB chunks...")
    bytes_read = 0
    t0 = time.time()

    for sf_file in safetensor_files:
        file_size = os.path.getsize(sf_file)
        with open(sf_file, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                bytes_read += len(chunk)

        elapsed = time.time() - t0
        speed = bytes_read / elapsed / 1024**3
        print(f"  {Path(sf_file).name}: {file_size/1024**3:.2f} GB | Running: {speed:.2f} GB/s")

    elapsed = time.time() - t0
    speed = bytes_read / elapsed / 1024**3

    return bytes_read, speed


def benchmark_mmap_read(model_path: str):
    """Benchmark memory-mapped read speed."""
    import mmap

    safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)

    # Drop caches first
    print("\nDropping caches for mmap test...")
    os.system('sync')
    os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1')
    time.sleep(1)

    print("Reading with mmap + sequential access...")
    bytes_read = 0
    t0 = time.time()

    for sf_file in safetensor_files:
        file_size = os.path.getsize(sf_file)
        with open(sf_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # Touch every 4KB page to force read
            for i in range(0, len(mm), 4096):
                _ = mm[i]
            bytes_read += file_size
            mm.close()

        elapsed = time.time() - t0
        speed = bytes_read / elapsed / 1024**3
        print(f"  {Path(sf_file).name}: {file_size/1024**3:.2f} GB | Running: {speed:.2f} GB/s")

    elapsed = time.time() - t0
    speed = bytes_read / elapsed / 1024**3

    return bytes_read, speed


def benchmark_posix_fadvise(model_path: str, chunk_size: int = 64 * 1024 * 1024):
    """Benchmark read speed with POSIX_FADV_WILLNEED."""
    import ctypes

    libc = ctypes.CDLL("libc.so.6")
    POSIX_FADV_WILLNEED = 3

    safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)

    # Drop caches first
    print("\nDropping caches for fadvise test...")
    os.system('sync')
    os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1')
    time.sleep(1)

    print("Reading with posix_fadvise(WILLNEED) + read...")
    bytes_read = 0
    t0 = time.time()

    for sf_file in safetensor_files:
        file_size = os.path.getsize(sf_file)
        with open(sf_file, 'rb') as f:
            fd = f.fileno()
            # Advise kernel to read ahead
            libc.posix_fadvise(fd, 0, file_size, POSIX_FADV_WILLNEED)
            # Now read
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                bytes_read += len(chunk)

        elapsed = time.time() - t0
        speed = bytes_read / elapsed / 1024**3
        print(f"  {Path(sf_file).name}: {file_size/1024**3:.2f} GB | Running: {speed:.2f} GB/s")

    elapsed = time.time() - t0
    speed = bytes_read / elapsed / 1024**3

    return bytes_read, speed


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-32B-Instruct"

    print("=" * 70)
    print("DISK READ BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")

    model_path = get_model_path(model_name)
    print(f"Path: {model_path}")

    # Test 1: Sequential read
    print("\n" + "-" * 50)
    print("TEST 1: Sequential read (64MB chunks)")
    print("-" * 50)
    bytes1, speed1 = benchmark_sequential_read(model_path)
    print(f"\nResult: {speed1:.2f} GB/s")

    # Test 2: Memory-mapped read
    print("\n" + "-" * 50)
    print("TEST 2: Memory-mapped read")
    print("-" * 50)
    bytes2, speed2 = benchmark_mmap_read(model_path)
    print(f"\nResult: {speed2:.2f} GB/s")

    # Test 3: posix_fadvise
    print("\n" + "-" * 50)
    print("TEST 3: posix_fadvise(WILLNEED) + read")
    print("-" * 50)
    bytes3, speed3 = benchmark_posix_fadvise(model_path)
    print(f"\nResult: {speed3:.2f} GB/s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Sequential read:  {speed1:.2f} GB/s")
    print(f"Memory-mapped:    {speed2:.2f} GB/s")
    print(f"With fadvise:     {speed3:.2f} GB/s")
    print("=" * 70)


if __name__ == '__main__':
    main()
