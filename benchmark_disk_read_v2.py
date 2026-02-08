#!/usr/bin/env python3
"""Benchmark disk read speed - optimized methods.

Testing different approaches to maximize read throughput.
"""
import os
import sys
import time
import glob
import mmap
import threading
import concurrent.futures
from pathlib import Path
from huggingface_hub import snapshot_download


def get_model_path(model_name: str) -> str:
    """Get local path for HuggingFace model."""
    return snapshot_download(model_name, local_files_only=True)


def drop_caches():
    """Drop OS page caches."""
    os.system('sync')
    os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1')
    time.sleep(1)


def benchmark_parallel_files(model_path: str, num_threads: int = 4):
    """Read files in parallel using thread pool."""
    safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)

    print(f"Total: {total_size / 1024**3:.2f} GB, {len(safetensor_files)} files, {num_threads} threads")

    drop_caches()

    def read_file(path):
        """Read a single file with mmap."""
        with open(path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # Touch every 4KB page
            for i in range(0, len(mm), 4096):
                _ = mm[i]
            size = len(mm)
            mm.close()
            return size

    t0 = time.time()
    bytes_read = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(read_file, f): f for f in safetensor_files}
        for future in concurrent.futures.as_completed(futures):
            bytes_read += future.result()

    elapsed = time.time() - t0
    speed = bytes_read / elapsed / 1024**3
    return bytes_read, speed


def benchmark_large_buffer_read(model_path: str, buffer_size_mb: int = 256):
    """Sequential read with very large buffer."""
    safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)
    buffer_size = buffer_size_mb * 1024 * 1024

    print(f"Total: {total_size / 1024**3:.2f} GB, buffer: {buffer_size_mb}MB")

    drop_caches()

    t0 = time.time()
    bytes_read = 0

    for sf_file in safetensor_files:
        with open(sf_file, 'rb', buffering=buffer_size) as f:
            while True:
                chunk = f.read(buffer_size)
                if not chunk:
                    break
                bytes_read += len(chunk)

    elapsed = time.time() - t0
    speed = bytes_read / elapsed / 1024**3
    return bytes_read, speed


def benchmark_readinto_buffer(model_path: str, buffer_size_mb: int = 256):
    """Use readinto() with pre-allocated buffer."""
    safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)
    buffer_size = buffer_size_mb * 1024 * 1024

    print(f"Total: {total_size / 1024**3:.2f} GB, buffer: {buffer_size_mb}MB")

    drop_caches()

    # Pre-allocate buffer
    buffer = bytearray(buffer_size)

    t0 = time.time()
    bytes_read = 0

    for sf_file in safetensor_files:
        with open(sf_file, 'rb') as f:
            while True:
                n = f.readinto(buffer)
                if n == 0:
                    break
                bytes_read += n

    elapsed = time.time() - t0
    speed = bytes_read / elapsed / 1024**3
    return bytes_read, speed


def benchmark_os_read(model_path: str, buffer_size_mb: int = 256):
    """Use os.read() for lower-level I/O."""
    safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)
    buffer_size = buffer_size_mb * 1024 * 1024

    print(f"Total: {total_size / 1024**3:.2f} GB, buffer: {buffer_size_mb}MB")

    drop_caches()

    t0 = time.time()
    bytes_read = 0

    for sf_file in safetensor_files:
        fd = os.open(sf_file, os.O_RDONLY)
        try:
            while True:
                data = os.read(fd, buffer_size)
                if not data:
                    break
                bytes_read += len(data)
        finally:
            os.close(fd)

    elapsed = time.time() - t0
    speed = bytes_read / elapsed / 1024**3
    return bytes_read, speed


def benchmark_dd_read(model_path: str):
    """Use dd command as baseline."""
    import subprocess

    safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)

    print(f"Total: {total_size / 1024**3:.2f} GB")

    drop_caches()

    t0 = time.time()

    # Read all files with dd
    for sf_file in safetensor_files:
        subprocess.run(
            ['dd', f'if={sf_file}', 'of=/dev/null', 'bs=1M', 'status=none'],
            check=True
        )

    elapsed = time.time() - t0
    speed = total_size / elapsed / 1024**3
    return total_size, speed


def benchmark_cat_read(model_path: str):
    """Use cat command to /dev/null."""
    import subprocess

    safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)

    print(f"Total: {total_size / 1024**3:.2f} GB")

    drop_caches()

    t0 = time.time()

    # Read all files with cat
    subprocess.run(
        ['cat'] + safetensor_files,
        stdout=subprocess.DEVNULL,
        check=True
    )

    elapsed = time.time() - t0
    speed = total_size / elapsed / 1024**3
    return total_size, speed


def benchmark_parallel_dd(model_path: str, num_jobs: int = 4):
    """Use parallel dd commands."""
    import subprocess

    safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    total_size = sum(os.path.getsize(f) for f in safetensor_files)

    print(f"Total: {total_size / 1024**3:.2f} GB, {num_jobs} parallel jobs")

    drop_caches()

    t0 = time.time()

    def dd_file(path):
        subprocess.run(['dd', f'if={path}', 'of=/dev/null', 'bs=1M', 'status=none'], check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as executor:
        executor.map(dd_file, safetensor_files)

    elapsed = time.time() - t0
    speed = total_size / elapsed / 1024**3
    return total_size, speed


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-32B-Instruct"

    print("=" * 70)
    print("OPTIMIZED DISK READ BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")

    model_path = get_model_path(model_name)
    print(f"Path: {model_path}")

    results = {}

    # Test 1: Parallel file reads with mmap
    print("\n" + "-" * 50)
    print("TEST 1: Parallel mmap (4 threads)")
    print("-" * 50)
    _, speed = benchmark_parallel_files(model_path, num_threads=4)
    print(f"Result: {speed:.2f} GB/s")
    results['parallel_mmap_4'] = speed

    # Test 2: Parallel file reads (8 threads)
    print("\n" + "-" * 50)
    print("TEST 2: Parallel mmap (8 threads)")
    print("-" * 50)
    _, speed = benchmark_parallel_files(model_path, num_threads=8)
    print(f"Result: {speed:.2f} GB/s")
    results['parallel_mmap_8'] = speed

    # Test 3: Large buffer sequential
    print("\n" + "-" * 50)
    print("TEST 3: Large buffer (256MB)")
    print("-" * 50)
    _, speed = benchmark_large_buffer_read(model_path, buffer_size_mb=256)
    print(f"Result: {speed:.2f} GB/s")
    results['large_buffer'] = speed

    # Test 4: readinto with pre-allocated buffer
    print("\n" + "-" * 50)
    print("TEST 4: readinto() pre-allocated buffer")
    print("-" * 50)
    _, speed = benchmark_readinto_buffer(model_path, buffer_size_mb=256)
    print(f"Result: {speed:.2f} GB/s")
    results['readinto'] = speed

    # Test 5: os.read()
    print("\n" + "-" * 50)
    print("TEST 5: os.read() low-level")
    print("-" * 50)
    _, speed = benchmark_os_read(model_path, buffer_size_mb=256)
    print(f"Result: {speed:.2f} GB/s")
    results['os_read'] = speed

    # Test 6: dd command
    print("\n" + "-" * 50)
    print("TEST 6: dd command (native)")
    print("-" * 50)
    _, speed = benchmark_dd_read(model_path)
    print(f"Result: {speed:.2f} GB/s")
    results['dd'] = speed

    # Test 7: cat command
    print("\n" + "-" * 50)
    print("TEST 7: cat command (native)")
    print("-" * 50)
    _, speed = benchmark_cat_read(model_path)
    print(f"Result: {speed:.2f} GB/s")
    results['cat'] = speed

    # Test 8: Parallel dd
    print("\n" + "-" * 50)
    print("TEST 8: Parallel dd (4 jobs)")
    print("-" * 50)
    _, speed = benchmark_parallel_dd(model_path, num_jobs=4)
    print(f"Result: {speed:.2f} GB/s")
    results['parallel_dd'] = speed

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, speed in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}: {speed:.2f} GB/s")
    print("=" * 70)
    print(f"Best method: {max(results.items(), key=lambda x: x[1])[0]}")


if __name__ == '__main__':
    main()
