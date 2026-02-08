#!/usr/bin/env python3
"""Benchmark memory transfer speeds: CPU↔GPU, pinned vs unpinned.

This tests what's physically possible for weight transfers.
"""
import os
import time
import torch

def benchmark_cpu_to_gpu(size_gb: float, pinned: bool = False, device: str = 'cuda:0'):
    """Benchmark CPU to GPU transfer speed."""
    size_bytes = int(size_gb * 1024**3)
    num_elements = size_bytes // 2  # bfloat16 = 2 bytes

    # Allocate CPU tensor
    if pinned:
        cpu_tensor = torch.empty(num_elements, dtype=torch.bfloat16, pin_memory=True)
    else:
        cpu_tensor = torch.empty(num_elements, dtype=torch.bfloat16)

    # Fill with random data
    cpu_tensor.fill_(1.0)

    # Warmup
    for _ in range(3):
        gpu = cpu_tensor.to(device, non_blocking=False)
        torch.cuda.synchronize()
        del gpu

    # Benchmark
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        gpu = cpu_tensor.to(device, non_blocking=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)
        del gpu
        torch.cuda.empty_cache()

    avg_time = sum(times) / len(times)
    speed = size_gb / avg_time

    return avg_time, speed


def benchmark_gpu_to_cpu(size_gb: float, pinned: bool = False, device: str = 'cuda:0'):
    """Benchmark GPU to CPU transfer speed."""
    size_bytes = int(size_gb * 1024**3)
    num_elements = size_bytes // 2  # bfloat16 = 2 bytes

    # Allocate GPU tensor
    gpu_tensor = torch.empty(num_elements, dtype=torch.bfloat16, device=device)
    gpu_tensor.fill_(1.0)

    # Warmup
    for _ in range(3):
        if pinned:
            cpu = gpu_tensor.to('cpu', non_blocking=False).pin_memory()
        else:
            cpu = gpu_tensor.to('cpu', non_blocking=False)
        torch.cuda.synchronize()
        del cpu

    # Benchmark
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        if pinned:
            cpu = gpu_tensor.to('cpu', non_blocking=False).pin_memory()
        else:
            cpu = gpu_tensor.to('cpu', non_blocking=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        times.append(elapsed)
        del cpu

    avg_time = sum(times) / len(times)
    speed = size_gb / avg_time

    del gpu_tensor
    torch.cuda.empty_cache()

    return avg_time, speed


def benchmark_nonblocking_transfer(size_gb: float, device: str = 'cuda:0'):
    """Test if non-blocking transfers are actually faster."""
    size_bytes = int(size_gb * 1024**3)
    num_elements = size_bytes // 2

    cpu_tensor = torch.empty(num_elements, dtype=torch.bfloat16, pin_memory=True)
    cpu_tensor.fill_(1.0)

    # Non-blocking transfer
    torch.cuda.synchronize()
    t0 = time.time()
    gpu = cpu_tensor.to(device, non_blocking=True)
    t_launch = time.time() - t0  # Time to launch the transfer
    torch.cuda.synchronize()
    t_total = time.time() - t0  # Time including transfer

    del gpu
    torch.cuda.empty_cache()

    return t_launch, t_total


def benchmark_multiple_streams(size_gb: float, num_streams: int = 4, device: str = 'cuda:0'):
    """Test parallel transfers using multiple CUDA streams."""
    size_bytes = int(size_gb * 1024**3)
    chunk_size = size_bytes // num_streams
    num_elements_per_chunk = chunk_size // 2

    # Create pinned CPU chunks
    cpu_chunks = [
        torch.empty(num_elements_per_chunk, dtype=torch.bfloat16, pin_memory=True)
        for _ in range(num_streams)
    ]
    for c in cpu_chunks:
        c.fill_(1.0)

    # Create streams
    streams = [torch.cuda.Stream(device) for _ in range(num_streams)]

    # Warmup
    for _ in range(2):
        gpu_chunks = []
        for i, (chunk, stream) in enumerate(zip(cpu_chunks, streams)):
            with torch.cuda.stream(stream):
                gpu_chunks.append(chunk.to(device, non_blocking=True))
        torch.cuda.synchronize()
        del gpu_chunks

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.time()

    gpu_chunks = []
    for chunk, stream in zip(cpu_chunks, streams):
        with torch.cuda.stream(stream):
            gpu_chunks.append(chunk.to(device, non_blocking=True))
    torch.cuda.synchronize()

    elapsed = time.time() - t0
    speed = size_gb / elapsed

    del gpu_chunks
    torch.cuda.empty_cache()

    return elapsed, speed


def main():
    print("=" * 70)
    print("MEMORY TRANSFER SPEED BENCHMARK")
    print("=" * 70)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    test_sizes = [1.0, 10.0, 30.0]  # GB

    results = {}

    for size in test_sizes:
        print(f"\n{'='*70}")
        print(f"Testing {size} GB transfers")
        print(f"{'='*70}")

        # CPU → GPU (unpinned)
        print(f"\n--- CPU → GPU (unpinned memory) ---")
        time_up, speed_up = benchmark_cpu_to_gpu(size, pinned=False)
        print(f"Time: {time_up:.3f}s, Speed: {speed_up:.2f} GB/s")
        results[f'{size}GB_cpu_to_gpu_unpinned'] = speed_up

        # CPU → GPU (pinned)
        print(f"\n--- CPU → GPU (pinned memory) ---")
        time_p, speed_p = benchmark_cpu_to_gpu(size, pinned=True)
        print(f"Time: {time_p:.3f}s, Speed: {speed_p:.2f} GB/s")
        results[f'{size}GB_cpu_to_gpu_pinned'] = speed_p

        # GPU → CPU (unpinned)
        print(f"\n--- GPU → CPU (unpinned memory) ---")
        time_down_up, speed_down_up = benchmark_gpu_to_cpu(size, pinned=False)
        print(f"Time: {time_down_up:.3f}s, Speed: {speed_down_up:.2f} GB/s")
        results[f'{size}GB_gpu_to_cpu_unpinned'] = speed_down_up

        # GPU → CPU (pinned)
        print(f"\n--- GPU → CPU (pinned memory) ---")
        time_down_p, speed_down_p = benchmark_gpu_to_cpu(size, pinned=True)
        print(f"Time: {time_down_p:.3f}s, Speed: {speed_down_p:.2f} GB/s")
        results[f'{size}GB_gpu_to_cpu_pinned'] = speed_down_p

        # Non-blocking test
        if size <= 10:
            print(f"\n--- Non-blocking transfer (pinned) ---")
            t_launch, t_total = benchmark_nonblocking_transfer(size)
            print(f"Launch time: {t_launch*1000:.1f}ms, Total time: {t_total:.3f}s")
            print(f"Speed: {size/t_total:.2f} GB/s")

        # Multi-stream test
        if size <= 10:
            print(f"\n--- Multi-stream transfer (4 streams, pinned) ---")
            time_ms, speed_ms = benchmark_multiple_streams(size, num_streams=4)
            print(f"Time: {time_ms:.3f}s, Speed: {speed_ms:.2f} GB/s")
            results[f'{size}GB_multi_stream_4'] = speed_ms

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - CPU ↔ GPU Transfer Speeds")
    print("=" * 70)
    print(f"{'Test':<40} {'Speed (GB/s)':>15}")
    print("-" * 55)
    for name, speed in sorted(results.items()):
        print(f"{name:<40} {speed:>15.2f}")
    print("=" * 70)

    # Implications
    print("\nIMPLICATIONS FOR MODEL LOADING:")
    if '30.0GB_cpu_to_gpu_pinned' in results:
        speed = results['30.0GB_cpu_to_gpu_pinned']
        time_62gb = 62 / speed
        print(f"  - 62 GB model from pinned RAM to GPU: ~{time_62gb:.1f}s at {speed:.1f} GB/s")
    if '30.0GB_cpu_to_gpu_unpinned' in results:
        speed = results['30.0GB_cpu_to_gpu_unpinned']
        time_62gb = 62 / speed
        print(f"  - 62 GB model from unpinned RAM to GPU: ~{time_62gb:.1f}s at {speed:.1f} GB/s")


if __name__ == '__main__':
    main()
