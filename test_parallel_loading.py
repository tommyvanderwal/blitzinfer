#!/usr/bin/env python3
"""
Test parallel weight loading approaches.

Question: Can weight loading be parallelized?
Answer: YES - there's no fundamental reason it needs to be sequential.

Each weight goes to a different parameter, so:
1. Reading from disk can be parallelized
2. CPU→GPU copies can happen on different CUDA streams
3. Pipelining can overlap read and copy operations

Let's test different approaches.
"""

import concurrent.futures
import gc
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import safe_open, load_file

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def find_safetensors(model_name: str) -> List[str]:
    """Find safetensors files for a model."""
    model_dir = model_name.replace("/", "--")
    for cache_dir in HF_CACHE.glob(f"models--{model_dir}*"):
        snapshots = cache_dir / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                st_files = list(snapshot.glob("model-*.safetensors"))
                if st_files:
                    return sorted([str(f) for f in st_files])
    raise FileNotFoundError(f"No safetensors found for {model_name}")


def get_mem():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024**3), free / (1024**3)


# Approach 1: Sequential (baseline)
def load_sequential(st_files: List[str], gpu_buffer: torch.Tensor) -> float:
    """Sequential loading - current vLLM approach."""
    t0 = time.perf_counter()

    for sf in st_files:
        with safe_open(sf, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                numel = tensor.numel()
                gpu_buffer[:numel].copy_(tensor.view(-1)[:numel])

    torch.cuda.synchronize()
    return time.perf_counter() - t0


# Approach 2: Pipelined with pinned memory
def load_pipelined(st_files: List[str], gpu_buffer: torch.Tensor, n_buffers: int = 2) -> float:
    """Pipelined: read into pinned buffer while copying previous to GPU."""
    t0 = time.perf_counter()

    # Pre-allocate pinned buffers
    max_size = 500_000_000  # 500M elements should be enough for any single tensor
    pinned_buffers = [torch.empty(max_size, dtype=torch.float16, pin_memory=True) for _ in range(n_buffers)]

    # Create CUDA streams for async copies
    streams = [torch.cuda.Stream() for _ in range(n_buffers)]

    buffer_idx = 0
    pending_copies = []

    for sf in st_files:
        with safe_open(sf, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                numel = tensor.numel()

                # Wait for previous copy on this buffer to complete
                if len(pending_copies) >= n_buffers:
                    old_stream, old_event = pending_copies.pop(0)
                    old_event.wait()

                # Copy to pinned buffer
                pinned_buf = pinned_buffers[buffer_idx]
                pinned_buf[:numel].copy_(tensor.view(-1)[:numel])

                # Async copy to GPU
                stream = streams[buffer_idx]
                with torch.cuda.stream(stream):
                    gpu_buffer[:numel].copy_(pinned_buf[:numel], non_blocking=True)

                event = torch.cuda.Event()
                event.record(stream)
                pending_copies.append((stream, event))

                buffer_idx = (buffer_idx + 1) % n_buffers

    # Wait for all pending copies
    torch.cuda.synchronize()
    return time.perf_counter() - t0


# Approach 3: Bulk load then parallel copy
def load_bulk_parallel(st_files: List[str], gpu_buffer: torch.Tensor, n_streams: int = 4) -> float:
    """Load all to CPU first, then copy to GPU using multiple streams."""
    t0 = time.perf_counter()

    # Phase 1: Load all tensors to CPU
    t_load = time.perf_counter()
    all_tensors = []
    for sf in st_files:
        with safe_open(sf, framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                all_tensors.append(tensor)
    load_time = (time.perf_counter() - t_load) * 1000

    # Phase 2: Copy to GPU using multiple streams
    t_copy = time.perf_counter()
    streams = [torch.cuda.Stream() for _ in range(n_streams)]

    for i, tensor in enumerate(all_tensors):
        stream = streams[i % n_streams]
        numel = tensor.numel()
        with torch.cuda.stream(stream):
            # Copy each tensor independently - don't try to pack into continuous buffer
            gpu_buffer[:numel].copy_(tensor.view(-1)[:numel], non_blocking=True)

    torch.cuda.synchronize()
    copy_time = (time.perf_counter() - t_copy) * 1000

    print(f"    Bulk load: {load_time:.0f}ms, Parallel copy: {copy_time:.0f}ms")
    return time.perf_counter() - t0


# Approach 4: Thread pool for file loading + pipelined GPU copy
def load_threaded_pipelined(st_files: List[str], gpu_buffer: torch.Tensor, n_threads: int = 4) -> float:
    """Multi-threaded file loading with pipelined GPU copies."""
    t0 = time.perf_counter()

    # Load files in parallel using thread pool
    def load_file(sf):
        return load_file(sf, device='cpu')

    all_state_dicts = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(lambda f=sf: load_file(f, device='cpu'), sf) for sf in st_files]
        for future in concurrent.futures.as_completed(futures):
            all_state_dicts.append(future.result())

    # Now copy all tensors to GPU
    for state_dict in all_state_dicts:
        for key, tensor in state_dict.items():
            numel = tensor.numel()
            gpu_buffer[:numel].copy_(tensor.view(-1)[:numel], non_blocking=True)

    torch.cuda.synchronize()
    return time.perf_counter() - t0


# Approach 5: Producer-consumer with queue
def load_producer_consumer(st_files: List[str], gpu_buffer: torch.Tensor) -> float:
    """Producer thread reads tensors, consumer thread copies to GPU."""
    t0 = time.perf_counter()

    # Queue for tensor transfer
    tensor_queue = queue.Queue(maxsize=8)  # Buffer up to 8 tensors
    done_event = threading.Event()

    # Pre-allocate pinned buffers for the queue
    max_size = 500_000_000
    pinned_pool = [torch.empty(max_size, dtype=torch.float16, pin_memory=True) for _ in range(8)]
    pool_idx = [0]  # Mutable for closure

    def producer():
        """Read tensors from disk into pinned memory."""
        for sf in st_files:
            with safe_open(sf, framework='pt') as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    numel = tensor.numel()

                    # Get a pinned buffer
                    buf = pinned_pool[pool_idx[0] % len(pinned_pool)]
                    pool_idx[0] += 1

                    # Copy to pinned
                    buf[:numel].copy_(tensor.view(-1)[:numel])

                    # Put (numel, buffer_slice) in queue
                    tensor_queue.put((numel, buf[:numel].clone()))

        done_event.set()

    def consumer():
        """Copy tensors from queue to GPU."""
        while not (done_event.is_set() and tensor_queue.empty()):
            try:
                numel, pinned_tensor = tensor_queue.get(timeout=0.1)
                gpu_buffer[:numel].copy_(pinned_tensor, non_blocking=True)
                tensor_queue.task_done()
            except queue.Empty:
                continue

    # Start threads
    prod_thread = threading.Thread(target=producer)
    cons_thread = threading.Thread(target=consumer)

    prod_thread.start()
    cons_thread.start()

    prod_thread.join()
    cons_thread.join()

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    print("="*70)
    print("PARALLEL WEIGHT LOADING INVESTIGATION")
    print("="*70)

    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    st_files = find_safetensors(model_name)
    total_bytes = sum(Path(f).stat().st_size for f in st_files)
    gb = total_bytes / (1024**3)

    print(f"\nModel: {model_name}")
    print(f"Total: {gb:.2f} GB in {len(st_files)} files")

    # Allocate GPU buffer
    gpu_buffer = torch.empty(500_000_000, dtype=torch.float16, device='cuda')

    results = []

    # Test 1: Sequential (baseline)
    print("\n" + "-"*70)
    print("1. Sequential (vLLM baseline)")
    print("-"*70)
    gc.collect()
    torch.cuda.synchronize()
    elapsed = load_sequential(st_files, gpu_buffer)
    bw = gb / elapsed
    print(f"   Time: {elapsed*1000:.0f}ms, Bandwidth: {bw:.1f} GB/s")
    results.append(("Sequential", elapsed*1000, bw))

    # Test 2: Pipelined with pinned memory
    print("\n" + "-"*70)
    print("2. Pipelined (2 pinned buffers + async copy)")
    print("-"*70)
    gc.collect()
    torch.cuda.synchronize()
    elapsed = load_pipelined(st_files, gpu_buffer, n_buffers=2)
    bw = gb / elapsed
    print(f"   Time: {elapsed*1000:.0f}ms, Bandwidth: {bw:.1f} GB/s")
    results.append(("Pipelined(2)", elapsed*1000, bw))

    # Test 3: Pipelined with more buffers
    print("\n" + "-"*70)
    print("3. Pipelined (4 pinned buffers + async copy)")
    print("-"*70)
    gc.collect()
    torch.cuda.synchronize()
    elapsed = load_pipelined(st_files, gpu_buffer, n_buffers=4)
    bw = gb / elapsed
    print(f"   Time: {elapsed*1000:.0f}ms, Bandwidth: {bw:.1f} GB/s")
    results.append(("Pipelined(4)", elapsed*1000, bw))

    # Test 4: Bulk load + parallel copy
    print("\n" + "-"*70)
    print("4. Bulk load + parallel GPU copy (4 streams)")
    print("-"*70)
    gc.collect()
    torch.cuda.synchronize()
    elapsed = load_bulk_parallel(st_files, gpu_buffer, n_streams=4)
    bw = gb / elapsed
    print(f"   Time: {elapsed*1000:.0f}ms, Bandwidth: {bw:.1f} GB/s")
    results.append(("Bulk+Parallel", elapsed*1000, bw))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    baseline = results[0][1]
    for name, time_ms, bw in results:
        speedup = baseline / time_ms
        print(f"  {name}: {time_ms:.0f}ms ({bw:.1f} GB/s) - {speedup:.2f}x")

    del gpu_buffer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
