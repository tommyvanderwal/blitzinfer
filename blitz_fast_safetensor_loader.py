#!/usr/bin/env python3
"""
Fast safetensor loader using pinned memory for maximum GPU transfer speed.

Key insight:
- Reading into pinned memory: 11.4 GB/s
- Pinned -> GPU: 48.5 GB/s
- Effective: 9.2 GB/s (vs 5-7 GB/s with safetensors mmap)

Strategy:
1. Read safetensor file raw bytes into pinned memory
2. Parse header to get tensor offsets
3. Create tensor views pointing to pinned memory
4. Transfer to GPU at full PCIe speed
"""

import os
import sys
import time
import json
import glob
import ctypes
from pathlib import Path
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np


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


def parse_safetensor_header(data: bytes) -> Tuple[Dict, int]:
    """Parse safetensor file header to get tensor metadata"""
    # First 8 bytes are header size (little endian)
    header_size = int.from_bytes(data[:8], 'little')
    # Header is JSON
    header_json = data[8:8+header_size].decode('utf-8')
    header = json.loads(header_json)
    data_start = 8 + header_size
    return header, data_start


DTYPE_MAP = {
    'F32': (torch.float32, 4),
    'F16': (torch.float16, 2),
    'BF16': (torch.bfloat16, 2),
    'I64': (torch.int64, 8),
    'I32': (torch.int32, 4),
    'I16': (torch.int16, 2),
    'I8': (torch.int8, 1),
    'U8': (torch.uint8, 1),
    'BOOL': (torch.bool, 1),
}


class PinnedBuffer:
    """Reusable pinned memory buffer for fast GPU transfers"""

    def __init__(self, size_bytes: int):
        """Pre-allocate pinned memory buffer"""
        self.size = size_bytes
        self.tensor = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
        self.ptr = self.tensor.data_ptr()
        self.ctypes_buffer = (ctypes.c_char * size_bytes).from_address(self.ptr)

    def read_file(self, file_path: str) -> int:
        """Read file directly into pinned memory. Returns bytes read."""
        file_size = os.path.getsize(file_path)
        if file_size > self.size:
            raise ValueError(f"File {file_size} bytes exceeds buffer {self.size} bytes")

        with open(file_path, 'rb') as f:
            bytes_read = f.readinto(self.ctypes_buffer)

        return bytes_read

    def get_tensor_view(self, offset: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Get a tensor view into the pinned buffer at given offset"""
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = 1
        for dim in shape:
            num_elements *= dim

        # Create view of the raw bytes
        byte_view = self.tensor[offset:offset + num_elements * element_size]
        # Reinterpret as the target dtype
        return byte_view.view(dtype).view(shape)


def load_safetensor_fast(file_path: str, pinned_buffer: PinnedBuffer = None) -> Dict[str, torch.Tensor]:
    """
    Load safetensor file using pinned memory for fast GPU transfer.

    Returns dict of GPU tensors.
    """
    file_size = os.path.getsize(file_path)

    # Use provided buffer or create temporary one
    if pinned_buffer is None:
        pinned_buffer = PinnedBuffer(file_size)

    # Read file into pinned memory
    bytes_read = pinned_buffer.read_file(file_path)

    # Parse header (need to copy small amount of header bytes)
    header_size = int.from_bytes(bytes(pinned_buffer.ctypes_buffer[:8]), 'little')
    header_bytes = bytes(pinned_buffer.ctypes_buffer[8:8+header_size])
    header = json.loads(header_bytes.decode('utf-8'))
    data_start = 8 + header_size

    # Create tensor views and transfer to GPU
    gpu_tensors = {}
    for name, meta in header.items():
        if name == '__metadata__':
            continue

        dtype_str = meta['dtype']
        shape = meta['shape']
        offsets = meta['data_offsets']
        start, end = offsets

        torch_dtype, elem_size = DTYPE_MAP[dtype_str]

        # Create view of pinned memory
        tensor_view = pinned_buffer.get_tensor_view(data_start + start, shape, torch_dtype)

        # Transfer to GPU (fast because pinned)
        gpu_tensors[name] = tensor_view.to('cuda', non_blocking=True)

    torch.cuda.synchronize()
    return gpu_tensors


def load_model_fast(model_path: Path, max_buffer_size: int = 8 * 1024**3) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Load all safetensor files for a model using fast pinned memory loading.

    Args:
        model_path: Path to model directory
        max_buffer_size: Maximum pinned buffer size (default 8GB)

    Returns:
        (tensor_dict, total_size_bytes)
    """
    safetensor_files = sorted(glob.glob(str(model_path / "*.safetensors")))

    # Find largest file to size buffer
    max_file_size = max(os.path.getsize(f) for f in safetensor_files)
    buffer_size = min(max_file_size, max_buffer_size)

    print(f"Allocating {buffer_size/1e9:.2f} GB pinned buffer...")
    t0 = time.time()
    pinned_buffer = PinnedBuffer(buffer_size)
    alloc_time = time.time() - t0
    print(f"Buffer allocated in {alloc_time:.2f}s")

    all_tensors = {}
    total_size = 0

    for sf in safetensor_files:
        tensors = load_safetensor_fast(sf, pinned_buffer)
        all_tensors.update(tensors)
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())

    return all_tensors, total_size


def benchmark_loading(model_name: str):
    """Benchmark fast loading vs safetensors default"""
    from safetensors import safe_open

    print("=" * 70)
    print(f"Benchmarking: {model_name}")
    print("=" * 70)

    model_path = get_model_path(model_name)
    safetensor_files = sorted(glob.glob(str(model_path / "*.safetensors")))
    total_file_size = sum(os.path.getsize(f) for f in safetensor_files)

    print(f"Path: {model_path}")
    print(f"Files: {len(safetensor_files)}")
    print(f"Total size: {total_file_size/1e9:.2f} GB")

    # Check GPU memory
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: {free/1e9:.2f}/{total/1e9:.2f} GiB free")

    # Warm page cache
    print("\nWarming page cache...")
    for sf in safetensor_files:
        with open(sf, 'rb') as f:
            _ = f.read()
    print("Cache warm.")

    results = {}

    # Method 1: Standard safetensors
    print("\n--- Method 1: Safetensors Direct to GPU (baseline) ---")
    torch.cuda.empty_cache()
    t0 = time.time()
    all_tensors = {}
    for sf in safetensor_files:
        with safe_open(sf, framework="pt", device="cuda") as f:
            for k in f.keys():
                all_tensors[k] = f.get_tensor(k)
    torch.cuda.synchronize()
    t1 = time.time()

    tensor_size = sum(t.numel() * t.element_size() for t in all_tensors.values())
    speed = tensor_size / (t1 - t0) / 1e9
    print(f"Time: {t1-t0:.2f}s, Size: {tensor_size/1e9:.2f} GB, Speed: {speed:.1f} GB/s")
    results['safetensors'] = {'time': t1-t0, 'speed': speed}
    del all_tensors
    torch.cuda.empty_cache()

    # Method 2: Fast pinned loader
    print("\n--- Method 2: Fast Pinned Memory Loader ---")
    torch.cuda.empty_cache()
    t0 = time.time()
    all_tensors, tensor_size = load_model_fast(model_path)
    t1 = time.time()

    speed = tensor_size / (t1 - t0) / 1e9
    print(f"Time: {t1-t0:.2f}s, Size: {tensor_size/1e9:.2f} GB, Speed: {speed:.1f} GB/s")
    results['fast_pinned'] = {'time': t1-t0, 'speed': speed}
    del all_tensors
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    baseline = results['safetensors']['time']
    for method, data in results.items():
        speedup = baseline / data['time']
        print(f"{method:20s}: {data['time']:.2f}s ({data['speed']:.1f} GB/s) - {speedup:.2f}x baseline")

    # Theoretical limits
    print(f"\nTheoretical limits:")
    print(f"  Read (11.4 GB/s): {tensor_size/1e9/11.4:.1f}s")
    print(f"  Transfer (48.5 GB/s): {tensor_size/1e9/48.5:.1f}s")
    print(f"  Combined: {tensor_size/1e9/11.4 + tensor_size/1e9/48.5:.1f}s")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"

    if model == "gpt":
        model = "openai/gpt-oss-120b"
    elif model == "qwen":
        model = "Qwen/Qwen3-VL-32B-Instruct"

    benchmark_loading(model)


if __name__ == "__main__":
    main()
