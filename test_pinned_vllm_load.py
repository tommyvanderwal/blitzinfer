#!/usr/bin/env python3
"""
Test: Load vLLM model from pinned memory arena (bypassing disk I/O).

The goal is to achieve ~6s model switch:
- Pinned → GPU transfer: ~1.4s (48 GB/s for 65GB)
- vLLM init: ~5s (tokenizer, KV cache, etc.)

Approach:
1. Pre-load safetensor bytes into pinned arena
2. Parse safetensor headers to get tensor metadata
3. Create tensor views from pinned memory
4. Transfer to GPU at full PCIe bandwidth
5. Initialize vLLM model skeleton (no weights)
6. Inject pre-loaded weights

This test validates steps 1-4. Step 5-6 require vLLM modification.
"""

import os
import gc
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import json
import struct

# Configure for single-process mode
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch
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
                return int(line.split()[1]) / 1024 / 1024


def cleanup():
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# Safetensor dtype mapping
SAFETENSOR_DTYPE_MAP = {
    'F64': torch.float64,
    'F32': torch.float32,
    'F16': torch.float16,
    'BF16': torch.bfloat16,
    'I64': torch.int64,
    'I32': torch.int32,
    'I16': torch.int16,
    'I8': torch.int8,
    'U8': torch.uint8,
    'BOOL': torch.bool,
}


def parse_safetensor_header(file_path: str) -> Tuple[int, Dict]:
    """Parse safetensor header to get tensor metadata."""
    with open(file_path, 'rb') as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        header_json = f.read(header_size)
        header = json.loads(header_json)
    return header_size, header


def get_safetensor_files(model_path: Path) -> List[Path]:
    """Get sorted safetensor files."""
    files = list(model_path.glob("*.safetensors"))
    def sort_key(p):
        name = p.stem
        if '-of-' in name:
            try:
                for part in name.split('-'):
                    if part.isdigit():
                        return (0, int(part), name)
            except (ValueError, IndexError):
                pass
        return (1, 0, name)
    return sorted(files, key=sort_key)


def main():
    print("=" * 80)
    print("PINNED MEMORY → GPU LOAD TEST")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")
    print(f"RAM free: {get_ram_free_gb():.1f} GB")

    # Initialize CUDA
    _ = torch.randn(1000, device='cuda')
    cleanup()

    model_name = "openai/gpt-oss-120b"
    model_path = Path(snapshot_download(model_name, local_files_only=True))
    sf_files = get_safetensor_files(model_path)
    total_bytes = sum(f.stat().st_size for f in sf_files)
    total_gb = total_bytes / 1e9

    print(f"\nModel: {model_name}")
    print(f"Files: {len(sf_files)}")
    print(f"Total size: {total_gb:.1f} GB")

    # PHASE 1: Pre-load into pinned arena
    print("\n" + "-" * 80)
    print("PHASE 1: Pre-load safetensor files into pinned arena")
    print("-" * 80)

    # Allocate pinned buffer
    print(f"Allocating {total_gb:.1f}GB pinned buffer...")
    t0 = time.perf_counter()
    pinned_buffer = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    alloc_time = time.perf_counter() - t0
    print(f"Allocation time: {alloc_time:.2f}s")
    print(f"RAM free after alloc: {get_ram_free_gb():.1f}GB")

    # Read files into pinned buffer
    print("\nReading files into pinned buffer...")
    t0 = time.perf_counter()
    offset = 0
    file_info = []  # (start, size, header_size, tensors)

    for i, sf_file in enumerate(sf_files):
        file_size = sf_file.stat().st_size

        # Parse header
        header_size, header = parse_safetensor_header(str(sf_file))
        data_offset = 8 + header_size  # 8 bytes for header size + header JSON

        # Parse tensor info
        tensors = {}
        for name, info in header.items():
            if name == '__metadata__':
                continue
            tensors[name] = {
                'dtype': SAFETENSOR_DTYPE_MAP.get(info['dtype'], torch.float16),
                'shape': tuple(info['shape']),
                'data_offsets': info['data_offsets'],
            }

        # Read file into buffer
        with open(sf_file, 'rb') as f:
            data = f.read()
            pinned_buffer[offset:offset+file_size].copy_(
                torch.frombuffer(bytearray(data), dtype=torch.uint8)
            )

        file_info.append((offset, file_size, data_offset, tensors))
        offset += file_size

        if (i + 1) % 5 == 0 or i == len(sf_files) - 1:
            print(f"  Loaded {i+1}/{len(sf_files)} files ({offset/1e9:.1f}GB)")

        del data
        gc.collect()

    read_time = time.perf_counter() - t0
    print(f"\nRead time: {read_time:.2f}s ({total_gb / read_time:.1f} GB/s)")

    # PHASE 2: Create tensor views from pinned buffer
    print("\n" + "-" * 80)
    print("PHASE 2: Create tensor views from pinned buffer")
    print("-" * 80)

    t0 = time.perf_counter()
    all_tensors = {}
    total_tensor_bytes = 0

    for file_start, file_size, data_offset, tensors in file_info:
        for name, info in tensors.items():
            dtype = info['dtype']
            shape = info['shape']
            start, end = info['data_offsets']
            tensor_size = end - start

            # Calculate absolute offset in pinned buffer
            abs_offset = file_start + data_offset + start

            # Create view into pinned buffer
            byte_view = pinned_buffer[abs_offset:abs_offset + tensor_size]

            # Reinterpret as correct dtype and shape
            # Note: This creates a view, not a copy
            element_size = byte_view.numel() // (torch.tensor([], dtype=dtype).numel() or 1)
            if dtype == torch.uint8:
                tensor_view = byte_view.view(shape)
            else:
                # Convert bytes to proper dtype
                tensor_view = byte_view.view(dtype).view(shape)

            all_tensors[name] = tensor_view
            total_tensor_bytes += tensor_size

    view_time = time.perf_counter() - t0
    print(f"Created {len(all_tensors)} tensor views in {view_time:.3f}s")
    print(f"Total tensor data: {total_tensor_bytes / 1e9:.1f}GB")

    # PHASE 3: Transfer to GPU
    print("\n" + "-" * 80)
    print("PHASE 3: Transfer pinned tensor views to GPU")
    print("-" * 80)

    cleanup()
    torch.cuda.synchronize()

    print("Transferring tensors to GPU...")
    t0 = time.perf_counter()
    gpu_tensors = {}
    transfer_bytes = 0

    for name, tensor in all_tensors.items():
        gpu_tensors[name] = tensor.to('cuda', non_blocking=True)
        transfer_bytes += tensor.numel() * tensor.element_size()

    torch.cuda.synchronize()
    transfer_time = time.perf_counter() - t0
    transfer_bw = (transfer_bytes / 1e9) / transfer_time

    print(f"Transfer time: {transfer_time:.2f}s ({transfer_bw:.1f} GB/s)")
    print(f"Transferred: {transfer_bytes / 1e9:.1f}GB, {len(gpu_tensors)} tensors")
    print(f"GPU free after: {nvidia_smi_free_gb():.1f} GB")

    # PHASE 4: Cleanup and summary
    print("\n" + "-" * 80)
    print("PHASE 4: Cleanup")
    print("-" * 80)

    del gpu_tensors
    torch.cuda.empty_cache()
    del all_tensors
    del pinned_buffer
    cleanup()

    print(f"GPU free after cleanup: {nvidia_smi_free_gb():.1f} GB")
    print(f"RAM free after cleanup: {get_ram_free_gb():.1f} GB")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    vllm_init_estimate = 5.0  # Estimated vLLM initialization time

    print(f"\n{'Phase':<35} {'Time':>10}")
    print("-" * 50)
    print(f"{'1. Pre-load to pinned arena':<35} {read_time:>9.2f}s")
    print(f"{'2. Create tensor views':<35} {view_time*1000:>8.0f}ms")
    print(f"{'3. Transfer to GPU':<35} {transfer_time:>9.2f}s")
    print(f"{'4. vLLM init (estimated)':<35} {vllm_init_estimate:>9.2f}s")

    total_no_prefetch = read_time + transfer_time + vllm_init_estimate
    total_with_prefetch = transfer_time + vllm_init_estimate

    print(f"\n{'TOTAL (no prefetch)':<35} {total_no_prefetch:>9.2f}s")
    print(f"{'TOTAL (with prefetch)':<35} {total_with_prefetch:>9.2f}s")

    print(f"""
ANALYSIS:
- Pre-load time ({read_time:.1f}s) can be hidden during inference
- GPU transfer achieved {transfer_bw:.1f} GB/s (vs 48 GB/s theoretical)
- With prefetch: switch takes ~{total_with_prefetch:.1f}s instead of ~20s

BOTTLENECK: GPU transfer is {transfer_bw:.1f} GB/s, not the theoretical 48 GB/s.
This is because we're transferring {len(gpu_tensors)} individual tensors.

OPTIMIZATION NEEDED:
1. Transfer as single contiguous buffer (should hit 48 GB/s)
2. Then slice into individual tensors on GPU
""")


if __name__ == '__main__':
    main()
