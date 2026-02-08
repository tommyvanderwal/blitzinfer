#!/usr/bin/env python3
"""
Profile the EXACT weight loading bottleneck for 60GB+ models.

Goal: Find why 60GB transfer takes 17s instead of <2s (PCIe 5.0 x16 = 64 GB/s)

MEMORY-EFFICIENT VERSION: Tests run sequentially, freeing memory between tests.
"""

import gc
import os
import time
import subprocess
from pathlib import Path
from typing import List, Dict

import torch
from safetensors.torch import safe_open, load_file
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


# =============================================================================
# BENCHMARK FUNCTIONS - MEMORY EFFICIENT
# =============================================================================

def bench_pcie_bandwidth_raw() -> Dict:
    """Raw PCIe bandwidth (pinned → GPU, large contiguous block)."""
    cleanup()

    # Test with 8GB - fits in memory easily
    size_gb = 8.0
    size_bytes = int(size_gb * 1024**3)

    cpu_tensor = torch.empty(size_bytes // 2, dtype=torch.float16, pin_memory=True)

    # Warm up
    gpu = cpu_tensor.to('cuda', non_blocking=False)
    torch.cuda.synchronize()
    del gpu
    cleanup()

    # Measure
    t0 = time.perf_counter()
    gpu = cpu_tensor.to('cuda', non_blocking=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del gpu, cpu_tensor
    cleanup()

    return {
        'name': f'Raw PCIe ({size_gb:.0f}GB pinned)',
        'time_s': elapsed,
        'bytes': size_bytes,
        'bandwidth_gbs': (size_bytes / 1e9) / elapsed,
    }


def bench_raw_file_read(files: List[Path]) -> Dict:
    """Raw file read speed (no parsing)."""
    cleanup()
    total_bytes = get_total_size(files)

    t0 = time.perf_counter()
    for f in files:
        with open(f, 'rb') as fp:
            data = fp.read()
        del data
        gc.collect()
    elapsed = time.perf_counter() - t0

    return {
        'name': 'Raw file read (page cache)',
        'time_s': elapsed,
        'bytes': total_bytes,
        'bandwidth_gbs': (total_bytes / 1e9) / elapsed,
    }


def bench_safetensor_header_only(files: List[Path]) -> Dict:
    """Just parse safetensor headers (no tensor load)."""
    cleanup()
    total_bytes = get_total_size(files)

    t0 = time.perf_counter()
    tensor_count = 0
    for f in files:
        with safe_open(str(f), framework='pt') as sf:
            keys = list(sf.keys())
            tensor_count += len(keys)
    elapsed = time.perf_counter() - t0

    return {
        'name': 'Safetensor header parsing',
        'time_s': elapsed,
        'tensor_count': tensor_count,
        'overhead_per_tensor_us': (elapsed * 1e6) / tensor_count if tensor_count > 0 else 0,
    }


def bench_direct_gpu_load_streaming(files: List[Path]) -> Dict:
    """Direct GPU load via safetensors - streaming (free after each file)."""
    cleanup()
    total_bytes = get_total_size(files)

    t0 = time.perf_counter()
    tensor_count = 0
    for f in files:
        state_dict = load_file(str(f), device='cuda')
        tensor_count += len(state_dict)
        # Delete immediately to avoid OOM
        del state_dict
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    cleanup()

    return {
        'name': 'Direct GPU load (streaming)',
        'time_s': elapsed,
        'bytes': total_bytes,
        'bandwidth_gbs': (total_bytes / 1e9) / elapsed,
        'tensor_count': tensor_count,
    }


def bench_single_file_detailed(file_path: Path) -> Dict:
    """Detailed breakdown for a single safetensor file."""
    cleanup()

    file_size = file_path.stat().st_size

    # Test 1: Raw read
    t0 = time.perf_counter()
    with open(file_path, 'rb') as f:
        data = f.read()
    raw_read_time = time.perf_counter() - t0
    del data
    gc.collect()

    # Test 2: Safetensor CPU load
    t0 = time.perf_counter()
    with safe_open(str(file_path), framework='pt') as sf:
        tensors = {k: sf.get_tensor(k) for k in sf.keys()}
    cpu_load_time = time.perf_counter() - t0
    tensor_count = len(tensors)
    del tensors
    gc.collect()

    # Test 3: Direct GPU load
    cleanup()
    t0 = time.perf_counter()
    state_dict = load_file(str(file_path), device='cuda')
    torch.cuda.synchronize()
    gpu_load_time = time.perf_counter() - t0
    del state_dict
    cleanup()

    return {
        'file': file_path.name,
        'size_gb': file_size / 1e9,
        'tensor_count': tensor_count,
        'raw_read_time': raw_read_time,
        'cpu_load_time': cpu_load_time,
        'gpu_load_time': gpu_load_time,
        'raw_read_gbs': (file_size / 1e9) / raw_read_time,
        'cpu_load_gbs': (file_size / 1e9) / cpu_load_time,
        'gpu_load_gbs': (file_size / 1e9) / gpu_load_time,
    }


def bench_many_small_transfers() -> Dict:
    """Many small transfers to measure per-transfer overhead."""
    cleanup()

    # 500 tensors of 20MB each = 10GB total (fits in memory)
    num_tensors = 500
    tensor_size = 20 * 1024 * 1024 // 2  # 20MB in float16 elements
    total_bytes = num_tensors * tensor_size * 2

    # Create pinned tensors
    tensors = [torch.empty(tensor_size, dtype=torch.float16, pin_memory=True) for _ in range(num_tensors)]

    torch.cuda.synchronize()

    # Measure individual transfers
    t0 = time.perf_counter()
    for t in tensors:
        gpu = t.to('cuda', non_blocking=False)
        del gpu
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del tensors
    cleanup()

    return {
        'name': f'Many small transfers ({num_tensors}x20MB)',
        'time_s': elapsed,
        'bytes': total_bytes,
        'bandwidth_gbs': (total_bytes / 1e9) / elapsed,
        'overhead_per_transfer_us': (elapsed * 1e6) / num_tensors,
    }


def bench_few_large_transfers() -> Dict:
    """Few large transfers to measure pure PCIe bandwidth."""
    cleanup()

    # 10 tensors of 1GB each = 10GB total
    num_tensors = 10
    tensor_size = 1024 * 1024 * 1024 // 2  # 1GB in float16 elements
    total_bytes = num_tensors * tensor_size * 2

    torch.cuda.synchronize()

    # Measure
    elapsed_total = 0
    for i in range(num_tensors):
        cpu_tensor = torch.empty(tensor_size, dtype=torch.float16, pin_memory=True)

        t0 = time.perf_counter()
        gpu = cpu_tensor.to('cuda', non_blocking=False)
        torch.cuda.synchronize()
        elapsed_total += time.perf_counter() - t0

        del gpu, cpu_tensor
        torch.cuda.empty_cache()

    cleanup()

    return {
        'name': f'Few large transfers ({num_tensors}x1GB)',
        'time_s': elapsed_total,
        'bytes': total_bytes,
        'bandwidth_gbs': (total_bytes / 1e9) / elapsed_total,
        'overhead_per_transfer_us': (elapsed_total * 1e6) / num_tensors,
    }


def bench_confirm_2x_data(files: List[Path], baseline_time: float) -> Dict:
    """Confirm bottleneck by loading data 2x."""
    cleanup()
    total_bytes = get_total_size(files)

    t0 = time.perf_counter()
    for _ in range(2):  # Load twice
        for f in files:
            state_dict = load_file(str(f), device='cuda')
            del state_dict
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    cleanup()

    slowdown = elapsed / baseline_time if baseline_time > 0 else 0

    return {
        'name': 'Confirm: 2x data transfer',
        'time_s': elapsed,
        'baseline_time': baseline_time,
        'slowdown': slowdown,
        'is_pcie_bottleneck': 1.8 <= slowdown <= 2.2,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("WEIGHT LOADING BOTTLENECK ANALYSIS")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    # Initialize CUDA
    _ = torch.randn(1000, device='cuda')
    cleanup()

    model_name = "openai/gpt-oss-120b"

    print(f"\nModel: {model_name}")
    model_path = find_model_path(model_name)
    files = get_safetensor_files(model_path)
    total_bytes = get_total_size(files)
    total_gb = total_bytes / 1e9
    print(f"Files: {len(files)}")
    print(f"Total size: {total_gb:.1f} GB")

    results = []

    # Test 0: Raw PCIe bandwidth (baseline)
    print("\n" + "-" * 80)
    print("TEST 0: Raw PCIe Bandwidth (baseline)")
    print("-" * 80)
    r = bench_pcie_bandwidth_raw()
    results.append(r)
    print(f"  {r['name']}: {r['time_s']*1000:.0f}ms, {r['bandwidth_gbs']:.1f} GB/s")
    pcie_bw = r['bandwidth_gbs']

    # Test 1: Raw file read (page cache)
    print("\n" + "-" * 80)
    print("TEST 1: Raw File Read (page cache)")
    print("-" * 80)
    r = bench_raw_file_read(files)
    results.append(r)
    print(f"  {r['name']}: {r['time_s']*1000:.0f}ms, {r['bandwidth_gbs']:.1f} GB/s")

    # Test 2: Safetensor header parsing
    print("\n" + "-" * 80)
    print("TEST 2: Safetensor Header Parsing")
    print("-" * 80)
    r = bench_safetensor_header_only(files)
    results.append(r)
    print(f"  {r['name']}: {r['time_s']*1000:.0f}ms")
    print(f"  Tensors: {r['tensor_count']}, Overhead per tensor: {r['overhead_per_tensor_us']:.1f} us")

    # Test 3: Many small transfers (overhead test)
    print("\n" + "-" * 80)
    print("TEST 3: Many Small Transfers (overhead test)")
    print("-" * 80)
    r = bench_many_small_transfers()
    results.append(r)
    print(f"  {r['name']}: {r['time_s']*1000:.0f}ms, {r['bandwidth_gbs']:.1f} GB/s")
    print(f"  Overhead per transfer: {r['overhead_per_transfer_us']:.1f} us")

    # Test 4: Few large transfers (pure PCIe)
    print("\n" + "-" * 80)
    print("TEST 4: Few Large Transfers (pure PCIe)")
    print("-" * 80)
    r = bench_few_large_transfers()
    results.append(r)
    print(f"  {r['name']}: {r['time_s']*1000:.0f}ms, {r['bandwidth_gbs']:.1f} GB/s")
    print(f"  Overhead per transfer: {r['overhead_per_transfer_us']:.1f} us")

    # Test 5: Direct GPU load (streaming)
    print("\n" + "-" * 80)
    print("TEST 5: Direct GPU Load (streaming)")
    print("-" * 80)
    r = bench_direct_gpu_load_streaming(files)
    results.append(r)
    print(f"  {r['name']}: {r['time_s']*1000:.0f}ms, {r['bandwidth_gbs']:.1f} GB/s")
    print(f"  Tensor count: {r['tensor_count']}")
    direct_load_time = r['time_s']
    direct_load_bw = r['bandwidth_gbs']

    # Test 6: Single file detailed breakdown
    print("\n" + "-" * 80)
    print("TEST 6: Single File Detailed Breakdown")
    print("-" * 80)
    # Use the largest file
    largest_file = max(files, key=lambda f: f.stat().st_size)
    r = bench_single_file_detailed(largest_file)
    print(f"  File: {r['file']} ({r['size_gb']:.2f} GB, {r['tensor_count']} tensors)")
    print(f"  Raw read:     {r['raw_read_time']*1000:6.0f}ms  ({r['raw_read_gbs']:.1f} GB/s)")
    print(f"  CPU load:     {r['cpu_load_time']*1000:6.0f}ms  ({r['cpu_load_gbs']:.1f} GB/s)")
    print(f"  Direct GPU:   {r['gpu_load_time']*1000:6.0f}ms  ({r['gpu_load_gbs']:.1f} GB/s)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Test':<40} {'Time':>10} {'Bandwidth':>12}")
    print("-" * 65)
    for r in results:
        time_str = f"{r.get('time_s', 0)*1000:.0f}ms"
        bw_str = f"{r.get('bandwidth_gbs', 0):.1f} GB/s" if 'bandwidth_gbs' in r else "N/A"
        print(f"{r['name']:<40} {time_str:>10} {bw_str:>12}")

    # Analysis
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)

    print(f"\nRaw PCIe bandwidth: {pcie_bw:.1f} GB/s")
    print(f"Direct GPU load:    {direct_load_bw:.1f} GB/s")

    efficiency = (direct_load_bw / pcie_bw) * 100 if pcie_bw > 0 else 0
    print(f"\nEfficiency: {efficiency:.0f}% of raw PCIe bandwidth")

    # Calculate overhead
    expected_time_at_pcie_bw = total_gb / pcie_bw
    actual_time = direct_load_time
    overhead = actual_time - expected_time_at_pcie_bw

    print(f"\nExpected time at {pcie_bw:.0f} GB/s: {expected_time_at_pcie_bw:.1f}s")
    print(f"Actual time: {actual_time:.1f}s")
    print(f"Overhead: {overhead:.1f}s ({overhead/actual_time*100:.0f}% of total)")

    # Per-tensor analysis
    tensor_count = results[2].get('tensor_count', 0)  # From header parsing
    if tensor_count > 0:
        overhead_per_tensor_ms = (overhead * 1000) / tensor_count
        print(f"\nPer-tensor overhead: {overhead_per_tensor_ms:.2f}ms ({tensor_count} tensors)")
        print(f"  If overhead is per-tensor, batch loading should be {overhead/actual_time*100:.0f}% faster")

    # Bottleneck identification
    print("\n" + "-" * 40)
    if efficiency < 30:
        print("*** BOTTLENECK: Per-tensor overhead ***")
        print("Safetensors/PyTorch overhead dominates PCIe transfer.")
        print("SOLUTION: Batch all tensors into single transfer.")
    elif efficiency < 70:
        print("*** PARTIAL BOTTLENECK: Mixed ***")
        print("Both per-tensor overhead and PCIe bandwidth matter.")
    else:
        print("*** BOTTLENECK: PCIe bandwidth ***")
        print("Already near hardware limit.")

    # Confirmation test
    print("\n" + "=" * 80)
    print("BOTTLENECK CONFIRMATION (2x test)")
    print("=" * 80)

    r = bench_confirm_2x_data(files, direct_load_time)
    print(f"\n  2x data transfer: {r['time_s']*1000:.0f}ms")
    print(f"  Baseline (1x):    {r['baseline_time']*1000:.0f}ms")
    print(f"  Slowdown:         {r['slowdown']:.2f}x")

    if r['is_pcie_bottleneck']:
        print("\n  --> CONFIRMED: PCIe bandwidth is the primary bottleneck")
        print("      (2x data = 2x time)")
    else:
        print(f"\n  --> Slowdown {r['slowdown']:.2f}x suggests:")
        if r['slowdown'] < 1.5:
            print("      Significant per-tensor overhead (not data-proportional)")
        else:
            print("      Mixed: some per-tensor overhead + some PCIe limitation")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print(f"""
Current state:
- Model size: {total_gb:.1f} GB
- Load time:  {direct_load_time:.1f}s
- Bandwidth:  {direct_load_bw:.1f} GB/s (effective)
- PCIe limit: {pcie_bw:.1f} GB/s

To achieve <2s weight loading:

1. SINGLE LARGE DMA TRANSFER:
   Current: {tensor_count} separate tensor transfers
   Target: 1 large contiguous transfer
   Expected time: {total_gb / pcie_bw:.1f}s at {pcie_bw:.0f} GB/s

2. PRE-PACKED WEIGHTS FORMAT:
   - Store model as single mmap'd binary file
   - Single cuda.memcpy from mmap region
   - Skip safetensor parsing entirely

3. BATCH SAFETENSOR LOADING:
   - load_file() loads all tensors at once (already used)
   - But still has per-tensor GPU allocation overhead
   - Need custom loader that pre-allocates GPU buffer
""")


if __name__ == '__main__':
    main()
