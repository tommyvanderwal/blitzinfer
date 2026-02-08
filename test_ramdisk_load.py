#!/usr/bin/env python3
"""Test loading model from RAM disk (tmpfs) vs regular disk.

This tests if we can get faster loading by putting weights in RAM.
"""
import os
import sys
import time
import shutil
import glob
import gc
import subprocess as sp
from pathlib import Path

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # Single-process mode with proper cleanup
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch
from huggingface_hub import snapshot_download


def nvidia_smi_free_gb():
    result = sp.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def get_model_size(model_path):
    """Get total size of safetensor files."""
    files = glob.glob(str(Path(model_path) / "*.safetensors"))
    return sum(os.path.getsize(f) for f in files)


def setup_ramdisk(size_gb: int = 70):
    """Create a tmpfs RAM disk."""
    ramdisk_path = "/tmp/blitz_ramdisk"

    # Check if already mounted
    result = sp.run(['mountpoint', '-q', ramdisk_path], capture_output=True)
    if result.returncode == 0:
        print(f"RAM disk already mounted at {ramdisk_path}")
        return ramdisk_path

    # Create and mount
    os.makedirs(ramdisk_path, exist_ok=True)
    result = sp.run(
        ['sudo', 'mount', '-t', 'tmpfs', '-o', f'size={size_gb}G', 'tmpfs', ramdisk_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Failed to mount RAM disk: {result.stderr}")
        return None

    print(f"Created {size_gb}GB RAM disk at {ramdisk_path}")
    return ramdisk_path


def copy_model_to_ramdisk(model_path: str, ramdisk_path: str):
    """Copy model files to RAM disk."""
    model_name = Path(model_path).name
    dest_path = Path(ramdisk_path) / model_name

    if dest_path.exists():
        print(f"Model already in RAM disk: {dest_path}")
        return str(dest_path)

    print(f"Copying model to RAM disk...")
    t0 = time.time()

    # Create destination
    dest_path.mkdir(parents=True, exist_ok=True)

    # Copy all files
    total_size = 0
    for f in Path(model_path).iterdir():
        if f.is_file():
            shutil.copy2(f, dest_path / f.name)
            total_size += f.stat().st_size

    elapsed = time.time() - t0
    speed = total_size / 1024**3 / elapsed
    print(f"Copied {total_size/1024**3:.2f} GB in {elapsed:.2f}s ({speed:.2f} GB/s)")

    return str(dest_path)


def load_model(model_path: str, label: str):
    """Load model and return timing."""
    from vllm import LLM, SamplingParams

    print(f"\n--- {label} ---")
    print(f"Path: {model_path}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    config = {
        "dtype": "bfloat16",
        "max_model_len": 100000,
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 8192,
        "enforce_eager": True,
        "trust_remote_code": True,
    }

    print("Loading model...")
    t0 = time.time()
    llm = LLM(model=model_path, **config)
    load_time = time.time() - t0

    model_size = get_model_size(model_path)
    print(f"Load time: {load_time:.2f}s")
    print(f"Effective speed: {model_size/1024**3/load_time:.2f} GB/s")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    # Quick inference
    sampling = SamplingParams(max_tokens=20, temperature=0.7)
    t0 = time.time()
    output = llm.generate(["Hello, I am"], sampling)
    inf_time = time.time() - t0
    print(f"Inference: {output[0].outputs[0].text[:40]}...")
    print(f"First inference: {inf_time:.2f}s")

    # Cleanup
    try:
        llm.llm_engine.engine_core.shutdown()
    except:
        pass
    time.sleep(1)
    del llm
    gc.collect()
    time.sleep(2)
    print(f"GPU free after cleanup: {nvidia_smi_free_gb():.1f} GB")

    return load_time


def main():
    print("=" * 70)
    print("RAM DISK vs REGULAR DISK LOADING TEST")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-32B-Instruct"
    print(f"Model: {model_name}")

    # Get original model path
    model_path = snapshot_download(model_name, local_files_only=True)
    model_size = get_model_size(model_path)
    print(f"Model size: {model_size/1024**3:.2f} GB")

    # Setup RAM disk
    print("\n" + "=" * 70)
    print("SETTING UP RAM DISK")
    print("=" * 70)
    ramdisk = setup_ramdisk(size_gb=70)
    if not ramdisk:
        print("Failed to create RAM disk. Exiting.")
        return

    # Copy model to RAM disk
    ramdisk_model_path = copy_model_to_ramdisk(model_path, ramdisk)

    results = {}

    # Test 1: Load from regular disk (cold)
    print("\n" + "=" * 70)
    print("TEST 1: LOAD FROM REGULAR DISK (cold)")
    print("=" * 70)
    os.system('sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null')
    time.sleep(1)
    results['disk_cold'] = load_model(model_path, "Regular disk (cold)")

    time.sleep(5)

    # Test 2: Load from regular disk (warm - page cache)
    print("\n" + "=" * 70)
    print("TEST 2: LOAD FROM REGULAR DISK (warm/page cache)")
    print("=" * 70)
    # Warm the page cache with parallel dd
    import concurrent.futures
    files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    def dd_file(f):
        sp.run(['dd', f'if={f}', 'of=/dev/null', 'bs=1M', 'status=none'])
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        ex.map(dd_file, files)
    print(f"Page cache warmed in {time.time()-t0:.2f}s")
    results['disk_warm'] = load_model(model_path, "Regular disk (warm)")

    time.sleep(5)

    # Test 3: Load from RAM disk
    print("\n" + "=" * 70)
    print("TEST 3: LOAD FROM RAM DISK (tmpfs)")
    print("=" * 70)
    os.system('sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null')
    time.sleep(1)
    results['ramdisk'] = load_model(ramdisk_model_path, "RAM disk (tmpfs)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model size:        {model_size/1024**3:.1f} GB")
    print(f"Disk (cold):       {results['disk_cold']:.1f}s")
    print(f"Disk (warm):       {results['disk_warm']:.1f}s")
    print(f"RAM disk:          {results['ramdisk']:.1f}s")
    print()
    print(f"RAM disk speedup over cold: {results['disk_cold']/results['ramdisk']:.2f}x")
    print(f"RAM disk speedup over warm: {results['disk_warm']/results['ramdisk']:.2f}x")
    print("=" * 70)


if __name__ == '__main__':
    main()
