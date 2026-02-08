#!/usr/bin/env python3
"""Profile vLLM model load time breakdown.

Measures where time is spent during model loading:
1. Weight file reading
2. Tensor creation/movement
3. Model initialization
4. KV cache allocation
5. Warmup
"""
import os
import sys
import time
import glob
import gc
import subprocess as sp
from pathlib import Path

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # Single-process mode with proper cleanup
os.environ['VLLM_SKIP_WARMUP'] = '1'  # Disable warmup for cleaner measurements

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


def drop_caches():
    """Drop OS page caches."""
    os.system('sync')
    os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1')
    time.sleep(1)


def warm_page_cache_fast(model_path):
    """Warm page cache using parallel dd (fastest method)."""
    import concurrent.futures

    files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))

    def dd_file(path):
        sp.run(['dd', f'if={path}', 'of=/dev/null', 'bs=1M', 'status=none'], check=True)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(dd_file, files)
    return time.time() - t0


def load_model_and_profile(model_name: str, warm_cache: bool = False):
    """Load model and return detailed timing breakdown."""
    from vllm import LLM, SamplingParams

    model_path = snapshot_download(model_name, local_files_only=True)
    model_size = get_model_size(model_path)

    print(f"\nModel: {model_name}")
    print(f"Size: {model_size / 1024**3:.2f} GB")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    # Drop or warm cache
    if warm_cache:
        print("Warming page cache with parallel dd...")
        warm_time = warm_page_cache_fast(model_path)
        print(f"Cache warmed in {warm_time:.2f}s ({model_size/1024**3/warm_time:.1f} GB/s)")
    else:
        print("Dropping page caches...")
        drop_caches()

    # Config - max_model_len set to 100K (largest that fits with 62GB model on 95GB GPU)
    # Note: Qwen3-VL-32B default is 262K but that needs 64GB KV cache alone
    config = {
        "dtype": "bfloat16",
        "max_model_len": 100000,  # 100K tokens - largest that fits
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 8192,
        "enforce_eager": True,
        "trust_remote_code": True,
    }

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    llm = LLM(model=model_name, **config)
    load_time = time.time() - t0

    print(f"Total load time: {load_time:.2f}s")
    print(f"Load speed: {model_size/1024**3/load_time:.2f} GB/s (effective)")
    print(f"GPU free after load: {nvidia_smi_free_gb():.1f} GB")

    # Quick inference test
    print("\nRunning inference test...")
    sampling = SamplingParams(max_tokens=20, temperature=0.7)
    t0 = time.time()
    output = llm.generate(["Hello, I am a"], sampling)
    inference_time = time.time() - t0
    print(f"Inference: {output[0].outputs[0].text[:40]}...")
    print(f"First inference time: {inference_time:.2f}s")

    # Cleanup
    print("\nCleaning up...")
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        print(f"Shutdown warning: {e}")
    time.sleep(1)
    del llm
    gc.collect()
    time.sleep(2)
    print(f"GPU free after cleanup: {nvidia_smi_free_gb():.1f} GB")

    return {
        'model': model_name,
        'size_gb': model_size / 1024**3,
        'load_time': load_time,
        'effective_speed': model_size / 1024**3 / load_time,
        'warm_cache': warm_cache,
    }


def main():
    print("=" * 70)
    print("VLLM MODEL LOAD TIME PROFILING")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-32B-Instruct"

    results = []

    # Test 1: Cold load
    print("\n" + "=" * 70)
    print("TEST 1: COLD LOAD (from disk)")
    print("=" * 70)
    r1 = load_model_and_profile(model, warm_cache=False)
    results.append(r1)

    # Wait for cleanup
    time.sleep(5)

    # Test 2: Warm load
    print("\n" + "=" * 70)
    print("TEST 2: WARM LOAD (from page cache)")
    print("=" * 70)
    r2 = load_model_and_profile(model, warm_cache=True)
    results.append(r2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model size:         {r1['size_gb']:.1f} GB")
    print(f"Cold load time:     {r1['load_time']:.1f}s")
    print(f"Warm load time:     {r2['load_time']:.1f}s")
    print(f"Cold effective:     {r1['effective_speed']:.2f} GB/s")
    print(f"Warm effective:     {r2['effective_speed']:.2f} GB/s")
    print(f"Speedup:            {r1['load_time']/r2['load_time']:.2f}x")
    print(f"Time saved:         {r1['load_time']-r2['load_time']:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
