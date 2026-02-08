#!/usr/bin/env python3
"""Test vLLM multi-threaded weight loading.

vLLM supports multi-threaded loading via model_loader_extra_config.
"""
import os
import sys
import time
import gc
import glob
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
    files = glob.glob(str(Path(model_path) / "*.safetensors"))
    return sum(os.path.getsize(f) for f in files)


def warm_page_cache(model_path):
    """Warm page cache using parallel dd."""
    import concurrent.futures
    files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    def dd_file(f):
        sp.run(['dd', f'if={f}', 'of=/dev/null', 'bs=1M', 'status=none'])
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        ex.map(dd_file, files)
    return time.time() - t0


def load_model(model_name: str, enable_multithread: bool, num_threads: int = 8):
    """Load model with optional multi-threading."""
    from vllm import LLM, SamplingParams

    model_path = snapshot_download(model_name, local_files_only=True)

    config = {
        "dtype": "bfloat16",
        "max_model_len": 100000,
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 8192,
        "enforce_eager": True,
        "trust_remote_code": True,
    }

    if enable_multithread:
        config["model_loader_extra_config"] = {
            "enable_multithread_load": True,
            "num_threads": num_threads,
        }

    label = f"multithread ({num_threads} threads)" if enable_multithread else "single-thread"
    print(f"\n--- {label} ---")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    print("Loading model...")
    t0 = time.time()
    llm = LLM(model=model_name, **config)
    load_time = time.time() - t0

    model_size = get_model_size(model_path)
    print(f"Load time: {load_time:.2f}s")
    print(f"Effective speed: {model_size/1024**3/load_time:.2f} GB/s")

    # Quick inference
    sampling = SamplingParams(max_tokens=20, temperature=0.7)
    output = llm.generate(["Hello"], sampling)
    print(f"Inference: {output[0].outputs[0].text[:40]}...")

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
    print("MULTI-THREADED WEIGHT LOADING TEST")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")

    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-32B-Instruct"
    print(f"Model: {model_name}")

    model_path = snapshot_download(model_name, local_files_only=True)
    model_size = get_model_size(model_path)
    print(f"Model size: {model_size/1024**3:.2f} GB")

    results = {}

    # Warm page cache first
    print("\nWarming page cache...")
    warm_time = warm_page_cache(model_path)
    print(f"Page cache warmed in {warm_time:.2f}s")

    # Test 1: Single-thread (default)
    print("\n" + "=" * 70)
    print("TEST 1: SINGLE-THREAD LOADING (default)")
    print("=" * 70)
    results['single_thread'] = load_model(model_name, enable_multithread=False)

    time.sleep(5)

    # Re-warm page cache
    print("\nRe-warming page cache...")
    warm_page_cache(model_path)

    # Test 2: Multi-thread (8 threads)
    print("\n" + "=" * 70)
    print("TEST 2: MULTI-THREAD LOADING (8 threads)")
    print("=" * 70)
    results['multithread_8'] = load_model(model_name, enable_multithread=True, num_threads=8)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, time_s in sorted(results.items(), key=lambda x: x[1]):
        speed = model_size / 1024**3 / time_s
        print(f"  {name:20s}: {time_s:.1f}s ({speed:.2f} GB/s)")
    print("=" * 70)


if __name__ == '__main__':
    main()
