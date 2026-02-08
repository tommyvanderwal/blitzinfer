#!/usr/bin/env python3
"""
Warm tier model pre-loading: Cache models in CPU RAM for fast GPU loading.

Strategy:
1. Pre-load model weights to CPU memory (warm tier)
2. When switching, transfer from RAM to GPU (fast)
3. Skip disk I/O entirely during switch

Expected improvement: 6.5s → ~2-3s (skip disk I/O, just RAM→GPU transfer)
"""

import os
import sys
import gc
import time
import types
import threading
from pathlib import Path

os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch
import torch.nn as nn
from safetensors.torch import load_file


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def get_ram():
    """Get free RAM in GB."""
    import psutil
    return psutil.virtual_memory().available / (1024**3)


def blazing_cleanup():
    """Fast cleanup: unfreeze + clear params + empty cache."""
    gc.unfreeze()
    for obj in gc.get_objects():
        try:
            if isinstance(obj, nn.Module):
                if hasattr(obj, '_parameters') and obj._parameters:
                    for key in list(obj._parameters.keys()):
                        param = obj._parameters.get(key)
                        if param is not None and param.is_cuda:
                            obj._parameters[key] = None
        except:
            pass
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


class WarmModelCache:
    """Cache models in CPU RAM for fast switching."""

    def __init__(self, max_models=2):
        self.max_models = max_models
        self.cache = {}  # model_name -> {weights: dict, config: dict}
        self.load_times = {}

    def get_model_path(self, model_name):
        """Get local path for a model."""
        from huggingface_hub import snapshot_download
        return snapshot_download(model_name)

    def preload_to_ram(self, model_name):
        """Load model weights to CPU RAM."""
        print(f"Pre-loading {model_name} to RAM...")
        start = time.perf_counter()

        model_path = self.get_model_path(model_name)
        safetensor_files = list(Path(model_path).glob("*.safetensors"))

        weights = {}
        for sf in safetensor_files:
            print(f"  Loading {sf.name}...")
            w = load_file(str(sf), device="cpu")
            weights.update(w)

        load_time = time.perf_counter() - start
        total_gb = sum(t.numel() * t.element_size() for t in weights.values()) / (1024**3)

        self.cache[model_name] = {
            'weights': weights,
            'path': model_path,
        }
        self.load_times[model_name] = load_time

        print(f"  Pre-loaded {len(weights)} tensors ({total_gb:.2f} GB) in {load_time:.2f}s")
        return weights

    def is_cached(self, model_name):
        return model_name in self.cache

    def get_weights(self, model_name):
        if model_name not in self.cache:
            self.preload_to_ram(model_name)
        return self.cache[model_name]['weights']


def measure_weight_loading():
    """Measure how much time is spent loading weights from disk vs RAM."""
    print("=" * 70)
    print("MEASURING WEIGHT LOADING TIME")
    print("=" * 70)

    from huggingface_hub import snapshot_download

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_path = snapshot_download(model_name)
    safetensor_files = list(Path(model_path).glob("*.safetensors"))

    print(f"\nModel: {model_name}")
    print(f"Files: {[f.name for f in safetensor_files]}")

    # Method 1: Load from disk to CPU
    print("\n>>> Method 1: Disk → CPU")
    gc.collect()
    t0 = time.perf_counter()
    cpu_weights = {}
    for sf in safetensor_files:
        w = load_file(str(sf), device="cpu")
        cpu_weights.update(w)
    disk_to_cpu = (time.perf_counter() - t0) * 1000
    total_gb = sum(t.numel() * t.element_size() for t in cpu_weights.values()) / (1024**3)
    print(f"  Time: {disk_to_cpu:.0f}ms for {total_gb:.2f} GB")
    print(f"  Bandwidth: {total_gb / (disk_to_cpu / 1000):.2f} GB/s")

    # Method 2: Load from disk directly to GPU
    print("\n>>> Method 2: Disk → GPU")
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    gpu_weights = {}
    for sf in safetensor_files:
        w = load_file(str(sf), device="cuda")
        gpu_weights.update(w)
    disk_to_gpu = (time.perf_counter() - t0) * 1000
    print(f"  Time: {disk_to_gpu:.0f}ms for {total_gb:.2f} GB")
    print(f"  Bandwidth: {total_gb / (disk_to_gpu / 1000):.2f} GB/s")

    # Clear GPU
    del gpu_weights
    gc.collect()
    torch.cuda.empty_cache()

    # Method 3: CPU → GPU transfer
    print("\n>>> Method 3: CPU → GPU transfer only")
    t0 = time.perf_counter()
    gpu_weights = {}
    for key, tensor in cpu_weights.items():
        gpu_weights[key] = tensor.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    cpu_to_gpu = (time.perf_counter() - t0) * 1000
    print(f"  Time: {cpu_to_gpu:.0f}ms for {total_gb:.2f} GB")
    print(f"  Bandwidth: {total_gb / (cpu_to_gpu / 1000):.2f} GB/s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Disk → CPU:  {disk_to_cpu:.0f}ms")
    print(f"Disk → GPU:  {disk_to_gpu:.0f}ms")
    print(f"CPU → GPU:   {cpu_to_gpu:.0f}ms")
    print(f"\nSavings if pre-cached: {disk_to_gpu - cpu_to_gpu:.0f}ms ({(disk_to_gpu - cpu_to_gpu) / disk_to_gpu * 100:.0f}%)")

    # Cleanup
    del cpu_weights
    del gpu_weights
    gc.collect()
    torch.cuda.empty_cache()


def test_warm_switching():
    """Test model switching with warm cache."""
    print("\n" + "=" * 70)
    print("WARM CACHE MODEL SWITCHING TEST")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    cache = WarmModelCache()
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Pre-load to RAM
    print(f"\n>>> Pre-loading {model_name} to RAM...")
    ram_before = get_ram()
    cache.preload_to_ram(model_name)
    ram_after = get_ram()
    print(f"RAM used: {ram_before - ram_after:.2f} GB")

    # First model load (cold - vLLM will still read from disk)
    print("\n>>> Loading first model (cold - vLLM reads from disk)...")
    initial = get_mem()
    print(f"Initial GPU: {initial:.2f} GB")

    t0 = time.perf_counter()
    llm = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.30,
        max_model_len=512,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=2 * 1024**3,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )
    cold_load = (time.perf_counter() - t0) * 1000
    print(f"Cold load: {cold_load:.0f}ms")

    out = llm.generate(["Hi"], SamplingParams(max_tokens=3))
    print(f"Output: {out[0].outputs[0].text}")
    print(f"GPU: {get_mem():.2f} GB")

    # Cleanup
    print("\n>>> Cleanup...")
    del llm
    del out
    t0 = time.perf_counter()
    blazing_cleanup()
    cleanup_time = (time.perf_counter() - t0) * 1000
    print(f"Cleanup: {cleanup_time:.0f}ms")
    print(f"GPU after cleanup: {get_mem():.2f} GB")

    # Second load (should benefit from OS page cache at least)
    print("\n>>> Loading second model (warm - should be faster)...")
    t0 = time.perf_counter()
    llm2 = LLM(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=0.30,
        max_model_len=512,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=2 * 1024**3,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )
    warm_load = (time.perf_counter() - t0) * 1000
    print(f"Warm load: {warm_load:.0f}ms")

    out2 = llm2.generate(["Test"], SamplingParams(max_tokens=3))
    print(f"Output: {out2[0].outputs[0].text}")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Cold load:  {cold_load:.0f}ms")
    print(f"Cleanup:    {cleanup_time:.0f}ms")
    print(f"Warm load:  {warm_load:.0f}ms")
    print(f"TOTAL SWITCH: {cleanup_time + warm_load:.0f}ms")

    # Cleanup
    del llm2
    del out2
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    measure_weight_loading()
    test_warm_switching()
