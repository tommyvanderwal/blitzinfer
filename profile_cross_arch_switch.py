#!/usr/bin/env python3
"""
Profile cross-architecture model switching to find optimization opportunities.

Target: Get ~6s down to ~2s
Current breakdown unknown - this will reveal where time goes.
"""

import gc
import os
import sys
import time
import types
from contextlib import contextmanager
from typing import Dict, List, Tuple

# ROCm setup
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# vLLM import workaround
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch


class SwitchProfiler:
    """Detailed profiler for model switching."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.current_section = None
        self.section_start = None

    @contextmanager
    def section(self, name: str):
        """Time a section of code."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)

    def report(self):
        """Print timing report."""
        print("\n" + "="*70)
        print("PROFILING RESULTS")
        print("="*70)

        total = 0
        for name, times in self.timings.items():
            avg = sum(times) / len(times)
            total += avg
            print(f"  {name}: {avg:.0f}ms")

        print(f"\n  TOTAL: {total:.0f}ms")
        print("="*70)


def get_mem():
    """Get GPU memory info."""
    free, total = torch.cuda.mem_get_info()
    return free / (1024**3), total / (1024**3)


def profile_switch():
    """Profile a complete model switch."""
    from vllm import LLM, SamplingParams

    profiler = SwitchProfiler()

    qwen_model = "Qwen/Qwen2.5-7B-Instruct"
    mistral_model = "mistralai/Mistral-7B-Instruct-v0.3"

    # Conservative config
    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.35,  # ~35GB max for model+KV
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    free_mem, total_mem = get_mem()
    print(f"Initial: {free_mem:.1f} GB free / {total_mem:.1f} GB total")

    # Load first model (Qwen)
    print(f"\n>>> Loading {qwen_model}...")
    with profiler.section("1. Initial load (Qwen)"):
        llm = LLM(model=qwen_model, **config)

    free_mem, _ = get_mem()
    print(f"After Qwen load: {free_mem:.1f} GB free")

    # Test inference
    params = SamplingParams(max_tokens=20, temperature=0.7)
    outputs = llm.generate(["Hello"], params)
    print(f"Qwen output: {outputs[0].outputs[0].text[:30]}...")

    # === PROFILE THE SWITCH ===
    print(f"\n>>> Profiling switch to {mistral_model}...")

    # Phase 1: Cleanup
    with profiler.section("2.1 Delete LLM object"):
        del llm

    with profiler.section("2.2 gc.collect()"):
        gc.collect()

    with profiler.section("2.3 torch.cuda.synchronize()"):
        torch.cuda.synchronize()

    with profiler.section("2.4 torch.cuda.empty_cache()"):
        torch.cuda.empty_cache()

    with profiler.section("2.5 Second gc.collect()"):
        gc.collect()

    free_mem, _ = get_mem()
    print(f"After cleanup: {free_mem:.1f} GB free")

    # Phase 2: Load new model - break down LLM() constructor
    # We need to dig into vLLM internals to profile this

    print("\n>>> Profiling Mistral load components...")

    # Profile tokenizer loading
    with profiler.section("3.1 Load tokenizer"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(mistral_model, trust_remote_code=True)
        del tokenizer

    # Profile config loading
    with profiler.section("3.2 Load config"):
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(mistral_model, trust_remote_code=True)
        del model_config

    # Profile full LLM load
    with profiler.section("3.3 Full LLM() constructor"):
        llm = LLM(model=mistral_model, **config)

    free_mem, _ = get_mem()
    print(f"After Mistral load: {free_mem:.1f} GB free")

    # Test inference
    outputs = llm.generate(["Hello"], params)
    print(f"Mistral output: {outputs[0].outputs[0].text[:30]}...")

    # Clean up
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    profiler.report()


def profile_vllm_internals():
    """Deep profile of vLLM LLM() constructor."""
    print("\n" + "="*70)
    print("DEEP PROFILE: vLLM LLM() Constructor")
    print("="*70)

    import functools
    import vllm
    from vllm import LLM

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.35,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Monkey-patch key functions to add timing
    timings = {}

    def timed(name):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                if name not in timings:
                    timings[name] = 0
                timings[name] += elapsed
                return result
            return wrapper
        return decorator

    # Patch key vLLM components
    original_funcs = {}

    # Weight loading
    try:
        import vllm.model_executor.model_loader.weight_utils as weight_utils
        original_funcs['safetensors_weights_iterator'] = weight_utils.safetensors_weights_iterator
        weight_utils.safetensors_weights_iterator = timed('weight_loading')(weight_utils.safetensors_weights_iterator)
    except:
        pass

    # Model creation
    try:
        import vllm.model_executor.model_loader.default_loader as default_loader
        if hasattr(default_loader, 'DefaultModelLoader'):
            original_load = default_loader.DefaultModelLoader.load_model
            default_loader.DefaultModelLoader.load_model = timed('model_loader.load_model')(original_load)
            original_funcs['load_model'] = original_load
    except:
        pass

    print(f"\nLoading {model_name}...")
    t0 = time.perf_counter()
    llm = LLM(model=model_name, **config)
    total_time = (time.perf_counter() - t0) * 1000

    print(f"\nTotal LLM() time: {total_time:.0f}ms")
    print("\nComponent breakdown:")
    for name, elapsed in sorted(timings.items(), key=lambda x: -x[1]):
        pct = elapsed / total_time * 100
        print(f"  {name}: {elapsed:.0f}ms ({pct:.1f}%)")

    # Restore original functions
    for name, func in original_funcs.items():
        if name == 'safetensors_weights_iterator':
            weight_utils.safetensors_weights_iterator = func
        elif name == 'load_model':
            default_loader.DefaultModelLoader.load_model = func

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def profile_weight_loading_options():
    """Compare different weight loading approaches."""
    print("\n" + "="*70)
    print("WEIGHT LOADING OPTIONS COMPARISON")
    print("="*70)

    from pathlib import Path
    from safetensors import safe_open
    from huggingface_hub import snapshot_download

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    # Get model path
    local_path = snapshot_download(model_name)
    safetensor_files = sorted(Path(local_path).glob("*.safetensors"))

    print(f"Model: {model_name}")
    print(f"Files: {len(safetensor_files)}")

    # Get total size
    total_size = sum(f.stat().st_size for f in safetensor_files)
    print(f"Total size: {total_size / (1024**3):.2f} GB")

    # Option 1: Standard safetensors loading (what vLLM does)
    print("\n>>> Option 1: Standard safetensors (lazy load to CPU, copy to GPU)")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    tensors = {}
    for sf in safetensor_files:
        with safe_open(str(sf), framework='pt', device='cpu') as f:
            for key in f.keys():
                t = f.get_tensor(key)
                tensors[key] = t.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    standard_time = (time.perf_counter() - t0) * 1000
    bandwidth1 = (total_size / (1024**3)) / (standard_time / 1000)
    print(f"  Time: {standard_time:.0f}ms ({bandwidth1:.1f} GB/s)")

    del tensors
    gc.collect()
    torch.cuda.empty_cache()

    # Option 2: Direct GPU loading
    print("\n>>> Option 2: Direct GPU load (safetensors device='cuda')")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    tensors = {}
    for sf in safetensor_files:
        with safe_open(str(sf), framework='pt', device='cuda') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    torch.cuda.synchronize()
    direct_time = (time.perf_counter() - t0) * 1000
    bandwidth2 = (total_size / (1024**3)) / (direct_time / 1000)
    print(f"  Time: {direct_time:.0f}ms ({bandwidth2:.1f} GB/s)")

    del tensors
    gc.collect()
    torch.cuda.empty_cache()

    # Option 3: Pinned memory bulk transfer
    print("\n>>> Option 3: Pinned memory + async bulk transfer")
    torch.cuda.synchronize()

    # Pre-allocate pinned buffer
    max_file_size = max(f.stat().st_size for f in safetensor_files)
    max_elements = max_file_size // 2 + 1000000  # bf16 = 2 bytes
    cpu_buffer = torch.empty(max_elements, dtype=torch.bfloat16, pin_memory=True)
    gpu_buffer = torch.empty(max_elements, dtype=torch.bfloat16, device='cuda')

    t0 = time.perf_counter()
    weight_data = {}

    for sf in safetensor_files:
        buf_offset = 0
        with safe_open(str(sf), framework='pt', device='cpu') as f:
            for key in f.keys():
                t = f.get_tensor(key)
                flat = t.reshape(-1)
                n = flat.numel()

                if flat.dtype == torch.bfloat16:
                    cpu_buffer[buf_offset:buf_offset+n].copy_(flat)
                else:
                    cpu_buffer[buf_offset:buf_offset+n].copy_(flat.view(torch.bfloat16) if flat.dtype == torch.float16 else flat.to(torch.bfloat16))

                weight_data[key] = (buf_offset, n, t.shape, t.dtype)
                buf_offset += n

        # Bulk transfer
        gpu_buffer[:buf_offset].copy_(cpu_buffer[:buf_offset], non_blocking=True)

    torch.cuda.synchronize()
    pinned_time = (time.perf_counter() - t0) * 1000
    bandwidth3 = (total_size / (1024**3)) / (pinned_time / 1000)
    print(f"  Time: {pinned_time:.0f}ms ({bandwidth3:.1f} GB/s)")

    del cpu_buffer, gpu_buffer, weight_data
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "-"*70)
    print("SUMMARY")
    print("-"*70)
    print(f"  Standard (CPU→GPU):     {standard_time:.0f}ms ({bandwidth1:.1f} GB/s)")
    print(f"  Direct GPU:             {direct_time:.0f}ms ({bandwidth2:.1f} GB/s)")
    print(f"  Pinned bulk:            {pinned_time:.0f}ms ({bandwidth3:.1f} GB/s)")
    print(f"\n  Best approach: {'Direct GPU' if direct_time < min(standard_time, pinned_time) else 'Pinned bulk' if pinned_time < standard_time else 'Standard'}")


def main():
    print("="*70)
    print("CROSS-ARCHITECTURE SWITCH PROFILER")
    print("Target: 6s → 2s")
    print("="*70)

    # First, profile the overall switch
    profile_switch()

    # Then, profile weight loading options
    profile_weight_loading_options()

    # Deep profile vLLM internals
    profile_vllm_internals()


if __name__ == '__main__':
    main()
