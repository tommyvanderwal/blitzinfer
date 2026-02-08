#!/usr/bin/env python3
"""Granular profiling of vLLM internal startup phases."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import sys
import types

# Patch torchvision._meta_registrations before import
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta

sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import time
import gc
from functools import wraps

# Global timing log
timing_log = []


def timed(name):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            print(f"  [{time.strftime('%H:%M:%S')}] START: {name}")
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                timing_log.append((name, elapsed))
                print(f"  [{time.strftime('%H:%M:%S')}] END:   {name} = {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                timing_log.append((f"{name} (FAILED)", elapsed))
                print(f"  [{time.strftime('%H:%M:%S')}] FAIL:  {name} = {elapsed:.3f}s - {e}")
                raise
        return wrapper
    return decorator


def install_hooks():
    """Install timing hooks into vLLM internals."""
    print("Installing timing hooks into vLLM internals...")

    # Hook into EngineCore._initialize_kv_caches
    from vllm.v1.engine import core as engine_core
    original_init_kv = engine_core.EngineCore._initialize_kv_caches

    @timed("EngineCore._initialize_kv_caches")
    def timed_init_kv(self, vllm_config):
        return original_init_kv(self, vllm_config)

    engine_core.EngineCore._initialize_kv_caches = timed_init_kv

    # Hook into Worker methods (v1 uses Worker class, not GPUWorker)
    from vllm.v1.worker import gpu_worker

    original_init_device = gpu_worker.Worker.init_device
    @timed("Worker.init_device")
    def timed_init_device(self):
        return original_init_device(self)
    gpu_worker.Worker.init_device = timed_init_device

    original_load_model = gpu_worker.Worker.load_model
    @timed("Worker.load_model")
    def timed_load_model(self):
        return original_load_model(self)
    gpu_worker.Worker.load_model = timed_load_model

    original_determine_memory = gpu_worker.Worker.determine_available_memory
    @timed("Worker.determine_available_memory")
    def timed_determine_memory(self):
        return original_determine_memory(self)
    gpu_worker.Worker.determine_available_memory = timed_determine_memory

    original_compile_warmup = gpu_worker.Worker.compile_or_warm_up_model
    @timed("Worker.compile_or_warm_up_model")
    def timed_compile_warmup(self):
        return original_compile_warmup(self)
    gpu_worker.Worker.compile_or_warm_up_model = timed_compile_warmup

    # Hook into GPUModelRunner methods
    from vllm.v1.worker.gpu import model_runner

    original_profile_run = model_runner.GPUModelRunner.profile_run
    @timed("GPUModelRunner.profile_run")
    def timed_profile_run(self):
        return original_profile_run(self)
    model_runner.GPUModelRunner.profile_run = timed_profile_run

    # Track _dummy_run calls
    original_dummy_run = model_runner.GPUModelRunner._dummy_run
    dummy_run_count = [0]

    def timed_dummy_run(self, num_tokens, **kwargs):
        dummy_run_count[0] += 1
        call_num = dummy_run_count[0]
        name = f"GPUModelRunner._dummy_run #{call_num} (tokens={num_tokens})"
        start = time.perf_counter()
        print(f"    [{time.strftime('%H:%M:%S')}] START: {name}")
        result = original_dummy_run(self, num_tokens, **kwargs)
        elapsed = time.perf_counter() - start
        timing_log.append((name, elapsed))
        print(f"    [{time.strftime('%H:%M:%S')}] END:   {name} = {elapsed:.3f}s")
        return result
    model_runner.GPUModelRunner._dummy_run = timed_dummy_run

    # Hook into kernel warmup
    from vllm.model_executor.warmup import kernel_warmup as kw_module
    original_kernel_warmup = kw_module.kernel_warmup

    @timed("kernel_warmup")
    def timed_kernel_warmup(worker):
        return original_kernel_warmup(worker)
    kw_module.kernel_warmup = timed_kernel_warmup

    # Hook into model loading (may not be directly importable in all versions)
    try:
        from vllm.model_executor.model_loader import loader
        if hasattr(loader, 'load_model'):
            original_load = loader.load_model
            @timed("loader.load_model")
            def timed_load(*args, **kwargs):
                return original_load(*args, **kwargs)
            loader.load_model = timed_load
    except ImportError:
        print("  (Could not hook model loader)")

    print("Hooks installed.")


def profile_startup():
    """Profile startup with internal timing."""
    print("\n" + "="*70)
    print("GRANULAR INTERNAL STARTUP PROFILING")
    print("="*70)

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    torch.cuda.empty_cache()
    gc.collect()

    # Install hooks before importing LLM
    install_hooks()

    from vllm import LLM, SamplingParams

    print(f"\n[{time.strftime('%H:%M:%S')}] Starting LLM initialization...")
    total_start = time.perf_counter()

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        max_model_len=1024,
        enforce_eager=True,
    )

    total_time = time.perf_counter() - total_start
    print(f"\n[{time.strftime('%H:%M:%S')}] LLM initialization complete: {total_time:.2f}s")

    # First inference
    print(f"\n[{time.strftime('%H:%M:%S')}] First inference...")
    start = time.perf_counter()
    out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    first_inf = time.perf_counter() - start
    print(f"  First inference: {first_inf:.2f}s")
    print(f"  Output: {out[0].outputs[0].text}")

    # Summary
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"{'Phase':<50} {'Time':>10}")
    print("-"*70)

    # Sort by start time (order they were logged)
    for phase, elapsed in timing_log:
        print(f"{phase:<50} {elapsed:>10.3f}s")

    print("-"*70)
    print(f"{'Total LLM init':<50} {total_time:>10.3f}s")
    print(f"{'First inference':<50} {first_inf:>10.3f}s")
    print("="*70)

    # Identify gaps
    print("\nGAP ANALYSIS:")
    accounted = sum(t for _, t in timing_log if "dummy_run" not in _)
    unaccounted = total_time - accounted
    print(f"  Accounted time: {accounted:.2f}s")
    print(f"  Unaccounted time: {unaccounted:.2f}s (multiprocessing overhead, misc)")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def profile_with_kv_cache_bytes():
    """Profile with kv_cache_memory_bytes to skip memory profiling."""
    print("\n" + "="*70)
    print("PROFILING WITH kv_cache_memory_bytes (SKIP MEMORY PROFILING)")
    print("="*70)

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    # Clear timing log
    timing_log.clear()

    install_hooks()

    from vllm import LLM, SamplingParams

    # 4GB KV cache
    kv_bytes = 4 * 1024 * 1024 * 1024

    print(f"\n[{time.strftime('%H:%M:%S')}] Starting LLM initialization with kv_cache_memory_bytes...")
    total_start = time.perf_counter()

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        kv_cache_memory_bytes=kv_bytes,
        max_model_len=1024,
        enforce_eager=True,
    )

    total_time = time.perf_counter() - total_start
    print(f"\n[{time.strftime('%H:%M:%S')}] LLM initialization complete: {total_time:.2f}s")

    # Summary
    print("\n" + "="*70)
    print("TIMING SUMMARY (with kv_cache_memory_bytes)")
    print("="*70)
    for phase, elapsed in timing_log:
        print(f"{phase:<50} {elapsed:>10.3f}s")
    print(f"{'Total':<50} {total_time:>10.3f}s")

    del llm
    gc.collect()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "kv":
            profile_with_kv_cache_bytes()
        else:
            profile_startup()
    else:
        print("Usage: python3.12 profile_internals.py [standard|kv]")
        print("  standard - Profile standard startup")
        print("  kv       - Profile with kv_cache_memory_bytes")
        print("\nRunning standard profile...")
        profile_startup()
