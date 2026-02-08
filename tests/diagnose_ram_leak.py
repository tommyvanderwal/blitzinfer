#!/usr/bin/env python3
"""Diagnose RAM leak during model switching.

This script tracks RAM usage at every step of the cleanup and load process
to find the exact moment when 19GB of RAM leaks.

Key hypothesis: During full_cleanup(), PyTorch may be allocating CPU memory
that isn't returned to the OS, OR the standby manager prefetch is creating
copies that accumulate.
"""

import os
import sys
import gc
import time
import ctypes

# Set single-process mode BEFORE importing vLLM
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

import torch

# Add blitzinfer to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_memory_info():
    """Get comprehensive memory state."""
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    gpu_allocated = torch.cuda.memory_allocated()
    gpu_reserved = torch.cuda.memory_reserved()

    # System RAM via /proc/meminfo
    with open('/proc/meminfo', 'r') as f:
        meminfo = {}
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                meminfo[parts[0].rstrip(':')] = int(parts[1]) * 1024  # KB to bytes

    ram_total = meminfo.get('MemTotal', 0)
    ram_free = meminfo.get('MemFree', 0)
    ram_available = meminfo.get('MemAvailable', 0)
    ram_shared = meminfo.get('Shmem', 0)
    ram_buffers = meminfo.get('Buffers', 0)
    ram_cached = meminfo.get('Cached', 0)

    return {
        'gpu_used_gb': (gpu_total - gpu_free) / 1024**3,
        'gpu_free_gb': gpu_free / 1024**3,
        'gpu_total_gb': gpu_total / 1024**3,
        'gpu_alloc_gb': gpu_allocated / 1024**3,
        'gpu_rsv_gb': gpu_reserved / 1024**3,
        'ram_total_gb': ram_total / 1024**3,
        'ram_free_gb': ram_free / 1024**3,
        'ram_avail_gb': ram_available / 1024**3,
        'ram_shared_gb': ram_shared / 1024**3,
        'ram_buffers_gb': ram_buffers / 1024**3,
        'ram_cached_gb': ram_cached / 1024**3,
        'ram_used_gb': (ram_total - ram_available) / 1024**3,
    }


def log_mem(label: str, baseline_avail=None):
    """Log memory state with optional delta from baseline."""
    mem = get_memory_info()
    delta_str = ""
    if baseline_avail is not None:
        delta = baseline_avail - mem['ram_avail_gb']
        delta_str = f" | RAM delta: {delta:+.2f}GB"
    print(f"[{label}] GPU: {mem['gpu_used_gb']:.1f}/{mem['gpu_total_gb']:.1f}GB "
          f"(alloc={mem['gpu_alloc_gb']:.1f}, rsv={mem['gpu_rsv_gb']:.1f}) | "
          f"RAM avail: {mem['ram_avail_gb']:.1f}GB, shared: {mem['ram_shared_gb']:.1f}GB{delta_str}")
    return mem


def malloc_trim():
    """Try to return freed memory to OS."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def detailed_cleanup(llm, baseline_avail):
    """Run cleanup with detailed tracking at each step."""
    from blitzinfer.engine.cleanup import (
        clear_vllm_caches,
        destroy_parallel_state,
        nuclear_cleanup,
    )

    print("\n=== DETAILED CLEANUP START ===")
    log_mem("0. Before any cleanup", baseline_avail)

    if llm is None:
        return 0.0

    # Get initial GPU memory for freed calculation
    free_before, total = torch.cuda.mem_get_info()

    # Get references to internal components
    engine = getattr(llm, 'llm_engine', None)
    if engine is None:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        return 0.0

    # Navigate vLLM V1 structure
    inproc_client = getattr(engine, 'engine_core', None)
    engine_core = getattr(inproc_client, 'engine_core', inproc_client) if inproc_client else None

    model_runner = None
    model = None
    worker = None
    inner_worker = None

    if engine_core is not None:
        executor = getattr(engine_core, 'model_executor', None)
        if executor is not None:
            worker = getattr(executor, 'driver_worker', None)
            if worker is not None:
                inner_worker = getattr(worker, 'worker', None)
                if inner_worker is not None:
                    model_runner = getattr(inner_worker, 'model_runner', None)

    if model_runner is not None:
        model = getattr(model_runner, 'model', None)

    log_mem("1. After getting references", baseline_avail)

    # Step 2: Clear KV caches
    if model_runner is not None and hasattr(model_runner, 'kv_caches'):
        kv_caches = model_runner.kv_caches
        for i, kv in enumerate(kv_caches):
            if kv is not None:
                if isinstance(kv, torch.Tensor):
                    try:
                        kv.storage().resize_(0)
                    except Exception:
                        kv.data = torch.empty(0, device='cpu')
                elif isinstance(kv, (list, tuple)):
                    for t in kv:
                        if isinstance(t, torch.Tensor):
                            try:
                                t.storage().resize_(0)
                            except Exception:
                                t.data = torch.empty(0, device='cpu')
        kv_caches.clear()
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("2. After KV cache clear", baseline_avail)

    # Step 3: Clear static forward context
    if model_runner is not None and hasattr(model_runner, 'static_forward_context'):
        ctx = model_runner.static_forward_context
        for name in list(ctx.keys()):
            layer = ctx[name]
            if hasattr(layer, 'kv_cache'):
                kv_list = layer.kv_cache if isinstance(layer.kv_cache, list) else [layer.kv_cache]
                for kv in kv_list:
                    if isinstance(kv, torch.Tensor):
                        try:
                            kv.storage().resize_(0)
                        except Exception:
                            kv.data = torch.empty(0, device='cpu')
                layer.kv_cache = []
        ctx.clear()
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("3. After static_forward_context clear", baseline_avail)

    # Step 4: Clear model parameters
    if model is not None:
        param_count = 0
        for name, param in list(model.named_parameters()):
            if param.device.type == 'cuda':
                try:
                    param.data.storage().resize_(0)
                except Exception:
                    param.data = torch.empty(0, device='cpu')
                param_count += 1
        print(f"   Cleared {param_count} model parameters")
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("4. After model params clear", baseline_avail)

    # Step 5: Clear model buffers
    if model is not None:
        buf_count = 0
        for name, buf in list(model.named_buffers()):
            if buf.device.type == 'cuda':
                try:
                    buf.storage().resize_(0)
                except Exception:
                    try:
                        buf.data = torch.empty(0, device='cpu')
                    except Exception:
                        pass
                buf_count += 1
        print(f"   Cleared {buf_count} model buffers")
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("5. After model buffers clear", baseline_avail)

    # Step 6: Delete model and clear references
    del model
    if model_runner is not None:
        model_runner.model = None
    if inner_worker is not None:
        inner_worker.model_runner = None
    if worker is not None:
        worker.worker = None
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("6. After del model + clear refs", baseline_avail)

    # Step 7: Engine shutdown
    if engine_core is not None:
        try:
            engine_core.shutdown()
        except Exception:
            pass
    if inproc_client is not None and inproc_client is not engine_core:
        try:
            if hasattr(inproc_client, 'shutdown'):
                inproc_client.shutdown()
        except Exception:
            pass
    if engine is not None:
        engine.engine_core = None
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("7. After engine shutdown", baseline_avail)

    # Step 8: Delete LLM object
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log_mem("8. After del llm", baseline_avail)

    # Step 9: Multiple GC rounds
    for i in range(5):
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log_mem("9. After 5x GC rounds", baseline_avail)

    # Step 10: Clear vLLM caches
    clear_vllm_caches()
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("10. After clear_vllm_caches", baseline_avail)

    # Step 11: Destroy parallel state
    destroy_parallel_state()
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("11. After destroy_parallel_state", baseline_avail)

    # Step 12: Nuclear cleanup
    nuclear_cleanup()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log_mem("12. After nuclear_cleanup", baseline_avail)

    # Step 13: malloc_trim to return memory to OS
    malloc_trim()
    log_mem("13. After malloc_trim", baseline_avail)

    # Final memory stats
    free_after, _ = torch.cuda.mem_get_info()
    freed_gb = (free_after - free_before) / 1024**3

    print(f"=== CLEANUP FREED {freed_gb:.2f}GB GPU ===\n")
    return freed_gb


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.orchestrator.standby_manager import StandbyManager

    print("=" * 80)
    print("RAM LEAK DIAGNOSTIC TEST")
    print("=" * 80)

    # Initial state
    initial_mem = log_mem("INITIAL STATE (no arena, no model)")
    baseline_avail = initial_mem['ram_avail_gb']

    # Test 1: Create arena WITHOUT loading any models
    print("\n" + "=" * 60)
    print("TEST 1: Create 80GB pinned arena")
    print("=" * 60)

    start = time.time()
    standby = StandbyManager(
        arena_size_gb=80.0,
        chunk_size_gb=16.0,
        pin_memory=True,
        lazy_arena=False,
    )
    print(f"Arena created in {time.time() - start:.1f}s")
    log_mem("After arena creation", baseline_avail)

    # Test 2: Load first model (gpt-oss-120b)
    print("\n" + "=" * 60)
    print("TEST 2: Load first model (gpt-oss-120b)")
    print("=" * 60)

    start = time.time()
    llm = LLM(
        model="openai/gpt-oss-120b",
        gpu_memory_utilization=0.94,
        max_model_len=131072,
        trust_remote_code=True,
        enforce_eager=True,
    )
    print(f"Model loaded in {time.time() - start:.1f}s")
    after_load1 = log_mem("After first model load", baseline_avail)

    # Quick inference to warm up
    print("\nWarming up with inference...")
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {outputs[0].outputs[0].text}")
    log_mem("After warmup inference", baseline_avail)

    # Test 3: Cleanup first model with detailed tracking
    print("\n" + "=" * 60)
    print("TEST 3: Cleanup first model (detailed tracking)")
    print("=" * 60)

    detailed_cleanup(llm, baseline_avail)
    llm = None

    # Test 4: Load second model (qwen3-32b) - this is where the leak was observed
    print("\n" + "=" * 60)
    print("TEST 4: Load second model (qwen3-32b)")
    print("=" * 60)

    start = time.time()
    llm = LLM(
        model="Qwen/Qwen3-32B-FP8",
        gpu_memory_utilization=0.94,
        max_model_len=131072,
        trust_remote_code=True,
        enforce_eager=True,
    )
    print(f"Model loaded in {time.time() - start:.1f}s")
    after_load2 = log_mem("After second model load", baseline_avail)

    # Quick inference
    print("\nWarming up with inference...")
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {outputs[0].outputs[0].text}")
    log_mem("After warmup inference", baseline_avail)

    # Test 5: Cleanup second model
    print("\n" + "=" * 60)
    print("TEST 5: Cleanup second model (detailed tracking)")
    print("=" * 60)

    detailed_cleanup(llm, baseline_avail)
    llm = None

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    final_mem = log_mem("FINAL STATE", baseline_avail)

    total_leak = baseline_avail - final_mem['ram_avail_gb']
    print(f"\nTotal RAM consumed: {total_leak:.2f}GB")
    print(f"  - Expected from arena: ~80GB")
    print(f"  - Unexpected leak: {max(0, total_leak - 80):.2f}GB")

    # Cleanup arena
    print("\nCleaning up arena...")
    del standby
    gc.collect()
    malloc_trim()
    log_mem("After arena cleanup", baseline_avail)


if __name__ == "__main__":
    main()
