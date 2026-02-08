#!/usr/bin/env python3
"""Test full vLLM model switch with large models using prefetch system.

Target: RTX PRO 6000 system
- 70GB pinned arena in system RAM
- 88GB+ GPU VRAM for vLLM
- 120K+ context length
"""

import os
import sys
import time
import logging
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_SKIP_WARMUP', '1')

import torch

from blitzinfer.memory import PinnedMemoryArena, ModelPrefetcher, PrefetchStatus
from blitzinfer.memory import get_model_size

# Models
MODEL_A = "openai/gpt-oss-120b"          # ~60GB
MODEL_B = "Qwen/Qwen3-VL-32B-Instruct"   # ~62GB

# Arena config
ARENA_SIZE_GB = 70.0

# vLLM config for RTX PRO 6000 (95GB VRAM, 120K+ context)
VLLM_CONFIG = {
    "dtype": "bfloat16",
    "max_model_len": 131072,  # 128K context
    "gpu_memory_utilization": 0.72,  # ~68GB - fits model + KV cache with headroom for switch
    "max_num_seqs": 16,
    "max_num_batched_tokens": 8192,
    "enforce_eager": True,
    "trust_remote_code": True,
}


def get_model_path(model_name: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(model_name, local_files_only=True)


def get_gpu_info():
    if not torch.cuda.is_available():
        return None
    return {
        "name": torch.cuda.get_device_name(),
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
    }


def cleanup_vllm(llm):
    """Properly cleanup vLLM to free GPU memory."""
    import torch._dynamo
    import multiprocessing

    # Clear model parameters and KV cache
    try:
        engine_core = llm.llm_engine.engine_core
        if hasattr(engine_core, 'engine_core'):
            core = engine_core.engine_core
        else:
            core = engine_core

        if hasattr(core, 'model_executor'):
            executor = core.model_executor
            if hasattr(executor, 'driver_worker'):
                worker = executor.driver_worker
                if hasattr(worker, 'worker') and worker.worker is not None:
                    model_runner = getattr(worker.worker, 'model_runner', None)
                    if model_runner is not None:
                        if hasattr(model_runner, 'model'):
                            model = model_runner.model
                            for param in model.parameters():
                                param.data = torch.empty(0, device='cpu')

                        if hasattr(model_runner, 'kv_caches'):
                            model_runner.kv_caches.clear()
    except Exception as e:
        logger.debug(f"Cleanup warning: {e}")

    del llm

    # Reset vLLM state
    try:
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            vllm_config.compilation_config.static_forward_context.clear()
    except Exception:
        pass

    try:
        import vllm.config.vllm as vllm_config_module
        vllm_config_module._current_vllm_config = None
        vllm_config_module._current_prefix = None
        vllm_config_module.get_cached_compilation_config.cache_clear()
    except Exception:
        pass

    try:
        torch._dynamo.reset()
    except Exception:
        pass

    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except Exception:
        pass

    for child in multiprocessing.active_children():
        child.join(timeout=5.0)
        if child.is_alive():
            child.terminate()
            child.join(timeout=2.0)

    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def test_cold_switch():
    """Baseline: cold model switch without prefetch."""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 70)
    print("TEST: COLD MODEL SWITCH (Baseline)")
    print("=" * 70)

    results = {}
    sampling_params = SamplingParams(max_tokens=50, temperature=0.7)

    # Load Model A
    print(f"\n--- Loading {MODEL_A} (cold) ---")
    t0 = time.time()
    llm_a = LLM(model=MODEL_A, **VLLM_CONFIG)
    load_a_time = time.time() - t0
    print(f"Model A loaded in {load_a_time:.2f}s")
    results['model_a_load'] = load_a_time

    # Inference
    output = llm_a.generate(["Hello, I am"], sampling_params)
    print(f"Output: {output[0].outputs[0].text[:80]}...")

    # Unload Model A with proper cleanup
    print(f"\n--- Unloading Model A ---")
    t0 = time.time()
    cleanup_vllm(llm_a)
    unload_time = time.time() - t0
    print(f"Unloaded in {unload_time:.2f}s")
    results['model_a_unload'] = unload_time

    gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
    print(f"GPU free after unload: {gpu_free:.1f} GB")

    # Load Model B
    print(f"\n--- Loading {MODEL_B} (cold) ---")
    t0 = time.time()
    llm_b = LLM(model=MODEL_B, **VLLM_CONFIG)
    load_b_time = time.time() - t0
    print(f"Model B loaded in {load_b_time:.2f}s")
    results['model_b_load'] = load_b_time

    # Inference
    output = llm_b.generate(["Hello, I am"], sampling_params)
    print(f"Output: {output[0].outputs[0].text[:80]}...")

    # Summary
    total_switch = unload_time + load_b_time
    print(f"\n--- Cold Switch Summary ---")
    print(f"Unload: {unload_time:.2f}s + Load: {load_b_time:.2f}s = {total_switch:.2f}s")
    results['total_switch'] = total_switch

    cleanup_vllm(llm_b)

    return results


def test_prefetch_switch():
    """Model switch with prefetch (SSD → Arena while serving, then Arena → GPU)."""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 70)
    print("TEST: PREFETCH MODEL SWITCH")
    print("=" * 70)

    results = {}
    sampling_params = SamplingParams(max_tokens=50, temperature=0.7)

    # Get paths and sizes
    path_a = get_model_path(MODEL_A)
    path_b = get_model_path(MODEL_B)
    size_a = get_model_size(path_a)
    size_b = get_model_size(path_b)

    print(f"Model A: {size_a / 1024**3:.1f} GB")
    print(f"Model B: {size_b / 1024**3:.1f} GB")

    # Create arena
    print(f"\nAllocating {ARENA_SIZE_GB} GB pinned arena...")
    t0 = time.time()
    arena = PinnedMemoryArena(ARENA_SIZE_GB)
    print(f"Arena allocated in {time.time() - t0:.2f}s (pinned: {arena.is_pinned})")

    # Create prefetcher
    prefetcher = ModelPrefetcher(arena)
    prefetcher.register_model(MODEL_A, path_a)
    prefetcher.register_model(MODEL_B, path_b)

    # Load Model A (cold)
    print(f"\n--- Loading {MODEL_A} (cold) ---")
    t0 = time.time()
    llm_a = LLM(model=MODEL_A, **VLLM_CONFIG)
    load_a_time = time.time() - t0
    print(f"Model A loaded in {load_a_time:.2f}s")
    results['model_a_load'] = load_a_time

    # Start prefetch for Model B while serving Model A
    print(f"\n--- Starting prefetch for {MODEL_B} (background) ---")
    t0_prefetch = time.time()
    prefetcher.start_prefetch(MODEL_B)

    # Serve requests while prefetching
    print("Serving requests while prefetch runs...")
    for i in range(3):
        output = llm_a.generate([f"Question {i+1}: Explain"], sampling_params)
        status = prefetcher.get_status(MODEL_B)
        print(f"  Request {i+1} done, prefetch: {status.name}")

    # Wait for prefetch
    print("Waiting for prefetch to complete...")
    prefetcher.wait_for_ready(MODEL_B, timeout=300)
    prefetch_time = time.time() - t0_prefetch
    prefetch_speed = (size_b / 1024**3) / prefetch_time
    print(f"Prefetch done: {size_b / 1024**3:.1f} GB in {prefetch_time:.2f}s ({prefetch_speed:.1f} GB/s)")
    results['prefetch_time'] = prefetch_time
    results['prefetch_speed'] = prefetch_speed

    # SWITCH: Unload A, transfer B from arena to GPU
    print(f"\n--- SWITCH: Unload A, Load B from arena ---")

    # Unload
    t0_unload = time.time()
    cleanup_vllm(llm_a)
    unload_time = time.time() - t0_unload
    print(f"Unloaded Model A in {unload_time:.2f}s")
    results['unload_time'] = unload_time

    gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
    print(f"GPU free after unload: {gpu_free:.1f} GB")

    # Transfer from arena to GPU
    print("Transferring from arena to GPU...")
    t0_transfer = time.time()
    tensors = prefetcher.get_tensors_for_gpu(MODEL_B)
    total_bytes = 0
    gpu_tensors = {}
    for name, tensor in tensors.items():
        gpu_tensors[name] = tensor.to('cuda', non_blocking=True)
        total_bytes += tensor.numel() * tensor.element_size()
    torch.cuda.synchronize()
    transfer_time = time.time() - t0_transfer
    transfer_speed = (total_bytes / 1024**3) / transfer_time
    print(f"Transferred {total_bytes / 1024**3:.1f} GB in {transfer_time:.2f}s ({transfer_speed:.1f} GB/s)")
    results['transfer_time'] = transfer_time
    results['transfer_speed'] = transfer_speed

    # Initialize vLLM (weights already in GPU)
    print("Initializing vLLM...")
    del gpu_tensors
    gc.collect()
    torch.cuda.empty_cache()

    t0_init = time.time()
    llm_b = LLM(model=MODEL_B, **VLLM_CONFIG)
    init_time = time.time() - t0_init
    print(f"vLLM initialized in {init_time:.2f}s")
    results['init_time'] = init_time

    # Inference
    output = llm_b.generate(["Hello, I am"], sampling_params)
    print(f"Output: {output[0].outputs[0].text[:80]}...")

    # Summary
    hot_switch = unload_time + transfer_time + init_time
    print(f"\n--- Prefetch Switch Summary ---")
    print(f"Prefetch (background): {prefetch_time:.2f}s @ {prefetch_speed:.1f} GB/s")
    print(f"Unload: {unload_time:.2f}s")
    print(f"Arena→GPU: {transfer_time:.2f}s @ {transfer_speed:.1f} GB/s")
    print(f"vLLM init: {init_time:.2f}s")
    print(f"HOT SWITCH: {hot_switch:.2f}s (user wait time)")
    results['hot_switch'] = hot_switch

    cleanup_vllm(llm_b)
    prefetcher.shutdown()

    return results


def main():
    print("=" * 70)
    print("LARGE MODEL SWITCH TEST")
    print("=" * 70)
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print(f"Arena: {ARENA_SIZE_GB} GB pinned")
    print(f"Context: {VLLM_CONFIG['max_model_len']} tokens")

    gpu = get_gpu_info()
    if gpu:
        print(f"GPU: {gpu['name']} ({gpu['total_gb']:.1f} GB)")

    all_results = {}

    # Test 1: Cold switch baseline
    print("\n" + "#" * 70)
    cold = test_cold_switch()
    all_results['cold'] = cold

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

    # Test 2: Prefetch switch
    print("\n" + "#" * 70)
    prefetch = test_prefetch_switch()
    all_results['prefetch'] = prefetch

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Cold switch:     {cold['total_switch']:.1f}s")
    print(f"Prefetch switch: {prefetch['hot_switch']:.1f}s")
    print(f"Speedup:         {cold['total_switch'] / prefetch['hot_switch']:.1f}x")
    print(f"Time saved:      {cold['total_switch'] - prefetch['hot_switch']:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
