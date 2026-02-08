#!/usr/bin/env python3
"""Test page cache warming approach for fast model switching.

Target: RTX PRO 6000 (95GB VRAM)
- Model A: GPT-OSS-120B (~60GB)
- Model B: Qwen3-VL-32B (~62GB)
- Context: 100K+ tokens
- Goal: Warm Model B cache while serving Model A, then switch

Expected performance:
- Cache warming: ~12 GB/s (background, no I/O competition)
- Cold weight loading: ~20s
- Warm weight loading: ~8s (from page cache)
- vLLM init overhead: ~17s (KV cache profiling, warmup)
"""

import os
import sys
import time
import logging
import gc
import subprocess
import multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# vLLM config - single-process mode with proper cleanup (see CLAUDE.md for details)
# engine_core.shutdown() + tensor clearing releases all GPU memory
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')  # Single-process mode with proper cleanup
os.environ.setdefault('VLLM_SKIP_WARMUP', '1')  # Skip warmup for faster startup

import torch

from blitzinfer.memory import PageCacheWarmer, WarmStatus

# Models for RTX PRO 6000
MODEL_A = "openai/gpt-oss-120b"
MODEL_B = "Qwen/Qwen3-VL-32B-Instruct"

# vLLM config for 95GB GPU
# Full GPU utilization - proper cleanup via engine_core.shutdown() releases all memory
# Note: Qwen3-VL-32B default is 262K but that needs 64GB KV cache, doesn't fit
VLLM_CONFIG = {
    "dtype": "bfloat16",
    "max_model_len": 100000,  # 100K tokens - largest that fits with these models
    "gpu_memory_utilization": 0.95,  # Full GPU - ~90GB
    "max_num_seqs": 16,
    "max_num_batched_tokens": 8192,
    "enforce_eager": True,
    "trust_remote_code": True,
}


def get_model_path(model_name: str) -> str:
    """Get local path for HuggingFace model."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model_name, local_files_only=True)


def drop_caches():
    """Drop OS page caches (requires sudo or appropriate permissions)."""
    os.system('sync')
    result = os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1')
    if result != 0:
        logger.warning("Could not drop caches - may need sudo permissions")
        return False
    time.sleep(1)
    return True


def load_vllm_model(model_name: str):
    """Load a model with vLLM and return load time."""
    from vllm import LLM, SamplingParams

    logger.info(f"Loading {model_name}...")
    t0 = time.time()
    llm = LLM(model=model_name, **VLLM_CONFIG)
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.2f}s")

    # Quick inference test
    sampling = SamplingParams(max_tokens=10, temperature=0.7)
    output = llm.generate(["Hello, I am"], sampling)
    logger.info(f"Output: {output[0].outputs[0].text[:50]}...")

    return llm, load_time


def unload_vllm_model(llm):
    """Unload vLLM model with proper cleanup.

    CRITICAL: Must call engine_core.shutdown() before del to release GPU memory.
    Without this, the main process inherits ~90GB of orphaned CUDA memory.
    """
    # Explicitly shutdown engine_core - this terminates subprocess and frees GPU memory
    try:
        llm.llm_engine.engine_core.shutdown()
        logger.info("Engine core shutdown successful")
    except Exception as e:
        logger.warning(f"Engine shutdown warning: {e}")

    time.sleep(1)
    del llm
    gc.collect()
    time.sleep(1)


def test_cache_warmer_switch():
    """Test full model switch with page cache warming.

    Flow:
    1. Load Model A (cold)
    2. Start warming Model B cache in background
    3. Serve requests on Model A while cache warms
    4. Switch to Model B (should be fast due to warm cache)
    """
    from vllm import SamplingParams

    print("\n" + "=" * 70)
    print("TEST: PAGE CACHE WARMING MODEL SWITCH")
    print("=" * 70)
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print(f"Context: {VLLM_CONFIG['max_model_len']} tokens")
    print(f"GPU util: {VLLM_CONFIG['gpu_memory_utilization']}")

    results = {}

    # Get model paths and sizes
    path_a = get_model_path(MODEL_A)
    path_b = get_model_path(MODEL_B)

    from blitzinfer.memory.cache_warmer import get_model_size
    size_a = get_model_size(path_a)
    size_b = get_model_size(path_b)
    print(f"\nModel A size: {size_a / 1024**3:.1f} GB")
    print(f"Model B size: {size_b / 1024**3:.1f} GB")

    # Create cache warmer
    warmer = PageCacheWarmer()
    warmer.register_model(MODEL_A, path_a)
    warmer.register_model(MODEL_B, path_b)

    # Drop caches to ensure cold start
    print("\nDropping page caches for cold start...")
    drop_caches()

    # === Phase 1: Load Model A (cold) ===
    print("\n" + "-" * 50)
    print("PHASE 1: Load Model A (cold)")
    print("-" * 50)

    llm_a, load_a_time = load_vllm_model(MODEL_A)
    results['model_a_cold_load'] = load_a_time

    # === Phase 2: Warm Model B cache while serving Model A ===
    print("\n" + "-" * 50)
    print("PHASE 2: Warm Model B cache (background)")
    print("-" * 50)

    # Start background warming
    t0_warm = time.time()
    warmer.start_warming(MODEL_B)

    # Serve requests on Model A while cache warms
    sampling = SamplingParams(max_tokens=50, temperature=0.7)
    print("Serving requests on Model A while warming Model B...")

    for i in range(5):
        prompt = f"Question {i+1}: Explain the concept of"
        output = llm_a.generate([prompt], sampling)

        status = warmer.get_status(MODEL_B)
        progress = warmer.get_progress(MODEL_B) * 100
        print(f"  Request {i+1} done | Warming: {status.name} ({progress:.0f}%)")

        if status == WarmStatus.WARM:
            break

    # Wait for warming to complete if not done
    if not warmer.is_warm(MODEL_B):
        print("Waiting for cache warming to complete...")
        warmer.wait_for_warm(MODEL_B, timeout=120)

    warm_time = time.time() - t0_warm
    warm_speed = (size_b / 1024**3) / warm_time
    print(f"\nCache warming complete: {warm_time:.2f}s ({warm_speed:.1f} GB/s)")
    results['model_b_warm_time'] = warm_time
    results['model_b_warm_speed'] = warm_speed

    # === Phase 3: Switch to Model B ===
    print("\n" + "-" * 50)
    print("PHASE 3: Switch to Model B (warm cache)")
    print("-" * 50)

    # Unload Model A
    t0_switch = time.time()
    print("Unloading Model A...")
    unload_vllm_model(llm_a)
    unload_time = time.time() - t0_switch
    print(f"Model A unloaded in {unload_time:.2f}s")

    # Load Model B (should be fast - weights in page cache)
    t0_load_b = time.time()
    llm_b, load_b_time = load_vllm_model(MODEL_B)
    results['model_b_warm_load'] = load_b_time

    total_switch = time.time() - t0_switch
    results['total_switch_time'] = total_switch

    print(f"\nSwitch complete: {total_switch:.2f}s")
    print(f"  Unload: {unload_time:.2f}s")
    print(f"  Load B: {load_b_time:.2f}s")

    # Clean up
    unload_vllm_model(llm_b)
    warmer.shutdown()

    return results


def test_cold_switch():
    """Baseline: cold model switch without cache warming."""
    print("\n" + "=" * 70)
    print("BASELINE: COLD MODEL SWITCH")
    print("=" * 70)

    results = {}

    # Drop caches
    print("Dropping page caches...")
    drop_caches()

    # Load Model A (cold)
    print("\n--- Loading Model A (cold) ---")
    llm_a, load_a_time = load_vllm_model(MODEL_A)
    results['model_a_cold_load'] = load_a_time

    # Unload
    print("\n--- Unloading Model A ---")
    t0 = time.time()
    unload_vllm_model(llm_a)
    unload_time = time.time() - t0
    print(f"Unloaded in {unload_time:.2f}s")

    # Drop caches again for cold Model B load
    print("\nDropping caches for cold Model B load...")
    drop_caches()

    # Load Model B (cold)
    print("\n--- Loading Model B (cold) ---")
    llm_b, load_b_time = load_vllm_model(MODEL_B)
    results['model_b_cold_load'] = load_b_time

    total_switch = unload_time + load_b_time
    results['total_switch_time'] = total_switch

    print(f"\n--- Cold Switch Summary ---")
    print(f"Unload: {unload_time:.2f}s")
    print(f"Load B: {load_b_time:.2f}s")
    print(f"Total: {total_switch:.2f}s")

    unload_vllm_model(llm_b)

    return results


def main():
    print("=" * 70)
    print("PAGE CACHE WARMING TEST")
    print("=" * 70)

    gpu_info = {
        "name": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
    }
    print(f"GPU: {gpu_info['name']} ({gpu_info['vram_gb']:.1f} GB)")

    all_results = {}

    # Test 1: Baseline cold switch
    print("\n" + "#" * 70)
    print("# TEST 1: COLD SWITCH BASELINE")
    print("#" * 70)
    cold = test_cold_switch()
    all_results['cold'] = cold

    # Wait for cleanup
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

    # Test 2: Cache warming switch
    print("\n" + "#" * 70)
    print("# TEST 2: CACHE WARMING SWITCH")
    print("#" * 70)
    warm = test_cache_warmer_switch()
    all_results['warm'] = warm

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Cold Model B load:  {cold['model_b_cold_load']:.1f}s")
    print(f"Warm Model B load:  {warm['model_b_warm_load']:.1f}s")
    print(f"Cache warm time:    {warm['model_b_warm_time']:.1f}s (background)")
    print(f"Cache warm speed:   {warm['model_b_warm_speed']:.1f} GB/s")
    print()
    print(f"Cold switch total:  {cold['total_switch_time']:.1f}s")
    print(f"Warm switch total:  {warm['total_switch_time']:.1f}s")
    print()

    speedup = cold['model_b_cold_load'] / warm['model_b_warm_load']
    time_saved = cold['model_b_cold_load'] - warm['model_b_warm_load']
    print(f"Load speedup:       {speedup:.1f}x")
    print(f"Time saved:         {time_saved:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
