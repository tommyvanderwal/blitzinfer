#!/usr/bin/env python3
"""Test cross-architecture switching with fixed cleanup.

This tests the fix for the InprocClient navigation bug that was causing
~42GB of memory to be stuck after model cleanup.

Expected: With the fix, cleanup should free ~38GB leaving only ~9GB
(CUDA context + PyTorch buffers), enabling gpt-oss-120b to load.
"""

import os
import gc
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_gpu_memory():
    free, total = torch.cuda.mem_get_info()
    return {
        'free_gb': free / 1024**3,
        'used_gb': (total - free) / 1024**3,
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
    }


def log_mem(label):
    m = get_gpu_memory()
    print(f"[{label}] Used: {m['used_gb']:.2f}GB, Alloc: {m['allocated_gb']:.2f}GB, Free: {m['free_gb']:.2f}GB")
    return m


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL_A = "Qwen/Qwen3-VL-32B-Thinking-FP8"  # ~34GB
    MODEL_B = "openai/gpt-oss-120b"              # ~60GB

    print("=" * 70)
    print("CROSS-ARCHITECTURE SWITCHING TEST (Fixed Cleanup)")
    print("=" * 70)
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print()

    log_mem("baseline")

    # === Phase 1: Load Model A ===
    print("\n=== PHASE 1: Load Model A (Qwen-32B) ===")
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL_A,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=True,
    )
    load_a_time = time.perf_counter() - t0
    print(f"Load time: {load_a_time:.1f}s")
    log_mem("after Model A load")

    # Test inference
    out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=20))
    print(f"Model A output: {out[0].outputs[0].text.strip()[:60]}")

    # === Phase 2: Cleanup Model A ===
    print("\n=== PHASE 2: Cleanup Model A ===")
    t0 = time.perf_counter()
    freed = full_cleanup(llm)
    cleanup_time = time.perf_counter() - t0
    print(f"Cleanup time: {cleanup_time:.1f}s")
    print(f"Memory freed: {freed:.1f}GB")
    llm = None

    m = log_mem("after cleanup")

    # Check if we have enough memory for Model B
    required_gb = 60  # gpt-oss-120b needs ~60GB for weights
    available_gb = m['free_gb']
    print(f"\nMemory check: {available_gb:.1f}GB available, {required_gb}GB required")

    if available_gb < required_gb:
        print(f"ERROR: Not enough memory! Need {required_gb}GB but only {available_gb:.1f}GB free")
        print("The cleanup fix did not work properly.")
        return False

    print("SUCCESS: Enough memory available for Model B!")

    # === Phase 3: Load Model B ===
    print("\n=== PHASE 3: Load Model B (gpt-oss-120b) ===")
    # gpt-oss-120b is ~66GB, need higher utilization and smaller context
    t0 = time.perf_counter()
    try:
        llm = LLM(
            model=MODEL_B,
            dtype="bfloat16",
            max_model_len=2048,  # Smaller context to leave room for KV cache
            gpu_memory_utilization=0.88,  # Model is 66GB, need ~80% just for weights
            enforce_eager=True,
            trust_remote_code=True,
        )
        load_b_time = time.perf_counter() - t0
        print(f"Load time: {load_b_time:.1f}s")
        log_mem("after Model B load")

        # Test inference
        out = llm.generate(["What is 3+3?"], SamplingParams(max_tokens=20))
        print(f"Model B output: {out[0].outputs[0].text.strip()[:60]}")

        # === Phase 4: Switch back to Model A ===
        print("\n=== PHASE 4: Switch back to Model A ===")
        freed = full_cleanup(llm)
        llm = None
        print(f"Freed: {freed:.1f}GB")
        log_mem("after Model B cleanup")

        t0 = time.perf_counter()
        llm = LLM(
            model=MODEL_A,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=True,
        )
        switch_back_time = time.perf_counter() - t0
        print(f"Switch back time: {switch_back_time:.1f}s")

        out = llm.generate(["What is 4+4?"], SamplingParams(max_tokens=20))
        print(f"Model A output: {out[0].outputs[0].text.strip()[:60]}")

        success = True

    except Exception as e:
        print(f"ERROR: Failed to load Model B: {e}")
        import traceback
        traceback.print_exc()
        success = False
        load_b_time = 0
        switch_back_time = 0

    finally:
        if llm is not None:
            full_cleanup(llm)
        gc.collect()
        torch.cuda.empty_cache()

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Model A (Qwen-32B) load:     {load_a_time:.1f}s
    Cleanup freed:              {freed:.1f}GB
    Model B (gpt-oss-120b) load: {load_b_time:.1f}s
    Switch back to A:           {switch_back_time:.1f}s

    Cross-architecture switching: {"SUCCESS" if success else "FAILED"}
    """)

    log_mem("final")
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
