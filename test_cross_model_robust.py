#!/usr/bin/env python3
"""
Robust cross-architecture model switching test with better memory management.

The previous test hit a GPU hang during Mistral inference. This version:
1. Uses more conservative memory settings
2. Adds explicit GPU synchronization and cache clearing
3. Tests models individually first before switching
"""

import gc
import os
import sys
import time
import types

# ROCm setup - more conservative settings
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:False'  # More predictable memory

# vLLM import workaround
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch

# Apply Blitz patch BEFORE importing vLLM
import blitz_vllm_patch
blitz_vllm_patch.patch_vllm()

from vllm import LLM, SamplingParams


def get_mem():
    """Get free GPU memory in GB."""
    return torch.cuda.mem_get_info()[0] / (1024**3)


def full_gpu_cleanup():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Try to release all cached memory
    if hasattr(torch.cuda, 'memory'):
        try:
            torch.cuda.memory.empty_cache()
        except:
            pass

    gc.collect()
    torch.cuda.synchronize()


def test_single_model(model_name, config, prompt, expected=None):
    """Test a single model in isolation."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")

    full_gpu_cleanup()
    print(f"Initial memory: {get_mem():.1f} GB free")

    try:
        t0 = time.perf_counter()
        llm = LLM(model=model_name, **config)
        load_time = (time.perf_counter() - t0) * 1000
        print(f"Load time: {load_time:.0f}ms")
        print(f"Memory after load: {get_mem():.1f} GB free")

        # Test inference
        params = SamplingParams(max_tokens=30, temperature=0.7)

        # Synchronize before inference
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        outputs = llm.generate([prompt], params)
        infer_time = (time.perf_counter() - t0) * 1000

        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        valid = len(text.strip()) > 0
        if expected:
            valid = valid and any(e.lower() in text.lower() for e in expected)

        print(f"Inference time: {infer_time:.0f}ms")
        print(f"Output [{['FAIL','OK'][valid]}]: {text[:60]}...")

        # Cleanup
        del llm
        full_gpu_cleanup()
        print(f"Memory after cleanup: {get_mem():.1f} GB free")

        return True, load_time, infer_time

    except Exception as e:
        print(f"ERROR: {e}")
        full_gpu_cleanup()
        return False, 0, 0


def test_switching_sequence(models, config):
    """Test switching between models in sequence."""
    print(f"\n{'='*70}")
    print("SEQUENTIAL SWITCHING TEST")
    print(f"{'='*70}")

    full_gpu_cleanup()
    initial_mem = get_mem()
    print(f"Initial memory: {initial_mem:.1f} GB free")

    results = []
    llm = None

    for i, (model_name, prompt, expected) in enumerate(models):
        print(f"\n--- Step {i+1}: {model_name.split('/')[-1]} ---")

        # Cleanup previous model if exists
        if llm is not None:
            print("Cleaning up previous model...")
            t0 = time.perf_counter()
            del llm
            llm = None

            # Fast cleanup
            cleanup_time, cleared = blitz_vllm_patch.fast_cleanup()
            full_gpu_cleanup()

            cleanup_total = (time.perf_counter() - t0) * 1000
            print(f"Cleanup: {cleanup_total:.0f}ms (fast_cleanup: {cleanup_time:.0f}ms, cleared {cleared} params)")
            print(f"Memory after cleanup: {get_mem():.1f} GB free")

        # Load new model
        t_switch = time.perf_counter()

        try:
            t0 = time.perf_counter()
            llm = LLM(model=model_name, **config)
            load_time = (time.perf_counter() - t0) * 1000
            print(f"Load time: {load_time:.0f}ms")

            # Synchronize before inference
            torch.cuda.synchronize()

            # Test inference
            params = SamplingParams(max_tokens=30, temperature=0.7)
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], params)
            infer_time = (time.perf_counter() - t0) * 1000

            text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            valid = len(text.strip()) > 0
            if expected:
                valid = valid and any(e.lower() in text.lower() for e in expected)

            switch_time = (time.perf_counter() - t_switch) * 1000

            print(f"Inference: {infer_time:.0f}ms")
            print(f"Output [{['FAIL','OK'][valid]}]: {text[:60]}...")
            print(f"Total switch time: {switch_time:.0f}ms")

            results.append((model_name.split('/')[-1], switch_time, valid))

        except Exception as e:
            print(f"ERROR during {model_name}: {e}")
            results.append((model_name.split('/')[-1], -1, False))
            llm = None
            full_gpu_cleanup()

    # Final cleanup
    if llm is not None:
        del llm
        full_gpu_cleanup()

    return results


def main():
    print("="*70)
    print("ROBUST CROSS-ARCHITECTURE MODEL SWITCHING TEST")
    print("="*70)

    # Models
    qwen_model = "Qwen/Qwen2.5-7B-Instruct"
    mistral_model = "mistralai/Mistral-7B-Instruct-v0.3"

    # More conservative config for stability
    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.20,  # Reduced from 0.25
        "max_model_len": 256,             # Reduced from 512
        "max_num_batched_tokens": 256,    # Reduced
        "kv_cache_memory_bytes": 1 * 1024**3,  # Reduced from 2GB
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    # Warmup GPU
    print("\nWarming up GPU...")
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    full_gpu_cleanup()

    # Phase 1: Test each model individually
    print("\n" + "="*70)
    print("PHASE 1: Individual Model Tests")
    print("="*70)

    qwen_ok, qwen_load, qwen_infer = test_single_model(
        qwen_model, config,
        "What is the capital of France?",
        ["paris"]
    )

    mistral_ok, mistral_load, mistral_infer = test_single_model(
        mistral_model, config,
        "What is 2 + 2?",
        ["4", "four"]
    )

    # Phase 2: Only test switching if both models work individually
    if qwen_ok and mistral_ok:
        print("\n" + "="*70)
        print("PHASE 2: Switching Test")
        print("="*70)

        models = [
            (qwen_model, "Capital of France?", ["paris"]),
            (mistral_model, "What is 2+2?", ["4", "four"]),
            (qwen_model, "Count 1 to 3:", ["1", "2", "3"]),
        ]

        results = test_switching_sequence(models, config)

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        print("\nIndividual tests:")
        print(f"  Qwen:    load={qwen_load:.0f}ms, infer={qwen_infer:.0f}ms, OK={qwen_ok}")
        print(f"  Mistral: load={mistral_load:.0f}ms, infer={mistral_infer:.0f}ms, OK={mistral_ok}")

        print("\nSwitching sequence:")
        all_valid = True
        for name, switch_time, valid in results:
            status = "OK" if valid else "FAIL"
            time_str = f"{switch_time:.0f}ms" if switch_time > 0 else "ERROR"
            print(f"  {name}: {time_str} [{status}]")
            all_valid = all_valid and valid

        # Calculate average switch time (excluding first load)
        switch_times = [r[1] for r in results[1:] if r[1] > 0]
        if switch_times:
            avg_switch = sum(switch_times) / len(switch_times)
            print(f"\n  Average switch time: {avg_switch:.0f}ms ({avg_switch/1000:.1f}s)")

        print(f"  All outputs valid: {all_valid}")

        if all_valid:
            print("\n[SUCCESS] Cross-architecture switching works!")
        else:
            print("\n[WARNING] Some tests failed")
    else:
        print("\n" + "="*70)
        print("SKIPPING SWITCH TEST - Individual model tests failed")
        print("="*70)
        if not qwen_ok:
            print("  Qwen failed")
        if not mistral_ok:
            print("  Mistral failed")


if __name__ == '__main__':
    main()
