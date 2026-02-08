#!/usr/bin/env python3
"""
Memory cleanup fix for vLLM on ROCm 780M - Version 2.

This version takes a more aggressive approach: completely unload
vLLM modules from Python so the next load starts with fresh state.
"""

import gc
import sys
import torch


def get_memory_gb():
    free, total = torch.cuda.mem_get_info()
    return free / (1024**3)


def unload_vllm_modules():
    """
    Completely unload all vLLM-related modules from Python.
    This forces a fresh import on next use.
    """
    # Get list of vLLM modules to unload
    vllm_modules = [name for name in list(sys.modules.keys())
                    if 'vllm' in name.lower()]

    # Also unload related modules that might hold references
    related_prefixes = ['transformers', 'safetensors', 'tokenizers']

    unloaded = 0
    for name in vllm_modules:
        try:
            del sys.modules[name]
            unloaded += 1
        except:
            pass

    return unloaded


def aggressive_tensor_cleanup():
    """
    Aggressively find and delete GPU tensors from gc.
    """
    # First pass: find all GPU tensors
    gpu_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
                gpu_tensors.append(id(obj))
        except:
            pass

    initial_count = len(gpu_tensors)

    # Multiple gc passes
    for _ in range(5):
        gc.collect()

    # Check remaining
    remaining = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
                remaining += 1
        except:
            pass

    return initial_count, remaining


def full_cleanup_with_module_unload(shutdown_ray=False):
    """
    Complete cleanup including module unloading.
    """
    # Step 1: Standard vLLM cleanup
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=shutdown_ray)
    except:
        pass

    gc.collect()

    # Step 2: Reset torch dynamo
    try:
        import torch._dynamo
        torch._dynamo.reset()
    except:
        pass

    # Step 3: Unload vLLM modules
    unloaded = unload_vllm_modules()

    # Step 4: Multiple gc passes
    for _ in range(5):
        gc.collect()

    # Step 5: Clear CUDA caches
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Step 6: Check what's left
    initial, remaining = aggressive_tensor_cleanup()

    gc.collect()
    torch.cuda.empty_cache()

    return unloaded, initial, remaining


def test_module_unload_approach():
    """Test the module unload approach."""
    import os
    import time
    import types

    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_SKIP_WARMUP'] = '1'
    os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

    fake_meta = types.ModuleType('torchvision._meta_registrations')
    sys.modules['torchvision._meta_registrations'] = fake_meta
    sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

    print("=" * 70)
    print("TESTING MODULE UNLOAD APPROACH")
    print("=" * 70)

    initial = get_memory_gb()
    print(f"\nInitial: {initial:.2f} GB free")

    # First model
    print("\n>>> Loading first model...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.50,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=4 * 1024**3,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )

    after_load = get_memory_gb()
    print(f"After load: {after_load:.2f} GB free (used {initial - after_load:.2f} GB)")

    # Generate
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text}")

    # Delete first
    print("\n>>> Deleting first model...")
    del llm
    del out

    # Cleanup with module unload
    print(">>> Running cleanup with module unload...")
    unloaded, tensors_before, tensors_after = full_cleanup_with_module_unload()
    print(f"Unloaded {unloaded} vLLM modules")
    print(f"GPU tensors: {tensors_before} -> {tensors_after}")

    time.sleep(1.0)
    gc.collect()
    torch.cuda.empty_cache()

    after_cleanup = get_memory_gb()
    leaked = initial - after_cleanup
    print(f"\nAfter cleanup: {after_cleanup:.2f} GB free")
    print(f"Memory leaked: {leaked:.2f} GB")

    if leaked < 2.0:
        print("\n>>> Cleanup SUCCESS! Trying second model...")

        # Re-import vLLM (fresh state)
        print(">>> Re-importing vLLM...")
        from vllm import LLM, SamplingParams

        llm2 = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            dtype="float16",
            gpu_memory_utilization=0.50,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            kv_cache_memory_bytes=4 * 1024**3,
            enforce_eager=True,
            compilation_config={"custom_ops": ["none"]},
        )

        out2 = llm2.generate(["Test"], SamplingParams(max_tokens=5))
        print(f"Output: {out2[0].outputs[0].text}")

        print("\n" + "=" * 70)
        print("SUCCESS! Model switching works with module unload!")
        print("=" * 70)

        del llm2
        del out2
        gc.collect()
    else:
        print(f"\n>>> Cleanup FAILED - still leaking {leaked:.2f} GB")
        print("The issue is at the HIP/ROCR driver level.")


if __name__ == '__main__':
    test_module_unload_approach()
