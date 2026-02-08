#!/usr/bin/env python3
"""
Memory cleanup fix for vLLM on ROCm 780M.

This module provides functions to properly release GPU memory after
deleting a vLLM LLM instance, working around circular reference issues.
"""

import gc
import sys
import weakref
import torch


def clear_model_weights(model):
    """
    Recursively clear all parameter and buffer data from a model.
    This breaks circular references that prevent garbage collection.
    """
    if model is None:
        return

    try:
        # First, set all gradients to None
        for param in model.parameters():
            try:
                param.grad = None
            except:
                pass

        # Clear _parameters and _buffers dicts in all modules
        # This removes references without moving data
        for module in list(model.modules()):
            try:
                # Clear parameters dict
                for key in list(module._parameters.keys()):
                    module._parameters[key] = None
                # Clear buffers dict
                for key in list(module._buffers.keys()):
                    module._buffers[key] = None
                # Clear child modules
                for key in list(module._modules.keys()):
                    module._modules[key] = None
            except:
                pass
    except:
        pass


def clear_gpu_tensors_in_gc():
    """
    Find and break references to GPU tensors remaining in gc.
    Returns the number of tensors where references were cleared.
    """
    cleared = 0
    # Multiple passes to handle nested references
    for _ in range(3):
        tensors_to_clear = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 0:
                    tensors_to_clear.append(obj)
            except:
                pass

        if not tensors_to_clear:
            break

        for tensor in tensors_to_clear:
            try:
                # Break referrer chains by setting dict/list entries to None
                referrers = gc.get_referrers(tensor)
                for ref in referrers:
                    try:
                        if isinstance(ref, dict):
                            keys = [k for k, v in list(ref.items()) if v is tensor]
                            for k in keys:
                                ref[k] = None
                            cleared += 1
                        elif isinstance(ref, list):
                            for i, v in enumerate(ref):
                                if v is tensor:
                                    ref[i] = None
                            cleared += 1
                    except:
                        pass
            except:
                pass

        gc.collect()

    return cleared


def cleanup_vllm_engine(llm):
    """
    Properly cleanup a vLLM LLM instance before deletion.
    Call this before `del llm` to ensure GPU memory is released.

    Usage:
        cleanup_vllm_engine(llm)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    """
    if llm is None:
        return

    try:
        # Get the model from the engine
        engine = getattr(llm, 'llm_engine', None)
        if engine is None:
            return

        # V1 engine path
        engine_core = getattr(engine, 'engine_core', None)
        if engine_core is not None:
            model_executor = getattr(engine_core, 'model_executor', None)
            if model_executor is not None:
                # Clear worker model
                driver_worker = getattr(model_executor, 'driver_worker', None)
                if driver_worker is not None:
                    worker = getattr(driver_worker, 'worker', None)
                    if worker is not None:
                        model_runner = getattr(worker, 'model_runner', None)
                        if model_runner is not None:
                            model = getattr(model_runner, 'model', None)
                            if model is not None:
                                clear_model_weights(model)
                                model_runner.model = None

                            # Clear KV cache
                            kv_caches = getattr(model_runner, 'kv_caches', None)
                            if kv_caches is not None:
                                for i, cache in enumerate(kv_caches):
                                    if cache is not None:
                                        try:
                                            cache.data = torch.empty(0, device='cpu')
                                        except:
                                            pass
                                    kv_caches[i] = None
                                model_runner.kv_caches = None

        # V0 engine path (fallback)
        model_executor = getattr(engine, 'model_executor', None)
        if model_executor is not None:
            driver_worker = getattr(model_executor, 'driver_worker', None)
            if driver_worker is not None:
                model_runner = getattr(driver_worker, 'model_runner', None)
                if model_runner is not None:
                    model = getattr(model_runner, 'model', None)
                    if model is not None:
                        clear_model_weights(model)

    except Exception as e:
        pass  # Best effort cleanup


def full_memory_cleanup(llm=None, shutdown_ray=False):
    """
    Complete memory cleanup for vLLM.

    Args:
        llm: Optional LLM instance to cleanup before deletion
        shutdown_ray: Whether to shutdown Ray

    Usage:
        full_memory_cleanup(llm)
        del llm
        # Memory should now be released
    """
    # Step 1: Clean up the LLM engine
    if llm is not None:
        cleanup_vllm_engine(llm)

    # Step 2: Standard vLLM cleanup
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=shutdown_ray)
    except:
        pass

    gc.collect()

    # Step 3: Clear remaining GPU tensors
    cleared = clear_gpu_tensors_in_gc()

    # Step 4: Reset torch dynamo (compilation cache)
    try:
        import torch._dynamo
        torch._dynamo.reset()
    except:
        pass

    # Step 5: Clear CUDA caches
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Step 6: Final gc pass
    gc.collect()

    return cleared


def test_memory_fix():
    """Test that the memory fix works."""
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

    def get_memory():
        return torch.cuda.mem_get_info()[0] / (1024**3)

    print("=" * 70)
    print("TESTING MEMORY FIX")
    print("=" * 70)

    initial = get_memory()
    print(f"\nInitial: {initial:.2f} GB free")

    from vllm import LLM, SamplingParams

    print("\n>>> Loading first model...")
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

    after_load = get_memory()
    print(f"After load: {after_load:.2f} GB free (used {initial - after_load:.2f} GB)")

    # Generate
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text}")

    # Cleanup with fix
    print("\n>>> Applying memory fix...")
    cleared = full_memory_cleanup(llm)
    print(f"Cleared {cleared} GPU tensors")

    del llm
    del out
    gc.collect()
    torch.cuda.empty_cache()

    time.sleep(1.0)
    gc.collect()
    torch.cuda.empty_cache()

    after_cleanup = get_memory()
    leaked = initial - after_cleanup
    print(f"\nAfter cleanup: {after_cleanup:.2f} GB free")
    print(f"Memory leaked: {leaked:.2f} GB")

    # Check remaining tensors
    gpu_tensors = sum(1 for obj in gc.get_objects()
                      if torch.is_tensor(obj) and obj.is_cuda)
    print(f"GPU tensors in gc: {gpu_tensors}")

    if leaked < 2.0:
        print("\n>>> Memory fix SUCCESS! Trying second model...")

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
        print("SUCCESS! Model switching works with memory fix!")
        print("=" * 70)

        full_memory_cleanup(llm2)
        del llm2
        gc.collect()

    else:
        print(f"\n>>> Memory fix FAILED - still leaking {leaked:.2f} GB")
        print("The issue is at the driver level, not Python level.")


if __name__ == '__main__':
    test_memory_fix()
