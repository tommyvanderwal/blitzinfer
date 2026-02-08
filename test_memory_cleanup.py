#!/usr/bin/env python3
"""
Test memory cleanup between model loads.
"""

import gc
import os
import sys
import time
import types

# Environment setup for ROCm
os.environ['HIP_VISIBLE_DEVICES'] = '0'
# Use single-process mode for fast switching - memory leak fixed by proper cleanup
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
# Don't skip warmup - it can cause instability
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# Fake torchvision module
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch


def get_mem():
    """Get GPU memory usage."""
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**3)
    return used, free / (1024**3)


def get_detailed_mem():
    """Get detailed CUDA memory stats."""
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**3)
    return {
        "total_used": used,
        "pytorch_allocated": allocated,
        "pytorch_reserved": reserved,
        "external": used - reserved,  # Memory used by other processes or driver
    }


def comprehensive_cleanup():
    """Comprehensive cleanup of vLLM state."""
    import gc
    import torch
    import torch._dynamo
    import multiprocessing

    # CRITICAL: Clear static_forward_context which holds model layer references
    try:
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            vllm_config.compilation_config.static_forward_context.clear()
            print("  Cleared static_forward_context")
    except Exception as e:
        print(f"  Could not clear static_forward_context: {e}")

    # CRITICAL: Reset the global _current_vllm_config
    try:
        import vllm.config.vllm as vllm_config_module
        vllm_config_module._current_vllm_config = None
        vllm_config_module._current_prefix = None
        vllm_config_module.get_cached_compilation_config.cache_clear()
        print("  Reset _current_vllm_config")
    except Exception as e:
        print(f"  Could not reset _current_vllm_config: {e}")

    # Reset torch.compile / dynamo state (holds compiled model closures)
    try:
        torch._dynamo.reset()
        print("  Reset torch._dynamo")
    except Exception as e:
        print(f"  torch._dynamo.reset failed: {e}")

    # Use vLLM's comprehensive cleanup function
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except Exception as e:
        print(f"cleanup_dist_env_and_memory failed: {e}")
        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
                destroy_distributed_environment,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass

    # Reset workspace manager (holds GPU memory)
    try:
        from vllm.v1.worker.workspace import reset_workspace_manager
        reset_workspace_manager()
    except Exception as e:
        print(f"reset_workspace_manager failed: {e}")

    # Reset environment variable cache
    try:
        import vllm.envs as envs
        envs.disable_envs_cache()
    except Exception:
        pass

    # Clear multimodal registry caches
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY
        if hasattr(MULTIMODAL_REGISTRY, '_processor_cache'):
            MULTIMODAL_REGISTRY._processor_cache.clear()
        if hasattr(MULTIMODAL_REGISTRY, 'clear_cache'):
            MULTIMODAL_REGISTRY.clear_cache()
    except Exception as e:
        print(f"  Could not clear multimodal registry: {e}")

    # Clear EngineArgs lru_cache
    try:
        from vllm.engine.arg_utils import EngineArgs
        if hasattr(EngineArgs, '_get_default_values'):
            EngineArgs._get_default_values.cache_clear()
    except Exception as e:
        print(f"  Could not clear EngineArgs cache: {e}")

    # Wait for child processes
    for child in multiprocessing.active_children():
        print(f"Waiting for child process {child.name} (pid={child.pid})")
        child.join(timeout=5.0)
        if child.is_alive():
            print(f"Force terminating {child.name}")
            child.terminate()
            child.join(timeout=2.0)

    # Aggressive GC
    gc.collect()
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Try to empty host cache
    try:
        torch._C._host_emptyCache()
    except AttributeError:
        pass

    gc.collect()


def main():
    from vllm import LLM, SamplingParams

    print("=" * 70)
    print("MEMORY CLEANUP TEST")
    print("=" * 70)

    # Warm up CUDA
    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    used, free = get_mem()
    print(f"\nInitial GPU: {used:.1f}GB used, {free:.1f}GB free")

    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    for i, model in enumerate(models):
        print(f"\n{'='*70}")
        print(f"LOAD {i+1}: {model}")
        print("=" * 70)

        used_before, free_before = get_mem()
        print(f"Before load: {used_before:.1f}GB used, {free_before:.1f}GB free")

        # Load model
        t0 = time.perf_counter()
        llm = LLM(model=model, **config)
        load_time = time.perf_counter() - t0

        used_after_load, free_after_load = get_mem()
        print(f"After load: {used_after_load:.1f}GB used, {free_after_load:.1f}GB free")
        print(f"Load time: {load_time:.2f}s")

        # Simple generation
        outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=10, temperature=0.7))
        text = outputs[0].outputs[0].text if outputs else ""
        print(f"Generated: {text[:50]}...")

        # Cleanup
        print("\nCleaning up...")

        # Move model weights and KV cache to CPU to force GPU memory release
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
                            # Clear model parameters
                            if hasattr(model_runner, 'model'):
                                model = model_runner.model
                                print("  Moving model parameters to CPU...")
                                for param in model.parameters():
                                    param.data = torch.empty(0, device='cpu')
                                print("  Model parameters cleared")

                            # Clear KV cache list on model_runner
                            if hasattr(model_runner, 'kv_caches'):
                                print(f"  Clearing kv_caches ({len(model_runner.kv_caches)} entries)...")
                                for i, cache in enumerate(model_runner.kv_caches):
                                    if cache is not None and hasattr(cache, 'device') and cache.device.type == 'cuda':
                                        model_runner.kv_caches[i] = torch.empty(0, device='cpu')
                                model_runner.kv_caches.clear()
                                print("  kv_caches cleared")

                            # Clear KV caches from attention layers in compilation config
                            if hasattr(model_runner, 'compilation_config'):
                                sfc = getattr(model_runner.compilation_config, 'static_forward_context', None)
                                if sfc:
                                    print(f"  Clearing KV cache from {len(sfc)} attention layers...")
                                    for layer_name, layer in sfc.items():
                                        if hasattr(layer, 'kv_cache') and layer.kv_cache:
                                            for j, kv in enumerate(layer.kv_cache):
                                                if kv is not None and hasattr(kv, 'device') and kv.device.type == 'cuda':
                                                    layer.kv_cache[j] = torch.empty(0, device='cpu')
                                            layer.kv_cache = []
                                    print("  Attention layer KV caches cleared")

                            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Could not clear model/KV cache: {e}")

        del llm
        comprehensive_cleanup()

        used_after_cleanup, free_after_cleanup = get_mem()
        mem_details = get_detailed_mem()
        print(f"After cleanup: {used_after_cleanup:.1f}GB used, {free_after_cleanup:.1f}GB free")
        print(f"  PyTorch allocated: {mem_details['pytorch_allocated']:.2f}GB")
        print(f"  PyTorch reserved:  {mem_details['pytorch_reserved']:.2f}GB")
        print(f"  External/driver:   {mem_details['external']:.2f}GB")

        leaked = used_after_cleanup - used_before
        if leaked > 1.0:
            print(f"WARNING: {leaked:.1f}GB leaked!")
        else:
            print(f"Memory change: {leaked:+.1f}GB")

        # Pause between loads
        time.sleep(2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    used, free = get_mem()
    print(f"Final GPU: {used:.1f}GB used, {free:.1f}GB free")


if __name__ == "__main__":
    main()
