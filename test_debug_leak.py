#!/usr/bin/env python3
"""
Debug script to find the exact cause of GPU hangs after multiple switches.
Tracks all memory (GPU, system RAM, PyTorch internals) through each switch.
"""

import gc
import os
import sys
import time
import types
import traceback

# Environment setup
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# Fake torchvision module
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch
import psutil


def get_all_memory_stats():
    """Get comprehensive memory stats."""
    stats = {}

    # GPU memory from torch.cuda
    if torch.cuda.is_available():
        stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        stats['gpu_max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)

        # Direct GPU query
        free, total = torch.cuda.mem_get_info()
        stats['gpu_used_gb'] = (total - free) / (1024**3)
        stats['gpu_free_gb'] = free / (1024**3)
        stats['gpu_total_gb'] = total / (1024**3)

    # System memory
    vm = psutil.virtual_memory()
    stats['sys_used_gb'] = vm.used / (1024**3)
    stats['sys_available_gb'] = vm.available / (1024**3)
    stats['sys_total_gb'] = vm.total / (1024**3)
    stats['sys_percent'] = vm.percent

    # Swap
    swap = psutil.swap_memory()
    stats['swap_used_gb'] = swap.used / (1024**3)
    stats['swap_total_gb'] = swap.total / (1024**3)

    # Process memory
    proc = psutil.Process()
    stats['proc_rss_gb'] = proc.memory_info().rss / (1024**3)
    stats['proc_vms_gb'] = proc.memory_info().vms / (1024**3)

    return stats


def print_memory_stats(label, stats):
    """Print memory stats in a readable format."""
    print(f"\n{'='*60}")
    print(f"MEMORY: {label}")
    print(f"{'='*60}")
    print(f"GPU:  allocated={stats['gpu_allocated_gb']:.2f}GB  reserved={stats['gpu_reserved_gb']:.2f}GB  used={stats['gpu_used_gb']:.2f}GB  free={stats['gpu_free_gb']:.2f}GB")
    print(f"SYS:  used={stats['sys_used_gb']:.1f}GB  available={stats['sys_available_gb']:.1f}GB  ({stats['sys_percent']:.1f}%)")
    print(f"SWAP: used={stats['swap_used_gb']:.2f}GB / {stats['swap_total_gb']:.2f}GB")
    print(f"PROC: rss={stats['proc_rss_gb']:.2f}GB  vms={stats['proc_vms_gb']:.2f}GB")


def count_cuda_tensors():
    """Count CUDA tensors tracked by garbage collector."""
    gc.collect()
    cuda_tensors = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                cuda_tensors.append({
                    'shape': tuple(obj.shape),
                    'dtype': str(obj.dtype),
                    'size_mb': obj.numel() * obj.element_size() / (1024**2),
                    'device': str(obj.device),
                })
        except Exception:
            pass
    return cuda_tensors


def count_vllm_objects():
    """Count vLLM-related objects in memory."""
    gc.collect()
    counts = {}
    for obj in gc.get_objects():
        try:
            type_name = type(obj).__name__
            module = type(obj).__module__ or ''
            if 'vllm' in module.lower():
                key = f"{module}.{type_name}"
                counts[key] = counts.get(key, 0) + 1
        except Exception:
            pass
    return counts


def comprehensive_cleanup(llm):
    """Comprehensive cleanup with detailed logging."""
    import torch._dynamo
    import multiprocessing

    print("\n--- Starting cleanup ---")

    # Step 1: Clear model weights and KV cache
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
                            param_count = sum(1 for _ in model.parameters())
                            print(f"  Clearing {param_count} model parameters...")
                            for param in model.parameters():
                                param.data = torch.empty(0, device='cpu')
                            print(f"  Model parameters cleared")

                        # Clear KV cache list
                        if hasattr(model_runner, 'kv_caches'):
                            kv_count = len(model_runner.kv_caches)
                            print(f"  Clearing {kv_count} kv_caches entries...")
                            for i, cache in enumerate(model_runner.kv_caches):
                                if cache is not None and hasattr(cache, 'device') and cache.device.type == 'cuda':
                                    model_runner.kv_caches[i] = torch.empty(0, device='cpu')
                            model_runner.kv_caches.clear()

                        # Clear KV caches from attention layers
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

                        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error clearing model/KV cache: {e}")
        traceback.print_exc()

    # Step 2: Delete LLM
    print("  Deleting LLM instance...")
    del llm

    # Step 3: Clear vLLM global state
    try:
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            sfc = vllm_config.compilation_config.static_forward_context
            print(f"  Clearing static_forward_context ({len(sfc)} entries)...")
            sfc.clear()
    except Exception as e:
        print(f"  Could not clear static_forward_context: {e}")

    try:
        import vllm.config.vllm as vllm_config_module
        vllm_config_module._current_vllm_config = None
        vllm_config_module._current_prefix = None
        vllm_config_module.get_cached_compilation_config.cache_clear()
        print("  Reset _current_vllm_config")
    except Exception as e:
        print(f"  Could not reset _current_vllm_config: {e}")

    # Step 4: Reset torch dynamo
    try:
        torch._dynamo.reset()
        print("  Reset torch._dynamo")
    except Exception as e:
        print(f"  torch._dynamo.reset failed: {e}")

    # Step 5: vLLM cleanup
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
        print("  Called cleanup_dist_env_and_memory")
    except Exception as e:
        print(f"  cleanup_dist_env_and_memory failed: {e}")

    # Step 6: Workspace manager
    try:
        from vllm.v1.worker.workspace import reset_workspace_manager
        reset_workspace_manager()
        print("  Reset workspace_manager")
    except Exception as e:
        print(f"  reset_workspace_manager failed: {e}")

    # Step 7: Clear various caches
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY
        if hasattr(MULTIMODAL_REGISTRY, '_processor_cache'):
            MULTIMODAL_REGISTRY._processor_cache.clear()
        if hasattr(MULTIMODAL_REGISTRY, 'clear_cache'):
            MULTIMODAL_REGISTRY.clear_cache()
        print("  Cleared multimodal registry")
    except Exception:
        pass

    try:
        from vllm.engine.arg_utils import EngineArgs
        if hasattr(EngineArgs, '_get_default_values'):
            EngineArgs._get_default_values.cache_clear()
        print("  Cleared EngineArgs cache")
    except Exception:
        pass

    # Step 8: Wait for child processes
    for child in multiprocessing.active_children():
        print(f"  Waiting for child process {child.name} (pid={child.pid})")
        child.join(timeout=5.0)
        if child.is_alive():
            print(f"  Force terminating {child.name}")
            child.terminate()
            child.join(timeout=2.0)

    # Step 9: Aggressive GC
    gc.collect()
    gc.collect()
    gc.collect()

    # Step 10: Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    try:
        torch._C._host_emptyCache()
    except AttributeError:
        pass

    gc.collect()
    print("--- Cleanup complete ---")


def main():
    from vllm import LLM, SamplingParams

    print("="*70)
    print("DEBUG: Finding the exact cause of GPU hangs")
    print("="*70)

    # Warm up
    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    initial_stats = get_all_memory_stats()
    print_memory_stats("INITIAL", initial_stats)

    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    # Alternate between two 7B models
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    # Track memory history
    memory_history = [initial_stats.copy()]

    for switch_num in range(1, 8):  # Try up to 7 switches
        model = models[(switch_num - 1) % 2]

        print(f"\n{'#'*70}")
        print(f"# SWITCH {switch_num}: Loading {model}")
        print(f"{'#'*70}")

        before_load = get_all_memory_stats()
        print_memory_stats(f"Before load {switch_num}", before_load)

        # Count objects before
        cuda_tensors_before = count_cuda_tensors()
        vllm_objects_before = count_vllm_objects()

        try:
            # Load model
            print(f"\nLoading model...")
            t0 = time.perf_counter()
            llm = LLM(model=model, **config)
            load_time = time.perf_counter() - t0
            print(f"Model loaded in {load_time:.2f}s")

            after_load = get_all_memory_stats()
            print_memory_stats(f"After load {switch_num}", after_load)

            # Generate
            print("\nGenerating...")
            outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=10, temperature=0.7))
            text = outputs[0].outputs[0].text if outputs else ""
            print(f"Generated: {text[:50]}...")

            after_gen = get_all_memory_stats()
            print_memory_stats(f"After generation {switch_num}", after_gen)

            # Cleanup
            comprehensive_cleanup(llm)

            after_cleanup = get_all_memory_stats()
            print_memory_stats(f"After cleanup {switch_num}", after_cleanup)

            # Count objects after
            cuda_tensors_after = count_cuda_tensors()
            vllm_objects_after = count_vllm_objects()

            # Report deltas
            print(f"\n--- Delta from initial ---")
            print(f"GPU allocated: {after_cleanup['gpu_allocated_gb'] - initial_stats['gpu_allocated_gb']:+.2f}GB")
            print(f"GPU reserved:  {after_cleanup['gpu_reserved_gb'] - initial_stats['gpu_reserved_gb']:+.2f}GB")
            print(f"GPU used:      {after_cleanup['gpu_used_gb'] - initial_stats['gpu_used_gb']:+.2f}GB")
            print(f"System RAM:    {after_cleanup['sys_used_gb'] - initial_stats['sys_used_gb']:+.1f}GB")
            print(f"Process RSS:   {after_cleanup['proc_rss_gb'] - initial_stats['proc_rss_gb']:+.2f}GB")

            print(f"\n--- CUDA tensors still in memory: {len(cuda_tensors_after)} ---")
            total_tensor_mb = sum(t['size_mb'] for t in cuda_tensors_after)
            print(f"Total tensor memory: {total_tensor_mb:.1f}MB")
            if cuda_tensors_after:
                # Sort by size and show top 10
                sorted_tensors = sorted(cuda_tensors_after, key=lambda x: -x['size_mb'])[:10]
                for t in sorted_tensors:
                    print(f"  {t['shape']} {t['dtype']}: {t['size_mb']:.1f}MB")

            print(f"\n--- vLLM objects still in memory ---")
            for key, count in sorted(vllm_objects_after.items(), key=lambda x: -x[1])[:15]:
                print(f"  {key}: {count}")

            memory_history.append(after_cleanup.copy())

            # Check for concerning accumulation
            if after_cleanup['gpu_allocated_gb'] > 2.0:
                print(f"\n*** WARNING: GPU allocated memory > 2GB after cleanup! ***")

            time.sleep(2)

        except Exception as e:
            print(f"\n*** CRASH on switch {switch_num}: {e} ***")
            traceback.print_exc()
            break

    print(f"\n{'='*70}")
    print("MEMORY HISTORY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Switch':<8} {'GPU Alloc':<12} {'GPU Resv':<12} {'GPU Used':<12} {'Proc RSS':<12}")
    print("-"*60)
    for i, stats in enumerate(memory_history):
        label = "Initial" if i == 0 else f"After {i}"
        print(f"{label:<8} {stats['gpu_allocated_gb']:<12.2f} {stats['gpu_reserved_gb']:<12.2f} {stats['gpu_used_gb']:<12.2f} {stats['proc_rss_gb']:<12.2f}")


if __name__ == "__main__":
    main()
