#!/usr/bin/env python3
"""Deep profiling to find exactly what's holding GPU memory after model cleanup.

This script will:
1. Load a model
2. Profile memory at every step of cleanup
3. Identify what's holding references
4. Find the exact source of the memory leak
"""

import os
import gc
import sys
import ctypes
import weakref

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_gpu_memory():
    """Get detailed GPU memory info."""
    free, total = torch.cuda.mem_get_info()
    return {
        'free_gb': free / 1024**3,
        'total_gb': total / 1024**3,
        'used_gb': (total - free) / 1024**3,
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
    }


def print_gpu_memory(label=""):
    """Print GPU memory status."""
    m = get_gpu_memory()
    print(f"[GPU {label}] Used: {m['used_gb']:.2f}GB, "
          f"Allocated: {m['allocated_gb']:.2f}GB, "
          f"Reserved: {m['reserved_gb']:.2f}GB, "
          f"Free: {m['free_gb']:.2f}GB")


def find_gpu_tensors():
    """Find all GPU tensors currently in memory."""
    gpu_tensors = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device.type == 'cuda':
                gpu_tensors.append({
                    'id': id(obj),
                    'shape': tuple(obj.shape),
                    'dtype': obj.dtype,
                    'size_mb': obj.numel() * obj.element_size() / 1024**2,
                    'refcount': sys.getrefcount(obj),
                })
        except Exception:
            pass
    return gpu_tensors


def find_referrers_for_tensor(tensor_id):
    """Find what's referring to a specific tensor."""
    for obj in gc.get_objects():
        if id(obj) == tensor_id:
            referrers = gc.get_referrers(obj)
            return referrers
    return []


def analyze_vllm_state(llm):
    """Analyze vLLM's internal state to find GPU tensor holders."""
    print("\n=== ANALYZING vLLM INTERNAL STATE ===\n")

    holders = {}

    # Engine
    engine = getattr(llm, 'llm_engine', None)
    if engine:
        print(f"Engine: {type(engine)}")
        holders['engine'] = engine

        # Engine core
        engine_core = getattr(engine, 'engine_core', None)
        if engine_core:
            print(f"  Engine core: {type(engine_core)}")
            holders['engine_core'] = engine_core

            # Model executor
            executor = getattr(engine_core, 'model_executor', None)
            if executor:
                print(f"    Executor: {type(executor)}")
                holders['executor'] = executor

                # Driver worker
                driver_worker = getattr(executor, 'driver_worker', None)
                if driver_worker:
                    print(f"      Driver worker: {type(driver_worker)}")
                    holders['driver_worker'] = driver_worker

                    # Inner worker
                    worker = getattr(driver_worker, 'worker', None)
                    if worker:
                        print(f"        Worker: {type(worker)}")
                        holders['worker'] = worker

                        # Model runner
                        model_runner = getattr(worker, 'model_runner', None)
                        if model_runner:
                            print(f"          Model runner: {type(model_runner)}")
                            holders['model_runner'] = model_runner

                            # Model
                            model = getattr(model_runner, 'model', None)
                            if model:
                                print(f"            Model: {type(model)}")
                                holders['model'] = model

                                # Count parameters
                                num_params = sum(1 for _ in model.parameters())
                                param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
                                print(f"            Parameters: {num_params}, Size: {param_size:.2f}GB")

                            # KV caches
                            kv_caches = getattr(model_runner, 'kv_caches', None)
                            if kv_caches:
                                print(f"          KV caches: {len(kv_caches)} entries")
                                kv_size = 0
                                for kv in kv_caches:
                                    if kv is not None:
                                        if hasattr(kv, 'numel'):
                                            kv_size += kv.numel() * kv.element_size()
                                print(f"            KV cache size: {kv_size / 1024**3:.2f}GB")

                            # Static forward context
                            ctx = getattr(model_runner, 'static_forward_context', None)
                            if ctx:
                                print(f"          Static forward context: {len(ctx)} entries")
                                for name, layer in list(ctx.items())[:3]:
                                    print(f"            {name}: {type(layer)}")

    return holders


def profile_cleanup_steps(llm):
    """Profile memory at each cleanup step."""
    print("\n=== STEP-BY-STEP CLEANUP PROFILING ===\n")

    print_gpu_memory("initial")

    engine = getattr(llm, 'llm_engine', None)
    engine_core = getattr(engine, 'engine_core', None) if engine else None
    executor = getattr(engine_core, 'model_executor', None) if engine_core else None
    driver_worker = getattr(executor, 'driver_worker', None) if executor else None
    worker = getattr(driver_worker, 'worker', None) if driver_worker else None
    model_runner = getattr(worker, 'model_runner', None) if worker else None
    model = getattr(model_runner, 'model', None) if model_runner else None

    # Step 1: Clear KV caches
    print("\n--- Step 1: Clear KV caches ---")
    if model_runner and hasattr(model_runner, 'kv_caches'):
        for kv in model_runner.kv_caches:
            if kv is not None and hasattr(kv, 'data'):
                kv.data = torch.empty(0, device='cpu')
        model_runner.kv_caches.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after KV clear")

    # Step 2: Clear static forward context
    print("\n--- Step 2: Clear static forward context ---")
    if model_runner and hasattr(model_runner, 'static_forward_context'):
        ctx = model_runner.static_forward_context
        for name in list(ctx.keys()):
            layer = ctx[name]
            if hasattr(layer, 'kv_cache'):
                layer.kv_cache.clear()
        ctx.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after context clear")

    # Step 3: Clear model parameters
    print("\n--- Step 3: Clear model parameters ---")
    if model:
        for name, param in list(model.named_parameters()):
            if param.device.type == 'cuda':
                param.data = torch.empty(0, device='cpu')
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after params clear")

    # Step 4: Clear model buffers
    print("\n--- Step 4: Clear model buffers ---")
    if model:
        for name, buf in list(model.named_buffers()):
            if buf.device.type == 'cuda':
                try:
                    buf.data = torch.empty(0, device='cpu')
                except:
                    pass
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after buffers clear")

    # Step 5: Delete model reference
    print("\n--- Step 5: Delete model reference ---")
    if model_runner:
        model_runner.model = None
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after model delete")

    # Step 6: Delete model_runner reference
    print("\n--- Step 6: Delete model_runner reference ---")
    if worker:
        worker.model_runner = None
    del model_runner
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after model_runner delete")

    # Step 7: Call engine_core.shutdown()
    print("\n--- Step 7: Call engine_core.shutdown() ---")
    if engine_core:
        try:
            engine_core.shutdown()
        except Exception as e:
            print(f"  shutdown error: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after shutdown")

    # Step 8: Clear engine references
    print("\n--- Step 8: Clear engine references ---")
    if engine:
        engine.engine_core = None
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after engine clear")

    # Step 9: Delete llm object
    print("\n--- Step 9: Delete llm object ---")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after llm delete")

    # Step 10: Multiple GC passes
    print("\n--- Step 10: Multiple GC passes ---")
    for i in range(5):
        gc.collect()
        torch.cuda.empty_cache()
    print_gpu_memory("after 5x GC")

    # Step 11: Check remaining GPU tensors
    print("\n--- Step 11: Check remaining GPU tensors ---")
    gpu_tensors = find_gpu_tensors()
    print(f"Found {len(gpu_tensors)} GPU tensors still in memory")
    total_size = sum(t['size_mb'] for t in gpu_tensors)
    print(f"Total size: {total_size / 1024:.2f}GB")

    # Show largest tensors
    sorted_tensors = sorted(gpu_tensors, key=lambda x: x['size_mb'], reverse=True)
    print("\nLargest remaining tensors:")
    for t in sorted_tensors[:20]:
        print(f"  {t['shape']} {t['dtype']} - {t['size_mb']:.1f}MB (refcount: {t['refcount']})")


def check_vllm_globals():
    """Check vLLM global/module-level state that might hold GPU memory."""
    print("\n=== CHECKING vLLM GLOBAL STATE ===\n")

    # Check RoPE cache
    try:
        from vllm.model_executor.layers import rotary_embedding
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rope_dict = rotary_embedding._ROPE_DICT
            print(f"RoPE cache (_ROPE_DICT): {len(rope_dict)} entries")
            for key, value in list(rope_dict.items())[:3]:
                print(f"  {key}: {type(value)}")
    except Exception as e:
        print(f"RoPE check failed: {e}")

    # Check parallel state
    try:
        from vllm.distributed import parallel_state
        print(f"\nParallel state module: {parallel_state}")

        # Check for various parallel groups
        attrs = ['_WORLD_GROUP', '_MODEL_PARALLEL_GROUP', '_TENSOR_PARALLEL_GROUP',
                 '_PIPELINE_PARALLEL_GROUP', '_DATA_PARALLEL_GROUP']
        for attr in attrs:
            if hasattr(parallel_state, attr):
                val = getattr(parallel_state, attr)
                if val is not None:
                    print(f"  {attr}: {type(val)}")
    except Exception as e:
        print(f"Parallel state check failed: {e}")

    # Check attention backends
    try:
        from vllm.attention.backends import flash_attn
        if hasattr(flash_attn, 'FlashAttentionBackend'):
            backend = flash_attn.FlashAttentionBackend
            print(f"\nFlash attention backend: {backend}")
    except Exception as e:
        print(f"Attention backend check failed: {e}")

    # Check workspace manager
    try:
        from vllm.model_executor.layers.fused_moe import workspace
        if hasattr(workspace, 'WorkspaceManager'):
            mgr = workspace.WorkspaceManager
            print(f"\nWorkspace manager: {mgr}")
            if hasattr(mgr, '_instance'):
                print(f"  Instance: {mgr._instance}")
    except Exception as e:
        print(f"Workspace check failed: {e}")

    # Check for any torch.nn.Module subclasses with GPU tensors
    print("\n--- Checking for leaked nn.Module instances ---")
    module_count = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.nn.Module):
                has_cuda = False
                for param in obj.parameters():
                    if param.device.type == 'cuda':
                        has_cuda = True
                        break
                if has_cuda:
                    module_count += 1
                    if module_count <= 5:
                        print(f"  Module with CUDA params: {type(obj).__name__}")
        except:
            pass
    print(f"Total nn.Module instances with CUDA params: {module_count}")


def main():
    from vllm import LLM, SamplingParams

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print("=" * 70)
    print("MEMORY LEAK PROFILING")
    print("=" * 70)

    print_gpu_memory("baseline")

    # Load model
    print("\n=== LOADING MODEL ===")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print_gpu_memory("after load")

    # Test inference
    out = llm.generate(["2+2="], SamplingParams(max_tokens=10))
    print(f"Test output: {out[0].outputs[0].text.strip()[:30]}")
    print_gpu_memory("after inference")

    # Analyze vLLM state
    analyze_vllm_state(llm)

    # Check global state before cleanup
    check_vllm_globals()

    # Profile cleanup
    profile_cleanup_steps(llm)

    # Check global state after cleanup
    print("\n=== CHECKING GLOBAL STATE AFTER CLEANUP ===")
    check_vllm_globals()

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)
    print_gpu_memory("final")


if __name__ == "__main__":
    main()
