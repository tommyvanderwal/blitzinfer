#!/usr/bin/env python3
"""Hunt down the remaining ~10GB memory drift per switch cycle.

Goal: Find EXACTLY what's holding memory after cleanup and eliminate it.
The GPU is dedicated - no display, no other processes. We should get to ~0GB.
"""

import os
import gc
import sys
import ctypes

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
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
    }


def log_mem(label):
    m = get_gpu_memory()
    print(f"[{label}] Used: {m['used_gb']:.2f}GB, Alloc: {m['allocated_gb']:.2f}GB, "
          f"Reserved: {m['reserved_gb']:.2f}GB, Free: {m['free_gb']:.2f}GB")
    return m


def find_all_cuda_tensors():
    """Find ALL CUDA tensors in memory."""
    tensors = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device.type == 'cuda':
                size_mb = obj.numel() * obj.element_size() / 1024**2
                tensors.append({
                    'id': id(obj),
                    'shape': tuple(obj.shape),
                    'dtype': str(obj.dtype),
                    'size_mb': size_mb,
                    'requires_grad': obj.requires_grad,
                    'refcount': sys.getrefcount(obj),
                })
        except Exception:
            pass
    return tensors


def find_all_modules_with_cuda():
    """Find all nn.Module instances with CUDA parameters."""
    modules = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.nn.Module):
                cuda_params = sum(1 for p in obj.parameters() if p.device.type == 'cuda')
                cuda_buffers = sum(1 for b in obj.buffers() if b.device.type == 'cuda')
                if cuda_params > 0 or cuda_buffers > 0:
                    modules.append({
                        'type': type(obj).__name__,
                        'cuda_params': cuda_params,
                        'cuda_buffers': cuda_buffers,
                    })
        except Exception:
            pass
    return modules


def check_vllm_global_state():
    """Check ALL vLLM global/module-level state."""
    print("\n=== CHECKING vLLM GLOBAL STATE ===\n")

    findings = []

    # 1. RoPE cache
    try:
        from vllm.model_executor.layers import rotary_embedding
        if hasattr(rotary_embedding, '_ROPE_DICT') and rotary_embedding._ROPE_DICT:
            findings.append(f"RoPE cache: {len(rotary_embedding._ROPE_DICT)} entries")
            for key in list(rotary_embedding._ROPE_DICT.keys())[:3]:
                findings.append(f"  - {key}")
    except Exception as e:
        findings.append(f"RoPE check error: {e}")

    # 2. WorkspaceManager
    try:
        from vllm.model_executor.layers.fused_moe import workspace
        if hasattr(workspace, 'WorkspaceManager'):
            mgr = workspace.WorkspaceManager
            if hasattr(mgr, '_instance') and mgr._instance is not None:
                findings.append(f"WorkspaceManager: instance exists")
            if hasattr(mgr, '_instances') and mgr._instances:
                findings.append(f"WorkspaceManager: {len(mgr._instances)} device instances")
    except Exception as e:
        findings.append(f"WorkspaceManager check error: {e}")

    # 3. Parallel state
    try:
        from vllm.distributed import parallel_state
        attrs = ['_WORLD_GROUP', '_MODEL_PARALLEL_GROUP', '_TENSOR_PARALLEL_GROUP',
                 '_PIPELINE_PARALLEL_GROUP', '_DATA_PARALLEL_GROUP', '_LOCAL_RANK']
        for attr in attrs:
            if hasattr(parallel_state, attr):
                val = getattr(parallel_state, attr)
                if val is not None:
                    findings.append(f"parallel_state.{attr}: {type(val).__name__}")
    except Exception as e:
        findings.append(f"Parallel state check error: {e}")

    # 4. torch.distributed
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            findings.append(f"torch.distributed: INITIALIZED (world_size={dist.get_world_size()})")
        else:
            findings.append("torch.distributed: not initialized")
    except Exception as e:
        findings.append(f"torch.distributed check error: {e}")

    # 5. Attention backends
    try:
        from vllm.attention.backends import flash_attn
        if hasattr(flash_attn, '_CACHED_ATTENTION') and flash_attn._CACHED_ATTENTION:
            findings.append("Flash attention: cached")
    except:
        pass

    try:
        from vllm.attention.backends import triton_attn
        if hasattr(triton_attn, '_CACHED_KERNELS') and triton_attn._CACHED_KERNELS:
            findings.append(f"Triton attention: {len(triton_attn._CACHED_KERNELS)} cached kernels")
    except:
        pass

    # 6. Custom ops cache
    try:
        from vllm.model_executor.custom_op import custom_ops
        if hasattr(custom_ops, '_CACHED_OPS') and custom_ops._CACHED_OPS:
            findings.append(f"Custom ops: {len(custom_ops._CACHED_OPS)} cached")
    except:
        pass

    # 7. Quantization caches
    try:
        from vllm.model_executor.layers.quantization import marlin
        if hasattr(marlin, '_CACHED_WORKSPACE') and marlin._CACHED_WORKSPACE:
            findings.append("Marlin: workspace cached")
    except:
        pass

    # 8. Model registry
    try:
        from vllm.model_executor.model_loader import model_loader
        if hasattr(model_loader, '_MODEL_REGISTRY') and model_loader._MODEL_REGISTRY:
            findings.append(f"Model registry: {len(model_loader._MODEL_REGISTRY)} entries")
    except:
        pass

    # 9. Tokenizer cache
    try:
        from transformers import AutoTokenizer
        if hasattr(AutoTokenizer, '_tokenizers') and AutoTokenizer._tokenizers:
            findings.append(f"Tokenizer cache: {len(AutoTokenizer._tokenizers)} entries")
    except:
        pass

    # 10. HuggingFace cache
    try:
        from huggingface_hub import constants
        # HF caches various things
    except:
        pass

    for f in findings:
        print(f"  {f}")

    return findings


def check_pytorch_cuda_state():
    """Check PyTorch CUDA internal state."""
    print("\n=== CHECKING PyTorch CUDA STATE ===\n")

    # Memory stats
    print("Memory allocator stats:")
    stats = torch.cuda.memory_stats()
    important_stats = [
        'allocated_bytes.all.current',
        'reserved_bytes.all.current',
        'active_bytes.all.current',
        'inactive_split_bytes.all.current',
        'num_alloc_retries',
    ]
    for stat in important_stats:
        if stat in stats:
            val = stats[stat]
            if 'bytes' in stat:
                print(f"  {stat}: {val / 1024**3:.2f}GB")
            else:
                print(f"  {stat}: {val}")

    # Check for CUDA graphs
    print("\nCUDA graphs:")
    try:
        # CUDA graphs can hold memory
        print(f"  (No direct API to check CUDA graph memory)")
    except:
        pass

    # Check NCCL
    print("\nNCCL state:")
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            print(f"  Process group initialized: {dist.get_backend()}")
            print(f"  World size: {dist.get_world_size()}")
        else:
            print("  Not initialized")
    except Exception as e:
        print(f"  Error: {e}")


def aggressive_cleanup():
    """Try every possible cleanup method."""
    print("\n=== AGGRESSIVE CLEANUP ===\n")

    log_mem("before cleanup")

    # 1. Clear all vLLM caches
    print("\n--- Step 1: Clear vLLM caches ---")
    try:
        from vllm.model_executor.layers import rotary_embedding
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            for key, rope in list(rotary_embedding._ROPE_DICT.items()):
                if hasattr(rope, 'cos_cached'):
                    if rope.cos_cached is not None and hasattr(rope.cos_cached, 'data'):
                        rope.cos_cached = None
                if hasattr(rope, 'sin_cached'):
                    if rope.sin_cached is not None:
                        rope.sin_cached = None
            rotary_embedding._ROPE_DICT.clear()
            print("  Cleared RoPE cache")
    except Exception as e:
        print(f"  RoPE clear error: {e}")

    try:
        from vllm.model_executor.layers.fused_moe import workspace
        if hasattr(workspace, 'WorkspaceManager'):
            mgr = workspace.WorkspaceManager
            if hasattr(mgr, '_instance'):
                mgr._instance = None
            if hasattr(mgr, '_instances'):
                mgr._instances = {}
            if hasattr(mgr, '_initialized'):
                mgr._initialized = False
            print("  Cleared WorkspaceManager")
    except Exception as e:
        print(f"  WorkspaceManager clear error: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after vLLM caches")

    # 2. Destroy parallel state
    print("\n--- Step 2: Destroy parallel state ---")
    try:
        from vllm.distributed import parallel_state
        if hasattr(parallel_state, 'destroy_model_parallel'):
            parallel_state.destroy_model_parallel()
        if hasattr(parallel_state, 'destroy_distributed_environment'):
            parallel_state.destroy_distributed_environment()

        # Force clear all group references
        for attr in dir(parallel_state):
            if attr.startswith('_') and 'GROUP' in attr:
                try:
                    setattr(parallel_state, attr, None)
                except:
                    pass
        print("  Destroyed vLLM parallel state")
    except Exception as e:
        print(f"  Parallel state error: {e}")

    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
            print("  Destroyed torch process group")
    except Exception as e:
        print(f"  Process group error: {e}")

    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after parallel state")

    # 3. Clear PyTorch CUDA caches
    print("\n--- Step 3: Clear PyTorch CUDA caches ---")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Try to release cached memory
    try:
        torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.0")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.6")
    except:
        pass

    log_mem("after PyTorch cache clear")

    # 4. Force garbage collection with multiple passes
    print("\n--- Step 4: Aggressive GC ---")
    for i in range(10):
        gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log_mem("after 10x GC")

    # 5. Check what's left
    print("\n--- Step 5: Check remaining allocations ---")
    tensors = find_all_cuda_tensors()
    print(f"CUDA tensors: {len(tensors)}")
    total_mb = sum(t['size_mb'] for t in tensors)
    print(f"Total tensor memory: {total_mb:.1f}MB")

    if tensors:
        print("\nRemaining tensors:")
        for t in sorted(tensors, key=lambda x: x['size_mb'], reverse=True)[:10]:
            print(f"  {t['shape']} {t['dtype']} - {t['size_mb']:.1f}MB (refs={t['refcount']})")

    modules = find_all_modules_with_cuda()
    print(f"\nnn.Modules with CUDA: {len(modules)}")
    for m in modules[:10]:
        print(f"  {m['type']}: {m['cuda_params']} params, {m['cuda_buffers']} buffers")

    return log_mem("final")


def try_cuda_device_reset():
    """Try resetting the CUDA device entirely."""
    print("\n=== TRYING CUDA DEVICE RESET ===\n")

    log_mem("before reset")

    print("WARNING: This will destroy all CUDA state!")

    # First sync and clear
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    # Try device reset (this is drastic but effective)
    # Note: This may not work if there are any active CUDA contexts
    try:
        # This releases all CUDA memory on the device
        torch.cuda.reset_peak_memory_stats()

        # Even more aggressive: reset the device
        # WARNING: This destroys ALL CUDA state
        # torch.cuda.device_reset()  # Uncomment to try
        print("  Device reset not attempted (commented out)")
    except Exception as e:
        print(f"  Reset error: {e}")

    gc.collect()
    torch.cuda.empty_cache()

    log_mem("after reset attempt")


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print("=" * 70)
    print("HUNTING DOWN REMAINING MEMORY LEAKS")
    print("=" * 70)

    baseline = log_mem("baseline")
    baseline_used = baseline['used_gb']

    # Check initial state
    check_vllm_global_state()
    check_pytorch_cuda_state()

    # Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=True,
    )
    log_mem("after load")

    # Test inference
    out = llm.generate(["2+2="], SamplingParams(max_tokens=10))
    print(f"Output: {out[0].outputs[0].text.strip()[:30]}")

    # Standard cleanup
    print("\n" + "=" * 70)
    print("STANDARD CLEANUP (full_cleanup)")
    print("=" * 70)

    freed = full_cleanup(llm)
    llm = None
    print(f"Freed: {freed:.1f}GB")
    after_standard = log_mem("after standard cleanup")

    # Check state after standard cleanup
    check_vllm_global_state()
    check_pytorch_cuda_state()

    # Aggressive cleanup
    print("\n" + "=" * 70)
    print("AGGRESSIVE CLEANUP")
    print("=" * 70)

    after_aggressive = aggressive_cleanup()

    # Check state after aggressive cleanup
    print("\n" + "=" * 70)
    print("STATE AFTER AGGRESSIVE CLEANUP")
    print("=" * 70)
    check_vllm_global_state()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Baseline:              {baseline_used:.2f}GB
    After standard cleanup: {after_standard['used_gb']:.2f}GB
    After aggressive:       {after_aggressive['used_gb']:.2f}GB

    Standard freed:         {baseline_used + 47 - after_standard['used_gb']:.1f}GB (approx)
    Remaining leak:         {after_aggressive['used_gb'] - baseline_used:.2f}GB
    """)

    if after_aggressive['used_gb'] - baseline_used > 1.0:
        print("LEAK DETECTED - investigating further...")

        # Try device reset as last resort
        print("\nTrying CUDA device reset...")
        try_cuda_device_reset()
        final = log_mem("after device reset")
        print(f"\nFinal remaining: {final['used_gb'] - baseline_used:.2f}GB above baseline")


if __name__ == "__main__":
    main()
