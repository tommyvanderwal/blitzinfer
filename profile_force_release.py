#!/usr/bin/env python3
"""Force release ALL GPU memory including PyTorch allocator internals.

Key finding: After cleanup, PyTorch reports 7.66GB "allocated" but 0 Python tensors.
The memory is in PyTorch's CUDA caching allocator or internal buffers.

This script tries aggressive methods to release this memory.
"""

import os
import gc
import ctypes

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

# IMPORTANT: Set this BEFORE importing torch to prevent caching
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False,max_split_size_mb:128'

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


def get_vllm_internals(llm):
    """Navigate to vLLM internals."""
    engine = llm.llm_engine
    inproc_client = engine.engine_core
    if hasattr(inproc_client, 'engine_core'):
        engine_core = inproc_client.engine_core
    else:
        engine_core = inproc_client

    result = {'engine_core': engine_core, 'inproc_client': inproc_client}

    if hasattr(engine_core, 'model_executor'):
        executor = engine_core.model_executor
        result['executor'] = executor
        if hasattr(executor, 'driver_worker'):
            driver = executor.driver_worker
            result['driver_worker'] = driver
            if hasattr(driver, 'worker'):
                worker = driver.worker
                result['worker'] = worker
                if hasattr(worker, 'model_runner'):
                    result['model_runner'] = worker.model_runner
                    if hasattr(worker.model_runner, 'model'):
                        result['model'] = worker.model_runner.model

    return result


def try_force_allocator_release():
    """Try various methods to force PyTorch allocator to release memory."""
    print("\n=== FORCE ALLOCATOR RELEASE ===\n")

    log_mem("before")

    # Method 1: Multiple rounds of empty_cache with sync
    print("\n--- Method 1: Multiple sync + empty_cache ---")
    for i in range(5):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    log_mem("after 5x sync+empty")

    # Method 2: Try to reset memory stats (sometimes helps)
    print("\n--- Method 2: Reset memory stats ---")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after reset stats")

    # Method 3: Try IPC collect (for shared memory)
    print("\n--- Method 3: IPC collect ---")
    try:
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"IPC collect error: {e}")
    log_mem("after ipc_collect")

    # Method 4: Try to release the memory pool itself
    print("\n--- Method 4: Memory pool manipulation ---")
    try:
        # Get the current allocator
        # This is a hack to try to release the memory pool
        stats = torch.cuda.memory_stats()
        print(f"  Large pool allocated: {stats.get('allocated_bytes.large_pool.current', 0) / 1024**3:.2f}GB")
        print(f"  Small pool allocated: {stats.get('allocated_bytes.small_pool.current', 0) / 1024**3:.2f}GB")

        # Try to force garbage collection at different thresholds
        for thresh in [0.5, 0.3, 0.1, 0.05]:
            try:
                torch.cuda.memory._set_allocator_settings(f"garbage_collection_threshold:{thresh}")
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Threshold {thresh} error: {e}")
                break

        # Reset to default
        torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.6")
    except Exception as e:
        print(f"Memory pool error: {e}")
    log_mem("after pool manipulation")

    # Method 5: Try to compact memory (PyTorch 2.0+)
    print("\n--- Method 5: Memory defragmentation ---")
    try:
        # This might help with fragmentation
        if hasattr(torch.cuda, 'memory'):
            # Check if there's a defragment function
            if hasattr(torch.cuda.memory, '_defragment'):
                torch.cuda.memory._defragment()
                print("  Called _defragment")
    except Exception as e:
        print(f"Defragment error: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after defragment attempt")

    return log_mem("final force release")


def try_cuda_context_operations():
    """Try CUDA context-level operations."""
    print("\n=== CUDA CONTEXT OPERATIONS ===\n")

    log_mem("before")

    # Check current device
    device = torch.cuda.current_device()
    print(f"Current device: {device}")

    # Try to get device properties
    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}")
    print(f"Total memory: {props.total_memory / 1024**3:.1f}GB")

    # Method 1: Synchronize all streams
    print("\n--- Method 1: Sync all streams ---")
    torch.cuda.synchronize(device)

    # Get default stream and sync
    default_stream = torch.cuda.default_stream(device)
    default_stream.synchronize()

    # Get current stream and sync
    current_stream = torch.cuda.current_stream(device)
    current_stream.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after stream sync")

    # Method 2: Try to access CUDA driver directly
    print("\n--- Method 2: Direct CUDA driver access ---")
    try:
        # Try to use cupy if available (has better CUDA access)
        import cupy as cp
        print("  CuPy available, trying direct memory release...")
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        torch.cuda.empty_cache()
        log_mem("after cupy free")
    except ImportError:
        print("  CuPy not available")
    except Exception as e:
        print(f"  CuPy error: {e}")

    # Method 3: Try to use CUDA runtime directly via ctypes
    print("\n--- Method 3: CUDA runtime via ctypes ---")
    try:
        # Find CUDA runtime library
        import ctypes.util
        cuda_lib_name = ctypes.util.find_library('cudart')
        if cuda_lib_name:
            cuda_rt = ctypes.CDLL(cuda_lib_name)
            print(f"  Found CUDA runtime: {cuda_lib_name}")

            # cudaDeviceSynchronize
            cuda_rt.cudaDeviceSynchronize()
            print("  Called cudaDeviceSynchronize")

            gc.collect()
            torch.cuda.empty_cache()
            log_mem("after cudaDeviceSynchronize")
        else:
            print("  Could not find CUDA runtime library")
    except Exception as e:
        print(f"  CUDA runtime error: {e}")

    return log_mem("final context ops")


def nuclear_cleanup(llm):
    """Nuclear option: try everything possible to free memory."""
    print("\n" + "=" * 70)
    print("NUCLEAR CLEANUP")
    print("=" * 70)

    log_mem("before nuclear")

    # Get all internals
    internals = get_vllm_internals(llm)

    # Step 1: Clear model parameters
    print("\n--- Step 1: Clear model ---")
    model = internals.get('model')
    if model:
        # Clear all parameters
        for param in model.parameters():
            if param.device.type == 'cuda':
                param.data = torch.empty(0, device='cpu')
        # Clear all buffers
        for buf in model.buffers():
            if buf.device.type == 'cuda':
                try:
                    buf.data = torch.empty(0, device='cpu')
                except:
                    pass
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after model clear")

    # Step 2: Clear KV caches with storage resize
    print("\n--- Step 2: Clear KV caches with storage.resize_(0) ---")
    runner = internals.get('model_runner')
    if runner and hasattr(runner, 'kv_caches'):
        for kv in runner.kv_caches:
            if isinstance(kv, torch.Tensor):
                try:
                    # This is more aggressive - resizes the underlying storage
                    kv.storage().resize_(0)
                except Exception as e:
                    # Fallback to data replacement
                    kv.data = torch.empty(0, device='cpu')
        runner.kv_caches.clear()
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after KV storage resize")

    # Step 3: Clear static forward context with storage resize
    print("\n--- Step 3: Clear static_forward_context ---")
    if runner and hasattr(runner, 'static_forward_context'):
        for name, layer in list(runner.static_forward_context.items()):
            if hasattr(layer, 'kv_cache'):
                for t in (layer.kv_cache if isinstance(layer.kv_cache, list) else [layer.kv_cache]):
                    if isinstance(t, torch.Tensor):
                        try:
                            t.storage().resize_(0)
                        except:
                            t.data = torch.empty(0, device='cpu')
                layer.kv_cache = []
        runner.static_forward_context.clear()
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after context clear")

    # Step 4: Delete all references
    print("\n--- Step 4: Delete all references ---")
    if runner:
        runner.model = None
        if internals.get('worker'):
            internals['worker'].model_runner = None
    del model
    del runner

    # Clear executor
    executor = internals.get('executor')
    if executor:
        if hasattr(executor, 'driver_worker'):
            executor.driver_worker = None

    # Clear engine core
    engine_core = internals.get('engine_core')
    if engine_core:
        if hasattr(engine_core, 'model_executor'):
            engine_core.model_executor = None
        try:
            engine_core.shutdown()
        except:
            pass

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after delete refs")

    # Step 5: Destroy parallel state
    print("\n--- Step 5: Destroy parallel state ---")
    try:
        from vllm.distributed import parallel_state
        if hasattr(parallel_state, 'destroy_model_parallel'):
            parallel_state.destroy_model_parallel()
        if hasattr(parallel_state, 'destroy_distributed_environment'):
            parallel_state.destroy_distributed_environment()
    except:
        pass

    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after parallel state")

    # Step 6: Clear RoPE and other caches
    print("\n--- Step 6: Clear vLLM caches ---")
    try:
        from vllm.model_executor.layers import rotary_embedding
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rotary_embedding._ROPE_DICT.clear()
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after vLLM caches")

    # Step 7: Force allocator release
    print("\n--- Step 7: Force allocator release ---")
    try_force_allocator_release()

    # Step 8: CUDA context operations
    print("\n--- Step 8: CUDA context operations ---")
    try_cuda_context_operations()

    return log_mem("NUCLEAR FINAL")


def main():
    from vllm import LLM, SamplingParams

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print("=" * 70)
    print("FORCE MEMORY RELEASE TEST")
    print("=" * 70)

    baseline = log_mem("baseline")

    # Load model
    print("\n=== Loading model ===")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=True,
    )
    log_mem("after load")

    # Quick inference
    out = llm.generate(["2+2="], SamplingParams(max_tokens=10))
    print(f"Output: {out[0].outputs[0].text.strip()[:30]}")

    # Nuclear cleanup
    final = nuclear_cleanup(llm)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Baseline:  {baseline['used_gb']:.2f}GB")
    print(f"Final:     {final['used_gb']:.2f}GB")
    print(f"Remaining: {final['used_gb'] - baseline['used_gb']:.2f}GB")

    if final['used_gb'] - baseline['used_gb'] > 1.0:
        print("\nSTILL LEAKING - The remaining memory is likely:")
        print("  1. CUDA context overhead (unavoidable, ~500MB-1GB)")
        print("  2. cuBLAS/cuDNN workspaces (lazy-allocated)")
        print("  3. PyTorch internal buffers")
        print("\nPossible solutions:")
        print("  1. Use subprocess isolation (spawn new process per model)")
        print("  2. Use torch.cuda.reset_device() if available")
        print("  3. Accept ~1GB overhead as baseline")


if __name__ == "__main__":
    main()
