#!/usr/bin/env python3
"""
Implement proper vLLM memory cleanup for in-process model switching.

This patches vLLM to add unload_model() functionality.
"""

import gc
import time
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

import torch


def unload_vllm_model(llm) -> dict:
    """
    Properly unload a vLLM model and free all GPU memory.

    Returns dict with timing and memory info.
    """
    start = time.time()
    free_before, total = torch.cuda.mem_get_info()

    timings = {}

    # Get internal components (vLLM V1 structure)
    # Path: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    engine = llm.llm_engine
    inproc_client = engine.engine_core
    engine_core = inproc_client.engine_core
    executor = engine_core.model_executor
    worker = executor.driver_worker.worker
    model_runner = worker.model_runner

    # 1. Clear the model weights
    t0 = time.time()
    model = model_runner.model
    if model is not None:
        # Clear all parameter data
        for name, param in model.named_parameters():
            param.data = torch.empty(0, device='cpu', dtype=param.dtype)

        # Clear all buffer data
        for name, buf in model.named_buffers():
            buf.data = torch.empty(0, device='cpu', dtype=buf.dtype)
    timings['clear_params'] = time.time() - t0

    # 2. Delete model reference
    t0 = time.time()
    del model_runner.model
    model_runner.model = None
    timings['del_model'] = time.time() - t0

    # 3. Clear KV caches from model runner
    t0 = time.time()
    if hasattr(model_runner, 'kv_caches') and model_runner.kv_caches:
        for i, cache in enumerate(model_runner.kv_caches):
            if cache is not None:
                # Each cache is a tuple of (key_cache, value_cache)
                if isinstance(cache, (list, tuple)):
                    for tensor in cache:
                        if tensor is not None:
                            tensor.data = torch.empty(0, device='cpu')
        model_runner.kv_caches = []
    timings['clear_kv_cache'] = time.time() - t0

    # 4. Clear static forward context (attention layer state)
    t0 = time.time()
    if hasattr(model_runner, 'static_forward_context'):
        for layer_name, layer in list(model_runner.static_forward_context.items()):
            if hasattr(layer, 'kv_cache'):
                layer.kv_cache = []
            # Clear any other cached tensors
            for attr in dir(layer):
                if attr.startswith('_'):
                    continue
                val = getattr(layer, attr, None)
                if isinstance(val, torch.Tensor):
                    setattr(layer, attr, None)
        model_runner.static_forward_context.clear()
    timings['clear_static_ctx'] = time.time() - t0

    # 5. Clear input/output buffers
    t0 = time.time()
    for attr in ['input_ids', 'positions', 'hidden_states', 'residual']:
        if hasattr(model_runner, attr):
            val = getattr(model_runner, attr)
            if isinstance(val, torch.Tensor):
                setattr(model_runner, attr, None)
    timings['clear_buffers'] = time.time() - t0

    # 6. Force garbage collection
    t0 = time.time()
    gc.collect()
    timings['gc'] = time.time() - t0

    # 7. Clear the KV cache tensors more thoroughly
    t0 = time.time()
    if hasattr(model_runner, 'kv_caches'):
        for cache in model_runner.kv_caches:
            if isinstance(cache, torch.Tensor):
                cache.data = torch.empty(0, device='cpu')
            elif isinstance(cache, (list, tuple)):
                for t in cache:
                    if isinstance(t, torch.Tensor):
                        t.data = torch.empty(0, device='cpu')
        model_runner.kv_caches.clear()
    timings['thorough_kv_clear'] = time.time() - t0

    # 8. Destroy distributed environment and model parallel using vLLM's functions
    t0 = time.time()
    try:
        from vllm.distributed import parallel_state
        # Use vLLM's official cleanup functions
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
    except Exception as e:
        print(f"Warning during parallel state cleanup: {e}")
        # Fallback: destroy torch distributed
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    timings['destroy_pg'] = time.time() - t0

    # 9. Force more aggressive garbage collection
    t0 = time.time()
    gc.collect()
    gc.collect()  # Second pass for cyclic refs
    timings['gc2'] = time.time() - t0

    # 10. Synchronize and empty cache
    t0 = time.time()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timings['cuda_empty'] = time.time() - t0

    # Final memory check
    free_after, _ = torch.cuda.mem_get_info()
    freed = free_after - free_before

    total_time = time.time() - start

    return {
        'total_time': total_time,
        'freed_gb': freed / 1e9,
        'free_before_gb': free_before / 1e9,
        'free_after_gb': free_after / 1e9,
        'timings': timings
    }


def test_cleanup():
    """Test the cleanup implementation"""
    from vllm import LLM, SamplingParams

    print("="*60)
    print("Testing vLLM Memory Cleanup Implementation")
    print("="*60)

    # Check initial memory
    free_init, total = torch.cuda.mem_get_info()
    print(f"\nInitial GPU memory: {free_init/1e9:.2f}/{total/1e9:.2f} GiB free")

    # Load first model
    print("\n--- Loading GPT-OSS-120B ---")
    t0 = time.time()
    llm = LLM(
        model="openai/gpt-oss-120b",
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=2,
        disable_log_stats=True,
        enforce_eager=True,
    )
    load_time = time.time() - t0

    free_after_load, _ = torch.cuda.mem_get_info()
    print(f"Load time: {load_time:.1f}s")
    print(f"GPU memory: {free_after_load/1e9:.2f} GiB free (used {(free_init-free_after_load)/1e9:.2f} GiB)")

    # Quick inference test
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text[:30]}")

    # Now unload
    print("\n--- Unloading model ---")
    result = unload_vllm_model(llm)

    print(f"Unload time: {result['total_time']:.3f}s")
    print(f"Memory freed: {result['freed_gb']:.2f} GiB")
    print(f"GPU memory now: {result['free_after_gb']:.2f} GiB free")
    print("\nTiming breakdown:")
    for stage, t in result['timings'].items():
        print(f"  {stage}: {t*1000:.1f}ms")

    # Delete LLM object and all internal refs
    del llm
    gc.collect()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    free_final, _ = torch.cuda.mem_get_info()
    print(f"\nAfter del llm: {free_final/1e9:.2f} GiB free")

    # Additional cleanup - aggressive CUDA memory release
    import time as time_mod

    # Clear all CUDA caches
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Clear triton cache if present
    try:
        import triton
        if hasattr(triton, 'runtime') and hasattr(triton.runtime, 'cache'):
            triton.runtime.cache.clear()
    except:
        pass

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    time_mod.sleep(1.0)  # Give CUDA time to fully release
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    free_extra, _ = torch.cuda.mem_get_info()
    print(f"After extra cleanup: {free_extra/1e9:.2f} GiB free")

    # Try loading second model
    print("\n--- Loading Qwen3-VL-32B ---")
    t0 = time.time()
    llm2 = LLM(
        model="Qwen/Qwen3-VL-32B-Instruct",
        dtype="float16",
        max_model_len=512,
        max_num_seqs=2,
        disable_log_stats=True,
        enforce_eager=True,
        gpu_memory_utilization=0.70,  # Lower to avoid memory check failure
    )
    load_time2 = time.time() - t0

    free_after_load2, _ = torch.cuda.mem_get_info()
    print(f"Load time: {load_time2:.1f}s")
    print(f"GPU memory: {free_after_load2/1e9:.2f} GiB free")

    # Quick inference test
    out = llm2.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text[:30]}")

    print("\n" + "="*60)
    print("SUCCESS: In-process model switching works!")
    print("="*60)


if __name__ == "__main__":
    test_cleanup()
