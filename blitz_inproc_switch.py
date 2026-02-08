#!/usr/bin/env python3
"""
In-process model switching for vLLM V1.

This module provides proper memory cleanup to enable switching models
without subprocess isolation.

Key findings:
- vLLM V1 doesn't properly free GPU memory when models are deleted
- The model weights, KV cache, and distributed state all need explicit cleanup
- Unload time is ~200ms, most of the switch time is in model loading

Usage:
    from blitz_inproc_switch import unload_vllm_model

    llm = LLM(model="model1", ...)
    # ... use model ...
    unload_vllm_model(llm)
    del llm

    llm2 = LLM(model="model2", ...)  # Now this works!
"""

import gc
import time
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

import torch


def unload_vllm_model(llm, verbose: bool = False) -> dict:
    """
    Properly unload a vLLM model and free all GPU memory.

    Args:
        llm: The vLLM LLM instance to unload
        verbose: If True, print timing details

    Returns:
        dict with timing and memory info
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

    # 7. Clear KV cache tensors more thoroughly
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

    # 8. Destroy distributed environment using vLLM's functions
    t0 = time.time()
    try:
        from vllm.distributed import parallel_state
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
    except Exception as e:
        if verbose:
            print(f"Warning during parallel state cleanup: {e}")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    timings['destroy_pg'] = time.time() - t0

    # 9. Force more aggressive garbage collection
    t0 = time.time()
    gc.collect()
    gc.collect()
    timings['gc2'] = time.time() - t0

    # 10. Synchronize and empty cache
    t0 = time.time()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    timings['cuda_empty'] = time.time() - t0

    # Final memory check
    free_after, _ = torch.cuda.mem_get_info()
    freed = free_after - free_before

    total_time = time.time() - start

    result = {
        'total_time': total_time,
        'freed_gb': freed / 1e9,
        'free_before_gb': free_before / 1e9,
        'free_after_gb': free_after / 1e9,
        'timings': timings
    }

    if verbose:
        print(f"Unload time: {total_time:.3f}s")
        print(f"Memory freed: {freed/1e9:.2f} GiB")
        print(f"GPU memory: {free_before/1e9:.2f} -> {free_after/1e9:.2f} GiB free")

    return result


def test_full_switching_cycle():
    """Test complete model switching cycle with timing."""
    from vllm import LLM, SamplingParams

    print("=" * 70)
    print("In-Process Model Switching Test")
    print("=" * 70)

    # Initial state
    free_init, total = torch.cuda.mem_get_info()
    print(f"\nInitial GPU memory: {free_init/1e9:.2f}/{total/1e9:.2f} GiB free")

    # Load Model 1
    print("\n" + "-" * 40)
    print("Phase 1: Load GPT-OSS-120B (MXFP4)")
    print("-" * 40)

    t0 = time.time()
    llm1 = LLM(
        model="openai/gpt-oss-120b",
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=2,
        disable_log_stats=True,
        enforce_eager=True,
    )
    load1_time = time.time() - t0

    free_after_1, _ = torch.cuda.mem_get_info()
    print(f"Load time: {load1_time:.1f}s")
    print(f"GPU memory: {free_after_1/1e9:.2f} GiB free (used {(free_init-free_after_1)/1e9:.2f} GiB)")

    # Verify model 1 works
    out = llm1.generate(["Hello world"], SamplingParams(max_tokens=10))
    print(f"Model 1 output: {out[0].outputs[0].text[:50]}...")

    # Switch to Model 2
    print("\n" + "-" * 40)
    print("Phase 2: Switch to Qwen3-VL-32B (FP16)")
    print("-" * 40)

    switch_start = time.time()

    # Unload model 1
    unload_result = unload_vllm_model(llm1, verbose=True)
    del llm1
    gc.collect()
    torch.cuda.empty_cache()

    # Load model 2
    t0 = time.time()
    llm2 = LLM(
        model="Qwen/Qwen3-VL-32B-Instruct",
        dtype="float16",
        max_model_len=512,
        max_num_seqs=2,
        disable_log_stats=True,
        enforce_eager=True,
        gpu_memory_utilization=0.70,
    )
    load2_time = time.time() - t0

    switch_total = time.time() - switch_start

    free_after_2, _ = torch.cuda.mem_get_info()
    print(f"Load time: {load2_time:.1f}s")
    print(f"GPU memory: {free_after_2/1e9:.2f} GiB free")

    # Verify model 2 works
    out = llm2.generate(["Hello world"], SamplingParams(max_tokens=10))
    print(f"Model 2 output: {out[0].outputs[0].text[:50]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model 1 load:     {load1_time:.1f}s")
    print(f"Model 1 unload:   {unload_result['total_time']:.2f}s")
    print(f"Model 2 load:     {load2_time:.1f}s")
    print(f"Total switch:     {switch_total:.1f}s")
    print(f"Memory freed:     {unload_result['freed_gb']:.1f} GiB")
    print("=" * 70)
    print("SUCCESS: In-process model switching works!")
    print("=" * 70)


if __name__ == "__main__":
    test_full_switching_cycle()
