#!/usr/bin/env python3
"""Investigate if KV cache buffers are the source of the 7.5GB leak.

The memory snapshot showed 261 allocations of 112-160MB each = ~7.5GB.
This looks exactly like KV cache blocks. Let's verify and fix.
"""

import os
import gc
import sys

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
          f"Reserved: {m['reserved_gb']:.2f}GB")
    return m


def get_vllm_internals(llm):
    """Navigate to vLLM internals properly."""
    engine = llm.llm_engine
    inproc_client = engine.engine_core

    # Unwrap InprocClient
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
                    runner = worker.model_runner
                    result['model_runner'] = runner

                    if hasattr(runner, 'model'):
                        result['model'] = runner.model

    return result


def analyze_kv_cache_structure(runner):
    """Analyze KV cache structure in detail."""
    print("\n=== KV CACHE ANALYSIS ===\n")

    if runner is None:
        print("No model runner")
        return

    # Check kv_caches attribute
    if hasattr(runner, 'kv_caches'):
        kv_caches = runner.kv_caches
        print(f"model_runner.kv_caches: {type(kv_caches)}, len={len(kv_caches) if kv_caches else 0}")
        if kv_caches:
            total_size = 0
            for i, kv in enumerate(kv_caches[:5]):  # First 5
                if isinstance(kv, torch.Tensor):
                    size = kv.numel() * kv.element_size()
                    total_size += size
                    print(f"  [{i}]: Tensor {kv.shape} {kv.dtype} = {size/1024**2:.1f}MB")
                elif kv is not None:
                    print(f"  [{i}]: {type(kv).__name__}")
            print(f"  Total (first 5): {total_size/1024**2:.1f}MB")
            if len(kv_caches) > 5:
                print(f"  ... and {len(kv_caches) - 5} more entries")
    else:
        print("No kv_caches attribute")

    # Check static_forward_context - this often holds KV cache references
    if hasattr(runner, 'static_forward_context'):
        ctx = runner.static_forward_context
        print(f"\nstatic_forward_context: {type(ctx)}, len={len(ctx) if ctx else 0}")
        if ctx:
            total_kv_size = 0
            for name, layer in list(ctx.items())[:3]:
                print(f"  {name}: {type(layer).__name__}")
                if hasattr(layer, 'kv_cache'):
                    kv = layer.kv_cache
                    print(f"    kv_cache: {type(kv)}")
                    if isinstance(kv, list):
                        for j, t in enumerate(kv):
                            if isinstance(t, torch.Tensor):
                                size = t.numel() * t.element_size()
                                total_kv_size += size
                                print(f"      [{j}]: {t.shape} = {size/1024**2:.1f}MB")
                    elif isinstance(kv, torch.Tensor):
                        size = kv.numel() * kv.element_size()
                        total_kv_size += size
                        print(f"      {kv.shape} = {size/1024**2:.1f}MB")
            print(f"  Total KV in context: {total_kv_size/1024**2:.1f}MB")

    # Check for flash attention state
    print("\n=== CHECKING FLASH ATTENTION STATE ===\n")
    try:
        from vllm.attention.backends import flash_attn
        print(f"Flash attention module: {flash_attn}")

        # Check for any cached state
        for attr in dir(flash_attn):
            if not attr.startswith('_'):
                continue
            val = getattr(flash_attn, attr, None)
            if val is not None and not callable(val):
                if isinstance(val, dict) and val:
                    print(f"  {attr}: dict with {len(val)} entries")
                elif isinstance(val, (list, tuple)) and val:
                    print(f"  {attr}: {type(val).__name__} with {len(val)} entries")
    except Exception as e:
        print(f"Flash attention check error: {e}")

    # Check for any attention backends holding state
    print("\n=== CHECKING ATTENTION BACKENDS ===\n")
    try:
        from vllm.attention import backends
        for name in dir(backends):
            if name.startswith('_'):
                continue
            mod = getattr(backends, name, None)
            if mod and hasattr(mod, '__dict__'):
                for attr in dir(mod):
                    if attr.startswith('_CACHED') or attr.startswith('_WORKSPACE'):
                        val = getattr(mod, attr, None)
                        if val is not None:
                            print(f"  {name}.{attr}: {type(val)}")
    except Exception as e:
        print(f"Attention backends check error: {e}")


def deep_kv_cache_cleanup(runner):
    """Deeply clean up ALL KV cache references."""
    print("\n=== DEEP KV CACHE CLEANUP ===\n")

    if runner is None:
        return

    log_mem("before deep cleanup")

    # 1. Clear kv_caches list
    if hasattr(runner, 'kv_caches') and runner.kv_caches:
        print("Clearing kv_caches...")
        for i, kv in enumerate(runner.kv_caches):
            if isinstance(kv, torch.Tensor):
                # Clear the data
                kv.data = torch.empty(0, device='cpu')
            elif isinstance(kv, (list, tuple)):
                for t in kv:
                    if isinstance(t, torch.Tensor):
                        t.data = torch.empty(0, device='cpu')
        runner.kv_caches.clear()
        print(f"  Cleared {i+1} entries")

    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after kv_caches clear")

    # 2. Clear static_forward_context
    if hasattr(runner, 'static_forward_context') and runner.static_forward_context:
        print("Clearing static_forward_context...")
        count = 0
        for name, layer in list(runner.static_forward_context.items()):
            if hasattr(layer, 'kv_cache'):
                kv = layer.kv_cache
                if isinstance(kv, list):
                    for t in kv:
                        if isinstance(t, torch.Tensor):
                            t.data = torch.empty(0, device='cpu')
                    layer.kv_cache = []
                elif isinstance(kv, torch.Tensor):
                    kv.data = torch.empty(0, device='cpu')
                    layer.kv_cache = None
                count += 1
        runner.static_forward_context.clear()
        print(f"  Cleared {count} layer contexts")

    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after static_forward_context clear")

    # 3. Clear any cached attention state
    print("Clearing attention backend state...")
    try:
        from vllm.attention.backends import flash_attn
        # Clear any module-level caches
        for attr in list(dir(flash_attn)):
            if attr.startswith('_CACHED') or attr.startswith('_WORKSPACE'):
                try:
                    setattr(flash_attn, attr, None)
                    print(f"  Cleared {attr}")
                except:
                    pass
    except:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after attention cache clear")

    # 4. Try to find and clear any hidden GPU tensors
    print("Searching for hidden GPU tensors...")
    found = 0
    freed = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device.type == 'cuda':
                size = obj.numel() * obj.element_size()
                found += 1
                # Try to free it
                try:
                    obj.data = torch.empty(0, device='cpu')
                    freed += 1
                except:
                    pass
        except:
            pass
    print(f"  Found {found} GPU tensors, freed {freed}")

    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after hidden tensor clear")

    # 5. Nuclear option: iterate through all module attributes
    print("Clearing all tensor attributes on runner...")
    cleared = 0
    for attr in dir(runner):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(runner, attr)
            if isinstance(val, torch.Tensor) and val.device.type == 'cuda':
                setattr(runner, attr, None)
                cleared += 1
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, torch.Tensor) and item.device.type == 'cuda':
                        item.data = torch.empty(0, device='cpu')
                        cleared += 1
        except:
            pass
    print(f"  Cleared {cleared} tensor attributes")

    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after runner attribute clear")


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.engine.cleanup import full_cleanup

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print("=" * 70)
    print("KV CACHE LEAK INVESTIGATION")
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

    # Get internals
    internals = get_vllm_internals(llm)
    runner = internals.get('model_runner')

    # Analyze KV cache structure
    analyze_kv_cache_structure(runner)

    # Do inference to populate KV cache
    print("\n=== Running inference ===")
    out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=20))
    print(f"Output: {out[0].outputs[0].text.strip()[:50]}")
    log_mem("after inference")

    # Re-analyze after inference
    print("\n=== After inference ===")
    analyze_kv_cache_structure(runner)

    # Try our deep cleanup
    print("\n" + "=" * 70)
    print("ATTEMPTING DEEP CLEANUP")
    print("=" * 70)

    # Get fresh references
    internals = get_vllm_internals(llm)
    runner = internals.get('model_runner')
    model = internals.get('model')

    # Clear model params first
    print("\n--- Clearing model params ---")
    if model:
        for name, param in model.named_parameters():
            if param.device.type == 'cuda':
                param.data = torch.empty(0, device='cpu')
        for name, buf in model.named_buffers():
            if buf.device.type == 'cuda':
                try:
                    buf.data = torch.empty(0, device='cpu')
                except:
                    pass
    gc.collect()
    torch.cuda.empty_cache()
    log_mem("after model params clear")

    # Deep KV cache cleanup
    deep_kv_cache_cleanup(runner)

    # Now do full cleanup
    print("\n--- Full cleanup ---")
    # Don't call full_cleanup since we've already cleared things
    # Just delete references
    if runner:
        internals['worker'].model_runner = None
    del model
    del runner
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Destroy parallel state
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
    log_mem("after delete all")

    # Final check
    print("\n=== FINAL CHECK ===")
    tensors_found = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device.type == 'cuda':
                size = obj.numel() * obj.element_size()
                tensors_found += 1
                if tensors_found <= 5:
                    print(f"  GPU tensor: {obj.shape} {obj.dtype} = {size/1024**2:.1f}MB")
        except:
            pass
    print(f"Total GPU tensors: {tensors_found}")

    # Final memory state
    final = log_mem("FINAL")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline: {baseline['used_gb']:.2f}GB")
    print(f"Final:    {final['used_gb']:.2f}GB")
    print(f"Leak:     {final['used_gb'] - baseline['used_gb']:.2f}GB")


if __name__ == "__main__":
    main()
