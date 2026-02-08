#!/usr/bin/env python3
"""Find what's holding the remaining ~8GB after model weight cleanup.

We know:
1. param.data = [] frees ~38GB (model weights)
2. ~8GB remains allocated after all cleanup attempts
3. Need to find what's holding this 8GB
"""

import os
import gc

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


def print_gpu_memory(label=""):
    m = get_gpu_memory()
    print(f"[{label}] Used: {m['used_gb']:.2f}GB, Alloc: {m['allocated_gb']:.2f}GB, Free: {m['free_gb']:.2f}GB")


def find_all_cuda_tensors():
    """Find ALL CUDA tensors and their sizes."""
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
                })
        except Exception:
            pass
    return tensors


def get_vllm_internals(llm):
    """Properly navigate vLLM V1's InprocClient structure."""
    engine = llm.llm_engine

    # V1 engine uses InprocClient
    inproc_client = engine.engine_core

    # InprocClient wraps the actual EngineCore
    if hasattr(inproc_client, 'engine_core'):
        engine_core = inproc_client.engine_core
    else:
        engine_core = inproc_client

    result = {'engine_core': engine_core}

    # Get model executor
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


def analyze_model_runner_memory(runner):
    """Analyze what the model runner is holding."""
    print("\n=== MODEL RUNNER MEMORY ANALYSIS ===\n")

    if runner is None:
        print("No model runner!")
        return

    attrs = [a for a in dir(runner) if not a.startswith('_')]
    print(f"Model runner attributes: {len(attrs)}")

    # Check specific known memory holders
    holders = [
        'model', 'kv_caches', 'static_forward_context', 'encoder_cache',
        'input_ids', 'positions', 'inputs_embeds', 'intermediate_tensors',
        'mm_inputs', 'mm_input_embeds', 'attn_metadata',
    ]

    for attr in holders:
        if hasattr(runner, attr):
            val = getattr(runner, attr)
            if val is None:
                print(f"  {attr}: None")
            elif isinstance(val, torch.Tensor):
                size_mb = val.numel() * val.element_size() / 1024**2
                print(f"  {attr}: Tensor {val.shape} {val.device} = {size_mb:.1f}MB")
            elif isinstance(val, list):
                total_size = 0
                for item in val:
                    if isinstance(item, torch.Tensor):
                        total_size += item.numel() * item.element_size()
                print(f"  {attr}: List[{len(val)}] = {total_size / 1024**2:.1f}MB")
            elif isinstance(val, dict):
                total_size = 0
                for k, v in val.items():
                    if isinstance(v, torch.Tensor):
                        total_size += v.numel() * v.element_size()
                    elif hasattr(v, '__dict__'):
                        for vv in v.__dict__.values():
                            if isinstance(vv, torch.Tensor):
                                total_size += vv.numel() * vv.element_size()
                print(f"  {attr}: Dict[{len(val)}] = {total_size / 1024**2:.1f}MB")
            else:
                print(f"  {attr}: {type(val).__name__}")


def analyze_kv_caches(runner):
    """Analyze KV cache structure in detail."""
    print("\n=== KV CACHE ANALYSIS ===\n")

    if not hasattr(runner, 'kv_caches'):
        print("No kv_caches attribute")
        return

    kv_caches = runner.kv_caches
    print(f"KV caches: {type(kv_caches)}, length: {len(kv_caches) if kv_caches else 0}")

    if kv_caches:
        total_size = 0
        for i, kv in enumerate(kv_caches):
            if kv is None:
                print(f"  [{i}]: None")
            elif isinstance(kv, torch.Tensor):
                size_mb = kv.numel() * kv.element_size() / 1024**2
                total_size += size_mb
                print(f"  [{i}]: Tensor {kv.shape} {kv.device} {kv.dtype} = {size_mb:.1f}MB")
            else:
                print(f"  [{i}]: {type(kv)}")
                # Some KV caches are tuples of (key, value)
                if isinstance(kv, tuple):
                    for j, t in enumerate(kv):
                        if isinstance(t, torch.Tensor):
                            size_mb = t.numel() * t.element_size() / 1024**2
                            total_size += size_mb
                            print(f"      [{j}]: Tensor {t.shape} = {size_mb:.1f}MB")
        print(f"\nTotal KV cache size: {total_size:.1f}MB = {total_size/1024:.2f}GB")


def analyze_encoder_cache(runner):
    """Analyze encoder cache (for vision models)."""
    print("\n=== ENCODER CACHE ANALYSIS ===\n")

    if not hasattr(runner, 'encoder_cache'):
        print("No encoder_cache attribute")
        return

    cache = runner.encoder_cache
    if cache is None:
        print("encoder_cache is None")
        return

    print(f"Encoder cache type: {type(cache)}")

    # It's typically a dict or custom cache object
    if hasattr(cache, '__dict__'):
        total_size = 0
        for name, val in cache.__dict__.items():
            if isinstance(val, torch.Tensor):
                size_mb = val.numel() * val.element_size() / 1024**2
                total_size += size_mb
                print(f"  {name}: Tensor {val.shape} = {size_mb:.1f}MB")
            elif isinstance(val, dict):
                dict_size = 0
                for k, v in val.items():
                    if isinstance(v, torch.Tensor):
                        dict_size += v.numel() * v.element_size()
                print(f"  {name}: Dict[{len(val)}] = {dict_size/1024**2:.1f}MB")
                total_size += dict_size / 1024**2
        print(f"\nTotal encoder cache size: {total_size:.1f}MB")


def analyze_static_forward_context(runner):
    """Analyze static forward context."""
    print("\n=== STATIC FORWARD CONTEXT ANALYSIS ===\n")

    if not hasattr(runner, 'static_forward_context'):
        print("No static_forward_context attribute")
        return

    ctx = runner.static_forward_context
    if ctx is None:
        print("static_forward_context is None")
        return

    print(f"Static forward context: {type(ctx)}, length: {len(ctx)}")

    if isinstance(ctx, dict):
        total_size = 0
        for name, layer in list(ctx.items())[:5]:  # First 5 layers
            print(f"  {name}: {type(layer).__name__}")
            if hasattr(layer, 'kv_cache'):
                kv = layer.kv_cache
                if isinstance(kv, list):
                    kv_size = 0
                    for t in kv:
                        if isinstance(t, torch.Tensor):
                            kv_size += t.numel() * t.element_size()
                    print(f"    kv_cache: List[{len(kv)}] = {kv_size/1024**2:.1f}MB")
                    total_size += kv_size
        print(f"\n(Showing first 5 of {len(ctx)} entries)")


def complete_cleanup_test():
    """Complete cleanup test with detailed tracing."""
    from vllm import LLM, SamplingParams

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print("=" * 70)
    print("COMPLETE CLEANUP TEST WITH TRACING")
    print("=" * 70)

    print_gpu_memory("baseline")

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
    print_gpu_memory("after load")

    # Get internals
    internals = get_vllm_internals(llm)
    runner = internals.get('model_runner')
    model = internals.get('model')

    # Analyze what's holding memory
    analyze_model_runner_memory(runner)
    analyze_kv_caches(runner)
    analyze_encoder_cache(runner)
    analyze_static_forward_context(runner)

    print("\n" + "=" * 70)
    print("STEP-BY-STEP CLEANUP")
    print("=" * 70)

    # Step 1: Clear KV caches
    print("\n--- Step 1: Clear KV caches ---")
    if runner and hasattr(runner, 'kv_caches') and runner.kv_caches:
        for kv in runner.kv_caches:
            if kv is not None:
                if isinstance(kv, torch.Tensor):
                    kv.data = torch.empty(0, device='cpu')
                elif isinstance(kv, tuple):
                    for t in kv:
                        if isinstance(t, torch.Tensor):
                            t.data = torch.empty(0, device='cpu')
        runner.kv_caches.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after KV clear")

    # Step 2: Clear static forward context
    print("\n--- Step 2: Clear static forward context ---")
    if runner and hasattr(runner, 'static_forward_context') and runner.static_forward_context:
        for name, layer in runner.static_forward_context.items():
            if hasattr(layer, 'kv_cache') and layer.kv_cache:
                for t in layer.kv_cache:
                    if isinstance(t, torch.Tensor):
                        t.data = torch.empty(0, device='cpu')
                layer.kv_cache.clear()
        runner.static_forward_context.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after context clear")

    # Step 3: Clear encoder cache
    print("\n--- Step 3: Clear encoder cache ---")
    if runner and hasattr(runner, 'encoder_cache') and runner.encoder_cache:
        ec = runner.encoder_cache
        if hasattr(ec, 'clear'):
            ec.clear()
        elif hasattr(ec, '__dict__'):
            for name in list(ec.__dict__.keys()):
                setattr(ec, name, None)
        runner.encoder_cache = None
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after encoder cache clear")

    # Step 4: Clear model parameters
    print("\n--- Step 4: Clear model parameters ---")
    if model:
        for name, param in model.named_parameters():
            if param.device.type == 'cuda':
                param.data = torch.empty(0, device='cpu')
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after params clear")

    # Step 5: Clear model buffers
    print("\n--- Step 5: Clear model buffers ---")
    if model:
        for name, buf in model.named_buffers():
            if buf.device.type == 'cuda':
                try:
                    buf.data = torch.empty(0, device='cpu')
                except:
                    pass
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after buffers clear")

    # Step 6: Check what's left
    print("\n--- Step 6: Check remaining CUDA tensors ---")
    tensors = find_all_cuda_tensors()
    print(f"Found {len(tensors)} CUDA tensors")
    total_mb = sum(t['size_mb'] for t in tensors)
    print(f"Total size: {total_mb:.1f}MB")

    # Show largest
    sorted_tensors = sorted(tensors, key=lambda x: x['size_mb'], reverse=True)[:20]
    print("\nLargest remaining tensors:")
    for t in sorted_tensors:
        print(f"  {t['shape']} {t['dtype']} - {t['size_mb']:.1f}MB (grad={t['requires_grad']})")

    # Step 7: Delete model reference
    print("\n--- Step 7: Delete model ---")
    if runner:
        runner.model = None
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after del model")

    # Step 8: Delete runner
    print("\n--- Step 8: Delete model_runner ---")
    if internals.get('worker'):
        internals['worker'].model_runner = None
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after del runner")

    # Step 9: Delete worker
    print("\n--- Step 9: Delete worker ---")
    if internals.get('driver_worker'):
        internals['driver_worker'].worker = None
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after del worker")

    # Step 10: Shutdown engine core
    print("\n--- Step 10: Shutdown engine_core ---")
    engine_core = internals.get('engine_core')
    if engine_core:
        try:
            engine_core.shutdown()
        except Exception as e:
            print(f"  Error: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after shutdown")

    # Step 11: Delete llm
    print("\n--- Step 11: Delete llm ---")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after del llm")

    # Step 12: Destroy process group
    print("\n--- Step 12: Destroy process group ---")
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"  Error: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after destroy_process_group")

    # Step 13: Clear RoPE cache
    print("\n--- Step 13: Clear RoPE cache ---")
    try:
        from vllm.model_executor.layers import rotary_embedding
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            for key, rope in list(rotary_embedding._ROPE_DICT.items()):
                # Clear any GPU tensors in the RoPE embedding
                if hasattr(rope, 'cos_cached'):
                    rope.cos_cached = None
                if hasattr(rope, 'sin_cached'):
                    rope.sin_cached = None
            rotary_embedding._ROPE_DICT.clear()
    except Exception as e:
        print(f"  Error: {e}")
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_memory("after RoPE clear")

    # Step 14: Final check
    print("\n--- Step 14: Final tensor check ---")
    tensors = find_all_cuda_tensors()
    print(f"Found {len(tensors)} CUDA tensors")
    total_mb = sum(t['size_mb'] for t in tensors)
    print(f"Total size: {total_mb:.1f}MB")
    for t in sorted(tensors, key=lambda x: x['size_mb'], reverse=True)[:10]:
        print(f"  {t['shape']} {t['dtype']} - {t['size_mb']:.1f}MB")

    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    print_gpu_memory("final")


if __name__ == "__main__":
    complete_cleanup_test()
