#!/usr/bin/env python3
"""
Deep profile of vLLM LLM() constructor to find remaining bottlenecks.

Weight loading is ~1.5s but constructor takes ~6s. Where's the other 4.5s?
"""

import gc
import os
import sys
import time
import types
import functools

# ROCm setup
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# vLLM import workaround
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch

# Timing storage
TIMINGS = {}


def timed(name, accumulate=True):
    """Decorator to time a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            if accumulate:
                if name not in TIMINGS:
                    TIMINGS[name] = 0
                TIMINGS[name] += elapsed
            else:
                TIMINGS[name] = elapsed
            return result
        return wrapper
    return decorator


def patch_vllm_for_profiling():
    """Monkey-patch vLLM to add timing instrumentation."""

    # Tokenizer loading
    try:
        import vllm.entrypoints.llm as llm_module
        original_init_tokenizer = llm_module.LLM._init_tokenizer
        llm_module.LLM._init_tokenizer = timed('tokenizer_init')(original_init_tokenizer)
    except Exception as e:
        print(f"Could not patch tokenizer: {e}")

    # Model loading
    try:
        import vllm.model_executor.model_loader.default_loader as loader
        if hasattr(loader, 'DefaultModelLoader'):
            original_load = loader.DefaultModelLoader.load_model
            loader.DefaultModelLoader.load_model = timed('model_load')(original_load)
    except Exception as e:
        print(f"Could not patch model loader: {e}")

    # Weight loading
    try:
        import vllm.model_executor.model_loader.weight_utils as weight_utils
        original_weights = weight_utils.safetensors_weights_iterator
        weight_utils.safetensors_weights_iterator = timed('weight_iterator')(original_weights)
    except Exception as e:
        print(f"Could not patch weight utils: {e}")

    # Engine core init
    try:
        import vllm.v1.engine.core as core
        original_core_init = core.EngineCore.__init__
        core.EngineCore.__init__ = timed('engine_core_init')(original_core_init)
    except Exception as e:
        print(f"Could not patch engine core: {e}")

    # GPU worker init
    try:
        import vllm.v1.worker.gpu_worker as gpu_worker
        original_device_init = gpu_worker.GPUWorker.init_device
        gpu_worker.GPUWorker.init_device = timed('gpu_worker_init_device')(original_device_init)
    except Exception as e:
        print(f"Could not patch gpu worker: {e}")

    # Model runner loading
    try:
        import vllm.v1.worker.gpu_model_runner as runner
        original_runner_load = runner.GPUModelRunner.load_model
        runner.GPUModelRunner.load_model = timed('model_runner_load')(original_runner_load)
    except Exception as e:
        print(f"Could not patch model runner: {e}")

    # KV cache creation
    try:
        import vllm.v1.worker.gpu_worker as gpu_worker
        if hasattr(gpu_worker.GPUWorker, '_create_kv_caches'):
            original_kv = gpu_worker.GPUWorker._create_kv_caches
            gpu_worker.GPUWorker._create_kv_caches = timed('kv_cache_create')(original_kv)
    except Exception as e:
        print(f"Could not patch kv cache: {e}")


def main():
    print("="*70)
    print("VLLM CONSTRUCTOR DEEP PROFILE")
    print("="*70)

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # Apply profiling patches
    patch_vllm_for_profiling()

    from vllm import LLM

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    print(f"\nLoading {model_name}...")
    TIMINGS.clear()

    t0 = time.perf_counter()
    llm = LLM(model=model_name, **config)
    total_time = (time.perf_counter() - t0) * 1000

    print(f"\nTotal LLM() time: {total_time:.0f}ms")
    print("\n" + "-"*70)
    print("COMPONENT BREAKDOWN")
    print("-"*70)

    accounted = 0
    for name, elapsed in sorted(TIMINGS.items(), key=lambda x: -x[1]):
        pct = elapsed / total_time * 100
        print(f"  {name}: {elapsed:.0f}ms ({pct:.1f}%)")
        accounted += elapsed

    unaccounted = total_time - accounted
    if unaccounted > 100:
        print(f"\n  Unaccounted: {unaccounted:.0f}ms ({unaccounted/total_time*100:.1f}%)")

    # Test inference
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=20, temperature=0.7)
    outputs = llm.generate(["Hello!"], params)
    print(f"\nTest output: {outputs[0].outputs[0].text[:40]}...")

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("OPTIMIZATION TARGETS")
    print("="*70)

    if 'tokenizer_init' in TIMINGS and TIMINGS['tokenizer_init'] > 200:
        print(f"  - Tokenizer: {TIMINGS['tokenizer_init']:.0f}ms → Cache tokenizers")

    if 'model_load' in TIMINGS:
        print(f"  - Model load: {TIMINGS['model_load']:.0f}ms → Optimize weight loading")

    if 'engine_core_init' in TIMINGS and TIMINGS['engine_core_init'] > 500:
        print(f"  - Engine core: {TIMINGS['engine_core_init']:.0f}ms → Investigate")

    print("="*70)


if __name__ == '__main__':
    main()
