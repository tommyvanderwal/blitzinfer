#!/usr/bin/env python3
"""
Detailed profiling of vLLM LLM() constructor to find optimization targets.

Goal: Understand where the ~5.7s switch time is spent.
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

# Global timing storage
TIMINGS = {}
CALL_COUNTS = {}


def timed(name):
    """Decorator to time a function and accumulate results."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            if name not in TIMINGS:
                TIMINGS[name] = 0
                CALL_COUNTS[name] = 0
            TIMINGS[name] += elapsed
            CALL_COUNTS[name] += 1
            return result
        return wrapper
    return decorator


def patch_vllm_for_profiling():
    """Add timing instrumentation to key vLLM components."""

    # LLM constructor phases
    try:
        import vllm.entrypoints.llm as llm_module
        original_init = llm_module.LLM.__init__

        def patched_init(self, *args, **kwargs):
            t0 = time.perf_counter()
            result = original_init(self, *args, **kwargs)
            TIMINGS['LLM.__init__'] = (time.perf_counter() - t0) * 1000
            return result

        llm_module.LLM.__init__ = patched_init
    except Exception as e:
        print(f"Could not patch LLM.__init__: {e}")

    # Tokenizer loading
    try:
        import vllm.transformers_utils.tokenizer as tokenizer_module
        if hasattr(tokenizer_module, 'get_tokenizer'):
            tokenizer_module.get_tokenizer = timed('get_tokenizer')(tokenizer_module.get_tokenizer)
    except Exception as e:
        print(f"Could not patch tokenizer: {e}")

    # Model config loading
    try:
        import vllm.config.model as model_config
        if hasattr(model_config, 'ModelConfig'):
            original_model_config_init = model_config.ModelConfig.__init__
            model_config.ModelConfig.__init__ = timed('ModelConfig.__init__')(original_model_config_init)
    except Exception as e:
        print(f"Could not patch ModelConfig: {e}")

    # Engine core init
    try:
        import vllm.v1.engine.core as core_module
        original_core_init = core_module.EngineCore.__init__
        core_module.EngineCore.__init__ = timed('EngineCore.__init__')(original_core_init)
    except Exception as e:
        print(f"Could not patch EngineCore: {e}")

    # GPU worker
    try:
        import vllm.v1.worker.gpu_worker as gpu_worker
        if hasattr(gpu_worker, 'GPUWorker'):
            original_worker_init = gpu_worker.GPUWorker.__init__
            gpu_worker.GPUWorker.__init__ = timed('GPUWorker.__init__')(original_worker_init)

            if hasattr(gpu_worker.GPUWorker, 'init_device'):
                original_init_device = gpu_worker.GPUWorker.init_device
                gpu_worker.GPUWorker.init_device = timed('GPUWorker.init_device')(original_init_device)
    except Exception as e:
        print(f"Could not patch GPUWorker: {e}")

    # Model runner
    try:
        import vllm.v1.worker.gpu_model_runner as runner_module
        if hasattr(runner_module, 'GPUModelRunner'):
            if hasattr(runner_module.GPUModelRunner, 'load_model'):
                original_load_model = runner_module.GPUModelRunner.load_model
                runner_module.GPUModelRunner.load_model = timed('GPUModelRunner.load_model')(original_load_model)
    except Exception as e:
        print(f"Could not patch GPUModelRunner: {e}")

    # Default loader
    try:
        import vllm.model_executor.model_loader.default_loader as loader_module
        if hasattr(loader_module, 'DefaultModelLoader'):
            if hasattr(loader_module.DefaultModelLoader, 'load_weights'):
                original_load_weights = loader_module.DefaultModelLoader.load_weights
                loader_module.DefaultModelLoader.load_weights = timed('DefaultModelLoader.load_weights')(original_load_weights)
    except Exception as e:
        print(f"Could not patch DefaultModelLoader: {e}")

    # Safetensors weight iterator
    try:
        import vllm.model_executor.model_loader.weight_utils as weight_utils
        original_safetensors_weights_iterator = weight_utils.safetensors_weights_iterator

        def timed_safetensors_iterator(*args, **kwargs):
            t0 = time.perf_counter()
            count = 0
            for item in original_safetensors_weights_iterator(*args, **kwargs):
                count += 1
                yield item
            elapsed = (time.perf_counter() - t0) * 1000
            TIMINGS['safetensors_weights_iterator'] = elapsed
            TIMINGS['weight_count'] = count

        weight_utils.safetensors_weights_iterator = timed_safetensors_iterator
    except Exception as e:
        print(f"Could not patch safetensors_weights_iterator: {e}")


def get_mem():
    """Get GPU memory (used, free) in GB."""
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**3)
    return used, free / (1024**3)


def cleanup():
    """Full cleanup."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def main():
    print("="*70)
    print("VLLM CONSTRUCTOR DETAILED PROFILE")
    print("="*70)

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    cleanup()

    # Apply patches
    patch_vllm_for_profiling()

    from vllm import LLM

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    used, free = get_mem()
    print(f"\nBefore load: {used:.1f}GB used, {free:.1f}GB free")

    TIMINGS.clear()
    CALL_COUNTS.clear()

    print(f"\nLoading {model_name}...")
    t0 = time.perf_counter()
    llm = LLM(model=model_name, **config)
    total_time = (time.perf_counter() - t0) * 1000

    used, free = get_mem()
    print(f"After load: {used:.1f}GB used")
    print(f"\nTotal time: {total_time:.0f}ms")

    print("\n" + "-"*70)
    print("TIMING BREAKDOWN")
    print("-"*70)

    # Sort by time
    sorted_timings = sorted(TIMINGS.items(), key=lambda x: -x[1])

    for name, elapsed in sorted_timings:
        if name == 'weight_count':
            continue
        pct = elapsed / total_time * 100
        count = CALL_COUNTS.get(name, 1)
        if count > 1:
            print(f"  {name}: {elapsed:.0f}ms ({pct:.1f}%) - {count} calls")
        else:
            print(f"  {name}: {elapsed:.0f}ms ({pct:.1f}%)")

    if 'weight_count' in TIMINGS:
        print(f"\n  Weights loaded: {TIMINGS['weight_count']} tensors")

    # Calculate unaccounted time
    # Note: Some timings are nested, so simple sum won't work
    key_phases = ['safetensors_weights_iterator', 'get_tokenizer']
    accounted = sum(TIMINGS.get(k, 0) for k in key_phases)
    unaccounted = total_time - accounted
    print(f"\n  Time in weight iterator: {TIMINGS.get('safetensors_weights_iterator', 0):.0f}ms")
    print(f"  Time in tokenizer: {TIMINGS.get('get_tokenizer', 0):.0f}ms")
    print(f"  Other overhead: ~{unaccounted:.0f}ms")

    # Quick validation
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=10, temperature=0.7)
    outputs = llm.generate(["Test"], params)
    print(f"\nOutput: {outputs[0].outputs[0].text[:30]}...")

    # Cleanup
    del llm
    cleanup()

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    weight_time = TIMINGS.get('safetensors_weights_iterator', 0)
    tokenizer_time = TIMINGS.get('get_tokenizer', 0)

    print(f"\nCurrent breakdown:")
    print(f"  - Weight loading:    {weight_time:.0f}ms")
    print(f"  - Tokenizer:         {tokenizer_time:.0f}ms")
    print(f"  - Other:             {total_time - weight_time - tokenizer_time:.0f}ms")

    print(f"\nTo reach 2s target:")
    print(f"  - Need to save:      {total_time - 2000:.0f}ms")
    print(f"  - Weight loading:    {weight_time:.0f}ms (max ~1.5s with 9GB/s)")
    print(f"  - Tokenizer:         {tokenizer_time:.0f}ms (can cache)")

    # Check if model files are already cached
    from pathlib import Path
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"\n  HF cache: {hf_cache}")


if __name__ == '__main__':
    main()
