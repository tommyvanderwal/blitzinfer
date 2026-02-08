#!/usr/bin/env python3
"""
Production-ready test: 10+ model swaps with page cache warming.

This test validates that:
1. Page cache warming works in practice
2. Model switches are faster with warm cache
3. Memory is properly cleaned up between switches
4. Inference works correctly after each switch

Target: RTX PRO 6000 with GPT-OSS-120B and Qwen3-VL-32B-FP8
"""

import os
import gc
import time
import subprocess
import random
import string

# Configure vLLM for single-process mode with proper cleanup
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch
from blitzinfer.memory import PageCacheWarmer, WarmStatus

# Models to switch between
MODEL_A = "openai/gpt-oss-120b"
MODEL_B = "Qwen/Qwen3-VL-32B-Thinking-FP8"

# vLLM config for RTX PRO 6000
VLLM_CONFIG = {
    "dtype": "bfloat16",
    "max_model_len": 32768,  # Reasonable context for switching tests
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 4096,
    "enforce_eager": True,
    "trust_remote_code": True,
}

NUM_SWITCHES = 12
REQUESTS_PER_MODEL = 3
INPUT_TOKENS = 200
OUTPUT_TOKENS = 100


def nvidia_smi_free_gb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def nvidia_smi_used_gb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def generate_random_prompt(num_words: int = 50) -> str:
    """Generate random prompt to avoid caching."""
    words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
             for _ in range(num_words)]
    return f"Please analyze the following terms and explain each: {' '.join(words)}"


def unload_model(llm):
    """Properly unload vLLM model with full cleanup."""
    t0 = time.time()

    try:
        import torch._dynamo

        # Access internal structures
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
                            for param in model.parameters():
                                param.data = torch.empty(0, device='cpu')

                        # Clear KV caches
                        if hasattr(model_runner, 'kv_caches'):
                            for i, cache in enumerate(model_runner.kv_caches):
                                if cache is not None:
                                    model_runner.kv_caches[i] = torch.empty(0, device='cpu')
                            model_runner.kv_caches.clear()

                        # Clear cross attention KV cache (for VL models)
                        if hasattr(model_runner, 'cross_layers_kv_cache'):
                            if model_runner.cross_layers_kv_cache is not None:
                                model_runner.cross_layers_kv_cache = torch.empty(0, device='cpu')

                        # Clear attention layer KV caches
                        if hasattr(model_runner, 'compilation_config'):
                            sfc = getattr(model_runner.compilation_config, 'static_forward_context', None)
                            if sfc:
                                for layer_name, layer in sfc.items():
                                    if hasattr(layer, 'kv_cache') and layer.kv_cache:
                                        for i, kv in enumerate(layer.kv_cache):
                                            if kv is not None:
                                                layer.kv_cache[i] = torch.empty(0, device='cpu')
                                        layer.kv_cache = []

        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Cleanup warning: {e}")

    # Delete the LLM
    del llm

    # Clear vLLM global state
    try:
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            vllm_config.compilation_config.static_forward_context.clear()
    except Exception:
        pass

    try:
        import vllm.config.vllm as vllm_config_module
        vllm_config_module._current_vllm_config = None
        vllm_config_module._current_prefix = None
        vllm_config_module.get_cached_compilation_config.cache_clear()
    except Exception:
        pass

    # Reset dynamo
    try:
        torch._dynamo.reset()
    except Exception:
        pass

    # vLLM cleanup
    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except Exception:
        pass

    # Reset workspace
    try:
        from vllm.v1.worker.workspace import reset_workspace_manager
        reset_workspace_manager()
    except Exception:
        pass

    # GC
    gc.collect()
    gc.collect()

    # Clear CUDA
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    gc.collect()

    return time.time() - t0


def load_model(model_name: str):
    """Load a model with vLLM."""
    from vllm import LLM
    t0 = time.time()
    llm = LLM(model=model_name, **VLLM_CONFIG)
    load_time = time.time() - t0
    return llm, load_time


def run_inference(llm, prompts: list[str]) -> tuple[list[str], float]:
    """Run inference on multiple prompts."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=OUTPUT_TOKENS,
        temperature=0.7,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    infer_time = time.time() - t0

    results = [o.outputs[0].text[:50] + "..." for o in outputs]
    return results, infer_time


def main():
    print("=" * 80)
    print("PAGE CACHE WARMING: 10+ SWAP STRESS TEST")
    print("=" * 80)
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print(f"Switches: {NUM_SWITCHES}")
    print(f"Requests per model: {REQUESTS_PER_MODEL}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU free: {nvidia_smi_free_gb():.1f} GB")

    # Initialize page cache warmer
    warmer = PageCacheWarmer()

    # Register both models (will resolve paths via HF cache)
    print("\nRegistering models...")
    from huggingface_hub import snapshot_download

    path_a = snapshot_download(MODEL_A, local_files_only=True)
    path_b = snapshot_download(MODEL_B, local_files_only=True)

    from blitzinfer.memory.cache_warmer import get_model_size
    size_a = get_model_size(path_a) / 1e9
    size_b = get_model_size(path_b) / 1e9

    warmer.register_model(MODEL_A, path_a)
    warmer.register_model(MODEL_B, path_b)

    print(f"  {MODEL_A}: {size_a:.1f} GB")
    print(f"  {MODEL_B}: {size_b:.1f} GB")

    # Results tracking
    results = []
    models = [MODEL_A, MODEL_B]
    initial_free = nvidia_smi_free_gb()
    current_model = None
    llm = None

    for i in range(NUM_SWITCHES):
        switch_num = i + 1
        # Alternate models
        next_model = models[i % 2]
        warming_model = models[(i + 1) % 2]

        print(f"\n{'#' * 80}")
        print(f"# SWITCH {switch_num}/{NUM_SWITCHES}: {next_model.split('/')[-1]}")
        print(f"{'#' * 80}")

        try:
            # Unload current model (if any)
            unload_time = 0
            if llm is not None:
                print(f"\nUnloading {current_model.split('/')[-1]}...")
                print(f"  GPU before: {nvidia_smi_free_gb():.1f} GB free")
                unload_time = unload_model(llm)
                llm = None
                print(f"  GPU after:  {nvidia_smi_free_gb():.1f} GB free")
                print(f"  Unload time: {unload_time:.2f}s")

            # Check if next model is warm
            is_warm = warmer.is_warm(next_model)
            warm_status = warmer.get_status(next_model).name

            print(f"\nLoading {next_model.split('/')[-1]}...")
            print(f"  Cache status: {warm_status}")

            # Load the model
            llm, load_time = load_model(next_model)
            current_model = next_model

            print(f"  Load time: {load_time:.2f}s {'[WARM]' if is_warm else '[COLD]'}")
            print(f"  GPU after load: {nvidia_smi_free_gb():.1f} GB free")

            # Start warming the OTHER model while this one is serving
            if not warmer.is_warm(warming_model) and not warmer.is_warming(warming_model):
                print(f"\n  Starting background warming of {warming_model.split('/')[-1]}...")
                warmer.start_warming(warming_model)

            # Run inference
            print(f"\n  Running {REQUESTS_PER_MODEL} inference requests...")
            prompts = [generate_random_prompt(INPUT_TOKENS // 5) for _ in range(REQUESTS_PER_MODEL)]

            responses, infer_time = run_inference(llm, prompts)

            for j, resp in enumerate(responses):
                print(f"    Request {j+1}: {resp}")
            print(f"  Inference time: {infer_time:.2f}s")

            # Check warming progress
            warming_status = warmer.get_status(warming_model)
            warming_progress = warmer.get_progress(warming_model) * 100
            print(f"\n  Background warming: {warming_model.split('/')[-1]} - {warming_status.name} ({warming_progress:.0f}%)")

            # Record results
            results.append({
                'switch': switch_num,
                'model': next_model.split('/')[-1],
                'unload_time': unload_time,
                'load_time': load_time,
                'infer_time': infer_time,
                'was_warm': is_warm,
                'gpu_free_after': nvidia_smi_free_gb(),
                'success': True,
            })

        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'switch': switch_num,
                'model': next_model.split('/')[-1],
                'success': False,
                'error': str(e),
            })

            # Try to recover
            if llm is not None:
                try:
                    unload_model(llm)
                except:
                    pass
                llm = None

            gc.collect()
            torch.cuda.empty_cache()

    # Final cleanup
    if llm is not None:
        unload_model(llm)

    warmer.shutdown()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    warm_loads = [r for r in successful if r['was_warm']]
    cold_loads = [r for r in successful if not r['was_warm']]

    print(f"\nTotal switches: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if cold_loads:
        avg_cold_load = sum(r['load_time'] for r in cold_loads) / len(cold_loads)
        print(f"\nCold loads: {len(cold_loads)}")
        print(f"  Average load time: {avg_cold_load:.2f}s")

    if warm_loads:
        avg_warm_load = sum(r['load_time'] for r in warm_loads) / len(warm_loads)
        print(f"\nWarm loads: {len(warm_loads)}")
        print(f"  Average load time: {avg_warm_load:.2f}s")

        if cold_loads:
            speedup = avg_cold_load / avg_warm_load
            time_saved = avg_cold_load - avg_warm_load
            print(f"  Speedup vs cold: {speedup:.2f}x")
            print(f"  Time saved per load: {time_saved:.1f}s")

    if successful:
        avg_unload = sum(r['unload_time'] for r in successful) / len(successful)
        avg_infer = sum(r['infer_time'] for r in successful) / len(successful)
        print(f"\nAverage unload time: {avg_unload:.2f}s")
        print(f"Average inference time: {avg_infer:.2f}s")

    # Memory check
    final_free = nvidia_smi_free_gb()
    memory_diff = initial_free - final_free

    print(f"\nMemory check:")
    print(f"  Initial free: {initial_free:.1f} GB")
    print(f"  Final free: {final_free:.1f} GB")
    print(f"  Difference: {memory_diff:+.1f} GB")

    if abs(memory_diff) < 2.0:
        print("  -> Memory stable (no significant leak)")
    else:
        print(f"  -> WARNING: Memory difference of {memory_diff:.1f} GB!")

    # GPU free over time
    print("\nGPU free after each switch:")
    for r in successful:
        warm_str = "[WARM]" if r['was_warm'] else "[COLD]"
        print(f"  Switch {r['switch']:2d} ({r['model'][:12]:12s}): {r['gpu_free_after']:.1f} GB {warm_str} load: {r['load_time']:.1f}s")

    if failed:
        print("\nFailed switches:")
        for r in failed:
            print(f"  Switch {r['switch']}: {r.get('error', 'Unknown')}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
