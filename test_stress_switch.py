#!/usr/bin/env python3
"""Stress test: 10 model switches with 2 parallel queries each time.

Tests single-process mode cleanup on RTX PRO 6000.
"""
import os
import gc
import time
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Single-process mode with proper cleanup
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch

# Two different architecture models for cross-arch switching
MODEL_A = "openai/gpt-oss-120b"
MODEL_B = "Qwen/Qwen3-32B"

# vLLM config for RTX PRO 6000 (95GB VRAM)
VLLM_CONFIG = {
    "dtype": "bfloat16",
    "max_model_len": 8192,  # Smaller for faster load during stress test
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 16,
    "max_num_batched_tokens": 4096,
    "enforce_eager": True,
    "trust_remote_code": True,
}

NUM_SWITCHES = 10
QUERIES_PER_MODEL = 2

TEST_PROMPTS = [
    "Explain quantum computing in one sentence:",
    "What is the capital of France? Answer:",
]


def nvidia_smi_free_gb():
    """Get free GPU memory from nvidia-smi."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def nvidia_smi_used_gb():
    """Get used GPU memory from nvidia-smi."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def gpu_processes():
    """Get GPU processes."""
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return result.stdout.strip() or "(none)"


def load_model(model_name: str):
    """Load a model with vLLM."""
    from vllm import LLM

    print(f"\n{'='*60}")
    print(f"Loading: {model_name}")
    print(f"GPU free before load: {nvidia_smi_free_gb():.1f}GB")

    t0 = time.time()
    llm = LLM(model=model_name, **VLLM_CONFIG)
    load_time = time.time() - t0

    print(f"Load time: {load_time:.2f}s")
    print(f"GPU free after load: {nvidia_smi_free_gb():.1f}GB")

    return llm, load_time


def run_parallel_queries(llm, prompts: list[str]):
    """Run multiple queries in parallel using vLLM's batch API."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=32,
        temperature=0.7,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    query_time = time.time() - t0

    results = []
    for output in outputs:
        text = output.outputs[0].text.strip()[:50]  # First 50 chars
        results.append(text)

    return results, query_time


def unload_model(llm, model_name: str):
    """Unload model with proper cleanup."""
    print(f"\nUnloading: {model_name}")
    print(f"GPU free before unload: {nvidia_smi_free_gb():.1f}GB")

    t0 = time.time()

    # Use our cleanup approach from vllm_adapter.py
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

    unload_time = time.time() - t0
    print(f"Unload time: {unload_time:.2f}s")
    print(f"GPU free after unload: {nvidia_smi_free_gb():.1f}GB")

    return unload_time


def main():
    print("="*60)
    print("STRESS TEST: 10 Model Switches with Parallel Queries")
    print("="*60)
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print(f"VLLM_ENABLE_V1_MULTIPROCESSING: {os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')}")
    print(f"GPU Total: ~95GB (RTX PRO 6000)")
    print(f"Initial GPU free: {nvidia_smi_free_gb():.1f}GB")
    print(f"GPU processes: {gpu_processes()}")

    models = [MODEL_A, MODEL_B]
    results = []

    initial_free = nvidia_smi_free_gb()

    for i in range(NUM_SWITCHES):
        model_name = models[i % 2]
        switch_num = i + 1

        print(f"\n{'#'*60}")
        print(f"SWITCH {switch_num}/{NUM_SWITCHES}: Loading {model_name.split('/')[-1]}")
        print(f"{'#'*60}")

        try:
            # Load model
            llm, load_time = load_model(model_name)

            # Run parallel queries
            print(f"\nRunning {QUERIES_PER_MODEL} parallel queries...")
            query_results, query_time = run_parallel_queries(llm, TEST_PROMPTS[:QUERIES_PER_MODEL])

            for j, (prompt, result) in enumerate(zip(TEST_PROMPTS[:QUERIES_PER_MODEL], query_results)):
                print(f"  Query {j+1}: {result}...")
            print(f"Query time: {query_time:.2f}s")

            # Unload model
            unload_time = unload_model(llm, model_name)

            # Record results
            free_after = nvidia_smi_free_gb()
            results.append({
                'switch': switch_num,
                'model': model_name.split('/')[-1],
                'load_time': load_time,
                'query_time': query_time,
                'unload_time': unload_time,
                'gpu_free_after': free_after,
                'success': True,
            })

            print(f"\nSwitch {switch_num} complete - GPU free: {free_after:.1f}GB")

        except Exception as e:
            print(f"\nERROR on switch {switch_num}: {e}")
            results.append({
                'switch': switch_num,
                'model': model_name.split('/')[-1],
                'success': False,
                'error': str(e),
            })

            # Try to recover
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"Successful switches: {len(successful)}/{NUM_SWITCHES}")
    print(f"Failed switches: {len(failed)}/{NUM_SWITCHES}")

    if successful:
        avg_load = sum(r['load_time'] for r in successful) / len(successful)
        avg_query = sum(r['query_time'] for r in successful) / len(successful)
        avg_unload = sum(r['unload_time'] for r in successful) / len(successful)

        print(f"\nAverage load time: {avg_load:.2f}s")
        print(f"Average query time: {avg_query:.2f}s")
        print(f"Average unload time: {avg_unload:.2f}s")

        # Check for memory leak
        final_free = nvidia_smi_free_gb()
        memory_diff = initial_free - final_free

        print(f"\nMemory check:")
        print(f"  Initial free: {initial_free:.1f}GB")
        print(f"  Final free: {final_free:.1f}GB")
        print(f"  Difference: {memory_diff:+.1f}GB")

        if abs(memory_diff) < 2.0:
            print("  -> Memory stable (no significant leak detected)")
        else:
            print(f"  -> WARNING: Memory difference of {memory_diff:.1f}GB detected!")

        # GPU free over time
        print("\nGPU free after each switch:")
        for r in successful:
            print(f"  Switch {r['switch']:2d} ({r['model'][:10]:10s}): {r['gpu_free_after']:.1f}GB")

    if failed:
        print("\nFailed switches:")
        for r in failed:
            print(f"  Switch {r['switch']}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
