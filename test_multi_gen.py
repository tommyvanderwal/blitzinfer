#!/usr/bin/env python3
"""
Test multiple generations per model to see if inference causes accumulation.
"""

import gc
import os
import sys
import time
import types

os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch
import psutil


def get_mem():
    """Get GPU and system memory."""
    free, total = torch.cuda.mem_get_info()
    gpu_used = (total - free) / (1024**3)
    proc_rss = psutil.Process().memory_info().rss / (1024**3)
    return gpu_used, proc_rss


def comprehensive_cleanup(llm):
    """Same cleanup as vllm_adapter."""
    import torch._dynamo
    import multiprocessing

    try:
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
                        if hasattr(model_runner, 'model'):
                            model = model_runner.model
                            for param in model.parameters():
                                param.data = torch.empty(0, device='cpu')

                        if hasattr(model_runner, 'kv_caches'):
                            for i, cache in enumerate(model_runner.kv_caches):
                                if cache is not None and hasattr(cache, 'device') and cache.device.type == 'cuda':
                                    model_runner.kv_caches[i] = torch.empty(0, device='cpu')
                            model_runner.kv_caches.clear()

                        if hasattr(model_runner, 'compilation_config'):
                            sfc = getattr(model_runner.compilation_config, 'static_forward_context', None)
                            if sfc:
                                for layer_name, layer in sfc.items():
                                    if hasattr(layer, 'kv_cache') and layer.kv_cache:
                                        for j, kv in enumerate(layer.kv_cache):
                                            if kv is not None and hasattr(kv, 'device') and kv.device.type == 'cuda':
                                                layer.kv_cache[j] = torch.empty(0, device='cpu')
                                        layer.kv_cache = []
                        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Cleanup error: {e}")

    del llm

    try:
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            vllm_config.compilation_config.static_forward_context.clear()
    except:
        pass

    try:
        import vllm.config.vllm as vllm_config_module
        vllm_config_module._current_vllm_config = None
        vllm_config_module._current_prefix = None
        vllm_config_module.get_cached_compilation_config.cache_clear()
    except:
        pass

    try:
        torch._dynamo.reset()
    except:
        pass

    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except:
        pass

    try:
        from vllm.v1.worker.workspace import reset_workspace_manager
        reset_workspace_manager()
    except:
        pass

    for child in multiprocessing.active_children():
        child.join(timeout=5.0)
        if child.is_alive():
            child.terminate()
            child.join(timeout=2.0)

    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    from vllm import LLM, SamplingParams

    print("="*70)
    print("TEST: Multiple generations per model (mimics queue test)")
    print("="*70)

    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    initial_gpu, initial_rss = get_mem()
    print(f"Initial: GPU={initial_gpu:.2f}GB, RSS={initial_rss:.2f}GB")

    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.25,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    prompts = [
        "Write a short paragraph about the color blue:",
        "Explain what makes the sky appear blue:",
        "List five things that are commonly blue:",
        "Describe a blue ocean sunset:",
    ]

    num_gens_per_model = 2  # Same as queue test

    for switch_num in range(1, 8):
        model = models[(switch_num - 1) % 2]

        print(f"\n{'='*70}")
        print(f"SWITCH {switch_num}: {model}")
        print(f"{'='*70}")

        gpu_before, rss_before = get_mem()
        print(f"Before load: GPU={gpu_before:.2f}GB, RSS={rss_before:.2f}GB")

        try:
            t0 = time.perf_counter()
            llm = LLM(model=model, **config)
            load_time = time.perf_counter() - t0
            print(f"Loaded in {load_time:.2f}s")

            gpu_after_load, rss_after_load = get_mem()
            print(f"After load: GPU={gpu_after_load:.2f}GB, RSS={rss_after_load:.2f}GB")

            # Do multiple generations (like queue test)
            sampling_params = SamplingParams(max_tokens=100, temperature=0.7)
            for gen_num in range(num_gens_per_model):
                prompt = prompts[(switch_num * 2 + gen_num) % len(prompts)]
                print(f"  Gen {gen_num + 1}: {prompt[:40]}...")

                t0 = time.perf_counter()
                outputs = llm.generate([prompt], sampling_params)
                gen_time = time.perf_counter() - t0

                text = outputs[0].outputs[0].text[:30] if outputs else ""
                print(f"    -> {text}... ({gen_time:.1f}s)")

            gpu_after_gen, rss_after_gen = get_mem()
            print(f"After gens: GPU={gpu_after_gen:.2f}GB, RSS={rss_after_gen:.2f}GB")

            # Cleanup
            comprehensive_cleanup(llm)

            gpu_after_clean, rss_after_clean = get_mem()
            print(f"After cleanup: GPU={gpu_after_clean:.2f}GB, RSS={rss_after_clean:.2f}GB")
            print(f"Delta from initial: GPU={gpu_after_clean - initial_gpu:+.2f}GB, RSS={rss_after_clean - initial_rss:+.2f}GB")

            time.sleep(1)

        except Exception as e:
            print(f"\n*** CRASH on switch {switch_num}: {e} ***")
            import traceback
            traceback.print_exc()
            break

    print(f"\n{'='*70}")
    print("FINAL")
    print(f"{'='*70}")
    final_gpu, final_rss = get_mem()
    print(f"Final: GPU={final_gpu:.2f}GB, RSS={final_rss:.2f}GB")
    print(f"Delta from initial: GPU={final_gpu - initial_gpu:+.2f}GB, RSS={final_rss - initial_rss:+.2f}GB")


if __name__ == "__main__":
    main()
