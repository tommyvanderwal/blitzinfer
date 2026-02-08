#!/usr/bin/env python3
"""
Stress test: 10 switches with 7B models to verify stability.
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
    free, total = torch.cuda.mem_get_info()
    gpu_used = (total - free) / (1024**3)
    proc_rss = psutil.Process().memory_info().rss / (1024**3)
    return gpu_used, proc_rss


def comprehensive_cleanup(llm):
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
                            for param in model_runner.model.parameters():
                                param.data = torch.empty(0, device='cpu')

                        if hasattr(model_runner, 'kv_caches'):
                            for i, cache in enumerate(model_runner.kv_caches):
                                if cache is not None and hasattr(cache, 'device') and cache.device.type == 'cuda':
                                    model_runner.kv_caches[i] = torch.empty(0, device='cpu')
                            model_runner.kv_caches.clear()

                        if hasattr(model_runner, 'compilation_config'):
                            sfc = getattr(model_runner.compilation_config, 'static_forward_context', None)
                            if sfc:
                                for layer in sfc.values():
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
    print("STRESS TEST: 10 switches with 7B models")
    print("="*70)

    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    initial_gpu, initial_rss = get_mem()
    print(f"Initial: GPU={initial_gpu:.2f}GB, RSS={initial_rss:.2f}GB\n")

    # Same config that works in queue test
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

    results = []

    for switch_num in range(1, 11):  # 10 switches
        model = models[(switch_num - 1) % 2]
        model_short = "Qwen-7B" if "Qwen" in model else "Mistral-7B"

        gpu_before, rss_before = get_mem()

        try:
            t0 = time.perf_counter()
            llm = LLM(model=model, **config)
            load_time = time.perf_counter() - t0

            # One quick generation
            outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5, temperature=0.7))
            text = outputs[0].outputs[0].text.strip() if outputs else ""

            gpu_loaded, _ = get_mem()

            # Cleanup
            comprehensive_cleanup(llm)

            gpu_after, rss_after = get_mem()

            result = {
                'switch': switch_num,
                'model': model_short,
                'load_time': load_time,
                'gpu_before': gpu_before,
                'gpu_loaded': gpu_loaded,
                'gpu_after': gpu_after,
                'rss_after': rss_after,
                'success': True,
            }
            results.append(result)

            print(f"Switch {switch_num:2d}: {model_short:12s} | Load: {load_time:5.1f}s | "
                  f"GPU: {gpu_before:.1f}→{gpu_loaded:.1f}→{gpu_after:.1f}GB | "
                  f"RSS: {rss_after:.2f}GB | ✓")

            time.sleep(1)

        except Exception as e:
            print(f"Switch {switch_num:2d}: {model_short:12s} | CRASHED: {e}")
            results.append({
                'switch': switch_num,
                'model': model_short,
                'success': False,
                'error': str(e),
            })
            break

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful = [r for r in results if r['success']]
    print(f"Successful switches: {len(successful)}/10")

    if successful:
        avg_load = sum(r['load_time'] for r in successful) / len(successful)
        avg_gpu = sum(r['gpu_after'] for r in successful) / len(successful)
        max_gpu = max(r['gpu_after'] for r in successful)
        final_rss = successful[-1]['rss_after']

        print(f"Average load time: {avg_load:.1f}s")
        print(f"Average GPU after cleanup: {avg_gpu:.2f}GB")
        print(f"Max GPU after cleanup: {max_gpu:.2f}GB")
        print(f"Final RSS: {final_rss:.2f}GB (delta from initial: {final_rss - initial_rss:+.2f}GB)")

    final_gpu, final_rss = get_mem()
    print(f"\nFinal: GPU={final_gpu:.2f}GB, RSS={final_rss:.2f}GB")
    print(f"Delta from initial: GPU={final_gpu - initial_gpu:+.2f}GB, RSS={final_rss - initial_rss:+.2f}GB")


if __name__ == "__main__":
    main()
