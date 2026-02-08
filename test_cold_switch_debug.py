#!/usr/bin/env python3
"""Debug test for cold-only model switching.

Tests pure cold loads (no standby) to isolate vLLM state issues:
1. Cold load gpt-oss-120b
2. Unload
3. Cold load Qwen3-32B-FP8
4. Unload
5. Cold load gpt-oss-120b again

This helps determine if switch failures are standby-related or vLLM state-related.
"""

import os
import sys
import time
import gc
import logging

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Configure environment before importing vLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch

# Configure logging with immediate flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def log(msg):
    """Print with immediate flush."""
    print(msg, flush=True)


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        return 0.0


def unload_llm_properly(llm):
    """Properly unload LLM and free GPU memory."""
    log(f"  GPU memory before cleanup: {get_gpu_memory_mb():.0f}MB")
    log("  Clearing model weights from GPU...")
    try:
        # Navigate to model runner - handle V1 engine structure
        engine_core = llm.llm_engine.engine_core
        if hasattr(engine_core, 'engine_core'):
            core = engine_core.engine_core
        else:
            core = engine_core

        model_runner = None
        if hasattr(core, 'model_executor'):
            executor = core.model_executor
            if hasattr(executor, 'driver_worker'):
                worker = executor.driver_worker
                # V1 has nested worker.worker structure
                if hasattr(worker, 'worker') and worker.worker is not None:
                    model_runner = getattr(worker.worker, 'model_runner', None)
                else:
                    model_runner = getattr(worker, 'model_runner', None)

        if model_runner is not None:
            # Clear model parameters
            if hasattr(model_runner, 'model') and model_runner.model is not None:
                model = model_runner.model
                param_count = 0
                for param in model.parameters():
                    param.data = torch.empty(0, device='cpu')
                    param_count += 1
                buf_count = 0
                for buf in model.buffers():
                    buf.data = torch.empty(0, device='cpu')
                    buf_count += 1
                log(f"  Cleared {param_count} params, {buf_count} buffers")

            # Clear KV caches
            if hasattr(model_runner, 'kv_caches') and model_runner.kv_caches:
                cache_count = len(model_runner.kv_caches)
                for i, cache in enumerate(model_runner.kv_caches):
                    if cache is not None:
                        model_runner.kv_caches[i] = None
                model_runner.kv_caches.clear()
                log(f"  Cleared {cache_count} KV caches")

            # Clear compilation_config.static_forward_context
            if hasattr(model_runner, 'compilation_config'):
                sfc = getattr(model_runner.compilation_config, 'static_forward_context', None)
                if sfc:
                    for layer in sfc.values():
                        if hasattr(layer, 'kv_cache'):
                            layer.kv_cache = []

            model_runner.model = None
            log(f"  GPU memory after clearing model: {get_gpu_memory_mb():.0f}MB")
        else:
            log("  Warning: Could not find model_runner")
    except Exception as e:
        log(f"  Warning: cleanup error: {e}")
        import traceback
        traceback.print_exc()

    gc.collect()
    torch.cuda.empty_cache()
    log(f"  GPU memory after gc+empty_cache: {get_gpu_memory_mb():.0f}MB")

    # Shutdown engine
    log("  Shutting down engine core...")
    try:
        llm.llm_engine.engine_core.shutdown()
        log("  Engine core shutdown")
    except Exception as e:
        log(f"  Engine shutdown error: {e}")

    log("  Deleting llm object...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  GPU memory after del llm: {get_gpu_memory_mb():.0f}MB")

    # Reset vLLM distributed state using the comprehensive cleanup
    try:
        from vllm.distributed import parallel_state
        if hasattr(parallel_state, 'cleanup_dist_env_and_memory'):
            parallel_state.cleanup_dist_env_and_memory(shutdown_ray=False)
            log("  Cleaned up dist env and memory")
        else:
            if hasattr(parallel_state, 'destroy_model_parallel'):
                parallel_state.destroy_model_parallel()
                log("  Destroyed model parallel state")
            if hasattr(parallel_state, 'destroy_distributed_environment'):
                parallel_state.destroy_distributed_environment()
                log("  Destroyed distributed environment")
    except Exception as e:
        log(f"  Could not reset parallel state: {e}")

    # Reset torch._dynamo cache (compilation artifacts)
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        log("  Reset torch._dynamo")
    except Exception as e:
        log(f"  Could not reset dynamo: {e}")

    # Clear any cached custom ops
    try:
        if hasattr(torch, '_custom_ops'):
            log("  Note: torch._custom_ops exists")
    except Exception:
        pass

    # Try to reset vLLM's custom op state
    try:
        from vllm.model_executor import custom_op
        if hasattr(custom_op, '_op_implementations'):
            custom_op._op_implementations.clear()
            log("  Cleared custom_op implementations")
    except Exception as e:
        log(f"  Could not reset custom_op: {e}")

    # CRITICAL: Clear the rotary embedding cache
    try:
        from vllm.model_executor.layers import rotary_embedding
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rope_dict = rotary_embedding._ROPE_DICT
            log(f"  Clearing _ROPE_DICT with {len(rope_dict)} entries")
            rope_dict.clear()
            log("  Cleared _ROPE_DICT")
    except Exception as e:
        log(f"  Could not clear _ROPE_DICT: {e}")

    # CUDA cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

    mem_after = get_gpu_memory_mb()
    log(f"  GPU memory after cleanup: {mem_after:.0f}MB")


def load_model(model_name: str, vllm_kwargs: dict):
    """Load a model and verify it works."""
    from vllm import LLM, SamplingParams

    log(f"  Loading {model_name}...")
    t0 = time.perf_counter()

    llm = LLM(model=model_name, **vllm_kwargs)

    load_time = time.perf_counter() - t0
    log(f"  Loaded in {load_time:.1f}s")

    # Verify model works
    log(f"  Verifying model...")
    outputs = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=10))
    log(f"  Output: {outputs[0].outputs[0].text.strip()}")

    return llm, load_time


def main():
    from vllm import LLM, SamplingParams

    # Configuration
    MODEL_A = "openai/gpt-oss-120b"     # ~60GB, MoE
    MODEL_B = "Qwen/Qwen3-32B-FP8"      # ~33GB, dense

    VLLM_KWARGS = {
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "trust_remote_code": True,
    }

    log("=" * 70)
    log("COLD-ONLY MODEL SWITCH DEBUG TEST")
    log("=" * 70)
    log(f"Model A: {MODEL_A}")
    log(f"Model B: {MODEL_B}")
    log("")

    llm = None

    try:
        # Step 1: Cold load gpt-oss-120b
        log("-" * 50)
        log("[STEP 1] Cold load gpt-oss-120b")
        log("-" * 50)
        llm, time1 = load_model(MODEL_A, VLLM_KWARGS)
        mem1 = get_gpu_memory_mb()
        log(f"  GPU memory: {mem1:.0f}MB")

        # Step 2: Unload
        log("\n[STEP 2] Unloading gpt-oss-120b...")
        unload_llm_properly(llm)
        llm = None

        # Step 3: Cold load Qwen3-32B-FP8
        log("-" * 50)
        log("[STEP 3] Cold load Qwen3-32B-FP8")
        log("-" * 50)
        llm, time2 = load_model(MODEL_B, VLLM_KWARGS)
        mem2 = get_gpu_memory_mb()
        log(f"  GPU memory: {mem2:.0f}MB")

        # Step 4: Unload
        log("\n[STEP 4] Unloading Qwen3-32B-FP8...")
        unload_llm_properly(llm)
        llm = None

        # Step 5: Cold load gpt-oss-120b again
        log("-" * 50)
        log("[STEP 5] Cold load gpt-oss-120b AGAIN")
        log("-" * 50)
        llm, time3 = load_model(MODEL_A, VLLM_KWARGS)
        mem3 = get_gpu_memory_mb()
        log(f"  GPU memory: {mem3:.0f}MB")

        # Summary
        log("\n" + "=" * 70)
        log("SUCCESS - All cold loads completed!")
        log("=" * 70)
        log(f"Load 1 (gpt-oss): {time1:.1f}s, {mem1:.0f}MB")
        log(f"Load 2 (Qwen):    {time2:.1f}s, {mem2:.0f}MB")
        log(f"Load 3 (gpt-oss): {time3:.1f}s, {mem3:.0f}MB")

    except Exception as e:
        log(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        log("\nCleaning up...")
        if llm is not None:
            try:
                unload_llm_properly(llm)
            except Exception as e:
                log(f"Final cleanup error: {e}")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
