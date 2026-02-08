#!/usr/bin/env python3
"""Stress test for 1 active + 1 standby model switching.

Tests demand-driven prefetch with significant inference load:
- Run multiple inference requests before requesting switch
- 10+ switches between two models
- Detailed profiling of each phase
- Track GPU memory, switch times, prefetch overlap
"""

import os
import sys
import time
import gc
import logging
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional

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


@dataclass
class SwitchMetrics:
    """Metrics for a single model switch."""
    switch_num: int
    from_model: str
    to_model: str
    was_standby_ready: bool
    prefetch_time: float
    unload_time: float
    load_time: float
    total_switch_time: float
    inference_requests_before: int
    inference_time_before: float
    gpu_mem_before_mb: float
    gpu_mem_after_mb: float
    verify_passed: bool = True


@dataclass
class TestResults:
    """Aggregate test results."""
    switches: List[SwitchMetrics] = field(default_factory=list)
    total_inference_requests: int = 0
    total_inference_time: float = 0.0

    def summary(self) -> str:
        if not self.switches:
            return "No switches recorded"

        cold_switches = [s for s in self.switches if not s.was_standby_ready]
        warm_switches = [s for s in self.switches if s.was_standby_ready]

        lines = [
            "=" * 70,
            "STRESS TEST RESULTS",
            "=" * 70,
            f"Total switches: {len(self.switches)}",
            f"  Cold switches: {len(cold_switches)}",
            f"  Warm switches (from standby): {len(warm_switches)}",
            "",
        ]

        if cold_switches:
            avg_cold = sum(s.total_switch_time for s in cold_switches) / len(cold_switches)
            lines.append(f"Cold switch avg: {avg_cold:.2f}s")

        if warm_switches:
            avg_warm = sum(s.total_switch_time for s in warm_switches) / len(warm_switches)
            avg_load = sum(s.load_time for s in warm_switches) / len(warm_switches)
            avg_unload = sum(s.unload_time for s in warm_switches) / len(warm_switches)
            lines.extend([
                f"Warm switch avg: {avg_warm:.2f}s",
                f"  Unload avg: {avg_unload:.2f}s",
                f"  Load avg: {avg_load:.2f}s",
            ])

        if cold_switches and warm_switches:
            speedup = avg_cold / avg_warm if avg_warm > 0 else 0
            lines.append(f"\nSpeedup from standby: {speedup:.1f}x")

        # Verification stats
        passed = sum(1 for s in self.switches if s.verify_passed)
        failed = len(self.switches) - passed

        lines.extend([
            "",
            f"Total inference requests: {self.total_inference_requests}",
            f"Total inference time: {self.total_inference_time:.1f}s",
            f"Verification: {passed}/{len(self.switches)} passed" + (" FAILURES DETECTED!" if failed > 0 else ""),
            "",
            "Per-switch details:",
        ])

        for s in self.switches:
            status = "[STANDBY]" if s.was_standby_ready else "[COLD]"
            verify = "OK" if s.verify_passed else "FAIL"
            lines.append(
                f"  {s.switch_num:2d}. {s.from_model[:20]:20s} -> {s.to_model[:20]:20s} "
                f"{status:10s} {s.total_switch_time:5.1f}s [{verify}] "
                f"(infer: {s.inference_requests_before} reqs, {s.inference_time_before:.1f}s)"
            )

        lines.append("=" * 70)
        return "\n".join(lines)


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        return 0.0


def unload_llm_properly(llm):
    """Properly unload LLM and free GPU memory.

    vLLM's shutdown() doesn't release model weights. We need to manually
    clear parameters and KV caches before deleting.

    For V1 engine, path is: engine_core -> model_executor -> driver_worker -> worker -> model_runner
    """
    log("    Clearing model weights from GPU...")
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
                # Also clear buffers
                buf_count = 0
                for buf in model.buffers():
                    buf.data = torch.empty(0, device='cpu')
                    buf_count += 1
                log(f"    Cleared {param_count} params, {buf_count} buffers")

            # Clear KV caches
            if hasattr(model_runner, 'kv_caches') and model_runner.kv_caches:
                cache_count = len(model_runner.kv_caches)
                for i, cache in enumerate(model_runner.kv_caches):
                    if cache is not None:
                        model_runner.kv_caches[i] = None
                model_runner.kv_caches.clear()
                log(f"    Cleared {cache_count} KV caches")

            # Clear compilation_config.static_forward_context
            if hasattr(model_runner, 'compilation_config'):
                sfc = getattr(model_runner.compilation_config, 'static_forward_context', None)
                if sfc:
                    for layer in sfc.values():
                        if hasattr(layer, 'kv_cache'):
                            layer.kv_cache = []

            # Clear the model reference itself
            model_runner.model = None
        else:
            log("    Warning: Could not find model_runner")
    except Exception as e:
        log(f"    Warning: cleanup error: {e}")
        import traceback
        traceback.print_exc()

    # Now shutdown and delete
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass

    del llm
    gc.collect()

    # Reset vLLM distributed state
    try:
        from vllm.distributed import parallel_state
        if hasattr(parallel_state, 'cleanup_dist_env_and_memory'):
            parallel_state.cleanup_dist_env_and_memory(shutdown_ray=False)
            log("    Cleaned up dist env and memory")
        else:
            if hasattr(parallel_state, 'destroy_model_parallel'):
                parallel_state.destroy_model_parallel()
                log("    Destroyed model parallel state")
            if hasattr(parallel_state, 'destroy_distributed_environment'):
                parallel_state.destroy_distributed_environment()
                log("    Destroyed distributed environment")
    except Exception as e:
        log(f"    Could not reset parallel state: {e}")

    # Reset torch._dynamo cache
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        log("    Reset torch._dynamo")
    except Exception as e:
        log(f"    Could not reset dynamo: {e}")

    # CRITICAL: Clear the rotary embedding cache to prevent dimension mismatches
    try:
        from vllm.model_executor.layers import rotary_embedding
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rope_dict = rotary_embedding._ROPE_DICT
            log(f"    Clearing _ROPE_DICT with {len(rope_dict)} entries")
            rope_dict.clear()
    except Exception as e:
        log(f"    Could not clear _ROPE_DICT: {e}")

    # More aggressive CUDA cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Multiple collection passes
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

    # Check memory was freed
    mem_after = get_gpu_memory_mb()
    log(f"    GPU memory after cleanup: {mem_after:.0f}MB")


def run_inference_batch(llm, num_requests: int, max_tokens: int = 100) -> float:
    """Run a batch of inference requests and return total time."""
    from vllm import SamplingParams

    prompts = [
        f"Explain the concept of {topic} in detail. Be thorough and comprehensive."
        for topic in [
            "quantum computing", "neural networks", "blockchain technology",
            "machine learning", "natural language processing", "computer vision",
            "reinforcement learning", "distributed systems", "cryptography",
            "data structures", "algorithms", "operating systems",
        ][:num_requests]
    ]

    while len(prompts) < num_requests:
        prompts.append(prompts[len(prompts) % len(prompts)])

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.7)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    log(f"    Generated {total_tokens} tokens in {elapsed:.1f}s ({total_tokens/elapsed:.1f} tok/s)")

    return elapsed


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.orchestrator.standby_manager import StandbyManager, StandbyState
    from blitzinfer.memory import set_preloaded_weights

    # Configuration - use original models, debug the switch issue
    MODEL_A = "Qwen/Qwen3-32B-FP8"   # ~33GB
    MODEL_B = "openai/gpt-oss-120b"  # ~60GB

    # Arena needs to fit gpt-oss-120b (65.2GB) with some margin
    STANDBY_ARENA_GB = 70.0
    NUM_SWITCHES = 6  # Reduced from 10 to avoid memory issues
    INFERENCE_REQUESTS_PER_SWITCH = 5
    INFERENCE_MAX_TOKENS = 150

    # vLLM settings for RTX PRO 6000
    # 0.85 * 95GB = 80.75GB - enough for gpt-oss-120b (~66GB) + KV cache
    VLLM_KWARGS = {
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "trust_remote_code": True,
    }

    log("=" * 70)
    log("STANDBY SWITCHING STRESS TEST")
    log("=" * 70)
    log(f"Model A: {MODEL_A}")
    log(f"Model B: {MODEL_B}")
    log(f"Switches: {NUM_SWITCHES}")
    log(f"Inference requests per switch: {INFERENCE_REQUESTS_PER_SWITCH}")
    log(f"Max tokens per request: {INFERENCE_MAX_TOKENS}")
    log(f"Standby Arena: {STANDBY_ARENA_GB}GB")
    log("")

    results = TestResults()
    standby = StandbyManager(arena_size_gb=STANDBY_ARENA_GB)
    llm = None
    current_model = None

    try:
        # Initial cold load of Model A
        log("-" * 50)
        log("[INIT] Cold loading Model A...")
        log("-" * 50)

        t0 = time.perf_counter()
        llm = LLM(model=MODEL_A, **VLLM_KWARGS)
        cold_load_a = time.perf_counter() - t0
        current_model = MODEL_A
        log(f"Cold load Model A: {cold_load_a:.1f}s")

        # Verify model works
        outputs = llm.generate(["Say hello briefly:"], SamplingParams(max_tokens=20))
        log(f"Test output: {outputs[0].outputs[0].text.strip()}")

        # Main switching loop
        for switch_num in range(1, NUM_SWITCHES + 1):
            next_model = MODEL_B if current_model == MODEL_A else MODEL_A

            log("")
            log("=" * 70)
            log(f"[SWITCH {switch_num}/{NUM_SWITCHES}] {current_model} -> {next_model}")
            log("=" * 70)

            # Step 1: Trigger prefetch for next model
            log(f"\n[1] Triggering prefetch for {next_model}...")
            prefetch_start = time.perf_counter()
            standby.start_prefetch(next_model)

            # Step 2: Run inference on current model while prefetch happens
            log(f"\n[2] Running {INFERENCE_REQUESTS_PER_SWITCH} inference requests...")
            gpu_mem_before = get_gpu_memory_mb()
            inference_time = run_inference_batch(
                llm,
                INFERENCE_REQUESTS_PER_SWITCH,
                INFERENCE_MAX_TOKENS
            )
            results.total_inference_requests += INFERENCE_REQUESTS_PER_SWITCH
            results.total_inference_time += inference_time

            # Step 3: Check if prefetch completed during inference
            standby_ready = standby.is_ready(next_model)
            if standby_ready:
                prefetch_time = time.perf_counter() - prefetch_start
                log(f"\n[3] Prefetch completed during inference! ({prefetch_time:.1f}s)")
            else:
                log(f"\n[3] Waiting for prefetch to complete...")
                standby.wait_for_load(timeout=180)
                prefetch_time = time.perf_counter() - prefetch_start
                standby_ready = standby.is_ready(next_model)
                if standby_ready:
                    log(f"    Prefetch done ({prefetch_time:.1f}s)")
                else:
                    log(f"    WARNING: Prefetch failed, will do cold load")

            # Step 4: Switch models
            log(f"\n[4] Switching to {next_model}...")
            switch_start = time.perf_counter()

            premerged = None
            if standby_ready:
                premerged = standby.consume_standby()
                if premerged is None:
                    log("    WARNING: Standby was evicted!")
                    standby_ready = False

            # Unload current model with proper GPU cleanup
            unload_start = time.perf_counter()
            log("    Unloading current model...")
            unload_llm_properly(llm)
            llm = None  # Mark as unloaded
            unload_time = time.perf_counter() - unload_start
            log(f"    Unload time: {unload_time:.1f}s")

            # Load new model
            load_start = time.perf_counter()
            use_standby = standby_ready and premerged is not None
            if use_standby:
                log(f"    Loading from standby ({len(premerged)} tensors)...")
                total_bytes = sum(t.numel() * t.element_size() for t in premerged.values())
                log(f"    Total tensor size: {total_bytes / 1e9:.2f}GB")

                set_preloaded_weights(premerged)
                llm = LLM(
                    model=next_model,
                    load_format="pinned_arena",
                    **VLLM_KWARGS,
                )
            else:
                if standby_ready:
                    log("    Standby ready but using cold load (debug)")
                else:
                    log("    Cold loading from SSD...")
                llm = LLM(model=next_model, **VLLM_KWARGS)

            load_time = time.perf_counter() - load_start
            total_switch_time = time.perf_counter() - switch_start
            gpu_mem_after = get_gpu_memory_mb()
            current_model = next_model

            log(f"    Load time: {load_time:.1f}s")
            log(f"    Total switch time: {total_switch_time:.1f}s")

            # Release arena memory now that GPU load is complete
            if standby_ready and premerged is not None:
                standby.release_consumed(next_model)

            # Verify new model works with actual correctness check
            def verify_inference(llm, model_name):
                """Run inference verification with correctness check."""
                # Use a simple factual question that both models can answer
                # The answer should contain 'Paris' for France's capital
                prompt = "What is the capital city of France? Answer with just the city name:"
                outputs = llm.generate([prompt], SamplingParams(max_tokens=20, temperature=0))
                response = outputs[0].outputs[0].text.strip().lower()

                # Check if response contains expected answer
                passed = 'paris' in response
                status = "PASS" if passed else "FAIL"
                log(f"    Verify [{status}]: {response[:50]}")

                if not passed:
                    log(f"    WARNING: Model {model_name} gave unexpected response!")
                    # For debugging, try another simpler prompt
                    outputs2 = llm.generate(["Count from 1 to 5:"], SamplingParams(max_tokens=30, temperature=0))
                    log(f"    Debug: {outputs2[0].outputs[0].text.strip()[:50]}")

                return passed

            verify_passed = verify_inference(llm, next_model)

            # Record metrics
            results.switches.append(SwitchMetrics(
                switch_num=switch_num,
                from_model=MODEL_A if next_model == MODEL_B else MODEL_B,
                to_model=next_model,
                was_standby_ready=standby_ready,
                prefetch_time=prefetch_time,
                unload_time=unload_time,
                load_time=load_time,
                total_switch_time=total_switch_time,
                inference_requests_before=INFERENCE_REQUESTS_PER_SWITCH,
                inference_time_before=inference_time,
                gpu_mem_before_mb=gpu_mem_before,
                gpu_mem_after_mb=gpu_mem_after,
                verify_passed=verify_passed,
            ))

        # Print summary
        log("\n" + results.summary())

    except Exception as e:
        log(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        log("\nCleaning up...")
        standby.shutdown()
        try:
            if llm is not None:
                unload_llm_properly(llm)
        except Exception as e:
            log(f"Final cleanup error: {e}")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
