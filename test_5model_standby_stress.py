#!/usr/bin/env python3
"""5-Model Standby Stress Test: Fast switching with pre-loading.

Uses StandbyManager for fast model switching:
- Background pre-load into pinned CPU RAM (hidden during inference)
- Fast switchover via pinned arena loader (user-facing)

Models:
1. Qwen/Qwen3-VL-32B-Thinking-FP8 (VL, FP8, ~35GB)
2. Qwen/Qwen3-VL-32B-Instruct (VL, bfloat16, ~65GB)
3. openai/gpt-oss-120b (LLM, MXFP4, MoE, ~65GB)
4. moonshotai/Kimi-VL-A3B-Thinking-2506 (VL, MoE, ~8GB)
5. mistralai/Mistral-Small-3.2-24B-Instruct-2506 (LLM, ~48GB)

Timing breakdown:
- Pre-load time: Time to load weights into pinned RAM (background)
- Switchover time: Cleanup + GPU injection (user-facing)
"""

import os
import gc
import sys
import time
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch

# Model configurations - must fit in 80GB arena
MODELS = {
    "qwen-32b-fp8": {
        "name": "Qwen/Qwen3-VL-32B-Thinking-FP8",
        "max_model_len": 131072,
        "gpu_memory_utilization": 0.95,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "size_gb": 35,  # Approximate
    },
    "gpt-oss-120b": {
        "name": "openai/gpt-oss-120b",
        "max_model_len": 131072,
        "gpu_memory_utilization": 0.90,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "size_gb": 65,
    },
    "kimi-vl": {
        "name": "moonshotai/Kimi-VL-A3B-Thinking-2506",
        "max_model_len": 131072,
        "gpu_memory_utilization": 0.95,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "size_gb": 8,
    },
    "mistral-24b": {
        "name": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "max_model_len": 131072,
        "gpu_memory_utilization": 0.95,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "size_gb": 48,
    },
}

# Simple questions to verify lucidity
LUCIDITY_QUESTIONS = [
    "What is 2 + 2? Answer with just the number.",
    "What color is the sky on a clear day? One word answer.",
    "What is the capital of France? One word answer.",
    "How many legs does a cat have? Answer with just the number.",
    "What planet do we live on? One word answer.",
]


@dataclass
class SwitchResult:
    """Results for a single model switch."""
    switch_num: int
    from_model: str
    to_model: str
    # Timing breakdown
    preload_time: float  # Time to load into pinned RAM (background)
    cleanup_time: float  # Time to cleanup previous model
    inject_time: float   # Time to inject from pinned to GPU
    vllm_init_time: float  # Time for vLLM initialization overhead
    total_switchover: float  # cleanup + inject + vllm_init (user-facing)
    # Status
    was_preloaded: bool
    lucid: bool
    verify_answer: str
    vram_after_gb: float


@dataclass
class TestResults:
    """Aggregate test results."""
    switches: List[SwitchResult] = field(default_factory=list)
    cold_load_time: float = 0.0
    arena_alloc_time: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 80,
            "5-MODEL STANDBY STRESS TEST RESULTS",
            "=" * 80,
            "",
            f"Arena allocation: {self.arena_alloc_time:.1f}s",
            f"Initial cold load: {self.cold_load_time:.1f}s",
            "",
            "TIMING BREAKDOWN:",
            "-" * 80,
            f"{'#':>3} {'From':<15} {'To':<15} {'Preload':>8} {'Cleanup':>8} {'Inject':>8} {'vLLM':>8} {'Switch':>8} {'Status'}",
            "-" * 80,
        ]

        preloaded_switches = [s for s in self.switches if s.was_preloaded]
        cold_switches = [s for s in self.switches if not s.was_preloaded]

        for s in self.switches:
            status = "PASS" if s.lucid else "FAIL"
            preload_str = f"{s.preload_time:.1f}s" if s.was_preloaded else "cold"
            lines.append(
                f"{s.switch_num:3d} {s.from_model:<15} {s.to_model:<15} "
                f"{preload_str:>8} {s.cleanup_time:>7.2f}s {s.inject_time:>7.2f}s "
                f"{s.vllm_init_time:>7.2f}s {s.total_switchover:>7.2f}s [{status}]"
            )

        lines.append("-" * 80)
        lines.append("")
        lines.append("SUMMARY:")

        if preloaded_switches:
            avg_preload = sum(s.preload_time for s in preloaded_switches) / len(preloaded_switches)
            avg_cleanup = sum(s.cleanup_time for s in preloaded_switches) / len(preloaded_switches)
            avg_inject = sum(s.inject_time for s in preloaded_switches) / len(preloaded_switches)
            avg_vllm = sum(s.vllm_init_time for s in preloaded_switches) / len(preloaded_switches)
            avg_switch = sum(s.total_switchover for s in preloaded_switches) / len(preloaded_switches)
            lines.extend([
                f"  Preloaded switches: {len(preloaded_switches)}",
                f"    Avg preload (background): {avg_preload:.1f}s",
                f"    Avg cleanup:              {avg_cleanup:.2f}s",
                f"    Avg inject:               {avg_inject:.2f}s",
                f"    Avg vLLM init:            {avg_vllm:.2f}s",
                f"    Avg switchover (user):    {avg_switch:.1f}s",
            ])

        if cold_switches:
            avg_cold = sum(s.total_switchover for s in cold_switches) / len(cold_switches)
            lines.append(f"  Cold switches: {len(cold_switches)}, avg: {avg_cold:.1f}s")

        if preloaded_switches and cold_switches:
            speedup = avg_cold / avg_switch if avg_switch > 0 else 0
            lines.append(f"\n  Speedup from preloading: {speedup:.1f}x")

        passed = sum(1 for s in self.switches if s.lucid)
        lines.extend([
            "",
            f"Lucidity: {passed}/{len(self.switches)} passed",
            "=" * 80,
        ])

        return "\n".join(lines)


def log(msg):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_memory():
    """Get GPU memory info."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        'used_gb': (total - free) / 1024**3,
        'free_gb': free / 1024**3,
    }


def test_lucidity(llm, model_key):
    """Test if model is lucid with a simple question."""
    from vllm import SamplingParams

    question = random.choice(LUCIDITY_QUESTIONS)
    log(f"  Lucidity test: '{question}'")

    try:
        out = llm.generate([question], SamplingParams(max_tokens=20, temperature=0.1))
        answer = out[0].outputs[0].text.strip()
        log(f"  Answer: '{answer}'")

        if len(answer) > 0 and len(answer) < 100:
            return True, answer
        else:
            return False, answer
    except Exception as e:
        log(f"  FAILED lucidity test: {e}")
        return False, str(e)


def cleanup_model(llm):
    """Clean up model and free memory using full_cleanup."""
    from blitzinfer.engine.cleanup import full_cleanup

    start = time.time()
    freed = full_cleanup(llm, nuclear=True)
    cleanup_time = time.time() - start

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    mem = get_memory()
    log(f"  Cleanup: freed {freed:.1f}GB in {cleanup_time:.2f}s, VRAM now: {mem['used_gb']:.2f}GB")
    return cleanup_time


def select_next_model(current_model, load_counts):
    """Select next model (not current, prefer least loaded)."""
    available = [k for k in MODELS.keys() if k != current_model]
    min_count = min(load_counts.get(k, 0) for k in available)
    candidates = [k for k in available if load_counts.get(k, 0) == min_count]
    return random.choice(candidates)


def main():
    from vllm import LLM, SamplingParams
    from blitzinfer.orchestrator.standby_manager import StandbyManager, StandbyState
    from blitzinfer.memory import set_preloaded_weights

    NUM_SWITCHES = 8  # At least 2 loads per model
    ARENA_SIZE_GB = 80.0
    CHUNK_SIZE_GB = 16.0  # Power of 2 for 0% overhead

    log("=" * 80)
    log("5-MODEL STANDBY STRESS TEST")
    log("=" * 80)
    log(f"Models: {list(MODELS.keys())}")
    log(f"Switches: {NUM_SWITCHES}")
    log(f"Arena: {ARENA_SIZE_GB}GB ({int(ARENA_SIZE_GB/CHUNK_SIZE_GB)}x{CHUNK_SIZE_GB}GB chunks)")
    log("")

    results = TestResults()

    # Initialize standby manager with static arena
    log("Allocating pinned memory arena...")
    arena_start = time.time()
    standby = StandbyManager(
        arena_size_gb=ARENA_SIZE_GB,
        chunk_size_gb=CHUNK_SIZE_GB,
        pin_memory=True,
        lazy_arena=False,  # Pre-allocate now
    )
    results.arena_alloc_time = time.time() - arena_start
    log(f"Arena allocated in {results.arena_alloc_time:.1f}s")

    load_counts = {k: 0 for k in MODELS}
    current_model = None
    llm = None

    gc.collect()
    torch.cuda.empty_cache()
    baseline = get_memory()
    log(f"Baseline VRAM: {baseline['used_gb']:.2f}GB")

    try:
        # Cold load first model
        first_model = random.choice(list(MODELS.keys()))
        config = MODELS[first_model]
        log(f"\nCold loading first model: {first_model}")

        cold_start = time.time()
        llm = LLM(
            model=config['name'],
            dtype=config['dtype'],
            max_model_len=config['max_model_len'],
            gpu_memory_utilization=config['gpu_memory_utilization'],
            enforce_eager=True,
            trust_remote_code=config['trust_remote_code'],
        )
        results.cold_load_time = time.time() - cold_start
        log(f"Cold load: {results.cold_load_time:.1f}s")

        load_counts[first_model] = 1
        current_model = first_model

        # Verify first model
        lucid, answer = test_lucidity(llm, current_model)
        if not lucid:
            log("WARNING: First model not lucid!")

        # Main switching loop with preloading
        for i in range(NUM_SWITCHES):
            log(f"\n{'='*80}")
            log(f"SWITCH {i+1}/{NUM_SWITCHES}")
            log("=" * 80)

            # Select next model
            next_model = select_next_model(current_model, load_counts)
            next_config = MODELS[next_model]
            log(f"Switching: {current_model} -> {next_model}")

            # Start preloading next model (background)
            log(f"Starting background preload of {next_model}...")
            preload_start = time.time()
            standby.start_prefetch(next_config['name'])

            # Simulate inference while preloading
            log("Running inference while preloading...")
            for j in range(3):
                out = llm.generate(
                    [f"Briefly explain concept {j+1}: machine learning."],
                    SamplingParams(max_tokens=50, temperature=0.7)
                )
                log(f"  Inference {j+1}: {out[0].outputs[0].text.strip()[:40]}...")

            # Wait for preload to complete
            was_preloaded = standby.is_ready(next_config['name'])
            if not was_preloaded:
                log("Waiting for preload to complete...")
                standby.wait_for_load(timeout=120)
                was_preloaded = standby.is_ready(next_config['name'])

            preload_time = time.time() - preload_start
            log(f"Preload {'completed' if was_preloaded else 'FAILED'} in {preload_time:.1f}s")

            # Get premerged weights before cleanup
            premerged = None
            if was_preloaded:
                premerged = standby.consume_standby()
                if premerged:
                    total_bytes = sum(t.numel() * t.element_size() for t in premerged.values())
                    log(f"Got {len(premerged)} premerged tensors ({total_bytes/1e9:.1f}GB)")

            # === SWITCHOVER STARTS HERE (user-facing time) ===
            switchover_start = time.time()

            # Cleanup current model
            cleanup_start = time.time()
            cleanup_model(llm)
            llm = None
            cleanup_time = time.time() - cleanup_start

            # Load new model
            if premerged:
                # Fast path: inject from pinned memory
                log(f"Injecting {next_model} from pinned memory...")
                inject_start = time.time()
                set_preloaded_weights(premerged)

                vllm_start = time.time()
                llm = LLM(
                    model=next_config['name'],
                    dtype=next_config['dtype'],
                    max_model_len=next_config['max_model_len'],
                    gpu_memory_utilization=next_config['gpu_memory_utilization'],
                    enforce_eager=True,
                    trust_remote_code=next_config['trust_remote_code'],
                    load_format="pinned_arena",
                )
                vllm_init_time = time.time() - vllm_start
                inject_time = vllm_init_time  # Injection happens during vLLM init

                # Release arena memory
                standby.release_consumed(next_config['name'])
            else:
                # Cold path: load from disk
                log(f"Cold loading {next_model}...")
                inject_start = time.time()
                llm = LLM(
                    model=next_config['name'],
                    dtype=next_config['dtype'],
                    max_model_len=next_config['max_model_len'],
                    gpu_memory_utilization=next_config['gpu_memory_utilization'],
                    enforce_eager=True,
                    trust_remote_code=next_config['trust_remote_code'],
                )
                inject_time = time.time() - inject_start
                vllm_init_time = inject_time  # All time is vLLM init for cold load

            total_switchover = time.time() - switchover_start
            # === SWITCHOVER ENDS HERE ===

            load_counts[next_model] += 1
            current_model = next_model

            # Verify model
            lucid, answer = test_lucidity(llm, next_model)
            mem = get_memory()

            # Record results
            results.switches.append(SwitchResult(
                switch_num=i + 1,
                from_model=current_model if i == 0 else results.switches[-1].to_model if results.switches else first_model,
                to_model=next_model,
                preload_time=preload_time if was_preloaded else 0,
                cleanup_time=cleanup_time,
                inject_time=inject_time,
                vllm_init_time=vllm_init_time,
                total_switchover=total_switchover,
                was_preloaded=was_preloaded,
                lucid=lucid,
                verify_answer=answer,
                vram_after_gb=mem['used_gb'],
            ))

            status = "PASS" if lucid else "FAIL"
            log(f"Result: [{status}] Switchover: {total_switchover:.1f}s (cleanup: {cleanup_time:.2f}s, inject: {inject_time:.2f}s)")

        # Print summary
        log("\n" + results.summary())

        # Final counts
        log(f"\nLoad counts: {load_counts}")

    except Exception as e:
        log(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        log("\nCleaning up...")
        standby.shutdown()
        if llm is not None:
            try:
                cleanup_model(llm)
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()

    passed = sum(1 for s in results.switches if s.lucid)
    if passed == len(results.switches):
        log("\n*** ALL TESTS PASSED ***")
        return 0
    else:
        log("\n*** SOME TESTS FAILED ***")
        return 1


if __name__ == "__main__":
    sys.exit(main())
