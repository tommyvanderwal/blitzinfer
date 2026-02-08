#!/usr/bin/env python3
"""Test 1 active + 1 standby model switching.

Tests demand-driven prefetch: prefetch starts when request arrives for different model.

This test demonstrates the standby switching workflow:
1. Cold load Model A (standard vLLM load from SSD)
2. Request for Model B arrives -> triggers background prefetch
3. Continue serving Model A while B prefetches in background
4. Fast switch to Model B from standby (pinned RAM -> GPU)
5. Request for Model A arrives -> triggers background prefetch
6. Fast switch back to Model A from standby

Expected performance on RTX PRO 6000:
- Cold load (SSD): ~18-30s depending on model size
- Standby switch: ~5-8s (PCIe transfer + vLLM init)
- Speedup: 3-4x faster switching when model is in standby
"""

import os
import sys
import time
import gc
import logging

# Configure environment before importing vLLM
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    import torch
    from vllm import LLM, SamplingParams

    from blitzinfer.orchestrator.standby_manager import StandbyManager, StandbyState
    from blitzinfer.memory import set_preloaded_weights

    # Configuration - adjust these for your setup
    # For RTX PRO 6000 testing:
    MODEL_A = "Qwen/Qwen3-VL-32B-Thinking-FP8"  # ~33GB
    MODEL_B = "openai/gpt-oss-120b"              # ~60GB (largest model)

    # For AMD iGPU testing (smaller models):
    # MODEL_A = "Qwen/Qwen2.5-7B-Instruct"
    # MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"

    # Standby arena size (must fit largest model)
    STANDBY_ARENA_GB = 70.0

    # vLLM settings (adjust for your hardware)
    VLLM_KWARGS = {
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.45,
        "enforce_eager": True,
        "trust_remote_code": True,
    }

    print("=" * 70)
    print("1 ACTIVE + 1 STANDBY MODEL SWITCHING TEST")
    print("=" * 70)
    print(f"\nModel A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print(f"Standby Arena: {STANDBY_ARENA_GB}GB")
    print()

    # Track metrics
    metrics = {
        'cold_load_a': 0.0,
        'cold_load_b': 0.0,
        'standby_switch_a_to_b': 0.0,
        'standby_switch_b_to_a': 0.0,
        'prefetch_time_b': 0.0,
        'prefetch_time_a': 0.0,
    }

    # Initialize standby manager
    standby = StandbyManager(arena_size_gb=STANDBY_ARENA_GB)

    try:
        # ==========================================
        # PHASE 1: Cold load Model A
        # ==========================================
        print("\n" + "-" * 50)
        print("[Phase 1] Cold load Model A...")
        print("-" * 50)

        t0 = time.perf_counter()
        llm = LLM(model=MODEL_A, **VLLM_KWARGS)
        cold_load_time = time.perf_counter() - t0
        metrics['cold_load_a'] = cold_load_time
        print(f"Cold load Model A: {cold_load_time:.1f}s")

        # Verify model works
        sampling_params = SamplingParams(max_tokens=20, temperature=0.7)
        outputs = llm.generate(["What is 2+2? Answer briefly:"], sampling_params)
        print(f"Model A test output: {outputs[0].outputs[0].text.strip()}")

        # ==========================================
        # PHASE 2: Request for Model B -> triggers prefetch
        # ==========================================
        print("\n" + "-" * 50)
        print("[Phase 2] Request for Model B arrives -> triggers prefetch")
        print("-" * 50)

        prefetch_start = time.perf_counter()
        started = standby.start_prefetch(MODEL_B)
        print(f"Prefetch started: {started}")

        # Continue serving Model A while prefetch happens
        print("\nServing Model A while Model B prefetches in background...")
        for i in range(3):
            prompt = f"Count from 1 to {i+3}. Just list the numbers:"
            outputs = llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            print(f"  Request {i+1}: {response[:60]}...")

            # Check prefetch status
            state = standby.get_state()
            if state == StandbyState.READY:
                prefetch_time = time.perf_counter() - prefetch_start
                metrics['prefetch_time_b'] = prefetch_time
                print(f"  [Prefetch complete: {prefetch_time:.1f}s]")
            else:
                print(f"  [Prefetch status: {state.name}]")

            time.sleep(1)  # Simulate request processing time

        # Wait for prefetch to complete if not done
        if standby.get_state() != StandbyState.READY:
            print("\nWaiting for prefetch to complete...")
            standby.wait_for_load(timeout=180)  # GPT-OSS-120B is large
            prefetch_time = time.perf_counter() - prefetch_start
            metrics['prefetch_time_b'] = prefetch_time

        print(f"\nStandby state: {standby.get_state().name}")
        assert standby.is_ready(MODEL_B), "Model B should be ready in standby"

        # ==========================================
        # PHASE 3: Fast switch to Model B (from standby)
        # ==========================================
        print("\n" + "-" * 50)
        print("[Phase 3] Fast switch to Model B (from standby)")
        print("-" * 50)

        # Get premerged tensors from standby
        premerged = standby.consume_standby()
        assert premerged is not None, "Should have premerged tensors"
        print(f"Got {len(premerged)} premerged tensors from standby")

        # Calculate tensor size
        total_bytes = sum(t.numel() * t.element_size() for t in premerged.values())
        print(f"Total tensor size: {total_bytes / 1e9:.2f}GB")

        # Unload Model A
        print("Unloading Model A...")
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        # Load Model B from standby (fast path)
        print("Loading Model B from standby...")
        t0 = time.perf_counter()
        set_preloaded_weights(premerged)
        llm = LLM(
            model=MODEL_B,
            load_format="pinned_arena",
            **VLLM_KWARGS,
        )
        standby_switch_time = time.perf_counter() - t0
        metrics['standby_switch_a_to_b'] = standby_switch_time
        print(f"Standby switch A->B: {standby_switch_time:.1f}s")

        # Verify Model B works
        outputs = llm.generate(["What is 3+3? Answer briefly:"], sampling_params)
        print(f"Model B test output: {outputs[0].outputs[0].text.strip()}")

        # ==========================================
        # PHASE 4: Request for Model A -> triggers prefetch
        # ==========================================
        print("\n" + "-" * 50)
        print("[Phase 4] Request for Model A arrives -> triggers prefetch")
        print("-" * 50)

        prefetch_start = time.perf_counter()
        started = standby.start_prefetch(MODEL_A)
        print(f"Prefetch started: {started}")

        # Serve a few requests while prefetching
        print("\nServing Model B while Model A prefetches...")
        for i in range(2):
            prompt = f"What is {(i+1)*10}+{(i+1)*5}? Just the number:"
            outputs = llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            print(f"  Request {i+1}: {response[:40]}...")

            state = standby.get_state()
            if state == StandbyState.READY:
                prefetch_time = time.perf_counter() - prefetch_start
                metrics['prefetch_time_a'] = prefetch_time
                print(f"  [Prefetch complete: {prefetch_time:.1f}s]")
            else:
                print(f"  [Prefetch status: {state.name}]")

            time.sleep(1)

        # Wait for prefetch
        if standby.get_state() != StandbyState.READY:
            print("\nWaiting for prefetch to complete...")
            standby.wait_for_load(timeout=120)
            prefetch_time = time.perf_counter() - prefetch_start
            metrics['prefetch_time_a'] = prefetch_time

        assert standby.is_ready(MODEL_A), "Model A should be ready in standby"

        # ==========================================
        # PHASE 5: Fast switch back to Model A
        # ==========================================
        print("\n" + "-" * 50)
        print("[Phase 5] Fast switch back to Model A (from standby)")
        print("-" * 50)

        premerged = standby.consume_standby()
        assert premerged is not None, "Should have premerged tensors"
        print(f"Got {len(premerged)} premerged tensors from standby")

        # Unload Model B
        print("Unloading Model B...")
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

        # Load Model A from standby
        print("Loading Model A from standby...")
        t0 = time.perf_counter()
        set_preloaded_weights(premerged)
        llm = LLM(
            model=MODEL_A,
            load_format="pinned_arena",
            **VLLM_KWARGS,
        )
        switch_back_time = time.perf_counter() - t0
        metrics['standby_switch_b_to_a'] = switch_back_time
        print(f"Standby switch B->A: {switch_back_time:.1f}s")

        # Verify Model A works
        outputs = llm.generate(["What is 4+4? Answer briefly:"], sampling_params)
        print(f"Model A test output: {outputs[0].outputs[0].text.strip()}")

        # ==========================================
        # SUMMARY
        # ==========================================
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        avg_cold_load = metrics['cold_load_a']  # Only measured A's cold load
        avg_standby_switch = (
            metrics['standby_switch_a_to_b'] + metrics['standby_switch_b_to_a']
        ) / 2
        speedup = avg_cold_load / avg_standby_switch if avg_standby_switch > 0 else 0

        print(f"""
    Cold load Model A (SSD):      {metrics['cold_load_a']:.1f}s
    Prefetch Model B (background):{metrics['prefetch_time_b']:.1f}s
    Prefetch Model A (background):{metrics['prefetch_time_a']:.1f}s

    Standby switch A->B:          {metrics['standby_switch_a_to_b']:.1f}s
    Standby switch B->A:          {metrics['standby_switch_b_to_a']:.1f}s
    Average standby switch:       {avg_standby_switch:.1f}s

    Speedup from standby:         {speedup:.1f}x faster than cold load
        """)

        print("=" * 70)
        print("TEST PASSED")
        print("=" * 70)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        print("\nCleaning up...")
        standby.shutdown()
        try:
            if 'llm' in dir():
                llm.llm_engine.engine_core.shutdown()
                del llm
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
