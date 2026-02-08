#!/usr/bin/env python3
"""Test multiple switches with same model (Qwen) to verify arena works.

This avoids the GPU memory leak issue by using same architecture.
"""

import os
import gc
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_mem():
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('Shmem:'):
                return int(line.split()[1]) / 1024 / 1024
    return 0


def main():
    from blitzinfer.orchestrator.standby_manager import StandbyManager, StandbyState
    from blitzinfer.memory import set_preloaded_weights
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers import rotary_embedding

    def clear_rope():
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rotary_embedding._ROPE_DICT.clear()

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    print("=" * 70)
    print("MULTI-SWITCH TEST (Same Model, 16GB Chunks)")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Arena: 48GB (3x 16GB pinned chunks)")
    print()

    log(f"START: Shmem={get_mem():.1f}GB")

    # Initialize standby with 48GB = 3x 16GB
    standby = StandbyManager(
        arena_size_gb=48.0,
        chunk_size_gb=16.0,
        pin_memory=True,
    )

    times = []
    llm = None

    try:
        # Cold load
        log("=== Cold Load ===")
        t0 = time.perf_counter()
        llm = LLM(
            model=MODEL,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=True,
        )
        cold = time.perf_counter() - t0
        log(f"Cold load: {cold:.1f}s")

        out = llm.generate(["2+2="], SamplingParams(max_tokens=20))
        log(f"Output: {out[0].outputs[0].text.strip()[:50]}")

        # 4 warm switches
        for i in range(4):
            log(f"\n=== Switch {i+1}/4 ===")

            # Prefetch
            standby.start_prefetch(MODEL)
            standby.wait_for_load(timeout=120)
            log(f"Prefetch done: Shmem={get_mem():.1f}GB")

            # Get tensors and unload
            premerged = standby.consume_standby()
            log(f"Got {len(premerged)} tensors")

            llm.llm_engine.engine_core.shutdown()
            del llm
            llm = None
            gc.collect()
            torch.cuda.empty_cache()
            clear_rope()
            time.sleep(0.5)

            # Load from standby
            t0 = time.perf_counter()
            set_preloaded_weights(premerged)
            llm = LLM(
                model=MODEL,
                load_format="pinned_arena",
                dtype="bfloat16",
                max_model_len=4096,
                gpu_memory_utilization=0.50,
                enforce_eager=True,
                trust_remote_code=True,
            )
            switch_time = time.perf_counter() - t0
            times.append(switch_time)
            log(f"Switch: {switch_time:.1f}s")

            # Verify output
            prompts = [
                f"{i+1}+{i+1}=",
                "The sky is",
                "Hello, my name is",
            ]
            out = llm.generate([prompts[i % 3]], SamplingParams(max_tokens=20))
            text = out[0].outputs[0].text.strip()[:50]
            log(f"Output: {text}")

            # Basic sanity check
            if '!!!!' in text or len(text) < 2:
                log("WARNING: Suspicious output!")

    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if llm:
            try:
                llm.llm_engine.engine_core.shutdown()
            except:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        standby.shutdown()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Cold load:         {cold:.1f}s")
    if times:
        avg = sum(times) / len(times)
        print(f"Warm switches:     {', '.join(f'{t:.1f}s' for t in times)}")
        print(f"Average warm:      {avg:.1f}s")
        print(f"Speedup:           {cold/avg:.1f}x")
        print(f"All {len(times)} switches: {'PASSED' if len(times) == 4 else 'INCOMPLETE'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
