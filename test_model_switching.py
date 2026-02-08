#!/usr/bin/env python3
"""Comprehensive model switching test with 16GB chunks.

Tests multiple switches between Qwen-32B and gpt-oss-120b, verifying
lucid output and measuring performance.
"""

import os
import gc
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_mem():
    with open('/proc/meminfo', 'r') as f:
        mem = {}
        for line in f:
            parts = line.split(':')
            if len(parts) == 2:
                mem[parts[0].strip()] = int(parts[1].strip().split()[0]) / 1024 / 1024
    return mem


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def verify_output(model_name, prompt, output):
    """Check if output is lucid (not garbage)."""
    # Basic checks
    if not output or len(output.strip()) < 2:
        return False, "Empty output"

    # Check for obvious garbage patterns
    garbage_patterns = ['!!!!', '????', '####', '.....' * 3, '\x00']
    for pattern in garbage_patterns:
        if pattern in output:
            return False, f"Contains garbage pattern: {pattern}"

    # Model-specific checks
    if 'gpt-oss' in model_name.lower():
        # GPT-OSS should give reasonable text
        if output.count('!') > len(output) / 4:
            return False, "Too many exclamation marks"

    return True, "OK"


def main():
    from blitzinfer.orchestrator.standby_manager import StandbyManager, StandbyState
    from blitzinfer.memory import set_preloaded_weights
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers import rotary_embedding

    def clear_rope():
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rotary_embedding._ROPE_DICT.clear()

    MODEL_A = "Qwen/Qwen3-VL-32B-Thinking-FP8"  # ~33GB
    MODEL_B = "openai/gpt-oss-120b"              # ~65GB

    TEST_PROMPTS = [
        ("What is 2+2?", "math"),
        ("The capital of France is", "geography"),
        ("Write a haiku about coding:", "creative"),
    ]

    print("=" * 70)
    print("MODEL SWITCHING TEST (16GB Chunks)")
    print("=" * 70)
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print(f"Arena: 80GB (5x 16GB pinned chunks)")
    print()

    mem = get_mem()
    log(f"START: Shmem={mem['Shmem']:.1f}GB, Avail={mem['MemAvailable']:.1f}GB")

    # Initialize StandbyManager with 80GB = 5x 16GB
    log("Initializing StandbyManager (80GB, 16GB chunks)...")
    standby = StandbyManager(
        arena_size_gb=80.0,
        chunk_size_gb=16.0,
        pin_memory=True,
    )

    results = []
    current_model = None
    llm = None
    cold_a = switch_ab = switch_ba = switch_ab2 = switch_ba2 = 0

    try:
        # === PHASE 1: Cold load Model A ===
        log(f"\n=== PHASE 1: Cold load {MODEL_A} ===")
        t0 = time.perf_counter()
        llm = LLM(
            model=MODEL_A,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=True,
        )
        cold_a = time.perf_counter() - t0
        current_model = MODEL_A
        log(f"Cold load: {cold_a:.1f}s")

        # Test output
        for prompt, category in TEST_PROMPTS[:1]:
            out = llm.generate([prompt], SamplingParams(max_tokens=30))
            text = out[0].outputs[0].text.strip()[:80]
            ok, reason = verify_output(MODEL_A, prompt, text)
            log(f"  {category}: {text[:50]}... [{reason}]")
            results.append(('A-cold', category, ok))

        # === PHASE 2: Prefetch Model B, switch A→B ===
        log(f"\n=== PHASE 2: Prefetch {MODEL_B}, switch A→B ===")
        standby.start_prefetch(MODEL_B)

        # Serve while prefetching
        log("Serving Model A while prefetching B...")
        for i in range(3):
            time.sleep(2)
            state = standby.get_state()
            if state == StandbyState.READY:
                break
            log(f"  Prefetch state: {state.name}")

        standby.wait_for_load(timeout=180)
        mem = get_mem()
        log(f"Prefetch done: Shmem={mem['Shmem']:.1f}GB")

        # Switch to Model B
        premerged = standby.consume_standby()
        log(f"Got {len(premerged)} premerged tensors")

        llm.llm_engine.engine_core.shutdown()
        del llm
        llm = None
        gc.collect()
        torch.cuda.empty_cache()
        clear_rope()
        time.sleep(1)  # Let GPU memory settle

        gpu_free = torch.cuda.mem_get_info()[0] / 1024**3
        log(f"After cleanup: {gpu_free:.1f}GB GPU free")

        t0 = time.perf_counter()
        set_preloaded_weights(premerged)
        llm = LLM(
            model=MODEL_B,
            load_format="pinned_arena",
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.55,  # Reduced due to incomplete GPU cleanup
            enforce_eager=True,
            trust_remote_code=True,
        )
        switch_ab = time.perf_counter() - t0
        current_model = MODEL_B
        log(f"Switch A→B: {switch_ab:.1f}s")

        # Test output
        for prompt, category in TEST_PROMPTS:
            out = llm.generate([prompt], SamplingParams(max_tokens=30))
            text = out[0].outputs[0].text.strip()[:80]
            ok, reason = verify_output(MODEL_B, prompt, text)
            log(f"  {category}: {text[:50]}... [{reason}]")
            results.append(('B-warm', category, ok))

        # === PHASE 3: Prefetch Model A, switch B→A ===
        log(f"\n=== PHASE 3: Prefetch {MODEL_A}, switch B→A ===")
        standby.start_prefetch(MODEL_A)
        standby.wait_for_load(timeout=180)

        premerged = standby.consume_standby()
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        clear_rope()

        t0 = time.perf_counter()
        set_preloaded_weights(premerged)
        llm = LLM(
            model=MODEL_A,
            load_format="pinned_arena",
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=True,
        )
        switch_ba = time.perf_counter() - t0
        current_model = MODEL_A
        log(f"Switch B→A: {switch_ba:.1f}s")

        # Test output
        for prompt, category in TEST_PROMPTS:
            out = llm.generate([prompt], SamplingParams(max_tokens=30))
            text = out[0].outputs[0].text.strip()[:80]
            ok, reason = verify_output(MODEL_A, prompt, text)
            log(f"  {category}: {text[:50]}... [{reason}]")
            results.append(('A-warm', category, ok))

        # === PHASE 4: One more round trip ===
        log(f"\n=== PHASE 4: Second round trip ===")

        # A→B
        standby.start_prefetch(MODEL_B)
        standby.wait_for_load(timeout=180)
        premerged = standby.consume_standby()
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        clear_rope()

        t0 = time.perf_counter()
        set_preloaded_weights(premerged)
        llm = LLM(
            model=MODEL_B,
            load_format="pinned_arena",
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            trust_remote_code=True,
        )
        switch_ab2 = time.perf_counter() - t0
        log(f"Switch A→B (2nd): {switch_ab2:.1f}s")

        out = llm.generate(["Hello, how are you?"], SamplingParams(max_tokens=30))
        text = out[0].outputs[0].text.strip()[:80]
        ok, _ = verify_output(MODEL_B, "greeting", text)
        log(f"  greeting: {text[:50]}...")
        results.append(('B-warm2', 'greeting', ok))

        # B→A
        standby.start_prefetch(MODEL_A)
        standby.wait_for_load(timeout=180)
        premerged = standby.consume_standby()
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        clear_rope()

        t0 = time.perf_counter()
        set_preloaded_weights(premerged)
        llm = LLM(
            model=MODEL_A,
            load_format="pinned_arena",
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=True,
        )
        switch_ba2 = time.perf_counter() - t0
        log(f"Switch B→A (2nd): {switch_ba2:.1f}s")

        out = llm.generate(["Count from 1 to 5:"], SamplingParams(max_tokens=30))
        text = out[0].outputs[0].text.strip()[:80]
        ok, _ = verify_output(MODEL_A, "counting", text)
        log(f"  counting: {text[:50]}...")
        results.append(('A-warm2', 'counting', ok))

    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        log("\n=== CLEANUP ===")
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except:
                pass
            del llm
        gc.collect()
        torch.cuda.empty_cache()
        standby.shutdown()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTiming:")
    print(f"  Cold load (Model A):    {cold_a:.1f}s")
    print(f"  Warm switch A→B:        {switch_ab:.1f}s")
    print(f"  Warm switch B→A:        {switch_ba:.1f}s")
    print(f"  Warm switch A→B (2nd):  {switch_ab2:.1f}s")
    print(f"  Warm switch B→A (2nd):  {switch_ba2:.1f}s")
    print(f"  Average warm switch:    {(switch_ab + switch_ba + switch_ab2 + switch_ba2)/4:.1f}s")
    print(f"  Speedup vs cold:        {cold_a / ((switch_ab + switch_ba + switch_ab2 + switch_ba2)/4):.1f}x")

    print(f"\nOutput verification:")
    passed = sum(1 for _, _, ok in results if ok)
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    for phase, category, ok in results:
        status = "✓" if ok else "✗"
        print(f"    {status} {phase}: {category}")

    mem = get_mem()
    print(f"\nFinal memory:")
    print(f"  Shmem: {mem['Shmem']:.1f}GB")
    print(f"  Available: {mem['MemAvailable']:.1f}GB")

    print("\n" + "=" * 70)
    if passed == total:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"SOME TESTS FAILED ({total - passed}/{total})")
    print("=" * 70)


if __name__ == "__main__":
    main()
