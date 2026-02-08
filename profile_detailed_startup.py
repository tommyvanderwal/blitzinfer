#!/usr/bin/env python3
"""Detailed profiling of vLLM startup phases."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import sys
import types

# Patch torchvision._meta_registrations before import
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta

sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import time
import gc


def profile_phase(name):
    """Context manager for timing phases."""
    class Timer:
        def __init__(self, name):
            self.name = name
            self.start = None
            self.end = None

        def __enter__(self):
            self.start = time.perf_counter()
            print(f"[{time.strftime('%H:%M:%S')}] START: {self.name}")
            return self

        def __exit__(self, *args):
            self.end = time.perf_counter()
            elapsed = self.end - self.start
            print(f"[{time.strftime('%H:%M:%S')}] END:   {self.name} = {elapsed:.3f}s")
            return False
    return Timer(name)


def run_detailed_profile(run_number=1):
    """Profile startup with detailed timing."""
    print(f"\n{'='*70}")
    print(f"RUN {run_number}: Detailed Startup Profile")
    print(f"{'='*70}")

    timings = {}
    total_start = time.perf_counter()

    # Phase 1: Import torch
    with profile_phase("1. Import torch") as t:
        import torch
    timings['import_torch'] = t.end - t.start

    # Phase 2: Import vllm
    with profile_phase("2. Import vllm") as t:
        from vllm import LLM, SamplingParams
    timings['import_vllm'] = t.end - t.start

    # Phase 3: Clear GPU
    with profile_phase("3. Clear GPU cache") as t:
        torch.cuda.empty_cache()
        gc.collect()
    timings['clear_cache'] = t.end - t.start

    # Phase 4: LLM instantiation (this is where most time is spent)
    print(f"\n[{time.strftime('%H:%M:%S')}] START: 4. LLM instantiation (detailed)")
    llm_start = time.perf_counter()

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        max_model_len=1024,
        enforce_eager=True,
    )

    llm_end = time.perf_counter()
    timings['llm_init'] = llm_end - llm_start
    print(f"[{time.strftime('%H:%M:%S')}] END:   4. LLM instantiation = {timings['llm_init']:.3f}s")

    # Phase 5: First inference (is this the "real" warmup?)
    with profile_phase("5. First inference (real warmup?)") as t:
        out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    timings['first_inference'] = t.end - t.start
    print(f"    Output: {out[0].outputs[0].text}")

    # Phase 6: Second inference (should be hot)
    with profile_phase("6. Second inference (hot)") as t:
        out = llm.generate(["Hi"], SamplingParams(max_tokens=10))
    timings['second_inference'] = t.end - t.start
    print(f"    Output: {out[0].outputs[0].text}")

    # Phase 7: Third inference
    with profile_phase("7. Third inference") as t:
        out = llm.generate(["Test"], SamplingParams(max_tokens=10))
    timings['third_inference'] = t.end - t.start

    total_end = time.perf_counter()
    timings['total'] = total_end - total_start

    # Summary
    print(f"\n{'='*70}")
    print(f"RUN {run_number} TIMING SUMMARY")
    print(f"{'='*70}")
    for phase, elapsed in timings.items():
        print(f"  {phase:25s}: {elapsed:7.3f}s")
    print(f"{'='*70}")

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return timings


def profile_without_warmup():
    """Test if skipping internal warmup affects first request."""
    print(f"\n{'='*70}")
    print("TEST: Can warmup be replaced by first request?")
    print(f"{'='*70}")

    import torch
    from vllm import LLM, SamplingParams

    torch.cuda.empty_cache()
    gc.collect()

    # Check if there's a way to skip warmup
    # Looking at vLLM code: enforce_eager=True already skips cudagraph
    # The profile_run() in model_runner is for compilation, not warmup

    start = time.perf_counter()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        max_model_len=1024,
        enforce_eager=True,
        # Try to minimize warmup
        max_num_batched_tokens=1,  # Minimal batch
    )
    init_time = time.perf_counter() - start
    print(f"Init time: {init_time:.2f}s")

    # First request timing
    start = time.perf_counter()
    out = llm.generate(["Hello world"], SamplingParams(max_tokens=20))
    first_req = time.perf_counter() - start
    print(f"First request: {first_req:.2f}s")
    print(f"Output: {out[0].outputs[0].text}")

    # Second request
    start = time.perf_counter()
    out = llm.generate(["Test"], SamplingParams(max_tokens=20))
    second_req = time.perf_counter() - start
    print(f"Second request: {second_req:.2f}s")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def profile_engine_internals():
    """Hook into vLLM to profile internal phases."""
    print(f"\n{'='*70}")
    print("PROFILING ENGINE INTERNALS")
    print(f"{'='*70}")

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    # Monkey-patch to add timing
    from vllm.v1.worker import gpu_worker
    from vllm.v1.worker.gpu import model_runner
    from vllm.v1.engine import core

    original_init_device = gpu_worker.GPUWorker.init_device
    original_profile_run = None
    original_init_kv = None

    timing_log = []

    def timed_init_device(self):
        start = time.perf_counter()
        print(f"  [{time.strftime('%H:%M:%S')}] GPUWorker.init_device START")
        result = original_init_device(self)
        elapsed = time.perf_counter() - start
        print(f"  [{time.strftime('%H:%M:%S')}] GPUWorker.init_device END = {elapsed:.3f}s")
        timing_log.append(('init_device', elapsed))
        return result

    gpu_worker.GPUWorker.init_device = timed_init_device

    # Profile the model loading
    from vllm import LLM, SamplingParams

    total_start = time.perf_counter()
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        max_model_len=1024,
        enforce_eager=True,
    )
    total_time = time.perf_counter() - total_start

    print(f"\nTotal LLM init: {total_time:.2f}s")
    print(f"Timing log: {timing_log}")

    # First inference
    start = time.perf_counter()
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"First inference: {time.perf_counter() - start:.2f}s")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def compare_runs():
    """Compare first and second startup timing."""
    print("\n" + "="*70)
    print("COMPARING FIRST vs SECOND STARTUP")
    print("="*70)

    # Run 1 (cold)
    run1 = run_detailed_profile(1)

    # Brief pause
    time.sleep(2)

    # Run 2 (warm - OS cache, Python imports cached)
    run2 = run_detailed_profile(2)

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON: Run 1 (cold) vs Run 2 (warm)")
    print("="*70)
    print(f"{'Phase':<25} {'Run1':>10} {'Run2':>10} {'Diff':>10} {'%Saved':>10}")
    print("-"*70)
    for phase in run1:
        r1 = run1[phase]
        r2 = run2.get(phase, 0)
        diff = r1 - r2
        pct = (diff / r1 * 100) if r1 > 0 else 0
        print(f"{phase:<25} {r1:>10.3f} {r2:>10.3f} {diff:>+10.3f} {pct:>9.1f}%")
    print("="*70)


if __name__ == '__main__':
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "single":
            run_detailed_profile(1)
        elif cmd == "compare":
            compare_runs()
        elif cmd == "nowarmup":
            profile_without_warmup()
        elif cmd == "internals":
            profile_engine_internals()
    else:
        print("\nUsage: python3.12 profile_detailed_startup.py [single|compare|nowarmup|internals]")
        print("  single    - Single detailed profile run")
        print("  compare   - Compare first vs second startup")
        print("  nowarmup  - Test skipping warmup")
        print("  internals - Profile engine internals")
        print("\nRunning 'compare' by default...")
        compare_runs()
