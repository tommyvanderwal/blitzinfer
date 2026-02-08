#!/usr/bin/env python3
"""
BlitzInfer Fast Switch v3: Optimized cleanup

Key insight: gc.collect() IS required to free GPU memory.
But we can make it faster by:
1. Explicitly nulling internal references
2. Running targeted gc.collect() on specific generations
3. Reducing object graph complexity before gc
"""

import gc
import os
import sys
import time
import types
from typing import Dict, List, Any, Tuple

# ROCm setup
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# vLLM import workaround
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch


def get_mem() -> Tuple[float, float]:
    """Get (used, free) GPU memory in GB."""
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**3)
    return used, free / (1024**3)


def optimized_cleanup(llm) -> Tuple[float, float]:
    """
    Optimized cleanup that properly frees GPU memory.

    Returns: (cleanup_time_ms, gc_time_ms)
    """
    t0 = time.perf_counter()

    # Phase 1: Explicitly clear internal references to help gc
    try:
        if hasattr(llm, 'llm_engine'):
            engine = llm.llm_engine
            if hasattr(engine, 'model_executor'):
                executor = engine.model_executor
                if hasattr(executor, 'driver_worker'):
                    worker = executor.driver_worker
                    if hasattr(worker, 'worker'):
                        inner = worker.worker
                        if hasattr(inner, 'model_runner'):
                            runner = inner.model_runner
                            if hasattr(runner, 'model'):
                                runner.model = None
                            if hasattr(runner, 'kv_caches'):
                                runner.kv_caches = None
    except Exception:
        pass

    # Phase 2: Delete the main object
    del llm

    # Phase 3: Match orchestrator cleanup pattern
    t_gc = time.perf_counter()
    gc.collect()   # First collect
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()   # Second collect (matches orchestrator)
    gc_time = (time.perf_counter() - t_gc) * 1000

    total_time = (time.perf_counter() - t0) * 1000
    return total_time, gc_time


class BlitzFastSwitchV3:
    """
    Fast cross-architecture switcher with optimized cleanup.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.llm = None
        self.current_model = None

        self.default_config = {
            "dtype": "float16",
            "gpu_memory_utilization": 0.25,
            "max_model_len": 512,
            "max_num_batched_tokens": 512,
            "kv_cache_memory_bytes": 2 * 1024**3,  # Fixed 2GB KV cache - skips profiling
            "enforce_eager": True,
            "compilation_config": {"custom_ops": ["none"]},
        }

    def _log(self, msg: str):
        if self.verbose:
            print(f"[FastSwitch] {msg}")

    def initialize(self, model_name: str, **config) -> Any:
        """Initialize with first model."""
        from vllm import LLM

        merged_config = {**self.default_config, **config}

        self._log(f"Loading {model_name}...")
        t0 = time.perf_counter()
        self.llm = LLM(model=model_name, **merged_config)
        load_time = (time.perf_counter() - t0) * 1000

        self.current_model = model_name
        used, free = get_mem()
        self._log(f"Loaded in {load_time:.0f}ms, VRAM: {used:.1f}GB used")

        return self.llm

    def switch(self, target_model: str, **config) -> Tuple[Any, float, Dict[str, float]]:
        """Fast switch to different model."""
        from vllm import LLM

        if self.llm is None:
            raise RuntimeError("Must call initialize() first")

        merged_config = {**self.default_config, **config}
        timings = {}

        self._log(f"Switching {self.current_model.split('/')[-1]} → {target_model.split('/')[-1]}")
        t_total = time.perf_counter()

        # Optimized cleanup
        cleanup_time, gc_time = optimized_cleanup(self.llm)
        self.llm = None
        timings['cleanup'] = cleanup_time
        timings['gc'] = gc_time

        used, free = get_mem()
        self._log(f"After cleanup: {used:.1f}GB used, {free:.1f}GB free (gc: {gc_time:.0f}ms)")

        # Load new model
        t0 = time.perf_counter()
        self.llm = LLM(model=target_model, **merged_config)
        timings['load'] = (time.perf_counter() - t0) * 1000

        self.current_model = target_model
        total_time = (time.perf_counter() - t_total) * 1000

        used, free = get_mem()
        self._log(f"Timing breakdown:")
        self._log(f"  Cleanup: {timings['cleanup']:.0f}ms (gc: {timings['gc']:.0f}ms)")
        self._log(f"  Load:    {timings['load']:.0f}ms")
        self._log(f"  TOTAL:   {total_time:.0f}ms")
        self._log(f"  VRAM: {used:.1f}GB used")

        return self.llm, total_time, timings

    def generate(self, prompts: List[str], **kwargs) -> List[Any]:
        from vllm import SamplingParams
        params = SamplingParams(**kwargs)
        return self.llm.generate(prompts, params)

    def cleanup(self):
        if self.llm is not None:
            optimized_cleanup(self.llm)
            self.llm = None
        self.current_model = None


def benchmark():
    """Benchmark fast switch v3."""
    print("="*70)
    print("BLITZ FAST SWITCH V3")
    print("Optimized gc.collect() with pre-cleared references")
    print("="*70)

    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    used, free = get_mem()
    print(f"\nInitial: {used:.1f}GB used, {free:.1f}GB free")

    switcher = BlitzFastSwitchV3(verbose=True)

    qwen_model = "Qwen/Qwen2.5-7B-Instruct"
    mistral_model = "mistralai/Mistral-7B-Instruct-v0.3"

    results = []

    # Initialize
    print("\n" + "-"*70)
    print("INITIALIZE")
    print("-"*70)

    t0 = time.perf_counter()
    llm = switcher.initialize(qwen_model)
    init_time = (time.perf_counter() - t0) * 1000

    outputs = switcher.generate(["Capital of France?"], max_tokens=20, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "paris" in text.lower()
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:40]}...")
    results.append(("Init (Qwen)", init_time, valid, {}))

    # Switch to Mistral
    print("\n" + "-"*70)
    print("SWITCH 1: Qwen → Mistral")
    print("-"*70)

    llm, switch_time, timings = switcher.switch(mistral_model)

    outputs = switcher.generate(["What is 2+2?"], max_tokens=20, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "4" in text
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:40]}...")
    results.append(("Qwen→Mistral", switch_time, valid, timings))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, time_ms, valid, timings in results:
        status = "OK" if valid else "FAIL"
        if timings:
            print(f"  {name}: {time_ms:.0f}ms (cleanup: {timings.get('cleanup',0):.0f}ms, gc: {timings.get('gc',0):.0f}ms, load: {timings.get('load',0):.0f}ms) [{status}]")
        else:
            print(f"  {name}: {time_ms:.0f}ms [{status}]")

    switch_times = [r[1] for r in results[1:]]
    if switch_times:
        avg_switch = sum(switch_times) / len(switch_times)
        baseline = 6000
        improvement = (baseline - avg_switch) / baseline * 100

        print(f"\n  Switch time: {avg_switch:.0f}ms ({avg_switch/1000:.2f}s)")
        print(f"  Baseline: {baseline}ms")

        if improvement > 0:
            print(f"  Improvement: {improvement:.0f}% faster")
        else:
            print(f"  Change: {improvement:.0f}%")

    used, free = get_mem()
    print(f"\n  Final VRAM: {used:.1f}GB used, {free:.1f}GB free")

    switcher.cleanup()

    print("\n" + "="*70)


if __name__ == '__main__':
    benchmark()
