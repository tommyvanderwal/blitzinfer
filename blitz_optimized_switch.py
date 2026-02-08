#!/usr/bin/env python3
"""
BlitzInfer Optimized Cross-Architecture Switch

Key insight: The pinned memory approach needs a reusable buffer, not
per-tensor allocation. But vLLM's iterator-based loading doesn't allow this
without major changes.

Instead, let's focus on what we CAN optimize:
1. Use kv_cache_memory_bytes to skip memory profiling (already done)
2. Minimize cleanup overhead
3. Profile to understand remaining bottlenecks

Current performance:
- Weight loading: ~4s (limited by safetensors + copy overhead)
- Model construction: ~1.5s
- Engine init: ~0.5s
- Total: ~6s

This is close to the practical minimum without modifying vLLM internals.
"""

import gc
import os
import sys
import time
import types
from typing import Any, Dict, List, Tuple

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


def cleanup(llm):
    """Optimized cleanup - minimal but effective."""
    if llm is not None:
        del llm

    # Single gc.collect is usually sufficient
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


class BlitzOptimizedSwitch:
    """
    Optimized cross-architecture switcher.

    Best achievable with current vLLM architecture:
    - Weight loading: ~3.5-4s (safetensors + copy overhead)
    - Model construction: ~1-1.5s
    - Engine init: ~0.5s
    - Total: ~5-6s

    Key optimizations applied:
    - kv_cache_memory_bytes: Skip expensive memory profiling
    - VLLM_SKIP_WARMUP: Skip kernel warmup
    - enforce_eager: Required for gfx1103
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
            "kv_cache_memory_bytes": 2 * 1024**3,  # CRITICAL: Skip memory profiling
            "enforce_eager": True,
            "compilation_config": {"custom_ops": ["none"]},
        }

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Blitz] {msg}")

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

        # Cleanup
        t0 = time.perf_counter()
        cleanup(self.llm)
        self.llm = None
        timings['cleanup'] = (time.perf_counter() - t0) * 1000

        used, free = get_mem()
        self._log(f"After cleanup: {used:.1f}GB used, {free:.1f}GB free (cleanup: {timings['cleanup']:.0f}ms)")

        # Load new model
        t0 = time.perf_counter()
        self.llm = LLM(model=target_model, **merged_config)
        timings['load'] = (time.perf_counter() - t0) * 1000

        self.current_model = target_model
        total_time = (time.perf_counter() - t_total) * 1000

        used, free = get_mem()
        self._log(f"Total: {total_time:.0f}ms (cleanup: {timings['cleanup']:.0f}ms, load: {timings['load']:.0f}ms)")
        self._log(f"VRAM: {used:.1f}GB used")

        return self.llm, total_time, timings

    def generate(self, prompts: List[str], **kwargs) -> List[Any]:
        from vllm import SamplingParams
        params = SamplingParams(**kwargs)
        return self.llm.generate(prompts, params)

    def cleanup_all(self):
        if self.llm is not None:
            cleanup(self.llm)
            self.llm = None
        self.current_model = None


def benchmark():
    """Benchmark optimized cross-architecture switching."""
    print("="*70)
    print("BLITZ OPTIMIZED CROSS-ARCHITECTURE SWITCH")
    print("="*70)
    print("""
Key optimizations:
- kv_cache_memory_bytes: Skip memory profiling
- VLLM_SKIP_WARMUP: Skip kernel warmup
- gc.collect() with proper cleanup

Target: Consistent ~5-6s switching
""")

    # Full cleanup
    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    used, free = get_mem()
    print(f"Initial: {used:.1f}GB used, {free:.1f}GB free")

    switcher = BlitzOptimizedSwitch(verbose=True)

    qwen_model = "Qwen/Qwen2.5-7B-Instruct"
    mistral_model = "mistralai/Mistral-7B-Instruct-v0.3"

    results = []

    # Initialize with Qwen
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

    # Multiple switch cycles to get average
    n_switches = 4

    for i in range(n_switches // 2):
        # Switch to Mistral
        print(f"\n{'-'*70}")
        print(f"SWITCH {i*2+1}: Qwen → Mistral")
        print(f"{'-'*70}")

        llm, switch_time, timings = switcher.switch(mistral_model)

        outputs = switcher.generate(["What is 2+2?"], max_tokens=20, temperature=0.7)
        text = outputs[0].outputs[0].text if outputs else ""
        valid = "4" in text
        print(f"Output [{['FAIL','OK'][valid]}]: {text[:40]}...")
        results.append((f"Qwen→Mistral #{i+1}", switch_time, valid, timings))

        # Switch back to Qwen
        print(f"\n{'-'*70}")
        print(f"SWITCH {i*2+2}: Mistral → Qwen")
        print(f"{'-'*70}")

        llm, switch_time, timings = switcher.switch(qwen_model)

        outputs = switcher.generate(["Largest planet?"], max_tokens=20, temperature=0.7)
        text = outputs[0].outputs[0].text if outputs else ""
        valid = "jupiter" in text.lower()
        print(f"Output [{['FAIL','OK'][valid]}]: {text[:40]}...")
        results.append((f"Mistral→Qwen #{i+1}", switch_time, valid, timings))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, time_ms, valid, timings in results:
        status = "OK" if valid else "FAIL"
        if timings:
            print(f"  {name}: {time_ms:.0f}ms (cleanup: {timings.get('cleanup',0):.0f}ms, load: {timings.get('load',0):.0f}ms) [{status}]")
        else:
            print(f"  {name}: {time_ms:.0f}ms [{status}]")

    switch_times = [r[1] for r in results[1:]]
    if switch_times:
        avg_switch = sum(switch_times) / len(switch_times)
        min_switch = min(switch_times)
        max_switch = max(switch_times)

        print(f"\n  Average switch: {avg_switch:.0f}ms ({avg_switch/1000:.2f}s)")
        print(f"  Min switch:     {min_switch:.0f}ms")
        print(f"  Max switch:     {max_switch:.0f}ms")

        if avg_switch < 6000:
            print(f"\n  [PASS] Sub-6s average switching achieved!")

    used, free = get_mem()
    print(f"\n  Final VRAM: {used:.1f}GB used, {free:.1f}GB free")

    switcher.cleanup_all()
    print("\n" + "="*70)


if __name__ == '__main__':
    benchmark()
