#!/usr/bin/env python3
"""
BlitzInfer Pinned Memory Fast Switch

Uses pinned CPU memory for 2.2x faster weight loading.

Benchmark results:
- Normal copy:  3.3s at 4.0 GB/s
- Pinned copy:  1.5s at 8.9 GB/s (2.2x faster)

Target: Reduce cross-architecture switch from ~5.7s to ~3.5s
"""

import gc
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

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
from safetensors.torch import safe_open



def pinned_safetensors_weights_iterator(
    hf_weights_files: List[str],
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str = "lazy",
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """
    Optimized weight iterator using pinned memory for CPU→GPU transfers.

    This replaces vLLM's safetensors_weights_iterator to achieve ~8.9 GB/s
    instead of the default ~4 GB/s.

    Strategy: For each tensor, create a pinned copy. The weight_loader will
    then copy this pinned tensor to GPU more efficiently than non-pinned.
    """
    from tqdm.auto import tqdm

    _BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"

    total_bytes = sum(Path(f).stat().st_size for f in hf_weights_files)
    t0 = time.perf_counter()

    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors (Pinned)",
        disable=not use_tqdm_on_load,
        bar_format=_BAR_FORMAT,
    ):
        with safe_open(st_file, framework='pt') as f:
            for name in f.keys():
                tensor = f.get_tensor(name)  # CPU tensor (non-pinned)

                # Create a pinned copy of this tensor
                # This is the key optimization: pinned memory transfers to GPU faster
                pinned_tensor = torch.empty_like(tensor, pin_memory=True)
                pinned_tensor.copy_(tensor)

                yield name, pinned_tensor

    load_time = (time.perf_counter() - t0) * 1000
    bandwidth = (total_bytes / (1024**3)) / (load_time / 1000)
    print(f"[PinnedMem] Loaded {total_bytes/(1024**3):.1f} GB in {load_time:.0f}ms ({bandwidth:.1f} GB/s)")


def patch_vllm_pinned_memory():
    """Patch vLLM to use pinned memory for weight loading."""
    import vllm.model_executor.model_loader.weight_utils as weight_utils
    import vllm.model_executor.model_loader.default_loader as default_loader

    # Save original
    if not hasattr(weight_utils, '_original_safetensors_weights_iterator'):
        weight_utils._original_safetensors_weights_iterator = weight_utils.safetensors_weights_iterator

    # Replace with pinned version
    weight_utils.safetensors_weights_iterator = pinned_safetensors_weights_iterator
    default_loader.safetensors_weights_iterator = pinned_safetensors_weights_iterator

    print("[PinnedMem] vLLM patched for pinned memory loading")


def unpatch_vllm():
    """Restore original vLLM loading."""
    import vllm.model_executor.model_loader.weight_utils as weight_utils

    if hasattr(weight_utils, '_original_safetensors_weights_iterator'):
        weight_utils.safetensors_weights_iterator = weight_utils._original_safetensors_weights_iterator
        print("[PinnedMem] vLLM restored to original loading")


def get_mem() -> Tuple[float, float]:
    """Get (used, free) GPU memory in GB."""
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**3)
    return used, free / (1024**3)


def cleanup(llm):
    """Full cleanup."""
    if llm is not None:
        del llm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


class BlitzPinnedMemorySwitch:
    """
    Fast cross-architecture switcher using pinned memory.

    Expected improvement: 3.5s → 2.0s for weight loading
    Total switch target: 5.7s → 3.5-4.0s
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
            "kv_cache_memory_bytes": 2 * 1024**3,  # Fixed KV cache skips profiling
            "enforce_eager": True,
            "compilation_config": {"custom_ops": ["none"]},
        }

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Blitz] {msg}")

    def initialize(self, model_name: str, **config) -> Any:
        """Initialize with first model."""
        from vllm import LLM

        # Apply pinned memory patch
        patch_vllm_pinned_memory()

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
        self._log(f"After cleanup: {used:.1f}GB used, {free:.1f}GB free")

        # Load new model with pinned memory
        t0 = time.perf_counter()
        self.llm = LLM(model=target_model, **merged_config)
        timings['load'] = (time.perf_counter() - t0) * 1000

        self.current_model = target_model
        total_time = (time.perf_counter() - t_total) * 1000

        used, free = get_mem()
        self._log(f"Timing breakdown:")
        self._log(f"  Cleanup: {timings['cleanup']:.0f}ms")
        self._log(f"  Load:    {timings['load']:.0f}ms")
        self._log(f"  TOTAL:   {total_time:.0f}ms")
        self._log(f"  VRAM: {used:.1f}GB used")

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
        unpatch_vllm()


def benchmark():
    """Benchmark pinned memory fast switch."""
    print("="*70)
    print("BLITZ PINNED MEMORY FAST SWITCH")
    print("Target: 5.7s → 3.5s via pinned memory (8.9 GB/s)")
    print("="*70)

    # Full cleanup
    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    used, free = get_mem()
    print(f"\nInitial: {used:.1f}GB used, {free:.1f}GB free")

    switcher = BlitzPinnedMemorySwitch(verbose=True)

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

    # Switch back to Qwen
    print("\n" + "-"*70)
    print("SWITCH 2: Mistral → Qwen")
    print("-"*70)

    llm, switch_time, timings = switcher.switch(qwen_model)

    outputs = switcher.generate(["What is the largest planet?"], max_tokens=20, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "jupiter" in text.lower()
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:40]}...")
    results.append(("Mistral→Qwen", switch_time, valid, timings))

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
        baseline = 5700  # Previous best
        improvement = (baseline - avg_switch) / baseline * 100

        print(f"\n  Average switch: {avg_switch:.0f}ms ({avg_switch/1000:.2f}s)")
        print(f"  Previous baseline: {baseline}ms")

        if improvement > 0:
            print(f"  Improvement: {improvement:.0f}% faster")
            if avg_switch < 4000:
                print(f"\n  [PASS] Sub-4s switching achieved!")
        else:
            print(f"  Change: {improvement:.0f}%")

    used, free = get_mem()
    print(f"\n  Final VRAM: {used:.1f}GB used, {free:.1f}GB free")

    switcher.cleanup_all()
    print("\n" + "="*70)


if __name__ == '__main__':
    benchmark()
