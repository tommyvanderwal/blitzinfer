#!/usr/bin/env python3
"""
BlitzInfer Direct GPU Loading Patch

Replaces vLLM's safetensors_weights_iterator with direct GPU loading.
This achieves 9.2 GB/s instead of vLLM's ~3.5 GB/s.

Expected savings: ~2.3s per model load
"""

import gc
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Any

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


def direct_gpu_weights_iterator(
    hf_weights_files: List[str],
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str = "lazy",
) -> Generator[Tuple[str, "torch.Tensor"], None, None]:
    """
    Direct GPU loading of safetensors weights.

    Instead of loading to CPU then copying to GPU (vLLM default ~3.5 GB/s),
    this loads directly to GPU (~9.2 GB/s on 780M).
    """
    from safetensors import safe_open
    from tqdm.auto import tqdm

    _BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"

    total_bytes = sum(Path(f).stat().st_size for f in hf_weights_files)
    t0 = time.perf_counter()

    for sf in tqdm(
        hf_weights_files,
        desc="Loading safetensors (Direct GPU)",
        disable=not use_tqdm_on_load,
        bar_format=_BAR_FORMAT,
    ):
        # Direct GPU loading - much faster than CPU→GPU copy
        with safe_open(sf, framework='pt', device='cuda') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                yield key, tensor

    load_time = (time.perf_counter() - t0) * 1000
    bandwidth = (total_bytes / (1024**3)) / (load_time / 1000)
    print(f"[DirectGPU] Loaded {total_bytes/(1024**3):.1f} GB in {load_time:.0f}ms ({bandwidth:.1f} GB/s)")


def patch_vllm_direct_gpu():
    """Patch vLLM to use direct GPU loading."""
    import vllm.model_executor.model_loader.weight_utils as weight_utils
    import vllm.model_executor.model_loader.default_loader as default_loader

    # Save originals
    weight_utils._original_safetensors_weights_iterator = weight_utils.safetensors_weights_iterator

    # Replace with direct GPU version
    weight_utils.safetensors_weights_iterator = direct_gpu_weights_iterator
    default_loader.safetensors_weights_iterator = direct_gpu_weights_iterator

    print("[DirectGPU] vLLM patched for direct GPU loading")


def unpatch_vllm():
    """Restore original vLLM loading."""
    import vllm.model_executor.model_loader.weight_utils as weight_utils

    if hasattr(weight_utils, '_original_safetensors_weights_iterator'):
        weight_utils.safetensors_weights_iterator = weight_utils._original_safetensors_weights_iterator
        print("[DirectGPU] vLLM restored to original loading")


class BlitzDirectGPUSwitcher:
    """
    Fast cross-architecture switcher using direct GPU loading.

    Performance targets:
    - Weight loading: ~1.5s (9.2 GB/s)
    - Total switch: ~3-4s (down from 6s)
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
            "enforce_eager": True,
            "compilation_config": {"custom_ops": ["none"]},
        }

    def _log(self, msg: str):
        if self.verbose:
            print(f"[DirectGPU] {msg}")

    def initialize(self, model_name: str, **config) -> Any:
        """Initialize with first model."""
        from vllm import LLM

        # Apply patch
        patch_vllm_direct_gpu()

        merged_config = {**self.default_config, **config}

        self._log(f"Loading {model_name}...")
        t0 = time.perf_counter()
        self.llm = LLM(model=model_name, **merged_config)
        load_time = (time.perf_counter() - t0) * 1000

        self.current_model = model_name
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / (1024**3)
        self._log(f"Loaded in {load_time:.0f}ms, VRAM: {used:.1f}GB")

        return self.llm

    def switch(self, target_model: str, **config) -> Tuple[Any, float]:
        """Fast switch to different model."""
        from vllm import LLM

        if self.llm is None:
            raise RuntimeError("Must call initialize() first")

        merged_config = {**self.default_config, **config}

        self._log(f"Switching {self.current_model.split('/')[-1]} → {target_model.split('/')[-1]}")
        t_total = time.perf_counter()

        # Fast cleanup (no gc.collect)
        t0 = time.perf_counter()
        gc.disable()

        del self.llm
        self.llm = None

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        cleanup_time = (time.perf_counter() - t0) * 1000
        self._log(f"Cleanup: {cleanup_time:.0f}ms")

        # Load new model
        t0 = time.perf_counter()
        self.llm = LLM(model=target_model, **merged_config)
        load_time = (time.perf_counter() - t0) * 1000

        gc.enable()

        self.current_model = target_model
        total_time = (time.perf_counter() - t_total) * 1000

        free, total = torch.cuda.mem_get_info()
        used = (total - free) / (1024**3)
        self._log(f"Load: {load_time:.0f}ms")
        self._log(f"TOTAL: {total_time:.0f}ms, VRAM: {used:.1f}GB")

        return self.llm, total_time

    def generate(self, prompts: List[str], **kwargs) -> List[Any]:
        from vllm import SamplingParams
        params = SamplingParams(**kwargs)
        return self.llm.generate(prompts, params)

    def cleanup(self):
        if self.llm is not None:
            del self.llm
            self.llm = None
        self.current_model = None
        gc.collect()
        torch.cuda.empty_cache()


def benchmark():
    """Benchmark the direct GPU switcher."""
    print("="*70)
    print("BLITZ DIRECT GPU LOADING SWITCHER")
    print("Target: 6s → 3s via direct GPU loading")
    print("="*70)

    # Warmup and cleanup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    print(f"\nInitial: {(total-free)/(1024**3):.1f}GB used, {free/(1024**3):.1f}GB free")

    switcher = BlitzDirectGPUSwitcher(verbose=True)

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
    results.append(("Init (Qwen)", init_time, valid))

    # Switch to Mistral
    print("\n" + "-"*70)
    print("SWITCH: Qwen → Mistral")
    print("-"*70)

    llm, switch_time = switcher.switch(mistral_model)

    outputs = switcher.generate(["What is 2+2?"], max_tokens=20, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "4" in text
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:40]}...")
    results.append(("Qwen→Mistral", switch_time, valid))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, time_ms, valid in results:
        status = "OK" if valid else "FAIL"
        print(f"  {name}: {time_ms:.0f}ms [{status}]")

    switch_times = [r[1] for r in results[1:]]
    if switch_times:
        avg_switch = sum(switch_times) / len(switch_times)
        print(f"\n  Switch time: {avg_switch:.0f}ms ({avg_switch/1000:.2f}s)")
        print(f"  Baseline: 6000ms")
        print(f"  Improvement: {(6000-avg_switch)/6000*100:.0f}%")

    switcher.cleanup()
    print("="*70)


if __name__ == '__main__':
    benchmark()
