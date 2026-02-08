#!/usr/bin/env python3
"""
BlitzInfer Simple Pinned Memory Loading

KISS approach: Single pre-allocated pinned buffer for fast CPU→GPU transfers.
No fancy pipelining - just pinned memory for ~2x faster transfers.

Based on profiling:
- Sequential (non-pinned): 3.4s at 4 GB/s
- Pinned (1 buffer): 1.5s at 9 GB/s
- More buffers don't help - the speedup is from pinned memory, not parallelism
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

# Single global pinned buffer - allocated once, reused forever
_PINNED_BUFFER = None
_BUFFER_SIZE = 600_000_000  # 600M elements, ~1.2GB in fp16


def get_pinned_buffer() -> torch.Tensor:
    """Get or create the single pinned buffer."""
    global _PINNED_BUFFER
    if _PINNED_BUFFER is None:
        _PINNED_BUFFER = torch.empty(_BUFFER_SIZE, dtype=torch.float16, pin_memory=True)
        print(f"[SimplePinned] Allocated pinned buffer: {_BUFFER_SIZE * 2 / (1024**3):.2f} GB")
    return _PINNED_BUFFER


def simple_pinned_weights_iterator(
    hf_weights_files: List[str],
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str = "lazy",
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """
    Simple weight iterator using a single pinned buffer.

    For each tensor:
    1. Copy to pinned buffer
    2. Yield a clone from the pinned buffer (clone is also pinned)
    3. vLLM copies to GPU using fast DMA
    """
    from tqdm.auto import tqdm

    pinned_buf = get_pinned_buffer()

    _BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"

    total_bytes = sum(Path(f).stat().st_size for f in hf_weights_files)
    t0 = time.perf_counter()
    tensor_count = 0

    for st_file in tqdm(
        hf_weights_files,
        desc="Loading weights (pinned)",
        disable=not use_tqdm_on_load,
        bar_format=_BAR_FORMAT,
    ):
        with safe_open(st_file, framework='pt') as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                numel = tensor.numel()
                shape = tensor.shape
                dtype = tensor.dtype
                tensor_count += 1

                # For supported dtypes, use the pinned buffer
                # NO CLONE - yield view directly. vLLM processes synchronously,
                # so buffer won't be reused until after GPU copy.
                if dtype == torch.float16 and numel <= _BUFFER_SIZE:
                    pinned_buf[:numel].copy_(tensor.view(-1))
                    yield name, pinned_buf[:numel].view(shape)
                elif dtype == torch.bfloat16 and numel <= _BUFFER_SIZE:
                    # bf16: copy via int16 view for bit-exact transfer
                    pinned_buf[:numel].view(torch.int16).copy_(tensor.view(-1).view(torch.int16))
                    yield name, pinned_buf[:numel].view(torch.bfloat16).view(shape)
                else:
                    # Fallback for other dtypes or oversized tensors
                    yield name, tensor

    load_time = (time.perf_counter() - t0) * 1000
    bandwidth = (total_bytes / (1024**3)) / (load_time / 1000)
    print(f"[SimplePinned] {tensor_count} tensors, {total_bytes/(1024**3):.1f} GB in {load_time:.0f}ms ({bandwidth:.1f} GB/s)")


def patch_vllm():
    """Patch vLLM to use simple pinned loading."""
    import vllm.model_executor.model_loader.weight_utils as weight_utils
    import vllm.model_executor.model_loader.default_loader as default_loader

    if not hasattr(weight_utils, '_original_iterator'):
        weight_utils._original_iterator = weight_utils.safetensors_weights_iterator
        default_loader._original_iterator = default_loader.safetensors_weights_iterator

    weight_utils.safetensors_weights_iterator = simple_pinned_weights_iterator
    default_loader.safetensors_weights_iterator = simple_pinned_weights_iterator
    print("[SimplePinned] vLLM patched")


def unpatch_vllm():
    """Restore original vLLM loading."""
    import vllm.model_executor.model_loader.weight_utils as weight_utils
    import vllm.model_executor.model_loader.default_loader as default_loader

    if hasattr(weight_utils, '_original_iterator'):
        weight_utils.safetensors_weights_iterator = weight_utils._original_iterator
        default_loader.safetensors_weights_iterator = default_loader._original_iterator
        print("[SimplePinned] vLLM restored")


def get_mem() -> Tuple[float, float]:
    """Get (used, free) GPU memory in GB."""
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1024**3)
    return used, free / (1024**3)


def cleanup_model(llm):
    """Clean up a model and free GPU memory."""
    if llm is not None:
        del llm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


class BlitzSimplePinned:
    """
    Simple, reliable model switcher using pinned memory.

    Uses a single pre-allocated pinned buffer for ~2x faster weight loading.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.llm = None
        self.current_model = None
        self.timings = {}

        self.default_config = {
            "dtype": "float16",
            "gpu_memory_utilization": 0.25,
            "max_model_len": 512,
            "max_num_batched_tokens": 512,
            "kv_cache_memory_bytes": 2 * 1024**3,
            "enforce_eager": True,
            "compilation_config": {"custom_ops": ["none"]},
        }

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Blitz] {msg}")

    def _profile(self, name: str):
        """Context manager for timing code sections."""
        class Timer:
            def __init__(self, timings, name):
                self.timings = timings
                self.name = name
            def __enter__(self):
                self.t0 = time.perf_counter()
                return self
            def __exit__(self, *args):
                self.timings[self.name] = (time.perf_counter() - self.t0) * 1000
        return Timer(self.timings, name)

    def initialize(self, model_name: str, **config) -> Any:
        """Initialize with first model - profiles all phases."""
        from vllm import LLM

        self.timings = {}
        t_total = time.perf_counter()

        # Patch vLLM
        with self._profile("patch"):
            patch_vllm()

        # Allocate pinned buffer
        with self._profile("alloc_pinned"):
            get_pinned_buffer()

        merged_config = {**self.default_config, **config}
        self._log(f"Loading {model_name}...")

        # Load model
        with self._profile("vllm_init"):
            self.llm = LLM(model=model_name, **merged_config)

        self.current_model = model_name
        total_time = (time.perf_counter() - t_total) * 1000

        used, free = get_mem()
        self._log(f"Initialization complete in {total_time:.0f}ms")
        self._log(f"  Patch: {self.timings.get('patch', 0):.0f}ms")
        self._log(f"  Pinned alloc: {self.timings.get('alloc_pinned', 0):.0f}ms")
        self._log(f"  vLLM init: {self.timings.get('vllm_init', 0):.0f}ms")
        self._log(f"  VRAM: {used:.1f}GB used, {free:.1f}GB free")

        return self.llm

    def switch(self, target_model: str, **config) -> Tuple[Any, float, Dict[str, float]]:
        """
        Switch to a different model - profiles all phases.

        Phases:
        1. Cleanup old model
        2. Load new model (weights, tokenizer, engine)
        """
        from vllm import LLM

        if self.llm is None:
            raise RuntimeError("Must call initialize() first")

        self.timings = {}
        merged_config = {**self.default_config, **config}

        self._log(f"Switching: {self.current_model.split('/')[-1]} → {target_model.split('/')[-1]}")
        t_total = time.perf_counter()

        # Phase 1: Cleanup old model
        with self._profile("cleanup"):
            cleanup_model(self.llm)
            self.llm = None

        used, free = get_mem()
        self._log(f"  After cleanup: {used:.1f}GB used, {free:.1f}GB free")

        # Phase 2: Load new model
        with self._profile("load"):
            self.llm = LLM(model=target_model, **merged_config)

        self.current_model = target_model
        total_time = (time.perf_counter() - t_total) * 1000

        used, free = get_mem()
        self._log(f"Switch complete in {total_time:.0f}ms")
        self._log(f"  Cleanup: {self.timings.get('cleanup', 0):.0f}ms")
        self._log(f"  Load: {self.timings.get('load', 0):.0f}ms")
        self._log(f"  VRAM: {used:.1f}GB used, {free:.1f}GB free")

        return self.llm, total_time, self.timings

    def generate(self, prompts: List[str], **kwargs) -> List[Any]:
        """Generate text with current model."""
        from vllm import SamplingParams
        params = SamplingParams(**kwargs)
        return self.llm.generate(prompts, params)

    def cleanup_all(self):
        """Full cleanup."""
        if self.llm is not None:
            cleanup_model(self.llm)
            self.llm = None
        self.current_model = None
        unpatch_vllm()


def benchmark():
    """Full benchmark with profiling."""
    print("="*70)
    print("BLITZ SIMPLE PINNED MEMORY")
    print("="*70)
    print("""
Approach: Single pre-allocated pinned buffer
Expected: ~2x faster weight loading (9 GB/s vs 4 GB/s)
""")

    # Initial GPU setup
    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    used, free = get_mem()
    print(f"Initial GPU state: {used:.1f}GB used, {free:.1f}GB free\n")

    switcher = BlitzSimplePinned(verbose=True)

    qwen = "Qwen/Qwen2.5-7B-Instruct"
    mistral = "mistralai/Mistral-7B-Instruct-v0.3"

    all_results = []

    # =========================================================================
    # PHASE 1: Initialize with Qwen
    # =========================================================================
    print("-"*70)
    print("PHASE 1: Initialize with Qwen")
    print("-"*70)

    t0 = time.perf_counter()
    llm = switcher.initialize(qwen)
    init_time = (time.perf_counter() - t0) * 1000

    # Verify it works
    outputs = switcher.generate(["What is 2+2?"], max_tokens=20, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "4" in text
    print(f"Verification: {'PASS' if valid else 'FAIL'} - {text[:50]}...")
    all_results.append(("Init Qwen", init_time, valid, switcher.timings.copy()))

    # =========================================================================
    # PHASE 2: Switch Qwen → Mistral
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 2: Switch Qwen → Mistral")
    print("-"*70)

    llm, switch_time, timings = switcher.switch(mistral)

    outputs = switcher.generate(["Capital of France?"], max_tokens=20, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "paris" in text.lower()
    print(f"Verification: {'PASS' if valid else 'FAIL'} - {text[:50]}...")
    all_results.append(("Qwen→Mistral", switch_time, valid, timings.copy()))

    # =========================================================================
    # PHASE 3: Switch Mistral → Qwen
    # =========================================================================
    print("\n" + "-"*70)
    print("PHASE 3: Switch Mistral → Qwen")
    print("-"*70)

    llm, switch_time, timings = switcher.switch(qwen)

    outputs = switcher.generate(["Largest planet in solar system?"], max_tokens=20, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "jupiter" in text.lower()
    print(f"Verification: {'PASS' if valid else 'FAIL'} - {text[:50]}...")
    all_results.append(("Mistral→Qwen", switch_time, valid, timings.copy()))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Phase':<20} {'Total':<10} {'Cleanup':<10} {'Load':<10} {'Status':<8}")
    print("-"*58)
    for name, total, valid, timings in all_results:
        cleanup = timings.get('cleanup', '-')
        load = timings.get('load', timings.get('vllm_init', '-'))
        cleanup_str = f"{cleanup:.0f}ms" if isinstance(cleanup, float) else cleanup
        load_str = f"{load:.0f}ms" if isinstance(load, float) else load
        status = "PASS" if valid else "FAIL"
        print(f"{name:<20} {total:<10.0f} {cleanup_str:<10} {load_str:<10} {status:<8}")

    # Calculate averages
    switch_results = [r for r in all_results if "→" in r[0]]
    if switch_results:
        avg_switch = sum(r[1] for r in switch_results) / len(switch_results)
        avg_cleanup = sum(r[3].get('cleanup', 0) for r in switch_results) / len(switch_results)
        avg_load = sum(r[3].get('load', 0) for r in switch_results) / len(switch_results)

        print(f"\n{'AVERAGE SWITCH':<20} {avg_switch:<10.0f} {avg_cleanup:<10.0f} {avg_load:<10.0f}")

        baseline = 5700  # Previous baseline
        improvement = (baseline - avg_switch) / baseline * 100
        print(f"\nBaseline: {baseline}ms")
        print(f"Current:  {avg_switch:.0f}ms")
        if improvement > 0:
            print(f"Improvement: {improvement:.1f}% faster")
        else:
            print(f"Regression: {-improvement:.1f}% slower")

    used, free = get_mem()
    print(f"\nFinal GPU state: {used:.1f}GB used, {free:.1f}GB free")

    switcher.cleanup_all()
    print("\n" + "="*70)


if __name__ == '__main__':
    benchmark()
