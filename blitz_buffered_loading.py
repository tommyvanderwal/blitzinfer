#!/usr/bin/env python3
"""
BlitzInfer Buffered Weight Loading

Uses a pool of pre-allocated pinned buffers that are reused across weights.
This avoids per-tensor allocation overhead while still achieving faster transfers.

Key insight: vLLM's weight_loader copies tensors to GPU immediately after receiving them.
So we can safely reuse buffers in a round-robin fashion with proper synchronization.

Approach:
1. Pre-allocate N pinned buffers at startup
2. For each weight: copy to pinned buffer, yield, then reuse on next iteration
3. The GPU copy happens before we yield the next tensor, so buffer is safe to reuse
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

# Global pinned buffer pool (reused across loads)
_PINNED_POOL = None
_POOL_SIZE = 4  # Number of buffers in pool


def init_pinned_pool():
    """Initialize pinned buffer pool once at startup."""
    global _PINNED_POOL

    if _PINNED_POOL is None:
        # 600M elements = 1.2GB per buffer in fp16
        # 4 buffers = 4.8GB total pinned memory
        max_elements = 600_000_000
        _PINNED_POOL = [
            torch.empty(max_elements, dtype=torch.float16, pin_memory=True)
            for _ in range(_POOL_SIZE)
        ]
        print(f"[BufferedLoad] Initialized {_POOL_SIZE} pinned buffers ({max_elements * 2 / (1024**3):.1f} GB each)")


def buffered_safetensors_weights_iterator(
    hf_weights_files: List[str],
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str = "lazy",
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """
    Buffered weight iterator using pre-allocated pinned memory.

    Uses round-robin buffer allocation. Since vLLM's weight_loader copies
    to GPU immediately, we can safely reuse buffers.
    """
    from tqdm.auto import tqdm

    init_pinned_pool()

    _BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"

    total_bytes = sum(Path(f).stat().st_size for f in hf_weights_files)
    t0 = time.perf_counter()

    buffer_idx = 0

    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors (Buffered)",
        disable=not use_tqdm_on_load,
        bar_format=_BAR_FORMAT,
    ):
        with safe_open(st_file, framework='pt') as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                numel = tensor.numel()
                shape = tensor.shape
                dtype = tensor.dtype

                # Get buffer from pool
                pinned_buf = _PINNED_POOL[buffer_idx]

                # Copy to pinned buffer, handling dtype
                # View as bytes for dtype-agnostic copy
                flat_tensor = tensor.view(-1)
                if dtype == torch.float16:
                    pinned_buf[:numel].copy_(flat_tensor)
                    yield_tensor = pinned_buf[:numel].view(shape)
                elif dtype == torch.bfloat16:
                    # bf16 -> view as int16 for bit-exact copy
                    pinned_buf[:numel].view(torch.int16).copy_(flat_tensor.view(torch.int16))
                    yield_tensor = pinned_buf[:numel].view(torch.bfloat16).view(shape)
                else:
                    # For other dtypes, create pinned copy
                    pinned_tensor = torch.empty_like(tensor, pin_memory=True)
                    pinned_tensor.copy_(tensor)
                    yield name, pinned_tensor
                    buffer_idx = (buffer_idx + 1) % _POOL_SIZE
                    continue

                yield name, yield_tensor

                # Rotate to next buffer
                buffer_idx = (buffer_idx + 1) % _POOL_SIZE

    load_time = (time.perf_counter() - t0) * 1000
    bandwidth = (total_bytes / (1024**3)) / (load_time / 1000)
    print(f"[BufferedLoad] Loaded {total_bytes/(1024**3):.1f} GB in {load_time:.0f}ms ({bandwidth:.1f} GB/s)")


def patch_vllm_buffered():
    """Patch vLLM to use buffered weight loading."""
    import vllm.model_executor.model_loader.weight_utils as weight_utils
    import vllm.model_executor.model_loader.default_loader as default_loader

    if not hasattr(weight_utils, '_original_safetensors_weights_iterator'):
        weight_utils._original_safetensors_weights_iterator = weight_utils.safetensors_weights_iterator

    weight_utils.safetensors_weights_iterator = buffered_safetensors_weights_iterator
    default_loader.safetensors_weights_iterator = buffered_safetensors_weights_iterator

    print("[BufferedLoad] vLLM patched for buffered loading")


def unpatch_vllm():
    """Restore original vLLM loading."""
    import vllm.model_executor.model_loader.weight_utils as weight_utils

    if hasattr(weight_utils, '_original_safetensors_weights_iterator'):
        weight_utils.safetensors_weights_iterator = weight_utils._original_safetensors_weights_iterator
        print("[BufferedLoad] vLLM restored to original loading")


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


class BlitzBufferedSwitch:
    """
    Fast cross-architecture switcher using buffered pinned memory.

    Uses pre-allocated pinned buffers to avoid allocation overhead
    while still achieving faster CPU->GPU transfers.
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
            "kv_cache_memory_bytes": 2 * 1024**3,
            "enforce_eager": True,
            "compilation_config": {"custom_ops": ["none"]},
        }

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Blitz] {msg}")

    def initialize(self, model_name: str, **config) -> Any:
        """Initialize with first model."""
        from vllm import LLM

        patch_vllm_buffered()

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

        self._log(f"Switching {self.current_model.split('/')[-1]} -> {target_model.split('/')[-1]}")
        t_total = time.perf_counter()

        # Cleanup
        t0 = time.perf_counter()
        cleanup(self.llm)
        self.llm = None
        timings['cleanup'] = (time.perf_counter() - t0) * 1000

        used, free = get_mem()
        self._log(f"After cleanup: {used:.1f}GB used, {free:.1f}GB free")

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
        unpatch_vllm()


def benchmark():
    """Benchmark buffered weight loading."""
    print("="*70)
    print("BLITZ BUFFERED WEIGHT LOADING")
    print("="*70)
    print("""
Pre-allocated pinned buffer pool:
- Avoids per-tensor allocation overhead
- Reuses buffers in round-robin fashion
- Target: ~4s switch (down from 5.7s)
""")

    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    used, free = get_mem()
    print(f"Initial: {used:.1f}GB used, {free:.1f}GB free")

    switcher = BlitzBufferedSwitch(verbose=True)

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
    print("SWITCH 1: Qwen -> Mistral")
    print("-"*70)

    llm, switch_time, timings = switcher.switch(mistral_model)

    outputs = switcher.generate(["What is 2+2?"], max_tokens=20, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "4" in text
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:40]}...")
    results.append(("Qwen->Mistral", switch_time, valid, timings))

    # Switch back to Qwen
    print("\n" + "-"*70)
    print("SWITCH 2: Mistral -> Qwen")
    print("-"*70)

    llm, switch_time, timings = switcher.switch(qwen_model)

    outputs = switcher.generate(["Largest planet?"], max_tokens=20, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs else ""
    valid = "jupiter" in text.lower()
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:40]}...")
    results.append(("Mistral->Qwen", switch_time, valid, timings))

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
        baseline = 5700

        print(f"\n  Average switch: {avg_switch:.0f}ms ({avg_switch/1000:.2f}s)")
        print(f"  Previous baseline: {baseline}ms")

        improvement = (baseline - avg_switch) / baseline * 100
        if improvement > 0:
            print(f"  Improvement: {improvement:.0f}% faster")
            if avg_switch < 4500:
                print(f"\n  [PASS] Sub-4.5s switching achieved!")
        else:
            print(f"  Change: {improvement:.0f}%")

    used, free = get_mem()
    print(f"\n  Final VRAM: {used:.1f}GB used, {free:.1f}GB free")

    switcher.cleanup_all()
    print("\n" + "="*70)


if __name__ == '__main__':
    benchmark()
