#!/usr/bin/env python3
"""
BlitzInfer vLLM Patch: Replace vLLM's weight iterator with our bulk loader.

This patch modifies vLLM's safetensors_weights_iterator to use our optimized
double-buffered pinned memory approach, achieving ~9 GB/s instead of ~2 GB/s.

Usage:
    # Import this module BEFORE importing vLLM
    import blitz_vllm_patch
    blitz_vllm_patch.patch_vllm()

    # Then use vLLM normally
    from vllm import LLM
    llm = LLM(model="...")  # Will use fast weight loading
"""

import gc
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, Generator, List, Tuple

# Pre-allocate these globally for reuse across model switches
_BLITZ_GPU_BUFFER = None
_BLITZ_CPU_BUFFERS = None
_BLITZ_INITIALIZED = False

# Configuration
BLITZ_MAX_MODEL_SIZE_GB = 16.0
BLITZ_MAX_FILE_SIZE_GB = 16.0  # Increased to handle single-file models like Mistral
BLITZ_NUM_CPU_BUFFERS = 2
BLITZ_VERBOSE = True


def _ensure_blitz_buffers():
    """Initialize Blitz buffers if not already done."""
    global _BLITZ_GPU_BUFFER, _BLITZ_CPU_BUFFERS, _BLITZ_INITIALIZED

    if _BLITZ_INITIALIZED:
        return

    import torch

    if BLITZ_VERBOSE:
        print("[BlitzPatch] Pre-allocating buffers for fast weight loading...")

    t0 = time.perf_counter()

    # GPU buffer for model weights (float16)
    max_elements = int(BLITZ_MAX_MODEL_SIZE_GB * 1024**3 / 2)
    _BLITZ_GPU_BUFFER = torch.empty(max_elements, dtype=torch.float16, device='cuda')
    torch.cuda.synchronize()

    # Pinned CPU buffers for fast DMA
    max_file_elements = int(BLITZ_MAX_FILE_SIZE_GB * 1024**3 / 2)
    _BLITZ_CPU_BUFFERS = [
        torch.empty(max_file_elements, dtype=torch.float16, pin_memory=True)
        for _ in range(BLITZ_NUM_CPU_BUFFERS)
    ]

    init_time = (time.perf_counter() - t0) * 1000
    _BLITZ_INITIALIZED = True

    if BLITZ_VERBOSE:
        print(f"[BlitzPatch]   GPU buffer: {BLITZ_MAX_MODEL_SIZE_GB:.1f} GB")
        print(f"[BlitzPatch]   CPU buffers: {BLITZ_NUM_CPU_BUFFERS} x {BLITZ_MAX_FILE_SIZE_GB:.1f} GB (pinned)")
        print(f"[BlitzPatch]   Pre-alloc time: {init_time:.0f}ms")


def blitz_safetensors_weights_iterator(
    hf_weights_files: List[str],
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str = "lazy",
) -> Generator[Tuple[str, "torch.Tensor"], None, None]:
    """
    Blitz-optimized safetensors iterator using bulk GPU transfer.

    This replaces vLLM's safetensors_weights_iterator with our optimized version
    that achieves ~9 GB/s instead of ~2 GB/s.
    """
    import torch
    from safetensors import safe_open
    from tqdm.auto import tqdm

    global _BLITZ_GPU_BUFFER, _BLITZ_CPU_BUFFERS

    # Ensure buffers are initialized
    _ensure_blitz_buffers()

    # Phase 1: Collect metadata for all tensors
    file_meta = []
    total_offset = 0

    for sf in hf_weights_files:
        tensors = []
        with safe_open(sf, framework='pt', device='cpu') as f:
            for key in f.keys():
                t = f.get_tensor(key)
                n = t.numel()
                shape = t.shape
                dtype = t.dtype
                tensors.append((key, total_offset, n, shape, dtype))
                total_offset += n
        file_meta.append((sf, tensors))

    total_gb = total_offset * 2 / (1024**3)

    if BLITZ_VERBOSE:
        print(f"[BlitzPatch] Loading {total_gb:.1f} GB via bulk transfer...")

    # Phase 2: Double-buffered loading
    t0 = time.perf_counter()

    offset = 0
    current_buf = 0

    # Progress bar compatible with vLLM's format
    _BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"

    for i, (sf, tensors) in enumerate(tqdm(
        file_meta,
        desc="Loading safetensors (Blitz)",
        disable=not use_tqdm_on_load,
        bar_format=_BAR_FORMAT,
    )):
        cpu_buf = _BLITZ_CPU_BUFFERS[current_buf]

        # Load file to CPU buffer (memory-mapped, fast)
        buf_offset = 0
        with safe_open(sf, framework='pt', device='cpu') as f:
            for key, _, n, _, _ in tensors:
                t = f.get_tensor(key)
                # Handle different dtypes by viewing as float16
                flat = t.reshape(-1)
                if flat.dtype == torch.bfloat16:
                    flat = flat.view(torch.float16)
                elif flat.dtype != torch.float16:
                    flat = flat.to(torch.float16)
                cpu_buf[buf_offset:buf_offset+n].copy_(flat)
                buf_offset += n

        # Wait for previous async transfer to complete
        if i > 0:
            torch.cuda.synchronize()

        # Start async transfer of this file's data
        file_size = sum(n for _, _, n, _, _ in tensors)
        _BLITZ_GPU_BUFFER[offset:offset+file_size].copy_(
            cpu_buf[:file_size],
            non_blocking=True
        )

        offset += file_size
        current_buf = (current_buf + 1) % len(_BLITZ_CPU_BUFFERS)

    # Wait for final transfer
    torch.cuda.synchronize()
    load_time = (time.perf_counter() - t0) * 1000

    if BLITZ_VERBOSE:
        bandwidth = total_gb / (load_time / 1000)
        print(f"[BlitzPatch] Loaded {total_gb:.1f} GB in {load_time:.0f}ms ({bandwidth:.1f} GB/s)")

    # Phase 3: Yield tensors with views into GPU buffer
    for sf, tensors in file_meta:
        for key, off, n, shape, dtype in tensors:
            # Create view into GPU buffer with correct shape
            tensor = _BLITZ_GPU_BUFFER[off:off+n].view(shape)
            # Handle dtype conversion if needed
            if dtype == torch.bfloat16:
                tensor = tensor.view(torch.bfloat16)
            yield key, tensor


def fast_cleanup() -> Tuple[float, int]:
    """
    Fast cleanup: bypass vLLM, directly clear nn.Module parameters.

    Returns:
        (cleanup_time_ms, parameters_cleared)
    """
    import torch
    import torch.nn as nn

    t0 = time.perf_counter()

    # Unfreeze gc (vLLM freezes objects)
    gc.unfreeze()

    # Clear parameters on all nn.Modules
    cleared = 0
    for obj in gc.get_objects():
        try:
            if isinstance(obj, nn.Module):
                if hasattr(obj, '_parameters') and obj._parameters:
                    for key in list(obj._parameters.keys()):
                        param = obj._parameters.get(key)
                        if param is not None and param.is_cuda:
                            obj._parameters[key] = None
                            cleared += 1
        except:
            pass

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    cleanup_time = (time.perf_counter() - t0) * 1000
    return cleanup_time, cleared


def patch_vllm():
    """
    Patch vLLM to use Blitz's fast weight loading.

    Call this BEFORE importing vLLM's LLM class.
    """
    # Import vLLM's weight_utils module
    import vllm.model_executor.model_loader.weight_utils as weight_utils
    import vllm.model_executor.model_loader.default_loader as default_loader

    # Save original for potential restoration
    weight_utils._original_safetensors_weights_iterator = weight_utils.safetensors_weights_iterator

    # Replace in weight_utils
    weight_utils.safetensors_weights_iterator = blitz_safetensors_weights_iterator

    # Also replace in default_loader (it imports at module load time)
    default_loader.safetensors_weights_iterator = blitz_safetensors_weights_iterator

    if BLITZ_VERBOSE:
        print("[BlitzPatch] vLLM patched for fast weight loading")


def unpatch_vllm():
    """Restore original vLLM weight loading."""
    import vllm.model_executor.model_loader.weight_utils as weight_utils

    if hasattr(weight_utils, '_original_safetensors_weights_iterator'):
        weight_utils.safetensors_weights_iterator = weight_utils._original_safetensors_weights_iterator
        print("[BlitzPatch] vLLM restored to original weight loading")


def test_patched_vllm():
    """Test vLLM with Blitz patch applied."""
    import os
    import sys
    import types

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
    from vllm import LLM, SamplingParams

    # Apply patch BEFORE creating LLM
    patch_vllm()

    print("=" * 70)
    print("BLITZ-PATCHED VLLM TEST")
    print("=" * 70)

    # Warmup GPU
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
    print(f"\nInitial GPU memory: {free_mem:.1f} GB free")

    # vLLM config
    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.30,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # === TEST 1: Load with Blitz patch ===
    print("\n" + "=" * 70)
    print("TEST 1: Load with Blitz Patch")
    print("=" * 70)

    t0 = time.perf_counter()
    llm = LLM(model=model_name, **config)
    load_time = (time.perf_counter() - t0) * 1000

    print(f"\nTotal load time: {load_time:.0f}ms")
    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
    print(f"GPU memory: {free_mem:.1f} GB free")

    # Verify output
    params = SamplingParams(max_tokens=50, temperature=0.7)
    outputs = llm.generate(["What is 2 + 2? Answer:"], params)
    text = outputs[0].outputs[0].text
    print(f"Output: {text[:80]}...")
    print(f"Valid: {bool(text.strip())}")

    # === TEST 2: Switch test ===
    print("\n" + "=" * 70)
    print("TEST 2: Fast Switch Cycle")
    print("=" * 70)

    # Cleanup
    del llm
    gc.collect()

    t_switch = time.perf_counter()

    cleanup_time, cleared = fast_cleanup()
    print(f"Cleanup: {cleanup_time:.0f}ms (cleared {cleared} params)")

    # Reload with patch
    t_load = time.perf_counter()
    llm2 = LLM(model=model_name, **config)
    reload_time = (time.perf_counter() - t_load) * 1000

    total_switch = (time.perf_counter() - t_switch) * 1000
    print(f"Reload: {reload_time:.0f}ms")
    print(f"TOTAL SWITCH: {total_switch:.0f}ms ({total_switch/1000:.2f}s)")

    # Verify
    outputs = llm2.generate(["Capital of France?"], params)
    text = outputs[0].outputs[0].text
    print(f"Output: {text[:60]}...")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Initial load: {load_time:.0f}ms")
    print(f"Switch time:  {total_switch:.0f}ms ({total_switch/1000:.2f}s)")
    print(f"Target: <3s   Achieved: {total_switch/1000:.2f}s")

    # Cleanup
    del llm2
    gc.collect()
    fast_cleanup()

    if total_switch < 3000:
        print("\n[PASS] Sub-3 second model switching achieved!")
    else:
        print(f"\n[INFO] Switch time {total_switch/1000:.2f}s > 3s target")

    print("=" * 70)


if __name__ == '__main__':
    test_patched_vllm()
