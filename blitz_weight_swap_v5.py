#!/usr/bin/env python3
"""
BlitzInfer Weight Swap v5: Hybrid bulk load with GPU conversion.

Strategy:
1. Load raw bf16 bytes to CPU buffer (fast mmap)
2. Bulk transfer to GPU as bf16 (fast DMA at ~10 GB/s)
3. Convert bf16→fp16 on GPU (fast compute)
4. Fuse and inject into model params

This should give us ~2s swap time with correct outputs.
"""

import gc
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
import torch.nn as nn
from safetensors import safe_open
from huggingface_hub import snapshot_download


def get_inner_model(llm) -> Optional[nn.Module]:
    """Extract the inner nn.Module from vLLM's LLM wrapper."""
    try:
        return llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    except AttributeError:
        return None


class BlitzWeightSwapperV5:
    """
    Ultra-fast weight swapping with bulk bf16 load + GPU conversion.
    """

    def __init__(
        self,
        max_model_size_gb: float = 16.0,
        max_file_size_gb: float = 4.0,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.max_model_size_gb = max_model_size_gb
        self.max_file_size_gb = max_file_size_gb

        # Buffers - now in bf16 for raw loading
        self.gpu_buffer_bf16 = None
        self.cpu_buffers_bf16 = None
        self.buffers_ready = False

        # Model state
        self.llm = None
        self.model = None

    def _ensure_buffers(self):
        """Pre-allocate pinned CPU and GPU buffers in bf16."""
        if self.buffers_ready:
            return

        if self.verbose:
            print("[BlitzSwapV5] Pre-allocating bf16 buffers...")

        t0 = time.perf_counter()

        # GPU buffer in bf16 (same byte size as fp16)
        max_elements = int(self.max_model_size_gb * 1024**3 / 2)
        self.gpu_buffer_bf16 = torch.empty(max_elements, dtype=torch.bfloat16, device='cuda')
        torch.cuda.synchronize()

        # Pinned CPU buffers in bf16
        max_file_elements = int(self.max_file_size_gb * 1024**3 / 2)
        self.cpu_buffers_bf16 = [
            torch.empty(max_file_elements, dtype=torch.bfloat16, pin_memory=True)
            for _ in range(2)
        ]

        init_time = (time.perf_counter() - t0) * 1000
        self.buffers_ready = True

        if self.verbose:
            print(f"[BlitzSwapV5]   Pre-alloc time: {init_time:.0f}ms")

    def _swap_weights_bulk(self, safetensor_files: List[Path]) -> Tuple[float, float, float]:
        """
        Load weights as bf16, transfer to GPU, convert to fp16, then inject.
        """
        t_total = time.perf_counter()
        self._ensure_buffers()

        # Phase 1: Bulk load bf16 weights to GPU
        t0 = time.perf_counter()

        weight_locations = {}  # key -> (buffer_offset, numel, shape)
        buffer_offset = 0
        current_buf = 0

        for i, sf in enumerate(safetensor_files):
            cpu_buf = self.cpu_buffers_bf16[current_buf]

            # Load file to CPU buffer - bf16 weights copied directly
            buf_pos = 0
            with safe_open(str(sf), framework='pt', device='cpu') as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    # Weights should be bf16, copy directly
                    flat = t.reshape(-1)
                    n = flat.numel()

                    # View as bf16 for the copy (same bytes, just interpreted as bf16)
                    if flat.dtype == torch.bfloat16:
                        cpu_buf[buf_pos:buf_pos+n].copy_(flat)
                    else:
                        # Handle non-bf16 (convert on CPU if needed)
                        cpu_buf[buf_pos:buf_pos+n].copy_(flat.to(torch.bfloat16))

                    weight_locations[key] = (buffer_offset + buf_pos, n, t.shape)
                    buf_pos += n

            # Wait for previous transfer
            if i > 0:
                torch.cuda.synchronize()

            # Bulk transfer to GPU (bf16 → bf16, fast DMA)
            file_size = buf_pos
            self.gpu_buffer_bf16[buffer_offset:buffer_offset+file_size].copy_(
                cpu_buf[:file_size],
                non_blocking=True
            )

            buffer_offset += file_size
            current_buf = (current_buf + 1) % 2

        torch.cuda.synchronize()
        load_time = (time.perf_counter() - t0) * 1000

        total_gb = buffer_offset * 2 / (1024**3)

        # Phase 2: Inject with on-the-fly bf16→fp16 conversion and fusion
        t1 = time.perf_counter()
        updated = 0
        skipped = 0

        for name, param in self.model.named_parameters():
            try:
                if 'qkv_proj.weight' in name:
                    # Fused QKV weight
                    base = name.replace('qkv_proj.weight', '')
                    keys = [base + 'q_proj.weight', base + 'k_proj.weight', base + 'v_proj.weight']

                    if all(k in weight_locations for k in keys):
                        parts = []
                        for k in keys:
                            off, n, shape = weight_locations[k]
                            # Get bf16 view, convert to fp16
                            bf16_view = self.gpu_buffer_bf16[off:off+n].view(shape)
                            parts.append(bf16_view.to(torch.float16))

                        merged = torch.cat(parts, dim=0)
                        if merged.shape == param.shape:
                            param.data.copy_(merged)
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1

                elif 'qkv_proj.bias' in name:
                    # Fused QKV bias
                    base = name.replace('qkv_proj.bias', '')
                    keys = [base + 'q_proj.bias', base + 'k_proj.bias', base + 'v_proj.bias']

                    if all(k in weight_locations for k in keys):
                        parts = []
                        for k in keys:
                            off, n, shape = weight_locations[k]
                            bf16_view = self.gpu_buffer_bf16[off:off+n].view(shape)
                            parts.append(bf16_view.to(torch.float16))

                        merged = torch.cat(parts, dim=0)
                        if merged.shape == param.shape:
                            param.data.copy_(merged)
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1

                elif 'gate_up_proj.weight' in name:
                    # Fused gate+up
                    base = name.replace('gate_up_proj.weight', '')
                    keys = [base + 'gate_proj.weight', base + 'up_proj.weight']

                    if all(k in weight_locations for k in keys):
                        parts = []
                        for k in keys:
                            off, n, shape = weight_locations[k]
                            bf16_view = self.gpu_buffer_bf16[off:off+n].view(shape)
                            parts.append(bf16_view.to(torch.float16))

                        merged = torch.cat(parts, dim=0)
                        if merged.shape == param.shape:
                            param.data.copy_(merged)
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1

                else:
                    # Direct copy with conversion
                    if name in weight_locations:
                        off, n, shape = weight_locations[name]
                        bf16_view = self.gpu_buffer_bf16[off:off+n].view(shape)
                        fp16_tensor = bf16_view.to(torch.float16)

                        if fp16_tensor.shape == param.shape:
                            param.data.copy_(fp16_tensor)
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1

            except Exception as e:
                if self.verbose and skipped < 3:
                    print(f"[BlitzSwapV5] Error on {name}: {e}")
                skipped += 1

        torch.cuda.synchronize()
        inject_time = (time.perf_counter() - t1) * 1000

        total_time = (time.perf_counter() - t_total) * 1000

        if self.verbose:
            bandwidth = total_gb / (load_time / 1000)
            print(f"[BlitzSwapV5] Load: {load_time:.0f}ms ({bandwidth:.1f} GB/s)")
            print(f"[BlitzSwapV5] Inject+Convert: {inject_time:.0f}ms")
            print(f"[BlitzSwapV5] Total: {total_time:.0f}ms, {updated} updated, {skipped} skipped")

        return load_time, inject_time, total_time

    def initialize(self, model_name: str, **config) -> Any:
        """Initialize with a model (one-time cost)."""
        from vllm import LLM

        import blitz_vllm_patch
        blitz_vllm_patch.patch_vllm()

        self._ensure_buffers()

        default_config = {
            "dtype": "float16",
            "gpu_memory_utilization": 0.30,
            "max_model_len": 512,
            "max_num_batched_tokens": 512,
            "kv_cache_memory_bytes": 2 * 1024**3,
            "enforce_eager": True,
            "compilation_config": {"custom_ops": ["none"]},
        }
        default_config.update(config)

        if self.verbose:
            print(f"[BlitzSwapV5] Initializing with {model_name}...")

        t0 = time.perf_counter()
        self.llm = LLM(model=model_name, **default_config)
        init_time = (time.perf_counter() - t0) * 1000

        self.model = get_inner_model(self.llm)
        if self.model is not None:
            if self.verbose:
                params = list(self.model.named_parameters())
                print(f"[BlitzSwapV5] Found {type(self.model).__name__} with {len(params)} params")
        else:
            if self.verbose:
                print("[BlitzSwapV5] Warning: Could not access inner model")

        if self.verbose:
            print(f"[BlitzSwapV5] Init complete: {init_time:.0f}ms")

        return self.llm

    def swap_weights(self, model_name: str) -> Tuple[float, float, float]:
        """Swap weights using bulk bf16 load + GPU conversion."""
        if self.model is None:
            raise RuntimeError("Must call initialize() first")

        local_path = snapshot_download(model_name)
        safetensor_files = sorted(Path(local_path).glob("*.safetensors"))

        if self.verbose:
            print(f"[BlitzSwapV5] Swapping to {model_name}...")

        return self._swap_weights_bulk(safetensor_files)

    def generate(self, prompts: List[str], **kwargs) -> List[Any]:
        """Generate outputs."""
        from vllm import SamplingParams
        params = SamplingParams(**kwargs)
        return self.llm.generate(prompts, params)

    def cleanup(self):
        """Clean up resources."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            self.model = None

        gc.collect()
        torch.cuda.empty_cache()


def test_weight_swap_v5():
    """Test the hybrid bulk+GPU conversion swapper."""
    print("=" * 70)
    print("WEIGHT SWAP V5 TEST (Bulk bf16 + GPU conversion)")
    print("=" * 70)

    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    swapper = BlitzWeightSwapperV5(verbose=True)
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Initialize
    print("\n>>> Initializing (one-time)...")
    t0 = time.perf_counter()
    llm = swapper.initialize(model_name)
    init_time = (time.perf_counter() - t0) * 1000
    print(f"Init time: {init_time:.0f}ms")

    # Test output before swap
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=30, temperature=0.7)
    outputs = llm.generate(["The capital of France is"], params)
    print(f"Output 1 (before swap): {outputs[0].outputs[0].text[:50]}...")

    # Swap weights
    print("\n>>> First swap...")
    load_time, inject_time, total_time = swapper.swap_weights(model_name)

    outputs = llm.generate(["The capital of France is"], params)
    print(f"Output 2 (after swap): {outputs[0].outputs[0].text[:50]}...")

    # Second swap
    print("\n>>> Second swap (warmed)...")
    load_time2, inject_time2, total_time2 = swapper.swap_weights(model_name)

    outputs = llm.generate(["2 + 2 equals"], params)
    print(f"Output 3: {outputs[0].outputs[0].text[:50]}...")

    # Third swap
    print("\n>>> Third swap...")
    load_time3, inject_time3, total_time3 = swapper.swap_weights(model_name)

    outputs = llm.generate(["Hello, my name is"], params)
    print(f"Output 4: {outputs[0].outputs[0].text[:50]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Init time (one-time):    {init_time:.0f}ms")
    print(f"First swap:              {total_time:.0f}ms (load: {load_time:.0f}ms, inject: {inject_time:.0f}ms)")
    print(f"Second swap:             {total_time2:.0f}ms (load: {load_time2:.0f}ms, inject: {inject_time2:.0f}ms)")
    print(f"Third swap:              {total_time3:.0f}ms (load: {load_time3:.0f}ms, inject: {inject_time3:.0f}ms)")

    avg_swap = (total_time + total_time2 + total_time3) / 3
    print(f"Average swap:            {avg_swap:.0f}ms")

    target = 3000
    if avg_swap < target:
        print(f"\n[PASS] Sub-{target}ms weight swap achieved! ({avg_swap:.0f}ms avg)")
    else:
        print(f"\n[INFO] {avg_swap:.0f}ms average swap time (target: {target}ms)")

    swapper.cleanup()


if __name__ == '__main__':
    test_weight_swap_v5()
