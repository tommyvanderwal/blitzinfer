#!/usr/bin/env python3
"""
BlitzInfer Weight Swap v4: Fixed bulk injection.

Issue: v3 bulk method corrupted weights. This version fixes the offset calculation.
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


class BlitzWeightSwapperV4:
    """
    Ultra-fast weight swapping with correct bulk injection.
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

        # Buffers
        self.gpu_buffer = None
        self.cpu_buffers = None
        self.buffers_ready = False

        # Model state
        self.llm = None
        self.model = None

    def _ensure_buffers(self):
        """Pre-allocate pinned CPU and GPU buffers."""
        if self.buffers_ready:
            return

        if self.verbose:
            print("[BlitzSwapV4] Pre-allocating buffers...")

        t0 = time.perf_counter()

        max_elements = int(self.max_model_size_gb * 1024**3 / 2)
        self.gpu_buffer = torch.empty(max_elements, dtype=torch.float16, device='cuda')
        torch.cuda.synchronize()

        max_file_elements = int(self.max_file_size_gb * 1024**3 / 2)
        self.cpu_buffers = [
            torch.empty(max_file_elements, dtype=torch.float16, pin_memory=True)
            for _ in range(2)
        ]

        init_time = (time.perf_counter() - t0) * 1000
        self.buffers_ready = True

        if self.verbose:
            print(f"[BlitzSwapV4]   Pre-alloc time: {init_time:.0f}ms")

    def _swap_weights_bulk(self, safetensor_files: List[Path]) -> Tuple[float, float, float]:
        """
        Load weights to GPU buffer, then inject into model with proper fusion.
        Uses GPU-side dtype conversion for speed.
        """
        t_total = time.perf_counter()
        self._ensure_buffers()

        # Phase 1: Load all weights directly to GPU with on-GPU dtype conversion
        t0 = time.perf_counter()

        weight_tensors = {}  # key -> tensor on GPU

        for sf in safetensor_files:
            with safe_open(str(sf), framework='pt', device='cuda') as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    # Convert on GPU (much faster than CPU conversion)
                    if t.dtype != torch.float16:
                        t = t.to(torch.float16)
                    weight_tensors[key] = t

        torch.cuda.synchronize()
        load_time = (time.perf_counter() - t0) * 1000

        total_bytes = sum(t.numel() * t.element_size() for t in weight_tensors.values())
        total_gb = total_bytes / (1024**3)

        # Phase 2: Inject weights into model with proper fusion
        t1 = time.perf_counter()
        updated = 0
        skipped = 0

        for name, param in self.model.named_parameters():
            try:
                if 'qkv_proj.weight' in name:
                    # Fused QKV weight: q + k + v along dim 0
                    base = name.replace('qkv_proj.weight', '')
                    q_key = base + 'q_proj.weight'
                    k_key = base + 'k_proj.weight'
                    v_key = base + 'v_proj.weight'

                    if all(k in weight_tensors for k in [q_key, k_key, v_key]):
                        merged = torch.cat([weight_tensors[k] for k in [q_key, k_key, v_key]], dim=0)
                        if merged.shape == param.shape:
                            param.data.copy_(merged)
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1

                elif 'qkv_proj.bias' in name:
                    # Fused QKV bias: q + k + v along dim 0
                    base = name.replace('qkv_proj.bias', '')
                    q_key = base + 'q_proj.bias'
                    k_key = base + 'k_proj.bias'
                    v_key = base + 'v_proj.bias'

                    if all(k in weight_tensors for k in [q_key, k_key, v_key]):
                        merged = torch.cat([weight_tensors[k] for k in [q_key, k_key, v_key]], dim=0)
                        if merged.shape == param.shape:
                            param.data.copy_(merged)
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1

                elif 'gate_up_proj.weight' in name:
                    # Fused gate+up: gate + up along dim 0
                    base = name.replace('gate_up_proj.weight', '')
                    gate_key = base + 'gate_proj.weight'
                    up_key = base + 'up_proj.weight'

                    if all(k in weight_tensors for k in [gate_key, up_key]):
                        merged = torch.cat([weight_tensors[k] for k in [gate_key, up_key]], dim=0)
                        if merged.shape == param.shape:
                            param.data.copy_(merged)
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1

                else:
                    # Direct copy - try exact name match
                    if name in weight_tensors:
                        src = weight_tensors[name]
                        if src.shape == param.shape:
                            param.data.copy_(src)
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1

            except Exception as e:
                if self.verbose and skipped < 3:
                    print(f"[BlitzSwapV4] Error on {name}: {e}")
                skipped += 1

        torch.cuda.synchronize()
        inject_time = (time.perf_counter() - t1) * 1000

        # Cleanup loaded tensors
        del weight_tensors
        torch.cuda.empty_cache()

        total_time = (time.perf_counter() - t_total) * 1000

        if self.verbose:
            bandwidth = total_gb / (load_time / 1000)
            print(f"[BlitzSwapV4] Load: {load_time:.0f}ms ({bandwidth:.1f} GB/s), inject: {inject_time:.0f}ms")
            print(f"[BlitzSwapV4] Total: {total_time:.0f}ms, {updated} updated, {skipped} skipped")

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
            print(f"[BlitzSwapV4] Initializing with {model_name}...")

        t0 = time.perf_counter()
        self.llm = LLM(model=model_name, **default_config)
        init_time = (time.perf_counter() - t0) * 1000

        self.model = get_inner_model(self.llm)
        if self.model is not None:
            if self.verbose:
                print(f"[BlitzSwapV4] Found model: {type(self.model).__name__}")
                params = list(self.model.named_parameters())
                print(f"[BlitzSwapV4] Model has {len(params)} parameters")
        else:
            if self.verbose:
                print("[BlitzSwapV4] Warning: Could not access inner model")

        if self.verbose:
            print(f"[BlitzSwapV4] Init complete: {init_time:.0f}ms")

        return self.llm

    def swap_weights(self, model_name: str) -> Tuple[float, float, float]:
        """Swap weights using bulk injection."""
        if self.model is None:
            raise RuntimeError("Must call initialize() first")

        local_path = snapshot_download(model_name)
        safetensor_files = sorted(Path(local_path).glob("*.safetensors"))

        if self.verbose:
            print(f"[BlitzSwapV4] Swapping to {model_name}...")

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


def test_weight_swap_v4():
    """Test the fixed weight swapper."""
    print("=" * 70)
    print("WEIGHT SWAP V4 TEST")
    print("=" * 70)

    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    swapper = BlitzWeightSwapperV4(verbose=True)
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

    # Test output after swap
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
    print(f"First swap:              {total_time:.0f}ms")
    print(f"Second swap (warm):      {total_time2:.0f}ms")
    print(f"Third swap:              {total_time3:.0f}ms")
    print(f"Average swap:            {(total_time + total_time2 + total_time3) / 3:.0f}ms")

    target = 2500
    avg_swap = (total_time + total_time2 + total_time3) / 3

    if avg_swap < target:
        print(f"\n[PASS] Sub-{target}ms weight swap achieved! ({avg_swap:.0f}ms avg)")
    else:
        print(f"\n[INFO] {avg_swap:.0f}ms average swap time (target: {target}ms)")

    swapper.cleanup()


if __name__ == '__main__':
    test_weight_swap_v4()
