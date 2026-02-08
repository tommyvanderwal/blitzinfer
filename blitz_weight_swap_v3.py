#!/usr/bin/env python3
"""
BlitzInfer Weight Swap v3: Optimized fusion without cloning.

Key optimization: Fuse weights in-place during injection, avoiding extra clones.
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


class BlitzWeightSwapperV3:
    """
    Ultra-fast weight swapping with direct fusion injection.

    Target: <2.5s total swap time
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

        # Pre-computed injection plan
        self.injection_plan = []

    def _ensure_buffers(self):
        """Pre-allocate pinned CPU and GPU buffers."""
        if self.buffers_ready:
            return

        if self.verbose:
            print("[BlitzSwapV3] Pre-allocating buffers...")

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
            print(f"[BlitzSwapV3]   Pre-alloc time: {init_time:.0f}ms")

    def _build_injection_plan(self, safetensor_files: List[Path]):
        """
        Pre-compute the injection plan: map weight file locations to model params.

        This builds a plan that can be executed quickly during swap.
        """
        self.injection_plan = []

        # Get all weight keys and their positions
        weight_meta = {}  # key -> (file_idx, shape, file_offset_within_file)
        for file_idx, sf in enumerate(safetensor_files):
            with safe_open(str(sf), framework='pt', device='cpu') as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    weight_meta[key] = {
                        'file_idx': file_idx,
                        'shape': t.shape,
                        'numel': t.numel(),
                        'dtype': t.dtype
                    }

        # Build plan for each model parameter
        for name, param in self.model.named_parameters():
            if 'qkv_proj.weight' in name:
                # Fused QKV weight
                base = name.replace('qkv_proj.weight', '')
                sources = [base + 'q_proj.weight', base + 'k_proj.weight', base + 'v_proj.weight']
                if all(s in weight_meta for s in sources):
                    self.injection_plan.append({
                        'type': 'fused',
                        'target': param,
                        'sources': sources,
                        'dim': 0
                    })
            elif 'qkv_proj.bias' in name:
                # Fused QKV bias
                base = name.replace('qkv_proj.bias', '')
                sources = [base + 'q_proj.bias', base + 'k_proj.bias', base + 'v_proj.bias']
                if all(s in weight_meta for s in sources):
                    self.injection_plan.append({
                        'type': 'fused',
                        'target': param,
                        'sources': sources,
                        'dim': 0
                    })
            elif 'gate_up_proj.weight' in name:
                # Fused gate+up weight
                base = name.replace('gate_up_proj.weight', '')
                sources = [base + 'gate_proj.weight', base + 'up_proj.weight']
                if all(s in weight_meta for s in sources):
                    self.injection_plan.append({
                        'type': 'fused',
                        'target': param,
                        'sources': sources,
                        'dim': 0
                    })
            else:
                # Direct copy
                if name in weight_meta:
                    self.injection_plan.append({
                        'type': 'direct',
                        'target': param,
                        'source': name
                    })

        if self.verbose:
            fused = sum(1 for p in self.injection_plan if p['type'] == 'fused')
            direct = sum(1 for p in self.injection_plan if p['type'] == 'direct')
            print(f"[BlitzSwapV3] Injection plan: {len(self.injection_plan)} ops ({fused} fused, {direct} direct)")

    def _load_and_inject(self, safetensor_files: List[Path]) -> Tuple[float, float, float]:
        """
        Load weights and inject in a single pass.

        Returns (load_time, inject_time, total_time).
        """
        t_total = time.perf_counter()

        # Load all weights to CPU first (using safetensors mmap)
        t0 = time.perf_counter()
        all_weights = {}
        for sf in safetensor_files:
            with safe_open(str(sf), framework='pt', device='cpu') as f:
                for key in f.keys():
                    all_weights[key] = f.get_tensor(key)

        cpu_load_time = (time.perf_counter() - t0) * 1000

        # Execute injection plan
        t1 = time.perf_counter()
        updated = 0

        for plan in self.injection_plan:
            if plan['type'] == 'fused':
                # Concatenate sources and copy to GPU
                sources = [all_weights[s] for s in plan['sources']]
                # Convert bfloat16 to float16 if needed
                sources = [s.to(torch.float16) if s.dtype == torch.bfloat16 else s for s in sources]
                merged = torch.cat(sources, dim=plan['dim'])
                plan['target'].data.copy_(merged.to('cuda'))
                updated += 1

            elif plan['type'] == 'direct':
                src = all_weights[plan['source']]
                if src.dtype == torch.bfloat16:
                    src = src.to(torch.float16)
                plan['target'].data.copy_(src.to('cuda'))
                updated += 1

        torch.cuda.synchronize()
        inject_time = (time.perf_counter() - t1) * 1000

        total_time = (time.perf_counter() - t_total) * 1000
        total_gb = sum(t.numel() * 2 for t in all_weights.values()) / (1024**3)

        if self.verbose:
            print(f"[BlitzSwapV3] CPU load: {cpu_load_time:.0f}ms, inject: {inject_time:.0f}ms, total: {total_time:.0f}ms")
            print(f"[BlitzSwapV3] Updated {updated} params, {total_gb:.1f} GB at {total_gb/(total_time/1000):.1f} GB/s effective")

        return cpu_load_time, inject_time, total_time

    def _load_and_inject_bulk(self, safetensor_files: List[Path]) -> Tuple[float, float, float]:
        """
        Load weights using bulk transfer, then inject with fusion.

        This is faster for unified memory systems.
        """
        t_total = time.perf_counter()
        self._ensure_buffers()

        # Phase 1: Bulk load to GPU buffer
        file_meta = []
        total_offset = 0

        for sf in safetensor_files:
            tensors = []
            with safe_open(str(sf), framework='pt', device='cpu') as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    n = t.numel()
                    tensors.append((key, total_offset, n, t.shape, t.dtype))
                    total_offset += n
            file_meta.append((sf, tensors))

        total_gb = total_offset * 2 / (1024**3)

        # Double-buffered loading
        t0 = time.perf_counter()
        offset = 0
        current_buf = 0

        for i, (sf, tensors) in enumerate(file_meta):
            cpu_buf = self.cpu_buffers[current_buf]

            buf_offset = 0
            with safe_open(str(sf), framework='pt', device='cpu') as f:
                for key, _, n, _, _ in tensors:
                    t = f.get_tensor(key)
                    flat = t.reshape(-1)
                    if flat.dtype == torch.bfloat16:
                        flat = flat.view(torch.float16)
                    elif flat.dtype != torch.float16:
                        flat = flat.to(torch.float16)
                    cpu_buf[buf_offset:buf_offset+n].copy_(flat)
                    buf_offset += n

            if i > 0:
                torch.cuda.synchronize()

            file_size = sum(n for _, _, n, _, _ in tensors)
            self.gpu_buffer[offset:offset+file_size].copy_(
                cpu_buf[:file_size],
                non_blocking=True
            )

            offset += file_size
            current_buf = (current_buf + 1) % 2

        torch.cuda.synchronize()
        load_time = (time.perf_counter() - t0) * 1000

        # Build weight dict (views into buffer, no clone)
        weights = {}
        for sf, tensors in file_meta:
            for key, off, n, shape, dtype in tensors:
                weights[key] = (off, n, shape, dtype)

        # Phase 2: Execute injection plan using buffer views
        t1 = time.perf_counter()
        updated = 0

        for plan in self.injection_plan:
            if plan['type'] == 'fused':
                # Get views for all sources
                parts = []
                for src_key in plan['sources']:
                    off, n, shape, dtype = weights[src_key]
                    tensor = self.gpu_buffer[off:off+n].view(shape)
                    parts.append(tensor)

                # Concatenate and copy
                merged = torch.cat(parts, dim=plan['dim'])
                plan['target'].data.copy_(merged)
                updated += 1

            elif plan['type'] == 'direct':
                off, n, shape, dtype = weights[plan['source']]
                src_view = self.gpu_buffer[off:off+n].view(shape)
                plan['target'].data.copy_(src_view)
                updated += 1

        torch.cuda.synchronize()
        inject_time = (time.perf_counter() - t1) * 1000

        total_time = (time.perf_counter() - t_total) * 1000

        if self.verbose:
            bandwidth = total_gb / (load_time / 1000)
            print(f"[BlitzSwapV3] Load: {load_time:.0f}ms ({bandwidth:.1f} GB/s), inject: {inject_time:.0f}ms")
            print(f"[BlitzSwapV3] Total: {total_time:.0f}ms, {updated} params updated")

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
            print(f"[BlitzSwapV3] Initializing with {model_name}...")

        t0 = time.perf_counter()
        self.llm = LLM(model=model_name, **default_config)
        init_time = (time.perf_counter() - t0) * 1000

        self.model = get_inner_model(self.llm)
        if self.model is not None:
            # Build injection plan
            local_path = snapshot_download(model_name)
            safetensor_files = sorted(Path(local_path).glob("*.safetensors"))
            self._build_injection_plan(safetensor_files)

            if self.verbose:
                print(f"[BlitzSwapV3] Found model: {type(self.model).__name__}")
        else:
            if self.verbose:
                print("[BlitzSwapV3] Warning: Could not access inner model")

        if self.verbose:
            print(f"[BlitzSwapV3] Init complete: {init_time:.0f}ms")

        return self.llm

    def swap_weights(self, model_name: str, use_bulk: bool = True) -> Tuple[float, float, float]:
        """Swap weights using optimized injection."""
        if self.model is None:
            raise RuntimeError("Must call initialize() first")

        local_path = snapshot_download(model_name)
        safetensor_files = sorted(Path(local_path).glob("*.safetensors"))

        if self.verbose:
            print(f"[BlitzSwapV3] Swapping to {model_name}...")

        if use_bulk:
            return self._load_and_inject_bulk(safetensor_files)
        else:
            return self._load_and_inject(safetensor_files)

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
            self.injection_plan = []

        gc.collect()
        torch.cuda.empty_cache()


def test_weight_swap_v3():
    """Test the optimized weight swapper."""
    print("=" * 70)
    print("WEIGHT SWAP V3 TEST")
    print("=" * 70)

    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    swapper = BlitzWeightSwapperV3(verbose=True)
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Initialize
    print("\n>>> Initializing (one-time)...")
    t0 = time.perf_counter()
    llm = swapper.initialize(model_name)
    init_time = (time.perf_counter() - t0) * 1000
    print(f"Init time: {init_time:.0f}ms")

    # Test output
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=30, temperature=0.7)
    outputs = llm.generate(["Hello, my name is"], params)
    print(f"Output 1: {outputs[0].outputs[0].text[:50]}...")

    # Swap weights
    print("\n>>> First swap (bulk)...")
    load_time, inject_time, total_time = swapper.swap_weights(model_name, use_bulk=True)

    # Test output after swap
    outputs = llm.generate(["The capital of France is"], params)
    print(f"Output 2: {outputs[0].outputs[0].text[:50]}...")

    # Second swap
    print("\n>>> Second swap (bulk, warmed)...")
    load_time2, inject_time2, total_time2 = swapper.swap_weights(model_name, use_bulk=True)

    outputs = llm.generate(["2 + 2 equals"], params)
    print(f"Output 3: {outputs[0].outputs[0].text[:50]}...")

    # Third swap (non-bulk for comparison)
    print("\n>>> Third swap (direct/non-bulk)...")
    load_time3, inject_time3, total_time3 = swapper.swap_weights(model_name, use_bulk=False)

    outputs = llm.generate(["Hello world!"], params)
    print(f"Output 4: {outputs[0].outputs[0].text[:50]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Init time (one-time):     {init_time:.0f}ms")
    print(f"First swap (bulk):        {total_time:.0f}ms")
    print(f"Second swap (bulk,warm):  {total_time2:.0f}ms")
    print(f"Third swap (direct):      {total_time3:.0f}ms")

    if total_time2 < 2500:
        print(f"\n[PASS] Sub-2.5s weight swap achieved!")
    else:
        print(f"\n[INFO] {total_time2:.0f}ms swap time")

    swapper.cleanup()


if __name__ == '__main__':
    test_weight_swap_v3()
