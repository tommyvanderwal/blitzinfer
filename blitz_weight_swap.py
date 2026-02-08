#!/usr/bin/env python3
"""
BlitzInfer Weight Swap: Fast in-place weight replacement for vLLM.

Approach:
1. Keep vLLM engine and model shell alive
2. Load new weights into GPU buffer (9.7 GB/s)
3. Swap weights directly into model parameters (in-place copy)

Target: <2s model switch
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
        # Try alternative path
        try:
            return llm.llm_engine.model_executor.driver_worker.model_runner.model
        except AttributeError:
            return None


class BlitzWeightSwapper:
    """
    Fast weight swapping for vLLM models.

    Keeps the model architecture alive and swaps weights in-place.
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

        # Pre-allocated buffers
        self.gpu_buffer = None
        self.cpu_buffers = None
        self.buffers_ready = False

        # Current model info
        self.llm = None
        self.model = None
        self.param_map: Dict[str, nn.Parameter] = {}

    def _ensure_buffers(self):
        """Pre-allocate pinned CPU and GPU buffers."""
        if self.buffers_ready:
            return

        if self.verbose:
            print("[BlitzSwap] Pre-allocating buffers...")

        t0 = time.perf_counter()

        # GPU buffer for model weights
        max_elements = int(self.max_model_size_gb * 1024**3 / 2)
        self.gpu_buffer = torch.empty(max_elements, dtype=torch.float16, device='cuda')
        torch.cuda.synchronize()

        # Pinned CPU buffers
        max_file_elements = int(self.max_file_size_gb * 1024**3 / 2)
        self.cpu_buffers = [
            torch.empty(max_file_elements, dtype=torch.float16, pin_memory=True)
            for _ in range(2)
        ]

        init_time = (time.perf_counter() - t0) * 1000
        self.buffers_ready = True

        if self.verbose:
            print(f"[BlitzSwap]   GPU buffer: {self.max_model_size_gb:.1f} GB")
            print(f"[BlitzSwap]   CPU buffers: 2 x {self.max_file_size_gb:.1f} GB (pinned)")
            print(f"[BlitzSwap]   Pre-alloc time: {init_time:.0f}ms")

    def _load_weights_bulk(self, safetensor_files: List[Path]) -> Tuple[Dict[str, torch.Tensor], float]:
        """Load weights using optimized bulk transfer."""
        self._ensure_buffers()

        # Phase 1: Collect metadata
        file_meta = []
        total_offset = 0

        for sf in safetensor_files:
            tensors = []
            with safe_open(str(sf), framework='pt', device='cpu') as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    n = t.numel()
                    shape = t.shape
                    dtype = t.dtype
                    tensors.append((key, total_offset, n, shape, dtype))
                    total_offset += n
            file_meta.append((sf, tensors))

        total_gb = total_offset * 2 / (1024**3)

        if self.verbose:
            print(f"[BlitzSwap] Loading {total_gb:.1f} GB weights...")

        # Phase 2: Double-buffered loading
        t0 = time.perf_counter()
        offset = 0
        current_buf = 0

        for i, (sf, tensors) in enumerate(file_meta):
            cpu_buf = self.cpu_buffers[current_buf]

            # Load to CPU buffer
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

            # Wait for previous async transfer
            if i > 0:
                torch.cuda.synchronize()

            # Start async transfer
            file_size = sum(n for _, _, n, _, _ in tensors)
            self.gpu_buffer[offset:offset+file_size].copy_(
                cpu_buf[:file_size],
                non_blocking=True
            )

            offset += file_size
            current_buf = (current_buf + 1) % 2

        torch.cuda.synchronize()
        load_time = (time.perf_counter() - t0) * 1000

        # Phase 3: Create weight dict with views
        weights = {}
        for sf, tensors in file_meta:
            for key, off, n, shape, dtype in tensors:
                tensor = self.gpu_buffer[off:off+n].view(shape)
                if dtype == torch.bfloat16:
                    tensor = tensor.view(torch.bfloat16)
                weights[key] = tensor

        if self.verbose:
            bandwidth = total_gb / (load_time / 1000)
            print(f"[BlitzSwap] Loaded in {load_time:.0f}ms ({bandwidth:.1f} GB/s)")

        return weights, load_time

    def _build_param_map(self):
        """Build mapping from weight file names to model parameters."""
        if self.model is None:
            return

        self.param_map.clear()

        for name, param in self.model.named_parameters():
            self.param_map[name] = param
            # Also add with "model." prefix if not present
            if not name.startswith("model."):
                self.param_map["model." + name] = param

        if self.verbose:
            print(f"[BlitzSwap] Built param map with {len(self.param_map)} entries")

    def _inject_weights(self, weights: Dict[str, torch.Tensor]) -> Tuple[int, int, float]:
        """
        Inject weights directly into model parameters.
        Returns (params_updated, params_skipped, time_ms).
        """
        if self.model is None or not self.param_map:
            return 0, len(weights), 0.0

        t0 = time.perf_counter()
        updated = 0
        skipped = 0
        skipped_names = []

        for name, new_weight in weights.items():
            if name in self.param_map:
                param = self.param_map[name]
                if param.shape == new_weight.shape:
                    # Direct in-place copy
                    param.data.copy_(new_weight)
                    updated += 1
                else:
                    skipped += 1
                    if len(skipped_names) < 5:
                        skipped_names.append(f"{name}: shape mismatch {param.shape} vs {new_weight.shape}")
            else:
                skipped += 1
                if len(skipped_names) < 5:
                    skipped_names.append(f"{name}: not in param map")

        torch.cuda.synchronize()
        inject_time = (time.perf_counter() - t0) * 1000

        if self.verbose:
            print(f"[BlitzSwap] Injected {updated} params, skipped {skipped}, time: {inject_time:.0f}ms")
            if skipped_names:
                print(f"[BlitzSwap] Sample skips: {skipped_names[:3]}")

        return updated, skipped, inject_time

    def initialize(self, model_name: str, **config) -> Any:
        """
        Initialize with a model. This does full vLLM init (one-time cost).
        """
        from vllm import LLM

        # Apply Blitz patch
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
            print(f"[BlitzSwap] Initializing with {model_name}...")

        t0 = time.perf_counter()
        self.llm = LLM(model=model_name, **default_config)
        init_time = (time.perf_counter() - t0) * 1000

        # Get inner model
        self.model = get_inner_model(self.llm)
        if self.model is not None:
            self._build_param_map()
            if self.verbose:
                print(f"[BlitzSwap] Found inner model: {type(self.model).__name__}")
        else:
            if self.verbose:
                print("[BlitzSwap] Warning: Could not access inner model")

        if self.verbose:
            print(f"[BlitzSwap] Init complete: {init_time:.0f}ms")

        return self.llm

    def swap_weights(self, model_name: str) -> Tuple[float, float, float]:
        """
        Swap weights to a different model (same architecture).

        Returns (load_time_ms, inject_time_ms, total_time_ms).
        """
        if self.model is None:
            raise RuntimeError("Must call initialize() first")

        t_total = time.perf_counter()

        # Get safetensor files
        local_path = snapshot_download(model_name)
        safetensor_files = sorted(Path(local_path).glob("*.safetensors"))

        if self.verbose:
            print(f"[BlitzSwap] Swapping to {model_name}...")

        # Load weights
        weights, load_time = self._load_weights_bulk(safetensor_files)

        # Inject into model
        updated, skipped, inject_time = self._inject_weights(weights)

        total_time = (time.perf_counter() - t_total) * 1000

        if self.verbose:
            print(f"[BlitzSwap] Total swap: {total_time:.0f}ms")

        return load_time, inject_time, total_time

    def generate(self, prompts: List[str], **kwargs) -> List[Any]:
        """Generate outputs using the current model."""
        from vllm import SamplingParams
        params = SamplingParams(**kwargs)
        return self.llm.generate(prompts, params)

    def cleanup(self):
        """Clean up resources."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            self.model = None
            self.param_map.clear()

        gc.collect()
        torch.cuda.empty_cache()


def test_weight_swap():
    """Test weight swapping functionality."""
    print("=" * 70)
    print("WEIGHT SWAP TEST")
    print("=" * 70)

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    swapper = BlitzWeightSwapper(verbose=True)

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

    # Swap weights (same model for testing)
    print("\n>>> Swapping weights (same model)...")
    load_time, inject_time, total_time = swapper.swap_weights(model_name)

    print(f"\nSwap breakdown:")
    print(f"  Load:   {load_time:.0f}ms")
    print(f"  Inject: {inject_time:.0f}ms")
    print(f"  Total:  {total_time:.0f}ms")

    # Test output after swap
    outputs = llm.generate(["The capital of France is"], params)
    print(f"Output 2: {outputs[0].outputs[0].text[:50]}...")

    # Second swap to verify consistency
    print("\n>>> Second swap (warmed up)...")
    load_time2, inject_time2, total_time2 = swapper.swap_weights(model_name)

    print(f"\nSecond swap breakdown:")
    print(f"  Load:   {load_time2:.0f}ms")
    print(f"  Inject: {inject_time2:.0f}ms")
    print(f"  Total:  {total_time2:.0f}ms")

    outputs = llm.generate(["2 + 2 = "], params)
    print(f"Output 3: {outputs[0].outputs[0].text[:50]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Init time (one-time):    {init_time:.0f}ms")
    print(f"First swap:              {total_time:.0f}ms")
    print(f"Second swap (warmed):    {total_time2:.0f}ms")
    print(f"Target: <2000ms")

    if total_time2 < 2000:
        print("\n[PASS] Sub-2 second weight swap achieved!")
    elif total_time2 < 3000:
        print(f"\n[GOOD] {total_time2:.0f}ms swap time")
    else:
        print(f"\n[INFO] {total_time2:.0f}ms - optimization needed")

    swapper.cleanup()


def test_different_models():
    """Test swapping between models of the same architecture."""
    print("=" * 70)
    print("CROSS-MODEL SWAP TEST")
    print("=" * 70)

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    swapper = BlitzWeightSwapper(verbose=True)

    # Models must have same architecture for weight swap
    model_a = "Qwen/Qwen2.5-7B-Instruct"
    model_b = "Qwen/Qwen2.5-7B"  # Base model, same architecture

    # Initialize with model A
    print(f"\n>>> Initializing with {model_a}...")
    t0 = time.perf_counter()
    llm = swapper.initialize(model_a)
    init_time = (time.perf_counter() - t0) * 1000
    print(f"Init time: {init_time:.0f}ms")

    # Test with model A
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=30, temperature=0.7)
    outputs = llm.generate(["Hello,"], params)
    print(f"Model A output: {outputs[0].outputs[0].text[:50]}...")

    # Swap to model B
    print(f"\n>>> Swapping to {model_b}...")
    try:
        load_time, inject_time, total_time = swapper.swap_weights(model_b)

        print(f"\nSwap breakdown:")
        print(f"  Load:   {load_time:.0f}ms")
        print(f"  Inject: {inject_time:.0f}ms")
        print(f"  Total:  {total_time:.0f}ms")

        # Test with model B
        outputs = llm.generate(["Hello,"], params)
        print(f"Model B output: {outputs[0].outputs[0].text[:50]}...")

        # Swap back to model A
        print(f"\n>>> Swapping back to {model_a}...")
        load_time2, inject_time2, total_time2 = swapper.swap_weights(model_a)

        print(f"\nSwap back breakdown:")
        print(f"  Load:   {load_time2:.0f}ms")
        print(f"  Inject: {inject_time2:.0f}ms")
        print(f"  Total:  {total_time2:.0f}ms")

        outputs = llm.generate(["Hello,"], params)
        print(f"Model A (restored) output: {outputs[0].outputs[0].text[:50]}...")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"A→B swap: {total_time:.0f}ms")
        print(f"B→A swap: {total_time2:.0f}ms")
        print(f"Average:  {(total_time + total_time2) / 2:.0f}ms")

    except Exception as e:
        print(f"\n[ERROR] Cross-model swap failed: {e}")
        import traceback
        traceback.print_exc()

    swapper.cleanup()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'cross':
        test_different_models()
    else:
        test_weight_swap()
