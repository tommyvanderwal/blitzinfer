#!/usr/bin/env python3
"""
BlitzInfer Weight Swap v2: Handle vLLM's fused parameters.

vLLM merges certain parameters for efficiency:
- q_proj, k_proj, v_proj -> qkv_proj
- gate_proj, up_proj -> gate_up_proj

We need to reconstruct fused weights from the source tensors.
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
        try:
            return llm.llm_engine.model_executor.driver_worker.model_runner.model
        except AttributeError:
            return None


class BlitzWeightSwapperV2:
    """
    Fast weight swapping with proper handling of vLLM's fused parameters.
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
        self.param_map: Dict[str, nn.Parameter] = {}
        self.model_config = None

    def _ensure_buffers(self):
        """Pre-allocate pinned CPU and GPU buffers."""
        if self.buffers_ready:
            return

        if self.verbose:
            print("[BlitzSwapV2] Pre-allocating buffers...")

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
            print(f"[BlitzSwapV2]   Pre-alloc time: {init_time:.0f}ms")

    def _load_weights_bulk(self, safetensor_files: List[Path]) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Load weights using optimized bulk transfer with pinned memory.
        Returns individual tensors as views into the GPU buffer.
        """
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

        # Phase 3: Create weight dict with views - BUT make clones so they're independent
        weights = {}
        for sf, tensors in file_meta:
            for key, off, n, shape, dtype in tensors:
                tensor = self.gpu_buffer[off:off+n].view(shape).clone()
                if dtype == torch.bfloat16:
                    tensor = tensor.view(torch.bfloat16)
                weights[key] = tensor

        if self.verbose:
            bandwidth = total_gb / (load_time / 1000)
            print(f"[BlitzSwapV2] Loaded {total_gb:.1f} GB in {load_time:.0f}ms ({bandwidth:.1f} GB/s)")

        return weights, load_time

    def _analyze_param_structure(self):
        """Analyze model parameter structure to understand fused params."""
        if self.model is None:
            return

        if self.verbose:
            print("\n[BlitzSwapV2] Analyzing model parameter structure...")

        # Get model params
        model_params = {}
        for name, param in self.model.named_parameters():
            model_params[name] = param.shape
            if self.verbose and len(model_params) <= 10:
                print(f"  Model param: {name}: {param.shape}")

        # Get weight file keys for comparison
        local_path = snapshot_download(self.current_model_name)
        safetensor_files = sorted(Path(local_path).glob("*.safetensors"))

        weight_keys = {}
        with safe_open(str(safetensor_files[0]), framework='pt', device='cpu') as f:
            for key in list(f.keys())[:20]:
                t = f.get_tensor(key)
                weight_keys[key] = t.shape

        if self.verbose:
            print("\n  Sample weight file keys:")
            for name, shape in list(weight_keys.items())[:10]:
                print(f"  Weight key: {name}: {shape}")

        return model_params, weight_keys

    def _build_fused_param_map(self):
        """
        Build mapping that handles fused parameters.

        vLLM fuses:
        - q_proj, k_proj, v_proj -> qkv_proj (concatenated along dim 0)
        - gate_proj, up_proj -> gate_up_proj (concatenated along dim 0)
        """
        if self.model is None:
            return

        self.param_map.clear()
        self.fused_groups = {}  # Map fused param name to source param names

        for name, param in self.model.named_parameters():
            self.param_map[name] = param

            # Detect fused params
            if 'qkv_proj.weight' in name:
                # This is a fused QKV weight
                base = name.replace('qkv_proj.weight', '')
                self.fused_groups[name] = {
                    'type': 'qkv_weight',
                    'sources': [
                        base + 'q_proj.weight',
                        base + 'k_proj.weight',
                        base + 'v_proj.weight',
                    ]
                }
            elif 'qkv_proj.bias' in name:
                # This is a fused QKV bias
                base = name.replace('qkv_proj.bias', '')
                self.fused_groups[name] = {
                    'type': 'qkv_bias',
                    'sources': [
                        base + 'q_proj.bias',
                        base + 'k_proj.bias',
                        base + 'v_proj.bias',
                    ]
                }
            elif 'gate_up_proj.weight' in name:
                # This is a fused gate+up weight
                base = name.replace('gate_up_proj.weight', '')
                self.fused_groups[name] = {
                    'type': 'gate_up',
                    'sources': [
                        base + 'gate_proj.weight',
                        base + 'up_proj.weight',
                    ]
                }

        if self.verbose:
            print(f"[BlitzSwapV2] Param map: {len(self.param_map)} params, {len(self.fused_groups)} fused groups")

    def _inject_weights_with_fusion(self, weights: Dict[str, torch.Tensor]) -> Tuple[int, int, float]:
        """
        Inject weights, handling fused parameters by merging source tensors.
        """
        if self.model is None:
            return 0, len(weights), 0.0

        t0 = time.perf_counter()
        updated = 0
        skipped = 0
        processed_sources = set()
        errors = []

        for name, param in self.param_map.items():
            # Check if this is a fused parameter
            if name in self.fused_groups:
                group = self.fused_groups[name]
                sources = group['sources']

                # Check if all sources are available
                if all(s in weights for s in sources):
                    try:
                        # Merge sources along dim 0
                        merged = torch.cat([weights[s] for s in sources], dim=0)
                        if merged.shape == param.shape:
                            param.data.copy_(merged)
                            updated += 1
                            processed_sources.update(sources)
                        else:
                            if len(errors) < 3:
                                errors.append(f"{name}: shape mismatch merged {merged.shape} vs param {param.shape}")
                            skipped += 1
                    except Exception as e:
                        if len(errors) < 3:
                            errors.append(f"{name}: {e}")
                        skipped += 1
                else:
                    missing = [s for s in sources if s not in weights]
                    if len(errors) < 3:
                        errors.append(f"{name}: missing sources {missing[:2]}")
                    skipped += 1

            else:
                # Direct parameter copy - try exact match first
                source_name = None

                # Try exact name
                if name in weights:
                    source_name = name
                # Try with 'model.' prefix
                elif 'model.' + name in weights:
                    source_name = 'model.' + name
                # Try without 'model.' prefix
                elif name.startswith('model.') and name[6:] in weights:
                    source_name = name[6:]

                if source_name and source_name not in processed_sources:
                    try:
                        if weights[source_name].shape == param.shape:
                            param.data.copy_(weights[source_name])
                            updated += 1
                            processed_sources.add(source_name)
                        else:
                            if len(errors) < 3:
                                errors.append(f"{name}: shape {weights[source_name].shape} vs {param.shape}")
                            skipped += 1
                    except Exception as e:
                        if len(errors) < 3:
                            errors.append(f"{name}: {e}")
                        skipped += 1
                elif source_name is None:
                    # Not a fused param and no direct match - this is expected for some params
                    skipped += 1

        torch.cuda.synchronize()
        inject_time = (time.perf_counter() - t0) * 1000

        if self.verbose:
            print(f"[BlitzSwapV2] Injected {updated} params, skipped {skipped}, time: {inject_time:.0f}ms")
            if errors:
                print(f"[BlitzSwapV2] Sample errors: {errors}")

        return updated, skipped, inject_time

    def initialize(self, model_name: str, **config) -> Any:
        """Initialize with a model (one-time cost)."""
        from vllm import LLM

        import blitz_vllm_patch
        blitz_vllm_patch.patch_vllm()

        self._ensure_buffers()
        self.current_model_name = model_name

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
            print(f"[BlitzSwapV2] Initializing with {model_name}...")

        t0 = time.perf_counter()
        self.llm = LLM(model=model_name, **default_config)
        init_time = (time.perf_counter() - t0) * 1000

        self.model = get_inner_model(self.llm)
        if self.model is not None:
            self._build_fused_param_map()
            if self.verbose:
                print(f"[BlitzSwapV2] Found model: {type(self.model).__name__}")
        else:
            if self.verbose:
                print("[BlitzSwapV2] Warning: Could not access inner model")

        if self.verbose:
            print(f"[BlitzSwapV2] Init complete: {init_time:.0f}ms")

        return self.llm

    def swap_weights(self, model_name: str) -> Tuple[float, float, float]:
        """Swap weights with proper fusion handling."""
        if self.model is None:
            raise RuntimeError("Must call initialize() first")

        t_total = time.perf_counter()

        local_path = snapshot_download(model_name)
        safetensor_files = sorted(Path(local_path).glob("*.safetensors"))

        if self.verbose:
            print(f"[BlitzSwapV2] Swapping to {model_name}...")

        # Load weights using fast bulk transfer
        weights, load_time = self._load_weights_bulk(safetensor_files)

        # Inject with fusion handling
        updated, skipped, inject_time = self._inject_weights_with_fusion(weights)

        # Cleanup weight dict
        del weights
        torch.cuda.empty_cache()

        total_time = (time.perf_counter() - t_total) * 1000

        if self.verbose:
            print(f"[BlitzSwapV2] Total swap: {total_time:.0f}ms")

        return load_time, inject_time, total_time

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
            self.param_map.clear()

        gc.collect()
        torch.cuda.empty_cache()


def analyze_model_structure():
    """Analyze the exact structure of model params vs weight file."""
    print("=" * 70)
    print("MODEL STRUCTURE ANALYSIS")
    print("=" * 70)

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    swapper = BlitzWeightSwapperV2(verbose=True)
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    print(f"\n>>> Loading model: {model_name}")
    swapper.initialize(model_name)

    # Analyze structure
    print("\n>>> Model parameters:")
    model_params = list(swapper.model.named_parameters())
    for name, param in model_params[:15]:
        print(f"  {name}: {param.shape}")

    print(f"\n  ... ({len(model_params)} total parameters)")

    # Get weight file structure
    print("\n>>> Weight file keys:")
    local_path = snapshot_download(model_name)
    safetensor_files = sorted(Path(local_path).glob("*.safetensors"))

    weight_keys = []
    for sf in safetensor_files[:1]:
        with safe_open(str(sf), framework='pt', device='cpu') as f:
            for key in f.keys():
                t = f.get_tensor(key)
                weight_keys.append((key, t.shape))

    for key, shape in weight_keys[:15]:
        print(f"  {key}: {shape}")

    print(f"\n  ... ({len(weight_keys)} keys in first file)")

    # Find mappings
    print("\n>>> Mapping analysis:")

    model_param_names = set(name for name, _ in model_params)
    weight_key_names = set(key for key, _ in weight_keys)

    # Direct matches
    direct = model_param_names.intersection(weight_key_names)
    print(f"  Direct matches: {len(direct)}")

    # Model params with 'qkv_proj'
    qkv = [n for n in model_param_names if 'qkv_proj' in n]
    print(f"  Model params with 'qkv_proj': {len(qkv)}")
    if qkv:
        print(f"    Example: {qkv[0]}")

    # Weight keys with q_proj, k_proj, v_proj
    q_proj = [k for k in weight_key_names if 'q_proj' in k]
    k_proj = [k for k in weight_key_names if 'k_proj' in k]
    v_proj = [k for k in weight_key_names if 'v_proj' in k]
    print(f"  Weight keys with q_proj: {len(q_proj)}")
    print(f"  Weight keys with k_proj: {len(k_proj)}")
    print(f"  Weight keys with v_proj: {len(v_proj)}")

    # Model params with 'gate_up_proj'
    gate_up = [n for n in model_param_names if 'gate_up_proj' in n]
    print(f"  Model params with 'gate_up_proj': {len(gate_up)}")

    # Weight keys with gate_proj, up_proj
    gate_proj = [k for k in weight_key_names if 'gate_proj' in k]
    up_proj = [k for k in weight_key_names if 'up_proj' in k]
    print(f"  Weight keys with gate_proj: {len(gate_proj)}")
    print(f"  Weight keys with up_proj: {len(up_proj)}")

    swapper.cleanup()


def test_weight_swap_v2():
    """Test the improved weight swapper."""
    print("=" * 70)
    print("WEIGHT SWAP V2 TEST")
    print("=" * 70)

    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    swapper = BlitzWeightSwapperV2(verbose=True)
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Initialize
    print("\n>>> Initializing...")
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
    print("\n>>> Swapping weights...")
    load_time, inject_time, total_time = swapper.swap_weights(model_name)

    print(f"\nSwap breakdown:")
    print(f"  Load:   {load_time:.0f}ms")
    print(f"  Inject: {inject_time:.0f}ms")
    print(f"  Total:  {total_time:.0f}ms")

    # Test output after swap
    outputs = llm.generate(["The capital of France is"], params)
    print(f"Output 2: {outputs[0].outputs[0].text[:50]}...")

    # Second swap
    print("\n>>> Second swap...")
    load_time2, inject_time2, total_time2 = swapper.swap_weights(model_name)

    print(f"\nSecond swap breakdown:")
    print(f"  Load:   {load_time2:.0f}ms")
    print(f"  Inject: {inject_time2:.0f}ms")
    print(f"  Total:  {total_time2:.0f}ms")

    outputs = llm.generate(["2 + 2 equals"], params)
    print(f"Output 3: {outputs[0].outputs[0].text[:50]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Init time:        {init_time:.0f}ms")
    print(f"First swap:       {total_time:.0f}ms")
    print(f"Second swap:      {total_time2:.0f}ms")

    swapper.cleanup()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze_model_structure()
    else:
        test_weight_swap_v2()
