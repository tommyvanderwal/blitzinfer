#!/usr/bin/env python3
"""
BlitzInfer Persistent Shell: Keep vLLM model shell alive, swap weights only.

Key insight from profiling:
- Model layer creation (__init__): ~2900ms for 59 layers
- Tokenizer loading: ~1300ms (loaded twice!)
- Weight transfer: ~1500ms (at 9.7 GB/s - already optimal)
- Config parsing: ~900ms

Strategy: Keep the vLLM engine and model shell alive between switches.
Only reload weights into existing parameters.

Target: <2.5s model switch (vs current ~5.2s)
"""

import gc
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

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


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    local_path: str
    param_count: int
    param_bytes: int


class BlitzPersistentShell:
    """
    Fast model switching by keeping the model shell alive.

    Instead of destroying and recreating the entire vLLM engine,
    we keep the model architecture and just swap the weights.
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

        # State
        self.llm = None
        self.current_model: Optional[ModelInfo] = None
        self.cached_tokenizers: Dict[str, Any] = {}

        # Pre-allocated buffers
        self.gpu_buffer = None
        self.cpu_buffers = None
        self._init_buffers()

    def _init_buffers(self):
        """Pre-allocate pinned CPU and GPU buffers."""
        if self.verbose:
            print("[BlitzPersistent] Pre-allocating buffers...")

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
        if self.verbose:
            print(f"[BlitzPersistent]   GPU buffer: {self.max_model_size_gb:.1f} GB")
            print(f"[BlitzPersistent]   CPU buffers: 2 x {self.max_file_size_gb:.1f} GB (pinned)")
            print(f"[BlitzPersistent]   Pre-alloc time: {init_time:.0f}ms")

    def _load_weights_bulk(self, safetensor_files: List[Path]) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Load weights using optimized bulk transfer.
        Returns (weights_dict, load_time_ms).
        """
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

        # Phase 3: Create weight dict
        weights = {}
        for sf, tensors in file_meta:
            for key, off, n, shape, dtype in tensors:
                tensor = self.gpu_buffer[off:off+n].view(shape)
                if dtype == torch.bfloat16:
                    tensor = tensor.view(torch.bfloat16)
                weights[key] = tensor

        if self.verbose:
            bandwidth = total_gb / (load_time / 1000)
            print(f"[BlitzPersistent] Loaded {total_gb:.1f} GB in {load_time:.0f}ms ({bandwidth:.1f} GB/s)")

        return weights, load_time

    def _inject_weights(self, model: nn.Module, weights: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        """
        Inject weights directly into model parameters.
        Returns (params_updated, params_skipped).
        """
        updated = 0
        skipped = 0

        # Build parameter name -> parameter mapping
        param_map = dict(model.named_parameters())

        for name, new_weight in weights.items():
            # vLLM uses slightly different naming, try variations
            param_name = name

            if param_name in param_map:
                param = param_map[param_name]
                if param.shape == new_weight.shape:
                    # Direct copy into existing parameter
                    param.data.copy_(new_weight)
                    updated += 1
                else:
                    skipped += 1
            else:
                # Try without "model." prefix
                alt_name = name.replace("model.", "", 1) if name.startswith("model.") else "model." + name
                if alt_name in param_map:
                    param = param_map[alt_name]
                    if param.shape == new_weight.shape:
                        param.data.copy_(new_weight)
                        updated += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1

        return updated, skipped

    def load_model(self, model_name: str, **config) -> Any:
        """
        Load a model. First load uses full vLLM init, subsequent loads reuse shell.
        """
        from vllm import LLM, SamplingParams

        t_total = time.perf_counter()

        # Get local path
        local_path = snapshot_download(model_name)
        safetensor_files = sorted(Path(local_path).glob("*.safetensors"))

        if self.llm is None:
            # First load - full vLLM initialization
            if self.verbose:
                print(f"[BlitzPersistent] First load: {model_name}")

            # Apply Blitz patch before loading
            import blitz_vllm_patch
            blitz_vllm_patch.patch_vllm()

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

            t0 = time.perf_counter()
            self.llm = LLM(model=model_name, **default_config)
            load_time = (time.perf_counter() - t0) * 1000

            if self.verbose:
                print(f"[BlitzPersistent] First load complete: {load_time:.0f}ms")

            self.current_model = ModelInfo(
                name=model_name,
                local_path=local_path,
                param_count=0,
                param_bytes=0
            )

            total_time = (time.perf_counter() - t_total) * 1000
            return self.llm

        else:
            # Fast switch - reuse shell, just swap weights
            if self.verbose:
                print(f"[BlitzPersistent] Fast switch to: {model_name}")

            # Load new weights
            t0 = time.perf_counter()
            weights, load_time = self._load_weights_bulk(safetensor_files)

            # Inject into existing model
            # Get the inner model from vLLM's engine
            model = self._get_inner_model()

            if model is not None:
                t1 = time.perf_counter()
                updated, skipped = self._inject_weights(model, weights)
                inject_time = (time.perf_counter() - t1) * 1000

                if self.verbose:
                    print(f"[BlitzPersistent] Injected {updated} params, skipped {skipped}, time: {inject_time:.0f}ms")
            else:
                if self.verbose:
                    print("[BlitzPersistent] Warning: Could not access inner model, falling back to full reload")
                # Fallback to full reload
                del self.llm
                gc.collect()
                torch.cuda.empty_cache()

                import blitz_vllm_patch
                blitz_vllm_patch.fast_cleanup()

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
                self.llm = LLM(model=model_name, **default_config)

            self.current_model = ModelInfo(
                name=model_name,
                local_path=local_path,
                param_count=len(weights),
                param_bytes=sum(w.numel() * w.element_size() for w in weights.values())
            )

            total_time = (time.perf_counter() - t_total) * 1000
            if self.verbose:
                print(f"[BlitzPersistent] Total switch time: {total_time:.0f}ms")

            return self.llm

    def _get_inner_model(self) -> Optional[nn.Module]:
        """Try to extract the inner nn.Module from vLLM's LLM wrapper."""
        try:
            # vLLM V1 structure
            if hasattr(self.llm, 'llm_engine'):
                engine = self.llm.llm_engine
                if hasattr(engine, 'model_executor'):
                    executor = engine.model_executor
                    if hasattr(executor, 'driver_worker'):
                        worker = executor.driver_worker
                        if hasattr(worker, 'model_runner'):
                            runner = worker.model_runner
                            if hasattr(runner, 'model'):
                                return runner.model

            # Try alternative path for V1
            if hasattr(self.llm, 'engine_core'):
                core = self.llm.engine_core
                if hasattr(core, 'model_executor'):
                    # ... similar traversal
                    pass

            return None
        except Exception as e:
            if self.verbose:
                print(f"[BlitzPersistent] Error accessing inner model: {e}")
            return None

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

        gc.collect()
        torch.cuda.empty_cache()

        import blitz_vllm_patch
        blitz_vllm_patch.fast_cleanup()


def test_weight_injection():
    """Test if we can inject weights into an existing model."""
    from vllm import LLM, SamplingParams

    print("=" * 70)
    print("WEIGHT INJECTION TEST")
    print("=" * 70)

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Load model normally
    import blitz_vllm_patch
    blitz_vllm_patch.patch_vllm()

    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.30,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 2 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    print("\n>>> Loading model...")
    t0 = time.perf_counter()
    llm = LLM(model=model_name, **config)
    load_time = (time.perf_counter() - t0) * 1000
    print(f"Load time: {load_time:.0f}ms")

    # Try to access the inner model
    print("\n>>> Exploring vLLM structure...")

    def explore_object(obj, path="llm", depth=0, max_depth=5):
        """Explore object structure to find the model."""
        if depth > max_depth:
            return

        indent = "  " * depth

        for attr in dir(obj):
            if attr.startswith('_'):
                continue

            try:
                val = getattr(obj, attr)
                val_type = type(val).__name__

                # Look for model-related attributes
                if any(x in attr.lower() for x in ['model', 'runner', 'worker', 'executor', 'engine']):
                    print(f"{indent}{path}.{attr}: {val_type}")

                    if isinstance(val, nn.Module):
                        print(f"{indent}  -> Found nn.Module!")
                        # Print first few parameters
                        params = list(val.named_parameters())
                        print(f"{indent}  -> {len(params)} parameters")
                        if params:
                            for name, p in params[:3]:
                                print(f"{indent}     {name}: {p.shape}")

                    elif hasattr(val, '__dict__'):
                        explore_object(val, f"{path}.{attr}", depth + 1, max_depth)
            except Exception:
                pass

    explore_object(llm)

    # Try direct access paths for vLLM V1
    print("\n>>> Testing access paths...")

    access_paths = [
        "llm.llm_engine",
        "llm.llm_engine.model_executor",
        "llm.llm_engine.model_executor.driver_worker",
        "llm.llm_engine.model_executor.driver_worker.model_runner",
        "llm.llm_engine.model_executor.driver_worker.model_runner.model",
    ]

    current = llm
    for i, path in enumerate(access_paths):
        attr = path.split('.')[-1]
        if hasattr(current, attr):
            current = getattr(current, attr)
            print(f"  [OK] {path}: {type(current).__name__}")

            if isinstance(current, nn.Module):
                params = list(current.named_parameters())
                print(f"       -> nn.Module with {len(params)} parameters!")
                break
        else:
            print(f"  [MISS] {path}")
            break

    # Test inference
    print("\n>>> Testing inference...")
    params = SamplingParams(max_tokens=20, temperature=0.7)
    outputs = llm.generate(["Hello, how are you?"], params)
    print(f"Output: {outputs[0].outputs[0].text[:50]}...")

    # Cleanup
    del llm
    gc.collect()
    blitz_vllm_patch.fast_cleanup()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


def test_persistent_shell():
    """Test the persistent shell approach."""
    print("=" * 70)
    print("PERSISTENT SHELL TEST")
    print("=" * 70)

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    shell = BlitzPersistentShell(verbose=True)

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # First load
    print("\n>>> First load (full init)...")
    t0 = time.perf_counter()
    llm = shell.load_model(model_name)
    first_load = (time.perf_counter() - t0) * 1000
    print(f"First load: {first_load:.0f}ms")

    # Test output
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=20, temperature=0.7)
    outputs = llm.generate(["Hello!"], params)
    print(f"Output 1: {outputs[0].outputs[0].text[:40]}...")

    # Second load (fast switch)
    print("\n>>> Second load (fast switch to same model)...")
    t0 = time.perf_counter()
    llm = shell.load_model(model_name)
    second_load = (time.perf_counter() - t0) * 1000
    print(f"Second load: {second_load:.0f}ms")

    # Test output again
    outputs = llm.generate(["World!"], params)
    print(f"Output 2: {outputs[0].outputs[0].text[:40]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"First load:  {first_load:.0f}ms")
    print(f"Second load: {second_load:.0f}ms")
    print(f"Speedup:     {first_load / second_load:.1f}x")

    shell.cleanup()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'inject':
        test_weight_injection()
    else:
        test_persistent_shell()
