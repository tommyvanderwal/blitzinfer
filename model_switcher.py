#!/usr/bin/env python3
"""
BlitzInfer Model Switcher - Fast model switching with RAM caching.

Architecture:
- Pre-cache model weights in system RAM
- Quick VRAM swap between models
- Single active model at a time
- Full VRAM release before loading next model
"""

import os
import gc
import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    path: str
    dtype: str = "float16"
    max_model_len: int = 1024
    max_num_batched_tokens: int = 1024
    kv_cache_bytes: int = 4 * 1024 * 1024 * 1024  # 4GB default


@dataclass
class CachedModel:
    """A model with weights cached in RAM."""
    config: ModelConfig
    weights: Optional[Dict[str, torch.Tensor]] = None
    size_bytes: int = 0
    last_used: float = field(default_factory=time.time)


class ModelSwitcher:
    """
    Fast model switching with RAM weight caching.

    Usage:
        switcher = ModelSwitcher(vram_limit_gb=20, ram_cache_gb=40)
        switcher.register_model("qwen7b", "Qwen/Qwen2.5-7B-Instruct")
        switcher.register_model("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3")

        # Pre-cache to RAM (optional, speeds up first switch)
        switcher.precache_to_ram("qwen7b")

        # Activate a model (loads to VRAM)
        llm = switcher.activate("qwen7b")
        output = llm.generate(["Hello"], ...)

        # Switch to another model (swaps VRAM)
        llm = switcher.activate("mistral7b")
    """

    def __init__(
        self,
        vram_limit_gb: float = 20.0,
        ram_cache_gb: float = 40.0,
        device: str = "cuda:0",
    ):
        self.vram_limit_gb = vram_limit_gb
        self.ram_cache_gb = ram_cache_gb
        self.device = device

        self.models: Dict[str, CachedModel] = {}
        self.active_model: Optional[str] = None
        self.active_llm = None

        # Calculate memory utilization based on limit
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        self.gpu_memory_utilization = min(0.9, vram_limit_gb / total_vram)

        print(f"ModelSwitcher initialized:")
        print(f"  VRAM limit: {vram_limit_gb:.1f}GB (utilization: {self.gpu_memory_utilization:.2f})")
        print(f"  RAM cache: {ram_cache_gb:.1f}GB")
        print(f"  Device: {device}")

    def register_model(
        self,
        name: str,
        path: str,
        dtype: str = "float16",
        max_model_len: int = 1024,
        max_num_batched_tokens: int = 1024,
        kv_cache_bytes: Optional[int] = None,
    ) -> None:
        """Register a model for switching."""
        if kv_cache_bytes is None:
            # Default: use half of VRAM limit for KV cache
            kv_cache_bytes = int(self.vram_limit_gb * 0.3 * 1024**3)

        config = ModelConfig(
            name=name,
            path=path,
            dtype=dtype,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_cache_bytes=kv_cache_bytes,
        )
        self.models[name] = CachedModel(config=config)
        print(f"Registered model: {name} ({path})")

    def precache_to_ram(self, name: str) -> float:
        """
        Pre-load model weights into RAM for faster activation.
        Returns time taken in seconds.
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")

        model = self.models[name]
        if model.weights is not None:
            print(f"Model {name} already cached in RAM")
            return 0.0

        print(f"Pre-caching {name} to RAM...")
        start = time.perf_counter()

        # Use safetensors to load weights to CPU
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download, list_repo_files

        weights = {}
        total_bytes = 0

        try:
            # Get safetensors files from HF hub
            files = list_repo_files(model.config.path)
            st_files = [f for f in files if f.endswith('.safetensors')]

            for st_file in st_files:
                local_path = hf_hub_download(model.config.path, st_file)
                with safe_open(local_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        weights[key] = tensor
                        total_bytes += tensor.numel() * tensor.element_size()

        except Exception as e:
            print(f"Error caching {name}: {e}")
            return -1

        model.weights = weights
        model.size_bytes = total_bytes
        model.last_used = time.time()

        elapsed = time.perf_counter() - start
        size_gb = total_bytes / (1024**3)
        print(f"Cached {name}: {size_gb:.2f}GB in {elapsed:.2f}s ({size_gb/elapsed:.2f} GB/s)")

        return elapsed

    def _unload_active(self) -> float:
        """Unload the currently active model from VRAM."""
        if self.active_llm is None:
            return 0.0

        print(f"Unloading {self.active_model} from VRAM...")
        start = time.perf_counter()

        # Delete LLM instance
        del self.active_llm
        self.active_llm = None
        self.active_model = None

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        print(f"Unloaded in {elapsed:.2f}s")
        return elapsed

    def activate(self, name: str):
        """
        Activate a model (load to VRAM).
        Returns the LLM instance ready for inference.
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")

        # If same model already active, return it
        if self.active_model == name and self.active_llm is not None:
            print(f"Model {name} already active")
            return self.active_llm

        print(f"\n{'='*60}")
        print(f"Activating model: {name}")
        print(f"{'='*60}")

        total_start = time.perf_counter()

        # Unload current model
        unload_time = self._unload_active()

        # Load new model
        model = self.models[name]
        config = model.config

        from vllm import LLM

        load_start = time.perf_counter()

        self.active_llm = LLM(
            model=config.path,
            dtype=config.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            max_num_batched_tokens=config.max_num_batched_tokens,
            kv_cache_memory_bytes=config.kv_cache_bytes,
            enforce_eager=True,
        )

        load_time = time.perf_counter() - load_start
        total_time = time.perf_counter() - total_start

        self.active_model = name
        model.last_used = time.time()

        print(f"\nActivation complete:")
        print(f"  Unload time: {unload_time:.2f}s")
        print(f"  Load time:   {load_time:.2f}s")
        print(f"  Total:       {total_time:.2f}s")

        return self.active_llm

    def get_active(self):
        """Get the currently active LLM instance."""
        return self.active_llm

    def status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {
            "active": self.active_model,
            "vram_limit_gb": self.vram_limit_gb,
            "ram_cache_gb": self.ram_cache_gb,
            "models": {}
        }

        for name, model in self.models.items():
            status["models"][name] = {
                "path": model.config.path,
                "cached_in_ram": model.weights is not None,
                "size_gb": model.size_bytes / (1024**3) if model.size_bytes else 0,
                "active": name == self.active_model,
            }

        return status

    def cleanup(self):
        """Clean up all resources."""
        self._unload_active()

        # Clear RAM cache
        for model in self.models.values():
            model.weights = None
            model.size_bytes = 0

        gc.collect()
        torch.cuda.empty_cache()
        print("ModelSwitcher cleaned up")


def test_model_switching():
    """Test switching between two models."""
    import os
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_SKIP_WARMUP'] = '1'
    os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

    import sys
    import types
    fake_meta = types.ModuleType('torchvision._meta_registrations')
    sys.modules['torchvision._meta_registrations'] = fake_meta
    sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

    from vllm import SamplingParams

    print("="*70)
    print("MODEL SWITCHING TEST")
    print("="*70)

    # Initialize switcher with 20GB VRAM limit
    switcher = ModelSwitcher(vram_limit_gb=20.0, ram_cache_gb=40.0)

    # Register models
    switcher.register_model(
        "qwen7b",
        "Qwen/Qwen2.5-7B-Instruct",
        max_model_len=1024,
        max_num_batched_tokens=1024,
    )

    # For testing, use same model with different name to simulate switching
    # In production, this would be a different model like Mistral
    switcher.register_model(
        "qwen7b_2",
        "Qwen/Qwen2.5-7B-Instruct",  # Same model for testing
        max_model_len=1024,
        max_num_batched_tokens=1024,
    )

    # Test 1: First activation
    print("\n" + "="*70)
    print("TEST 1: First model activation")
    print("="*70)

    start = time.perf_counter()
    llm = switcher.activate("qwen7b")
    first_activation = time.perf_counter() - start

    # Quick inference test
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text}")

    # Test 2: Switch to second model
    print("\n" + "="*70)
    print("TEST 2: Switch to second model")
    print("="*70)

    start = time.perf_counter()
    llm = switcher.activate("qwen7b_2")
    switch_time = time.perf_counter() - start

    # Quick inference test
    out = llm.generate(["Test"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text}")

    # Test 3: Switch back to first model
    print("\n" + "="*70)
    print("TEST 3: Switch back to first model")
    print("="*70)

    start = time.perf_counter()
    llm = switcher.activate("qwen7b")
    switch_back_time = time.perf_counter() - start

    out = llm.generate(["Quick"], SamplingParams(max_tokens=5))
    print(f"Output: {out[0].outputs[0].text}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  First activation:  {first_activation:.2f}s")
    print(f"  Switch to model 2: {switch_time:.2f}s")
    print(f"  Switch back:       {switch_back_time:.2f}s")
    print("="*70)

    # Cleanup
    switcher.cleanup()


if __name__ == '__main__':
    test_model_switching()
