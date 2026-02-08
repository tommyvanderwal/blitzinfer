#!/usr/bin/env python3
"""
BlitzInfer Model Switcher - Multi-model pre-loading approach.

Architecture:
- Pre-load multiple models into memory at startup
- Keep all models loaded simultaneously
- "Switching" is instant - just change which model handles requests
- No unload/reload cycle means no GPU context corruption

Trade-off: Uses more VRAM but achieves instant switching.
With 96GB unified memory, can hold 4-6 7B models (~14GB each).
"""

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

import gc
import time
import torch
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from vllm import LLM, SamplingParams


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    path: str
    dtype: str = "float16"
    max_model_len: int = 1024
    max_num_batched_tokens: int = 1024
    kv_cache_bytes: int = 2 * 1024 * 1024 * 1024  # Smaller per-model KV cache


class ModelSwitcherMultiLoad:
    """
    Multi-model pre-loading for instant switching.

    All registered models are loaded at startup and kept in memory.
    Switching between them is instant (just change active pointer).
    """

    def __init__(self, vram_limit_gb: float = 80.0, max_models: int = 4):
        """
        Args:
            vram_limit_gb: Total VRAM budget for all models
            max_models: Maximum number of models to keep loaded
        """
        self.vram_limit_gb = vram_limit_gb
        self.max_models = max_models
        self.configs: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, LLM] = {}
        self.active_model: Optional[str] = None

        # Calculate per-model GPU memory utilization
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        per_model_vram = vram_limit_gb / max_models
        self.gpu_memory_utilization = min(0.9, per_model_vram / total_vram)

        print(f"ModelSwitcherMultiLoad initialized:")
        print(f"  Total VRAM budget: {vram_limit_gb:.1f}GB")
        print(f"  Max models: {max_models}")
        print(f"  Per-model VRAM: {per_model_vram:.1f}GB")
        print(f"  GPU utilization per model: {self.gpu_memory_utilization:.2%}")
        print(f"  Total GPU memory: {total_vram:.1f}GB")

    def register_model(
        self,
        name: str,
        path: str,
        dtype: str = "float16",
        max_model_len: int = 1024,
        max_num_batched_tokens: int = 1024,
        kv_cache_bytes: Optional[int] = None,
    ) -> None:
        """Register a model for loading."""
        if kv_cache_bytes is None:
            # Smaller KV cache per model since we're loading multiple
            kv_cache_bytes = int(2 * 1024**3)  # 2GB per model

        config = ModelConfig(
            name=name,
            path=path,
            dtype=dtype,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_cache_bytes=kv_cache_bytes,
        )
        self.configs[name] = config
        print(f"Registered model: {name} ({path})")

    def load_all(self) -> Dict[str, float]:
        """
        Load all registered models into memory.
        Returns timing for each model load.
        """
        if len(self.configs) > self.max_models:
            raise ValueError(f"Too many models ({len(self.configs)}) for max_models ({self.max_models})")

        print(f"\n{'='*60}")
        print(f"Loading {len(self.configs)} models into memory...")
        print(f"{'='*60}")

        timings = {}
        total_start = time.perf_counter()

        for name, config in self.configs.items():
            print(f"\nLoading {name}...")
            load_start = time.perf_counter()

            llm = LLM(
                model=config.path,
                dtype=config.dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=config.max_model_len,
                max_num_batched_tokens=config.max_num_batched_tokens,
                kv_cache_memory_bytes=config.kv_cache_bytes,
                enforce_eager=True,
                compilation_config={"custom_ops": ["none"]},
            )

            load_time = time.perf_counter() - load_start
            self.loaded_models[name] = llm
            timings[name] = load_time
            print(f"  Loaded {name} in {load_time:.2f}s")

            # Report memory
            free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
            used_mem = torch.cuda.mem_get_info()[1] / (1024**3) - free_mem
            print(f"  GPU memory: {used_mem:.1f}GB used, {free_mem:.1f}GB free")

        total_time = time.perf_counter() - total_start
        print(f"\nAll models loaded in {total_time:.2f}s")

        # Set first model as active
        if self.loaded_models:
            self.active_model = list(self.loaded_models.keys())[0]
            print(f"Active model: {self.active_model}")

        return timings

    def activate(self, name: str) -> float:
        """
        Activate a model (instant - just changes pointer).
        Returns activation time (should be ~0).
        """
        if name not in self.loaded_models:
            if name in self.configs:
                raise RuntimeError(f"Model {name} registered but not loaded. Call load_all() first.")
            raise ValueError(f"Model {name} not registered")

        if self.active_model == name:
            print(f"Model {name} already active")
            return 0.0

        start = time.perf_counter()
        self.active_model = name
        switch_time = time.perf_counter() - start

        print(f"Switched to {name} in {switch_time*1000:.2f}ms (instant)")
        return switch_time

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 20,
        temperature: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Generate responses using the active model."""
        if self.active_model is None:
            raise RuntimeError("No model active")

        llm = self.loaded_models[self.active_model]
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

        start = time.perf_counter()
        outputs = llm.generate(prompts, params)
        gen_time = time.perf_counter() - start

        results = [
            {"text": o.outputs[0].text, "tokens": len(o.outputs[0].token_ids)}
            for o in outputs
        ]
        return results, gen_time

    def cleanup(self):
        """Clean up all resources."""
        print("Cleaning up all models...")
        for name in list(self.loaded_models.keys()):
            del self.loaded_models[name]
        self.loaded_models.clear()
        self.active_model = None
        gc.collect()
        torch.cuda.empty_cache()
        print("ModelSwitcherMultiLoad cleaned up")


def test_multiload_switching():
    """Test instant model switching with pre-loaded models."""
    print("="*70)
    print("MODEL SWITCHING TEST (Multi-Load / Instant Switch)")
    print("="*70)

    # Use 40GB for 2 models (20GB each) - conservative for testing
    switcher = ModelSwitcherMultiLoad(vram_limit_gb=40.0, max_models=2)

    # Register two different model "configurations" (same underlying model for testing)
    switcher.register_model(
        "qwen7b_config_a",
        "Qwen/Qwen2.5-7B-Instruct",
        max_model_len=1024,
        max_num_batched_tokens=1024,
    )
    switcher.register_model(
        "qwen7b_config_b",
        "Qwen/Qwen2.5-7B-Instruct",
        max_model_len=1024,
        max_num_batched_tokens=1024,
    )

    # Load all models upfront
    load_timings = switcher.load_all()

    results = []

    # Test 1: First generation (model A already active)
    print("\n" + "="*60)
    print("TEST 1: Generate with model A (already active)")
    print("="*60)
    out, gen_time = switcher.generate(["Hello, how are you?"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s: {out[0]['text']}")

    # Test 2: Switch to model B (instant)
    print("\n" + "="*60)
    print("TEST 2: Switch to model B (INSTANT)")
    print("="*60)
    switch_start = time.perf_counter()
    switch_time = switcher.activate("qwen7b_config_b")
    results.append(("Switch A->B", switch_time))

    out, gen_time = switcher.generate(["Tell me a joke"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s: {out[0]['text']}")

    # Test 3: Switch back to model A (instant)
    print("\n" + "="*60)
    print("TEST 3: Switch back to model A (INSTANT)")
    print("="*60)
    switch_time = switcher.activate("qwen7b_config_a")
    results.append(("Switch B->A", switch_time))

    out, gen_time = switcher.generate(["What is 2+2?"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s: {out[0]['text']}")

    # Test 4: Rapid switching
    print("\n" + "="*60)
    print("TEST 4: Rapid switching (10 switches)")
    print("="*60)
    rapid_start = time.perf_counter()
    for i in range(10):
        model = "qwen7b_config_a" if i % 2 == 0 else "qwen7b_config_b"
        switcher.activate(model)
    rapid_time = time.perf_counter() - rapid_start
    print(f"10 switches completed in {rapid_time*1000:.2f}ms ({rapid_time*100:.2f}ms per switch)")
    results.append(("10 rapid switches", rapid_time))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (Multi-Load / Instant Switch)")
    print("="*70)
    print(f"Initial load times:")
    for name, t in load_timings.items():
        print(f"  {name}: {t:.2f}s")
    print(f"\nSwitch times:")
    for name, t in results:
        print(f"  {name}: {t*1000:.2f}ms")
    print("="*70)

    switcher.cleanup()


if __name__ == '__main__':
    test_multiload_switching()
