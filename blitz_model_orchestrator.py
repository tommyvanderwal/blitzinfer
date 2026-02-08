#!/usr/bin/env python3
"""
BlitzInfer Model Orchestrator: Unified model switching with automatic strategy selection.

Strategies:
1. Same-architecture switch: Fast weight swap (~2.7s)
2. Cross-architecture switch: Standard vLLM reload (~12s)

The orchestrator automatically detects whether models are compatible for weight swap
and uses the optimal strategy.
"""

import gc
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

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
from huggingface_hub import snapshot_download


class SwitchStrategy(Enum):
    WEIGHT_SWAP = "weight_swap"      # Same architecture, ~2.7s
    FULL_RELOAD = "full_reload"      # Different architecture, ~12s


@dataclass
class ModelInfo:
    name: str
    architecture: str
    num_layers: int
    hidden_size: int
    num_params: int


# Known model architectures
ARCHITECTURE_MAP = {
    "Qwen2ForCausalLM": "qwen2",
    "MistralForCausalLM": "mistral",
    "LlamaForCausalLM": "llama",
    "Phi3ForCausalLM": "phi3",
}


class BlitzModelOrchestrator:
    """
    Unified model orchestrator with automatic strategy selection.

    Performance:
    - Same architecture (weight swap): ~2.7s
    - Cross architecture (full reload): ~12s
    """

    def __init__(
        self,
        max_model_size_gb: float = 16.0,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.max_model_size_gb = max_model_size_gb

        # Current state
        self.llm = None
        self.model = None
        self.current_model_name = None
        self.current_architecture = None

        # Weight swap components (initialized on demand)
        self.weight_swapper = None

        # Default config
        self.default_config = {
            "dtype": "float16",
            "gpu_memory_utilization": 0.25,
            "max_model_len": 512,
            "max_num_batched_tokens": 512,
            "kv_cache_memory_bytes": 2 * 1024**3,
            "enforce_eager": True,
            "compilation_config": {"custom_ops": ["none"]},
        }

    def _get_architecture(self, model_name: str) -> str:
        """Get the architecture type for a model."""
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            arch = config.architectures[0] if config.architectures else "unknown"
            return ARCHITECTURE_MAP.get(arch, arch.lower())
        except Exception as e:
            if self.verbose:
                print(f"[Orchestrator] Warning: Could not get architecture for {model_name}: {e}")
            return "unknown"

    def _determine_strategy(self, target_model: str) -> SwitchStrategy:
        """Determine the best switching strategy."""
        if self.current_architecture is None:
            return SwitchStrategy.FULL_RELOAD

        target_arch = self._get_architecture(target_model)

        if target_arch == self.current_architecture:
            return SwitchStrategy.WEIGHT_SWAP
        else:
            return SwitchStrategy.FULL_RELOAD

    def _full_cleanup(self):
        """Full GPU cleanup."""
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    def _load_model_standard(self, model_name: str, config: Dict) -> Any:
        """Load model using standard vLLM (no Blitz patch)."""
        from vllm import LLM

        if self.verbose:
            print(f"[Orchestrator] Loading {model_name} (standard vLLM)...")

        t0 = time.perf_counter()
        llm = LLM(model=model_name, **config)
        load_time = (time.perf_counter() - t0) * 1000

        if self.verbose:
            print(f"[Orchestrator] Load time: {load_time:.0f}ms")

        return llm, load_time

    def _load_model_blitz(self, model_name: str, config: Dict) -> Any:
        """Load model using Blitz patch for fast loading."""
        import blitz_vllm_patch
        blitz_vllm_patch.patch_vllm()

        from vllm import LLM

        if self.verbose:
            print(f"[Orchestrator] Loading {model_name} (Blitz patch)...")

        t0 = time.perf_counter()
        llm = LLM(model=model_name, **config)
        load_time = (time.perf_counter() - t0) * 1000

        if self.verbose:
            print(f"[Orchestrator] Load time: {load_time:.0f}ms")

        return llm, load_time

    def _get_inner_model(self, llm) -> Optional[nn.Module]:
        """Extract the inner nn.Module from vLLM."""
        try:
            return llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        except AttributeError:
            return None

    def initialize(self, model_name: str, use_blitz: bool = True, **config) -> Any:
        """
        Initialize with a model.

        Args:
            model_name: HuggingFace model name
            use_blitz: Whether to use Blitz patch for initial load
            **config: Additional vLLM config options
        """
        merged_config = {**self.default_config, **config}

        self._full_cleanup()

        if use_blitz:
            self.llm, load_time = self._load_model_blitz(model_name, merged_config)
        else:
            self.llm, load_time = self._load_model_standard(model_name, merged_config)

        self.model = self._get_inner_model(self.llm)
        self.current_model_name = model_name
        self.current_architecture = self._get_architecture(model_name)

        if self.verbose:
            print(f"[Orchestrator] Architecture: {self.current_architecture}")

        return self.llm

    def switch_model(self, target_model: str, **config) -> Tuple[Any, float, SwitchStrategy]:
        """
        Switch to a different model using the optimal strategy.

        Returns:
            (llm, switch_time_ms, strategy_used)
        """
        if self.llm is None:
            raise RuntimeError("Must call initialize() first")

        merged_config = {**self.default_config, **config}
        strategy = self._determine_strategy(target_model)

        if self.verbose:
            print(f"\n[Orchestrator] Switching {self.current_model_name.split('/')[-1]} → {target_model.split('/')[-1]}")
            print(f"[Orchestrator] Strategy: {strategy.value}")

        t0 = time.perf_counter()

        if strategy == SwitchStrategy.WEIGHT_SWAP:
            switch_time = self._switch_weight_swap(target_model)
        else:
            switch_time = self._switch_full_reload(target_model, merged_config)

        total_time = (time.perf_counter() - t0) * 1000

        self.current_model_name = target_model
        self.current_architecture = self._get_architecture(target_model)

        if self.verbose:
            print(f"[Orchestrator] Switch complete: {total_time:.0f}ms")

        return self.llm, total_time, strategy

    def _switch_weight_swap(self, target_model: str) -> float:
        """Switch using fast weight swap (same architecture)."""
        # Lazy initialization of weight swapper
        if self.weight_swapper is None:
            from blitz_weight_swap_final import BlitzWeightSwapperFinal
            self.weight_swapper = BlitzWeightSwapperFinal(verbose=self.verbose)
            self.weight_swapper.llm = self.llm
            self.weight_swapper.model = self.model

            # Build fusion plan from first model
            local_path = snapshot_download(self.current_model_name)
            safetensor_files = sorted(Path(local_path).glob("*.safetensors"))
            self.weight_swapper._build_fusion_plan(safetensor_files)

        return self.weight_swapper.swap_weights(target_model)

    def _switch_full_reload(self, target_model: str, config: Dict) -> float:
        """Switch using full reload (cross architecture)."""
        # Clean up current model
        del self.llm
        self.llm = None
        self.model = None
        self.weight_swapper = None  # Reset swapper for new architecture

        self._full_cleanup()

        if self.verbose:
            free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
            print(f"[Orchestrator] Memory after cleanup: {free_mem:.1f} GB free")

        # Load new model (standard, no Blitz patch for cross-architecture)
        self.llm, load_time = self._load_model_standard(target_model, config)
        self.model = self._get_inner_model(self.llm)

        return load_time

    def generate(self, prompts: List[str], **kwargs) -> List[Any]:
        """Generate outputs from the current model."""
        from vllm import SamplingParams
        params = SamplingParams(**kwargs)
        return self.llm.generate(prompts, params)

    def get_current_model(self) -> Optional[str]:
        """Get the current model name."""
        return self.current_model_name

    def get_current_architecture(self) -> Optional[str]:
        """Get the current model architecture."""
        return self.current_architecture

    def cleanup(self):
        """Clean up all resources."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            self.model = None

        if self.weight_swapper is not None:
            self.weight_swapper.cleanup()
            self.weight_swapper = None

        self.current_model_name = None
        self.current_architecture = None

        self._full_cleanup()


def test_orchestrator():
    """Test the model orchestrator."""
    print("="*70)
    print("BLITZ MODEL ORCHESTRATOR TEST")
    print("="*70)

    # Warmup
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    orchestrator = BlitzModelOrchestrator(verbose=True)

    qwen_model = "Qwen/Qwen2.5-7B-Instruct"
    mistral_model = "mistralai/Mistral-7B-Instruct-v0.3"

    results = []

    # Initialize with Qwen
    print("\n" + "-"*70)
    print("INITIALIZING")
    print("-"*70)

    t0 = time.perf_counter()
    llm = orchestrator.initialize(qwen_model, use_blitz=False)  # No blitz for cross-arch compatibility
    init_time = (time.perf_counter() - t0) * 1000

    # Test output
    outputs = orchestrator.generate(["Capital of France?"], max_tokens=30, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    valid = "paris" in text.lower()
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:50]}...")
    results.append(("Initialize (Qwen)", init_time, valid))

    # Switch to Mistral (cross-architecture)
    print("\n" + "-"*70)
    print("SWITCH: Qwen → Mistral")
    print("-"*70)

    llm, switch_time, strategy = orchestrator.switch_model(mistral_model)

    outputs = orchestrator.generate(["What is 2+2?"], max_tokens=30, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    valid = "4" in text or "four" in text.lower()
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:50]}...")
    results.append((f"Qwen→Mistral ({strategy.value})", switch_time, valid))

    # Switch back to Qwen (cross-architecture)
    print("\n" + "-"*70)
    print("SWITCH: Mistral → Qwen")
    print("-"*70)

    llm, switch_time, strategy = orchestrator.switch_model(qwen_model)

    outputs = orchestrator.generate(["Count 1 to 3:"], max_tokens=30, temperature=0.7)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    valid = "1" in text and "2" in text and "3" in text
    print(f"Output [{['FAIL','OK'][valid]}]: {text[:50]}...")
    results.append((f"Mistral→Qwen ({strategy.value})", switch_time, valid))

    # Same-architecture switch (if we had another Qwen model)
    # This would use weight_swap strategy

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_valid = True
    for name, time_ms, valid in results:
        status = "OK" if valid else "FAIL"
        print(f"  {name}: {time_ms:.0f}ms [{status}]")
        all_valid = all_valid and valid

    # Calculate average switch time
    switch_times = [r[1] for r in results[1:]]  # Exclude init
    avg_switch = sum(switch_times) / len(switch_times) if switch_times else 0

    print(f"\n  Average switch time: {avg_switch:.0f}ms ({avg_switch/1000:.1f}s)")
    print(f"  All outputs valid: {all_valid}")

    orchestrator.cleanup()

    print("\n" + "="*70)
    if all_valid:
        print("[SUCCESS] Model orchestrator works!")
    else:
        print("[WARNING] Some tests failed")
    print("="*70)


if __name__ == '__main__':
    test_orchestrator()
