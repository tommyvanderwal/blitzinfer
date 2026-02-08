#!/usr/bin/env python3
"""
BlitzInfer Model Switcher - Single process, in-memory switching.

Architecture:
- Single process, no IPC overhead
- Claim one block of GPU memory
- Properly cleanup between model switches
- Use native PyTorch ops to avoid _C module issues
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
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    path: str
    dtype: str = "float16"
    max_model_len: int = 1024
    max_num_batched_tokens: int = 1024
    kv_cache_bytes: int = 4 * 1024 * 1024 * 1024


class ModelSwitcherInProc:
    """
    Single-process model switching.

    Uses one GPU memory block, properly cleans up between switches.
    """

    def __init__(self, vram_limit_gb: float = 20.0):
        self.vram_limit_gb = vram_limit_gb
        self.models: Dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None
        self.active_llm = None

        # Calculate GPU memory utilization
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        self.gpu_memory_utilization = min(0.9, vram_limit_gb / total_vram)

        print(f"ModelSwitcherInProc initialized:")
        print(f"  VRAM limit: {vram_limit_gb:.1f}GB")
        print(f"  GPU utilization: {self.gpu_memory_utilization:.2%}")
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
        """Register a model for switching."""
        if kv_cache_bytes is None:
            kv_cache_bytes = int(self.vram_limit_gb * 0.3 * 1024**3)

        config = ModelConfig(
            name=name,
            path=path,
            dtype=dtype,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_cache_bytes=kv_cache_bytes,
        )
        self.models[name] = config
        print(f"Registered model: {name} ({path})")

    def _unload_active(self) -> float:
        """Unload the currently active model from VRAM."""
        if self.active_llm is None:
            return 0.0

        print(f"Unloading {self.active_model} from VRAM...")
        start = time.perf_counter()

        # Delete main LLM reference first
        del self.active_llm
        self.active_llm = None
        self.active_model = None

        # Aggressive cleanup
        gc.collect()
        gc.collect()

        # Use vLLM's cleanup function which properly resets parallel state
        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            print("  Calling vLLM cleanup_dist_env_and_memory...")
            cleanup_dist_env_and_memory(shutdown_ray=False)
        except Exception as e:
            print(f"  Warning during vLLM cleanup: {e}")
            # Fallback: try manual cleanup
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
            except:
                pass

        # Clear GPU memory
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        gc.collect()

        elapsed = time.perf_counter() - start

        # Report memory state
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        print(f"  Unloaded in {elapsed:.2f}s, free VRAM: {free_mem:.1f}GB")

        return elapsed

    def activate(self, name: str) -> Dict[str, float]:
        """
        Activate a model (load to VRAM).
        Returns timing breakdown.
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")

        # If same model already active, return it
        if self.active_model == name and self.active_llm is not None:
            print(f"Model {name} already active")
            return {"total": 0.0}

        print(f"\n{'='*60}")
        print(f"Activating model: {name}")
        print(f"{'='*60}")

        total_start = time.perf_counter()

        # Unload current model
        unload_time = self._unload_active()

        # Longer delay to let GPU settle after cleanup
        time.sleep(0.5)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Load new model
        config = self.models[name]

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
            # Use native PyTorch ops to avoid _C module issues
            compilation_config={"custom_ops": ["none"]},
        )

        load_time = time.perf_counter() - load_start
        total_time = time.perf_counter() - total_start

        self.active_model = name

        print(f"\nActivation complete:")
        print(f"  Unload time: {unload_time:.2f}s")
        print(f"  Load time:   {load_time:.2f}s")
        print(f"  Total:       {total_time:.2f}s")

        return {
            "unload_time": unload_time,
            "load_time": load_time,
            "total": total_time,
        }

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 20,
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Generate responses using the active model."""
        if self.active_llm is None:
            raise RuntimeError("No model active")

        from vllm import SamplingParams
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

        start = time.perf_counter()
        outputs = self.active_llm.generate(prompts, params)
        gen_time = time.perf_counter() - start

        results = [
            {"text": o.outputs[0].text, "tokens": len(o.outputs[0].token_ids)}
            for o in outputs
        ]
        return results, gen_time

    def cleanup(self):
        """Clean up all resources."""
        self._unload_active()
        gc.collect()
        torch.cuda.empty_cache()
        print("ModelSwitcherInProc cleaned up")


def test_inproc_switching():
    """Test single-process model switching."""
    print("="*70)
    print("MODEL SWITCHING TEST (Single Process, In-Memory)")
    print("="*70)

    switcher = ModelSwitcherInProc(vram_limit_gb=20.0)

    # Register models (using same model twice for testing)
    switcher.register_model(
        "qwen7b",
        "Qwen/Qwen2.5-7B-Instruct",
        max_model_len=1024,
        max_num_batched_tokens=1024,
    )
    switcher.register_model(
        "qwen7b_v2",
        "Qwen/Qwen2.5-7B-Instruct",
        max_model_len=1024,
        max_num_batched_tokens=1024,
    )

    results = []

    # Test 1: First activation
    print("\n" + "="*60)
    print("TEST 1: First model activation")
    print("="*60)
    timing1 = switcher.activate("qwen7b")
    results.append(("First activation", timing1["total"]))

    # Generate
    out, gen_time = switcher.generate(["Hello, how are you?"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s: {out[0]['text']}")

    # Test 2: Switch to second model
    print("\n" + "="*60)
    print("TEST 2: Switch to second model (IN-PROCESS)")
    print("="*60)
    timing2 = switcher.activate("qwen7b_v2")
    results.append(("Switch to model 2", timing2["total"]))

    out, gen_time = switcher.generate(["Tell me a joke"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s: {out[0]['text']}")

    # Test 3: Switch back
    print("\n" + "="*60)
    print("TEST 3: Switch back to first model (IN-PROCESS)")
    print("="*60)
    timing3 = switcher.activate("qwen7b")
    results.append(("Switch back", timing3["total"]))

    out, gen_time = switcher.generate(["What is 2+2?"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s: {out[0]['text']}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (Single Process)")
    print("="*70)
    for name, t in results:
        print(f"  {name}: {t:.2f}s")
    print("="*70)

    switcher.cleanup()


if __name__ == '__main__':
    test_inproc_switching()
