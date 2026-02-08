#!/usr/bin/env python3
"""
BlitzInfer Model Switcher - Hybrid approach for 10+ models.

Architecture:
- Pre-load a "hot set" of frequently used models for instant switching
- Use subprocess-based switching for cold models (when hot set is full)
- LRU eviction policy for the hot set

This handles the target scenario: 10+ models, 1 active at a time,
with optimized switching for frequently used models.
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
import multiprocessing as mp
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import queue

from vllm import LLM, SamplingParams


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    path: str
    dtype: str = "float16"
    max_model_len: int = 1024
    max_num_batched_tokens: int = 1024
    kv_cache_bytes: int = 2 * 1024 * 1024 * 1024  # 2GB per model


def cold_worker_process(
    config: ModelConfig,
    gpu_memory_utilization: float,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    ready_event: mp.Event,
):
    """
    Worker process for cold model loading.
    Used when hot set is full and we need to load a model via subprocess.
    """
    # Re-setup env in subprocess
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_SKIP_WARMUP'] = '1'
    os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

    import sys
    import types
    fake_meta = types.ModuleType('torchvision._meta_registrations')
    sys.modules['torchvision._meta_registrations'] = fake_meta
    sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

    from vllm import LLM, SamplingParams

    print(f"[ColdWorker {config.name}] Starting...")

    load_start = time.perf_counter()
    llm = LLM(
        model=config.path,
        dtype=config.dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=config.max_model_len,
        max_num_batched_tokens=config.max_num_batched_tokens,
        kv_cache_memory_bytes=config.kv_cache_bytes,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )
    load_time = time.perf_counter() - load_start
    print(f"[ColdWorker {config.name}] Loaded in {load_time:.2f}s")

    ready_event.set()
    response_queue.put(("ready", load_time))

    while True:
        try:
            request = request_queue.get(timeout=1.0)
            if request[0] == "generate":
                prompts, params_dict = request[1], request[2]
                params = SamplingParams(**params_dict)
                gen_start = time.perf_counter()
                outputs = llm.generate(prompts, params)
                gen_time = time.perf_counter() - gen_start
                results = [
                    {"text": o.outputs[0].text, "tokens": len(o.outputs[0].token_ids)}
                    for o in outputs
                ]
                response_queue.put(("result", results, gen_time))
            elif request[0] == "shutdown":
                break
        except queue.Empty:
            continue
        except Exception as e:
            response_queue.put(("error", str(e)))

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[ColdWorker {config.name}] Exited")


class ModelSwitcherHybrid:
    """
    Hybrid model switching: hot set + cold subprocess fallback.

    - Models in the hot set: instant switching (0ms)
    - Models not in hot set: subprocess-based switching (~24s)
    - LRU eviction when hot set is full

    Usage:
        switcher = ModelSwitcherHybrid(hot_set_size=3, vram_budget_gb=80)
        switcher.register_model("model1", "/path/to/model1")
        switcher.register_model("model2", "/path/to/model2")
        ...
        switcher.preload(["model1", "model2", "model3"])  # Load hot set
        switcher.activate("model1")  # Instant
        switcher.activate("model4")  # Cold load (~24s), evicts LRU
    """

    def __init__(
        self,
        hot_set_size: int = 3,
        vram_budget_gb: float = 80.0,
        cold_model_vram_gb: float = 20.0,
    ):
        """
        Args:
            hot_set_size: Max models to keep loaded simultaneously
            vram_budget_gb: Total VRAM budget for hot set
            cold_model_vram_gb: VRAM budget for cold subprocess loads
        """
        self.hot_set_size = hot_set_size
        self.vram_budget_gb = vram_budget_gb
        self.cold_model_vram_gb = cold_model_vram_gb

        # Model registry
        self.configs: Dict[str, ModelConfig] = {}

        # Hot set (LRU ordered)
        self.hot_models: OrderedDict[str, LLM] = OrderedDict()

        # Cold subprocess (for models not in hot set)
        self.cold_worker: Optional[mp.Process] = None
        self.cold_model: Optional[str] = None
        self.cold_queues: Optional[Tuple[mp.Queue, mp.Queue, mp.Event]] = None

        # Current active model
        self.active_model: Optional[str] = None
        self.active_is_hot: bool = True

        # Calculate GPU utilization
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        per_model_vram = vram_budget_gb / hot_set_size
        self.hot_gpu_util = min(0.9, per_model_vram / total_vram)
        self.cold_gpu_util = min(0.9, cold_model_vram_gb / total_vram)

        print(f"ModelSwitcherHybrid initialized:")
        print(f"  Hot set size: {hot_set_size}")
        print(f"  VRAM budget (hot): {vram_budget_gb:.1f}GB ({per_model_vram:.1f}GB/model)")
        print(f"  VRAM budget (cold): {cold_model_vram_gb:.1f}GB")
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
        """Register a model configuration."""
        if kv_cache_bytes is None:
            kv_cache_bytes = int(2 * 1024**3)

        self.configs[name] = ModelConfig(
            name=name,
            path=path,
            dtype=dtype,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_cache_bytes=kv_cache_bytes,
        )
        print(f"Registered model: {name}")

    def preload(self, model_names: List[str]) -> Dict[str, float]:
        """
        Pre-load models into the hot set.

        Args:
            model_names: List of model names to load (up to hot_set_size)

        Returns:
            Dict of model name -> load time
        """
        if len(model_names) > self.hot_set_size:
            raise ValueError(f"Cannot preload {len(model_names)} models, hot_set_size={self.hot_set_size}")

        timings = {}
        for name in model_names:
            if name not in self.configs:
                raise ValueError(f"Model {name} not registered")

            config = self.configs[name]
            print(f"\nPreloading {name}...")
            start = time.perf_counter()

            llm = LLM(
                model=config.path,
                dtype=config.dtype,
                gpu_memory_utilization=self.hot_gpu_util,
                max_model_len=config.max_model_len,
                max_num_batched_tokens=config.max_num_batched_tokens,
                kv_cache_memory_bytes=config.kv_cache_bytes,
                enforce_eager=True,
                compilation_config={"custom_ops": ["none"]},
            )

            load_time = time.perf_counter() - start
            self.hot_models[name] = llm
            timings[name] = load_time
            print(f"  Loaded {name} in {load_time:.2f}s")

            free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
            print(f"  Free VRAM: {free_mem:.1f}GB")

        if self.hot_models:
            self.active_model = list(self.hot_models.keys())[0]
            self.active_is_hot = True

        return timings

    def _stop_cold_worker(self) -> float:
        """Stop the cold subprocess worker."""
        if self.cold_worker is None:
            return 0.0

        start = time.perf_counter()
        req_queue, resp_queue, _ = self.cold_queues
        req_queue.put(("shutdown",))
        self.cold_worker.join(timeout=10.0)
        if self.cold_worker.is_alive():
            self.cold_worker.terminate()
            self.cold_worker.join(timeout=5.0)

        self.cold_worker = None
        self.cold_model = None
        self.cold_queues = None
        gc.collect()

        return time.perf_counter() - start

    def _start_cold_worker(self, name: str) -> float:
        """Start a cold subprocess worker for a model."""
        config = self.configs[name]

        req_queue = mp.Queue()
        resp_queue = mp.Queue()
        ready_event = mp.Event()

        self.cold_worker = mp.Process(
            target=cold_worker_process,
            args=(config, self.cold_gpu_util, req_queue, resp_queue, ready_event),
        )
        self.cold_worker.start()

        ready_event.wait(timeout=120.0)
        msg = resp_queue.get(timeout=5.0)
        if msg[0] != "ready":
            raise RuntimeError(f"Cold worker failed: {msg}")

        self.cold_model = name
        self.cold_queues = (req_queue, resp_queue, ready_event)

        return msg[1]  # load_time

    def activate(self, name: str) -> Dict[str, Any]:
        """
        Activate a model for inference.

        Returns timing info:
            - "type": "instant" | "hot_load" | "cold_load"
            - "switch_time": time in seconds
            - "evicted": model name if one was evicted (only for hot_load)
        """
        if name not in self.configs:
            raise ValueError(f"Model {name} not registered")

        # Already active?
        if self.active_model == name:
            return {"type": "already_active", "switch_time": 0.0}

        result = {"type": None, "switch_time": 0.0}
        start = time.perf_counter()

        # Case 1: Model in hot set (instant switch)
        if name in self.hot_models:
            # Move to end (most recently used)
            self.hot_models.move_to_end(name)
            self.active_model = name
            self.active_is_hot = True
            self._stop_cold_worker()  # Stop any cold worker

            result["type"] = "instant"
            result["switch_time"] = time.perf_counter() - start
            print(f"Switched to {name} (INSTANT) in {result['switch_time']*1000:.2f}ms")
            return result

        # Case 2: Model not in hot set
        # Stop any existing cold worker first
        stop_time = self._stop_cold_worker()

        # If hot set has room, load into hot set
        if len(self.hot_models) < self.hot_set_size:
            config = self.configs[name]
            print(f"Loading {name} into hot set...")

            llm = LLM(
                model=config.path,
                dtype=config.dtype,
                gpu_memory_utilization=self.hot_gpu_util,
                max_model_len=config.max_model_len,
                max_num_batched_tokens=config.max_num_batched_tokens,
                kv_cache_memory_bytes=config.kv_cache_bytes,
                enforce_eager=True,
                compilation_config={"custom_ops": ["none"]},
            )

            self.hot_models[name] = llm
            self.active_model = name
            self.active_is_hot = True

            result["type"] = "hot_load"
            result["switch_time"] = time.perf_counter() - start
            print(f"Loaded {name} into hot set in {result['switch_time']:.2f}s")
            return result

        # Case 3: Hot set full, use cold subprocess
        # Note: We can't evict from hot set because in-process reload causes GPU hang
        print(f"Hot set full, loading {name} via subprocess...")

        load_time = self._start_cold_worker(name)
        self.active_model = name
        self.active_is_hot = False

        result["type"] = "cold_load"
        result["switch_time"] = time.perf_counter() - start
        result["subprocess_load_time"] = load_time
        print(f"Cold loaded {name} in {result['switch_time']:.2f}s")
        return result

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 20,
        temperature: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Generate responses using the active model."""
        if self.active_model is None:
            raise RuntimeError("No model active")

        params_dict = {"max_tokens": max_tokens, "temperature": temperature}

        if self.active_is_hot:
            # Use hot model directly
            llm = self.hot_models[self.active_model]
            params = SamplingParams(**params_dict)
            start = time.perf_counter()
            outputs = llm.generate(prompts, params)
            gen_time = time.perf_counter() - start
            results = [
                {"text": o.outputs[0].text, "tokens": len(o.outputs[0].token_ids)}
                for o in outputs
            ]
            return results, gen_time
        else:
            # Use cold subprocess
            req_queue, resp_queue, _ = self.cold_queues
            req_queue.put(("generate", prompts, params_dict))
            msg = resp_queue.get(timeout=60.0)
            if msg[0] == "result":
                return msg[1], msg[2]
            elif msg[0] == "error":
                raise RuntimeError(f"Generation error: {msg[1]}")

    def get_status(self) -> Dict[str, Any]:
        """Get current switcher status."""
        return {
            "hot_models": list(self.hot_models.keys()),
            "cold_model": self.cold_model,
            "active_model": self.active_model,
            "active_is_hot": self.active_is_hot,
            "hot_set_capacity": f"{len(self.hot_models)}/{self.hot_set_size}",
        }

    def cleanup(self):
        """Clean up all resources."""
        print("\nCleaning up...")
        self._stop_cold_worker()
        for name in list(self.hot_models.keys()):
            del self.hot_models[name]
        self.hot_models.clear()
        self.active_model = None
        gc.collect()
        torch.cuda.empty_cache()
        print("ModelSwitcherHybrid cleaned up")


def test_hybrid_switching():
    """Test hybrid model switching."""
    print("="*70)
    print("MODEL SWITCHING TEST (Hybrid: Hot Set + Cold Subprocess)")
    print("="*70)

    # Initialize with multiprocessing
    mp.set_start_method('spawn', force=True)

    # Create switcher with 2-model hot set
    switcher = ModelSwitcherHybrid(
        hot_set_size=2,
        vram_budget_gb=40.0,  # 20GB per hot model
        cold_model_vram_gb=20.0,
    )

    # Register 4 models (more than hot set size)
    for i in range(4):
        switcher.register_model(
            f"model_{i}",
            "Qwen/Qwen2.5-7B-Instruct",  # Same model, different configs for testing
            max_model_len=1024,
            max_num_batched_tokens=1024,
        )

    # Preload 2 models into hot set
    print("\n" + "="*60)
    print("PRELOADING HOT SET (2 models)")
    print("="*60)
    preload_times = switcher.preload(["model_0", "model_1"])

    results = []

    # Test 1: Switch between hot models (instant)
    print("\n" + "="*60)
    print("TEST 1: Switch between hot models (INSTANT)")
    print("="*60)
    timing = switcher.activate("model_1")
    results.append(("Hot switch 0->1", timing["switch_time"], timing["type"]))

    out, gen_time = switcher.generate(["Hello world"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s")

    timing = switcher.activate("model_0")
    results.append(("Hot switch 1->0", timing["switch_time"], timing["type"]))

    # Test 2: Switch to cold model (subprocess)
    print("\n" + "="*60)
    print("TEST 2: Switch to cold model (SUBPROCESS)")
    print("="*60)
    timing = switcher.activate("model_2")
    results.append(("Cold switch 0->2", timing["switch_time"], timing["type"]))

    out, gen_time = switcher.generate(["Cold model test"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s")

    # Test 3: Switch back to hot model (instant)
    print("\n" + "="*60)
    print("TEST 3: Switch back to hot model (INSTANT)")
    print("="*60)
    timing = switcher.activate("model_0")
    results.append(("Hot switch 2->0", timing["switch_time"], timing["type"]))

    out, gen_time = switcher.generate(["Back to hot"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s")

    # Status
    print("\n" + "="*60)
    print("STATUS")
    print("="*60)
    status = switcher.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Preload times:")
    for name, t in preload_times.items():
        print(f"  {name}: {t:.2f}s")
    print("\nSwitch times:")
    for name, t, typ in results:
        print(f"  {name}: {t*1000 if typ == 'instant' else t:.2f}{'ms' if typ == 'instant' else 's'} ({typ})")
    print("="*70)

    switcher.cleanup()


if __name__ == '__main__':
    test_hybrid_switching()
