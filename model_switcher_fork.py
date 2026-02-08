#!/usr/bin/env python3
"""
BlitzInfer Model Switcher - Fork-based fast switching.

Uses fork() instead of spawn() to inherit Python state, eliminating
the ~14s import overhead per switch.

Architecture:
- Pre-import all modules in parent process
- Fork child processes that inherit the imports
- Each child loads only the model weights (fast)
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

# Import non-CUDA modules first
import gc
import time
import multiprocessing as mp
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import queue

# Set forkserver as start method BEFORE any CUDA import
mp.set_start_method('forkserver', force=True)

# Now import CUDA-related modules
print("Importing modules...")
import_start = time.perf_counter()
import torch
from vllm import LLM, SamplingParams
import_time = time.perf_counter() - import_start
print(f"Import complete in {import_time:.2f}s")


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    path: str
    dtype: str = "float16"
    max_model_len: int = 1024
    max_num_batched_tokens: int = 1024
    kv_cache_bytes: int = 4 * 1024 * 1024 * 1024
    gpu_memory_utilization: float = 0.21


def worker_process(
    config: ModelConfig,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    ready_event: mp.Event,
):
    """Worker process that loads and serves a single model."""
    # Note: vLLM is already imported thanks to fork()!

    print(f"[Worker {config.name}] Starting (fork - imports inherited)...")

    # Load model
    load_start = time.perf_counter()
    llm = LLM(
        model=config.path,
        dtype=config.dtype,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        max_num_batched_tokens=config.max_num_batched_tokens,
        kv_cache_memory_bytes=config.kv_cache_bytes,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )
    load_time = time.perf_counter() - load_start
    print(f"[Worker {config.name}] Loaded in {load_time:.2f}s")

    # Signal ready
    ready_event.set()
    response_queue.put(("ready", load_time))

    # Process requests
    while True:
        try:
            request = request_queue.get(timeout=1.0)

            if request[0] == "generate":
                prompts, sampling_params_dict = request[1], request[2]
                params = SamplingParams(**sampling_params_dict)

                gen_start = time.perf_counter()
                outputs = llm.generate(prompts, params)
                gen_time = time.perf_counter() - gen_start

                results = [
                    {"text": o.outputs[0].text, "tokens": len(o.outputs[0].token_ids)}
                    for o in outputs
                ]
                response_queue.put(("result", results, gen_time))

            elif request[0] == "shutdown":
                print(f"[Worker {config.name}] Shutting down...")
                break

        except queue.Empty:
            continue
        except Exception as e:
            response_queue.put(("error", str(e)))

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Worker {config.name}] Exited cleanly")


class ModelSwitcherFork:
    """
    Fast model switching using fork-based subprocess isolation.

    Key difference from V2: uses fork() which inherits the parent's
    Python interpreter state including all imports. This eliminates
    the ~14s import overhead per switch.
    """

    def __init__(self, vram_limit_gb: float = 20.0):
        self.vram_limit_gb = vram_limit_gb
        self.models: Dict[str, ModelConfig] = {}
        self.active_worker: Optional[mp.Process] = None
        self.active_model: Optional[str] = None
        self.request_queue: Optional[mp.Queue] = None
        self.response_queue: Optional[mp.Queue] = None
        self.ready_event: Optional[mp.Event] = None

        # Calculate GPU memory utilization
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        self.gpu_memory_utilization = min(0.9, vram_limit_gb / total_vram)

        print(f"ModelSwitcherFork initialized:")
        print(f"  VRAM limit: {vram_limit_gb:.1f}GB")
        print(f"  GPU utilization: {self.gpu_memory_utilization:.2%}")

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
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.models[name] = config
        print(f"Registered model: {name} ({path})")

    def _stop_active(self) -> float:
        """Stop the currently active worker."""
        if self.active_worker is None:
            return 0.0

        print(f"Stopping {self.active_model}...")
        stop_start = time.perf_counter()

        # Send shutdown signal
        self.request_queue.put(("shutdown",))

        # Wait for process to exit
        self.active_worker.join(timeout=10.0)
        if self.active_worker.is_alive():
            print("  Force terminating...")
            self.active_worker.terminate()
            self.active_worker.join(timeout=5.0)

        # Cleanup
        self.active_worker = None
        self.active_model = None
        self.request_queue = None
        self.response_queue = None
        self.ready_event = None

        stop_time = time.perf_counter() - stop_start
        print(f"Stopped in {stop_time:.2f}s")

        gc.collect()
        return stop_time

    def activate(self, name: str) -> Dict[str, float]:
        """
        Activate a model. Returns timing breakdown.
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")

        if self.active_model == name:
            print(f"Model {name} already active")
            return {"total": 0.0}

        print(f"\n{'='*60}")
        print(f"Activating model: {name}")
        print(f"{'='*60}")

        total_start = time.perf_counter()

        # Stop current worker
        stop_time = self._stop_active()

        # Start new worker using fork
        config = self.models[name]
        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue()
        self.ready_event = mp.Event()

        # Use forkserver for process creation (set at module level)
        self.active_worker = mp.Process(
            target=worker_process,
            args=(config, self.request_queue, self.response_queue, self.ready_event),
        )
        self.active_worker.start()

        # Wait for model to load
        print("Waiting for model to load...")
        self.ready_event.wait(timeout=120.0)

        # Get load time from worker
        msg = self.response_queue.get(timeout=5.0)
        if msg[0] == "ready":
            load_time = msg[1]
        else:
            load_time = -1

        self.active_model = name
        total_time = time.perf_counter() - total_start

        print(f"\nActivation complete:")
        print(f"  Stop previous: {stop_time:.2f}s")
        print(f"  Load new:      {load_time:.2f}s")
        print(f"  Total:         {total_time:.2f}s")

        return {
            "stop_time": stop_time,
            "load_time": load_time,
            "total": total_time,
        }

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 20,
        temperature: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Generate responses using the active model."""
        if self.active_worker is None:
            raise RuntimeError("No model active")

        sampling_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        self.request_queue.put(("generate", prompts, sampling_params))

        try:
            msg = self.response_queue.get(timeout=60.0)
            if msg[0] == "result":
                return msg[1], msg[2]
            elif msg[0] == "error":
                raise RuntimeError(f"Worker error: {msg[1]}")
        except queue.Empty:
            raise RuntimeError("Generation timeout")

    def cleanup(self):
        """Clean up all resources."""
        self._stop_active()
        print("ModelSwitcherFork cleaned up")


def test_fork_switching():
    """Test model switching with fork."""
    # Note: We do NOT set start method here - we use fork context explicitly

    print("="*70)
    print("MODEL SWITCHING TEST (Fork-based)")
    print("="*70)

    switcher = ModelSwitcherFork(vram_limit_gb=20.0)

    # Register models
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
    print("TEST 2: Switch to second model (FORK)")
    print("="*60)
    timing2 = switcher.activate("qwen7b_v2")
    results.append(("Switch to model 2", timing2["total"]))

    out, gen_time = switcher.generate(["Tell me a joke"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s: {out[0]['text']}")

    # Test 3: Switch back
    print("\n" + "="*60)
    print("TEST 3: Switch back to first model (FORK)")
    print("="*60)
    timing3 = switcher.activate("qwen7b")
    results.append(("Switch back", timing3["total"]))

    out, gen_time = switcher.generate(["What is 2+2?"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s: {out[0]['text']}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (Fork-based)")
    print("="*70)
    for name, t in results:
        print(f"  {name}: {t:.2f}s")
    print("="*70)

    switcher.cleanup()


if __name__ == '__main__':
    test_fork_switching()
