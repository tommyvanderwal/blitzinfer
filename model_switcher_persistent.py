#!/usr/bin/env python3
"""
BlitzInfer Model Switcher - Persistent worker with model reloading.

Architecture:
- Single persistent worker subprocess
- Worker can load/unload/reload models without respawning
- Avoids the ~14s import overhead per switch
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
import queue


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


def persistent_worker(
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    ready_event: mp.Event,
):
    """
    Persistent worker that can load multiple models without respawning.
    Handles: load, unload, generate, shutdown commands.
    """
    # Import vLLM once
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_SKIP_WARMUP'] = '1'
    os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

    import gc
    import torch
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    print("[Persistent Worker] Started, vLLM imported")

    current_llm = None
    current_model = None

    # Signal ready
    ready_event.set()
    response_queue.put(("worker_ready",))

    while True:
        try:
            request = request_queue.get(timeout=1.0)
            cmd = request[0]

            if cmd == "load":
                config = request[1]
                print(f"[Worker] Loading model: {config.name}")

                # Unload current model first if any
                if current_llm is not None:
                    print(f"[Worker] Unloading {current_model}...")
                    unload_start = time.perf_counter()

                    del current_llm
                    current_llm = None
                    current_model = None

                    gc.collect()
                    gc.collect()

                    # Clean up vLLM state
                    try:
                        cleanup_dist_env_and_memory(shutdown_ray=False)
                    except Exception as e:
                        print(f"[Worker] Warning during cleanup: {e}")

                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    unload_time = time.perf_counter() - unload_start
                    print(f"[Worker] Unloaded in {unload_time:.2f}s")

                    # Let GPU settle
                    time.sleep(0.5)
                    gc.collect()
                    torch.cuda.empty_cache()

                # Load new model
                load_start = time.perf_counter()
                try:
                    current_llm = LLM(
                        model=config.path,
                        dtype=config.dtype,
                        gpu_memory_utilization=config.gpu_memory_utilization,
                        max_model_len=config.max_model_len,
                        max_num_batched_tokens=config.max_num_batched_tokens,
                        kv_cache_memory_bytes=config.kv_cache_bytes,
                        enforce_eager=True,
                        compilation_config={"custom_ops": ["none"]},
                    )
                    current_model = config.name
                    load_time = time.perf_counter() - load_start
                    print(f"[Worker] Loaded {config.name} in {load_time:.2f}s")
                    response_queue.put(("loaded", config.name, load_time))
                except Exception as e:
                    print(f"[Worker] Load FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    response_queue.put(("error", f"Load failed: {e}"))

            elif cmd == "generate":
                if current_llm is None:
                    response_queue.put(("error", "No model loaded"))
                    continue

                prompts, sampling_params_dict = request[1], request[2]
                params = SamplingParams(**sampling_params_dict)

                gen_start = time.perf_counter()
                try:
                    outputs = current_llm.generate(prompts, params)
                    gen_time = time.perf_counter() - gen_start

                    results = [
                        {"text": o.outputs[0].text, "tokens": len(o.outputs[0].token_ids)}
                        for o in outputs
                    ]
                    response_queue.put(("result", results, gen_time))
                except Exception as e:
                    print(f"[Worker] Generate FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    response_queue.put(("error", f"Generate failed: {e}"))

            elif cmd == "shutdown":
                print("[Worker] Shutting down...")
                if current_llm is not None:
                    del current_llm
                gc.collect()
                torch.cuda.empty_cache()
                break

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker] Error: {e}")
            import traceback
            traceback.print_exc()
            response_queue.put(("error", str(e)))

    print("[Worker] Exited cleanly")


class ModelSwitcherPersistent:
    """
    Fast model switching with a persistent worker.

    The worker stays alive across model switches, avoiding the
    ~14s import overhead per switch.
    """

    def __init__(self, vram_limit_gb: float = 20.0):
        self.vram_limit_gb = vram_limit_gb
        self.models: Dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None

        # Calculate GPU memory utilization
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        self.gpu_memory_utilization = min(0.9, vram_limit_gb / total_vram)

        # Start persistent worker
        mp.set_start_method('spawn', force=True)
        self.request_queue = mp.Queue()
        self.response_queue = mp.Queue()
        self.ready_event = mp.Event()

        self.worker = mp.Process(
            target=persistent_worker,
            args=(self.request_queue, self.response_queue, self.ready_event),
        )
        self.worker.start()

        # Wait for worker to be ready
        print("Waiting for persistent worker to start...")
        self.ready_event.wait(timeout=60.0)
        msg = self.response_queue.get(timeout=5.0)
        if msg[0] != "worker_ready":
            raise RuntimeError(f"Worker failed to start: {msg}")

        print(f"ModelSwitcherPersistent initialized:")
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

        # Send load command to worker
        config = self.models[name]
        self.request_queue.put(("load", config))

        # Wait for response
        try:
            msg = self.response_queue.get(timeout=120.0)
            if msg[0] == "loaded":
                self.active_model = msg[1]
                load_time = msg[2]
            elif msg[0] == "error":
                raise RuntimeError(f"Load failed: {msg[1]}")
            else:
                raise RuntimeError(f"Unexpected response: {msg}")
        except queue.Empty:
            raise RuntimeError("Load timeout")

        total_time = time.perf_counter() - total_start

        print(f"\nActivation complete:")
        print(f"  Load time:   {load_time:.2f}s")
        print(f"  Total:       {total_time:.2f}s")

        return {
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
        if self.active_model is None:
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
                raise RuntimeError(f"Generate error: {msg[1]}")
        except queue.Empty:
            raise RuntimeError("Generation timeout")

    def cleanup(self):
        """Clean up all resources."""
        self.request_queue.put(("shutdown",))
        self.worker.join(timeout=10.0)
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=5.0)
        print("ModelSwitcherPersistent cleaned up")


def test_persistent_switching():
    """Test model switching with persistent worker."""
    print("="*70)
    print("MODEL SWITCHING TEST (Persistent Worker)")
    print("="*70)

    switcher = ModelSwitcherPersistent(vram_limit_gb=20.0)

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

    # Test 2: Switch to second model (WITHIN SAME WORKER)
    print("\n" + "="*60)
    print("TEST 2: Switch to second model (PERSISTENT WORKER)")
    print("="*60)
    timing2 = switcher.activate("qwen7b_v2")
    results.append(("Switch to model 2", timing2["total"]))

    out, gen_time = switcher.generate(["Tell me a joke"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s: {out[0]['text']}")

    # Test 3: Switch back (WITHIN SAME WORKER)
    print("\n" + "="*60)
    print("TEST 3: Switch back to first model (PERSISTENT WORKER)")
    print("="*60)
    timing3 = switcher.activate("qwen7b")
    results.append(("Switch back", timing3["total"]))

    out, gen_time = switcher.generate(["What is 2+2?"], max_tokens=10)
    print(f"Generated in {gen_time:.2f}s: {out[0]['text']}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (Persistent Worker)")
    print("="*70)
    for name, t in results:
        print(f"  {name}: {t:.2f}s")
    print("="*70)

    switcher.cleanup()


if __name__ == '__main__':
    test_persistent_switching()
