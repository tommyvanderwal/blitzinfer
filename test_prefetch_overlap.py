#!/usr/bin/env python3
"""Test prefetch overlap: pre-warm next model while current model serves requests.

Flow:
1. Load Model A
2. Queue 5 requests for Model A (1000 in, 2000 out tokens)
3. Queue 5 requests for Model B → triggers page cache warming
4. Serve Model A batch (~10-15s inference)
5. Switch to Model B (should load faster due to warm page cache)
6. Repeat cycling between models

Measures:
- Weight loading time (separate from KV init)
- Page cache warming progress during inference
- Cold vs warm model load times
- Hardware utilization (GPU, disk I/O)
"""
import os
import sys
import gc
import time
import random
import string
import threading
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# Single-process mode with proper cleanup
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Models
MODEL_A = "openai/gpt-oss-120b"
MODEL_B = "Qwen/Qwen3-VL-32B-Thinking-FP8"

# Request config
NUM_REQUESTS_PER_BATCH = 5
INPUT_TOKENS = 1000
OUTPUT_TOKENS = 2000

# vLLM config for RTX PRO 6000 (95GB VRAM)
VLLM_CONFIG = {
    "dtype": "auto",  # Let vLLM choose based on model (FP8 for Qwen-VL, BF16 for gpt-oss)
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 16,
    "max_num_batched_tokens": 8192,
    "enforce_eager": True,
    "trust_remote_code": True,
}

NUM_CYCLES = 5  # Number of A→B→A cycles


@dataclass
class TimingStats:
    """Detailed timing statistics."""
    model_name: str
    # Load breakdown
    config_init_time: float = 0.0
    weight_download_time: float = 0.0
    weight_load_time: float = 0.0
    kv_cache_init_time: float = 0.0
    total_load_time: float = 0.0
    # Inference
    batch_inference_time: float = 0.0
    tokens_per_second: float = 0.0
    # Unload
    unload_time: float = 0.0
    # Memory
    gpu_free_before_load: float = 0.0
    gpu_free_after_load: float = 0.0
    gpu_free_after_unload: float = 0.0
    # Prefetch
    prefetch_overlap_time: float = 0.0  # Time prefetch ran during inference
    prefetch_complete_before_switch: bool = False
    page_cache_warm: bool = False


@dataclass
class PrefetchState:
    """Track prefetch progress."""
    model_name: str = ""
    started: bool = False
    completed: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    bytes_read: int = 0
    total_bytes: int = 0
    speed_gbps: float = 0.0


def nvidia_smi_free_gb() -> float:
    """Get free GPU memory from nvidia-smi."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def nvidia_smi_used_gb() -> float:
    """Get used GPU memory."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip()) / 1024


def get_disk_read_stats():
    """Get disk read stats from /proc/diskstats."""
    try:
        with open('/proc/diskstats', 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and parts[2] == 'nvme0n1':  # Adjust device name as needed
                    # Field 6 is sectors read, multiply by 512 for bytes
                    return int(parts[5]) * 512
    except Exception:
        pass
    return 0


def generate_random_prompt(target_tokens: int) -> str:
    """Generate a random prompt that won't hit prefix cache.

    Uses random text to ensure each prompt is unique.
    ~4 chars per token on average.
    """
    # Add timestamp and random seed to ensure uniqueness
    prefix = f"[{time.time():.6f}][{random.randint(0, 999999):06d}] "

    # Generate random words to fill the rest
    chars_needed = target_tokens * 4 - len(prefix)

    # Mix of random words and characters
    words = []
    while len(' '.join(words)) < chars_needed:
        word_len = random.randint(3, 12)
        word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
        words.append(word)

    text = prefix + ' '.join(words)
    return text[:target_tokens * 4]  # Approximate token count


class PageCacheWarmer:
    """Background page cache warming using file reads."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._state = PrefetchState()
        self._lock = threading.Lock()
        self._future = None

    def start_warming(self, model_name: str):
        """Start background page cache warming."""
        with self._lock:
            if self._state.started and not self._state.completed:
                logger.warning(f"Warming already in progress for {self._state.model_name}")
                return

            self._state = PrefetchState(model_name=model_name, started=True, start_time=time.time())

        self._future = self._executor.submit(self._warm_worker, model_name)
        logger.info(f"[PREFETCH] Started page cache warming for {model_name}")

    def _warm_worker(self, model_name: str):
        """Worker thread: read model files into page cache."""
        from pathlib import Path
        from huggingface_hub import snapshot_download

        try:
            # Get model path
            model_path = Path(snapshot_download(model_name, local_files_only=True))

            # Find safetensor files
            safetensor_files = sorted(model_path.glob('*.safetensors'))
            if not safetensor_files:
                logger.warning(f"[PREFETCH] No safetensor files found for {model_name}")
                return

            total_size = sum(f.stat().st_size for f in safetensor_files)
            with self._lock:
                self._state.total_bytes = total_size

            logger.info(f"[PREFETCH] Warming {len(safetensor_files)} files, {total_size / 1024**3:.2f}GB total")

            bytes_read = 0
            chunk_size = 64 * 1024 * 1024  # 64MB chunks

            for sf in safetensor_files:
                file_size = sf.stat().st_size
                logger.debug(f"[PREFETCH] Reading {sf.name} ({file_size / 1024**3:.2f}GB)")

                with open(sf, 'rb') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        bytes_read += len(chunk)

                        with self._lock:
                            self._state.bytes_read = bytes_read

            elapsed = time.time() - self._state.start_time
            speed = (bytes_read / 1024**3) / elapsed if elapsed > 0 else 0

            with self._lock:
                self._state.completed = True
                self._state.end_time = time.time()
                self._state.speed_gbps = speed

            logger.info(f"[PREFETCH] Complete: {bytes_read / 1024**3:.2f}GB in {elapsed:.2f}s ({speed:.1f} GB/s)")

        except Exception as e:
            logger.error(f"[PREFETCH] Error warming {model_name}: {e}")
            with self._lock:
                self._state.completed = True
                self._state.end_time = time.time()

    def get_state(self) -> PrefetchState:
        """Get current prefetch state."""
        with self._lock:
            return PrefetchState(
                model_name=self._state.model_name,
                started=self._state.started,
                completed=self._state.completed,
                start_time=self._state.start_time,
                end_time=self._state.end_time,
                bytes_read=self._state.bytes_read,
                total_bytes=self._state.total_bytes,
                speed_gbps=self._state.speed_gbps,
            )

    def is_complete(self) -> bool:
        """Check if warming is complete."""
        with self._lock:
            return self._state.completed

    def wait_for_complete(self, timeout: float = None) -> bool:
        """Wait for warming to complete."""
        if self._future:
            try:
                self._future.result(timeout=timeout)
                return True
            except Exception:
                return False
        return True

    def reset(self):
        """Reset state for next warming."""
        with self._lock:
            self._state = PrefetchState()


def load_model_with_timing(model_name: str, stats: TimingStats):
    """Load model with detailed timing breakdown."""
    from vllm import LLM

    stats.gpu_free_before_load = nvidia_smi_free_gb()
    logger.info(f"[LOAD] Starting load of {model_name}")
    logger.info(f"[LOAD] GPU free before: {stats.gpu_free_before_load:.1f}GB")

    total_start = time.time()

    # Phase 1: Config initialization (includes model resolution)
    config_start = time.time()
    logger.info(f"[LOAD] Phase 1: Config initialization...")

    # We can't easily separate config from weight loading in vLLM's current API
    # So we'll measure total and use vLLM's internal logs for breakdown

    # Create LLM with timing hooks
    llm = LLM(
        model=model_name,
        **VLLM_CONFIG,
    )

    total_end = time.time()
    stats.total_load_time = total_end - total_start

    stats.gpu_free_after_load = nvidia_smi_free_gb()
    gpu_used = stats.gpu_free_before_load - stats.gpu_free_after_load

    logger.info(f"[LOAD] Complete: {stats.total_load_time:.2f}s total")
    logger.info(f"[LOAD] GPU used: {gpu_used:.1f}GB (free: {stats.gpu_free_after_load:.1f}GB)")

    return llm


def unload_model_with_timing(llm, model_name: str, stats: TimingStats):
    """Unload model with proper cleanup."""
    import torch._dynamo

    logger.info(f"[UNLOAD] Starting unload of {model_name}")
    start = time.time()

    # Detailed cleanup
    try:
        engine_core = llm.llm_engine.engine_core
        if hasattr(engine_core, 'engine_core'):
            core = engine_core.engine_core
        else:
            core = engine_core

        if hasattr(core, 'model_executor'):
            executor = core.model_executor
            if hasattr(executor, 'driver_worker'):
                worker = executor.driver_worker
                if hasattr(worker, 'worker') and worker.worker is not None:
                    model_runner = getattr(worker.worker, 'model_runner', None)
                    if model_runner is not None:
                        # Clear model parameters
                        if hasattr(model_runner, 'model'):
                            model = model_runner.model
                            for param in model.parameters():
                                param.data = torch.empty(0, device='cpu')

                        # Clear KV caches
                        if hasattr(model_runner, 'kv_caches'):
                            for i, cache in enumerate(model_runner.kv_caches):
                                if cache is not None:
                                    model_runner.kv_caches[i] = torch.empty(0, device='cpu')
                            model_runner.kv_caches.clear()

                        # Clear cross attention KV cache (for VL models)
                        if hasattr(model_runner, 'cross_layers_kv_cache'):
                            if model_runner.cross_layers_kv_cache is not None:
                                model_runner.cross_layers_kv_cache = torch.empty(0, device='cpu')

                        # Clear attention layer KV caches
                        if hasattr(model_runner, 'compilation_config'):
                            sfc = getattr(model_runner.compilation_config, 'static_forward_context', None)
                            if sfc:
                                for layer_name, layer in sfc.items():
                                    if hasattr(layer, 'kv_cache') and layer.kv_cache:
                                        for i, kv in enumerate(layer.kv_cache):
                                            if kv is not None:
                                                layer.kv_cache[i] = torch.empty(0, device='cpu')
                                        layer.kv_cache = []

        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"[UNLOAD] Cleanup warning: {e}")

    del llm

    # Clear vLLM global state
    try:
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            vllm_config.compilation_config.static_forward_context.clear()
    except Exception:
        pass

    try:
        import vllm.config.vllm as vllm_config_module
        vllm_config_module._current_vllm_config = None
        vllm_config_module._current_prefix = None
        vllm_config_module.get_cached_compilation_config.cache_clear()
    except Exception:
        pass

    try:
        torch._dynamo.reset()
    except Exception:
        pass

    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except Exception:
        pass

    try:
        from vllm.v1.worker.workspace import reset_workspace_manager
        reset_workspace_manager()
    except Exception:
        pass

    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    stats.unload_time = time.time() - start
    stats.gpu_free_after_unload = nvidia_smi_free_gb()

    logger.info(f"[UNLOAD] Complete: {stats.unload_time:.2f}s")
    logger.info(f"[UNLOAD] GPU free after: {stats.gpu_free_after_unload:.1f}GB")


def run_batch_inference(llm, prompts: list[str], max_tokens: int, stats: TimingStats):
    """Run batch inference with timing."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
    )

    logger.info(f"[INFERENCE] Starting batch of {len(prompts)} requests, max_tokens={max_tokens}")

    total_input_tokens = 0
    start = time.time()

    outputs = llm.generate(prompts, sampling_params)

    elapsed = time.time() - start

    total_output_tokens = 0
    for output in outputs:
        total_input_tokens += len(output.prompt_token_ids)
        total_output_tokens += len(output.outputs[0].token_ids)

    tokens_per_second = total_output_tokens / elapsed if elapsed > 0 else 0

    stats.batch_inference_time = elapsed
    stats.tokens_per_second = tokens_per_second

    logger.info(f"[INFERENCE] Complete: {elapsed:.2f}s")
    logger.info(f"[INFERENCE] Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}")
    logger.info(f"[INFERENCE] Throughput: {tokens_per_second:.1f} tok/s")

    return outputs


def main():
    logger.info("=" * 70)
    logger.info("PREFETCH OVERLAP TEST")
    logger.info("=" * 70)
    logger.info(f"Model A: {MODEL_A}")
    logger.info(f"Model B: {MODEL_B}")
    logger.info(f"Requests per batch: {NUM_REQUESTS_PER_BATCH}")
    logger.info(f"Input tokens: {INPUT_TOKENS}, Output tokens: {OUTPUT_TOKENS}")
    logger.info(f"Cycles: {NUM_CYCLES}")
    logger.info(f"VLLM_ENABLE_V1_MULTIPROCESSING: {os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')}")
    logger.info(f"Initial GPU free: {nvidia_smi_free_gb():.1f}GB")
    logger.info("=" * 70)

    warmer = PageCacheWarmer()
    all_stats: list[TimingStats] = []

    models = [MODEL_A, MODEL_B]
    current_model_idx = 0

    # Pre-generate all prompts (random to avoid prefix caching)
    logger.info("Generating random prompts...")
    all_prompts = []
    for _ in range(NUM_CYCLES * 2 * NUM_REQUESTS_PER_BATCH):
        all_prompts.append(generate_random_prompt(INPUT_TOKENS))
    prompt_idx = 0
    logger.info(f"Generated {len(all_prompts)} prompts")

    for cycle in range(NUM_CYCLES):
        for model_switch in range(2):  # A then B
            current_model = models[current_model_idx]
            next_model = models[1 - current_model_idx]

            logger.info("")
            logger.info("#" * 70)
            logger.info(f"CYCLE {cycle + 1}/{NUM_CYCLES} - MODEL: {current_model.split('/')[-1]}")
            logger.info("#" * 70)

            stats = TimingStats(model_name=current_model)

            # Check if page cache is warm from previous prefetch
            prefetch_state = warmer.get_state()
            if prefetch_state.model_name == current_model and prefetch_state.completed:
                stats.page_cache_warm = True
                logger.info(f"[CACHE] Page cache is WARM for {current_model}")
            else:
                logger.info(f"[CACHE] Page cache is COLD for {current_model}")

            # Load model
            llm = load_model_with_timing(current_model, stats)

            # Get prompts for this batch
            batch_prompts = all_prompts[prompt_idx:prompt_idx + NUM_REQUESTS_PER_BATCH]
            prompt_idx += NUM_REQUESTS_PER_BATCH

            # Start prefetch for next model (unless last iteration)
            is_last = (cycle == NUM_CYCLES - 1 and model_switch == 1)
            if not is_last:
                warmer.reset()
                warmer.start_warming(next_model)
                prefetch_start = time.time()

            # Run inference
            inference_start = time.time()
            run_batch_inference(llm, batch_prompts, OUTPUT_TOKENS, stats)
            inference_end = time.time()

            # Check prefetch progress
            if not is_last:
                prefetch_state = warmer.get_state()
                if prefetch_state.completed:
                    stats.prefetch_complete_before_switch = True
                    overlap_time = min(prefetch_state.end_time, inference_end) - max(prefetch_start, inference_start)
                    stats.prefetch_overlap_time = max(0, overlap_time)
                    logger.info(f"[PREFETCH] Completed BEFORE switch! Overlap: {stats.prefetch_overlap_time:.2f}s")
                else:
                    progress = (prefetch_state.bytes_read / prefetch_state.total_bytes * 100) if prefetch_state.total_bytes > 0 else 0
                    logger.info(f"[PREFETCH] Still in progress: {progress:.1f}% ({prefetch_state.bytes_read / 1024**3:.2f}GB)")
                    # Wait for prefetch to complete
                    warmer.wait_for_complete(timeout=60)
                    final_state = warmer.get_state()
                    logger.info(f"[PREFETCH] Finished after inference at {final_state.speed_gbps:.1f} GB/s")

            # Unload model
            unload_model_with_timing(llm, current_model, stats)

            all_stats.append(stats)
            current_model_idx = 1 - current_model_idx

            # Log summary for this switch
            logger.info("")
            logger.info(f"--- SWITCH SUMMARY ---")
            logger.info(f"Model: {stats.model_name.split('/')[-1]}")
            logger.info(f"Page cache warm: {stats.page_cache_warm}")
            logger.info(f"Load time: {stats.total_load_time:.2f}s")
            logger.info(f"Inference time: {stats.batch_inference_time:.2f}s ({stats.tokens_per_second:.1f} tok/s)")
            logger.info(f"Unload time: {stats.unload_time:.2f}s")
            logger.info(f"Prefetch overlap: {stats.prefetch_overlap_time:.2f}s")
            logger.info(f"Prefetch done before switch: {stats.prefetch_complete_before_switch}")
            logger.info(f"GPU free after unload: {stats.gpu_free_after_unload:.1f}GB")

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    # Separate stats by model
    model_a_stats = [s for s in all_stats if MODEL_A in s.model_name]
    model_b_stats = [s for s in all_stats if MODEL_B in s.model_name]

    for model_stats, model_name in [(model_a_stats, MODEL_A), (model_b_stats, MODEL_B)]:
        if not model_stats:
            continue

        logger.info(f"\n{model_name.split('/')[-1]}:")

        cold_loads = [s for s in model_stats if not s.page_cache_warm]
        warm_loads = [s for s in model_stats if s.page_cache_warm]

        if cold_loads:
            avg_cold = sum(s.total_load_time for s in cold_loads) / len(cold_loads)
            logger.info(f"  Cold load time (n={len(cold_loads)}): {avg_cold:.2f}s")

        if warm_loads:
            avg_warm = sum(s.total_load_time for s in warm_loads) / len(warm_loads)
            logger.info(f"  Warm load time (n={len(warm_loads)}): {avg_warm:.2f}s")
            if cold_loads:
                speedup = avg_cold / avg_warm if avg_warm > 0 else 0
                logger.info(f"  Speedup from warm cache: {speedup:.2f}x")

        avg_inference = sum(s.batch_inference_time for s in model_stats) / len(model_stats)
        avg_tps = sum(s.tokens_per_second for s in model_stats) / len(model_stats)
        logger.info(f"  Avg inference time: {avg_inference:.2f}s")
        logger.info(f"  Avg throughput: {avg_tps:.1f} tok/s")

        prefetch_success = sum(1 for s in model_stats if s.prefetch_complete_before_switch)
        logger.info(f"  Prefetch completed before switch: {prefetch_success}/{len(model_stats)}")

    # Memory stability
    logger.info(f"\nMemory stability:")
    logger.info(f"  Initial GPU free: {nvidia_smi_free_gb():.1f}GB")
    for i, s in enumerate(all_stats):
        logger.info(f"  After switch {i+1}: {s.gpu_free_after_unload:.1f}GB")

    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST COMPLETE")
    logger.info("=" * 70)

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
