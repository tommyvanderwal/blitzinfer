#!/usr/bin/env python3
"""Persistent vLLM engine that stays loaded to avoid startup overhead."""

import os
import time
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _get_mp_context():
    """Get a spawn multiprocessing context."""
    import multiprocessing as mp
    return mp.get_context('spawn')


@dataclass
class EngineRequest:
    """Request to the engine process."""
    action: str  # 'load', 'generate', 'unload', 'shutdown'
    model: Optional[str] = None
    prompts: Optional[list[str]] = None
    max_tokens: int = 256
    temperature: float = 0.7
    request_id: int = 0


@dataclass
class EngineResponse:
    """Response from the engine process."""
    success: bool
    request_id: int
    data: any = None  # Results or error message
    timing: float = 0.0


def _engine_worker(
    request_queue,  # mp.Queue
    response_queue,  # mp.Queue
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
):
    """Worker process that keeps vLLM loaded."""
    # Set environment before any imports
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    # Import vLLM in the worker process (this is the slow part we want to amortize)
    import_start = time.time()
    from vllm import LLM, SamplingParams
    import_time = time.time() - import_start
    logger.info(f"Worker: vLLM imported in {import_time:.2f}s")

    llm: Optional[LLM] = None
    current_model: Optional[str] = None

    while True:
        try:
            req = request_queue.get(timeout=60)
        except:
            continue

        start_time = time.time()

        if req.action == 'shutdown':
            response_queue.put(EngineResponse(True, req.request_id))
            break

        elif req.action == 'load':
            try:
                # Unload current model if different
                if llm is not None and current_model != req.model:
                    del llm
                    llm = None
                    import gc
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()

                if llm is None or current_model != req.model:
                    llm = LLM(
                        model=req.model,
                        gpu_memory_utilization=gpu_memory_utilization,
                        max_model_len=max_model_len,
                        max_num_seqs=max_num_seqs,
                        max_num_batched_tokens=max_num_batched_tokens,
                        enforce_eager=True,
                    )
                    current_model = req.model

                response_queue.put(EngineResponse(
                    True, req.request_id,
                    timing=time.time() - start_time
                ))
            except Exception as e:
                response_queue.put(EngineResponse(
                    False, req.request_id,
                    data=str(e),
                    timing=time.time() - start_time
                ))

        elif req.action == 'generate':
            try:
                if llm is None:
                    raise RuntimeError("No model loaded")

                outputs = llm.generate(
                    req.prompts,
                    SamplingParams(
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                    )
                )

                results = [o.outputs[0].text for o in outputs]
                response_queue.put(EngineResponse(
                    True, req.request_id,
                    data=results,
                    timing=time.time() - start_time
                ))
            except Exception as e:
                response_queue.put(EngineResponse(
                    False, req.request_id,
                    data=str(e),
                    timing=time.time() - start_time
                ))

        elif req.action == 'unload':
            if llm is not None:
                del llm
                llm = None
                current_model = None
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            response_queue.put(EngineResponse(
                True, req.request_id,
                timing=time.time() - start_time
            ))


class PersistentEngine:
    """Engine that keeps vLLM process alive for faster subsequent loads."""

    def __init__(
        self,
        gpu_memory_utilization: float = 0.20,
        max_model_len: int = 4096,
        max_num_seqs: int = 20,
        max_num_batched_tokens: int = 512,
    ):
        self.config = {
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_model_len': max_model_len,
            'max_num_seqs': max_num_seqs,
            'max_num_batched_tokens': max_num_batched_tokens,
        }
        self._ctx = _get_mp_context()
        self._request_queue = None
        self._response_queue = None
        self._process = None
        self._request_counter = 0
        self._current_model: Optional[str] = None
        self._started = False

    def start(self):
        """Start the persistent engine process."""
        if self._started:
            return

        logger.info("Starting persistent engine process...")
        start_time = time.time()

        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()

        self._process = self._ctx.Process(
            target=_engine_worker,
            args=(
                self._request_queue,
                self._response_queue,
                self.config['gpu_memory_utilization'],
                self.config['max_model_len'],
                self.config['max_num_seqs'],
                self.config['max_num_batched_tokens'],
            ),
            daemon=False,  # Non-daemon so vLLM can spawn its own children
        )
        self._process.start()
        self._started = True

        # Register cleanup on exit
        import atexit
        atexit.register(self.shutdown)

        # Wait for process to be ready (it will import vLLM)
        # This is the slow part, but only happens once
        elapsed = time.time() - start_time
        logger.info(f"Persistent engine process started in {elapsed:.2f}s")

    def load_model(self, model: str, timeout: float = 300) -> float:
        """Load a model, return load time."""
        if not self._started:
            self.start()

        self._request_counter += 1
        req = EngineRequest('load', model=model, request_id=self._request_counter)
        self._request_queue.put(req)

        resp = self._response_queue.get(timeout=timeout)
        if not resp.success:
            raise RuntimeError(f"Failed to load model: {resp.data}")

        self._current_model = model
        return resp.timing

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        timeout: float = 300,
    ) -> list[str]:
        """Generate text."""
        if not self._started:
            raise RuntimeError("Engine not started")

        self._request_counter += 1
        req = EngineRequest(
            'generate',
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            request_id=self._request_counter,
        )
        self._request_queue.put(req)

        resp = self._response_queue.get(timeout=timeout)
        if not resp.success:
            raise RuntimeError(f"Generation failed: {resp.data}")

        return resp.data

    def unload(self, timeout: float = 60):
        """Unload current model."""
        if not self._started:
            return

        self._request_counter += 1
        req = EngineRequest('unload', request_id=self._request_counter)
        self._request_queue.put(req)
        self._response_queue.get(timeout=timeout)
        self._current_model = None

    def shutdown(self):
        """Shutdown the engine process."""
        if not self._started:
            return

        self._request_counter += 1
        req = EngineRequest('shutdown', request_id=self._request_counter)
        self._request_queue.put(req)

        try:
            self._process.join(timeout=10)
        except:
            self._process.terminate()

        self._started = False
        self._current_model = None

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model

    @property
    def is_running(self) -> bool:
        return self._started and self._process is not None and self._process.is_alive()
