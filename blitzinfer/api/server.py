"""OpenAI-compatible API server for BlitzInfer with per-model request queues.

Architecture:
    FastAPI (:8000)
      -> Per-model asyncio.Queue (ChatCompletionRequest objects)
      -> Queue processor (dequeue active model, forward to engine)
      -> SGLang Engine (in-process, replaces vLLM)

Queue state machine:
    SERVING   - Dequeuing from active model's queue, calling engine.generate()
    DRAINING  - Active queue empty, waiting for in_flight == 0
    SWITCHING - Shutting down engine, loading new model
"""

import asyncio
import gc
import json
import logging
import os
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# SGLang Engine (replaces vLLM)
# Import Engine directly to avoid namespace conflict with local sglang/ source directory
from sglang.srt.entrypoints.engine import Engine as SglEngine

# Harmony utils from SGLang (native support, replaces ~300 lines of manual code)
try:
    from sglang.srt.entrypoints.harmony_utils import (
        get_system_message,
        get_developer_message,
        render_for_completion,
        get_stop_tokens_for_assistant_actions,
        parse_output_into_messages,
        parse_output_message,
        parse_chat_input,
        parse_response_input,
        parse_response_output,
    )
    from openai.types.responses import ResponseFunctionToolCall
    HAS_HARMONY = True
    HARMONY_STOP_TOKENS = list(get_stop_tokens_for_assistant_actions())
except ImportError:
    HAS_HARMONY = False
    HARMONY_STOP_TOKENS = []

# BlitzInfer imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from blitzinfer.orchestrator.standby_manager import StandbyManager


# =============================================================================
# Logging
# =============================================================================

class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()
        if self.stream:
            os.fsync(self.stream.fileno())

class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

CRASH_LOG_FILE = os.path.expanduser('~/blitzinfer_crash.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-30s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        FlushingStreamHandler(sys.stdout),
        FlushingFileHandler(CRASH_LOG_FILE, mode='a'),
    ]
)
logger = logging.getLogger('blitzinfer.api')

logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


def crash_log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] CRASH_LOG: {msg}\n"
    with open(CRASH_LOG_FILE, 'a') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    print(line, end='', flush=True)


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelInfo:
    name: str
    hf_path: str
    context_length: Optional[int] = None  # None = auto-detect from model config
    mem_fraction_static: float = 0.88
    supports_vision: bool = False
    quantization: Optional[str] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)


GPU_MEM_FRAC = 0.94

AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    "gpt-oss-120b": ModelInfo(
        name="gpt-oss-120b",
        hf_path="openai/gpt-oss-120b",
        context_length=131072,  # GPT-OSS supports 128K natively
        mem_fraction_static=GPU_MEM_FRAC,
    ),
    # NOTE: Qwen3-VL-32B produces garbage on SM120/Blackwell desktop GPUs
    # (MROPE position embedding bug). Use kimi-vl for vision instead.
    # Keeping entry commented out until SGLang fixes MROPE for SM120.
    # "qwen3-vl-32b-thinking": ModelInfo(
    #     name="qwen3-vl-32b-thinking",
    #     hf_path="Qwen/Qwen3-VL-32B-Thinking-FP8",
    #     mem_fraction_static=GPU_MEM_FRAC,
    #     supports_vision=True,
    #     quantization="fp8",
    # ),
    "qwen3-32b": ModelInfo(
        name="qwen3-32b",
        hf_path="Qwen/Qwen3-32B",
        # bf16 - FP8 not supported on SM120/Blackwell (deep_gemm + flashinfer both fail)
        mem_fraction_static=GPU_MEM_FRAC,
    ),
    # NOTE: Mistral-Small-3.2 dropped - non-standard tokenizer (MistralCommonTokenizer),
    # non-standard weight format (consolidated.safetensors), no preprocessor_config.json.
    # Would need 3 separate workarounds. Not worth it until Mistral fixes their HF integration.
    "llama-3.1-70b": ModelInfo(
        name="llama-3.1-70b",
        hf_path="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        context_length=131072,  # Llama 3.1 supports 128K
        mem_fraction_static=GPU_MEM_FRAC,
        quantization="awq",
    ),
    # NOTE: GLM-4.6V-AWQ requires transformers >= 5.0, incompatible with SGLang.
    # Qwen2.5-72B weights not downloaded.
    "qwen2.5-7b": ModelInfo(
        name="qwen2.5-7b",
        hf_path="Qwen/Qwen2.5-7B-Instruct",
        mem_fraction_static=GPU_MEM_FRAC,
    ),
    "kimi-vl": ModelInfo(
        name="kimi-vl",
        hf_path="moonshotai/Kimi-VL-A3B-Thinking-2506",
        # Auto-detect context from model config
        mem_fraction_static=GPU_MEM_FRAC,
        supports_vision=True,
    ),
}

MODEL_ALIASES = {
    "gpt-oss": "gpt-oss-120b",
    "qwen-vl": "kimi-vl",  # Qwen3-VL broken on SM120, use kimi-vl instead
    "qwen": "qwen3-32b",
    "qwen-small": "qwen2.5-7b",
    "llama": "llama-3.1-70b",
    "kimi": "kimi-vl",
}


# =============================================================================
# Request/Response Models (OpenAI-compatible)
# =============================================================================

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "blitzinfer"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]


# =============================================================================
# Queue State Machine
# =============================================================================

class QueueState(Enum):
    SERVING = auto()
    DRAINING = auto()
    SWITCHING = auto()


@dataclass
class QueuedRequest:
    """A request waiting in a model queue."""
    request: ChatCompletionRequest
    request_id: str
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.time)


# =============================================================================
# Memory Monitoring
# =============================================================================

def get_memory_info() -> Dict[str, float]:
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    gpu_allocated = torch.cuda.memory_allocated()
    gpu_reserved = torch.cuda.memory_reserved()
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(':')] = int(parts[1]) * 1024
        ram_total = meminfo.get('MemTotal', 0)
        ram_available = meminfo.get('MemAvailable', 0)
        ram_shared = meminfo.get('Shmem', 0)
    except Exception:
        ram_total = ram_available = ram_shared = 0

    return {
        'gpu_used_gb': (gpu_total - gpu_free) / 1024**3,
        'gpu_free_gb': gpu_free / 1024**3,
        'gpu_total_gb': gpu_total / 1024**3,
        'gpu_allocated_gb': gpu_allocated / 1024**3,
        'gpu_reserved_gb': gpu_reserved / 1024**3,
        'ram_total_gb': ram_total / 1024**3,
        'ram_available_gb': ram_available / 1024**3,
        'ram_shared_gb': ram_shared / 1024**3,
    }

def log_memory_state(label: str):
    mem = get_memory_info()
    logger.info(f"[MEMORY {label}] GPU: {mem['gpu_used_gb']:.1f}/{mem['gpu_total_gb']:.1f}GB used "
                f"(alloc={mem['gpu_allocated_gb']:.1f}GB, rsv={mem['gpu_reserved_gb']:.1f}GB) | "
                f"RAM: {mem['ram_available_gb']:.1f}GB avail, {mem['ram_shared_gb']:.1f}GB shared")
    return mem

def verify_memory_available(min_gpu_free_gb: float = 10.0, min_ram_avail_gb: float = 5.0) -> Tuple[bool, str]:
    mem = get_memory_info()
    issues = []
    if mem['gpu_free_gb'] < min_gpu_free_gb:
        issues.append(f"GPU free {mem['gpu_free_gb']:.1f}GB < {min_gpu_free_gb}GB required")
    if mem['ram_available_gb'] < min_ram_avail_gb:
        issues.append(f"RAM available {mem['ram_available_gb']:.1f}GB < {min_ram_avail_gb}GB required")
    if issues:
        return False, "; ".join(issues)
    return True, "OK"


# =============================================================================
# Server State with Per-Model Queues
# =============================================================================

MAX_QUEUE_SIZE = 64
DRAIN_TIMEOUT = 30.0

class ServerState:
    """Global server state with per-model request queues."""

    def __init__(self):
        self.engine: Optional[SglEngine] = None
        self.current_model: Optional[str] = None
        self.standby: Optional[StandbyManager] = None
        self.request_count: int = 0
        self.switch_count: int = 0
        self.total_tokens: int = 0
        self.start_time: float = time.time()

        # Queue state
        self.queues: Dict[str, asyncio.Queue] = {}
        self.queue_state: QueueState = QueueState.SERVING
        self.in_flight: int = 0
        self._switch_lock = asyncio.Lock()
        self._switch_event = asyncio.Event()  # Signals queue processor to check for switch
        self._queue_processor_task: Optional[asyncio.Task] = None

    async def initialize(self):
        logger.info("=" * 80)
        logger.info("BLITZINFER SERVER INITIALIZING (SGLang + Queues)")
        logger.info("=" * 80)

        # Create queues for all models
        for model_id in AVAILABLE_MODELS:
            self.queues[model_id] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        logger.info(f"Created {len(self.queues)} model queues (max_size={MAX_QUEUE_SIZE})")

        # Initialize standby manager
        logger.info("Creating StandbyManager with 80GB pinned arena (5x16GB chunks)...")
        start = time.time()
        self.standby = StandbyManager(
            arena_size_gb=80.0,
            chunk_size_gb=16.0,
            pin_memory=True,
            lazy_arena=False,
        )
        elapsed = time.time() - start
        logger.info(f"StandbyManager initialized in {elapsed:.1f}s")

        # Load initial model
        initial_model = "gpt-oss-120b"
        logger.info(f"Loading initial model: {initial_model}")
        await self._load_engine(initial_model)

        # Start background queue processor
        self._queue_processor_task = asyncio.create_task(self._queue_processor())
        logger.info("Queue processor started")

        logger.info("=" * 80)
        logger.info("BLITZINFER SERVER READY")
        logger.info(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        logger.info("=" * 80)

    async def _load_engine(self, model_id: str):
        """Create a new SGLang Engine for the given model."""
        model_info = AVAILABLE_MODELS.get(model_id)
        if not model_info:
            raise ValueError(f"Unknown model: {model_id}")

        start = time.time()
        crash_log(f"_load_engine: {model_id}, hf_path={model_info.hf_path}")
        logger.info(f"Loading SGLang engine for {model_id}...")

        engine_kwargs = {
            "model_path": model_info.hf_path,
            "mem_fraction_static": model_info.mem_fraction_static,
            "trust_remote_code": True,
            "disable_cuda_graph": True,  # SGLang equivalent of vLLM's enforce_eager
            "attention_backend": "triton",  # SM120 (Blackwell desktop): flashinfer not supported
            "log_level": "info",
        }

        # Only set context_length if explicitly configured (None = auto-detect from model config)
        if model_info.context_length is not None:
            engine_kwargs["context_length"] = model_info.context_length

        if model_info.quantization:
            engine_kwargs["quantization"] = model_info.quantization

        # SM120 (Blackwell desktop: RTX PRO 6000, RTX 5090) compatibility:
        # - DeepGemm FP8 kernels crash with "Unknown recipe" on SM120
        # - Cutlass FP8 produces garbage on Qwen3-VL (works for non-VL)
        # - Triton FP8 is the safest fallback for SM120
        if model_info.quantization == "fp8":
            engine_kwargs["fp8_gemm_runner_backend"] = "triton"

        # SM120 (Blackwell desktop): Vision encoder triton attention exceeds 99KB shared memory
        # Use PyTorch's SDPA backend for vision encoder attention (safe on all CUDA devices)
        # Decoder MLA attention is handled by patched triton block sizes in extend_attention.py
        if model_info.supports_vision:
            engine_kwargs["mm_attention_backend"] = "sdpa"

        # GPT-OSS needs Harmony tool call parser + triton_kernel for MXFP4 MoE
        if model_id == "gpt-oss-120b":
            engine_kwargs["tool_call_parser"] = "harmony"
            # triton_kernel keeps weights in native mxfp4 (no upcast to bf16)
            engine_kwargs["moe_runner_backend"] = "triton_kernel"
            # gpt-oss config has torch_dtype=float32, auto-downcasts to float16,
            # but triton MoE kernels require bfloat16 hidden states
            engine_kwargs["dtype"] = "bfloat16"

        # Apply any model-specific extra args
        if model_info.extra_args:
            engine_kwargs.update(model_info.extra_args)

        crash_log(f"_load_engine: SglEngine() starting")
        self.engine = SglEngine(**engine_kwargs)
        crash_log(f"_load_engine: SglEngine() done")

        self.current_model = model_id
        elapsed = time.time() - start
        crash_log(f"_load_engine: {model_id} loaded in {elapsed:.1f}s")
        logger.info(f"Engine {model_id} loaded in {elapsed:.1f}s")

    async def _shutdown_engine(self):
        """Shutdown current SGLang Engine and release GPU memory."""
        if self.engine is None:
            return

        crash_log(f"_shutdown_engine: shutting down {self.current_model}")
        logger.info(f"Shutting down engine for {self.current_model}...")

        mem_before = log_memory_state("BEFORE_ENGINE_SHUTDOWN")

        try:
            self.engine.shutdown()
        except Exception as e:
            logger.warning(f"Engine shutdown error: {e}")

        self.engine = None

        # Cleanup GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Extra gc passes
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()

        # Release memory to OS
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

        mem_after = log_memory_state("AFTER_ENGINE_SHUTDOWN")
        freed = mem_before['gpu_used_gb'] - mem_after['gpu_used_gb']
        crash_log(f"_shutdown_engine: freed ~{freed:.1f}GB GPU")
        logger.info(f"Engine shutdown complete, freed ~{freed:.1f}GB GPU")

    def _build_reload_overrides(self, model_id: str, model_info: ModelInfo) -> Dict[str, Any]:
        """Build server_args overrides dict for reload_model()."""
        overrides = {}
        if model_info.context_length is not None:
            overrides["context_length"] = model_info.context_length
        if model_info.quantization:
            overrides["quantization"] = model_info.quantization
        if model_info.quantization == "fp8":
            overrides["fp8_gemm_runner_backend"] = "triton"
        if model_id == "gpt-oss-120b":
            overrides["moe_runner_backend"] = "triton_kernel"
            # gpt-oss config has torch_dtype=float32, which auto-downcasts to
            # float16, but triton MoE kernels require bfloat16 hidden states.
            overrides["dtype"] = "bfloat16"
        return overrides

    async def _switch_model(self, target_model: str):
        """Switch to a different model (called from queue processor).

        Uses Engine.reload_model() for fast in-process switching (~4-10s warm).
        Falls back to full Engine restart if reload fails.
        """
        if self.current_model == target_model:
            return

        self.switch_count += 1
        crash_log(f"=== SWITCH #{self.switch_count}: {self.current_model} -> {target_model} ===")
        logger.info("=" * 60)
        logger.info(f"MODEL SWITCH #{self.switch_count}: {self.current_model} -> {target_model}")
        logger.info("=" * 60)

        start = time.time()
        target_info = AVAILABLE_MODELS[target_model]

        # Try fast in-process reload first.
        # Previously skipped for gpt-oss-120b due to mxfp4 bf16 upcast OOM,
        # but now fixed by using triton_kernels backend (keeps weights in mxfp4).
        if self.engine is not None:
            reload_success = await self._try_reload_model(target_model, target_info)
            if reload_success:
                self.current_model = target_model
                log_memory_state("AFTER_SWITCH")
                elapsed = time.time() - start
                crash_log(f"=== SWITCH #{self.switch_count} COMPLETE (reload) in {elapsed:.1f}s ===")
                logger.info(f"Switch complete via reload_model in {elapsed:.1f}s")
                return

        # Fallback: full Engine restart
        logger.info("Using full Engine restart (fallback)")

        # Wait for standby prefetch if running
        standby_state = self.standby.get_state()
        if standby_state.name == "LOADING":
            logger.info("Waiting for background prefetch...")
            self.standby.wait_for_load(timeout=120.0)

        await self._shutdown_engine()

        mem_ok, mem_msg = verify_memory_available(min_gpu_free_gb=5.0, min_ram_avail_gb=2.0)
        if not mem_ok:
            raise MemoryError(f"Insufficient memory to load {target_model}: {mem_msg}")

        await self._load_engine(target_model)

        log_memory_state("AFTER_SWITCH")
        elapsed = time.time() - start
        crash_log(f"=== SWITCH #{self.switch_count} COMPLETE (restart) in {elapsed:.1f}s ===")
        logger.info(f"Switch complete via Engine restart in {elapsed:.1f}s")

    async def _try_reload_model(self, target_model: str, target_info: ModelInfo) -> bool:
        """Attempt fast in-process model reload. Returns True on success."""
        from sglang.srt.managers.io_struct import ReloadModelReqInput

        overrides = self._build_reload_overrides(target_model, target_info)
        logger.info(f"Attempting reload_model: {self.current_model} -> {target_model} "
                    f"(overrides={overrides})")

        try:
            # Call tokenizer_manager directly (async) to avoid
            # Engine.reload_model()'s run_until_complete which conflicts
            # with the already-running FastAPI event loop.
            obj = ReloadModelReqInput(
                model_path=target_info.hf_path,
                server_args_overrides=overrides,
                flush_cache=True,
            )
            success, message = await self.engine.tokenizer_manager.reload_model(
                obj, None
            )

            if success:
                # Reload tokenizer in the main process (mirrors Engine.reload_model)
                try:
                    from sglang.srt.utils.hf_transformers_utils import get_tokenizer
                    self.engine.tokenizer_manager.tokenizer = get_tokenizer(
                        target_info.hf_path,
                        tokenizer_mode=self.engine.server_args.tokenizer_mode,
                        trust_remote_code=self.engine.server_args.trust_remote_code,
                    )
                except Exception as e:
                    logger.warning(f"Tokenizer reload in server (non-fatal): {e}")

                logger.info(f"reload_model succeeded: {message}")
                return True
            else:
                logger.warning(f"reload_model failed: {message}")
                return False

        except Exception as e:
            logger.warning(f"reload_model exception: {e}")
            logger.warning(traceback.format_exc())
            return False

    def _find_next_model(self) -> Optional[str]:
        """Find the next model that has queued requests.

        Returns model_id if a non-active model has pending requests, None otherwise.
        """
        for model_id, queue in self.queues.items():
            if model_id != self.current_model and not queue.empty():
                return model_id
        return None

    async def _queue_processor(self):
        """Background task: dequeue requests from active model and process them.

        State machine:
            SERVING  -> dequeue from active queue, generate, respond
            DRAINING -> active queue empty, wait for in_flight == 0, then switch
            SWITCHING -> shutting down engine, loading new model
        """
        logger.info("Queue processor running")

        while True:
            try:
                if self.queue_state == QueueState.SERVING:
                    await self._process_serving()

                elif self.queue_state == QueueState.DRAINING:
                    await self._process_draining()

                elif self.queue_state == QueueState.SWITCHING:
                    await self._process_switching()

            except asyncio.CancelledError:
                logger.info("Queue processor cancelled")
                return
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1.0)

    async def _process_serving(self):
        """SERVING state: dequeue from active model's queue and process."""
        if self.current_model is None:
            await asyncio.sleep(0.1)
            return

        active_queue = self.queues.get(self.current_model)
        if active_queue is None:
            await asyncio.sleep(0.1)
            return

        # Check if another model has pending requests and active queue is empty
        if active_queue.empty() and self.in_flight == 0:
            next_model = self._find_next_model()
            if next_model:
                logger.info(f"Active queue empty, switching to {next_model}")
                self.queue_state = QueueState.DRAINING
                self._target_switch_model = next_model
                return

        # Try to dequeue (with timeout so we can check for switches)
        try:
            queued: QueuedRequest = await asyncio.wait_for(
                active_queue.get(), timeout=0.5
            )
        except asyncio.TimeoutError:
            return

        # Process the request
        self.in_flight += 1
        try:
            result = await self._generate_response(queued.request, queued.request_id)
            if not queued.future.done():
                queued.future.set_result(result)
        except Exception as e:
            if not queued.future.done():
                queued.future.set_exception(e)
        finally:
            self.in_flight -= 1

    async def _process_draining(self):
        """DRAINING state: wait for in_flight to reach 0, then switch."""
        if self.in_flight > 0:
            logger.debug(f"Draining: {self.in_flight} requests still in flight")
            await asyncio.sleep(0.1)
            return

        # All in-flight done, transition to switching
        target = getattr(self, '_target_switch_model', None)
        if target:
            self.queue_state = QueueState.SWITCHING
        else:
            self.queue_state = QueueState.SERVING

    async def _process_switching(self):
        """SWITCHING state: shutdown current engine, load new model."""
        target = getattr(self, '_target_switch_model', None)
        if not target:
            self.queue_state = QueueState.SERVING
            return

        try:
            await self._switch_model(target)
        except Exception as e:
            logger.error(f"Model switch failed: {e}")
            # Fail all queued requests for the target model
            target_queue = self.queues.get(target)
            if target_queue:
                while not target_queue.empty():
                    try:
                        queued = target_queue.get_nowait()
                        if not queued.future.done():
                            queued.future.set_exception(e)
                    except asyncio.QueueEmpty:
                        break

        self._target_switch_model = None
        self.queue_state = QueueState.SERVING

    async def enqueue_request(self, request: ChatCompletionRequest, model_id: str) -> asyncio.Future:
        """Enqueue a request for a specific model. Returns a Future with the response."""
        queue = self.queues.get(model_id)
        if queue is None:
            raise HTTPException(status_code=404, detail=f"No queue for model: {model_id}")

        if queue.full():
            raise HTTPException(status_code=429, detail=f"Queue full for model {model_id}")

        request_id = str(uuid.uuid4())[:8]
        future = asyncio.get_event_loop().create_future()

        queued = QueuedRequest(
            request=request,
            request_id=request_id,
            future=future,
        )

        await queue.put(queued)
        logger.info(f"[{request_id}] Enqueued for {model_id} (depth={queue.qsize()})")

        # If this is for a non-active model, trigger prefetch
        if model_id != self.current_model:
            model_info = AVAILABLE_MODELS.get(model_id)
            if model_info:
                logger.info(f"[{request_id}] Triggering prefetch for {model_id}")
                self.standby.start_prefetch(model_info.hf_path)

        return future, request_id

    async def _generate_response(
        self, request: ChatCompletionRequest, request_id: str
    ) -> Union[ChatCompletionResponse, StreamingResponse]:
        """Generate a response using the SGLang engine."""
        model_id = self.current_model
        model_info = AVAILABLE_MODELS[model_id]

        self.request_count += 1

        logger.info(f"[{request_id}] Generating: model={model_id}, "
                    f"msgs={len(request.messages)}, max_tokens={request.max_tokens}")

        use_harmony = HAS_HARMONY and model_id == "gpt-oss-120b"
        has_tools = request.tools and len(request.tools) > 0

        # Build prompt
        prompt = None
        input_ids = None
        image_data = None

        if use_harmony:
            input_ids, image_data = self._build_harmony_prompt(
                request, request_id, has_tools
            )
        if input_ids is None:
            prompt, image_data = self._build_chat_prompt(
                request, request_id, model_info
            )

        # Build sampling params dict (SGLang uses plain dicts)
        sampling_params = {
            "max_new_tokens": request.max_tokens or 2048,
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 0.95,
            "presence_penalty": request.presence_penalty or 0.0,
            "frequency_penalty": request.frequency_penalty or 0.0,
        }
        if request.stop:
            sampling_params["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]
        if use_harmony:
            sampling_params["stop_token_ids"] = HARMONY_STOP_TOKENS

        # Generate
        start = time.time()

        gen_kwargs = {"sampling_params": sampling_params}
        if input_ids is not None:
            gen_kwargs["input_ids"] = input_ids
        else:
            gen_kwargs["prompt"] = prompt
        if image_data is not None:
            gen_kwargs["image_data"] = image_data

        if request.stream:
            return self._build_streaming_response(
                request, request_id, model_id, gen_kwargs,
                use_harmony, has_tools
            )

        # Non-streaming generation
        output = await self.engine.async_generate(**gen_kwargs)

        elapsed = time.time() - start
        generated_text = output["text"]
        output_token_ids = output.get("output_ids", [])
        meta = output.get("meta_info", {})
        prompt_tokens = meta.get("prompt_tokens", 0)
        completion_tokens = len(output_token_ids)
        total_tokens = prompt_tokens + completion_tokens
        self.total_tokens += total_tokens

        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(f"[{request_id}] Generated {completion_tokens} tokens "
                    f"in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

        # Parse finish reason
        finish_reason_info = meta.get("finish_reason", {})
        finish_reason = finish_reason_info.get("type", "stop") if isinstance(finish_reason_info, dict) else "stop"

        # Parse Harmony output for GPT-OSS
        tool_calls_list = []
        if use_harmony and output_token_ids:
            generated_text, tool_calls_list = self._parse_harmony_output(
                output_token_ids, request_id, has_tools
            )
            if tool_calls_list:
                finish_reason = "tool_calls"

        # Log response
        if generated_text:
            preview = generated_text[:500] + "..." if len(generated_text) > 500 else generated_text
        else:
            generated_text = ""
            preview = "(empty)"
        logger.info(f"[{request_id}] RESPONSE: {preview}")

        response_message = ResponseMessage(
            role="assistant",
            content=generated_text if generated_text else None,
            tool_calls=tool_calls_list if tool_calls_list else None,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            created=int(time.time()),
            model=model_id,
            choices=[ChatCompletionChoice(
                index=0,
                message=response_message,
                finish_reason=finish_reason,
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    # -------------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------------

    def _build_harmony_prompt(
        self, request: ChatCompletionRequest, request_id: str, has_tools: bool
    ) -> Tuple[Optional[List[int]], Optional[Any]]:
        """Build Harmony-encoded prompt for GPT-OSS. Returns (input_ids, image_data)."""
        try:
            harmony_msgs = []

            # System message
            sys_msg = get_system_message()
            harmony_msgs.append(sys_msg)

            # Developer message with tools
            if has_tools:
                from openai.types.responses import FunctionTool
                tools_for_harmony = []
                for tool in request.tools:
                    t = FunctionTool(
                        type="function",
                        name=tool.function.name,
                        description=tool.function.description or "",
                        parameters=tool.function.parameters or {},
                    )
                    tools_for_harmony.append(t)
                dev_msg = get_developer_message(tools=tools_for_harmony)
                harmony_msgs.append(dev_msg)
                logger.info(f"[{request_id}] Added {len(request.tools)} tools to Harmony")

            # Convert chat messages to Harmony format
            # parse_chat_input handles basic messages but not tool_calls or tool results
            from openai.types.responses import ResponseFunctionToolCall as HarmonyToolCall
            # Track tool calls so parse_response_input can look up call_id for tool results
            prev_tool_calls: list[HarmonyToolCall] = []
            for msg in request.messages:
                if msg.role == "tool" and msg.tool_call_id:
                    # Tool result message - convert to FunctionCallOutput
                    from openai.types.responses.response_input_item_param import FunctionCallOutput
                    tool_output = FunctionCallOutput(
                        type="function_call_output",
                        call_id=msg.tool_call_id,
                        output=msg.content or "",
                    )
                    harmony_msgs.append(parse_response_input(tool_output, prev_tool_calls))
                elif msg.role == "assistant" and msg.tool_calls:
                    # Assistant message with tool calls
                    for tc in msg.tool_calls:
                        tool_call = HarmonyToolCall(
                            id=tc.id,
                            type="function_call",
                            call_id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        )
                        prev_tool_calls.append(tool_call)
                        harmony_msgs.append(parse_response_output(tool_call))
                elif msg.content is not None:
                    harmony_msgs.append(parse_chat_input(msg))
                else:
                    logger.debug(f"[{request_id}] Skipping msg role={msg.role} (content=None)")

            input_ids = render_for_completion(harmony_msgs)
            logger.debug(f"[{request_id}] Harmony prompt: {len(input_ids)} tokens")
            return input_ids, None

        except Exception as e:
            logger.warning(f"[{request_id}] Harmony encoding failed: {e}, falling back to chat template")
            logger.warning(traceback.format_exc())
            return None, None

    def _build_chat_prompt(
        self, request: ChatCompletionRequest, request_id: str, model_info: ModelInfo
    ) -> Tuple[str, Optional[Any]]:
        """Build a text prompt using the model's chat template. Returns (prompt, image_data)."""
        image_data = None

        # Extract image data from multimodal messages
        if model_info.supports_vision:
            image_data = self._extract_image_data(request.messages)

        # Get tokenizer from engine and apply chat template
        messages_dicts = []
        for msg in request.messages:
            d = {"role": msg.role}
            if isinstance(msg.content, str):
                d["content"] = msg.content
            elif isinstance(msg.content, list):
                # Multimodal content - extract text parts for prompt, images handled separately
                text_parts = []
                for part in msg.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            # Image placeholder - SGLang handles via image_data param
                            text_parts.append("<image>")
                d["content"] = "\n".join(text_parts) if text_parts else ""
            else:
                d["content"] = str(msg.content) if msg.content else ""
            messages_dicts.append(d)

        # Use SGLang's tokenizer to apply the chat template
        try:
            tokenizer = self.engine.tokenizer_manager.tokenizer
            prompt = tokenizer.apply_chat_template(
                messages_dicts, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"[{request_id}] Chat template failed: {e}, using fallback")
            prompt = _format_chat_messages_fallback(request.messages)

        logger.debug(f"[{request_id}] Prompt length: {len(prompt)} chars")
        return prompt, image_data

    def _extract_image_data(self, messages: List[ChatMessage]) -> Optional[List[str]]:
        """Extract image URLs/paths from multimodal messages."""
        images = []
        for msg in messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url_data = part.get("image_url", {})
                        url = url_data.get("url", "") if isinstance(url_data, dict) else str(url_data)
                        if url:
                            images.append(url)
        return images if images else None

    # -------------------------------------------------------------------------
    # Harmony output parsing
    # -------------------------------------------------------------------------

    def _parse_harmony_output(
        self, output_token_ids: List[int], request_id: str, has_tools: bool
    ) -> Tuple[str, List[ToolCall]]:
        """Parse Harmony output tokens into text and tool calls."""
        tool_calls_list = []
        final_text = ""

        try:
            parser = parse_output_into_messages(list(output_token_ids))
            reasoning_parts = []
            final_parts = []

            for msg in parser.messages:
                items = parse_output_message(msg)
                for item in items:
                    if isinstance(item, ResponseFunctionToolCall):
                        tool_calls_list.append(ToolCall(
                            id=item.call_id,
                            type="function",
                            function=FunctionCall(
                                name=item.name,
                                arguments=item.arguments,
                            ),
                        ))
                        logger.info(f"[{request_id}] Tool call: {item.name}({item.arguments[:100]}...)")
                    elif hasattr(item, 'type'):
                        if item.type == "reasoning":
                            for c in getattr(item, 'content', []):
                                if hasattr(c, 'text'):
                                    reasoning_parts.append(c.text)
                        elif item.type == "message":
                            for c in getattr(item, 'content', []):
                                if hasattr(c, 'text'):
                                    final_parts.append(c.text)

            # Check partial content in parser
            if hasattr(parser, 'current_content') and parser.current_content:
                channel = getattr(parser, 'current_channel', None)
                if channel == "final":
                    final_parts.append(parser.current_content)
                elif channel == "commentary":
                    recipient = getattr(parser, 'current_recipient', None)
                    if recipient and recipient.startswith("functions."):
                        func_name = recipient.split(".")[-1]
                        tool_calls_list.append(ToolCall(
                            id=f"call_{uuid.uuid4().hex[:24]}",
                            type="function",
                            function=FunctionCall(
                                name=func_name,
                                arguments=parser.current_content,
                            ),
                        ))

            if final_parts:
                final_text = "\n".join(final_parts)
            if tool_calls_list:
                final_text = ""

            logger.info(f"[{request_id}] Harmony parsed: "
                       f"reasoning={len(''.join(reasoning_parts))} chars, "
                       f"final={len(final_text)} chars, "
                       f"tool_calls={len(tool_calls_list)}")

        except Exception as e:
            logger.warning(f"[{request_id}] Harmony parse failed: {e}")

        return final_text, tool_calls_list

    # -------------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------------

    def _build_streaming_response(
        self, request: ChatCompletionRequest, request_id: str,
        model_id: str, gen_kwargs: dict,
        use_harmony: bool, has_tools: bool
    ) -> StreamingResponse:
        """Build a streaming SSE response."""

        async def event_stream():
            created = int(time.time())
            prev_text = ""

            try:
                gen_kwargs["stream"] = True
                generator = await self.engine.async_generate(**gen_kwargs)

                if use_harmony:
                    # Harmony models: buffer entire output, parse channels, then emit
                    # This avoids leaking "analysis" (reasoning) channel to the client
                    last_chunk = None
                    async for chunk in generator:
                        last_chunk = chunk

                    if last_chunk:
                        output_ids = last_chunk.get("output_ids", [])
                        meta = last_chunk.get("meta_info", {})
                        finish_info = meta.get("finish_reason", {})
                        finish = finish_info.get("type", "stop") if isinstance(finish_info, dict) else "stop"

                        final_text = ""
                        tool_calls = []
                        if output_ids:
                            final_text, tool_calls = self._parse_harmony_output(
                                output_ids, request_id, has_tools
                            )

                        if tool_calls:
                            finish = "tool_calls"
                            for i, tc in enumerate(tool_calls):
                                tc_data = {
                                    "id": f"chatcmpl-{request_id}",
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_id,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [{
                                                "index": i,
                                                "id": tc.id,
                                                "type": "function",
                                                "function": {
                                                    "name": tc.function.name,
                                                    "arguments": tc.function.arguments,
                                                },
                                            }],
                                        },
                                        "finish_reason": None,
                                    }],
                                }
                                yield f"data: {json.dumps(tc_data)}\n\n"
                        elif final_text:
                            # Emit parsed final content as a single chunk
                            data = {
                                "id": f"chatcmpl-{request_id}",
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": final_text},
                                    "finish_reason": None,
                                }],
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                        else:
                            # Fallback: emit raw text if Harmony parsing returned nothing
                            raw = last_chunk.get("text", "")
                            if raw:
                                data = {
                                    "id": f"chatcmpl-{request_id}",
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_id,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": raw},
                                        "finish_reason": None,
                                    }],
                                }
                                yield f"data: {json.dumps(data)}\n\n"
                else:
                    # Non-Harmony models: stream deltas in real-time
                    async for chunk in generator:
                        chunk_text = chunk.get("text", "")
                        # SGLang streaming returns cumulative text, extract delta
                        delta = chunk_text[len(prev_text):]
                        prev_text = chunk_text

                        if not delta:
                            continue

                        data = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": delta},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                # Final chunk with finish_reason
                if not use_harmony:
                    meta = chunk.get("meta_info", {}) if chunk else {}
                    finish_info = meta.get("finish_reason", {})
                    finish = finish_info.get("type", "stop") if isinstance(finish_info, dict) else "stop"

                final_data = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish,
                    }],
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"[{request_id}] Streaming error: {e}")
                error_data = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"\n\n[Error: {e}]"},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    def start_prefetch(self, model_id: str):
        model_info = AVAILABLE_MODELS.get(model_id)
        if model_info and model_id != self.current_model:
            logger.info(f"Starting prefetch for {model_id}")
            self.standby.start_prefetch(model_info.hf_path)


state = ServerState()


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await state.initialize()
    yield
    logger.info("Server shutting down...")
    if state._queue_processor_task:
        state._queue_processor_task.cancel()
        try:
            await state._queue_processor_task
        except asyncio.CancelledError:
            pass
    if state.engine:
        try:
            state.engine.shutdown()
        except Exception:
            pass


app = FastAPI(
    title="BlitzInfer API",
    description="Fast LLM serving with queue-driven model switching (SGLang)",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    models = [
        ModelObject(id=model_id, created=int(state.start_time))
        for model_id in AVAILABLE_MODELS.keys()
    ]
    return ModelsResponse(data=models)


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    model_id = MODEL_ALIASES.get(model_id, model_id)
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return ModelObject(id=model_id, created=int(state.start_time))


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion request via per-model queue."""
    # Resolve model
    model_id = MODEL_ALIASES.get(request.model, request.model)
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model}")

    # Log request
    for i, msg in enumerate(request.messages):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        preview = content[:500] + "..." if len(content) > 500 else content
        logger.debug(f"MSG[{i}] {msg.role}: {preview}")

    try:
        # If this model is already active and queue is empty, fast-path directly
        if model_id == state.current_model and state.queues[model_id].empty() and state.in_flight == 0:
            request_id = str(uuid.uuid4())[:8]
            state.request_count += 1
            logger.info(f"[{request_id}] Fast-path: model={model_id}")
            return await state._generate_response(request, request_id)

        # Otherwise, enqueue and wait
        future, request_id = await state.enqueue_request(request, model_id)
        return await future

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    queue_depths = {mid: q.qsize() for mid, q in state.queues.items() if q.qsize() > 0}
    return {
        "status": "healthy",
        "current_model": state.current_model,
        "queue_state": state.queue_state.name,
        "in_flight": state.in_flight,
        "queue_depths": queue_depths,
        "request_count": state.request_count,
        "switch_count": state.switch_count,
        "total_tokens": state.total_tokens,
        "uptime_seconds": int(time.time() - state.start_time),
    }


@app.get("/status")
async def status():
    gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info()
    gpu_mem_used = (gpu_mem_total - gpu_mem_free) / 1024**3
    gpu_mem_total_gb = gpu_mem_total / 1024**3

    standby_status = None
    if state.standby:
        standby_model = state.standby.get_standby_model()
        standby_status = {
            "state": state.standby.get_state().name,
            "standby_model": standby_model,
            "is_ready": state.standby.is_ready(standby_model) if standby_model else False,
        }

    queue_info = {}
    for mid, q in state.queues.items():
        queue_info[mid] = {"depth": q.qsize(), "is_active": mid == state.current_model}

    return {
        "current_model": state.current_model,
        "available_models": list(AVAILABLE_MODELS.keys()),
        "queue_state": state.queue_state.name,
        "in_flight": state.in_flight,
        "queues": queue_info,
        "request_count": state.request_count,
        "switch_count": state.switch_count,
        "total_tokens": state.total_tokens,
        "uptime_seconds": int(time.time() - state.start_time),
        "gpu_memory_used_gb": round(gpu_mem_used, 2),
        "gpu_memory_total_gb": round(gpu_mem_total_gb, 2),
        "standby": standby_status,
    }


@app.post("/prefetch/{model_id}")
async def prefetch_model(model_id: str):
    model_id = MODEL_ALIASES.get(model_id, model_id)
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    state.start_prefetch(model_id)
    return {"status": "prefetch_started", "model": model_id}


# =============================================================================
# Helpers
# =============================================================================

def _format_chat_messages_fallback(messages: List[ChatMessage]) -> str:
    """Fallback prompt formatting when chat template is unavailable."""
    parts = []
    for msg in messages:
        role = msg.role
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting BlitzInfer API server (SGLang + Queues)...")
    logger.info(f"Crash log: {CRASH_LOG_FILE}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
