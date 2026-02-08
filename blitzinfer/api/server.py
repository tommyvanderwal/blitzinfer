"""OpenAI-compatible API server for BlitzInfer with per-model request queues.

Architecture:
    FastAPI (:8000)
      -> Per-model asyncio.Queue (ChatCompletionRequest objects)
      -> Queue processor (dequeue active model, forward to engine)
      -> vLLM Engine (in-process)

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
import re
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

# vLLM imports
from vllm import LLM, SamplingParams

# Harmony utils from vLLM (for GPT-OSS tool calling)
try:
    from vllm.entrypoints.openai.parser.harmony_utils import (
        get_system_message,
        get_developer_message,
        render_for_completion,
        get_stop_tokens_for_assistant_actions,
        parse_output_into_messages,
        parse_output_message,
        parse_chat_inputs_to_harmony_messages,
    )
    from openai.types.responses import ResponseFunctionToolCall
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
        FunctionDefinition as VLLMFunctionDefinition,
    )
    HAS_HARMONY = True
    HARMONY_STOP_TOKENS = list(get_stop_tokens_for_assistant_actions())
except ImportError:
    HAS_HARMONY = False
    HARMONY_STOP_TOKENS = []

# BlitzInfer imports - sys.path must be set before any blitzinfer imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Reasoning content extraction for thinking models (Kimi-VL, Qwen3, GLM, etc.)
from blitzinfer.api.reasoning import extract_reasoning_content
from blitzinfer.orchestrator.standby_manager import StandbyManager
from blitzinfer.engine.cleanup import full_cleanup
from blitzinfer.memory import set_preloaded_weights


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
SESSION_LOG_FILE = os.path.expanduser('~/blitzinfer_sessions.log')

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

# Dedicated session logger for full request/response debugging
session_logger = logging.getLogger('blitzinfer.sessions')
session_logger.setLevel(logging.DEBUG)
session_logger.propagate = False  # Don't spam main log
_session_handler = FlushingFileHandler(SESSION_LOG_FILE, mode='a')
_session_handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
session_logger.addHandler(_session_handler)

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
    gpu_memory_utilization: float = 0.94
    supports_vision: bool = False
    quantization: Optional[str] = None
    dtype: Optional[str] = None
    is_thinking_model: bool = False  # Model outputs <think>...</think> reasoning blocks
    extra_args: Dict[str, Any] = field(default_factory=dict)


# GPU memory utilization for vLLM
# NOTE: MXFP4 models (gpt-oss-120b) require ~61GB and have opaque CUDA allocations
# that can't be fully freed via Python. Model switching from MXFP4 models may
# require process restart.
GPU_MEM_FRAC = 0.94

AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    "gpt-oss-120b": ModelInfo(
        name="gpt-oss-120b",
        hf_path="openai/gpt-oss-120b",
        # No context_length - use model's native 128K
        gpu_memory_utilization=GPU_MEM_FRAC,
        dtype="bfloat16",  # Required for MXFP4 MoE kernels
        # max_num_seqs=256: gpt-oss-120b has 200K vocab. Default 1024 seqs causes
        # sampler warmup to allocate 1024*200K*4B=2.3 GiB of temporary memory.
        # After model switches, C++ internal allocations (Triton JIT, NCCL, etc.)
        # accumulate ~1 GB/switch in PyTorch's allocator, reducing headroom.
        # With max_num_seqs=256, warmup only needs ~0.6 GiB (fits even after 20+ switches).
        # KV cache is unaffected - max concurrency is ~3x at 131K context anyway.
        extra_args={"max_num_seqs": 256},
    ),
    "qwen3-32b": ModelInfo(
        name="qwen3-32b",
        hf_path="Qwen/Qwen3-32B",
        gpu_memory_utilization=GPU_MEM_FRAC,
        is_thinking_model=True,  # Qwen3 outputs <think>...</think> tags
    ),
    "llama-3.1-70b": ModelInfo(
        name="llama-3.1-70b",
        hf_path="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        # No context_length - use model's native 128K
        gpu_memory_utilization=GPU_MEM_FRAC,
        quantization="awq",
    ),
    "qwen2.5-7b": ModelInfo(
        name="qwen2.5-7b",
        hf_path="Qwen/Qwen2.5-7B-Instruct",
        gpu_memory_utilization=GPU_MEM_FRAC,
    ),
    "kimi-vl": ModelInfo(
        name="kimi-vl",
        hf_path="moonshotai/Kimi-VL-A3B-Thinking-2506",
        gpu_memory_utilization=GPU_MEM_FRAC,
        supports_vision=True,
        is_thinking_model=True,  # Uses ◁think▷...◁/think▷ tokens
    ),
    "qwen3-coder-next": ModelInfo(
        name="qwen3-coder-next",
        hf_path="Qwen/Qwen3-Coder-Next-FP8",
        # Weights: 74.89 GiB. CUDA graphs enabled for ~2-3x decode speedup.
        # max_num_batched_tokens=2048: reduces peak activation memory during
        # chunked prefill and CUDA graph capture (default 16384 uses too much).
        # max_num_seqs=32: reduces sampler warmup memory (default 256 OOMs after
        # many switches when residual non-pytorch memory eats into margin).
        gpu_memory_utilization=0.90,
        context_length=110000,
        is_thinking_model=True,  # Uses <think>...</think> tags
        extra_args={"max_num_seqs": 32, "max_num_batched_tokens": 2048},
    ),
    # GLM-4.6V-NVFP4 disabled: hangs during model construction after switch
    # (0% CPU, stuck at "slow image processor" message). Works on cold start only.
    # "glm-4.6v-nvfp4": ModelInfo(
    #     name="glm-4.6v-nvfp4",
    #     hf_path="GadflyII/GLM-4.6V-NVFP4",
    #     gpu_memory_utilization=GPU_MEM_FRAC,
    #     context_length=100000,
    #     supports_vision=True,
    #     is_thinking_model=True,
    # ),
}

MODEL_ALIASES = {
    "gpt-oss": "gpt-oss-120b",
    "qwen-vl": "kimi-vl",
    "qwen": "qwen3-32b",
    "qwen-small": "qwen2.5-7b",
    "llama": "llama-3.1-70b",
    "kimi": "kimi-vl",
    "coder": "qwen3-coder-next",
    "qwen-coder": "qwen3-coder-next",
    # "glm": "glm-4.6v-nvfp4",  # disabled: hangs after model switch
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
    reasoning_content: Optional[str] = None
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
MODEL_LOAD_TIMEOUT = 300.0   # 5 min max for loading a model
GENERATION_TIMEOUT = 180.0   # 3 min max for a single generation

class ServerState:
    """Global server state with per-model request queues."""

    def __init__(self):
        self.llm: Optional[LLM] = None
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
        self._generate_lock = asyncio.Lock()  # Serialize llm.generate() calls
        self._switch_event = asyncio.Event()
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._failed_models: Dict[str, float] = {}  # model_id -> failure timestamp
        self._failed_model_cooldown = 120.0  # seconds before retrying failed model

        # First-request timing: tracks when a model becomes ready and whether
        # the first request after a switch has been served yet
        self._first_request_pending: bool = True   # True until first request completes
        self._model_ready_time: float = 0.0        # When _load_engine() completed
        self._switch_start_time: float = 0.0       # When _switch_model() started

    async def initialize(self):
        logger.info("=" * 80)
        logger.info("BLITZINFER SERVER INITIALIZING (vLLM + Queues)")
        logger.info("=" * 80)

        # Create queues for all models
        for model_id in AVAILABLE_MODELS:
            self.queues[model_id] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        logger.info(f"Created {len(self.queues)} model queues (max_size={MAX_QUEUE_SIZE})")

        # Initialize standby manager
        # Mixed chunk sizes: 64+16+1 = 81GB (all power-of-2, 0% PyTorch overhead)
        # Fits qwen3-coder-next (~80.4GB) with margin
        logger.info("Creating StandbyManager with 81GB pinned arena (64+16+1 GB)...")
        start = time.time()
        self.standby = StandbyManager(
            chunk_sizes_gb=[64, 16, 1],
            pin_memory=True,
            lazy_arena=False,
        )
        elapsed = time.time() - start
        logger.info(f"StandbyManager initialized in {elapsed:.1f}s")

        # Load initial model via pinned arena for O_DIRECT speed (~10 GB/s vs ~3 GB/s buffered)
        initial_model = "gpt-oss-120b"
        logger.info(f"Loading initial model: {initial_model} (via pinned arena)")
        initial_info = AVAILABLE_MODELS.get(initial_model)
        preloaded_weights = None
        if self.standby and initial_info:
            self.standby.start_prefetch(initial_info.hf_path)
            ready = self.standby.wait_for_load(timeout=120.0)
            if ready and self.standby.is_ready(initial_info.hf_path):
                preloaded_weights = self.standby.consume_standby()
                logger.info(f"Initial model prefetched to arena ({len(preloaded_weights)} tensors)")
        await self._load_engine(initial_model, preloaded_weights=preloaded_weights)
        if preloaded_weights is not None and self.standby:
            self.standby.free_arena()

        # Start background queue processor
        self._queue_processor_task = asyncio.create_task(self._queue_processor())
        logger.info("Queue processor started")

        logger.info("=" * 80)
        logger.info("BLITZINFER SERVER READY")
        logger.info(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        logger.info("=" * 80)

    async def _load_engine(self, model_id: str, preloaded_weights: Optional[Dict] = None):
        """Create a new vLLM LLM for the given model.

        Args:
            model_id: Model identifier from AVAILABLE_MODELS.
            preloaded_weights: If provided, use pinned arena loader for fast GPU transfer.
                Dict of tensor name -> torch.Tensor in pinned CPU memory.
        """
        model_info = AVAILABLE_MODELS.get(model_id)
        if not model_info:
            raise ValueError(f"Unknown model: {model_id}")

        start = time.time()
        use_pinned = preloaded_weights is not None
        crash_log(f"_load_engine: {model_id}, hf_path={model_info.hf_path}, pinned={use_pinned}")
        logger.info(f"Loading vLLM engine for {model_id} (pinned_arena={use_pinned})...")

        # vLLM V1 single-process mode for fast switching
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        # Use Marlin MXFP4 backend on SM120 (RTX PRO 6000 Blackwell)
        # Marlin + CUDA graphs = ~200 tok/s (vs Triton StridedLayout = ~31 tok/s)
        os.environ["VLLM_MXFP4_USE_MARLIN"] = "1"

        # Ensure nvcc is available for FlashInfer JIT compilation
        cuda_bin = "/usr/local/cuda-13.1/bin"
        if cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")

        # Dynamic gpu_memory_utilization: adapt to actual free memory.
        # C++ internal allocations (Triton JIT, NCCL, CUDA driver) accumulate
        # ~0.5-1 GB per model switch and cannot be freed from Python.
        # Without adaptation, the sampler warmup + CUDA graph capture OOMs
        # after ~6 switches. We ensure at least HEADROOM_GB remains free
        # after KV cache allocation for CUDA graphs, warmup, and overhead.
        HEADROOM_GB = 3.0  # For CUDA graphs (~0.8GB) + warmup + activations
        free_gb, total_gb = [x / 1024**3 for x in torch.cuda.mem_get_info()]
        max_safe_util = (free_gb - HEADROOM_GB) / total_gb
        configured_util = model_info.gpu_memory_utilization
        actual_util = min(configured_util, max(0.5, max_safe_util))

        if actual_util < configured_util:
            lost_kv_gb = (configured_util - actual_util) * total_gb
            logger.warning(
                f"Reducing gpu_memory_utilization: {configured_util:.2f} -> {actual_util:.2f} "
                f"(leaked {total_gb - free_gb:.1f}GB residual, {lost_kv_gb:.1f}GB less KV cache)"
            )

        engine_kwargs = {
            "model": model_info.hf_path,
            "gpu_memory_utilization": actual_util,
            "trust_remote_code": True,
            # CUDA graphs provide ~3x decode speedup (62→200 tok/s for gpt-oss-120b)
            # Trade-off: ~15s extra for graph capture on first load
        }

        # Only set max_model_len if explicitly configured
        if model_info.context_length is not None:
            engine_kwargs["max_model_len"] = model_info.context_length

        if model_info.quantization:
            engine_kwargs["quantization"] = model_info.quantization

        if model_info.dtype:
            engine_kwargs["dtype"] = model_info.dtype

        # Apply extra args
        if model_info.extra_args:
            engine_kwargs.update(model_info.extra_args)

        # Use pinned arena loader for fast GPU transfer if weights are preloaded
        if use_pinned:
            set_preloaded_weights(preloaded_weights)
            engine_kwargs["load_format"] = "pinned_arena"
            logger.info(f"Using pinned arena loader ({len(preloaded_weights)} tensors)")

        crash_log(f"_load_engine: LLM() starting")

        # Run model loading in thread pool with timeout to avoid blocking event loop
        loop = asyncio.get_event_loop()
        try:
            self.llm = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: LLM(**engine_kwargs)),
                timeout=MODEL_LOAD_TIMEOUT,
            )
        except asyncio.TimeoutError:
            crash_log(f"_load_engine: TIMEOUT loading {model_id} after {MODEL_LOAD_TIMEOUT}s")
            await self._cleanup_failed_load()
            raise RuntimeError(f"Model {model_id} load timed out after {MODEL_LOAD_TIMEOUT}s")
        except Exception as e:
            err_str = str(e)
            # Retry with reduced max_model_len if KV cache is marginally too small.
            # After many switches, non-pytorch memory drift can reduce available KV
            # cache by a fraction of a GiB, causing the model's native context to
            # just barely not fit.
            m = re.search(r"estimated maximum model length is (\d+)", err_str)
            if m and "max_model_len" not in engine_kwargs:
                reduced_len = int(m.group(1))
                logger.warning(
                    f"KV cache marginal for {model_id}, retrying with "
                    f"max_model_len={reduced_len} (was native)"
                )
                crash_log(f"_load_engine: KV margin retry max_model_len={reduced_len}")
                await self._cleanup_failed_load()
                engine_kwargs["max_model_len"] = reduced_len
                if use_pinned:
                    set_preloaded_weights(preloaded_weights)
                try:
                    self.llm = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: LLM(**engine_kwargs)),
                        timeout=MODEL_LOAD_TIMEOUT,
                    )
                except Exception as e2:
                    crash_log(f"_load_engine: FAILED retry {model_id}: {e2}")
                    logger.error(f"Failed to load model {model_id} (retry): {e2}")
                    await self._cleanup_failed_load()
                    raise
            else:
                crash_log(f"_load_engine: FAILED loading {model_id}: {e}")
                logger.error(f"Failed to load model {model_id}: {e}")
                await self._cleanup_failed_load()
                raise

        crash_log(f"_load_engine: LLM() done")

        self.current_model = model_id
        self._model_ready_time = time.time()
        self._first_request_pending = True

        elapsed = time.time() - start
        crash_log(f"_load_engine: {model_id} loaded in {elapsed:.1f}s")
        logger.info(f"Engine {model_id} loaded in {elapsed:.1f}s")

    async def _cleanup_failed_load(self):
        """Clean up GPU memory after a failed model load (OOM, timeout, etc.).

        When LLM() constructor fails mid-construction (e.g. OOM during CUDA graph
        capture), model weights + KV cache (80+ GB) remain allocated on GPU.

        CRITICAL: The partially-constructed LLM has NO llm_engine attribute set
        (the exception propagates before the assignment in LLM.__init__), so
        cleanup_vllm_model() can't navigate its internals. The model weights
        live inside EngineCore → ModelRunner objects that are separate in gc.

        Strategy (layered):
        1. Find orphaned LLM instances and try cleanup_vllm_model()
        2. Find ModelRunner objects directly via gc and walk their model graphs
        3. Run detach_all_cuda_tensors() as safety net for any remaining tensors
        4. Destroy CUDA graphs (releases references to captured tensors)
        5. Final gc + empty_cache
        """
        crash_log("_cleanup_failed_load: cleaning up after failed model load")
        logger.warning("Cleaning up GPU memory after failed model load...")

        mem_before = log_memory_state("BEFORE_FAILED_LOAD_CLEANUP")

        try:
            from blitzinfer.engine.cleanup import (
                _walk_and_free_cuda_tensors,
                cleanup_vllm_model,
                detach_all_cuda_tensors,
                clear_cuda_graph_pools,
                destroy_parallel_state,
                clear_vllm_caches,
            )

            # Run gc to ensure the failed constructor's frame is cleaned up
            gc.collect()
            gc.collect()

            # Strategy 1: Find orphaned LLM instances (works when llm_engine is set)
            orphaned_count = 0
            for obj in gc.get_objects():
                if type(obj).__name__ == 'LLM' and obj is not self.llm:
                    has_engine = hasattr(obj, 'llm_engine') and obj.llm_engine is not None
                    logger.info(f"Found orphaned LLM (has_engine={has_engine}), running cleanup_vllm_model...")
                    try:
                        cleanup_vllm_model(obj)
                        orphaned_count += 1
                    except Exception as e:
                        logger.warning(f"Orphaned LLM cleanup error: {e}")

            alloc_after_llm = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"After LLM cleanup ({orphaned_count} LLMs): alloc={alloc_after_llm:.1f}GB")

            # Strategy 2: Find ModelRunner objects directly via gc.
            # When LLM() fails mid-construction, llm_engine is never set, so
            # cleanup_vllm_model can't navigate to the model. But ModelRunner
            # objects exist separately in gc with .model and .kv_caches attributes.
            runners_cleaned = 0
            tensors_freed = 0
            for obj in gc.get_objects():
                try:
                    typename = type(obj).__name__
                    # GPUModelRunner (V1) or ModelRunner (V0) - both have .model and .kv_caches
                    if 'ModelRunner' in typename:
                        model = getattr(obj, 'model', None)
                        if model is not None and isinstance(model, torch.nn.Module):
                            # Clear static_forward_context first (same as cleanup_vllm_model)
                            for module in model.modules():
                                vc = getattr(module, 'vllm_config', None)
                                if vc is not None:
                                    cc = getattr(vc, 'compilation_config', None)
                                    if cc is not None:
                                        sfc = getattr(cc, 'static_forward_context', None)
                                        if sfc is not None and isinstance(sfc, dict) and len(sfc) > 0:
                                            sfc.clear()
                                            break

                            n = _walk_and_free_cuda_tensors(model)
                            tensors_freed += n
                            obj.model = None
                            logger.info(f"ModelRunner walker freed {n} CUDA tensors")

                        # Free KV caches
                        kv_caches = getattr(obj, 'kv_caches', None)
                        if kv_caches:
                            for kv in kv_caches:
                                if isinstance(kv, torch.Tensor) and kv.device.type == 'cuda':
                                    try:
                                        kv.storage().resize_(0)
                                        tensors_freed += 1
                                    except Exception:
                                        pass
                            kv_caches.clear()

                        runners_cleaned += 1
                except Exception:
                    pass

            if runners_cleaned > 0:
                gc.collect()
                torch.cuda.empty_cache()
                alloc_after_runners = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"After ModelRunner cleanup ({runners_cleaned} runners, {tensors_freed} tensors): alloc={alloc_after_runners:.1f}GB")

            # Strategy 3: Detach ALL remaining CUDA tensors via gc scan
            detach_all_cuda_tensors()

            # Clear CUDA graphs (releases references to captured tensors)
            clear_cuda_graph_pools()

            # Clear vLLM caches and parallel state
            clear_vllm_caches()
            destroy_parallel_state()

            # Clear cuBLAS workspaces
            try:
                torch._C._cuda_clearCublasWorkspaces()
            except Exception:
                pass

            # Multi-round GC: destroying CUDA graphs and vLLM caches may have
            # released the last references to tensor storages. GC + empty_cache
            # converts those to free memory.
            for _ in range(3):
                gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        except Exception as e:
            logger.warning(f"Failed load cleanup error: {e}")
            traceback.print_exc()

        mem_after = log_memory_state("AFTER_FAILED_LOAD_CLEANUP")
        freed = mem_before['gpu_used_gb'] - mem_after['gpu_used_gb']
        crash_log(f"_cleanup_failed_load: freed ~{freed:.1f}GB GPU")
        logger.info(f"Failed load cleanup freed ~{freed:.1f}GB GPU")

    async def _shutdown_engine(self):
        """Shutdown current vLLM LLM and release GPU memory."""
        if self.llm is None:
            return

        crash_log(f"_shutdown_engine: shutting down {self.current_model}")
        logger.info(f"Shutting down engine for {self.current_model}...")

        mem_before = log_memory_state("BEFORE_ENGINE_SHUTDOWN")

        try:
            # Use full_cleanup from blitzinfer which properly handles vLLM V1 InprocClient
            freed = full_cleanup(self.llm)
            logger.info(f"full_cleanup freed {freed:.1f}GB")
        except Exception as e:
            logger.warning(f"Engine cleanup error: {e}")

        self.llm = None

        # Extra gc passes
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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

    async def _switch_model(self, target_model: str):
        """Switch to a different model, using pinned arena if available."""
        if self.current_model == target_model:
            return

        self.switch_count += 1
        crash_log(f"=== SWITCH #{self.switch_count}: {self.current_model} -> {target_model} ===")
        logger.info("=" * 60)
        logger.info(f"MODEL SWITCH #{self.switch_count}: {self.current_model} -> {target_model}")
        logger.info("=" * 60)

        start = time.time()
        self._switch_start_time = start

        # Try to get preloaded weights from standby manager.
        # The pinned arena loader delegates to vLLM's model.load_weights(),
        # so it works for ALL model types (vision, AWQ, MoE, FLA, etc.)
        preloaded_weights = None
        model_info = AVAILABLE_MODELS.get(target_model)

        if self.standby and model_info:
            hf_path = model_info.hf_path
            standby_state = self.standby.get_state()

            if self.standby.is_ready(hf_path):
                # Weights already prefetched and ready
                logger.info(f"Standby READY for {target_model} - using pinned arena")
                preloaded_weights = self.standby.consume_standby()

            elif standby_state.name == "LOADING":
                # Prefetch in progress - wait for it
                logger.info("Prefetch in progress, waiting...")
                wait_start = time.time()
                ready = self.standby.wait_for_load(timeout=120.0)
                wait_time = time.time() - wait_start
                if ready and self.standby.is_ready(hf_path):
                    logger.info(f"Prefetch completed in {wait_time:.1f}s - using pinned arena")
                    preloaded_weights = self.standby.consume_standby()
                else:
                    logger.warning(f"Prefetch wait failed after {wait_time:.1f}s, falling back to disk")

            else:
                # No prefetch running - start one and wait
                logger.info(f"No prefetch for {target_model}, starting now...")
                prefetch_start = time.time()
                self.standby.start_prefetch(hf_path)
                ready = self.standby.wait_for_load(timeout=120.0)
                prefetch_time = time.time() - prefetch_start
                if ready and self.standby.is_ready(hf_path):
                    logger.info(f"On-demand prefetch completed in {prefetch_time:.1f}s - using pinned arena")
                    preloaded_weights = self.standby.consume_standby()
                else:
                    logger.warning(f"On-demand prefetch failed after {prefetch_time:.1f}s, falling back to disk")

        t_prefetch = time.time() - start

        t_shutdown_start = time.time()
        await self._shutdown_engine()
        # Clear current model tracking AFTER shutdown - if new model fails to load,
        # this ensures recovery logic triggers (checks self.current_model is None)
        self.current_model = None
        t_shutdown = time.time() - t_shutdown_start

        mem_ok, mem_msg = verify_memory_available(min_gpu_free_gb=5.0, min_ram_avail_gb=2.0)
        if not mem_ok:
            raise MemoryError(f"Insufficient memory to load {target_model}: {mem_msg}")

        t_load_start = time.time()
        await self._load_engine(target_model, preloaded_weights=preloaded_weights)
        t_load = time.time() - t_load_start

        # Free pinned arena immediately - data is on GPU now, no reason to keep it
        t_arena_start = time.time()
        if preloaded_weights is not None and self.standby:
            self.standby.free_arena()
        t_arena_free = time.time() - t_arena_start

        log_memory_state("AFTER_SWITCH")
        elapsed = time.time() - start
        crash_log(f"=== SWITCH #{self.switch_count} COMPLETE in {elapsed:.1f}s ===")

        # Detailed timing breakdown
        logger.info(
            f"SWITCH TIMING #{self.switch_count} ({self.current_model}): "
            f"total={elapsed:.2f}s | "
            f"prefetch_wait={t_prefetch:.2f}s | "
            f"shutdown={t_shutdown:.2f}s | "
            f"load_engine={t_load:.2f}s | "
            f"arena_free={t_arena_free:.3f}s"
        )

    def _find_next_model(self) -> Optional[str]:
        """Find the next model that has queued requests."""
        for model_id, queue in self.queues.items():
            if model_id != self.current_model and not queue.empty():
                return model_id
        return None

    async def _queue_processor(self):
        """Background task: dequeue requests from active model and process them.

        CRITICAL: This task must NEVER die from an unhandled exception.
        If it does, all queued requests will hang forever.
        """
        logger.info("Queue processor running")

        consecutive_errors = 0
        max_consecutive_errors = 10

        while True:
            try:
                if self.queue_state == QueueState.SERVING:
                    await self._process_serving()

                elif self.queue_state == QueueState.DRAINING:
                    await self._process_draining()

                elif self.queue_state == QueueState.SWITCHING:
                    await self._process_switching()

                consecutive_errors = 0  # Reset on success

            except asyncio.CancelledError:
                logger.info("Queue processor cancelled")
                return
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Queue processor error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                logger.error(traceback.format_exc())

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Queue processor hit {max_consecutive_errors} consecutive errors, resetting to SERVING")
                    self.queue_state = QueueState.SERVING
                    self._target_switch_model = None
                    consecutive_errors = 0

                await asyncio.sleep(min(1.0 * consecutive_errors, 5.0))

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

        # Process the request (serialize with _generate_lock to prevent
        # concurrent llm.generate() from fast-path requests)
        self.in_flight += 1
        try:
            async with self._generate_lock:
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
            logger.error(f"Model switch to {target} failed: {e}")
            logger.error(traceback.format_exc())

            # Mark model as temporarily failed
            self._failed_models[target] = time.time()
            crash_log(f"Model {target} marked as failed: {e}")

            # Fail all queued requests for the target model with a proper error
            target_queue = self.queues.get(target)
            error = RuntimeError(f"Model {target} failed to load: {e}")
            if target_queue:
                drained = 0
                while not target_queue.empty():
                    try:
                        queued = target_queue.get_nowait()
                        if not queued.future.done():
                            queued.future.set_exception(error)
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
                logger.info(f"Drained {drained} requests from failed model {target} queue")

            # If we have no active engine, try to recover by loading a known-good model
            if self.llm is None and self.current_model is None:
                logger.warning("No active engine after failed switch, attempting recovery...")
                for fallback in ["gpt-oss-120b", "qwen3-32b", "qwen2.5-7b"]:
                    if fallback != target:
                        try:
                            await self._load_engine(fallback)
                            logger.info(f"Recovery: loaded fallback model {fallback}")
                            break
                        except Exception as fe:
                            logger.error(f"Recovery failed for {fallback}: {fe}")

        self._target_switch_model = None
        self.queue_state = QueueState.SERVING

    async def enqueue_request(self, request: ChatCompletionRequest, model_id: str) -> asyncio.Future:
        """Enqueue a request for a specific model. Returns a Future with the response."""
        queue = self.queues.get(model_id)
        if queue is None:
            raise HTTPException(status_code=404, detail=f"No queue for model: {model_id}")

        if queue.full():
            raise HTTPException(status_code=429, detail=f"Queue full for model {model_id}")

        # Check if model recently failed to load
        if model_id in self._failed_models:
            fail_time = self._failed_models[model_id]
            elapsed = time.time() - fail_time
            if elapsed < self._failed_model_cooldown:
                remaining = self._failed_model_cooldown - elapsed
                raise HTTPException(
                    status_code=503,
                    detail=f"Model {model_id} temporarily unavailable (failed to load {elapsed:.0f}s ago, retry in {remaining:.0f}s)"
                )
            else:
                # Cooldown expired, allow retry
                del self._failed_models[model_id]
                logger.info(f"Model {model_id} cooldown expired, allowing retry")

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
            if model_info and self.standby:
                logger.info(f"[{request_id}] Triggering prefetch for {model_id}")
                self.standby.start_prefetch(model_info.hf_path)

        return future, request_id

    async def _generate_response(
        self, request: ChatCompletionRequest, request_id: str
    ) -> Union[ChatCompletionResponse, StreamingResponse]:
        """Generate a response using the vLLM engine."""
        if self.llm is None:
            raise RuntimeError(f"Engine not available (model switching in progress)")

        # Capture local reference to prevent race with _shutdown_engine
        llm = self.llm
        model_id = self.current_model
        model_info = AVAILABLE_MODELS[model_id]

        self.request_count += 1
        is_first = self._first_request_pending
        t_request_start = time.time()

        logger.info(f"[{request_id}] Generating: model={model_id}, "
                    f"msgs={len(request.messages)}, max_tokens={request.max_tokens}"
                    f"{' [FIRST REQUEST]' if is_first else ''}")

        # Session logging: full request details
        session_logger.info(f"=== REQUEST {request_id} model={model_id} ===")
        session_logger.info(f"Messages ({len(request.messages)}):")
        for i, msg in enumerate(request.messages):
            session_logger.info(f"  [{i}] role={msg.role} content={str(msg.content)[:500]}")
            if msg.tool_calls:
                session_logger.info(f"      tool_calls={msg.tool_calls}")
            if msg.tool_call_id:
                session_logger.info(f"      tool_call_id={msg.tool_call_id}")
        if request.tools:
            session_logger.info(f"Tools: {json.dumps([t.model_dump() for t in request.tools], indent=2)[:1000]}")
        session_logger.info(f"Params: max_tokens={request.max_tokens} temp={request.temperature} stream={request.stream}")

        use_harmony = HAS_HARMONY and model_id == "gpt-oss-120b"
        has_tools = request.tools and len(request.tools) > 0

        # Build prompt
        prompt = None
        prompt_token_ids = None
        image_data = None

        if use_harmony:
            prompt_token_ids = self._build_harmony_prompt(
                request, request_id, has_tools
            )
        if prompt_token_ids is None:
            prompt, image_data = self._build_chat_prompt(
                request, request_id, model_info, has_tools=has_tools
            )

        # Build sampling params
        stop = None
        if request.stop:
            stop = request.stop if isinstance(request.stop, list) else [request.stop]

        stop_token_ids = HARMONY_STOP_TOKENS if use_harmony else None

        sampling_params = SamplingParams(
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            presence_penalty=request.presence_penalty or 0.0,
            frequency_penalty=request.frequency_penalty or 0.0,
            stop=stop,
            stop_token_ids=stop_token_ids,
        )

        # Session logging: prompt
        if prompt is not None:
            session_logger.info(f"[{request_id}] PROMPT ({len(prompt)} chars):\n{prompt[:2000]}")
        elif prompt_token_ids is not None:
            session_logger.info(f"[{request_id}] PROMPT_TOKEN_IDS ({len(prompt_token_ids)} tokens)")

        # Generate
        t_prompt_built = time.time()
        prompt_build_time = t_prompt_built - t_request_start

        if request.stream:
            return self._build_streaming_response(
                request, request_id, model_id, prompt, prompt_token_ids,
                sampling_params, use_harmony, has_tools, llm
            )

        # Non-streaming generation - run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        t_gen_start = time.time()

        try:
            if prompt_token_ids is not None:
                outputs = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: llm.generate(
                            [{"prompt_token_ids": prompt_token_ids}],
                            sampling_params
                        )
                    ),
                    timeout=GENERATION_TIMEOUT,
                )
            else:
                outputs = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: llm.generate([prompt], sampling_params)
                    ),
                    timeout=GENERATION_TIMEOUT,
                )
        except asyncio.TimeoutError:
            logger.error(f"[{request_id}] Generation timed out after {GENERATION_TIMEOUT}s")
            raise RuntimeError(f"Generation timed out after {GENERATION_TIMEOUT}s")

        t_gen_end = time.time()
        output = outputs[0]
        gen_time = t_gen_end - t_gen_start

        generated_text = output.outputs[0].text
        output_token_ids = output.outputs[0].token_ids
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output_token_ids)
        total_tokens = prompt_tokens + completion_tokens
        self.total_tokens += total_tokens

        elapsed = t_gen_end - t_request_start
        tokens_per_sec = completion_tokens / gen_time if gen_time > 0 else 0
        logger.info(f"[{request_id}] Generated {completion_tokens} tokens "
                    f"in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")

        # Detailed timing for first request after model load/switch
        if is_first:
            self._first_request_pending = False
            time_since_ready = t_request_start - self._model_ready_time
            time_since_switch = t_request_start - self._switch_start_time if self._switch_start_time > 0 else 0
            # Estimate TTFT: prompt processing is roughly (gen_time - completion_tokens/tok_rate)
            # but vLLM batches, so gen_time includes both prefill + decode
            # A rough estimate: if decode is at tok/s rate, prefill is the remainder
            decode_time_est = completion_tokens / tokens_per_sec if tokens_per_sec > 0 else gen_time
            prefill_time_est = gen_time - decode_time_est
            ttft_est = prompt_build_time + max(prefill_time_est, 0)
            logger.info(
                f"[{request_id}] FIRST REQUEST TIMING ({model_id}):\n"
                f"  prompt_build={prompt_build_time:.3f}s | "
                f"llm.generate={gen_time:.2f}s | "
                f"total={elapsed:.2f}s\n"
                f"  prompt_tokens={prompt_tokens} | "
                f"completion_tokens={completion_tokens} | "
                f"tok/s={tokens_per_sec:.1f}\n"
                f"  time_since_model_ready={time_since_ready:.2f}s | "
                f"time_since_switch_start={time_since_switch:.2f}s\n"
                f"  est_prefill={prefill_time_est:.3f}s | "
                f"est_decode={decode_time_est:.2f}s | "
                f"est_ttft={ttft_est:.3f}s"
            )

        # Get finish reason
        finish_reason = output.outputs[0].finish_reason or "stop"

        # Parse Harmony output for GPT-OSS
        tool_calls_list = []
        if use_harmony and output_token_ids:
            generated_text, tool_calls_list = self._parse_harmony_output(
                list(output_token_ids), request_id, has_tools
            )
            if tool_calls_list:
                finish_reason = "tool_calls"

        # Parse tool calls from non-Harmony models (e.g. Qwen3 XML format)
        if not use_harmony and has_tools and "<tool_call>" in generated_text:
            tool_calls_list, remaining_text = self._parse_tool_calls(generated_text, request_id)
            if tool_calls_list:
                generated_text = remaining_text
                finish_reason = "tool_calls"

        # Extract reasoning content from thinking models (Kimi-VL, Qwen3, etc.)
        reasoning_content = None
        if not use_harmony and model_info.is_thinking_model and generated_text:
            reasoning_content, content = extract_reasoning_content(generated_text)
            if content is not None:
                generated_text = content
            elif reasoning_content is not None:
                # Model was truncated mid-reasoning, no final content
                generated_text = ""

        # Session logging: raw output + parsed result
        session_logger.info(f"[{request_id}] RAW OUTPUT ({len(output.outputs[0].text)} chars):\n{output.outputs[0].text[:3000]}")
        session_logger.info(
            f"[{request_id}] PARSED: content={generated_text[:500] if generated_text else None} "
            f"reasoning={reasoning_content[:200] if reasoning_content else None} "
            f"tool_calls={tool_calls_list}"
        )

        # Log response
        if generated_text:
            preview = generated_text[:500] + "..." if len(generated_text) > 500 else generated_text
        else:
            generated_text = ""
            preview = "(empty)"
        if reasoning_content:
            logger.info(f"[{request_id}] REASONING: {reasoning_content[:200]}...")
        logger.info(f"[{request_id}] RESPONSE: {preview}")

        response_message = ResponseMessage(
            role="assistant",
            content=generated_text if generated_text else None,
            reasoning_content=reasoning_content if reasoning_content else None,
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
    ) -> Optional[List[int]]:
        """Build Harmony-encoded prompt for GPT-OSS. Returns prompt_token_ids or None."""
        try:
            # Convert messages to dicts for Harmony
            chat_msgs = [{"role": m.role, "content": m.content} for m in request.messages if m.content]

            # Get system message (with custom tools flag)
            sys_msg = get_system_message(with_custom_tools=has_tools)

            # Parse chat inputs to harmony messages
            harmony_msgs = [sys_msg] + parse_chat_inputs_to_harmony_messages(chat_msgs)

            # Add developer message with tools if needed
            if has_tools:
                tools_for_harmony = []
                for tool in request.tools:
                    func_def = VLLMFunctionDefinition(
                        name=tool.function.name,
                        description=tool.function.description,
                        parameters=tool.function.parameters,
                    )
                    tool_param = ChatCompletionToolsParam(type="function", function=func_def)
                    tools_for_harmony.append(tool_param)
                dev_msg = get_developer_message(tools=tools_for_harmony)
                harmony_msgs.insert(1, dev_msg)  # After system, before user msgs
                logger.info(f"[{request_id}] Added {len(request.tools)} tools to Harmony")

            prompt_token_ids = render_for_completion(harmony_msgs)
            logger.debug(f"[{request_id}] Harmony prompt: {len(prompt_token_ids)} tokens")
            return prompt_token_ids

        except Exception as e:
            logger.warning(f"[{request_id}] Harmony encoding failed: {e}, falling back to chat template")
            logger.warning(traceback.format_exc())
            return None

    def _build_chat_prompt(
        self, request: ChatCompletionRequest, request_id: str, model_info: ModelInfo,
        has_tools: bool = False
    ) -> Tuple[str, Optional[Any]]:
        """Build a text prompt using the model's chat template."""
        image_data = None

        # Extract image data from multimodal messages
        if model_info.supports_vision:
            image_data = self._extract_image_data(request.messages)

        # Convert to dicts for tokenizer
        messages_dicts = []
        for msg in request.messages:
            d = {"role": msg.role}
            if isinstance(msg.content, str):
                d["content"] = msg.content
            elif isinstance(msg.content, list):
                text_parts = []
                for part in msg.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            text_parts.append("<image>")
                d["content"] = "\n".join(text_parts) if text_parts else ""
            else:
                d["content"] = str(msg.content) if msg.content else ""

            # Preserve tool_calls for assistant messages
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": (
                                json.loads(tc.function.arguments)
                                if isinstance(tc.function.arguments, str)
                                else tc.function.arguments
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Preserve tool_call_id for tool result messages
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id

            messages_dicts.append(d)

        # Use vLLM's tokenizer to apply the chat template
        try:
            tokenizer = self.llm.get_tokenizer()
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if has_tools and request.tools:
                template_kwargs["tools"] = [t.model_dump() for t in request.tools]
            prompt = tokenizer.apply_chat_template(
                messages_dicts, **template_kwargs
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
    # Non-Harmony tool call parsing (Qwen3 XML format)
    # -------------------------------------------------------------------------

    def _parse_tool_calls(
        self, text: str, request_id: str
    ) -> Tuple[List[ToolCall], str]:
        """Parse tool calls from Qwen3-style XML output.

        Format:
            <tool_call>
            <function=function_name>
            <parameter=param_name>value</parameter>
            </function>
            </tool_call>

        Returns (tool_calls_list, remaining_text_before_tool_calls).
        """
        tool_calls_list = []

        # Split text at first <tool_call> to get content before tool calls
        first_tc = text.find("<tool_call>")
        remaining_text = text[:first_tc].strip() if first_tc >= 0 else text

        # Find all <tool_call>...</tool_call> blocks
        tc_pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
        for match in tc_pattern.finditer(text):
            block = match.group(1)

            # Extract function name
            func_match = re.search(r'<function=(\S+?)>', block)
            if not func_match:
                logger.warning(f"[{request_id}] Tool call block missing function name: {block[:100]}")
                continue
            func_name = func_match.group(1)

            # Extract parameters
            params = {}
            param_pattern = re.compile(r'<parameter=(\S+?)>(.*?)</parameter>', re.DOTALL)
            for pm in param_pattern.finditer(block):
                params[pm.group(1)] = pm.group(2).strip()

            call_id = f"call_{uuid.uuid4().hex[:24]}"
            tool_calls_list.append(ToolCall(
                id=call_id,
                type="function",
                function=FunctionCall(
                    name=func_name,
                    arguments=json.dumps(params),
                ),
            ))
            logger.info(f"[{request_id}] Parsed tool call: {func_name}({json.dumps(params)[:200]})")

        return tool_calls_list, remaining_text

    # -------------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------------

    def _build_streaming_response(
        self, request: ChatCompletionRequest, request_id: str,
        model_id: str, prompt: Optional[str], prompt_token_ids: Optional[List[int]],
        sampling_params: SamplingParams, use_harmony: bool, has_tools: bool,
        llm: "LLM" = None,
    ) -> StreamingResponse:
        """Build a streaming SSE response."""

        async def event_stream():
            created = int(time.time())
            prev_text = ""
            loop = asyncio.get_event_loop()

            try:
                # vLLM streaming - use generate() with stream=False but iterate outputs
                # For true streaming, we'd use AsyncLLMEngine, but for simplicity
                # we buffer the response here

                if prompt_token_ids is not None:
                    outputs = await loop.run_in_executor(
                        None,
                        lambda: llm.generate(
                            [{"prompt_token_ids": prompt_token_ids}],
                            sampling_params
                        )
                    )
                else:
                    outputs = await loop.run_in_executor(
                        None,
                        lambda: llm.generate([prompt], sampling_params)
                    )

                output = outputs[0]
                generated_text = output.outputs[0].text
                output_token_ids = list(output.outputs[0].token_ids)
                finish_reason = output.outputs[0].finish_reason or "stop"

                if use_harmony and output_token_ids:
                    final_text, tool_calls = self._parse_harmony_output(
                        output_token_ids, request_id, has_tools
                    )

                    if tool_calls:
                        finish_reason = "tool_calls"
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
                    # Non-Harmony: parse tool calls, then extract reasoning
                    model_info = AVAILABLE_MODELS[model_id]

                    # Parse tool calls from XML format (Qwen3, etc.)
                    tool_calls = []
                    if has_tools and "<tool_call>" in generated_text:
                        tool_calls, generated_text = self._parse_tool_calls(generated_text, request_id)
                        if tool_calls:
                            finish_reason = "tool_calls"

                    reasoning_content = None
                    content = generated_text
                    if model_info.is_thinking_model and generated_text:
                        reasoning_content, content = extract_reasoning_content(generated_text)
                        if content is None and reasoning_content is not None:
                            content = ""  # Truncated mid-reasoning

                    # Session logging
                    session_logger.info(f"[{request_id}] STREAM RAW ({len(output.outputs[0].text)} chars):\n{output.outputs[0].text[:3000]}")
                    session_logger.info(
                        f"[{request_id}] STREAM PARSED: content={content[:500] if content else None} "
                        f"reasoning={reasoning_content[:200] if reasoning_content else None} "
                        f"tool_calls={tool_calls}"
                    )

                    # Emit reasoning_content chunk if present
                    if reasoning_content:
                        data = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {"reasoning_content": reasoning_content},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                    # Emit tool calls as chunks
                    if tool_calls:
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
                    elif content:
                        # Emit content chunk (only if no tool calls)
                        data = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                # Final chunk with finish_reason
                final_data = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
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
        if model_info and model_id != self.current_model and self.standby:
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
    if state.llm:
        try:
            full_cleanup(state.llm)
        except Exception:
            pass


app = FastAPI(
    title="BlitzInfer API",
    description="Fast LLM serving with queue-driven model switching (vLLM)",
    version="0.3.0",
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
        # If this model is already active and queue is empty, fast-path directly.
        # Use _generate_lock to prevent concurrent llm.generate() calls
        # (vLLM's LLM class is not thread-safe for concurrent generate calls).
        if (model_id == state.current_model and state.llm is not None
                and state.queues[model_id].empty() and state.in_flight == 0
                and state.queue_state == QueueState.SERVING):
            async with state._generate_lock:
                # Re-check conditions after acquiring lock (another request may have started)
                if (model_id == state.current_model and state.llm is not None
                        and state.queue_state == QueueState.SERVING):
                    request_id = str(uuid.uuid4())[:8]
                    state.request_count += 1
                    state.in_flight += 1
                    try:
                        logger.info(f"[{request_id}] Fast-path: model={model_id}")
                        return await state._generate_response(request, request_id)
                    finally:
                        state.in_flight -= 1

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
    failed = {mid: int(time.time() - ts) for mid, ts in state._failed_models.items()}
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
        "failed_models": failed,
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

    logger.info("Starting BlitzInfer API server (vLLM + Queues)...")
    logger.info(f"Crash log: {CRASH_LOG_FILE}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
