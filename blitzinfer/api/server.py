"""OpenAI-compatible API server for BlitzInfer."""

import asyncio
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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Set single-process mode BEFORE importing vLLM
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
# Allow overriding model context lengths
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

from vllm import LLM, SamplingParams

# Import Harmony parser for GPT-OSS input/output formatting
try:
    from vllm.entrypoints.openai.parser.harmony_utils import (
        parse_chat_output,
        parse_chat_inputs_to_harmony_messages,
        render_for_completion,
        get_system_message,
        get_developer_message,
        parse_output_into_messages,
        parse_output_message,
        get_stop_tokens_for_assistant_actions,
    )
    from openai.types.responses import ResponseFunctionToolCall
    HAS_HARMONY = True
    # GPT-OSS requires specific stop tokens to properly terminate generation
    HARMONY_STOP_TOKENS = get_stop_tokens_for_assistant_actions()
except ImportError:
    HAS_HARMONY = False
    HARMONY_STOP_TOKENS = []

# BlitzInfer imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from blitzinfer.engine.cleanup import full_cleanup
from blitzinfer.memory import set_preloaded_weights
from blitzinfer.orchestrator.standby_manager import StandbyManager

# Configure extensive logging with IMMEDIATE flush to disk
# This ensures we capture logs up to milliseconds before any crash
class FlushingFileHandler(logging.FileHandler):
    """File handler that flushes and fsyncs after every write."""
    def emit(self, record):
        super().emit(record)
        self.flush()
        if self.stream:
            os.fsync(self.stream.fileno())

class FlushingStreamHandler(logging.StreamHandler):
    """Stream handler that flushes after every write."""
    def emit(self, record):
        super().emit(record)
        self.flush()

# Use home directory for crash log so it survives reboots
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

# Reduce noise from some libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


def crash_log(msg: str):
    """Write directly to crash log with immediate sync - for critical moments."""
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
    """Model configuration."""
    name: str  # Display name / model ID for API
    hf_path: str  # HuggingFace path or local path
    context_length: int = 131072  # 128K default
    gpu_memory_utilization: float = 0.88
    supports_vision: bool = False
    quantization: Optional[str] = None  # 'fp8', 'awq', etc.
    extra_args: Dict[str, Any] = field(default_factory=dict)


# Available models on RTX PRO 6000 (95GB VRAM)
# Use 94% GPU memory - slightly reduced to leave headroom during model switches
GPU_MEM_UTIL = 0.94  # Use ~89GB of 95GB VRAM

AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    # GPT-OSS-120B - largest model, MXFP4 quantized (native 128K)
    "gpt-oss-120b": ModelInfo(
        name="gpt-oss-120b",
        hf_path="openai/gpt-oss-120b",
        context_length=131072,  # 128K native
        gpu_memory_utilization=GPU_MEM_UTIL,
    ),
    # Qwen3 VL 32B with thinking - vision model (limited to 128K for memory)
    "qwen3-vl-32b-thinking": ModelInfo(
        name="qwen3-vl-32b-thinking",
        hf_path="Qwen/Qwen3-VL-32B-Thinking-FP8",
        context_length=131072,  # 128K (256K native but limited for KV cache memory)
        gpu_memory_utilization=GPU_MEM_UTIL,
        supports_vision=True,
        quantization="fp8",
    ),
    # Qwen3 32B FP8 - fast text model (RoPE scaling to 128K)
    "qwen3-32b": ModelInfo(
        name="qwen3-32b",
        hf_path="Qwen/Qwen3-32B-FP8",
        context_length=131072,  # 128K with RoPE scaling
        gpu_memory_utilization=GPU_MEM_UTIL,
        quantization="fp8",
    ),
    # Mistral Small 24B - efficient model (native 128K)
    "mistral-small-24b": ModelInfo(
        name="mistral-small-24b",
        hf_path="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        context_length=131072,  # 128K native
        gpu_memory_utilization=GPU_MEM_UTIL,
    ),
    # Llama 3.1 70B AWQ - popular model (native 128K)
    "llama-3.1-70b": ModelInfo(
        name="llama-3.1-70b",
        hf_path="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        context_length=131072,  # 128K native
        gpu_memory_utilization=GPU_MEM_UTIL,
        quantization="awq",
    ),
    # Qwen 2.5 72B - large Qwen model (RoPE scaling to 128K)
    "qwen2.5-72b": ModelInfo(
        name="qwen2.5-72b",
        hf_path="Qwen/Qwen2.5-72B-Instruct",
        context_length=131072,  # 128K with RoPE scaling
        gpu_memory_utilization=GPU_MEM_UTIL,
    ),
    # Kimi VL - vision model (native 128K)
    "kimi-vl": ModelInfo(
        name="kimi-vl",
        hf_path="moonshotai/Kimi-VL-A3B-Thinking-2506",
        context_length=131072,  # 128K
        gpu_memory_utilization=GPU_MEM_UTIL,
        supports_vision=True,
    ),
    # GLM 4.6V Flash - DISABLED: vLLM processor incompatibility
    # Error: "Invalid type of HuggingFace processor. Expected ProcessorMixin, found PreTrainedTokenizerFast"
    # "glm-4.6v-flash": ModelInfo(
    #     name="glm-4.6v-flash",
    #     hf_path="zai-org/GLM-4.6V-Flash",
    #     context_length=131072,
    #     gpu_memory_utilization=GPU_MEM_UTIL,
    #     supports_vision=True,
    # ),
}

# Model aliases for convenience
MODEL_ALIASES = {
    "gpt-oss": "gpt-oss-120b",
    "qwen-vl": "qwen3-vl-32b-thinking",
    "qwen": "qwen3-32b",
    "mistral": "mistral-small-24b",
    "llama": "llama-3.1-70b",
    "kimi": "kimi-vl",
    "glm": "glm-4.6v-flash",
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
# Memory Monitoring
# =============================================================================

def get_memory_info() -> Dict[str, float]:
    """Get current memory state for logging."""
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    gpu_allocated = torch.cuda.memory_allocated()
    gpu_reserved = torch.cuda.memory_reserved()

    # System memory via /proc/meminfo
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(':')] = int(parts[1]) * 1024  # Convert KB to bytes
        ram_total = meminfo.get('MemTotal', 0)
        ram_free = meminfo.get('MemFree', 0)
        ram_available = meminfo.get('MemAvailable', 0)
        ram_shared = meminfo.get('Shmem', 0)
    except Exception:
        ram_total = ram_free = ram_available = ram_shared = 0

    return {
        'gpu_used_gb': (gpu_total - gpu_free) / 1024**3,
        'gpu_free_gb': gpu_free / 1024**3,
        'gpu_total_gb': gpu_total / 1024**3,
        'gpu_allocated_gb': gpu_allocated / 1024**3,
        'gpu_reserved_gb': gpu_reserved / 1024**3,
        'ram_total_gb': ram_total / 1024**3,
        'ram_free_gb': ram_free / 1024**3,
        'ram_available_gb': ram_available / 1024**3,
        'ram_shared_gb': ram_shared / 1024**3,
    }


def log_memory_state(label: str):
    """Log current memory state."""
    mem = get_memory_info()
    logger.info(f"[MEMORY {label}] GPU: {mem['gpu_used_gb']:.1f}/{mem['gpu_total_gb']:.1f}GB used "
                f"(alloc={mem['gpu_allocated_gb']:.1f}GB, rsv={mem['gpu_reserved_gb']:.1f}GB) | "
                f"RAM: {mem['ram_available_gb']:.1f}GB avail, {mem['ram_shared_gb']:.1f}GB shared")
    return mem


def verify_memory_available(min_gpu_free_gb: float = 10.0, min_ram_avail_gb: float = 5.0) -> Tuple[bool, str]:
    """Verify sufficient memory is available before model load."""
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
# Server State
# =============================================================================

class ServerState:
    """Global server state."""

    def __init__(self):
        self.llm: Optional[LLM] = None
        self.current_model: Optional[str] = None
        self.standby: Optional[StandbyManager] = None
        self.request_count: int = 0
        self.switch_count: int = 0
        self.total_tokens: int = 0
        self.start_time: float = time.time()
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the standby manager."""
        logger.info("=" * 80)
        logger.info("BLITZINFER SERVER INITIALIZING")
        logger.info("=" * 80)

        # Initialize standby manager with 80GB arena
        logger.info("Creating StandbyManager with 80GB pinned arena (5x16GB chunks)...")
        start = time.time()
        self.standby = StandbyManager(
            arena_size_gb=80.0,
            chunk_size_gb=16.0,
            pin_memory=True,
            lazy_arena=False,  # Pre-allocate now
        )
        elapsed = time.time() - start
        logger.info(f"StandbyManager initialized in {elapsed:.1f}s")

        # Load initial model (gpt-oss-120b as default)
        initial_model = "gpt-oss-120b"
        logger.info(f"Loading initial model: {initial_model}")
        await self._load_model(initial_model)

        logger.info("=" * 80)
        logger.info("BLITZINFER SERVER READY")
        logger.info(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        logger.info("=" * 80)

    async def _load_model(self, model_id: str, use_standby: bool = False):
        """Load a model (cold or from standby)."""
        model_info = AVAILABLE_MODELS.get(model_id)
        if not model_info:
            raise ValueError(f"Unknown model: {model_id}")

        start = time.time()
        crash_log(f"_load_model: {model_id}, hf_path={model_info.hf_path}, standby={use_standby}")
        logger.info(f"Loading model {model_id} (standby={use_standby})...")

        if use_standby and self.standby.is_ready(model_info.hf_path):
            # Fast load from standby
            crash_log(f"_load_model: consuming standby weights")
            logger.info(f"Using standby weights for {model_id}")
            premerged = self.standby.consume_standby()
            set_preloaded_weights(premerged)
            load_format = "pinned_arena"
        else:
            load_format = "auto"

        crash_log(f"_load_model: LLM() constructor starting, load_format={load_format}")
        self.llm = LLM(
            model=model_info.hf_path,
            gpu_memory_utilization=model_info.gpu_memory_utilization,
            max_model_len=model_info.context_length,
            trust_remote_code=True,
            load_format=load_format,
            enforce_eager=True,
        )
        crash_log(f"_load_model: LLM() constructor done")

        self.current_model = model_id
        elapsed = time.time() - start
        crash_log(f"_load_model: {model_id} loaded in {elapsed:.1f}s")
        logger.info(f"Model {model_id} loaded in {elapsed:.1f}s")

    async def switch_model(self, target_model: str):
        """Switch to a different model."""
        async with self._lock:
            if self.current_model == target_model:
                logger.debug(f"Already on model {target_model}")
                return

            self.switch_count += 1
            crash_log(f"=== SWITCH #{self.switch_count} START: {self.current_model} -> {target_model} ===")
            logger.info("=" * 60)
            logger.info(f"MODEL SWITCH #{self.switch_count}: {self.current_model} -> {target_model}")
            logger.info("=" * 60)

            start = time.time()

            # Log memory state before cleanup
            crash_log(f"STEP: log_memory_state BEFORE_CLEANUP")
            mem_before = log_memory_state("BEFORE_CLEANUP")

            # CRITICAL: Ensure clean GPU state before model switch
            # This prevents race conditions that can trigger GPU driver bugs
            # causing kernel panics (full machine freeze requiring power cycle)

            # Step 1: Synchronize GPU to ensure all operations complete
            crash_log(f"STEP: torch.cuda.synchronize() starting")
            torch.cuda.synchronize()
            crash_log(f"STEP: torch.cuda.synchronize() done")
            logger.info("GPU synchronized before switch")

            # Step 2: Wait for any ongoing background prefetch to complete
            standby_state = self.standby.get_state()
            crash_log(f"STEP: standby_state = {standby_state.name}")
            logger.info(f"Standby state: {standby_state.name}")
            if standby_state.name == "LOADING":
                crash_log(f"STEP: wait_for_load starting (prefetch running)")
                logger.info("Waiting for background prefetch to complete...")
                prefetch_waited = self.standby.wait_for_load(timeout=120.0)
                crash_log(f"STEP: wait_for_load done, result={prefetch_waited}")
                if prefetch_waited:
                    logger.info("Background prefetch completed")
                else:
                    logger.warning("Background prefetch timeout - proceeding anyway")

            # Step 3: Reset torch.compile/dynamo state to prevent accumulation
            crash_log(f"STEP: dynamo.reset() starting")
            try:
                import torch._dynamo as dynamo
                dynamo.reset()
                logger.debug("torch._dynamo reset")
            except Exception as e:
                logger.debug(f"dynamo reset: {e}")
            crash_log(f"STEP: dynamo.reset() done")

            log_memory_state("AFTER_PRE_SWITCH_SYNC")

            # Check if standby is ready
            model_info = AVAILABLE_MODELS.get(target_model)
            if not model_info:
                raise ValueError(f"Unknown model: {target_model}")

            use_standby = self.standby.is_ready(model_info.hf_path)
            crash_log(f"STEP: use_standby = {use_standby}")

            # Cleanup current model with detailed tracking
            if self.llm:
                crash_log(f"STEP: cleanup {self.current_model} starting")
                logger.info(f"Cleaning up {self.current_model}...")
                cleanup_start = time.time()

                # Step 1: Call full_cleanup
                crash_log(f"STEP: full_cleanup() starting")
                freed_gb = full_cleanup(self.llm)
                crash_log(f"STEP: full_cleanup() done, freed={freed_gb:.1f}GB")
                self.llm = None
                mem_after_cleanup = log_memory_state("AFTER_FULL_CLEANUP")

                # Step 2: Extra gc passes to release CPU memory
                crash_log(f"STEP: gc.collect() x3 starting")
                import gc
                for i in range(3):
                    gc.collect()
                crash_log(f"STEP: gc.collect() x3 done")
                mem_after_gc = log_memory_state("AFTER_GC_PASSES")

                # Step 3: Try to release memory back to OS (Linux-specific)
                crash_log(f"STEP: malloc_trim starting")
                try:
                    import ctypes
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                    log_memory_state("AFTER_MALLOC_TRIM")
                except Exception as e:
                    logger.debug(f"malloc_trim not available: {e}")
                crash_log(f"STEP: malloc_trim done")

                logger.info(f"Cleanup freed {freed_gb:.2f}GB GPU in {time.time() - cleanup_start:.1f}s")

                # Calculate RAM delta
                ram_delta = mem_after_gc['ram_available_gb'] - mem_before['ram_available_gb']
                if ram_delta < -1.0:
                    logger.warning(f"[MEMORY LEAK?] RAM available dropped by {-ram_delta:.1f}GB during cleanup!")

            # Verify memory is available before loading
            crash_log(f"STEP: verify_memory_available starting")
            mem_ok, mem_msg = verify_memory_available(min_gpu_free_gb=5.0, min_ram_avail_gb=2.0)
            crash_log(f"STEP: verify_memory_available done, ok={mem_ok}")
            if not mem_ok:
                logger.error(f"[MEMORY CRITICAL] {mem_msg}")
                # Don't proceed if we're critically low on memory
                raise MemoryError(f"Insufficient memory to load {target_model}: {mem_msg}")
            else:
                logger.info(f"[MEMORY CHECK] OK - sufficient memory available")

            # Load new model
            crash_log(f"STEP: _load_model({target_model}) starting")
            await self._load_model(target_model, use_standby=use_standby)
            crash_log(f"STEP: _load_model({target_model}) done")

            # Log memory state after load
            log_memory_state("AFTER_LOAD")

            elapsed = time.time() - start
            crash_log(f"=== SWITCH #{self.switch_count} COMPLETE in {elapsed:.1f}s ===")
            logger.info(f"Switch complete in {elapsed:.1f}s (standby={use_standby})")

            # Start prefetching previous model for potential switch back
            if self.current_model:
                prev_info = AVAILABLE_MODELS.get(self.current_model)
                if prev_info:
                    crash_log(f"STEP: start_prefetch({prev_info.hf_path}) starting")
                    logger.info(f"Starting background prefetch of previous model...")
                    self.standby.start_prefetch(prev_info.hf_path)
                    crash_log(f"STEP: start_prefetch done")

    def start_prefetch(self, model_id: str):
        """Start prefetching a model in background."""
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
    """Application lifespan handler."""
    await state.initialize()
    yield
    logger.info("Server shutting down...")
    if state.llm:
        full_cleanup(state.llm)


app = FastAPI(
    title="BlitzInfer API",
    description="Fast LLM serving with instant model switching",
    version="0.1.0",
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
    """List available models."""
    logger.debug("GET /v1/models")
    models = [
        ModelObject(
            id=model_id,
            created=int(state.start_time),
        )
        for model_id in AVAILABLE_MODELS.keys()
    ]
    return ModelsResponse(data=models)


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get model details."""
    logger.debug(f"GET /v1/models/{model_id}")

    # Resolve alias
    model_id = MODEL_ALIASES.get(model_id, model_id)

    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    return ModelObject(
        id=model_id,
        created=int(state.start_time),
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion request."""
    request_id = str(uuid.uuid4())[:8]
    state.request_count += 1

    logger.info(f"[{request_id}] POST /v1/chat/completions")
    logger.info(f"[{request_id}] model={request.model}, messages={len(request.messages)}, max_tokens={request.max_tokens}, stream={request.stream}")

    # Log actual message content for debugging
    for i, msg in enumerate(request.messages):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        content_preview = content[:500] + "..." if len(content) > 500 else content
        logger.info(f"[{request_id}] MSG[{i}] {msg.role}: {content_preview}")

    try:
        # Resolve model alias and validate model exists BEFORE any operations
        model_id = MODEL_ALIASES.get(request.model, request.model)

        if model_id not in AVAILABLE_MODELS:
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model}")

        # Switch model if needed (after validation)
        if model_id != state.current_model:
            logger.info(f"[{request_id}] Switching to model: {model_id}")
            await state.switch_model(model_id)

        # Build prompt from messages
        # For GPT-OSS, use Harmony encoding for proper structured output
        use_harmony = HAS_HARMONY and model_id == "gpt-oss-120b"
        prompt_token_ids = None
        has_tools = request.tools and len(request.tools) > 0

        if use_harmony:
            try:
                # Convert messages to dict format for Harmony
                # Include tool_calls and tool_call_id for multi-turn tool conversations
                chat_msgs = []
                for m in request.messages:
                    msg_dict = {"role": m.role, "content": m.content}
                    if m.tool_calls:
                        msg_dict["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                            }
                            for tc in m.tool_calls
                        ]
                    if m.tool_call_id:
                        msg_dict["tool_call_id"] = m.tool_call_id
                    chat_msgs.append(msg_dict)

                harmony_msgs = []

                # Add system message if not present
                if not any(m.get("role") == "system" for m in chat_msgs):
                    sys_msg = get_system_message(with_custom_tools=has_tools)
                    harmony_msgs.append(sys_msg)

                # Add developer message with tools if tools are provided
                if has_tools:
                    # Convert tools to ChatCompletionToolsParam format expected by Harmony
                    from vllm.entrypoints.openai.chat_completion.protocol import (
                        ChatCompletionToolsParam,
                        FunctionDefinition as VLLMFunctionDefinition,
                    )
                    tools_for_harmony = []
                    for tool in request.tools:
                        func_def = VLLMFunctionDefinition(
                            name=tool.function.name,
                            description=tool.function.description,
                            parameters=tool.function.parameters,
                        )
                        tool_param = ChatCompletionToolsParam(
                            type="function",
                            function=func_def,
                        )
                        tools_for_harmony.append(tool_param)
                    dev_msg = get_developer_message(tools=tools_for_harmony)
                    harmony_msgs.append(dev_msg)
                    logger.info(f"[{request_id}] Added {len(request.tools)} tools to Harmony prompt")

                # Parse chat messages into Harmony format
                harmony_msgs.extend(parse_chat_inputs_to_harmony_messages(chat_msgs))
                prompt_token_ids = render_for_completion(harmony_msgs)
                logger.debug(f"[{request_id}] Harmony prompt: {len(prompt_token_ids)} tokens")
            except Exception as e:
                logger.warning(f"[{request_id}] Harmony encoding failed: {e}, using text prompt")
                logger.warning(traceback.format_exc())
                use_harmony = False

        if not use_harmony:
            prompt = _format_chat_messages(request.messages)
            logger.debug(f"[{request_id}] Text prompt length: {len(prompt)} chars")

        # Create sampling params
        # For GPT-OSS with Harmony encoding, add special stop tokens
        stop_token_ids = HARMONY_STOP_TOKENS if use_harmony else None
        sampling_params = SamplingParams(
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            max_tokens=request.max_tokens or 2048,
            stop=request.stop if request.stop else None,
            stop_token_ids=stop_token_ids,
            presence_penalty=request.presence_penalty or 0.0,
            frequency_penalty=request.frequency_penalty or 0.0,
        )
        if stop_token_ids:
            logger.debug(f"[{request_id}] Using Harmony stop tokens: {stop_token_ids}")

        # Generate
        start = time.time()
        if prompt_token_ids:
            # Pass as TokensPrompt dict
            outputs = state.llm.generate([{"prompt_token_ids": prompt_token_ids}], sampling_params)
        else:
            outputs = state.llm.generate([prompt], sampling_params)
        elapsed = time.time() - start

        output = outputs[0]
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason
        output_token_ids = output.outputs[0].token_ids

        # Count tokens
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output_token_ids)
        total_tokens = prompt_tokens + completion_tokens
        state.total_tokens += total_tokens

        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(f"[{request_id}] Generated {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

        # For GPT-OSS models, parse the Harmony format to extract final content and tool calls
        reasoning_content = None
        tool_calls_list = []
        if HAS_HARMONY and model_id == "gpt-oss-120b":
            try:
                if has_tools:
                    # For tool requests, use structured parsing to get tool calls
                    parser = parse_output_into_messages(list(output_token_ids))
                    parsed_messages = parser.messages

                    reasoning_parts = []
                    final_parts = []

                    for msg in parsed_messages:
                        response_items = parse_output_message(msg)
                        for item in response_items:
                            if isinstance(item, ResponseFunctionToolCall):
                                tool_calls_list.append(ToolCall(
                                    id=item.call_id,
                                    type="function",
                                    function=FunctionCall(
                                        name=item.name,
                                        arguments=item.arguments,
                                    )
                                ))
                                logger.info(f"[{request_id}] Parsed tool call: {item.name}({item.arguments[:100]}...)")
                            elif hasattr(item, 'type'):
                                if item.type == "reasoning":
                                    for content in getattr(item, 'content', []):
                                        if hasattr(content, 'text'):
                                            reasoning_parts.append(content.text)
                                elif item.type == "message":
                                    for content in getattr(item, 'content', []):
                                        if hasattr(content, 'text'):
                                            final_parts.append(content.text)

                    # Check for partial content in parser
                    if parser.current_content:
                        if parser.current_channel == "analysis":
                            reasoning_parts.append(parser.current_content)
                        elif parser.current_channel == "final":
                            final_parts.append(parser.current_content)
                        elif parser.current_channel == "commentary" and parser.current_recipient:
                            if parser.current_recipient.startswith("functions."):
                                func_name = parser.current_recipient.split(".")[-1]
                                tool_calls_list.append(ToolCall(
                                    id=f"call_{uuid.uuid4().hex[:24]}",
                                    type="function",
                                    function=FunctionCall(
                                        name=func_name,
                                        arguments=parser.current_content,
                                    )
                                ))

                    if reasoning_parts:
                        reasoning_content = "\n".join(reasoning_parts)
                    if final_parts:
                        generated_text = "\n".join(final_parts)

                    # If tool calls present, clear content per OpenAI spec
                    if tool_calls_list:
                        generated_text = ""

                    logger.info(f"[{request_id}] Harmony (tools) parsed: reasoning={len(reasoning_content or '')} chars, "
                               f"final={len(generated_text)} chars, tool_calls={len(tool_calls_list)}")
                else:
                    # For non-tool requests, use simpler parse_chat_output
                    # Returns (reasoning, final_content, has_content)
                    reasoning, final_content, has_content = parse_chat_output(list(output_token_ids))
                    logger.debug(f"[{request_id}] parse_chat_output returned: reasoning={len(reasoning or '')} chars, "
                                f"final={len(final_content or '')} chars, has_content={has_content}")
                    logger.debug(f"[{request_id}] final_content preview: {(final_content or '')[:200]}")
                    if reasoning:
                        reasoning_content = reasoning
                    if final_content:
                        generated_text = final_content
                    else:
                        # If parse_chat_output couldn't extract content, the model may not have
                        # used proper Harmony tokens. Log this for debugging.
                        logger.warning(f"[{request_id}] parse_chat_output returned no final content - model may not have used Harmony tokens")
                    logger.info(f"[{request_id}] Harmony (simple) parsed: reasoning={len(reasoning_content or '')} chars, "
                               f"final={len(generated_text)} chars (from_parser={final_content is not None})")

            except Exception as e:
                logger.warning(f"[{request_id}] Harmony parse failed: {e}, using raw output")

        # Log response content for debugging
        response_preview = generated_text[:500] + "..." if len(generated_text) > 500 else generated_text
        logger.info(f"[{request_id}] RESPONSE: {response_preview}")

        # Determine finish reason based on content
        if tool_calls_list:
            finish_reason = "tool_calls"
        elif finish_reason is None:
            finish_reason = "stop"

        # Handle streaming response
        if request.stream:
            async def generate_stream():
                created = int(time.time())

                # If we have tool calls, stream them first
                if tool_calls_list:
                    # Send tool calls in chunks
                    for i, tc in enumerate(tool_calls_list):
                        # First chunk with tool call id and name
                        data = {
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
                                            "arguments": ""
                                        }
                                    }]
                                },
                                "finish_reason": None,
                            }]
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                        # Stream arguments in chunks
                        args = tc.function.arguments
                        arg_chunk_size = 50
                        for j in range(0, len(args), arg_chunk_size):
                            arg_chunk = args[j:j+arg_chunk_size]
                            data = {
                                "id": f"chatcmpl-{request_id}",
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{
                                            "index": i,
                                            "function": {"arguments": arg_chunk}
                                        }]
                                    },
                                    "finish_reason": None,
                                }]
                            }
                            yield f"data: {json.dumps(data)}\n\n"

                # Stream content if present
                if generated_text:
                    chunk_size = 20  # characters per chunk
                    for i in range(0, len(generated_text), chunk_size):
                        chunk = generated_text[i:i+chunk_size]
                        data = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }]
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                # Send final chunk with finish_reason
                data = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Build non-streaming response
        response_message = ResponseMessage(
            role="assistant",
            content=generated_text if generated_text else None,
            tool_calls=tool_calls_list if tool_calls_list else None,
        )

        response = ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            created=int(time.time()),
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=response_message,
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions (404, etc.) as-is
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "current_model": state.current_model,
        "request_count": state.request_count,
        "switch_count": state.switch_count,
        "total_tokens": state.total_tokens,
        "uptime_seconds": int(time.time() - state.start_time),
    }


@app.get("/status")
async def status():
    """Detailed status endpoint."""
    gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info()
    gpu_mem_used = (gpu_mem_total - gpu_mem_free) / 1024**3
    gpu_mem_total_gb = gpu_mem_total / 1024**3

    standby_status = None
    if state.standby:
        standby_model = state.standby.get_standby_model()
        standby_status = {
            "arena_size_gb": state.standby._arena.size_bytes / 1024**3 if state.standby._arena else 0,
            "standby_model": standby_model,
            "is_ready": state.standby.is_ready(standby_model) if standby_model else False,
        }

    return {
        "current_model": state.current_model,
        "available_models": list(AVAILABLE_MODELS.keys()),
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
    """Start prefetching a model in background."""
    model_id = MODEL_ALIASES.get(model_id, model_id)

    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    state.start_prefetch(model_id)
    return {"status": "prefetch_started", "model": model_id}


# =============================================================================
# Helpers
# =============================================================================

def _format_chat_messages(messages: List[ChatMessage]) -> str:
    """Format chat messages into a prompt string."""
    # Simple format - can be enhanced per model
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

    logger.info("Starting BlitzInfer API server...")
    logger.info(f"Log file: /tmp/blitzinfer_server.log")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
