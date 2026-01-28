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
from typing import Any, Dict, List, Optional, Union

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

# BlitzInfer imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from blitzinfer.engine.cleanup import full_cleanup
from blitzinfer.memory import set_preloaded_weights
from blitzinfer.orchestrator.standby_manager import StandbyManager

# Configure extensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-30s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/blitzinfer_server.log', mode='a'),
    ]
)
logger = logging.getLogger('blitzinfer.api')

# Reduce noise from some libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


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
# Use 95% GPU memory and maximum context lengths (with RoPE scaling where needed)
GPU_MEM_UTIL = 0.95  # Use ~90GB of 95GB VRAM

AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    # GPT-OSS-120B - largest model, MXFP4 quantized (native 128K)
    "gpt-oss-120b": ModelInfo(
        name="gpt-oss-120b",
        hf_path="openai/gpt-oss-120b",
        context_length=131072,  # 128K native
        gpu_memory_utilization=GPU_MEM_UTIL,
    ),
    # Qwen3 VL 32B with thinking - vision model (native 256K, expandable to 1M)
    "qwen3-vl-32b-thinking": ModelInfo(
        name="qwen3-vl-32b-thinking",
        hf_path="Qwen/Qwen3-VL-32B-Thinking-FP8",
        context_length=262144,  # 256K native
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
    # GLM 4.6V Flash - fast vision (native 128K)
    "glm-4.6v-flash": ModelInfo(
        name="glm-4.6v-flash",
        hf_path="zai-org/GLM-4.6V-Flash",
        context_length=131072,  # 128K
        gpu_memory_utilization=GPU_MEM_UTIL,
        supports_vision=True,
    ),
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

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None


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


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
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
        logger.info(f"Loading model {model_id} (standby={use_standby})...")

        if use_standby and self.standby.is_ready(model_info.hf_path):
            # Fast load from standby
            logger.info(f"Using standby weights for {model_id}")
            premerged = self.standby.consume_standby()
            set_preloaded_weights(premerged)
            load_format = "pinned_arena"
        else:
            load_format = "auto"

        self.llm = LLM(
            model=model_info.hf_path,
            gpu_memory_utilization=model_info.gpu_memory_utilization,
            max_model_len=model_info.context_length,
            trust_remote_code=True,
            load_format=load_format,
            enforce_eager=True,
        )

        self.current_model = model_id
        elapsed = time.time() - start
        logger.info(f"Model {model_id} loaded in {elapsed:.1f}s")

    async def switch_model(self, target_model: str):
        """Switch to a different model."""
        async with self._lock:
            if self.current_model == target_model:
                logger.debug(f"Already on model {target_model}")
                return

            self.switch_count += 1
            logger.info("=" * 60)
            logger.info(f"MODEL SWITCH #{self.switch_count}: {self.current_model} -> {target_model}")
            logger.info("=" * 60)

            start = time.time()

            # Check if standby is ready
            model_info = AVAILABLE_MODELS.get(target_model)
            if not model_info:
                raise ValueError(f"Unknown model: {target_model}")

            use_standby = self.standby.is_ready(model_info.hf_path)

            # Cleanup current model
            if self.llm:
                logger.info(f"Cleaning up {self.current_model}...")
                cleanup_start = time.time()
                freed_gb = full_cleanup(self.llm)
                self.llm = None
                logger.info(f"Cleanup freed {freed_gb:.2f}GB in {time.time() - cleanup_start:.1f}s")

            # Load new model
            await self._load_model(target_model, use_standby=use_standby)

            elapsed = time.time() - start
            logger.info(f"Switch complete in {elapsed:.1f}s (standby={use_standby})")

            # Start prefetching previous model for potential switch back
            if self.current_model:
                prev_info = AVAILABLE_MODELS.get(self.current_model)
                if prev_info:
                    logger.info(f"Starting background prefetch of previous model...")
                    self.standby.start_prefetch(prev_info.hf_path)

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
    logger.info(f"[{request_id}] model={request.model}, messages={len(request.messages)}, max_tokens={request.max_tokens}")

    try:
        # Resolve model alias
        model_id = MODEL_ALIASES.get(request.model, request.model)

        if model_id not in AVAILABLE_MODELS:
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model}")

        # Switch model if needed
        if model_id != state.current_model:
            logger.info(f"[{request_id}] Switching to model: {model_id}")
            await state.switch_model(model_id)

        # Build prompt from messages
        prompt = _format_chat_messages(request.messages)
        logger.debug(f"[{request_id}] Prompt length: {len(prompt)} chars")

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            max_tokens=request.max_tokens or 2048,
            stop=request.stop if request.stop else None,
            presence_penalty=request.presence_penalty or 0.0,
            frequency_penalty=request.frequency_penalty or 0.0,
        )

        # Generate
        start = time.time()
        outputs = state.llm.generate([prompt], sampling_params)
        elapsed = time.time() - start

        output = outputs[0]
        generated_text = output.outputs[0].text
        finish_reason = output.outputs[0].finish_reason

        # Count tokens
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)
        total_tokens = prompt_tokens + completion_tokens
        state.total_tokens += total_tokens

        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info(f"[{request_id}] Generated {completion_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

        # Build response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            created=int(time.time()),
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason=finish_reason or "stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

        return response

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
