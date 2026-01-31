# BlitzInfer Optimization Architecture

This document catalogs all code modifications, optimizations, and architectural decisions in the BlitzInfer project for fast LLM model switching.

**Total Custom Code**: ~6,500 lines across 15+ core modules

---

## Table of Contents

1. [Optimization Categories Summary](#optimization-categories-summary)
2. [Category 1: Memory Management](#category-1-memory-management)
3. [Category 2: Weight Loading Pipeline](#category-2-weight-loading-pipeline)
4. [Category 3: GPU Memory Cleanup](#category-3-gpu-memory-cleanup)
5. [Category 4: vLLM Integration & Patches](#category-4-vllm-integration--patches)
6. [Category 5: Orchestration Layer](#category-5-orchestration-layer)
7. [Category 6: API & Serving](#category-6-api--serving)
8. [Performance Summary](#performance-summary)
9. [Architectural Recommendations](#architectural-recommendations)

---

## Optimization Categories Summary

| Category | Files | Lines | Key Optimization |
|----------|-------|-------|------------------|
| **Memory Management** | arena.py, prefetcher.py, cache_warmer.py | ~1,300 | Pinned memory arena with power-of-2 chunks |
| **Weight Loading** | fast_loader.py, pinned_loader.py, premerge.py, blitz_vllm_patch.py | ~1,760 | 45 GB/s pinned→GPU transfer |
| **GPU Cleanup** | cleanup.py | ~900 | 7-level cleanup hierarchy for vLLM V1 |
| **vLLM Integration** | vllm_adapter.py, pinned_loader.py | ~1,030 | Custom model loader, InprocClient navigation |
| **Orchestration** | standby_manager.py, controller.py, model_state.py | ~1,100 | Queue-driven prefetch, state machine |
| **API & Serving** | server.py | ~1,100 | Harmony encoding, multi-model switching |

---

## Category 1: Memory Management

### 1.1 Pinned Memory Arena (`blitzinfer/memory/arena.py`)

**Problem**: Standard malloc/cudaMalloc too slow for model switching. Need pre-allocated buffer for fast DMA.

**Solution**: Chunked pinned memory arena with LRU eviction.

**Key Implementation**:
```python
class PinnedMemoryArena:
    def __init__(self, size_gb=80.0, chunk_size_gb=16.0, pin_memory=True):
        # CRITICAL: Chunk size MUST be power of 2
        # PyTorch rounds to next power of 2, causing 60% overhead otherwise
        # 5GB → 8GB (60% overhead) vs 16GB → 16GB (0% overhead)
        self._chunks = [
            torch.empty(chunk_size_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(num_chunks)
        ]
```

**Optimizations**:
- **Power-of-2 chunks**: Avoids PyTorch allocation overhead (0% vs 60%)
- **Direct ctypes read**: `read_file_into()` uses ctypes for zero-copy disk reads
- **LRU eviction**: Automatic model eviction when space needed
- **Status tracking**: `allocated/loading/ready/transferring` states

**Memory Layout**:
```
┌─────────────────────────────────────────────────────────┐
│  80GB Arena (5x 16GB power-of-2 chunks)                │
├─────────────────────────────────────────────────────────┤
│ [Chunk 0: 16GB] [Chunk 1: 16GB] ... [Chunk 4: 16GB]   │
│    Model A data spans across chunks as needed          │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Background Prefetcher (`blitzinfer/memory/prefetcher.py`)

**Problem**: Model loading blocks inference.

**Solution**: Background thread loads next model while current one serves.

**Key Features**:
- ThreadPoolExecutor for async loading
- Callback system: `on_ready(model_name, callback)`
- Status machine: `COLD → LOADING → READY → TRANSFERRING`

### 1.3 Page Cache Warmer (`blitzinfer/memory/cache_warmer.py`)

**Problem**: Pinned memory requires large RAM allocation.

**Solution**: Alternative using OS page cache (simpler, less optimal).

**Trade-offs**:
| Approach | Speed | RAM Required | Complexity |
|----------|-------|--------------|------------|
| Pinned Arena | 45 GB/s | 80GB dedicated | High |
| Page Cache | 7-12 GB/s | OS-managed | Low |

---

## Category 2: Weight Loading Pipeline

### 2.1 Fast Safetensor Loader (`blitzinfer/memory/fast_loader.py`)

**Problem**: vLLM's safetensor loading is single-threaded (~2-3 GB/s).

**Solution**: Parallel file loading with 16 workers.

**Key Implementation**:
```python
def load_model_to_arena(model_path, arena, parallel_workers=16):
    # Parse headers (metadata only, no weight data)
    for sf_path in get_safetensor_files(model_path):
        header_size, header = parse_safetensor_header(sf_path)

    # Parallel file reads directly into pinned arena
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(arena.read_file_into, ...) for ...]
```

**Performance**: 10-12 GB/s (NVMe limited)

### 2.2 Custom vLLM Loader (`blitzinfer/memory/pinned_loader.py`)

**Problem**: vLLM doesn't support pre-loaded weights.

**Solution**: Register custom loader `"pinned_arena"` that injects weights.

**Key Implementation**:
```python
@register_model_loader("pinned_arena")
class PinnedArenaModelLoader(BaseModelLoader):
    def load_weights(self, model, model_config):
        weights = get_preloaded_weights()  # From global storage
        for name, param in model.named_parameters():
            if name in weights:
                param.data.copy_(weights[name])  # Direct injection
```

**Transfer Speed**: 39-45 GB/s (PCIe 5.0 limit)

### 2.3 Weight Pre-merging (`blitzinfer/memory/premerge.py`)

**Problem**: vLLM expects merged weights (qkv_proj, gate_up_proj).

**Solution**: Pre-compute merged tensors during background load.

**Model-Specific Transforms**:

| Model | Transformation |
|-------|----------------|
| **Mistral** | `wq/wk/wv → q_proj/k_proj/v_proj` + rotary permutation |
| **GPT-OSS-120B** | 4D MoE tensors → 3D + name remapping |
| **Qwen VL** | `model.language_model.X → language_model.model.X` |

**Memory Optimization**: Default `skip_merge=True` - merge on-the-fly during GPU transfer to avoid copies.

### 2.4 vLLM Monkey Patch (`blitz_vllm_patch.py`)

**Problem**: Alternative approach - patch vLLM's iterator directly.

**Solution**: Replace `safetensors_weights_iterator` with bulk loader.

**Key Implementation**:
```python
def patch_vllm():
    import vllm.model_executor.model_loader.weight_utils as weight_utils
    weight_utils.safetensors_weights_iterator = blitz_safetensors_weights_iterator
```

**Features**:
- Double-buffered pinned memory
- Async GPU transfers
- Achieves ~9 GB/s (vs ~2 GB/s original)

**Status**: Available but not primary path. `pinned_arena` loader preferred.

---

## Category 3: GPU Memory Cleanup

### 3.1 Aggressive Cleanup (`blitzinfer/engine/cleanup.py`)

**Problem**: vLLM V1 single-process mode leaks GPU memory on model unload.

**Root Cause**: `del llm` doesn't release GPU memory - storage stays allocated.

**Critical Discovery**:
```python
# WRONG - GPU memory NOT freed
param.data = torch.empty(0, device='cpu')

# CORRECT - GPU memory actually freed
param.data.storage().resize_(0)
```

### 3.2 Cleanup Hierarchy (7 Levels)

| Level | Function | Target |
|-------|----------|--------|
| 1 | `cleanup_vllm_model()` | Model params, buffers, KV caches |
| 2 | `clear_vllm_caches()` | RoPE, WorkspaceManager, ForwardContext |
| 3 | `clear_current_vllm_config()` | Global vLLM config singleton |
| 4 | `destroy_parallel_state()` | NCCL, Ray, distributed groups |
| 5 | `clear_triton_caches()` | Triton JIT, autotuner |
| 6 | `clear_cudnn_cublas_workspaces()` | CUDA library internals |
| 7 | `force_free_phantom_blocks()` | C++ allocated blocks w/o Python refs |

### 3.3 InprocClient Navigation

**Problem**: vLLM V1 wraps EngineCore in InprocClient, hiding real engine.

**Solution**:
```python
# Must navigate wrapper structure
engine_core = llm.llm_engine.engine_core
if hasattr(engine_core, 'engine_core'):  # InprocClient wrapper
    engine_core = engine_core.engine_core  # Actual EngineCore

model_runner = engine_core.model_executor.driver_worker.worker.model_runner
model = model_runner.model  # PyTorch model
kv_caches = model_runner.kv_caches  # KV cache tensors
```

### 3.4 Phantom Block Cleanup

**Problem**: Flash Attention, NCCL, cuBLAS allocate memory without Python references.

**Solution**: Force-free via CUDA allocator:
```python
snapshot = torch.cuda.memory._snapshot()
for block in snapshot['blocks']:
    if block['size'] > threshold and not has_python_ref(block):
        torch.cuda.caching_allocator_delete(block['address'])
```

### 3.5 Residual Drift

**Unavoidable**: ~0.47GB per switch from CUDA library workspaces.

**Mitigation**:
- Size system for 60-70 switches
- Periodic process restart for long-running servers

---

## Category 4: vLLM Integration & Patches

### 4.1 Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `'0'` | Single-process mode (6s vs 30s switch) |
| `VLLM_SKIP_WARMUP` | `'1'` | Skip kernel warmup (ROCm issues) |
| `VLLM_WORKER_MULTIPROC_METHOD` | `'spawn'` | Clean child process spawning |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `'1'` | Allow 128K+ context overrides |

### 4.2 vLLM Adapter (`blitzinfer/engine/vllm_adapter.py`)

**Key Methods**:
- `load_model()` - Standard vLLM loading
- `load_from_prefetch()` - Fast path with pinned weights
- `unload_model()` - Calls full_cleanup()
- `switch_model()` - Atomic unload + load

### 4.3 Harmony Encoding (GPT-OSS-120B)

**Special handling for GPT-OSS structured output**:
```python
from vllm.entrypoints.openai.parser.harmony_utils import (
    parse_chat_output,           # Returns (reasoning, final_content, has_tool_call)
    parse_chat_inputs_to_harmony_messages,
    render_for_completion,
    get_stop_tokens_for_assistant_actions,
)
```

**Channels**:
- `<|channel|>analysis<|message|>` - Hidden reasoning
- `<|channel|>final<|message|>` - User response
- `<|channel|>commentary to=functions.{name}` - Tool calls

---

## Category 5: Orchestration Layer

### 5.1 Standby Manager (`blitzinfer/orchestrator/standby_manager.py`)

**Pattern**: 1 active (GPU) + 1 standby (pinned CPU)

**Architecture**:
```
GPU VRAM:     [Active Model - 65GB]
Pinned RAM:   [Standby Model - 35GB, ready for fast transfer]
NVMe SSD:     [All other models - cold storage]
```

**State Machine**:
```
EMPTY → LOADING → READY → (consume) → EMPTY
           ↓
         ERROR
```

### 5.2 Queue-Driven Prefetch

**Key Method**: `on_request_queued(model_name)`

```python
def on_request_queued(self, model_name: str):
    """Called when request arrives for a model."""
    if model_name != self.current_model:
        # Start prefetching - deterministic, not speculative
        self.standby.start_prefetch(model_name)
```

### 5.3 Model State Machine (`blitzinfer/orchestrator/model_state.py`)

**States**:
- `COLD` - On disk
- `WARM` - In CPU RAM (future)
- `HOT` - In GPU VRAM
- `SERVING` - Processing requests
- `SWITCHING` - Being swapped

---

## Category 6: API & Serving

### 6.1 OpenAI-Compatible Server (`blitzinfer/api/server.py`)

**Endpoints**:
- `POST /v1/chat/completions` - Chat with streaming
- `GET /v1/models` - List available models

**Features**:
- StandbyManager integration
- Multi-model switching
- Harmony encoding for GPT-OSS
- Vision support for Qwen VL
- Crash logging with fsync

### 6.2 Available Models

| Model ID | Context | Notes |
|----------|---------|-------|
| `gpt-oss-120b` | 131K | Harmony encoding, tool calls |
| `qwen3-vl-32b-thinking` | 128K | Vision support |
| `qwen3-32b` | 128K | Text only |
| `mistral-small-24b` | 128K | Text only |
| `llama-3.1-70b` | 128K | AWQ quantized |
| `kimi-vl` | 128K | Vision support |

---

## Performance Summary

### Bandwidth Achieved

| Operation | Speed | Hardware Limit |
|-----------|-------|----------------|
| SSD → Pinned RAM | 10-12 GB/s | NVMe PCIe 5.0: ~14 GB/s |
| Pinned RAM → GPU | 39-45 GB/s | PCIe 5.0: ~48 GB/s |
| Standard vLLM load | 2-3 GB/s | Safetensor parsing bound |

### Switch Time Breakdown

| Phase | Time | Notes |
|-------|------|-------|
| Prefetch (background) | 3-8s | Happens during inference |
| GPU cleanup | 0.5s | full_cleanup() |
| Weight injection | 1-2s | 35-65GB at 39 GB/s |
| vLLM init | 3-5s | Model construction, KV cache |
| **Total warm switch** | **5-10s** | vs 20-30s cold |

### Memory Efficiency

| Metric | Value |
|--------|-------|
| Arena overhead | 0% (power-of-2 chunks) |
| Drift per switch | 0.48 GB |
| Max switches before restart | 60-70 |

---

## Architectural Recommendations

### Potential Further Optimizations

1. **Persistent Engine Process**
   - `persistent_engine.py` exists but not integrated
   - Could amortize vLLM init overhead across switches
   - Estimated saving: 3-5s per switch

2. **KV Cache Profiling Skip**
   - vLLM re-profiles KV cache on each load
   - If models share architecture, could cache profile
   - Estimated saving: 1-2s per switch

3. **Weight Deduplication**
   - Models with shared layers (same base) could share weights
   - Requires vLLM model structure awareness

4. **Tensor Parallelism Reuse**
   - Currently destroys distributed state on switch
   - Could maintain parallel state for same-TP-size models

5. **Lazy Encoder Init (Vision Models)**
   - Qwen VL takes +6s for encoder cache init
   - Could pre-initialize encoder during prefetch

### Current Bottleneck Analysis

```
Total switch time: ~10s

Breakdown:
├── Weight transfer: ~2s (optimized, near hardware limit)
├── vLLM model init: ~3s (PyTorch model construction)
├── KV cache profiling: ~2s (GPU memory scan)
├── Warmup: ~2s (first inference pass)
└── Cleanup overhead: ~1s (previous model teardown)

Optimization potential:
- Persistent engine could eliminate init/warmup: -5s
- Cached KV profiles could eliminate profiling: -2s
- Best case: ~3s switches (weight transfer + cleanup)
```

### Code Quality Notes

1. **Well-structured**: Clear separation of concerns
2. **Thread-safe**: Proper locking in critical sections
3. **Documented**: Comprehensive docstrings and comments
4. **Tested**: Multiple test files cover edge cases

### Known Issues to Address

1. **Tool calls from external clients** - Harmony parsing needs adjustment
2. **Mistral premerge** - 216 tensors skipped (name mapping incomplete)
3. **Vision encoder overhead** - Inherent to model architecture

---

## File Reference

| File | Lines | Category |
|------|-------|----------|
| `blitzinfer/memory/arena.py` | 592 | Memory Management |
| `blitzinfer/memory/fast_loader.py` | 363 | Weight Loading |
| `blitzinfer/memory/pinned_loader.py` | 472 | Weight Loading |
| `blitzinfer/memory/premerge.py` | 560 | Weight Loading |
| `blitzinfer/memory/prefetcher.py` | 367 | Memory Management |
| `blitzinfer/memory/cache_warmer.py` | 346 | Memory Management |
| `blitzinfer/engine/cleanup.py` | 897 | GPU Cleanup |
| `blitzinfer/engine/vllm_adapter.py` | 557 | vLLM Integration |
| `blitzinfer/orchestrator/standby_manager.py` | 500+ | Orchestration |
| `blitzinfer/orchestrator/controller.py` | 450+ | Orchestration |
| `blitzinfer/orchestrator/model_state.py` | 150 | Orchestration |
| `blitzinfer/api/server.py` | 1100+ | API & Serving |
| `blitz_vllm_patch.py` | 365 | vLLM Patch (alt) |

**Total**: ~6,500+ lines of optimization code
