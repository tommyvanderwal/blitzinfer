# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BlitzInfer** is a high-performance LLM serving orchestrator optimized for fast model switching with intelligent queue management and tiered memory caching.

**Current Status**: Phase 2-3 - Fast model switching achieved. Queue-driven prefetch working with ~5-10s warm switches.

## Latest Test Results (Jan 2026)

**Queue-Driven Test (`test_queue_simple.py`):**

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Preload time (max) | 8.41s | < 10s | **PASS** |
| Memory drift | 0.48GB | < 2GB | **PASS** |
| Weight injection | 39.3 GB/s | - | Excellent |
| GPT-OSS switch (warm) | 5.36s | < 10s | **PASS** |
| Qwen VL switch | 16.60s | < 10s | FAIL (vision overhead) |
| GPT-OSS tool calls | ✓ | - | **PASS** |
| Qwen VL vision | ✓ | - | **PASS** |

**Key findings:**
- Weight injection is fast (39.3 GB/s) - bottleneck is vLLM model init
- Vision models have inherent encoder setup overhead (~6s extra)
- Warm switches achieve <10s target for text models
- Memory cleanup is stable (0.48GB drift over 3 switches)

**Known issues:**
- Tool calls from OpenCode (Claude Code) don't work with the API server
- Harmony tool call parsing may need adjustment for external clients

## Completed

- [x] vLLM compiled and working on AMD Radeon 780M (gfx1103) with ROCm 7.2
- [x] Basic model switching working between different models (Qwen-7B, Mistral-7B)
- [x] Model orchestrator with state machine (COLD/HOT/SERVING)
- [x] vLLM engine adapter with proper cleanup for process termination
- [x] **Fast weight swap** (~2.7s) for same-architecture models
- [x] **Cross-architecture switching** (~6s) between Qwen and Mistral
- [x] Blitz vLLM patch for fast initial loading at 9.7 GB/s
- [x] Unified model orchestrator with automatic strategy selection
- [x] **Page cache warming** for background model prefetch (~12 GB/s)
- [x] **Orchestrator integration** with PageCacheWarmer (1.44x speedup, 10/10 switches)
- [x] **Pinned arena loader** for fast CPU→GPU transfer (44.4 GB/s, 1.37x speedup)
- [x] **Standby manager** (1 active + 1 standby pattern) - **5.36s warm switches**
- [x] **Static 80GB pinned arena** (5x 16GB power-of-2 chunks, 0% overhead)
- [x] **Aggressive GPU cleanup** for vLLM V1 single-process mode
- [x] **Cross-architecture switching** (Qwen-32B ↔ gpt-oss-120b) - InprocClient navigation fix
- [x] **GPT-OSS Harmony tool calls** - Working with proper encoding
- [x] **Qwen VL vision input** - Working with PIL Image format

## WARNING: AWQ Marlin Intermittent Crash Bug (Jan 2026)

**BUG**: The `awq_marlin` quantization kernel causes **intermittent hard system freeze** on RTX PRO 6000 Blackwell.

| Quantization | Result |
|-------------|--------|
| `awq_marlin` | ⚠️ **INTERMITTENT CRASH** (may require power cycle) |
| `awq` standard | ✅ Works |
| `mxfp4` Marlin | ✅ Works |

**Findings**:
- Fresh cold boots: 6/6 tests passed (both awq_marlin and awq)
- After model switching/extended tests: Crashes occurred
- Suspected trigger: Accumulated GPU/driver state from model switching

**Workaround**: Use `quantization='awq'` for production reliability:
```python
llm = LLM(
    model='hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4',
    quantization='awq',  # NOT 'awq_marlin' - safer for Blackwell
    ...
)
```

**Trade-off**: Standard AWQ is ~10x slower than Marlin but works reliably.

**Bug Report**: `tests/AWQ_MARLIN_CRASH_BUG_REPORT.md`

## Architecture

```
REQUEST GATEWAY (OpenAI API - planned)
    ↓
PER-MODEL REQUEST QUEUES (implemented)
    ↓
MODEL ORCHESTRATOR (BlitzInferOrchestrator)
    - State machine: COLD → HOT → SERVING
    - Model registry with queue tracking
    ↓
vLLM ENGINE ADAPTER (VLLMEngine)
    - Model loading/unloading
    - Proper multiprocess cleanup
```

### Key Design Principles

- **One model active at a time** - maximizes KV cache utilization
- **Proper cleanup on switch** - wait for child processes, clear GPU memory
- **Minimal vLLM modifications** - use vLLM as library, orchestrate externally

## Target Hardware

### PRIMARY: RTX PRO 6000 System (REMOTE)
**SSH**: `ssh tommy@192.168.2.90`
**Path**: `~/pythonprojects/blitzinfer/`

- **GPU**: NVIDIA RTX PRO 6000 (95GB VRAM)
- **Context**: 100K+ tokens (NEVER reduce below 100K)
- **Target Models**: GPT-OSS-120B (~60GB), Qwen3-VL-32B (~62GB)
- **vLLM Config**: Single-process mode (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)

**IMPORTANT**: Most development and testing should target this system. Always sync code changes.

### Secondary: AMD iGPU (LOCAL)
**Testing**: Ryzen 7840HS with Radeon 780M iGPU, ~109GB DDR5 unified memory, ROCm 7.2

## Development Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Framework research | Complete - vLLM chosen |
| 1 | Prototype weight offloading | Complete - basic switching works |
| 2 | Core orchestrator | **Complete** - fast model switching |
| 3 | Memory manager - tiered cache | Not started |
| 4 | Startup optimization | **Partial** - Blitz patch implemented |
| 5 | Integration & API | Not started |

## Package Structure

```
blitzinfer/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py        # ModelConfig, BlitzInferConfig
├── orchestrator/
│   ├── __init__.py
│   ├── model_state.py     # ModelState, ModelStatus, ModelRegistry
│   └── controller.py      # BlitzInferOrchestrator
├── engine/
│   ├── __init__.py
│   └── vllm_adapter.py    # VLLMEngine adapter
├── memory/                # Pinned memory arena and fast loading
│   ├── __init__.py
│   ├── arena.py           # PinnedMemoryArena for fast GPU transfers
│   ├── fast_loader.py     # Safetensor parsing and arena loading
│   ├── pinned_loader.py   # Custom vLLM loader for pinned weights
│   ├── premerge.py        # Pre-merge weights for vLLM injection
│   ├── prefetcher.py      # Background model prefetching
│   └── cache_warmer.py    # Page cache warming
└── api/                   # Future: OpenAI-compatible endpoint
    └── __init__.py
```

## Key Files

### Core Components
- `blitz_model_orchestrator.py` - Unified orchestrator with automatic strategy selection
- `blitz_weight_swap_final.py` - Production-ready fast weight swapper (~2.7s)
- `blitz_vllm_patch.py` - vLLM patch for fast initial loading (9.7 GB/s)

### Test Files
- `test_working.py` - Basic vLLM test (single model)
- `test_direct_switch.py` - Cross-architecture switch (no Blitz patch)
- `test_cross_model_robust.py` - Robust switching test with error handling

### Documentation
- `WEIGHT_SWAP_RESULTS.md` - Detailed performance results and benchmarks

## Key Configuration for iGPU

```python
llm = LLM(
    model='...',
    gpu_memory_utilization=0.20,     # Conservative for iGPU
    max_model_len=4096,
    max_num_seqs=20,
    max_num_batched_tokens=512,      # Small batches for stability
    enforce_eager=True,              # Required on gfx1100
)
```

## Performance (iGPU - AMD Radeon 780M)

| Metric | Value | Notes |
|--------|-------|-------|
| Initial model load (7B) | ~10s | Standard vLLM |
| **Same-arch weight swap** | **~2.7s** | Blitz weight swapper |
| **Cross-arch switch** | **~6s** | Full reload (standard vLLM) |
| Inference speed | ~4 tok/s | |
| Weight loading bandwidth | 9.7 GB/s | Near hardware limit |

### Switching Strategies

| Strategy | Time | Use Case |
|----------|------|----------|
| Weight swap | ~2.7s | Same architecture (e.g., Qwen→Qwen) |
| Full reload | ~6s | Different architecture (e.g., Qwen→Mistral) |

## Target Models (Future)

1. `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
2. `openai/gpt-oss-120b`
3. `Qwen/Qwen3-VL-32B-Thinking-FP8`

## Important Notes

- See `LESSONS.md` for detailed ROCm 7.2 setup instructions
- See `SPEC.md` for the complete technical specification
- vLLM V1 engine uses multiprocessing - must clean up child processes on unload
- NumPy must be < 2.3 for numba compatibility

## Memory Leak Fix (SOLVED - Jan 2026)

**Problem**: vLLM V1's single-process mode had a memory leak - model weights and KV cache were not freed when `del llm` is called.

### Issue 1: InprocClient Navigation
The cleanup code was not properly navigating vLLM V1's `InprocClient` wrapper structure:
```
llm.llm_engine.engine_core (InprocClient)
    └── .engine_core (EngineCore)  ← MUST unwrap this!
        └── .model_executor → .driver_worker → .worker → .model_runner → .model
```

### Issue 2: storage().resize_(0) vs data = empty (CRITICAL)
**CRITICAL DISCOVERY**: `tensor.data = torch.empty(0, device='cpu')` does NOT release GPU memory!
The underlying CUDA storage is kept alive. You MUST use `tensor.storage().resize_(0)` to actually free memory.

```python
# WRONG - leaves GPU memory allocated:
for param in model.parameters():
    param.data = torch.empty(0, device='cpu')  # Storage NOT freed!

# CORRECT - actually releases GPU memory:
for param in model.parameters():
    param.data.storage().resize_(0)  # Storage freed!
```

This applies to:
- Model parameters (`model.named_parameters()`)
- Model buffers (`model.named_buffers()`)
- KV cache tensors (`model_runner.kv_caches`)
- Static forward context (`model_runner.static_forward_context`)

### Cross-Architecture Stress Test (RTX PRO 6000, 5 rounds, 128K context)

| Model | Memory Used | Memory Freed | Context |
|-------|-------------|--------------|---------|
| Qwen-32B-FP8 | 92GB | 89GB | 128K tokens |
| gpt-oss-120b | 89GB | 85GB | 128K tokens |

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Memory drift/round | ~10GB | ~0.6GB |
| 5 rounds total drift | ~50GB | 3.0GB |
| Stress test | FAIL | **5/5 PASS** |

**Result**: Cross-architecture switching (Qwen-32B ↔ gpt-oss-120b) now works with 128K context!

### Residual Memory Drift (~0.47GB per switch) - INVESTIGATED

**Problem**: After fixing the major leak, there's still ~0.47GB drift per model switch.

**Root Cause Analysis** (Jan 2026):
- **Driver memory** (+0.186GB): One-time NCCL init cost, doesn't grow
- **PyTorch allocated** (+0.47GB/round): Internal CUDA library allocations with NO Python stack traces

**Findings**:
1. Memory has no Python traces - allocated by CUDA libraries (NCCL, Flash Attention, cuBLAS) through PyTorch's allocator
2. These are internal workspaces that aren't being reused across model loads
3. Cannot be freed from Python - no Python references to clear

**Attempted fixes that didn't help**:
- Clearing triton/dynamo/inductor caches
- Clearing cuDNN/cuBLAS workspaces
- Clearing vLLM WorkspaceManager, ForwardContext, UBatchContext
- Aggressive allocator GC settings
- CUBLAS_WORKSPACE_CONFIG=:0:0

**Impact**:
- With 95GB VRAM: ~60-70 switches before running low
- Projected drift over 100 switches: ~47GB

**Recommended approach**:
1. Accept drift and size system accordingly
2. Restart Python process periodically for long-running servers
3. This appears to be a vLLM V1 bug - internal resources not reused

| Mode | Memory Cleanup | Switch Time | Notes |
|------|---------------|-------------|-------|
| **Single-process (`=0`)** | ✓ Fixed | ~10-13s | Use this for fast switching |
| Multiprocessing (`=1`) | ✓ Works | ~30s | Slower but more isolated |

## BIOS Crash with llama-AWQ (CRITICAL - Jan 2026)

**Problem**: Loading `llama-3.1-70B-AWQ-INT4` after multiple model switches with StandbyManager causes system-level crash (BIOS freeze, requires power cycle).

### Crash Pattern
- Crash is **non-deterministic** (sometimes passes, sometimes crashes)
- Occurs when loading AWQ Marlin model AFTER prior model switches
- Crash happens during safetensors loading or marlin initialization
- 100% CPU on 1 core for 2-7 seconds before crash

### Root Cause (Investigation Jan 2026)
Cumulative GPU state corruption from multiple load/cleanup cycles:
1. Multiple model switches corrupt internal CUDA/driver state
2. AWQ Marlin kernel (used by llama-AWQ) encounters corrupted state
3. Marlin initialization fails catastrophically → BIOS crash

### Tested Scenarios
| Scenario | Result |
|----------|--------|
| full_cleanup() without StandbyManager | **PASS** |
| StandbyManager (lazy) + 1 switch + llama | **CRASH** (non-deterministic) |
| StandbyManager deleted before cleanup + llama | **PASS** |
| llama-AWQ as FIRST model (no prior switches) | **PASS** |
| Multiple switches (3+) then llama-AWQ | **CRASH** (consistent) |

### Workarounds
For reliable llama-3.1-70B-AWQ-INT4 loading:

1. **Delete StandbyManager before cleanup** (safest):
```python
standby.shutdown()
del standby
gc.collect()
freed = full_cleanup(llm)
# Now safe to load llama-AWQ
```

2. **Restart Python process** before loading llama-AWQ after multiple switches

3. **Load llama-AWQ first** (no prior model switches)

### Not Affected
- gpt-oss-120b (mxfp4 quantization) - works with any switch count
- Qwen3-32B-FP8 (fp8 quantization) - works with any switch count
- Mistral-Small-24B (bfloat16) - works with any switch count

**Key Files**: `~/crash_isolation_results.md` (test results on RTX PRO 6000)

## vLLM Model Cleanup (CRITICAL)

**Use `full_cleanup()` from cleanup.py - it properly navigates InprocClient:**

```python
from blitzinfer.engine.cleanup import full_cleanup

# CORRECT - releases all GPU memory (38GB+ freed)
freed_gb = full_cleanup(llm)
llm = None

# WRONG - leaves 42GB orphaned (only KV cache freed)
del llm

# WRONG - leaves ~90GB orphaned CUDA memory
del llm  # Memory not released!
```

Without explicit shutdown, the main Python process inherits GPU memory from the terminated subprocess, blocking subsequent model loads.

## Page Cache Warming (Fast Model Prefetch)

Pre-read model safetensor files into OS page cache in background while current model is serving. When switching, vLLM reads weights from RAM (page cache) instead of SSD.

**Orchestrator Integration Test (RTX PRO 6000, 10 switches)**:
| Metric | Value |
|--------|-------|
| Total switches | 10/10 successful |
| Cold load average | 20.04s |
| **Warm load average** | **13.88s** |
| **Speedup** | **1.44x** |
| Time saved per switch | 6.2s |
| Memory drift | 1.6GB over 10 switches (stable) |

**Per-switch details**:
```
Switch  1 (gpt-oss-120b): 19.5s [COLD]
Switch  2 (Qwen3-VL-32B): 20.5s [COLD]
Switch  3 (gpt-oss-120b): 14.9s [WARM]
Switch  4 (Qwen3-VL-32B): 12.9s [WARM]
Switch  5 (gpt-oss-120b): 14.6s [WARM]
Switch  6 (Qwen3-VL-32B): 13.1s [WARM]
...
```

**Usage via Orchestrator**:
```python
from blitzinfer.config import BlitzInferConfig, ModelConfig, PrefetchConfig
from blitzinfer.orchestrator import BlitzInferOrchestrator

config = BlitzInferConfig(
    models=[ModelConfig(name="openai/gpt-oss-120b", ...)],
    prefetch=PrefetchConfig(
        enabled=True,
        use_page_cache=True,  # Default: use page cache warming
    ),
)

orch = BlitzInferOrchestrator(config)

# Register model paths for warming
orch.register_model_path("openai/gpt-oss-120b", "/path/to/model")

# Start warming a model explicitly
orch.start_warming("other_model")

# Check warming status
if orch.is_model_warm("other_model"):
    # Ready for fast switch
    pass

# Generate triggers automatic prefetch
result = await orch.generate(model="gpt-oss-120b", prompt="Hello", max_tokens=50)
```

**Direct PageCacheWarmer usage**:
```python
from blitzinfer.memory import PageCacheWarmer, WarmStatus

warmer = PageCacheWarmer()
warmer.register_model('qwen-32b', '/path/to/model')

# Start background warming while serving current model
warmer.start_warming('qwen-32b')

# Check if ready
if warmer.is_warm('qwen-32b'):
    # Safe to switch - weights will load fast
    pass
```

**Key files**: `blitzinfer/memory/cache_warmer.py`, `blitzinfer/orchestrator/controller.py`, `test_orchestrator_integration.py`

## Pinned Arena Loader (Fast CPU→GPU Transfer)

Pre-load weights into pinned CPU memory during inference, then transfer at PCIe speed when switching models. This bypasses safetensors parsing overhead.

**CRITICAL: Use Power-of-2 Chunk Sizes for Pinned Memory**

PyTorch rounds pinned memory allocations to the next power of 2 (see [#150517](https://github.com/pytorch/pytorch/issues/150517)).
Non-power-of-2 allocations have massive overhead:

| Allocation Size | Actual Shmem | Overhead |
|-----------------|--------------|----------|
| 5GB | 8GB | 60% |
| 10GB | 16GB | 60% |
| **1GB** | 1GB | **0%** |
| **2GB** | 2GB | **0%** |
| **4GB** | 4GB | **0%** |
| **16GB** | 16GB | **0%** |

**Recommended: 16GB chunks** (power of 2, fewer allocations, 0% overhead)

**Standby Manager Test Results (RTX PRO 6000, Qwen3-VL-32B-FP8)**:
| Metric | Value |
|--------|-------|
| Arena | **80GB pinned (5x 16GB chunks)** - static pre-allocation |
| Memory overhead | **0%** |
| Weight injection | **25.4 GB/s** |
| Cold load | 25.8s |
| **Warm switch** | **6.8s** |
| **Speedup** | **3.8x** |

**Architecture**:
```
At startup:
1. Pre-allocate static 80GB pinned arena (5x 16GB chunks) - ~33s

Background (hidden during inference):
2. Load safetensors into arena (~3s at 10 GB/s)
3. Pre-merge qkv_proj, gate_up_proj (~1.8s)

User-facing switch:
4. vLLM init + weight injection (6.8s total)
   - Weight transfer: ~1.4s at 25.4 GB/s
   - vLLM overhead: ~5.4s (model init, KV cache profiling)
```

**Usage**:
```python
from blitzinfer.orchestrator.standby_manager import StandbyManager
from blitzinfer.memory import set_preloaded_weights
from blitzinfer.engine.cleanup import full_cleanup
from vllm import LLM

# Initialize with static 80GB arena (pre-allocated at startup)
standby = StandbyManager(
    arena_size_gb=80.0,   # Fits gpt-oss-120b (~65GB) with margin
    chunk_size_gb=16.0,   # MUST be power of 2 (1, 2, 4, 8, 16, 32GB)
    pin_memory=True,
    lazy_arena=False,     # Pre-allocate now (default)
)

# Prefetch in background while serving current model
standby.start_prefetch("Qwen/Qwen3-VL-32B-Thinking-FP8")

# When ready to switch
if standby.is_ready("Qwen/Qwen3-VL-32B-Thinking-FP8"):
    premerged = standby.consume_standby()
    full_cleanup(current_llm)  # Release GPU memory
    set_preloaded_weights(premerged)
    llm = LLM(model=model_name, load_format="pinned_arena", ...)
```

**LIMITATION: Cross-Architecture Switching**

vLLM V1 single-process mode has a memory leak that prevents switching between different architectures (e.g., Qwen → gpt-oss-120b). The old model's GPU memory (~42GB) stays allocated in PyTorch's caching allocator, and when the new model tries to construct, both models' memory requirements exceed available VRAM.

| Scenario | Status | Notes |
|----------|--------|-------|
| Same architecture (Qwen → Qwen) | ✅ Works | 3.8x speedup |
| Cross architecture (Qwen → gpt-oss-120b) | ❌ OOM | Memory not released |

**Workarounds**:
- Use multiprocessing mode (`VLLM_ENABLE_V1_MULTIPROCESSING=1`) - but slower switches
- Only switch between same-architecture models
- Restart Python process for architecture changes

**Direct Arena Usage** (for custom workflows):
```python
from blitzinfer.memory import (
    PinnedMemoryArena,
    load_model_to_arena,
    get_premerged_tensors_for_vllm,
    set_preloaded_weights,
)
from vllm import LLM

# Background: pre-load into pinned arena (use 1GB chunks!)
arena = PinnedMemoryArena(40, chunk_size_gb=1.0, pin_memory=True)
load_model_to_arena(model_path, arena, model_name)

# Pre-merge weights (can be done during inference)
pinned_tensors = arena.get_all_tensors(model_name)
premerged = get_premerged_tensors_for_vllm(pinned_tensors)

# On switch: inject weights fast
set_preloaded_weights(premerged)
llm = LLM(model=model_name, load_format="pinned_arena", ...)
```

**Key files**: `blitzinfer/memory/arena.py`, `blitzinfer/memory/pinned_loader.py`, `blitzinfer/memory/premerge.py`

## Bottleneck Analysis

**GPU Transfer Speed**: Raw pinned → GPU achieves **44-48 GB/s** on RTX PRO 6000 (PCIe 5.0).

| Test | Bandwidth | Notes |
|------|-----------|-------|
| Raw pinned → GPU | 44.4-48 GB/s | Hardware limit |
| **Pinned arena loader** | **44.4 GB/s** | ✓ Achieved |
| Safetensors direct | 2.7 GB/s | Old vLLM approach |
| Page cache warm | ~7 GB/s | Simpler but slower |

**Current Status**: Weight transfer bottleneck is SOLVED at 44.4 GB/s.

**Remaining bottleneck**: vLLM initialization overhead (~12.6s)
- Model construction: ~5s
- KV cache profiling: ~4s
- Engine core init: ~4s

**Actual results** (Qwen3-VL-32B-FP8, 35.5GB):
- Weight transfer: **0.80s** (at 44.4 GB/s)
- vLLM overhead: **12.6s** (dominated by KV cache profiling)
- Total switch: **13.39s** (1.37x faster than baseline)
- **Total: ~6.4s** (vs current ~15s cold, ~11s warm)

**Key profiling files**: `profile_weight_loading.py`, `profile_pinned_arena_load.py`

## Standby Manager (1 Active + 1 Standby Pattern)

Pre-load the next model into pinned CPU RAM while serving the current model on GPU. When switching, transfer from pinned RAM to GPU at ~45 GB/s instead of loading from SSD.

**Tested Configuration (RTX PRO 6000, 124GB RAM)**:
| Parameter | Value |
|-----------|-------|
| Arena size | **40GB** (fits Qwen3-32B-FP8) |
| GPU utilization | 0.85 (80.75GB VRAM) |
| Models | Qwen3-32B-FP8 (~33GB), gpt-oss-120b (~65GB) |

**Performance (6 switches, 6/6 verification passed)**:
| Metric | Value |
|--------|-------|
| Cold switches (gpt-oss) | avg 30.5s |
| **Warm switches (Qwen)** | **avg 6.2s** |
| **Speedup** | **4.9x** |
| Weight injection | 45.1 GB/s |

**Memory Constraints**:
```
System RAM:     124GB
GPU VRAM:       95GB (CUDA uses ~77GB for 65GB model)
Arena limit:    ~40GB (CUDA uses shared memory for GPU buffers)

70GB arena + 77GB GPU = OOM (kernel kills process)
40GB arena + 77GB GPU = OK (stays under 124GB)
```

**Key insight**: CUDA uses shared/pinned memory internally for GPU buffers. A 65GB model uses ~77GB of shared memory. Combined with a 70GB pinned arena, this exceeds 124GB RAM.

**Usage**:
```python
from blitzinfer.orchestrator.standby_manager import StandbyManager
from blitzinfer.memory import set_preloaded_weights
from vllm import LLM, SamplingParams

# Initialize with 40GB arena (fits Qwen, not gpt-oss)
standby = StandbyManager(arena_size_gb=40.0)

# Cold load initial model
llm = LLM(model="openai/gpt-oss-120b", ...)

# Request arrives for Qwen → triggers prefetch
standby.start_prefetch("Qwen/Qwen3-32B-FP8")

# ... continue serving gpt-oss ...

# When ready to switch
if standby.is_ready("Qwen/Qwen3-32B-FP8"):
    premerged = standby.consume_standby()

    # Unload current model (proper cleanup!)
    unload_llm(llm)

    # Fast load from standby
    set_preloaded_weights(premerged)
    llm = LLM(model="Qwen/Qwen3-32B-FP8", load_format="pinned_arena", ...)
```

**Critical vLLM cleanup** (required for reliable switching):
```python
def unload_llm(llm):
    # Clear model weights
    model_runner = llm.llm_engine.engine_core.engine_core.model_executor \
                      .driver_worker.worker.model_runner
    for param in model_runner.model.parameters():
        param.data = torch.empty(0, device='cpu')
    model_runner.kv_caches.clear()
    model_runner.model = None

    # Shutdown engine
    llm.llm_engine.engine_core.shutdown()
    del llm

    # Reset vLLM state
    from vllm.distributed import parallel_state
    parallel_state.cleanup_dist_env_and_memory(shutdown_ray=False)

    # CRITICAL: Clear rotary embedding cache
    from vllm.model_executor.layers import rotary_embedding
    rotary_embedding._ROPE_DICT.clear()

    # Reset torch._dynamo
    import torch._dynamo as dynamo
    dynamo.reset()

    gc.collect()
    torch.cuda.empty_cache()
```

**Key files**: `blitzinfer/orchestrator/standby_manager.py`, `test_standby_safe.py`

## OpenAI-Compatible API Server (Jan 2026)

**File**: `blitzinfer/api/server.py`

FastAPI server providing OpenAI-compatible `/v1/chat/completions` endpoint with:
- Multi-model support with fast switching
- StandbyManager integration for prefetch
- GPT-OSS-120B Harmony encoding support
- Tool/function calling support

### GPT-OSS-120B Harmony Encoding (CRITICAL)

GPT-OSS-120B uses "Harmony" format for structured output with channels:
- `<|channel|>analysis<|message|>` - Internal reasoning (hidden from user)
- `<|channel|>final<|message|>` - User-visible response
- `<|channel|>commentary to=functions.{name}<|message|>{args}` - Tool calls

**Required imports**:
```python
from vllm.entrypoints.openai.parser.harmony_utils import (
    parse_chat_output,
    parse_chat_inputs_to_harmony_messages,
    render_for_completion,
    get_system_message,
    get_developer_message,
    parse_output_into_messages,
    parse_output_message,
)
from openai.types.responses import ResponseFunctionToolCall
```

**Input encoding** (convert chat messages to Harmony tokens):
```python
# MUST use Harmony encoding for GPT-OSS input
chat_msgs = [{"role": m.role, "content": m.content} for m in messages]
sys_msg = get_system_message(with_custom_tools=has_tools)
harmony_msgs = [sys_msg] + parse_chat_inputs_to_harmony_messages(chat_msgs)
prompt_token_ids = render_for_completion(harmony_msgs)
# Pass to vLLM as: llm.generate([{"prompt_token_ids": prompt_token_ids}], ...)
```

**Tool encoding** (CRITICAL - must use proper objects, not dicts):
```python
# WRONG - causes AttributeError: 'dict' object has no attribute 'type'
tools_for_harmony = [{"type": "function", "function": {...}}]

# CORRECT - use ChatCompletionToolsParam objects
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
    tool_param = ChatCompletionToolsParam(type="function", function=func_def)
    tools_for_harmony.append(tool_param)
dev_msg = get_developer_message(tools=tools_for_harmony)
```

**Output parsing** (extract tool calls and final content):
```python
parser = parse_output_into_messages(list(output_token_ids))
for msg in parser.messages:
    response_items = parse_output_message(msg)
    for item in response_items:
        if isinstance(item, ResponseFunctionToolCall):
            # Tool call: item.call_id, item.name, item.arguments
            pass
        elif item.type == "message":
            # Final content: item.content[0].text
            pass
```

**Stop tokens** (required for proper generation termination):
```python
from vllm.entrypoints.openai.parser.harmony_utils import get_stop_tokens_for_assistant_actions
stop_tokens = get_stop_tokens_for_assistant_actions()  # Returns [200002, 200012]
sampling_params = SamplingParams(..., stop_token_ids=stop_tokens)
```

**Token limits** (IMPORTANT):
GPT-OSS outputs in Harmony format with reasoning (analysis channel) before the response (final channel). If `max_tokens` is too low, the model may be truncated mid-reasoning and never output the final channel:
- `max_tokens=50` → Often truncated during reasoning, returns corrupted raw text
- `max_tokens=200` → Usually sufficient for simple responses
- For complex queries, use higher limits (500+)

When the model is truncated (`finish_reason="length"`), `parse_chat_output()` may return `reasoning` content but no `final_content`. The server should handle this gracefully.

**Conversation history corruption** (CRITICAL):
If Harmony encoding fails (e.g., due to dict vs object issue), the raw output looks like:
```
assistantanalysisWe need to...assistantanalysis to=functions.bash code{"command":"ls"}
```
This is TEXT that LOOKS like Harmony but isn't properly tokenized. If this gets stored in client conversation history and sent back in subsequent requests, the model will imitate this pattern instead of generating proper Harmony tokens.

**Fix**: Client must clear conversation history completely when this corruption occurs.

### Model Compatibility

| Model | Status | Notes |
|-------|--------|-------|
| gpt-oss-120b | ✅ Works | Requires Harmony encoding |
| qwen3-vl-32b-thinking | ✅ Works | Vision model, 128K context max |
| qwen3-32b | ✅ Works | Text model |
| mistral-small-24b | ✅ Works | Text model |
| llama-3.1-70b | ✅ Works | AWQ quantized |
| kimi-vl | ✅ Works | Vision model, tested lucid |
| glm-4.6v-awq | ✅ Works | Requires config fix (see below) |

### GLM-4.6V-AWQ Config Fix (Jan 2026)

The cyankiwi/GLM-4.6V-AWQ-4bit model has typos in its processor config files (`Glm46V` instead of `Glm4v`). Fix by editing the model's local cache:

```python
# Fix preprocessor_config.json and video_preprocessor_config.json
import json, os

model_path = os.path.expanduser(
    '~/.cache/huggingface/hub/models--cyankiwi--GLM-4.6V-AWQ-4bit/snapshots/*'
)
# Use glob to find the actual path

# Fix preprocessor_config.json
preproc = json.load(open(f'{model_path}/preprocessor_config.json'))
preproc['image_processor_type'] = 'Glm4vImageProcessor'  # was Glm46VImageProcessor
preproc['processor_class'] = 'Glm4vProcessor'  # was Glm46VProcessor
json.dump(preproc, open(f'{model_path}/preprocessor_config.json', 'w'), indent=2)

# Fix video_preprocessor_config.json
video_preproc = json.load(open(f'{model_path}/video_preprocessor_config.json'))
video_preproc['video_processor_type'] = 'Glm4vVideoProcessor'  # was Glm46VVideoProcessor
video_preproc['processor_class'] = 'Glm4vProcessor'  # was Glm46VProcessor
json.dump(video_preproc, open(f'{model_path}/video_preprocessor_config.json', 'w'), indent=2)
```

After this fix, GLM-4.6V-AWQ loads and passes all lucidity tests (math, knowledge, self-awareness).

### Non-Harmony Models

For models other than GPT-OSS, skip Harmony encoding entirely:
```python
use_harmony = HAS_HARMONY and model_id == "gpt-oss-120b"
if not use_harmony:
    # Use standard chat template formatting
    prompt = _format_chat_messages(request.messages)
```
