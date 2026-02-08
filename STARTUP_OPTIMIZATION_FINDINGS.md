# vLLM V1 Startup Optimization Findings

## Executive Summary

Achieved **instant model switching (0ms)** between pre-loaded models using multi-load architecture.

| Approach | Switch Time | Memory Usage |
|----------|-------------|--------------|
| Multi-Load (hot set) | **0ms** | ~32GB per 7B model |
| Cold Subprocess | ~28s | 20GB during load |
| Hybrid | 0ms hot / 28s cold | Configurable |

Reduced initial Qwen2.5-7B startup time from **80s to 14s** (82% improvement) through configuration changes and a new `VLLM_SKIP_WARMUP` env var.

## Hardware Profile
- **GPU**: AMD Radeon 780M iGPU (gfx1100, ~4 TFLOPS)
- **Memory**: 96GB unified memory (shared CPU/GPU)
- **Platform**: ROCm 7.2 + PyTorch 2.9.1

## Timing Breakdown (Default Configuration)

| Phase | Time | Notes |
|-------|------|-------|
| Process spawn | ~5s | vLLM V1 uses multiprocessing |
| Import vllm | 2.5s | First run only, cached after |
| Model loading | 5.2s | ~14GB at ~2.8GB/s |
| Memory profiling | **60s** | Forward pass with 16384 tokens |
| KV cache creation | ~1s | Tensor allocation |
| Warmup | ~4s | Sampler/kernel warmup |
| **Total** | **~80s** | |

## Root Cause: Memory Profiling

The 60-second bottleneck was in `determine_available_memory()`:

```python
# vllm/v1/worker/gpu_worker.py
def determine_available_memory(self) -> int:
    with memory_profiling(...) as profile_result:
        self.model_runner.profile_run()  # Forward pass with max_num_batched_tokens
```

The `profile_run()` runs a forward pass with `max_num_batched_tokens=16384` tokens.
On a slow iGPU (~4 TFLOPS), this takes ~60 seconds.

## Optimizations Applied

### 1. Reduce max_num_batched_tokens (MAJOR)
```python
LLM(..., max_num_batched_tokens=1024)  # Default is 16384
```
- **Impact**: 65s → 11s init engine
- **Trade-off**: Limits concurrent token processing

### 2. Pre-specify KV cache size
```python
LLM(..., kv_cache_memory_bytes=4 * 1024 * 1024 * 1024)
```
- **Impact**: Skips memory profiling (~5s savings)
- **Trade-off**: Manual KV cache management

### 3. Use enforce_eager (already applied)
```python
LLM(..., enforce_eager=True)
```
- **Impact**: Skips CUDA graph capture
- **Trade-off**: Slightly slower inference per-token

## Optimized Configuration

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="float16",
    gpu_memory_utilization=0.40,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    kv_cache_memory_bytes=4 * 1024 * 1024 * 1024,
    enforce_eager=True,
)
```

**Result**: 24s total startup (vs 80s default)

## Timing Breakdown (Optimized)

| Phase | Time | Notes |
|-------|------|-------|
| Process spawn | ~5s | Still required |
| Model loading | 5.2s | Unchanged (I/O bound) |
| Init engine | 9.6s | Profile + warmup |
| **Total** | **24s** | |

## Hardware Utilization During Startup

- **GPU**: 99% utilized during memory profiling
- **CPU**: ~15% utilized
- **I/O**: Negligible (model cached in page cache)

The bottleneck is **GPU computation**, not I/O or CPU.

## Remaining Overhead (24s)

| Component | Time | Cacheable? |
|-----------|------|------------|
| Process spawn | ~5s | No (could keep worker alive) |
| Model loading | ~5s | Yes (keep in RAM) |
| Profile run | ~5s | Partially (compilation artifacts) |
| Warmup | ~4s | No (GPU-bound) |
| Other | ~5s | Misc initialization |

## Path to <10 Second Startup

To achieve <10s second-run startup:

1. **Keep worker process alive** (-5s)
   - Don't spawn new process for each model switch
   - Requires architectural change to BlitzInfer

2. **Keep weights in system RAM** (-5s)
   - Pre-load model weights to RAM
   - Use mmap with MAP_POPULATE
   - iGPU can read from system RAM directly

3. **Skip profile_run on subsequent loads** (-5s)
   - Cache compilation artifacts
   - Use pre-computed tensor shapes

4. **Warmup optimization** (-4s)
   - Serve first request as warmup
   - Pre-warm common batch sizes

### Theoretical minimum: ~5-8s
- Model transfer from RAM: ~5s (14GB at ~3GB/s)
- Minimal warmup: ~2-3s

## Files Modified/Created

- `test_reduced_batch.py` - Batch size testing
- `test_fast_startup.py` - Optimized startup testing
- `profile_with_hw_monitor.py` - Hardware utilization monitoring
- `profile_internals.py` - Internal timing hooks

## Next Steps

1. Implement persistent worker pool (avoid process spawn)
2. Implement model pre-loading to RAM
3. Investigate triton cache for kernel warmup
4. Test with keep-alive architecture

## Key Learnings

1. **GPU computation dominates** - Not I/O, not CPU, not synchronization
2. **Batch size matters** - max_num_batched_tokens directly affects warmup time
3. **Memory profiling is expensive** - Profile run with large batch is the bottleneck
4. **iGPU is slow** - 4 TFLOPS vs 40+ TFLOPS on discrete GPUs
5. **First request is not warmed up** - But it's the same speed as subsequent requests
6. **No caching between runs** - vLLM spawns new worker process each time
7. **Process spawn is fixed overhead** - ~5s for multiprocessing setup

## Second Run Results

With vLLM's multiprocessing architecture, each LLM instantiation:
- Spawns a new worker process
- Loads model weights fresh
- Runs warmup from scratch

**Result**: Second run is same as first run (~24s)

To achieve faster second-run times, BlitzInfer must:
1. Keep worker processes alive between model switches
2. Pre-load model weights into shared memory
3. Potentially use vLLM V0 (in-process) instead of V1 (multiprocessing)

## Recommendation for BlitzInfer

For fast model switching on iGPU:

```python
# Optimal configuration
llm = LLM(
    model=model_name,
    dtype="float16",
    max_model_len=1024,
    max_num_batched_tokens=1024,
    kv_cache_memory_bytes=4 * 1024 * 1024 * 1024,
    enforce_eager=True,
)
```

For sub-10-second switching, architectural changes needed:
1. Persistent worker pool
2. Model weight pre-staging
3. Skip redundant warmup on model reload

## ROCm _C Module Workaround

When the vLLM _C custom ops module fails to load (due to ROCm version mismatch), use
native PyTorch ops via `compilation_config={"custom_ops": ["none"]}`.

### Modified Files in vLLM (for native ops fallback)
- `vllm/model_executor/layers/activation.py` - Added `self.enabled()` guard before
  loading _C ops to prevent crash when ops are disabled

### Usage
```python
llm = LLM(
    model="...",
    compilation_config={"custom_ops": ["none"]},  # Use PyTorch native ops
    ...
)
```

This bypasses the broken _C module and uses PyTorch's native implementations
(e.g., `F.silu(x[..., :d]) * x[..., d:]` instead of `torch.ops._C.silu_and_mul`).

## NEW: VLLM_SKIP_WARMUP Environment Variable

Added `VLLM_SKIP_WARMUP=1` to vLLM source to skip warmup phases entirely.
The first inference request acts as the warmup instead.

### Modified Files in vLLM
- `vllm/envs.py` - Added VLLM_SKIP_WARMUP environment variable
- `vllm/v1/worker/gpu_worker.py` - Check env var to skip profile_run, kernel_warmup, sampler warmup

### Results with VLLM_SKIP_WARMUP=1

| Metric | Without Skip | With Skip | Improvement |
|--------|--------------|-----------|-------------|
| Init time | 24s | **14s** | -10s |
| Init engine | 9.6s | **0.24s** | -9.4s |
| First inference | 2.3s | 4.7s | +2.4s (lazy warmup) |
| Time to first response | 26.3s | **19s** | **-7.3s** |

The first few inferences are slower due to Triton JIT compilation, but by the
third request, inference is at normal speed (~2s).

### Final Optimized Configuration

```bash
# Environment variables
export VLLM_SKIP_WARMUP=1
export VLLM_DEEP_GEMM_WARMUP=skip
```

```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="float16",
    max_model_len=1024,
    max_num_batched_tokens=1024,
    kv_cache_memory_bytes=4 * 1024 * 1024 * 1024,
    enforce_eager=True,
)
```

### Current Achievable Times
- **Init**: 14s
- **Time to first response** (with lazy warmup): 19s total
- **Subsequent responses**: ~2s

### Path to Sub-10-Second Startup
With `VLLM_SKIP_WARMUP=1`, remaining overhead is:
- Process spawn: ~5s
- Model loading: ~5s
- KV cache + misc: ~4s

To achieve <10s:
1. Keep worker alive between switches: -5s → **9s init**
2. Pre-stage weights in RAM: -5s → **4s init** (theoretical minimum)

## Model Switching Implementation

### RECOMMENDED: Multi-Load Instant Switching (model_switcher_multiload.py)

The fastest approach: **load all models at startup, switch instantly**.

| Operation | Time | Notes |
|-----------|------|-------|
| First model load | ~10s | Includes Triton JIT compilation |
| Subsequent model loads | ~5-6s | Triton already warmed |
| **Switch time** | **0ms** | Instant (pointer change) |
| Memory per 7B model | ~32GB | Includes KV cache |

```python
# Example: 2 models in 65GB, instant switching
switcher = ModelSwitcherMultiLoad(vram_limit_gb=80.0, max_models=4)
switcher.register_model("model_a", "path/to/model_a")
switcher.register_model("model_b", "path/to/model_b")
switcher.load_all()  # Load both at startup

# Instant switches (0ms)
switcher.activate("model_a")
switcher.generate(["prompt"])
switcher.activate("model_b")  # INSTANT!
switcher.generate(["prompt"])
```

**Capacity on 96GB unified memory:**
- 2x 7B models: ~65GB, 30GB free
- 4x 7B models: ~80-90GB (with smaller KV caches)
- Larger models scale proportionally

### Alternative: Subprocess-based Isolation (model_switcher_v2.py)

For scenarios where multi-load isn't feasible (e.g., too many large models):
- Each model runs in a separate subprocess
- Switching involves terminating one subprocess and starting another
- ~24s per switch but uses less peak memory

| Operation | Time | Notes |
|-----------|------|-------|
| Stop previous worker | ~1s | Clean shutdown |
| Subprocess spawn + imports | ~14s | Python + vLLM imports |
| Model weight loading | ~5s | From disk/page cache |
| Engine initialization | ~5s | KV cache setup |
| **Total switch time** | **~24s** | |

### Why In-Process Reload Doesn't Work

On ROCm/HIP with AMD 780M iGPU, deleting an LLM instance and creating a new one
in the same process causes GPU context corruption and hangs. Tested approaches:
- `model_switcher_inproc.py`: First load works, second switch causes GPU hang
- `model_switcher_persistent.py`: Same issue within persistent subprocess
- Memory not fully released (42GB free vs 63GB initial after cleanup)

This is a hardware/driver limitation, not a Python/vLLM issue.

### Key Configuration for Model Switching
```python
LLM(
    model=config.path,
    dtype="float16",
    gpu_memory_utilization=0.20,  # Respect VRAM limit
    max_model_len=1024,
    max_num_batched_tokens=1024,
    kv_cache_memory_bytes=2 * 1024**3,  # Smaller per-model for multi-load
    enforce_eager=True,
    compilation_config={"custom_ops": ["none"]},  # Native ops
)
```

### Hybrid Approach (model_switcher_hybrid.py)

Best of both worlds for 10+ model scenarios:
- Pre-load a "hot set" of frequently used models for instant switching
- Use subprocess-based switching for cold models when hot set is full
- LRU eviction policy for hot set management

```python
switcher = ModelSwitcherHybrid(
    hot_set_size=3,        # Keep 3 models loaded
    vram_budget_gb=80.0,   # Total VRAM for hot set
    cold_model_vram_gb=20.0,  # VRAM for cold subprocess
)
switcher.register_model("model_0", "path/to/model_0")
# ... register all 10+ models
switcher.preload(["model_0", "model_1", "model_2"])  # Load hot set

# Instant switches between hot models
switcher.activate("model_0")  # 0ms
switcher.activate("model_1")  # 0ms

# Cold switch for models not in hot set
switcher.activate("model_5")  # ~28s subprocess

# Return to hot model
switcher.activate("model_0")  # ~1s (stop subprocess)
```

| Switch Type | Time | Notes |
|-------------|------|-------|
| Hot → Hot | **0ms** | Instant |
| Hot → Cold | ~28s | Subprocess spawn |
| Cold → Hot | ~1s | Stop subprocess |

### Environment Variables
```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0  # In-process mode within worker
export VLLM_SKIP_WARMUP=1  # Skip warmup (first request warms up)
export VLLM_DEEP_GEMM_WARMUP=skip  # Skip DeepGEMM warmup
```

## Implementation Files Summary

| File | Approach | Switch Time | Best For |
|------|----------|-------------|----------|
| `model_switcher_multiload.py` | Multi-load | **0ms** | Few models, lots of RAM |
| `model_switcher_hybrid.py` | Hot set + cold subprocess | 0ms hot / 28s cold | Many models |
| `model_switcher_v2.py` | Subprocess isolation | ~24s | Memory constrained |
| `model_switcher_inproc.py` | Single process | ❌ GPU hang | Not recommended (ROCm issue) |
| `model_switcher_persistent.py` | Persistent worker | ❌ GPU hang | Not recommended (ROCm issue) |
| `model_switcher_fork.py` | Fork-based | ~24s | Not recommended (CUDA issues) |

## Recommendations

**For 96GB unified memory with 7B models:**

1. **Best option**: Use `model_switcher_multiload.py` to pre-load 4-5 models
   - Instant switching (0ms) between all loaded models
   - ~80GB for 4 models, leaves ~16GB free

2. **For 10+ models**: Use `model_switcher_hybrid.py`
   - Keep 3-4 most-used models in hot set (instant)
   - Cold load remaining models on demand (~28s)
   - Good usage patterns = mostly instant switches

3. **Avoid**: In-process model switching on ROCm (GPU hang issues)
