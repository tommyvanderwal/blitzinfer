# vLLM Caching Analysis for Fast Model Switching

## Executive Summary

With pinned arena loading + premerge + warm caches, we achieve **4.88s** model switch time (down from 18.74s cold start). The remaining ~3s overhead is from KV cache profiling that runs on every engine init.

## Performance Breakdown

| Scenario | Time | Savings |
|----------|------|---------|
| Cold start (first load) | 18.74s | baseline |
| Same process (warm cache) | 8.27s | 10.47s (56%) |
| Pinned arena (1st) | 4.88s | 13.86s (74%) |
| Pinned arena (2nd) | 4.84s | 13.90s (74%) |

## What Gets Cached

### 1. Triton Kernel Compilation (~5-7s savings)
- **Location**: `~/.triton/cache/`
- **Persists**: Across processes, across reboots
- **Contents**: Compiled PTX/CUBIN for FlashAttention, FP8 quantization ops, etc.

### 2. PyTorch CUDA Context (~2-3s savings)
- **Location**: In-memory (per process)
- **Persists**: Same process only
- **Contents**: CUDA driver init, cuBLAS/cuDNN handles, memory allocator state

### 3. Tokenizer/Config (~1-2s savings)
- **Location**: In-memory (per process) + HuggingFace cache
- **Persists**: Partially across processes (config files cached on disk)
- **Contents**: Tokenizer vocabulary, model config JSON

## Remaining Overhead Analysis (4.88s pinned arena load)

| Component | Time | Notes |
|-----------|------|-------|
| Weight transfer | 0.80s | 45.1 GB/s from pinned memory |
| Model construction | ~0.15s | Creating PyTorch modules |
| **KV cache profiling** | **~3.2s** | `profile_run()` + memory allocation |
| Other init | ~0.7s | Tokenizer, config, scheduler |

## The `profile_run()` Problem

Even when `kv_cache_memory_bytes` is set, vLLM still runs `profile_run()`:

```python
# From vllm/v1/worker/gpu_worker.py:296
if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
    # still need a profile run which compiles the model for
    # max_num_batched_tokens
    self.model_runner.profile_run()  # ← ALWAYS RUNS!
    return kv_cache_memory_bytes
```

The `profile_run()` does:
1. MM encoder warmup (for vision models) - skippable with `VLLM_SKIP_WARMUP=1`
2. `_dummy_run()` with `is_profile=True` - **NOT skippable**

The `_dummy_run()` pre-allocates communication buffers and triggers Triton compilation for the specific batch size.

## Optimization Opportunities

### Option 1: Skip `profile_run()` (saves ~3s)
**Difficulty**: Medium (requires vLLM patch)
**Approach**: When `kv_cache_memory_bytes` is set AND Triton cache is warm, skip the entire `profile_run()`
**Risk**: May miss buffer pre-allocation, first inference might be slightly slower

### Option 2: Persistent Engine (saves ~4s)
**Difficulty**: High (complex state management)
**Approach**: Keep engine alive, only swap model weights in-place
**Benefit**: Skip model construction + profile_run entirely
**Result**: ~0.8s switch time (weight transfer only)

### Option 3: Pre-computed KV Cache Size (saves ~0.5s)
**Difficulty**: Low
**Approach**: Store computed KV cache size for known models, pass via `kv_cache_memory_bytes`
**Benefit**: Skip memory profiling portion of profile_run

## kv_cache_memory_bytes Test Results

Testing with `kv_cache_memory_bytes` to skip memory profiling:

| Scenario | Init Time | Notes |
|----------|-----------|-------|
| Without kv_cache_memory_bytes | 14.48s | Normal path |
| With kv_cache_memory_bytes | 23.45s | **SLOWER!** |

**Finding**: Setting `kv_cache_memory_bytes` actually makes things SLOWER because:
1. The `VLLM_SKIP_WARMUP` check in vLLM V1 doesn't work as expected
2. `profile_run()` still executes even when `kv_cache_memory_bytes` is set
3. Something in that code path takes longer

**Conclusion**: `kv_cache_memory_bytes` is NOT a viable optimization path.

## Recommended Path Forward

1. **Immediate**: Use pinned arena with pre-merge (4.88s achieved) - **BEST OPTION**
2. **Future**: Investigate persistent engine for sub-1s weight swaps
3. **Not recommended**: Setting `kv_cache_memory_bytes` (makes things slower)

## Test Results Detail

```
Load 1 - Standard vLLM:
  - Model loading took 33.46 GiB memory and 5.32s
  - init engine (profile, create kv cache, warmup) took 3.71s
  - Total: 18.74s

Load 2 - Same process:
  - Model loading took 33.21 GiB memory and 4.89s
  - init engine took 2.78s
  - Total: 8.27s

Load 3 - Pinned arena:
  - Weight injection: 0.79s (45.1 GB/s)
  - Model loading took 0.93s total
  - init engine took 3.28s
  - Total: 4.88s

Load 4 - Pinned arena (2nd):
  - Weight injection: 0.80s (44.3 GB/s)
  - Model loading took 0.99s total
  - init engine took 3.22s
  - Total: 4.84s
```

## Key Insight

The warm cache savings (10.47s) happen automatically via Triton's disk cache. The **remaining 3s bottleneck** is the `profile_run()` which runs a forward pass to:
1. Compile model for specific batch size
2. Pre-allocate communication buffers
3. Profile encoder cache for vision models

This runs every time a new LLM engine is created, even when Triton kernels are cached, because it also handles buffer allocation.
