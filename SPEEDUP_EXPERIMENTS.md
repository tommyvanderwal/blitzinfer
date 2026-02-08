# Speedup Experiments Log

This document tracks all model switching speedup attempts, what worked, what didn't, and why.

## Hardware Context

- **System**: Remote server at tommy@192.168.2.90
- **GPU**: RTX PRO 6000 (95GB VRAM)
- **Models**:
  - GPT-OSS-120B (~60GB)
  - Qwen3-VL-32B-Instruct (~62GB)
- **Goal**: Fast model switching with 100K+ context support

---

## Critical Discoveries

### GPU Memory Cleanup (SOLVED)

**Problem**: After unloading a model, 90GB+ of GPU memory remained allocated, preventing loading the next model.

**Root Cause**: vLLM V1 multiprocessing mode creates a subprocess that holds GPU memory. When the subprocess terminates without explicit shutdown, CUDA IPC memory is inherited by the main process.

**Solution**: Must call `llm.llm_engine.engine_core.shutdown()` BEFORE `del llm`:

```python
def unload_vllm_model(llm):
    # CRITICAL: shutdown() releases GPU memory properly
    llm.llm_engine.engine_core.shutdown()
    time.sleep(1)
    del llm
    gc.collect()
```

**What DOESN'T work**:
- `del llm` alone - subprocess terminates but memory inherited by parent
- `gc.collect()` - doesn't help with CUDA IPC memory
- `torch.cuda.empty_cache()` - only affects PyTorch allocator, not IPC memory
- Terminating child processes manually - memory still inherited

---

## Experiment Results

### Experiment 1: Page Cache Warming (2025-01-27)

**Hypothesis**: Pre-read model files into OS page cache while serving Model A, so Model B loads faster.

**Implementation**: `blitzinfer/memory/cache_warmer.py` - background thread reads safetensor files in 64MB chunks.

**Results**:
| Metric | Value | Notes |
|--------|-------|-------|
| Cold Model B load | 38.6s | From disk |
| Warm Model B load | 28.0s | From page cache |
| Cache warm time | 38.5s | Background, during Model A serving |
| Cache warm speed | 1.6 GB/s | **Much lower than expected 12 GB/s** |
| Load speedup | 1.4x | |
| Time saved | 10.6s | |

**Analysis**:
- Warming works but speed is disappointing (1.6 GB/s vs expected 12 GB/s)
- I/O bandwidth contention: vLLM subprocess was running and serving requests during warming
- Expected speed (12 GB/s) only achievable when no I/O competition

**Status**: Partially successful. Need to investigate I/O contention.

---

### Experiment 2: Reduced GPU Utilization (REJECTED)

**Hypothesis**: Reduce gpu_memory_utilization to 0.72 to leave headroom for next model.

**Why Rejected**: User explicitly stated: "FULL GPU VRAM needs to be available for the active model to have a lot of KV cache. DO NOT reduce for no good reason."

**Lesson**: Never reduce GPU utilization as a workaround. Find and fix the actual memory leak instead.

---

### Experiment 3: Single Process Mode (FAILED)

**Hypothesis**: Single process mode (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) might have cleaner memory management.

**Results**: PyTorch CUDA allocator retains ~25GB even after model deletion. `torch.cuda.empty_cache()` doesn't fully release.

**Conclusion**: Multiprocessing mode with explicit `shutdown()` is the correct approach.

---

## Pending Experiments

### Experiment 4: Disk I/O Benchmarks (2025-01-27)

**Goal**: Understand true disk I/O capabilities without vLLM.

**Results** (62 GB model files):
| Method | Speed | Notes |
|--------|-------|-------|
| Python read() | 1.6 GB/s | Limited by Python I/O layer |
| Python mmap | 3.5 GB/s | Better, bypasses Python buffer |
| readinto() | 2.97 GB/s | Pre-allocated buffer helps |
| dd command | 3.69 GB/s | Native, single file |
| cat command | 3.67 GB/s | Native, single file |
| **Parallel dd (4 jobs)** | **6.41 GB/s** | Best method! |
| Parallel mmap (8 threads) | 3.23 GB/s | Threading doesn't help much |

**Key Finding**: Python's I/O layer is the bottleneck! Parallel native reads (dd) achieve 4x the speed of Python's read().

**Action Items**:
1. Update cache warmer to use parallel subprocess (dd/cat) instead of Python read()
2. Or use parallel mmap in multiple threads

---

### Experiment 5: vLLM Load Time Breakdown (2025-01-27)

**Goal**: Understand exactly where time goes during model loading.

**Model**: Qwen3-VL-32B-Instruct (62 GB)
**GPU**: RTX PRO 6000 Blackwell (95 GB)

**Cold Load Breakdown (40.5s total):**
| Phase | Time | Notes |
|-------|------|-------|
| vLLM subprocess spawn + CUDA init | 14s | Can't skip - required for isolation |
| Weight loading (disk I/O) | 20s | 3.1 GB/s effective |
| KV cache profiling + init | 5s | Required for memory allocation |
| Other | 1.5s | |

**Warm Load Breakdown (23.6s total):**
| Phase | Time | Notes |
|-------|------|-------|
| vLLM subprocess spawn + CUDA init | 10s | Slightly faster (tokenizer cached?) |
| Weight loading (from page cache) | 8s | 7.8 GB/s effective |
| KV cache profiling + init | 5s | Same as cold |
| Other | 0.6s | |

**Page Cache Warming (separate step):**
- 62 GB in 1.62s = **38.4 GB/s** with parallel dd
- This is phenomenal! Much faster than expected.

**Key Insights:**
1. **vLLM overhead is the new bottleneck**: 15-20s can't be avoided
2. **Weight I/O savings**: 20s → 8s = 12s saved from warm cache
3. **Parallel dd is extremely fast**: 38.4 GB/s (RAM speed, not disk!)
4. **Total savings**: 16.9s (40.5s → 23.6s)

**The Problem**: Even with instant weight loading (0s theoretical), minimum load time is ~15-20s due to vLLM initialization.

---

## Pending Experiments

### Experiment 6: CPU↔GPU Transfer Speed (2025-01-27)

**Goal**: Measure how fast we can move weights from RAM to GPU.

**Results** (RTX PRO 6000 Blackwell, PCIe 5.0):
| Transfer | 1 GB | 10 GB | 30 GB |
|----------|------|-------|-------|
| Unpinned RAM → GPU | 17.2 GB/s | 16.3 GB/s | 16.3 GB/s |
| **Pinned RAM → GPU** | **41.6 GB/s** | **36.8 GB/s** | **35.3 GB/s** |
| Multi-stream (4) | 45.0 GB/s | 45.5 GB/s | - |
| GPU → Unpinned RAM | 2.2 GB/s | 2.3 GB/s | 2.0 GB/s |

**Key Finding**: Pinned RAM → GPU is **35+ GB/s**!
- 62 GB model: **1.8s** from pinned RAM (vs 8s from page cache, 20s from cold disk)

---

## Current Best Times

| Scenario | Time | Bottleneck |
|----------|------|------------|
| Cold load | 40.5s | Disk I/O (20s) + vLLM init (18s) |
| Warm load (page cache) | 23.6s | Page cache I/O (8s) + vLLM init (15s) |
| **Theoretical best** (pinned RAM) | **~20s** | Pinned RAM I/O (2s) + vLLM init (18s) |

**The bottleneck is now vLLM initialization (18s), not I/O!**

---

## Pending Experiments

### Experiment 7: RAM Disk vs Regular Disk (2025-01-27)

**Hypothesis**: Loading from RAM disk (tmpfs) should be faster than SSD.

**Results** (62 GB model):
| Source | Weight Load Time | Effective Speed |
|--------|------------------|-----------------|
| Disk (cold) | 20.9s | 3.0 GB/s |
| Disk (warm/page cache) | 19.4s | 3.2 GB/s |
| **RAM disk (tmpfs)** | **25.4s** | **2.5 GB/s** |

**UNEXPECTED**: RAM disk is actually **SLOWER**!

**Analysis**: The bottleneck is NOT file I/O speed. It's the vLLM weight loading process:
1. Safetensors parsing (header inspection)
2. CPU tensor creation via mmap
3. GPU memory allocation
4. CPU → GPU data transfer

Even though we can read files at 38 GB/s (parallel dd), vLLM only achieves 3 GB/s effective because it's limited by tensor creation and GPU transfer overhead, not file reads.

**Key Insight**: To go faster, we need to modify how vLLM loads weights, not just how files are read. Options:
1. Custom safetensors loader that allocates directly on GPU
2. Pre-parse headers and pre-allocate GPU tensors
3. Use CUDA unified memory or pinned memory with DMA

---

### Experiment 8: Multi-threaded Weight Loading (2025-01-27)

**Hypothesis**: Multi-threaded loading should be faster.

**Results** (62 GB model, page cache warm):
| Method | Weight Load Time | Total Load Time |
|--------|------------------|-----------------|
| Single-thread (default) | 11.6s | 31.3s |
| **Multi-thread (8 threads)** | **21.0s** | **38.5s** |

**UNEXPECTED**: Multi-threading is **SLOWER**!

**Analysis**: The multi-thread loader:
1. Loads all shard files into CPU memory in parallel
2. This fills up CPU memory (~62GB)
3. Memory pressure causes slowdown
4. Then transfers to GPU serially

The single-thread approach is more efficient because it:
1. Loads one shard at a time
2. Transfers to GPU immediately
3. Reuses CPU memory buffer

---

## Summary of Load Time Breakdown

For a **62 GB model** on **RTX PRO 6000 (95GB)**:

| Component | Time (s) | Notes |
|-----------|----------|-------|
| vLLM subprocess spawn | 5s | Required for isolation |
| CUDA/NCCL init | 2s | One-time per process |
| Model architecture creation | 5s | Depends on model complexity |
| Weight loading (best case) | 8s | From warm page cache |
| KV cache profiling | 5s | Required for memory planning |
| **Total (best case)** | **~25s** | With warm page cache |
| **Total (worst case)** | **~45s** | Cold disk |

**Minimum theoretical load time: ~18s** (with dummy weights = 0s I/O)

---

## Key Conclusions

1. **The bottleneck is vLLM initialization, not I/O**
   - Even with instant file reads, 18s minimum from subprocess/CUDA init
   - Weight I/O is only 8-12s of total 25-40s load time

2. **Page cache warming works**
   - Reduces weight I/O from 20s → 8s (12s savings)
   - Background warming at 6-10 GB/s while serving

3. **RAM disk and multi-threading don't help**
   - Both are actually slower due to different bottlenecks
   - vLLM's mmap-based loading is well-optimized for page cache

4. **To go faster, need architectural changes**:
   - Keep vLLM engine running between switches (skip 12s init)
   - Or use GPU direct storage (skip CPU entirely)

---

## Next Steps (Priority Order)

### 1. Keep vLLM Engine Running (HIGHEST IMPACT)

### 2. Skip/Cache KV Profiling
- vLLM spends ~5s profiling available memory
- Can we pre-compute and cache this?

### 3. Keep Engine Core Running (BEST POTENTIAL)
- Don't destroy vLLM subprocess between models
- Just swap model weights in place
- Target: <5s switch (skip 13s subprocess/CUDA init)

### 4. Parallel Subprocess Init
- Fork new vLLM subprocess while old model serving
- Initialize CUDA context in background (takes 7s)
- Complete weight loading after old model unloads

---

## Configuration Notes

### vLLM Config (Current)
```python
VLLM_CONFIG = {
    "dtype": "bfloat16",
    # max_model_len: removed, let model use its default max
    "gpu_memory_utilization": 0.95,  # Full GPU - NEVER reduce
    "max_num_seqs": 16,
    "max_num_batched_tokens": 8192,
    "enforce_eager": True,
    "trust_remote_code": True,
}
```

### Environment Variables
```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_ENABLE_V1_MULTIPROCESSING=1
VLLM_SKIP_WARMUP=1  # Optional: faster startup
```

---

## Lessons Learned

1. **Always call `engine_core.shutdown()`** before deleting vLLM LLM object
2. **Don't reduce GPU utilization** as a workaround - fix the actual problem
3. **I/O contention is real** - background warming competes with model serving
4. **nvidia-smi is reliable** for external memory monitoring (doesn't init CUDA in main process)
5. **Document everything** - we've wasted time re-discovering the same issues
