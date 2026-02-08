# Plan: Fast In-Process Model Switching

## Goal
Switch models in ONE process as close to pure hardware speed as possible.

**Target:** 66GB weights @ PCIe 5.0 x16 (~64 GB/s) = ~1s theoretical minimum

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Memory Cleanup | **COMPLETE** | unload_vllm_model() implemented |
| Phase 2: Profile Bottleneck | **COMPLETE** | Weight loading is bottleneck (3-7 GB/s) |
| Phase 3: Optimize Loading | **IN PROGRESS** | Contiguous/pinned memory didn't help |
| Phase 4: ROCm Investigation | DEFERRED | Focus on NVIDIA first |

## Phase 1: Implement Proper vLLM Memory Cleanup ✓ COMPLETE

### Problem (SOLVED)
vLLM's `shutdown()` only cleans profiler, not model weights.

### Solution: `unload_vllm_model()` in `blitz_inproc_switch.py`

Implemented proper cleanup that:
1. Clears model weights (`param.data = torch.empty(0, device='cpu')`)
2. Clears buffers
3. Clears KV cache tensors
4. Clears static forward context
5. Destroys distributed process groups via `parallel_state.destroy_distributed_environment()`
6. Flushes CUDA cache

**Results:**
- Unload time: **0.21s** (very fast!)
- Memory freed: **71.3 GiB**
- In-process switching: **WORKS**

## Phase 2: Profile Actual Loading Bottleneck ✓ COMPLETE

### Findings (RTX PRO 6000)

| Transfer Method | Speed | Notes |
|-----------------|-------|-------|
| Regular CPU→GPU | 8.6 GB/s | Non-contiguous safetensor mmap |
| Contiguous CPU→GPU | 18.9 GB/s | After `.clone()` |
| Pinned memory→GPU | 45 GB/s | Near hardware limit |

**But**: Clone and pin overhead negates the transfer speedup!

Full model loading:
- Direct GPU load: **9.09s** (7.2 GB/s) - BEST
- Contiguous sequential: 15.00s (4.4 GB/s) - SLOWER
- Contiguous parallel: 12.74s (5.1 GB/s) - SLOWER

**Root cause**: Safetensors uses mmap which creates non-contiguous tensors. The `.clone()` to make them contiguous takes longer than the transfer speedup saves.

### Current Switch Performance

| Operation | Time |
|-----------|------|
| GPT-OSS-120B load | 18.0s |
| Unload | **0.21s** |
| Qwen3-VL-32B load | 36.0s |
| **Total switch** | **36.2s** |

## Phase 3: Optimize Loading Path - IN PROGRESS

### Attempted Optimizations

| Approach | Result | Notes |
|----------|--------|-------|
| Contiguous memory | **SLOWER** | Clone overhead > transfer speedup |
| Pinned memory | **SLOWER** | Copy-to-pinned overhead > transfer speedup |
| Parallel file loading | Marginal | GIL limits parallelism |

### Remaining Ideas

1. **Custom safetensor loader**: Load directly to contiguous/pinned memory without intermediate mmap
2. **Model skeleton caching**: Keep model structure, only reload weights for same-architecture switches
3. **Prefetch/pipelining**: Start loading next model while serving current one
4. **RAM cache**: Keep model weights in system RAM for faster reload
5. **GPUDirect Storage**: Bypass CPU entirely (requires compatible hardware)

## Phase 4: Document ROCm Driver State Corruption - DEFERRED

Documented separately in `ROCM_780M_MEMORY_LEAK_BUG.md`. Will investigate after NVIDIA optimization is complete.

## Key Files

- `blitz_inproc_switch.py` - Production-ready unload implementation
- `blitz_raw_profile.py` - Profiling script for bottleneck analysis
- `blitz_contiguous_loader.py` - Contiguous memory optimization tests
- `INPROC_SWITCHING_RESULTS.md` - Detailed performance results

## Success Criteria Progress

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Load time (66GB) | ~24s | ~9s (weights) | <5s |
| Memory after unload | Not freed | Freed 71 GiB | <1GB held |
| In-process switches | FAILS | **WORKS** | Works |

## Next Steps

1. Investigate custom safetensor loading to avoid mmap overhead
2. Profile Qwen loading (slower than expected at 3.2 GB/s)
3. Consider model skeleton caching for same-architecture switches
