# Cross-Architecture Memory Leak Investigation Plan

## Problem Statement
When switching from Qwen-32B to gpt-oss-120b, ~42GB of GPU memory remains allocated even after cleanup. This prevents loading the new model.

## Key Observation
- Model A (Qwen-32B): ~34GB weights + ~7GB KV cache = ~41GB
- After cleanup: Only ~5.7GB freed (KV cache), ~34GB weights remain
- PyTorch reports 93.66GB allocated during new model construction

## Investigation Steps

### Phase 1: Memory Profiling
1. **Baseline measurement**: Track exactly what's allocated before/after each step
2. **PyTorch memory snapshot**: Use torch.cuda.memory._snapshot() to see all allocations
3. **Reference counting**: Find what's holding references to GPU tensors

### Phase 2: vLLM Internal State Analysis
1. **Model references**: Trace all paths from LLM object to GPU tensors
2. **Global state**: Find all vLLM singletons/module-level caches
3. **Worker state**: Check if worker threads hold references
4. **Attention backends**: Check for cached attention state

### Phase 3: PyTorch/CUDA Analysis
1. **Caching allocator**: Understand why cached memory isn't released
2. **NCCL state**: Check distributed communication buffers
3. **CUDA context**: Check for context-level allocations

### Phase 4: Fix Implementation
1. Implement proper cleanup for each identified leak
2. Verify memory is truly released
3. Test cross-architecture switching

## Success Criteria
- GPU memory usage after cleanup < 5GB (CUDA context overhead only)
- Switching Qwen-32B ↔ gpt-oss-120b works in < 10 seconds
- Multiple switches work without memory accumulation
