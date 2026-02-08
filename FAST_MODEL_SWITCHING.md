# Fast Model Switching on AMD 780M APU

## Summary

Achieved **~5.2 second full vLLM model switching** for 14GB 7B models on AMD Radeon 780M (gfx1103) with unified memory.

| Metric | Before (vLLM) | After (Blitz) | Improvement |
|--------|---------------|---------------|-------------|
| Weight Loading | 3.7s | **1.5s** | **2.5x faster** |
| Transfer Bandwidth | 2.0 GB/s | **9.6 GB/s** | **4.8x faster** |
| Full vLLM Switch | ~7s | **~5.2s** | **26% faster** |
| Cleanup | 1700ms | ~500ms | 3.4x faster |

Note: vLLM has ~3.2s fixed overhead for model architecture setup, tokenizer loading, and engine initialization that cannot be optimized without modifying vLLM core.

## Standalone Weight Loading (No vLLM Overhead)

For pure weight loading benchmarks:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Weight Load | 7000ms | **1600ms** | **4.4x faster** |
| Bandwidth | 2.0 GB/s | 9.0 GB/s | 4.5x faster |
| Cleanup | 1700ms | 55ms | 31x faster |

## Key Optimizations

### 1. Cleanup Optimization (1700ms → 55ms)

**Problem**: vLLM's `cleanup_dist_env_and_memory()` is slow due to multiple `gc.collect()` calls.

**Solution**: Bypass vLLM cleanup, directly clear nn.Module parameters:

```python
gc.unfreeze()  # Expose frozen objects
for obj in gc.get_objects():
    if isinstance(obj, nn.Module):
        if hasattr(obj, '_parameters') and obj._parameters:
            for key in list(obj._parameters.keys()):
                param = obj._parameters.get(key)
                if param is not None and param.is_cuda:
                    obj._parameters[key] = None
torch.cuda.empty_cache()
```

### 2. Weight Loading Optimization (7000ms → 1600ms)

**Problem**: Safetensors `load_file(device='cuda')` allocates each tensor individually, achieving only 2 GB/s.

**Solution**: Double-buffered loading with pinned memory:

1. **Pre-allocate pinned CPU buffers** (one-time startup cost)
2. **Pre-allocate GPU buffer** (reuse across loads)
3. **Double buffering**: Load to CPU buffer A while transferring buffer B to GPU
4. **Non-blocking transfers**: Overlap I/O and compute

```python
# One-time setup
cpu_bufs = [torch.empty(max_size, pin_memory=True) for _ in range(2)]
gpu_buffer = torch.empty(total_elements, device='cuda')

# Per-load (achieves 9 GB/s)
for i, (file, size) in enumerate(files):
    cpu_buf = cpu_bufs[i % 2]
    load_to_cpu(file, cpu_buf)
    if i > 0: torch.cuda.synchronize()  # Wait for prev transfer
    gpu_buffer[offset:].copy_(cpu_buf[:size], non_blocking=True)
```

### 3. Bandwidth Analysis

| Operation | Bandwidth | Notes |
|-----------|-----------|-------|
| Theoretical max | ~20 GB/s | In-GPU operations |
| CPU→GPU single tensor | 10-11 GB/s | Hardware limit for APU |
| Safetensors default | 2.0 GB/s | Per-tensor allocation |
| **Optimized bulk** | **9.0 GB/s** | Near hardware limit |

## Implementation Files

| File | Purpose |
|------|---------|
| `blitz_vllm_patch.py` | **vLLM patch for fast weight loading (recommended)** |
| `blitz_vllm_integration.py` | Full integration test and benchmarks |
| `blitz_fast_switcher.py` | Standalone fast switcher (without vLLM patch) |
| `test_full_vllm_switch.py` | Comprehensive vLLM switching tests |
| `fast_weight_loader.py` | Weight loading optimization tests |
| `bandwidth_investigation.py` | Bandwidth analysis |

## Usage

### Option 1: vLLM Patch (Recommended)

Patch vLLM to use fast weight loading:

```python
# Import and apply patch BEFORE importing vLLM
import blitz_vllm_patch
blitz_vllm_patch.patch_vllm()

# Then use vLLM normally - it will use fast weight loading
from vllm import LLM, SamplingParams

# First load (includes 2.3s one-time buffer allocation)
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", ...)

# Fast cleanup + switch (~5.2s total)
del llm
gc.collect()
blitz_vllm_patch.fast_cleanup()
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", ...)  # Uses 9.6 GB/s loading
```

### Option 2: Standalone Fast Switcher

For custom integration without modifying vLLM:

```python
from blitz_fast_switcher import BlitzFastSwitcher

# Initialize (pre-allocates buffers)
switcher = BlitzFastSwitcher(max_model_size_gb=16)

# First load
llm = switcher.load("Qwen/Qwen2.5-7B-Instruct")
output = llm.generate(["Hello"], params)

# Fast switch
llm = switcher.switch("mistralai/Mistral-7B-v0.3")
```

## Hardware Requirements

- AMD Radeon 780M (gfx1103) or similar APU
- ROCm 7.2+
- 64GB+ unified memory
- PyTorch with ROCm support

## Lessons Learned

1. **Pinned memory is critical** for fast CPU→GPU transfers on APU
2. **Pre-allocation eliminates variance** - first allocations are slow
3. **Double buffering overlaps I/O** - load next while transferring current
4. **Single bulk transfers** beat many small transfers (2 GB/s → 9 GB/s)
5. **gc.freeze()** hides objects - need `gc.unfreeze()` before cleanup
6. **Skip vLLM's gc.collect()** calls - they're expensive (~240ms each)

## Theoretical Limits

For 14.2 GB model at 10 GB/s hardware limit:
- **Minimum transfer time**: 1.4 seconds
- **Achieved**: 1.6 seconds (88% of theoretical)
- **Overhead**: ~200ms for file parsing, buffer management

## vLLM Overhead Analysis

Full vLLM switch time breakdown (5.2s total):

| Component | Time | Notes |
|-----------|------|-------|
| Cleanup (fast_cleanup) | 500ms | Could be reduced with warm objects |
| **Blitz weight transfer** | **1500ms** | At hardware limit (9.6 GB/s) |
| vLLM model setup | ~1200ms | Creating layers, parameter allocation |
| Tokenizer loading | ~500ms | Could cache for same-model switches |
| Engine/scheduler init | ~120ms | KV cache allocation |
| Config/validation | ~1000ms | Model config parsing, dtype checks |

### Potential Further Optimizations

To get below 3 seconds would require:

1. **Cache tokenizer** - Save ~500ms when switching between same-family models
2. **Persistent model shell** - Keep vLLM engine alive, only swap weights
3. **Warm parameter allocation** - Pre-allocate parameter tensors

These would require significant vLLM modifications and are beyond the scope of the current patch approach.

## Test Results (2026-01-26)

```
BLITZ-PATCHED VLLM TEST
======================================================================

TEST 1: Load with Blitz Patch
[BlitzPatch] Pre-allocating buffers for fast weight loading...
[BlitzPatch]   GPU buffer: 16.0 GB
[BlitzPatch]   CPU buffers: 2 x 4.0 GB (pinned)
[BlitzPatch]   Pre-alloc time: 2278ms
[BlitzPatch] Loading 14.2 GB via bulk transfer...
[BlitzPatch] Loaded 14.2 GB in 1466ms (9.7 GB/s)
INFO: Loading weights took 3.93 seconds (first load includes buffer alloc)

TEST 2: Fast Switch Cycle
Cleanup: 527ms (cleared 199 params)
[BlitzPatch] Loading 14.2 GB via bulk transfer...
[BlitzPatch] Loaded 14.2 GB in 1471ms (9.6 GB/s)
INFO: Loading weights took 1.66 seconds
Reload: 4685ms
TOTAL SWITCH: 5212ms (5.21s)

Output: "Paris. Capital of Spain? Madrid. Capital of Italy? Rome..."
Valid: True
```
