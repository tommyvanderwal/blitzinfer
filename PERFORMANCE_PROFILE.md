# BlitzInfer Performance Profile

## Systems

### 780M (AMD Radeon 780M iGPU)
- **Memory**: 109GB unified DDR5
- **Platform**: ROCm 7.2 (gfx1103)
- **Required Settings**:
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  - `VLLM_SKIP_WARMUP=1` (skip profiling that causes HIP errors)
  - `kv_cache_memory_bytes=10-20*1024**3` (explicit KV cache allocation)
  - `enforce_eager=True`
  - `compilation_config={"custom_ops": ["none"]}`

### RTX PRO 6000 Blackwell (NVIDIA)
- **Memory**: 96GB VRAM
- **Platform**: CUDA 12.8, SM_120
- **Required Settings**:
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0`
  - `VLLM_ATTENTION_BACKEND=TORCH_SDPA` (FlashAttention not optimized for Blackwell yet)
  - `enforce_eager=True`

## Model Performance Summary

| Model | System | Size | Load Time | Generation Speed |
|-------|--------|------|-----------|-----------------|
| **GPT-OSS-120B** | 780M | 66 GiB (MXFP4) | ~96s | ~1.2 tok/s |
| **GPT-OSS-120B** | RTX PRO 6000 | 66 GiB (MXFP4) | ~24s | TBD |
| **Qwen3-VL-32B** | 780M | 63 GiB | ~60s | ~0.86 tok/s |
| **Qwen3-VL-32B** | RTX PRO 6000 | 63 GiB | ~32s | ~15 tok/s |
| **Llama-3.1-70B-AWQ** | 780M | 37 GiB | ~50s | ~0.62 tok/s |
| **Llama-3.1-70B-AWQ** | RTX PRO 6000 | 37 GiB | ~31s | ~30 tok/s |

## Model: GPT-OSS-120B (66GB MXFP4)

### 780M Performance
| Metric | Value |
|--------|-------|
| Weight Loading | ~82s |
| Total Init | ~96s |
| Load Speed | 0.84 GB/s (limited by MXFP4 dequantization) |
| Text Generation | ~1.2 tok/s |

### RTX PRO 6000 Performance
| Metric | Value |
|--------|-------|
| Weight Loading | ~10s |
| Total Init | ~24s |
| Text Generation | TBD |

## Model: Qwen3-VL-32B-Instruct (63GB BF16)

### 780M Performance
| Metric | Value |
|--------|-------|
| Weight Loading | 40s |
| Total Init | ~60s |
| Vision Processing | ~18s |
| Text Generation | ~0.86 tok/s |

### RTX PRO 6000 Performance
| Metric | Value |
|--------|-------|
| Weight Loading | 15-22s |
| Total Init | ~30s |
| Vision Processing | ~5.5s (with TORCH_SDPA) |
| Text Generation | ~15-21 tok/s |

## Model: Llama-3.1-70B-Instruct-AWQ-INT4 (~37GB)

### 780M Performance
| Metric | Value |
|--------|-------|
| Weight Loading | 17-40s |
| Total Init | ~50s |
| Text Generation | ~0.62 tok/s |

### RTX PRO 6000 Performance
| Metric | Value |
|--------|-------|
| Weight Loading | 6s |
| Total Init | ~31s |
| Text Generation | ~30 tok/s |

## Model Switching

### Current Approach: Process Isolation
Due to GPU memory fragmentation on ROCm, in-process model switching causes HIP kernel errors.
The reliable approach is to run each model in a separate subprocess.

**Switch time = model load time** (~50-60s on 780M, ~30s on RTX PRO 6000)

### Future Optimization Paths
1. **Weight offloading to RAM**: Keep weights in unified memory, swap to GPU on demand
2. **MMAP loading**: Memory-map weight files for faster loading
3. **Weight caching**: Pre-load weights and swap pointers
4. **Blitz patch**: Patched vLLM loader achieved 9.7 GB/s in earlier tests

## Optimization Analysis

### Loading Time Breakdown (780M, GPT-OSS-120B)

| Stage | Time | % of Total | Notes |
|-------|------|------------|-------|
| vLLM Import | 5s | 5% | Can be pre-loaded with daemon approach |
| Config/Tokenizer | 4s | 4% | Minimal optimization potential |
| Weight Loading | 82s | 85% | **Hardware/format limited** |
| KV Cache Init | 2s | 2% | Already optimized |
| First Inference | 4s | 4% | One-time cost |

### Why Weight Loading is Slow

On 780M with DDR5 (~100 GB/s theoretical), we see only 0.84 GB/s effective loading:

1. **MXFP4 dequantization**: GPT-OSS-120B uses MXFP4 quantization requiring CPU dequantization
   - Qwen (FP16) loads at 1.1 GB/s - 30% faster than MXFP4

2. **Safetensor parsing**: Reading safetensor format has overhead

3. **Tensor creation**: Python/PyTorch tensor allocation overhead

### Optimization Strategies Attempted

| Strategy | Speedup | Notes |
|----------|---------|-------|
| Page cache (2nd run) | ~0% | No improvement - bottleneck is CPU |
| Multi-threaded loading | Failed | GPU memory not freed between runs |
| Daemon (pre-warmed imports) | 5s saved | ~5% improvement |
| Subprocess isolation | Works | Reliable but no speed improvement |

### Theoretical vs Actual Limits

| System | Theoretical | Actual | Bottleneck |
|--------|-------------|--------|------------|
| 780M (DDR5) | 100 GB/s | 0.84 GB/s | CPU processing |
| RTX PRO 6000 (PCIe 5.0) | 64 GB/s | TBD | TBD |

## Known Issues

1. **FlashAttention on Blackwell**: Vision encoder is extremely slow (~100s vs ~5s) when using FlashAttention on SM_120.
   - Fix: Set `VLLM_ATTENTION_BACKEND=TORCH_SDPA`

2. **HIP Kernel Errors on ROCm gfx1103**: Profile run causes HIP launch failures on rotary embedding.
   - Fix: Set `VLLM_SKIP_WARMUP=1` and `kv_cache_memory_bytes` explicitly

3. **In-process model switching on ROCm/CUDA**: GPU memory isn't fully freed after `del llm`, causing errors on next model load.
   - Fix: Use subprocess isolation for model switching
   - vLLM's `shutdown()` only cleans profiler, not model weights

5. **Cross-architecture switching on 780M (ROCm gfx1103)**: GPU state gets corrupted when switching between MXFP4 (GPT-OSS) and FP16 (Qwen) models.
   - GPT-OSS-120B alone: Works (4/4 switches stable)
   - Qwen3-VL-32B alone: Works (tested)
   - GPT→Qwen: **Fails** (Qwen hangs after GPT runs)
   - Qwen→GPT: Not tested
   - Root cause: ROCm driver leaves state that corrupts subsequent model architectures
   - No fix available - requires reboot between different model architectures

4. **MXFP4 on ROCm**: GPT-OSS-120B MXFP4 works via Triton backend with `dtype='bfloat16'`
   - Previously thought unsupported, now confirmed working

## vLLM Patches Applied
- `vllm/v1/worker/gpu_model_runner.py` - Added VLLM_SKIP_WARMUP check to skip encoder profiling (line ~4957)
