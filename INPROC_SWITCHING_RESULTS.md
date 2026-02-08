# In-Process Model Switching Results

## Summary

Successfully implemented proper GPU memory cleanup for vLLM V1, enabling in-process model switching without subprocess isolation.

## Key Results (RTX PRO 6000, 102 GiB VRAM)

| Metric | Value | Notes |
|--------|-------|-------|
| Model 1 load (GPT-OSS-120B MXFP4) | 18.0s | 9.2s for weight loading |
| Model 1 unload | **0.21s** | Very fast! |
| Model 2 load (Qwen3-VL-32B FP16) | 36.0s | 19.7s for weight loading |
| Total switch time | 36.2s | Dominated by model loading |
| Memory freed by unload | 71.3 GiB | Model weights + buffers |

## Bottleneck Analysis

The unload is **fast** (210ms). The switch time is dominated by:

1. **Weight loading** (62%): 19.7s for Qwen weights (62.4 GiB at 3.2 GB/s)
2. **vLLM initialization** (38%): ~14s for tokenizer, engine setup, KV cache

Weight loading performance:
- GPT-OSS-120B: 65.97 GiB in 9.2s = **7.2 GB/s**
- Qwen3-VL-32B: 62.43 GiB in 19.7s = **3.2 GB/s** (slower due to FP16 vs MXFP4)

## Implementation

The `unload_vllm_model()` function in `blitz_inproc_switch.py` properly cleans up:

1. Model weights (clear parameter data)
2. Model buffers
3. KV cache tensors
4. Static forward context (attention layer state)
5. Input/output buffers
6. Distributed process groups (NCCL)
7. CUDA memory cache

**Timing breakdown:**
```
clear_params:     2.0ms
del_model:        0.0ms
clear_kv_cache:   0.0ms
clear_static_ctx: 0.0ms
clear_buffers:    0.0ms
gc:               0.4ms
destroy_pg:      42.1ms   <- NCCL cleanup
gc2:              0.4ms
cuda_empty:     161.3ms   <- CUDA cache flush
-----------------------
Total:          206ms
```

## How to Use

```python
from blitz_inproc_switch import unload_vllm_model
from vllm import LLM, SamplingParams

# Load model 1
llm = LLM(model="model1", ...)
output = llm.generate(["prompt"], SamplingParams(...))

# Switch to model 2
unload_vllm_model(llm)
del llm

# Now model 2 can use the freed memory
llm2 = LLM(model="model2", ...)
```

## Requirements

- `VLLM_ENABLE_V1_MULTIPROCESSING=0` (single-process mode)
- vLLM V1 engine (tested with v0.14.0rc2)

## Next Steps for Faster Switching

1. **Optimize weight loading**: Current bottleneck is 3-7 GB/s. Hardware can do ~45 GB/s with pinned memory, but safetensors mmap overhead prevents this.

2. **Keep model skeleton**: For same-architecture switches, keep the model structure and only reload weights.

3. **Prefetch weights**: Start loading next model weights while current model is still serving.

4. **RAM cache**: Keep frequently-used model weights in system RAM for faster reloading.

## Comparison with Previous Approaches

| Approach | Switch Time | Clean? | Notes |
|----------|-------------|--------|-------|
| Subprocess (old) | ~45s | Yes | Slow due to process overhead |
| In-process (new) | ~36s | Yes | Fast unload, loading is bottleneck |
| No cleanup | N/A | No | OOM on second model |
