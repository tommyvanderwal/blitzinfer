# AWQ Marlin Crash Investigation Log

## Date: 2026-01-29

## Summary

The `awq_marlin` quantization with `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` causes **intermittent hard system freezes** requiring power cycle on RTX PRO 6000 Blackwell GPU.

## Test Results

### Marathon Test (25 runs, 4-model switching)

**Test sequence per run:**
1. openai/gpt-oss-120b (mxfp4) - ALWAYS works
2. Qwen/Qwen3-VL-32B-Thinking-FP8 (fp8) - ALWAYS works
3. openai/gpt-oss-120b (mxfp4) - ALWAYS works
4. hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 (awq_marlin) - **CRASHES ~52%**

| Metric | Value |
|--------|-------|
| Total runs | 25 |
| Passes | 12 (48%) |
| Crashes | 13 (52%) |
| Avg pass time | 107.2s |
| Avg crash time | 178.2s |

**Pattern:** `CCC P CCC PPP C PP CC P CC P C PP C PP`

### AWQ-First Test (AWQ Marlin as only/first model)

- Test 1: PASS
- Test 2: CRASH

**Conclusion:** Crash is NOT dependent on loading other models first. AWQ Marlin has inherent instability.

### Exact Crash Point (100% consistent)

```
INFO [gpu_model_runner.py:4118] Model loading took 37.09 GiB memory and 14.2 seconds
INFO [gpu_worker.py:356] Available KV cache memory: 38.71 GiB
INFO [kv_cache_utils.py:1307] GPU KV cache size: 126,832 tokens
INFO [core.py:272] init engine (profile, create kv cache, warmup model) took 7.65 seconds
<< SYSTEM FREEZE - NO FURTHER OUTPUT >>
```

The crash ALWAYS occurs:
- After model loads successfully
- After KV cache profiling completes
- After engine warmup completes
- **BEFORE first inference can start**

## Hardware

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX PRO 6000 Blackwell (SM120, 95GB VRAM) |
| Driver | 570.195.03 |
| CUDA | 12.8 |
| vLLM | 0.14.0rc2.dev348+gdcd80206b |

## Related vLLM GitHub Issues

### Blackwell/SM120 Specific
- [#19166](https://github.com/vllm-project/vllm/issues/19166) - CUDA Illegal Access Error on 2x Blackwell
- [#21336](https://github.com/vllm-project/vllm/issues/21336) - vLLM crashes with Blackwell PRO 6000
- [#20522](https://github.com/vllm-project/vllm/issues/20522) - "no kernel image available" on RTX Pro 6000
- [#23497](https://github.com/vllm-project/vllm/issues/23497) - SM120 not recognized in MXFP4 backend
- [#26211](https://github.com/vllm-project/vllm/issues/26211) - DeepSeek not supported on SM120
- [#31085](https://github.com/vllm-project/vllm/issues/31085) - Feature request for SM120 NVFP4 MoE kernels

### AWQ Marlin Specific
- [#21339](https://github.com/vllm-project/vllm/issues/21339) - AWQ INT4 hard crashes on RTX 3090s (requires reboot)
- [#6985](https://github.com/vllm-project/vllm/issues/6985) - AWQ Marlin conflicts with vLLM
- [#3392](https://github.com/vllm-project/vllm/issues/3392) - AWQ + Marlin Error
- [#7297](https://github.com/vllm-project/vllm/issues/7297) - vLLM hangs with Meta-Llama-3.1-70B-Instruct-AWQ-INT4
- [#13119](https://github.com/vllm-project/vllm/pull/13119) - Bugfix for AWQ Marlin fallback

### Most Relevant Issue
**[#21339](https://github.com/vllm-project/vllm/issues/21339)** describes similar hard crashes with AWQ INT4:
> "HARD CRASHES, and nvidia-smi shows no cards until I reboot"

However, this is on RTX 3090s and the system stays online (only GPU becomes unresponsive).
Our issue is worse - complete system freeze requiring power cycle.

## Root Cause Analysis

The crash appears to be caused by:
1. **SM120 (Blackwell) incomplete support** in vLLM
2. **AWQ Marlin kernel** triggering hardware-level fault
3. **Timing-sensitive** - ~52% crash rate suggests race condition or thermal

## Workarounds

1. Use `quantization='awq'` instead of `quantization='awq_marlin'` (slower but stable)
2. Avoid this model entirely on Blackwell GPUs
3. Use different quantization (fp8, mxfp4 work fine)

## Files

- Marathon results: `~/blitz_marathon_logs/marathon_results.csv`
- Marathon live log: `~/blitz_marathon_logs/marathon_live.log`
- Individual run logs: `~/blitz_marathon_logs/run_*.log`
- Test script: `~/blitz_marathon_logs/marathon_4switch.py`

## Conclusion

**This model/quantization combo should be avoided on Blackwell GPUs until vLLM adds proper SM120 support.**

Other Marlin variants (mxfp4) work fine. The issue is specific to AWQ Marlin on SM120.
