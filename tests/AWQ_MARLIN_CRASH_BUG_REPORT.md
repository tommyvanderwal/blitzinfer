# Bug Report: AWQ Marlin Kernel Causes Intermittent System Crash on RTX PRO 6000 Blackwell

## Summary

The vLLM `awq_marlin` quantization kernel causes **intermittent hard system freezes** (requiring power cycle) on NVIDIA RTX PRO 6000 Blackwell Workstation Edition. The standard `awq` kernel works correctly on the same hardware.

## Severity

**High** - Intermittent complete system freeze requiring physical power cycle. No kernel panic, no logs, no recovery possible without power cycling.

## Reproducibility Update (2026-01-29)

**IMPORTANT**: Extensive marathon testing (18+ runs) reveals a **61% crash rate** with consistent crash point.

### Marathon Test Results (4-model switching, no manual reboot):

**Test sequence per run:**
1. Load gpt-oss-120b (mxfp4) → inference → cleanup
2. Load Qwen3-VL-32B-FP8 → inference → cleanup
3. Load gpt-oss-120b (mxfp4) → inference → cleanup
4. Load Meta-Llama-3.1-70B-AWQ-INT4 (**awq_marlin**) → **CRASH HERE**

**Statistics (18 runs):**
| Metric | Value |
|--------|-------|
| Total runs | 18 |
| Passes | 7 (38.9%) |
| Crashes | 11 (61.1%) |
| Avg pass time | 104.4s |
| Avg crash time | 179.6s |

**Pattern:** `CCC P CCC PPP C PP CC P CC` (C=crash, P=pass)

### Exact Crash Point (100% consistent when crash occurs):

```
INFO [gpu_model_runner.py:4118] Model loading took 37.09 GiB memory and 14.2 seconds
INFO [gpu_worker.py:356] Available KV cache memory: 38.71 GiB
INFO [kv_cache_utils.py:1307] GPU KV cache size: 126,832 tokens
INFO [core.py:272] init engine (profile, create kv cache, warmup model) took 7.65 seconds
<< SYSTEM FREEZE - NO FURTHER OUTPUT >>
```

**Critical observation**: The crash ALWAYS occurs:
- After AWQ Marlin model loads successfully
- After KV cache profiling completes successfully
- After engine warmup completes successfully
- **BEFORE first inference can start**

### Suspected Pattern:
The crash appears to be triggered by:
1. **AWQ Marlin kernel warmup completion** - crash occurs right after `init engine ... took 7.6s`
2. **Intermittent hardware/driver interaction** - ~61% crash rate suggests timing-sensitive issue
3. **Multiple reboots may help** - passes tend to occur after 2-3 consecutive crashes/reboots
4. **Not accumulated state** - crashes occur even on first run after fresh boot

## Environment

| Component | Version/Details |
|-----------|----------------|
| **GPU** | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| **GPU PCI ID** | 0x2BB110DE |
| **VBIOS** | 98.02.81.00.07 |
| **Driver** | 570.195.03 (Open Kernel Module) |
| **CUDA** | 12.8 |
| **PyTorch** | 2.9.1+cu128 |
| **vLLM** | 0.14.0rc2.dev348+gdcd80206b |
| **Linux Kernel** | 6.14.0-37-generic |
| **OS** | Ubuntu 24.04.3 LTS |
| **CPU** | AMD Ryzen 7 7800X3D |
| **RAM** | 124 GB |
| **Motherboard** | ASUS (model TBD) |

## Reproducer

### Minimal Reproduction (CRASHES)

```python
import os
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

from vllm import LLM, SamplingParams

llm = LLM(
    model='hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4',
    max_model_len=4096,
    quantization='awq_marlin',  # THIS CAUSES CRASH
    enforce_eager=True,
)

# Crash occurs during or shortly after model warmup
# System becomes completely unresponsive
```

### Working Alternative

```python
import os
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

from vllm import LLM, SamplingParams

llm = LLM(
    model='hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4',
    max_model_len=4096,
    quantization='awq',  # Standard AWQ - WORKS
    enforce_eager=True,
)

out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
print(out[0].outputs[0].text)  # Works correctly
```

## Behavior

### Crash Timeline (precise from marathon testing)

1. Model weights load successfully (14.2 seconds, 37.09 GiB)
2. KV cache profiling completes (38.71 GiB available, 126,832 tokens)
3. Engine initialization completes ("init engine ... took 7.65 seconds")
4. **CRASH** - System freezes **immediately** after warmup, before inference
5. No logs, no kernel panic, no dmesg output
6. System requires physical power cycle (30s off recommended for PSU drain)

### Comparison Matrix

| Quantization | Kernel Backend | Model | Result |
|-------------|---------------|-------|--------|
| `awq_marlin` | Marlin AWQ | llama-3.1-70B-AWQ-INT4 | ❌ **CRASH** |
| `awq` | Standard GEMM | llama-3.1-70B-AWQ-INT4 | ✅ Works |
| `mxfp4` | Marlin FP4 | gpt-oss-120b | ✅ Works |

## Investigation Results

### Ruled Out

- **Prior model usage**: Crash occurs even when llama-AWQ is first model loaded on cold boot
- **Triton cache corruption**: Crash occurs with fresh caches cleared
- **Memory issues**: 124 GB RAM, 95 GB VRAM - plenty of headroom
- **Other Marlin variants**: mxfp4 Marlin works fine with same setup

### Identified

- **Specific to AWQ Marlin**: Only `awq_marlin` crashes, `awq` standard works
- **Specific to Blackwell**: May be related to new Blackwell architecture
- **Timing**: Crash occurs during model warmup phase, not loading

## Workaround

Use `quantization='awq'` instead of `quantization='awq_marlin'`:

```python
llm = LLM(
    model='hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4',
    quantization='awq',  # Use standard AWQ, not Marlin
    ...
)
```

**Trade-off**: Standard AWQ is slower than Marlin (~10x slower inference) but doesn't crash.

## Possible Root Causes

1. **AWQ Marlin kernel incompatibility with Blackwell SM architecture**
   - Marlin kernels are optimized for specific GPU architectures
   - Blackwell may have different warp scheduling or memory access patterns

2. **CUDA driver bug with specific kernel patterns**
   - The AWQ Marlin kernel may trigger a driver bug
   - Open kernel module may behave differently than proprietary

3. **Hardware-level issue triggered by specific compute patterns**
   - Certain instruction sequences may cause GPU hangs that propagate to system

## Requested Actions

1. **vLLM team**: Investigate AWQ Marlin kernel compatibility with Blackwell GPUs
2. **NVIDIA**: Check if driver 570.195.03 has known issues with Marlin-style kernels on Blackwell

## Additional Notes

- The crash has **61% probability** - tested over 18+ runs with automated power cycling
- Crash point is **100% consistent** when it occurs: after AWQ Marlin engine warmup, before inference
- Models 1-3 in the test sequence (gpt-oss-120b mxfp4, Qwen3-VL-32B fp8) work perfectly
- Only Model 4 (AWQ Marlin) causes crashes
- Crash occurs ~170-180 seconds into test (consistent with Model 4 being the last)
- When test passes, it completes in ~104 seconds
- No error messages, no stack traces, no kernel logs before crash
- System appears to be in a GPU hang state that requires power cycle
- Other Marlin variants (mxfp4) work fine with same setup
- Multiple reboots may help - passes tend to cluster after 2-3 consecutive crashes
- 30-second power-off recommended to ensure full PSU drain before restart

### Marathon test logs preserved at:
- Results: `~/blitz_marathon_logs/marathon_results.csv`
- Live log: `~/blitz_marathon_logs/marathon_live.log`
- Individual run logs: `~/blitz_marathon_logs/run_*.log`

## Files

- Reproducer script: `test_awq_marlin_crash.py`
- Working alternative: `test_awq_standard.py`
- Hardware info collection: `collect_hw_info.sh`
