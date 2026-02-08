# Lessons Learned - BlitzInfer Development

## ROCm 7.2 + Radeon 780M (gfx1100) Setup

### System Configuration
- **APU**: Ryzen 7840HS with Radeon 780M iGPU
- **ROCm Version**: 7.2.0
- **Architecture**: gfx1100 (RDNA 3 / Navi31)
- **RAM**: 109GB DDR5
- **Unified Memory**: iGPU shares system RAM - no PCIe transfers needed

### Critical Setup for vLLM on ROCm 7.2

#### 1. Use AMD's ROCm 7.2 Specific Wheels
**DO NOT** use generic PyTorch ROCm wheels or vLLM prebuilt wheels - they are incompatible with ROCm 7.2.

```bash
# AMD's official ROCm 7.2 wheels repository
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.git7e1940d4-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-manylinux_2_28_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl

uv pip install ./torch-*.whl ./triton-*.whl ./torchvision-*.whl
```

#### 2. Install amdsmi from ROCm
vLLM's ROCm platform detection requires amdsmi:
```bash
cp -r /opt/rocm/share/amd_smi/ ./amd_smi_build
uv pip install ./amd_smi_build/
```

#### 3. Build vLLM from Source
```bash
git clone --depth 1 https://github.com/vllm-project/vllm.git
cd vllm

uv pip install -r requirements/rocm.txt

export PYTORCH_ROCM_ARCH="gfx1100"
export ROCM_HOME=/opt/rocm
export CMAKE_PREFIX_PATH="/opt/rocm:$CMAKE_PREFIX_PATH"

python setup.py develop
```

#### 4. NumPy Compatibility
```bash
# numba requires numpy < 2.3
uv pip install "numpy<2.3"
```

### Working vLLM Configuration for iGPU

Key settings that make vLLM work on 780M iGPU:

```python
llm = LLM(
    model='Qwen/Qwen2.5-7B-Instruct',
    gpu_memory_utilization=0.20,     # Conservative for iGPU shared memory
    max_model_len=32768,
    max_num_seqs=20,
    max_num_batched_tokens=512,      # Small batches for stability
    enforce_eager=True,              # Required - no CUDA graphs on gfx1100
)
```

**CRITICAL**:
- `gpu_memory_utilization=0.20` - Higher values (0.85) cause kernel launch failures
- `max_num_batched_tokens=512` - Large batches cause instability
- `enforce_eager=True` - CUDA graphs don't work properly on gfx1100

### Working Package Versions
```
torch: 2.9.1+rocm7.2.0.git7e1940d4
triton: 3.5.1+rocm7.2.0.gita272dfa8
torchvision: 0.24.0+rocm7.2.0.gitb919bd0c
vllm: built from source (commit a698e8e or later)
numpy: 2.2.6 (< 2.3 for numba compatibility)
amdsmi: 26.2.1+fc0010cf6a (from /opt/rocm/share/amd_smi/)
```

### Performance Observations
- Model load: ~5.5s for 7B model weights
- First token latency: Higher due to unified memory
- Throughput: ~3.5 tok/s output (limited by iGPU)
- Memory: 14.34 GiB for Qwen2.5-7B in bfloat16

### Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `hipErrorLaunchFailure` | Memory/batch too large | Use `gpu_memory_utilization=0.20`, `max_num_batched_tokens=512` |
| `UnspecifiedPlatform` | amdsmi not installed | Install from `/opt/rocm/share/amd_smi/` |
| `Numba needs NumPy 2.2 or less` | numpy too new | `uv pip install "numpy<2.3"` |
| `No module named 'amdsmi'` | pip installed to wrong location | Use `uv pip install` not regular pip |
| `hipblas-common not found` | CMAKE_PREFIX_PATH missing | `export CMAKE_PREFIX_PATH="/opt/rocm:$CMAKE_PREFIX_PATH"` |

### Environment Variables
```bash
export HIP_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_ROCM_ARCH=gfx1100
export ROCM_HOME=/opt/rocm
export CMAKE_PREFIX_PATH="/opt/rocm:$CMAKE_PREFIX_PATH"
```

### Model Switching Performance (iGPU)

Successfully tested model switching between Qwen-7B and Mistral-7B on Radeon 780M:

| Operation | Time |
|-----------|------|
| Model unload (including process cleanup) | ~3s |
| Model load (from disk cache) | ~16-19s |
| Total switch time | ~19s |

**Key requirements for switching to work:**
- Must properly clean up child processes from vLLM V1 engine
- Need explicit `multiprocessing.active_children()` cleanup
- 1-second delay after cleanup ensures GPU memory is fully released
- Without proper cleanup, loading second model causes HIP kernel failures

### Target Models for BlitzInfer
The following models need to work with fast switching:
1. `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
2. `openai/gpt-oss-120b`
3. `Qwen/Qwen3-VL-32B-Thinking-FP8`
