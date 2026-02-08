# BlitzInfer Cutting-Edge Stack Upgrade Plan

## Target Versions (February 2026)

| Component | Current | Target | Status |
|-----------|---------|--------|--------|
| **Python** | 3.12.12 | **3.13.x** | Upgrade |
| **PyTorch** | 2.9.1+cu128 | **2.10.0+cu129** | Upgrade |
| **CUDA Runtime** | 12.8 | **12.9** | Upgrade |
| **SGLang** | 0.5.8 | 0.5.8 | Keep (latest) |
| **flashinfer** | 0.6.1 | **0.6.2** | Upgrade |
| **sgl-kernel** | 0.3.21 | 0.3.21 | Keep (latest) |
| **triton** | 3.5.1 | **Latest** | Check |

## Why Upgrade?

### Python 3.13 Benefits
- **Free-threading (experimental)**: Disable GIL for true parallelism
- **JIT compiler**: Up to 5% general performance gains
- **Better error messages**: Improved debugging
- **Faster startup**: Reduced import times

### CUDA 12.9 Benefits for Blackwell SM120
- **FP32 emulation with BF16x9**: Up to 3x speedup on compute-bound FP32 matmul
- **CUTLASS SM120 optimizations**: Better blockwise dense gemm
- **NVFP4 support**: Next-gen quantization format

### PyTorch 2.10 Benefits
- Full Python 3.13 support
- Improved torch.compile for SM120
- Better CUDA 12.9 integration

## Upgrade Steps

### Phase 1: Create Fresh Python 3.13 Environment

```bash
# On remote (tommy@192.168.2.90)
cd ~/pythonprojects/blitzinfer

# Install Python 3.13 via deadsnakes PPA (Ubuntu)
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev -y

# Create new venv with Python 3.13
python3.13 -m venv venv313
source venv313/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Phase 2: Install Core Stack (CUDA 12.9)

```bash
# PyTorch 2.10 with CUDA 12.9
pip install torch==2.10.0+cu129 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# Verify
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Phase 3: Install SGLang + Dependencies

```bash
# SGLang with all dependencies
pip install sglang[all]

# flashinfer (CUDA 12.9 compatible)
pip install flashinfer-python==0.6.2

# Note: flashinfer wheels may need specific CUDA version
# If issues, install from source or use FLASHINFER_DISABLE_VERSION_CHECK=1
```

### Phase 4: Install BlitzInfer Dependencies

```bash
pip install fastapi uvicorn httpx pillow pydantic
pip install huggingface_hub transformers safetensors
```

### Phase 5: Verify Installation

```bash
python -c "
import torch
import sglang
import flashinfer

print('Python:', __import__('sys').version)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('SGLang:', sglang.__version__)
print('FlashInfer:', flashinfer.__version__)
print('GPU:', torch.cuda.get_device_name(0))
print('SM:', torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor)
"
```

## Potential Issues & Solutions

### Issue 1: flashinfer CUDA version mismatch
**Solution**: Set `FLASHINFER_DISABLE_VERSION_CHECK=1` or build from source

### Issue 2: sgl-kernel requires specific torch version
**Solution**: May need to install sgl-kernel from nightly/source

### Issue 3: Python 3.13 package compatibility
**Solution**: Some packages may need updates; fall back to 3.12 if critical failures

### Issue 4: Triton SM120 issues
**Solution**: Use `attention_backend=triton` and `disable_cuda_graph=True`

## Rollback Plan

Keep old venv intact:
```bash
# Old venv preserved at ~/pythonprojects/blitzinfer/venv
# Switch back if needed:
source venv/bin/activate
```

## Post-Upgrade Testing

1. Run basic model load test
2. Run edge case tests: `python tests/test_all_models_edge_cases.py --quick`
3. Run full test suite: `python tests/test_all_models.py`
4. Monitor memory drift over 5 switches
5. Test all model switching pairs

## References

- [PyTorch cu129 wheels](https://download.pytorch.org/whl/cu129)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [CUDA 12.9 Release Notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/)
- [Python 3.13 What's New](https://docs.python.org/3.13/whatsnew/3.13.html)
