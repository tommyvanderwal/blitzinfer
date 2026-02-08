# BlitzInfer NVIDIA Setup Notes

## Remote System: 192.168.2.90

### Hardware
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- **VRAM**: ~98GB (95GB usable)
- **RAM**: 128GB DDR5
- **Storage**: NVMe PCIe 5.0 (3.6TB, 479GB free)
- **OS**: Ubuntu 24.04 (Linux 6.14.0)

### Software Stack
- **NVIDIA Driver**: 570.195.03
- **CUDA**: 12.8 (multiple versions: 12.6, 12.8 in /usr/local)
- **Python**: 3.12.3
- **PyTorch**: 2.9.1+cu128
- **vLLM**: 0.14.1
- **triton**: 3.5.1

### Setup Steps Performed

1. **Created project folder**:
   ```bash
   mkdir -p ~/pythonprojects/blitzinfer
   ```

2. **Transferred BlitzInfer code** (from local AMD machine):
   ```bash
   rsync -avz --exclude '__pycache__' --exclude 'venv' \
       /home/tommy/pythonprojects/blitzinfer/ \
       tommy@192.168.2.90:~/pythonprojects/blitzinfer/
   ```

3. **Created virtual environment**:
   ```bash
   cd ~/pythonprojects/blitzinfer
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip wheel setuptools
   ```

4. **Installed PyTorch with CUDA 12.8**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

5. **Installed vLLM**:
   ```bash
   pip install vllm
   ```
   Note: vLLM 0.14.1 pins torch 2.9.1 and triton 3.5.1

### Model Downloads

Started parallel downloads on remote system:

```bash
# GPT-OSS-120B (~60GB MXFP4 quantized)
huggingface-cli download openai/gpt-oss-120b \
    --include '*.safetensors' '*.json' '*.txt' '*.model' '*.tiktoken' \
    > /tmp/gpt_download.log 2>&1 &

# GLM-4.6V-AWQ-4bit (~65GB AWQ quantized)
huggingface-cli download cyankiwi/GLM-4.6V-AWQ-4bit \
    --include '*.safetensors' '*.json' '*.txt' '*.model' \
    > /tmp/glm_download.log 2>&1 &
```

Monitor with:
```bash
# Check processes
pgrep -f huggingface-cli

# Check progress
tail -f /tmp/gpt_download.log
tail -f /tmp/glm_download.log

# Check disk usage
du -sh ~/.cache/huggingface/hub/models--openai--gpt-oss-120b
du -sh ~/.cache/huggingface/hub/models--cyankiwi--GLM-4.6V-AWQ-4bit
```

### Key Differences from AMD/ROCm Setup

| Feature | AMD (780M iGPU) | NVIDIA (RTX PRO 6000) |
|---------|-----------------|----------------------|
| VRAM | ~96GB (shared DDR5) | 95GB (dedicated GDDR7) |
| Memory BW | ~89 GB/s | ~1024+ GB/s |
| CUDA graphs | Not supported | Supported |
| `enforce_eager` | Required (True) | Optional (False) |
| `compilation_config` | `{"custom_ops": ["none"]}` | Not needed |
| Expected load time | ~17-20s (7B) | Much faster |
| Expected tok/s | ~4 tok/s (7B) | ~100+ tok/s |

### Running Tests

```bash
cd ~/pythonprojects/blitzinfer
source venv/bin/activate

# Run large model swap test
python test_nvidia_swap.py 10
```

### Expected Performance

With RTX PRO 6000's ~1TB/s memory bandwidth vs 780M's ~89 GB/s:
- Model loading should be ~10x faster
- Inference throughput should be ~20-50x higher
- Swap cycles should be significantly quicker

### Troubleshooting

1. **Out of memory**: Reduce `gpu_memory_utilization` from 0.90
2. **CUDA errors**: Check `nvidia-smi` for other processes using GPU
3. **Model not found**: Verify HF cache with `ls ~/.cache/huggingface/hub/`
