# Dynamic Multi-Model LLM Serving System

## Project Vision

Build a high-performance LLM serving orchestrator optimized for **fast model switching** with intelligent queue management and tiered memory caching. The system should seamlessly swap models in ~1 second (from DDR5 WARM tier) while maintaining high throughput for concurrent requests.

---

## vLLM vs SGLang Comparison

| Aspect | vLLM | SGLang | Winner for This Project |
|--------|------|--------|-------------------------|
| **Throughput** | Good, ~12-16k tok/s | Better, ~16k tok/s (29% faster in some tests) | SGLang |
| **Memory Usage** | Higher (21GB for 7B) | Lower (7GB for 7B) - more KV cache room | SGLang |
| **TTFT** | Faster first token | Slightly slower | vLLM |
| **KV Cache Reuse** | Manual prefix caching | RadixAttention (automatic) | SGLang |
| **Multi-Model Gateway** | No built-in | Model Gateway v0.3.0 | SGLang |
| **AMD/ROCm Support** | Good | Good (native support) | Tie |
| **Model Switching** | Not built-in | Not built-in | Neither - must implement |
| **Ecosystem** | Larger community | Growing rapidly | vLLM (slightly) |

**Recommendation**: Both require similar modifications. Decision requires hands-on testing on your target hardware.

### Framework Decision Criteria

| Criteria | vLLM | SGLang | How to Test |
|----------|------|--------|-------------|
| Memory overhead | Measure | Measure | Load same model, compare free VRAM |
| Startup time | Measure | Measure | Time from launch to first inference |
| Weight offload support | CuMemAllocator sleep mode | Unknown | Check if either supports GPU↔CPU weight movement |
| AMD ROCm maturity | Good | Good | Test on 7840HS iGPU |
| Extensibility | V1 scheduler hooks | Model Gateway | Review code, assess modification points |
| Startup caching | Partial (CUDA graphs) | RadixAttention state | Investigate what each caches |

### Key Experiments Before Deciding

1. **Load same 7B model on both** - Compare memory usage, startup time
2. **Test on AMD iGPU (ROCm)** - Which has better support for Radeon 780M?
3. **Explore model unloading** - Can either cleanly unload a model from GPU?
4. **Review source code** - Which has cleaner extension points for orchestrator?
5. **Community activity** - Check GitHub issues for multi-model discussions

**Sources**:
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang vs vLLM Comparison](https://www.gpu-mart.com/blog/sglang-vs-vllm)
- [vLLM vs SGLang KV Cache](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache)

---

## Target Hardware Configurations

### Configuration A: High-End Desktop
| Component | Spec | Role |
|-----------|------|------|
| CPU | AM5 Platform | System orchestration |
| RAM | 128GB DDR5 | WARM tier model cache (configurable budget) |
| GPU | RTX PRO 6000 Blackwell 96GB | HOT tier + inference |
| GPU Link | PCIe 5.0 x16 (~64 GB/s) | Fast DDR5→VRAM transfer |
| Storage | 4TB+ NVMe PCIe 5.0 x4 (~14 GB/s) | COLD tier model archive |

### Configuration B: Compact/Portable (iGPU)
| Component | Spec | Role |
|-----------|------|------|
| APU | Ryzen 7840HS (Radeon 780M) | Unified compute |
| RAM | 128GB DDR5 (~96GB iGPU / 32GB system) | Unified memory architecture |
| Storage | NVMe PCIe 4.0 x4 (~7 GB/s) | Model storage |

**iGPU Strategy**: Strict single-model loading. Zero-copy switching (no PCIe transfer). Maximize KV cache by keeping only one model in memory at a time.

---

## Design Decisions (Finalized)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Switch Policy** | Drain-then-switch | Complete all queued jobs (snapshot at switch decision time) |
| **Starvation** | No limit | New requests after switch decision go to "next model" queue |
| **WARM Tier Budget** | Configurable at startup | If 0, skip DDR5 and load direct SSD→GPU |
| **Concurrency** | Unlimited (KV threshold ~80%) | Let KV cache pressure be the natural limit |
| **API** | OpenAI-compatible | Standard endpoint, broad client support |
| **Model Source** | Local files only (initially) | Full control, predictable load times |
| **Startup Caching** | Cache all artifacts | CUDA graphs, kernels, memory profiles - 2nd boot = weight transfer only |

**Model Sizes**: All models downloaded from HF, pre-quantized to fit comfortably in 90GB VRAM. Example: GPT-OSS-120B is ~60-65GB.

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REQUEST GATEWAY (OpenAI API)                 │
│  POST /v1/chat/completions with model="gpt-oss-120b"            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PER-MODEL REQUEST QUEUES                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  GPT-120B    │  │   Qwen-80B   │  │  Llama-70B   │          │
│  │  Queued: 20  │  │  Active: 5   │  │  Queued: 3   │          │
│  │  State: WARM │  │  State: HOT  │  │  State: COLD │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  Switch Logic: When Qwen-80B queue drains →                     │
│                Snapshot queues → Switch to GPT-120B             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL ORCHESTRATOR                            │
│  State Machine: COLD → WARM → HOT → SERVING                     │
│                                                                 │
│  • Monitors queue depths for predictive warming                 │
│  • Coordinates graceful drain-then-switch                       │
│  • Manages startup cache artifacts                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 TIERED MEMORY MANAGER                           │
│                                                                 │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐            │
│  │    HOT     │◄──►│    WARM    │◄──►│    COLD    │            │
│  │   (VRAM)   │    │   (DDR5)   │    │   (NVMe)   │            │
│  │   96GB     │    │ Configurable│    │   4TB+     │            │
│  │ 1 model +  │    │  0-100GB   │    │ All models │            │
│  │ KV cache   │    │            │    │ + caches   │            │
│  └────────────┘    └────────────┘    └────────────┘            │
│       ▲                  ▲                  ▲                   │
│       │ ~1s              │ ~5s              │                   │
│       │ (64 GB/s)        │ (14 GB/s)        │                   │
│       │                  │                  │                   │
│  Models pre-quantized from HF, all fit in 90GB VRAM easily      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              vLLM/SGLang ENGINE (Long-Lived Process)            │
│                                                                 │
│  • NEVER kill/restart - sleep/wake weights within same process  │
│  • One model active → maximize KV cache size                    │
│  • KV threshold (~80%) for natural backpressure                 │
│  • CUDA graphs captured once per model, persisted in memory     │
│  • Minimal core modifications for merge compatibility           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Long-Lived Single Engine Process (CRITICAL)
- **Never kill/restart the engine** - saves 10-30+ seconds per switch
- Sleep/wake model weights within same process
- Internal scheduler integration for queue snapshot logic
- One process owns GPU memory for entire session lifetime

### 2. Fast Model Switching (~1 second target)
- Pinned host memory (DDR5) + async multi-stream DMA transfers
- Target sustained ~52-58+ GB/s on PCIe 5.0 x16 (within 10% of hardware max)
- Use `torch.cuda.Stream` + `non_blocking=True` + custom allocator hooks
- 65GB model @ 55 GB/s ≈ 1.2s practical

### 3. One-Time CUDA Graph Warm-Up Per Model
- On first load: force small dummy batch → full CUDA graph capture
- Use piecewise/hybrid graph capture (not enforce-eager → avoid 15-30% decode penalty)
- Persist graphs in engine memory - never pay capture cost again
- Subsequent switches = pure weight transfer, graphs already cached

### 4. DDR5 WARM Pool Management
- Allocate ~70% of free RAM as pinned pool
- Use `mmap` + `mlock` or torch pinned tensors for zero-copy-ready weights
- LRU eviction when pool full
- Predictive prefetch based on queue depth

### 5. Intelligent Queue Management
- Per-model request queues (FCFS)
- "Snapshot" queue at switch decision time - new arrivals go to next model
- No starvation limits - drain completely before switching
- **One model active at a time** - maximize KV cache size

### 6. Predictive Loading
- Monitor queue depths across all models
- Start COLD→WARM transfer when next model has requests and current is draining
- Queue depth threshold triggers pre-warming

### 7. Merge-Compatible Design
- Minimize modifications to core vLLM/SGLang
- Use hooks/callbacks where possible
- Orchestrator layer wraps engine cleanly
- Goal: Easy rebase onto new vLLM/SGLang releases

---

## Transfer Time Estimates

Models are pre-quantized from HF to fit in 90GB VRAM. Transfer times based on actual model sizes:

| Model | Size on Disk | WARM→HOT (PCIe5 x16) | COLD→WARM (PCIe5 x4) |
|-------|-------------|----------------------|----------------------|
| Small (7B class) | ~15GB | 0.25s | 1.1s |
| Medium (70B class) | ~40GB | 0.65s | 2.9s |
| Large (GPT-OSS-120B) | ~65GB | 1.0s | 4.6s |

**All models fit comfortably in 96GB VRAM** with room for KV cache.

---

## iGPU-Specific Considerations (7840HS)

- **Strict Single-Model Loading**: One model active at a time - avoids bandwidth contention
- **Unified Memory**: No PCIe transfer cost on switch - models already in shared DDR5
- **Max iGPU VRAM**: Allocate ~96GB to iGPU, 32GB for OS/system
- **Bandwidth Limit**: ~90 GB/s shared DDR5 - one active model respects this
- **ROCm Support**: Both vLLM and SGLang viable - test startup/transfer times
- **Cold Load**: PCIe 4.0 x4 NVMe (~7 GB/s) is the bottleneck for SSD→RAM transfers

---

## Implementation Approach

### Phase 0: Framework Research (Before Committing)

**Goal**: Make informed vLLM vs SGLang decision based on your hardware.

| Experiment | What to Measure | Tools |
|------------|-----------------|-------|
| Memory overhead | VRAM used for 7B model | `nvidia-smi`, `rocm-smi` |
| Startup time | Cold boot to first inference | `time`, Python profiler |
| ROCm/iGPU support | Does it run on 7840HS? | Direct testing |
| Weight offload | GPU→CPU→GPU round-trip | Custom test script |
| Code review | Extension points, complexity | Manual code reading |

**Decision Framework**:
- If SGLang works well on ROCm + has lower memory → Use SGLang
- If vLLM has better offload support + cleaner hooks → Use vLLM
- If both work equally → SGLang (Model Gateway head start)

### Phase 1: Research & Prototype
1. Run Phase 0 experiments to choose framework
2. Prototype model weight offloading (GPU↔DDR5)
3. Measure actual transfer times on target hardware
4. Validate startup caching possibilities

### Phase 2: Core Orchestrator
1. Implement per-model request queues
2. Build model state machine (COLD/WARM/HOT/SERVING)
3. Implement drain-then-switch logic with queue snapshots
4. Add predictive warming based on queue depth

### Phase 3: Memory Manager
1. Implement tiered cache (VRAM/DDR5/NVMe)
2. Add configurable DDR5 budget
3. Implement LRU eviction for WARM tier
4. Async transfer handlers

### Phase 4: Startup Optimization
1. Identify all cacheable startup artifacts
2. Implement disk caching layer
3. Benchmark 1st vs 2nd boot times
4. Target: 2nd boot = SSD read time only

### Phase 5: Integration & Testing
1. OpenAI-compatible API gateway
2. Multi-model stress testing
3. Switch latency benchmarks
4. Memory pressure testing

---

## Open Questions for Future Consideration

1. **Multi-GPU**: How to handle tensor parallelism for 120B+ models?
2. **LoRA Adapters**: Can we keep base model HOT and swap only LoRA weights?
3. **Speculative Decoding**: Small draft model + large target model - switch implications?
4. **KV Cache Persistence**: For same-model hot-reload, persist KV to DDR5?
5. **Distributed**: Scale to multiple machines with shared model cache?

---

## File Structure (Proposed)

```
dynamic-llm-server/
├── orchestrator/
│   ├── model_state.py        # COLD/WARM/HOT state machine
│   ├── request_queue.py      # Per-model queues with snapshot
│   ├── switch_controller.py  # Drain-then-switch logic
│   └── predictor.py          # Predictive warming
├── memory/
│   ├── tier_manager.py       # VRAM/DDR5/NVMe coordination
│   ├── transfer.py           # Async weight transfers
│   └── cache.py              # Startup artifact caching
├── api/
│   ├── gateway.py            # OpenAI-compatible endpoint
│   └── router.py             # Model routing
├── engine/
│   ├── vllm_adapter.py       # vLLM integration (minimal mods)
│   └── sglang_adapter.py     # SGLang integration (alternative)
├── config/
│   └── settings.py           # DDR5 budget, thresholds, etc.
└── tests/
    └── ...
```

---

## Verification Plan

1. **Unit Tests**: State machine transitions, queue snapshot logic
2. **Integration Tests**: Full switch cycle with mock models
3. **Benchmark Suite**:
   - Measure actual PCIe transfer speeds
   - Time model switches under load
   - Compare 1st vs 2nd boot times
4. **Stress Tests**:
   - 200+ concurrent requests
   - Rapid model switching
   - Memory pressure scenarios
