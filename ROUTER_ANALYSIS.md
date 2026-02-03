# Router & Queue Architecture Analysis

## Research Summary (Feb 2026)

Analysis of routing/queue approaches for BlitzInfer's multi-model serving architecture.

## Options Evaluated

### 1. LiteLLM Router
- **What**: Python proxy that routes OpenAI-compatible requests to multiple backends
- **Verdict**: REJECTED - adds HTTP hop latency, doesn't solve our core problem (in-process model switching)
- LiteLLM assumes multiple backends are already running; we have one GPU

### 2. vLLM Router
- **What**: vLLM's built-in multi-model serving
- **Verdict**: REJECTED - requires separate vLLM server processes per model, doesn't support single-GPU switching

### 3. TGI (Text Generation Inference)
- **What**: HuggingFace's serving runtime
- **Verdict**: REJECTED - Rust-based, no in-process Python API, no model switching support

### 4. SGLang Engine (In-Process)
- **What**: SGLang's `Engine()` class - in-process LLM engine with subprocess scheduler
- **Verdict**: CHOSEN
- Native Harmony support for GPT-OSS-120B
- `update_weights_from_disk()` for same-architecture switching
- `release_memory_occupation()` / `resume_memory_occupation()` for GPU memory management
- `generate()` returns simple dicts, clean API
- Streaming via `stream=True` parameter
- Vision/multimodal support built-in

### 5. Custom Queue (asyncio.Queue)
- **What**: Simple per-model asyncio.Queue upstream of the engine
- **Verdict**: CHOSEN - minimal, sufficient for our needs
- One queue per registered model
- Background task dequeues from active model's queue
- Drain-before-switch protocol for clean model transitions

## Architecture Decision

```
FastAPI (:8000)
  -> Per-model asyncio.Queue (raw ChatCompletionRequest objects)
  -> Queue processor (dequeue active model, forward to engine)
  -> SGLang Engine (in-process, replaces vLLM LLM())
```

### Queue State Machine

```
SERVING: Dequeuing from active model's queue, calling engine.generate()
DRAINING: Active queue empty, waiting for in_flight == 0
SWITCHING: Shutting down engine, loading new model
```

### Switch Protocol

1. Request for model-B arrives while serving model-A
2. Enqueue in `queues["model-B"]`, start prefetch(model-B)
3. Continue serving model-A until its queue is empty AND in_flight == 0
4. Transition to SWITCHING state
5. `engine.shutdown()` + cleanup GPU
6. Create new `Engine(model_path=model-B)`
7. Transition to SERVING, start dequeuing model-B

### Why Not `update_weights_from_disk()`?

SGLang's weight update is tempting but:
- Only works for same-architecture models
- Our models span different architectures (GPT-OSS MoE, Qwen, Mistral, Llama)
- Full engine restart is more reliable for cross-architecture switching
- We already have StandbyManager for fast prefetch

### Harmony Handling

SGLang has native Harmony support via `sglang.srt.entrypoints.harmony_utils`:
- Same API as vLLM's harmony_utils (both adapted from openai_harmony)
- `render_for_completion()` for input encoding
- `parse_output_into_messages()` for output parsing
- `get_stop_tokens_for_assistant_actions()` for stop tokens
- Streaming parser via `sglang.srt.parser.harmony_parser`

This eliminates ~300 lines of manual Harmony code from server.py.

### Tokenization

SGLang Engine handles tokenization internally via its TokenizerManager subprocess.
For chat messages, we use `tokenizer.apply_chat_template()` to format prompts.
For Harmony (GPT-OSS), we use `render_for_completion()` to get token IDs directly.

## Key SGLang API Reference

```python
# Create engine
engine = sgl.Engine(model_path="openai/gpt-oss-120b", context_length=131072)

# Generate (sync)
output = engine.generate(prompt="Hello", sampling_params={"max_new_tokens": 100})
# Returns: {"text": "...", "output_ids": [...], "meta_info": {...}}

# Generate with token IDs (for Harmony)
output = engine.generate(input_ids=[...], sampling_params={...})

# Async generate
output = await engine.async_generate(prompt="Hello", sampling_params={...})

# Streaming
async for chunk in await engine.async_generate(prompt="Hello", stream=True):
    # chunk["text"] is CUMULATIVE (not delta)
    pass

# Shutdown
engine.shutdown()
```

## Performance Expectations

| Metric | vLLM (current) | SGLang (expected) |
|--------|---------------|-------------------|
| Cold model load | ~20s | ~20s (similar) |
| Warm switch (standby) | ~6s | ~6s (standby still works) |
| Inference throughput | baseline | ~1.1-1.3x (better scheduler) |
| Harmony parsing | manual (~300 LOC) | native (0 LOC) |
| Memory cleanup | complex (InprocClient nav) | `engine.shutdown()` (kills process tree) |
