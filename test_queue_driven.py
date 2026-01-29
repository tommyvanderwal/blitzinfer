#!/usr/bin/env python3
"""Queue-driven model switching test with vision, tool calls, and memory monitoring.

Requirements tested:
- Switch time < 10 seconds (warm)
- Preload time < 10 seconds
- Memory drift < 2GB total
- Vision model with actual images
- Tool call verification
- Multiple models with real request patterns
"""

import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import gc
import json
import time
import base64
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

from blitzinfer.engine.cleanup import full_cleanup
from blitzinfer.orchestrator.standby_manager import StandbyManager
from blitzinfer.memory import set_preloaded_weights


# HuggingFace cache path resolver
def resolve_hf_path(model_id: str) -> str:
    """Resolve HuggingFace model ID to local cache path."""
    from huggingface_hub import snapshot_download
    import os

    # Check if it's already a local path
    if os.path.exists(model_id):
        return model_id

    # Use huggingface_hub to get the local cache path
    try:
        local_path = snapshot_download(model_id, local_files_only=True)
        return local_path
    except Exception:
        # Fall back to manual path construction
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_dir = f"models--{model_id.replace('/', '--')}"
        full_path = os.path.join(cache_dir, model_dir, "snapshots")
        if os.path.exists(full_path):
            # Get the most recent snapshot
            snapshots = sorted(os.listdir(full_path))
            if snapshots:
                return os.path.join(full_path, snapshots[-1])
        raise FileNotFoundError(f"Could not find local path for {model_id}")


# Test configuration
MODELS = {
    "gpt-oss-120b": {
        "hf_id": "openai/gpt-oss-120b",
        "type": "text",
        "max_model_len": 128000,
        "gpu_memory_utilization": 0.85,
        "quantization": None,
        "has_tools": True,
        "has_harmony": True,
    },
    "qwen3-vl-32b": {
        "hf_id": "Qwen/Qwen3-VL-32B-Thinking-FP8",
        "type": "vision",
        "max_model_len": 128000,
        "gpu_memory_utilization": 0.85,
        "quantization": None,
        "has_tools": False,
        "has_harmony": False,
    },
    "mistral-small-24b": {
        "hf_id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "type": "text",
        "max_model_len": 128000,
        "gpu_memory_utilization": 0.85,
        "quantization": None,
        "has_tools": True,
        "has_harmony": False,
    },
}

# Thresholds
MAX_SWITCH_TIME = 10.0  # seconds
MAX_PRELOAD_TIME = 10.0  # seconds
MAX_MEMORY_DRIFT = 2.0  # GB total


@dataclass
class Request:
    """Simulated user request."""
    model: str
    prompt: str
    image_path: Optional[str] = None
    tools: Optional[List[Dict]] = None
    expected_tool_call: Optional[str] = None
    max_tokens: int = 100


@dataclass
class Result:
    """Result of processing a request."""
    model: str
    prompt: str
    response: str
    switch_time: float
    preload_time: float
    inference_time: float
    tool_call: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3


def get_total_gpu_memory_gb() -> float:
    """Get total GPU memory in use (allocated + reserved)."""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return max(allocated, reserved)


def create_test_image() -> str:
    """Create a simple test image and return its path."""
    try:
        from PIL import Image
        import numpy as np

        # Create a simple test image: 256x256 with some shapes
        img = Image.new('RGB', (256, 256), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        # Draw a red circle
        draw.ellipse([50, 50, 150, 150], fill='red', outline='darkred')

        # Draw a blue rectangle
        draw.rectangle([160, 100, 240, 200], fill='blue', outline='darkblue')

        # Add text
        draw.text((80, 220), "Test Image", fill='black')

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            return f.name
    except ImportError:
        # If PIL not available, return None
        return None


def create_request_queue() -> List[Request]:
    """Create a realistic request queue that tests various scenarios."""
    test_image = create_test_image()

    requests = []

    # 1. Start with text model (gpt-oss)
    requests.append(Request(
        model="gpt-oss-120b",
        prompt="What is 7 + 8? Answer with just the number.",
        max_tokens=50,
    ))

    # 2. Tool call test (gpt-oss)
    requests.append(Request(
        model="gpt-oss-120b",
        prompt="Use the bash tool to list files in the current directory.",
        tools=[{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to run"}
                    },
                    "required": ["command"]
                }
            }
        }],
        expected_tool_call="bash",
        max_tokens=200,
    ))

    # 3. Switch to vision model (qwen3-vl)
    if test_image:
        requests.append(Request(
            model="qwen3-vl-32b",
            prompt="Describe the shapes and colors you see in this image.",
            image_path=test_image,
            max_tokens=200,
        ))
    else:
        # Fallback text-only request
        requests.append(Request(
            model="qwen3-vl-32b",
            prompt="What is the capital of France?",
            max_tokens=50,
        ))

    # 4. Another vision request (same model - no switch)
    if test_image:
        requests.append(Request(
            model="qwen3-vl-32b",
            prompt="How many distinct objects are in this image?",
            image_path=test_image,
            max_tokens=100,
        ))

    # 5. Switch to mistral
    requests.append(Request(
        model="mistral-small-24b",
        prompt="Write a haiku about programming.",
        max_tokens=100,
    ))

    # 6. Tool call test on mistral
    requests.append(Request(
        model="mistral-small-24b",
        prompt="Use the calculator function to compute 123 * 456.",
        tools=[{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform arithmetic calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        }],
        expected_tool_call="calculator",
        max_tokens=200,
    ))

    # 7. Switch back to gpt-oss
    requests.append(Request(
        model="gpt-oss-120b",
        prompt="Explain quantum entanglement in one sentence.",
        max_tokens=100,
    ))

    # 8. Switch to qwen again
    requests.append(Request(
        model="qwen3-vl-32b",
        prompt="What year did humans first land on the moon?",
        max_tokens=50,
    ))

    # 9. Final switch to mistral
    requests.append(Request(
        model="mistral-small-24b",
        prompt="List the first 5 prime numbers.",
        max_tokens=50,
    ))

    return requests


def load_model(model_key: str, standby: StandbyManager) -> LLM:
    """Load a model, using standby if available."""
    config = MODELS[model_key]
    hf_id = config["hf_id"]

    # Check if we have preloaded weights
    premerged = None
    load_format = "auto"

    if standby.is_ready(hf_id):
        premerged = standby.consume_standby()
        set_preloaded_weights(premerged)
        load_format = "pinned_arena"
        print(f"  Using preloaded weights for {model_key}")

    kwargs = {
        "model": hf_id,
        "trust_remote_code": True,
        "max_model_len": config["max_model_len"],
        "gpu_memory_utilization": config["gpu_memory_utilization"],
        "enforce_eager": True,
        "load_format": load_format,
    }

    if config.get("quantization"):
        kwargs["quantization"] = config["quantization"]

    return LLM(**kwargs)


def process_request_gpt_oss(llm: LLM, request: Request) -> str:
    """Process request for GPT-OSS with Harmony encoding."""
    try:
        from vllm.entrypoints.openai.parser.harmony_utils import (
            parse_chat_output,
            parse_chat_inputs_to_harmony_messages,
            render_for_completion,
            get_system_message,
            get_developer_message,
            get_stop_tokens_for_assistant_actions,
        )
        HAS_HARMONY = True
    except ImportError:
        HAS_HARMONY = False

    if not HAS_HARMONY:
        # Fallback to regular generation
        sp = SamplingParams(max_tokens=request.max_tokens, temperature=0.0)
        out = llm.generate([request.prompt], sp)
        return out[0].outputs[0].text.strip()

    # Build Harmony messages
    chat_msgs = [{"role": "user", "content": request.prompt}]
    sys_msg = get_system_message(with_custom_tools=bool(request.tools))
    harmony_msgs = [sys_msg] + parse_chat_inputs_to_harmony_messages(chat_msgs)

    # Add tools if present
    if request.tools:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
            FunctionDefinition as VLLMFunctionDefinition,
        )
        tools_for_harmony = []
        for tool in request.tools:
            func = tool["function"]
            func_def = VLLMFunctionDefinition(
                name=func["name"],
                description=func.get("description", ""),
                parameters=func.get("parameters", {}),
            )
            tool_param = ChatCompletionToolsParam(type="function", function=func_def)
            tools_for_harmony.append(tool_param)
        dev_msg = get_developer_message(tools=tools_for_harmony)
        harmony_msgs.insert(1, dev_msg)

    # Render to token IDs
    prompt_token_ids = render_for_completion(harmony_msgs)
    stop_tokens = get_stop_tokens_for_assistant_actions()

    sp = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=0.0,
        stop_token_ids=stop_tokens,
    )

    out = llm.generate([{"prompt_token_ids": prompt_token_ids}], sp)
    output_text = out[0].outputs[0].text
    output_token_ids = out[0].outputs[0].token_ids

    # Parse Harmony output
    result = parse_chat_output(list(output_token_ids))

    # Check for tool calls
    if hasattr(result, 'tool_calls') and result.tool_calls:
        return f"[TOOL:{result.tool_calls[0].function.name}] {result.tool_calls[0].function.arguments}"

    return result.final_content or output_text


def process_request_vision(llm: LLM, request: Request) -> str:
    """Process request for vision model with optional image."""
    if request.image_path and os.path.exists(request.image_path):
        # Load image as base64
        with open(request.image_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')

        # Multi-modal prompt format
        prompt = {
            "prompt": request.prompt,
            "multi_modal_data": {
                "image": [f"data:image/png;base64,{img_data}"]
            }
        }
        sp = SamplingParams(max_tokens=request.max_tokens, temperature=0.0)
        out = llm.generate([prompt], sp)
    else:
        sp = SamplingParams(max_tokens=request.max_tokens, temperature=0.0)
        out = llm.generate([request.prompt], sp)

    return out[0].outputs[0].text.strip()


def process_request_standard(llm: LLM, request: Request) -> str:
    """Process request for standard text model."""
    # Build prompt with tools if present
    prompt = request.prompt

    if request.tools:
        # Add tool info to prompt for models without native tool support
        tool_desc = "\n".join([
            f"- {t['function']['name']}: {t['function'].get('description', '')}"
            for t in request.tools
        ])
        prompt = f"""You have access to these tools:
{tool_desc}

To use a tool, respond with: [TOOL:tool_name] {{arguments}}

User: {request.prompt}"""

    sp = SamplingParams(max_tokens=request.max_tokens, temperature=0.0)
    out = llm.generate([prompt], sp)
    return out[0].outputs[0].text.strip()


def process_request(llm: LLM, request: Request, model_config: Dict) -> str:
    """Process a request using the appropriate method for the model."""
    if model_config.get("has_harmony"):
        return process_request_gpt_oss(llm, request)
    elif model_config.get("type") == "vision":
        return process_request_vision(llm, request)
    else:
        return process_request_standard(llm, request)


def run_queue_test():
    """Run the queue-driven test."""
    print("=" * 60)
    print("QUEUE-DRIVEN MODEL SWITCHING TEST")
    print("=" * 60)
    print(f"\nRequirements:")
    print(f"  - Switch time: < {MAX_SWITCH_TIME}s")
    print(f"  - Preload time: < {MAX_PRELOAD_TIME}s")
    print(f"  - Memory drift: < {MAX_MEMORY_DRIFT} GB total")
    print()

    # Initialize
    torch.cuda.empty_cache()
    gc.collect()
    baseline_memory = get_gpu_memory_gb()
    print(f"Baseline GPU memory: {baseline_memory:.2f} GB")

    # Create standby manager
    print("\nInitializing StandbyManager (80GB arena)...")
    standby = StandbyManager(
        arena_size_gb=80.0,
        chunk_size_gb=16.0,
        pin_memory=True,
        lazy_arena=False,
    )

    # Resolve all model paths and register
    print("Resolving model paths...")
    for key, config in MODELS.items():
        try:
            local_path = resolve_hf_path(config["hf_id"])
            config["local_path"] = local_path
            standby.register_model(config["hf_id"], local_path)
            print(f"  {key}: {local_path}")
        except FileNotFoundError as e:
            print(f"  {key}: NOT FOUND - {e}")
            config["local_path"] = None

    # Create request queue
    requests = create_request_queue()
    print(f"\nRequest queue: {len(requests)} requests")
    for i, req in enumerate(requests):
        img_marker = " [IMG]" if req.image_path else ""
        tool_marker = " [TOOL]" if req.tools else ""
        print(f"  {i+1}. {req.model}{img_marker}{tool_marker}: {req.prompt[:50]}...")

    # Process queue
    results: List[Result] = []
    current_llm = None
    current_model = None
    memory_readings = [baseline_memory]

    print("\n" + "=" * 60)
    print("PROCESSING QUEUE")
    print("=" * 60)

    for i, request in enumerate(requests):
        print(f"\n--- Request {i+1}/{len(requests)}: {request.model} ---")

        switch_time = 0.0
        preload_time = 0.0
        preload_was_ready = False

        # Check if we need to switch models
        if request.model != current_model:
            print(f"  Switching from {current_model or 'None'} to {request.model}")

            # Start prefetch for next model if not already ready
            config = MODELS[request.model]
            hf_id = config["hf_id"]

            if not standby.is_ready(hf_id):
                print(f"  Starting prefetch...")
                preload_start = time.time()
                standby.start_prefetch(hf_id)

                # Wait for prefetch to complete
                while not standby.is_ready(hf_id):
                    time.sleep(0.1)

                preload_time = time.time() - preload_start
                print(f"  Preload completed in {preload_time:.2f}s")
            else:
                preload_was_ready = True
                print(f"  Preload already ready")

            # Cleanup current model
            if current_llm is not None:
                print(f"  Cleaning up {current_model}...")
                freed = full_cleanup(current_llm)
                print(f"  Freed {freed:.1f} GB")
                current_llm = None

            # Load new model
            switch_start = time.time()
            try:
                current_llm = load_model(request.model, standby)
                switch_time = time.time() - switch_start
                current_model = request.model
                print(f"  Model loaded in {switch_time:.2f}s")
            except Exception as e:
                print(f"  ERROR loading model: {e}")
                results.append(Result(
                    model=request.model,
                    prompt=request.prompt,
                    response="",
                    switch_time=switch_time,
                    preload_time=preload_time,
                    inference_time=0.0,
                    success=False,
                    error=str(e),
                ))
                continue

        # Process the request
        model_config = MODELS[request.model]
        inference_start = time.time()

        try:
            response = process_request(current_llm, request, model_config)
            inference_time = time.time() - inference_start

            # Check for tool call
            tool_call = None
            if "[TOOL:" in response:
                tool_call = response.split("[TOOL:")[1].split("]")[0]

            results.append(Result(
                model=request.model,
                prompt=request.prompt,
                response=response,
                switch_time=switch_time,
                preload_time=preload_time,
                inference_time=inference_time,
                tool_call=tool_call,
                success=True,
            ))

            print(f"  Response ({inference_time:.2f}s): {response[:80]}...")

            if tool_call:
                expected = request.expected_tool_call
                if expected and tool_call == expected:
                    print(f"  Tool call: {tool_call} [EXPECTED]")
                elif expected:
                    print(f"  Tool call: {tool_call} [UNEXPECTED, expected {expected}]")
                else:
                    print(f"  Tool call: {tool_call}")

        except Exception as e:
            inference_time = time.time() - inference_start
            print(f"  ERROR: {e}")
            results.append(Result(
                model=request.model,
                prompt=request.prompt,
                response="",
                switch_time=switch_time,
                preload_time=preload_time,
                inference_time=inference_time,
                success=False,
                error=str(e),
            ))

        # Record memory after each request
        current_memory = get_gpu_memory_gb()
        memory_readings.append(current_memory)

        # Start prefetch for next different model (if any)
        next_different_model = None
        for future_req in requests[i+1:]:
            if future_req.model != current_model:
                next_different_model = future_req.model
                break

        if next_different_model:
            next_hf_id = MODELS[next_different_model]["hf_id"]
            if not standby.is_ready(next_hf_id):
                print(f"  Starting background prefetch for {next_different_model}...")
                standby.start_prefetch(next_hf_id)

    # Final cleanup
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)

    if current_llm:
        freed = full_cleanup(current_llm)
        print(f"Final cleanup freed: {freed:.1f} GB")

    standby.shutdown()
    del standby
    gc.collect()
    torch.cuda.empty_cache()

    final_memory = get_gpu_memory_gb()
    memory_readings.append(final_memory)

    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Calculate metrics
    successful = [r for r in results if r.success]
    switch_times = [r.switch_time for r in results if r.switch_time > 0]
    preload_times = [r.preload_time for r in results if r.preload_time > 0]
    total_drift = final_memory - baseline_memory
    max_memory = max(memory_readings)

    print(f"\nRequests: {len(successful)}/{len(results)} successful")

    if switch_times:
        avg_switch = sum(switch_times) / len(switch_times)
        max_switch = max(switch_times)
        print(f"\nSwitch times:")
        print(f"  Average: {avg_switch:.2f}s")
        print(f"  Max: {max_switch:.2f}s")
        print(f"  Target: < {MAX_SWITCH_TIME}s - {'PASS' if max_switch < MAX_SWITCH_TIME else 'FAIL'}")

    if preload_times:
        avg_preload = sum(preload_times) / len(preload_times)
        max_preload = max(preload_times)
        print(f"\nPreload times:")
        print(f"  Average: {avg_preload:.2f}s")
        print(f"  Max: {max_preload:.2f}s")
        print(f"  Target: < {MAX_PRELOAD_TIME}s - {'PASS' if max_preload < MAX_PRELOAD_TIME else 'FAIL'}")

    print(f"\nMemory:")
    print(f"  Baseline: {baseline_memory:.2f} GB")
    print(f"  Final: {final_memory:.2f} GB")
    print(f"  Peak: {max_memory:.2f} GB")
    print(f"  Drift: {total_drift:.2f} GB")
    print(f"  Target: < {MAX_MEMORY_DRIFT} GB - {'PASS' if total_drift < MAX_MEMORY_DRIFT else 'FAIL'}")

    # Tool call verification
    tool_requests = [r for r in results if MODELS[r.model].get("has_tools")]
    tool_results = [(r, r.tool_call) for r in results if r.tool_call]
    print(f"\nTool calls:")
    print(f"  Requests with tools: {len([r for r in requests if r.tools])}")
    print(f"  Tool calls detected: {len(tool_results)}")

    for r, tc in tool_results:
        print(f"    - {r.model}: {tc}")

    # Vision requests
    vision_requests = [r for r in requests if r.image_path]
    vision_results = [r for r in results if r.success and requests[results.index(r)].image_path]
    print(f"\nVision requests:")
    print(f"  With images: {len(vision_requests)}")
    print(f"  Successful: {len(vision_results)}")

    # Overall verdict
    print("\n" + "=" * 60)
    all_pass = True

    if switch_times and max(switch_times) >= MAX_SWITCH_TIME:
        all_pass = False
    if preload_times and max(preload_times) >= MAX_PRELOAD_TIME:
        all_pass = False
    if total_drift >= MAX_MEMORY_DRIFT:
        all_pass = False
    if len(successful) < len(results):
        all_pass = False

    if all_pass:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    import sys
    success = run_queue_test()
    sys.exit(0 if success else 1)
