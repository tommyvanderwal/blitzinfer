#!/usr/bin/env python3
"""Trace exactly when the 160MB blocks are allocated during vLLM model load."""

import os
import gc
import sys
import time

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'

import torch


def get_memory():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3


def count_large_blocks(min_size_mb=100):
    """Count blocks larger than min_size_mb."""
    snapshot = torch.cuda.memory._snapshot()
    if not snapshot or 'segments' not in snapshot:
        return 0, []

    large_blocks = []
    for segment in snapshot['segments']:
        for block in segment.get('blocks', []):
            if block.get('state') == 'active_allocated':
                size = block.get('size', block.get('requested_size', 0))
                if size > min_size_mb * 1024 * 1024:
                    # Get trace if available
                    frames = block.get('frames', [])
                    trace = []
                    if frames:
                        for f in frames[:5]:
                            fname = f.get('filename', '?')
                            line = f.get('line', '?')
                            name = f.get('name', '?')
                            trace.append(f"{fname}:{line}:{name}")
                    large_blocks.append({
                        'size_mb': size / 1024**2,
                        'address': block.get('address', segment.get('address', 0)),
                        'trace': trace,
                    })

    return len(large_blocks), large_blocks


def trace_vllm_load():
    """Trace memory allocations during vLLM model load."""
    from vllm import LLM, SamplingParams

    MODEL = "Qwen/Qwen3-VL-32B-Thinking-FP8"

    # Enable detailed tracing
    torch.cuda.memory._record_memory_history(
        enabled='all',
        context='all',
        stacks='all',
        max_entries=500000
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("=" * 70)
    print("TRACING 160MB BLOCK ALLOCATION SOURCE")
    print("=" * 70)

    baseline = get_memory()
    num_large, _ = count_large_blocks()
    print(f"\n[BASELINE] {baseline:.3f} GB, {num_large} large blocks (>100MB)")

    # Step-by-step load with monitoring
    print("\n--- Step 1: Import vLLM engine core ---")
    from vllm.v1.engine.core import EngineCore
    step1 = get_memory()
    n1, blocks1 = count_large_blocks()
    print(f"    Allocated: {step1:.3f} GB, {n1} large blocks")

    print("\n--- Step 2: Create LLM (starts init) ---")
    print("    This will trace where 160MB blocks appear...")

    # Intercept at key points
    import vllm.v1.engine.core as core_module

    original_init = core_module.EngineCore.__init__

    def traced_init(self, *args, **kwargs):
        print("      [EngineCore.__init__ start]")
        mem_before = get_memory()
        n_before, _ = count_large_blocks()
        print(f"        Before: {mem_before:.3f} GB, {n_before} large blocks")

        result = original_init(self, *args, **kwargs)

        mem_after = get_memory()
        n_after, blocks_after = count_large_blocks()
        print(f"        After: {mem_after:.3f} GB, {n_after} large blocks")
        if n_after > n_before:
            print(f"        NEW LARGE BLOCKS:")
            for b in blocks_after[-3:]:  # Show last 3 new blocks
                print(f"          {b['size_mb']:.1f} MB at 0x{b['address']:x}")
                if b['trace']:
                    print(f"            Trace: {b['trace'][0] if b['trace'] else 'no trace'}")

        return result

    core_module.EngineCore.__init__ = traced_init

    # Also trace model_runner creation
    try:
        import vllm.v1.worker.gpu_model_runner as runner_module

        original_runner_init = runner_module.GPUModelRunner.__init__

        def traced_runner_init(self, *args, **kwargs):
            print("      [GPUModelRunner.__init__ start]")
            mem_before = get_memory()
            n_before, _ = count_large_blocks()
            print(f"        Before: {mem_before:.3f} GB, {n_before} large blocks")

            result = original_runner_init(self, *args, **kwargs)

            mem_after = get_memory()
            n_after, blocks_after = count_large_blocks()
            print(f"        After: {mem_after:.3f} GB, {n_after} large blocks")
            if n_after > n_before:
                print(f"        NEW LARGE BLOCKS:")
                for b in blocks_after[-(n_after-n_before):]:
                    print(f"          {b['size_mb']:.1f} MB at 0x{b['address']:x}")
                    if b['trace']:
                        for t in b['trace'][:3]:
                            print(f"            {t}")

            return result

        runner_module.GPUModelRunner.__init__ = traced_runner_init
    except Exception as e:
        print(f"    Could not trace GPUModelRunner: {e}")

    # Also trace the load_model call
    try:
        import vllm.v1.worker.gpu_model_runner as runner_module

        original_load = runner_module.GPUModelRunner.load_model

        def traced_load(self, *args, **kwargs):
            print("      [GPUModelRunner.load_model start]")
            mem_before = get_memory()
            n_before, _ = count_large_blocks()
            print(f"        Before: {mem_before:.3f} GB, {n_before} large blocks")

            result = original_load(self, *args, **kwargs)

            mem_after = get_memory()
            n_after, blocks_after = count_large_blocks()
            print(f"        After: {mem_after:.3f} GB, {n_after} large blocks")
            if n_after > n_before:
                print(f"        NEW LARGE BLOCKS during model load:")
                for b in blocks_after[-(n_after-n_before):]:
                    print(f"          {b['size_mb']:.1f} MB at 0x{b['address']:x}")
                    if b['trace']:
                        for t in b['trace'][:3]:
                            print(f"            {t}")

            return result

        runner_module.GPUModelRunner.load_model = traced_load
    except Exception as e:
        print(f"    Could not trace load_model: {e}")

    # Now create the LLM
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
    )

    print("\n--- Step 3: Generate (first forward pass) ---")
    mem_before_gen = get_memory()
    n_before_gen, _ = count_large_blocks()
    print(f"    Before generate: {mem_before_gen:.3f} GB, {n_before_gen} large blocks")

    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    _ = out[0].outputs[0].text

    mem_after_gen = get_memory()
    n_after_gen, blocks_after_gen = count_large_blocks()
    print(f"    After generate: {mem_after_gen:.3f} GB, {n_after_gen} large blocks")
    if n_after_gen > n_before_gen:
        print(f"    NEW LARGE BLOCKS during generate:")
        for b in blocks_after_gen[-(n_after_gen-n_before_gen):]:
            print(f"      {b['size_mb']:.1f} MB at 0x{b['address']:x}")
            if b['trace']:
                for t in b['trace'][:5]:
                    print(f"        {t}")

    # Cleanup and check what remains
    print("\n--- Step 4: Cleanup ---")
    from blitzinfer.engine.cleanup import full_cleanup
    freed = full_cleanup(llm, nuclear=True)
    llm = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    final = get_memory()
    n_final, blocks_final = count_large_blocks()
    print(f"\n[FINAL] {final:.3f} GB, {n_final} large blocks")

    if blocks_final:
        print(f"\nRemaining large blocks (source of leak):")
        for b in blocks_final:
            print(f"  {b['size_mb']:.1f} MB at 0x{b['address']:x}")
            if b['trace']:
                print(f"    Stack trace:")
                for t in b['trace']:
                    print(f"      {t}")
            else:
                print(f"    No stack trace (internal allocation)")

    torch.cuda.memory._record_memory_history(enabled=None)

    return 0


if __name__ == "__main__":
    sys.exit(trace_vllm_load())
