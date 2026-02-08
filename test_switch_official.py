#!/usr/bin/env python3
"""Test switching between GPT-OSS-120B and Qwen3-VL-32B using subprocess isolation"""

import time
import subprocess
import sys
import os

# Detect platform
IS_ROCM = os.path.exists("/opt/rocm")

GPT_OSS_TEST = '''
import time
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "1"
if {is_rocm}:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
else:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

from vllm import LLM, SamplingParams

KV_CACHE_BYTES = 10 * 1024**3

start = time.time()
llm = LLM(
    model="openai/gpt-oss-120b",
    dtype="bfloat16",
    max_model_len=1024,
    gpu_memory_utilization=0.85,
    max_num_seqs=2,
    disable_log_stats=True,
    enforce_eager=True,
    compilation_config={{"custom_ops": ["none"]}} if {is_rocm} else {{}},
    kv_cache_memory_bytes=KV_CACHE_BYTES if {is_rocm} else None,
)
load_time = time.time() - start
print(f"LOAD_TIME:{{load_time:.2f}}")

start = time.time()
out = llm.generate(["The meaning of life is"], SamplingParams(max_tokens=20))
gen_time = time.time() - start
print(f"GEN_TIME:{{gen_time:.2f}}")
print(f"OUTPUT:{{out[0].outputs[0].text[:80]}}")
'''

QWEN_TEST = '''
import time
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "1"
if {is_rocm}:
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
else:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

from vllm import LLM, SamplingParams

KV_CACHE_BYTES = 10 * 1024**3

start = time.time()
llm = LLM(
    model="Qwen/Qwen3-VL-32B-Instruct",
    dtype="float16",
    max_model_len=1024,
    gpu_memory_utilization=0.85,
    max_num_seqs=2,
    disable_log_stats=True,
    enforce_eager=True,
    compilation_config={{"custom_ops": ["none"]}} if {is_rocm} else {{}},
    kv_cache_memory_bytes=KV_CACHE_BYTES if {is_rocm} else None,
)
load_time = time.time() - start
print(f"LOAD_TIME:{{load_time:.2f}}")

start = time.time()
out = llm.generate(["Hello, I am"], SamplingParams(max_tokens=20))
gen_time = time.time() - start
print(f"GEN_TIME:{{gen_time:.2f}}")
print(f"OUTPUT:{{out[0].outputs[0].text[:80]}}")
'''

def run_model_test(code, model_name, is_rocm):
    """Run model test in subprocess"""
    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    print(f"{'='*60}")

    formatted_code = code.format(is_rocm=is_rocm)

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-c", formatted_code],
        capture_output=True,
        text=True,
        timeout=600
    )
    total_time = time.time() - start

    load_time = gen_time = output = None

    for line in result.stdout.split('\n'):
        if line.startswith("LOAD_TIME:"):
            load_time = float(line.split(":")[1])
        elif line.startswith("GEN_TIME:"):
            gen_time = float(line.split(":")[1])
        elif line.startswith("OUTPUT:"):
            output = line.split(":", 1)[1]

    if result.returncode != 0:
        print(f"ERROR: {result.stderr[-1000:]}")
        return None

    print(f"Load time: {load_time:.1f}s")
    print(f"Gen time: {gen_time:.1f}s")
    print(f"Output: {output}")
    print(f"Total: {total_time:.1f}s")

    return {"load": load_time, "gen": gen_time, "total": total_time}

def main():
    platform = "780M (ROCm)" if IS_ROCM else "RTX PRO 6000 (CUDA)"

    print("="*60)
    print(f"Model Switching Test - {platform}")
    print("="*60)
    print(f"Models: GPT-OSS-120B (MXFP4) <-> Qwen3-VL-32B")

    results = []
    num_cycles = 2

    for i in range(num_cycles):
        print(f"\n{'#'*60}")
        print(f"# Switch Cycle {i+1}/{num_cycles}")
        print(f"{'#'*60}")

        cycle_start = time.time()

        # GPT-OSS-120B
        gpt_result = run_model_test(GPT_OSS_TEST, "GPT-OSS-120B", IS_ROCM)

        # Qwen3-VL-32B
        qwen_result = run_model_test(QWEN_TEST, "Qwen3-VL-32B", IS_ROCM)

        cycle_time = time.time() - cycle_start

        if gpt_result and qwen_result:
            results.append({
                "gpt": gpt_result,
                "qwen": qwen_result,
                "cycle": cycle_time
            })
            print(f"\nCycle {i+1} total: {cycle_time:.1f}s")

    # Summary
    if results:
        print("\n" + "="*60)
        print(f"RESULTS SUMMARY - {platform}")
        print("="*60)

        gpt_loads = [r["gpt"]["load"] for r in results]
        qwen_loads = [r["qwen"]["load"] for r in results]
        cycles = [r["cycle"] for r in results]

        print(f"GPT-OSS-120B load: {gpt_loads} -> avg {sum(gpt_loads)/len(gpt_loads):.1f}s")
        print(f"Qwen3-VL-32B load: {qwen_loads} -> avg {sum(qwen_loads)/len(qwen_loads):.1f}s")
        print(f"Full cycle: {cycles} -> avg {sum(cycles)/len(cycles):.1f}s")
        print(f"\nSwitch times (subprocess overhead + load):")
        print(f"  GPT-OSS → Qwen: ~{sum(qwen_loads)/len(qwen_loads):.0f}s")
        print(f"  Qwen → GPT-OSS: ~{sum(gpt_loads)/len(gpt_loads):.0f}s")

if __name__ == "__main__":
    main()
