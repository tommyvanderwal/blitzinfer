#!/usr/bin/env python3
"""Test model switching using subprocess isolation"""

import time
import os
import subprocess
import sys

# Test scripts for each model
QWEN_TEST = '''
import time
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

from vllm import LLM, SamplingParams

KV_CACHE_BYTES = 10 * 1024**3

start = time.time()
llm = LLM(
    model="Qwen/Qwen3-VL-32B-Instruct",
    dtype="float16",
    max_model_len=1024,
    gpu_memory_utilization=0.85,
    max_num_seqs=4,
    disable_log_stats=True,
    enforce_eager=True,
    compilation_config={"custom_ops": ["none"]},
    kv_cache_memory_bytes=KV_CACHE_BYTES,
)
load_time = time.time() - start
print(f"LOAD_TIME:{load_time:.2f}")

start = time.time()
out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
gen_time = time.time() - start
print(f"GEN_TIME:{gen_time:.2f}")
print(f"OUTPUT:{out[0].outputs[0].text[:50]}")
'''

LLAMA_TEST = '''
import time
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "1"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

from vllm import LLM, SamplingParams

KV_CACHE_BYTES = 10 * 1024**3

start = time.time()
llm = LLM(
    model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    dtype="float16",
    max_model_len=1024,
    gpu_memory_utilization=0.85,
    max_num_seqs=4,
    disable_log_stats=True,
    enforce_eager=True,
    compilation_config={"custom_ops": ["none"]},
    kv_cache_memory_bytes=KV_CACHE_BYTES,
)
load_time = time.time() - start
print(f"LOAD_TIME:{load_time:.2f}")

start = time.time()
out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
gen_time = time.time() - start
print(f"GEN_TIME:{gen_time:.2f}")
print(f"OUTPUT:{out[0].outputs[0].text[:50]}")
'''

def run_model_test(code, model_name):
    """Run model test in subprocess"""
    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300
    )
    total_time = time.time() - start

    load_time = None
    gen_time = None
    output = None

    for line in result.stdout.split('\n'):
        if line.startswith("LOAD_TIME:"):
            load_time = float(line.split(":")[1])
        elif line.startswith("GEN_TIME:"):
            gen_time = float(line.split(":")[1])
        elif line.startswith("OUTPUT:"):
            output = line.split(":", 1)[1]

    if result.returncode != 0:
        print(f"ERROR: {result.stderr[-500:]}")
        return None

    print(f"Load time: {load_time:.2f}s")
    print(f"Gen time: {gen_time:.2f}s")
    print(f"Output: {output}")
    print(f"Total subprocess time: {total_time:.2f}s")

    return {"load": load_time, "gen": gen_time, "total": total_time}

def main():
    print("="*60)
    print("BlitzInfer Model Switching Test (Subprocess Isolation)")
    print("="*60)

    results = []
    num_cycles = 2

    for i in range(num_cycles):
        print(f"\n{'#'*60}")
        print(f"# Cycle {i+1}/{num_cycles}")
        print(f"{'#'*60}")

        cycle_start = time.time()

        # Test Qwen
        qwen_result = run_model_test(QWEN_TEST, "Qwen3-VL-32B")

        # Test Llama
        llama_result = run_model_test(LLAMA_TEST, "Llama-3.1-70B-AWQ")

        cycle_time = time.time() - cycle_start

        if qwen_result and llama_result:
            results.append({
                "qwen": qwen_result,
                "llama": llama_result,
                "cycle": cycle_time
            })

        print(f"\nCycle {i+1} total: {cycle_time:.2f}s")

    # Summary
    if results:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

        qwen_loads = [r["qwen"]["load"] for r in results]
        llama_loads = [r["llama"]["load"] for r in results]
        cycles = [r["cycle"] for r in results]

        print(f"Qwen load times: {qwen_loads}")
        print(f"  Average: {sum(qwen_loads)/len(qwen_loads):.1f}s")
        print(f"Llama load times: {llama_loads}")
        print(f"  Average: {sum(llama_loads)/len(llama_loads):.1f}s")
        print(f"Full cycle times: {cycles}")
        print(f"  Average: {sum(cycles)/len(cycles):.1f}s")
        print(f"\nModel switch time (subprocess overhead + loading):")
        print(f"  Qwen → Llama: ~{sum(llama_loads)/len(llama_loads):.0f}s")
        print(f"  Llama → Qwen: ~{sum(qwen_loads)/len(qwen_loads):.0f}s")

if __name__ == "__main__":
    main()
