#!/usr/bin/env python3
"""Test 4 model switches between GPT-OSS-120B and Qwen3-VL-32B"""

import subprocess
import sys
import time
import os

IS_ROCM = os.path.exists("/opt/rocm")
PLATFORM = "780M" if IS_ROCM else "RTX_PRO_6000"

# Simpler test scripts
GPT_TEST = f'''
import time, os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
{"os.environ['VLLM_SKIP_WARMUP'] = '1'" if IS_ROCM else "os.environ['VLLM_ATTENTION_BACKEND'] = 'TORCH_SDPA'"}
{"os.environ['HIP_VISIBLE_DEVICES'] = '0'; os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'" if IS_ROCM else ""}
from vllm import LLM, SamplingParams
t=time.time()
llm=LLM(model="openai/gpt-oss-120b",dtype="bfloat16",max_model_len=512,max_num_seqs=2,disable_log_stats=True,enforce_eager=True,{"compilation_config={'custom_ops':['none']},kv_cache_memory_bytes=10*1024**3" if IS_ROCM else ""})
print(f"LOAD:{{time.time()-t:.1f}}")
t=time.time()
o=llm.generate(["Hi"],SamplingParams(max_tokens=10))
print(f"GEN:{{time.time()-t:.1f}}")
print(f"OUT:{{o[0].outputs[0].text[:30]}}")
'''

QWEN_TEST = f'''
import time, os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
{"os.environ['VLLM_SKIP_WARMUP'] = '1'" if IS_ROCM else "os.environ['VLLM_ATTENTION_BACKEND'] = 'TORCH_SDPA'"}
{"os.environ['HIP_VISIBLE_DEVICES'] = '0'; os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'" if IS_ROCM else ""}
from vllm import LLM, SamplingParams
t=time.time()
llm=LLM(model="Qwen/Qwen3-VL-32B-Instruct",dtype="float16",max_model_len=512,max_num_seqs=2,disable_log_stats=True,enforce_eager=True,{"compilation_config={'custom_ops':['none']},kv_cache_memory_bytes=10*1024**3" if IS_ROCM else ""})
print(f"LOAD:{{time.time()-t:.1f}}")
t=time.time()
o=llm.generate(["Hi"],SamplingParams(max_tokens=10))
print(f"GEN:{{time.time()-t:.1f}}")
print(f"OUT:{{o[0].outputs[0].text[:30]}}")
'''

def run_test(code, name):
    """Run model test in subprocess"""
    start = time.time()
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    total = time.time() - start

    load = gen = out = None
    for line in result.stdout.split('\n'):
        if line.startswith("LOAD:"): load = float(line.split(":")[1])
        elif line.startswith("GEN:"): gen = float(line.split(":")[1])
        elif line.startswith("OUT:"): out = line.split(":", 1)[1][:30]

    if result.returncode != 0:
        print(f"  {name}: FAILED - {result.stderr[-200:]}")
        return None

    print(f"  {name}: load={load:.1f}s gen={gen:.1f}s total={total:.1f}s out='{out}'")
    return {"load": load, "gen": gen, "total": total}

def main():
    print(f"="*60)
    print(f"4-Switch Stability Test - {PLATFORM}")
    print(f"="*60)

    results = {"gpt": [], "qwen": []}

    for i in range(4):
        print(f"\nSwitch {i+1}/4:")

        gpt = run_test(GPT_TEST, "GPT-OSS-120B")
        if gpt: results["gpt"].append(gpt)

        qwen = run_test(QWEN_TEST, "Qwen3-VL-32B")
        if qwen: results["qwen"].append(qwen)

    print(f"\n{'='*60}")
    print(f"SUMMARY - {PLATFORM}")
    print(f"{'='*60}")

    if results["gpt"]:
        loads = [r["load"] for r in results["gpt"]]
        print(f"GPT-OSS-120B: {len(loads)}/4 successful, avg load={sum(loads)/len(loads):.1f}s")

    if results["qwen"]:
        loads = [r["load"] for r in results["qwen"]]
        print(f"Qwen3-VL-32B: {len(loads)}/4 successful, avg load={sum(loads)/len(loads):.1f}s")

if __name__ == "__main__":
    main()
