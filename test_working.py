#!/usr/bin/env python3
"""Test vLLM with settings from working reference project."""
import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def main():
    from vllm import LLM, SamplingParams
    print("Loading model...")
    llm = LLM(
        model='Qwen/Qwen2.5-7B-Instruct',
        gpu_memory_utilization=0.20,
        max_model_len=32768,
        max_num_seqs=20,
        max_num_batched_tokens=512,
        enforce_eager=True,
    )
    print("Generating...")
    out = llm.generate(['Hello, how are you?'], SamplingParams(max_tokens=20))
    print(f'Output: {out[0].outputs[0].text}')
    print("SUCCESS")

if __name__ == '__main__':
    main()
