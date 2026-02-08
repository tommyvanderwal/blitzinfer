#!/usr/bin/env python3
"""Test Mistral model directly."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def main():
    from vllm import LLM, SamplingParams

    print("Loading Mistral-7B...")
    llm = LLM(
        model='mistralai/Mistral-7B-Instruct-v0.3',
        gpu_memory_utilization=0.20,
        max_model_len=4096,
        max_num_seqs=20,
        max_num_batched_tokens=512,
        enforce_eager=True,
    )

    print("Generating...")
    out = llm.generate(
        ['What is the capital of Germany? Answer briefly.'],
        SamplingParams(max_tokens=20, temperature=0)
    )
    print(f'Output: {out[0].outputs[0].text}')
    print("SUCCESS")


if __name__ == '__main__':
    main()
