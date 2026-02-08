#!/usr/bin/env python3
"""Test basic inference with transformers on ROCm."""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    print(f"Loading {model_name}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    print(f"Model memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()

    # Test inference
    prompt = "Hello, my name is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"Prompt: {prompt!r}")
    print("Generating...")

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )
    gen_time = time.time() - start

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {response!r}")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Tokens generated: {outputs.shape[1] - inputs['input_ids'].shape[1]}")

if __name__ == "__main__":
    main()
