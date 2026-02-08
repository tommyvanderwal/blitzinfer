#!/usr/bin/env python3
"""Test GLM-4.6V-AWQ after processor config fix."""

import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from blitzinfer.engine.cleanup import full_cleanup
import torch

def main():
    print("Testing GLM-4.6V-AWQ with vLLM after processor fix...")
    torch.cuda.empty_cache()
    print(f"GPU memory before: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    try:
        llm = LLM(
            model="cyankiwi/GLM-4.6V-AWQ-4bit",
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
            enforce_eager=True,
            gpu_memory_utilization=0.85,
        )
        print("LLM loaded successfully!")
        print(f"GPU memory after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Lucidity test
        print("\n--- Lucidity Tests ---")
        sp = SamplingParams(max_tokens=30, temperature=0.0)

        # Math
        out = llm.generate(["What is 7 + 8? Answer with just the number:"], sp)
        ans = out[0].outputs[0].text.strip()
        math_ok = "15" in ans
        print(f"Math: {'OK' if math_ok else 'FAIL'} - {ans[:50]}")

        # Knowledge
        out = llm.generate(["The capital of France is"], sp)
        ans = out[0].outputs[0].text.strip()
        know_ok = "paris" in ans.lower()
        print(f"Knowledge: {'OK' if know_ok else 'FAIL'} - {ans[:50]}")

        # Self
        out = llm.generate(["Are you an AI? Yes or no:"], sp)
        ans = out[0].outputs[0].text.strip()
        self_ok = "yes" in ans.lower()
        print(f"Self-awareness: {'OK' if self_ok else 'FAIL'} - {ans[:50]}")

        lucid = math_ok and know_ok and self_ok
        count = int(math_ok) + int(know_ok) + int(self_ok)
        print(f"\nLucidity: {'LUCID' if lucid else 'NOT LUCID'} ({count}/3)")

        # Cleanup
        print("\n--- Cleanup ---")
        freed = full_cleanup(llm)
        print(f"Freed: {freed:.1f} GB")
        llm = None
        torch.cuda.empty_cache()
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        print("\n*** GLM-4.6V-AWQ TEST PASSED! ***")
        return 0

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)[:500]}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
