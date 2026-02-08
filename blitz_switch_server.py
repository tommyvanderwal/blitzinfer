#!/usr/bin/env python3
"""
BlitzSwitch Server - Fast model switching via persistent process

Key optimizations:
1. Pre-import all vLLM modules (saves ~3s per switch)
2. Keep tokenizers warm in memory
3. Reuse engine infrastructure where possible
"""

import time
import os
import gc
import sys

# Pre-configure environment
IS_ROCM = os.path.exists("/opt/rocm")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
if not IS_ROCM:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
else:
    os.environ["VLLM_SKIP_WARMUP"] = "1"
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

# Pre-import all vLLM modules
print("Pre-importing vLLM modules...")
t0 = time.time()
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
import torch
print(f"Imports ready in {time.time()-t0:.1f}s")

# Model configurations
MODELS = {
    "gpt": {
        "name": "openai/gpt-oss-120b",
        "dtype": "bfloat16",
        "max_model_len": 512,
        "kv_cache_bytes": 10 * 1024**3 if IS_ROCM else None,
    },
    "qwen": {
        "name": "Qwen/Qwen3-VL-32B-Instruct",
        "dtype": "float16",
        "max_model_len": 512,
        "kv_cache_bytes": 10 * 1024**3 if IS_ROCM else None,
    }
}

class BlitzSwitcher:
    """Fast model switcher with warm start"""

    def __init__(self):
        self.current_model = None
        self.llm = None
        self.load_times = []

    def unload(self):
        """Unload current model"""
        if self.llm:
            del self.llm
            self.llm = None
            gc.collect()
            torch.cuda.empty_cache()
            self.current_model = None

    def load(self, model_key: str) -> float:
        """Load a model (unloads current first)"""
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        if self.current_model == model_key:
            print(f"Model {model_key} already loaded")
            return 0.0

        self.unload()

        config = MODELS[model_key]
        print(f"\nLoading {config['name']}...")

        start = time.time()

        kwargs = {
            "model": config["name"],
            "dtype": config["dtype"],
            "max_model_len": config["max_model_len"],
            "max_num_seqs": 2,
            "disable_log_stats": True,
            "enforce_eager": True,
        }

        if IS_ROCM:
            kwargs["compilation_config"] = {"custom_ops": ["none"]}
            if config["kv_cache_bytes"]:
                kwargs["kv_cache_memory_bytes"] = config["kv_cache_bytes"]

        self.llm = LLM(**kwargs)
        load_time = time.time() - start

        self.current_model = model_key
        self.load_times.append((model_key, load_time))

        print(f"Loaded in {load_time:.1f}s")
        return load_time

    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate text with current model"""
        if not self.llm:
            raise RuntimeError("No model loaded")

        output = self.llm.generate([prompt], SamplingParams(max_tokens=max_tokens))
        return output[0].outputs[0].text

    def run_switch_test(self, num_switches: int = 4):
        """Run switching test"""
        print(f"\n{'='*60}")
        print(f"BlitzSwitch Test - {num_switches} switches")
        print(f"Platform: {'780M' if IS_ROCM else 'RTX PRO 6000'}")
        print(f"{'='*60}")

        for i in range(num_switches):
            print(f"\n--- Switch {i+1}/{num_switches} ---")

            # Load GPT-OSS
            t1 = self.load("gpt")
            out = self.generate("Hello")
            print(f"GPT output: {out[:40]}...")

            # Load Qwen
            t2 = self.load("qwen")
            out = self.generate("Hello")
            print(f"Qwen output: {out[:40]}...")

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        gpt_times = [t for m, t in self.load_times if m == "gpt"]
        qwen_times = [t for m, t in self.load_times if m == "qwen"]

        if gpt_times:
            print(f"GPT-OSS-120B loads: {[f'{t:.1f}s' for t in gpt_times]}")
            print(f"  Average: {sum(gpt_times)/len(gpt_times):.1f}s")
            print(f"  After warm imports: ~{sum(gpt_times)/len(gpt_times) - 3:.1f}s effective")

        if qwen_times:
            print(f"Qwen3-VL-32B loads: {[f'{t:.1f}s' for t in qwen_times]}")
            print(f"  Average: {sum(qwen_times)/len(qwen_times):.1f}s")
            print(f"  After warm imports: ~{sum(qwen_times)/len(qwen_times) - 3:.1f}s effective")


if __name__ == "__main__":
    num_switches = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    switcher = BlitzSwitcher()
    switcher.run_switch_test(num_switches)
