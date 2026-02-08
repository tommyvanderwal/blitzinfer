#!/usr/bin/env python3
"""
Multi-model GPU switching: Keep multiple models loaded, instant switch.
Target: Sub-1s model switching by avoiding weight loading entirely.
"""

import os
import sys
import gc
import time
import types

os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_SKIP_WARMUP'] = '1'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch


def get_mem():
    return torch.cuda.mem_get_info()[0] / (1024**3)


def test_dual_model():
    """Test loading two models simultaneously."""
    print("=" * 70)
    print("DUAL MODEL GPU TEST")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    initial = get_mem()
    print(f"Initial GPU: {initial:.2f} GB")

    # Can we fit two 7B models?
    # Each model: ~14GB weights + 2GB KV = ~16GB
    # Two models: ~32GB
    # Available: ~63GB - should fit!

    print("\n>>> Loading first model...")
    t0 = time.perf_counter()
    llm1 = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.25,  # Lower to leave room for second model
        max_model_len=512,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=1 * 1024**3,  # 1GB KV cache
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
    )
    load1 = (time.perf_counter() - t0) * 1000
    after1 = get_mem()
    print(f"Model 1 loaded: {load1:.0f}ms")
    print(f"GPU: {after1:.2f} GB (used {initial - after1:.2f} GB)")

    # Test first model
    out1 = llm1.generate(["Who are you?"], SamplingParams(max_tokens=20))
    print(f"Model 1 response: {out1[0].outputs[0].text}")

    print("\n>>> Loading second model...")
    t0 = time.perf_counter()
    try:
        llm2 = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",  # Same model for testing
            dtype="float16",
            gpu_memory_utilization=0.25,
            max_model_len=512,
            max_num_batched_tokens=512,
            kv_cache_memory_bytes=1 * 1024**3,
            enforce_eager=True,
            compilation_config={"custom_ops": ["none"]},
        )
        load2 = (time.perf_counter() - t0) * 1000
        after2 = get_mem()
        print(f"Model 2 loaded: {load2:.0f}ms")
        print(f"GPU: {after2:.2f} GB (used {initial - after2:.2f} GB)")

        # Test second model
        out2 = llm2.generate(["What is 2+2?"], SamplingParams(max_tokens=20))
        print(f"Model 2 response: {out2[0].outputs[0].text}")

        # Now test switching between them
        print("\n>>> Testing rapid switching...")

        for i in range(3):
            t0 = time.perf_counter()
            out1 = llm1.generate([f"Count to {i+1}"], SamplingParams(max_tokens=10))
            t1 = time.perf_counter()
            out2 = llm2.generate([f"What is {i+1}+{i+1}?"], SamplingParams(max_tokens=10))
            t2 = time.perf_counter()

            print(f"  Round {i+1}: LLM1={t1-t0:.2f}s, LLM2={t2-t1:.2f}s")
            print(f"    LLM1: {out1[0].outputs[0].text[:30]}")
            print(f"    LLM2: {out2[0].outputs[0].text[:30]}")

        print("\nSUCCESS! Both models running simultaneously")
        print(f"Total GPU used: {initial - get_mem():.2f} GB")

        del llm1, llm2

    except Exception as e:
        print(f"FAILED to load second model: {e}")
        del llm1

    gc.collect()
    torch.cuda.empty_cache()


def test_model_pool():
    """Test a pool of pre-loaded models for instant switching."""
    print("\n" + "=" * 70)
    print("MODEL POOL TEST")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    class ModelPool:
        def __init__(self):
            self.models = {}
            self.active = None

        def preload(self, name, model_id, config):
            print(f"Pre-loading {name}...")
            t0 = time.perf_counter()
            self.models[name] = LLM(model=model_id, **config)
            print(f"  Loaded in {(time.perf_counter() - t0)*1000:.0f}ms")
            print(f"  GPU: {get_mem():.2f} GB")

        def switch(self, name):
            t0 = time.perf_counter()
            self.active = self.models[name]
            switch_time = (time.perf_counter() - t0) * 1000
            print(f"Switched to {name} in {switch_time:.2f}ms")
            return self.active

        def generate(self, prompts, params):
            return self.active.generate(prompts, params)

    config = {
        "dtype": "float16",
        "gpu_memory_utilization": 0.20,  # 20% each = 40% total for 2 models
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "kv_cache_memory_bytes": 1 * 1024**3,
        "enforce_eager": True,
        "compilation_config": {"custom_ops": ["none"]},
    }

    initial = get_mem()
    print(f"Initial GPU: {initial:.2f} GB")

    pool = ModelPool()

    # Pre-load models
    try:
        pool.preload("qwen", "Qwen/Qwen2.5-7B-Instruct", config)
        pool.preload("qwen2", "Qwen/Qwen2.5-7B-Instruct", config)  # Same for testing

        print(f"\nAll models loaded. GPU: {get_mem():.2f} GB")
        print(f"Total used: {initial - get_mem():.2f} GB")

        params = SamplingParams(max_tokens=20)

        # Test instant switching
        print("\n>>> Testing instant switching...")

        for i in range(5):
            model = "qwen" if i % 2 == 0 else "qwen2"
            t0 = time.perf_counter()

            llm = pool.switch(model)
            switch_time = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            out = llm.generate([f"Say {i}"], params)
            gen_time = (time.perf_counter() - t1) * 1000

            print(f"  {model}: switch={switch_time:.2f}ms, gen={gen_time:.0f}ms")
            print(f"    Output: {out[0].outputs[0].text[:40]}")

        print("\n" + "=" * 70)
        print("RESULT: INSTANT SWITCHING ACHIEVED")
        print("=" * 70)
        print(f"Switch time: <1ms (just pointer assignment)")
        print(f"GPU used: {initial - get_mem():.2f} GB for 2 models")

    except Exception as e:
        print(f"FAILED: {e}")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_dual_model()
    test_model_pool()
