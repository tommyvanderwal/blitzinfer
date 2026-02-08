#!/usr/bin/env python3
"""Test if warmup can be skipped entirely."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'  # Skip deep gemm warmup

import sys
import types

# Patch torchvision._meta_registrations
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta

sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import time
import gc


def patch_skip_warmup():
    """Patch vLLM to skip warmup phases."""
    print("Patching vLLM to skip warmup...")

    # Patch profile_run to be a no-op
    from vllm.v1.worker.gpu import model_runner

    original_profile_run = model_runner.GPUModelRunner.profile_run
    def skip_profile_run(self):
        print("    [SKIPPED] profile_run")
        # Still need to sync device to ensure model is loaded
        import torch
        torch.cuda.synchronize()
    model_runner.GPUModelRunner.profile_run = skip_profile_run

    # Patch kernel_warmup to be a no-op
    from vllm.model_executor.warmup import kernel_warmup as kw_module
    original_kernel_warmup = kw_module.kernel_warmup
    def skip_kernel_warmup(worker):
        print("    [SKIPPED] kernel_warmup")
    kw_module.kernel_warmup = skip_kernel_warmup

    # Patch compile_or_warm_up_model to skip most work
    from vllm.v1.worker import gpu_worker
    original_compile_warmup = gpu_worker.Worker.compile_or_warm_up_model
    def minimal_compile_warmup(self):
        print("    [SKIPPED] compile_or_warm_up_model (keeping sampler init)")
        # Only do the essential sampler warmup for the last rank
        from vllm.distributed import get_pp_group
        if get_pp_group().is_last_rank:
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )
            # Minimal dummy run for sampler
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
            )
            self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)
        from vllm.utils import set_random_seed
        set_random_seed(self.model_config.seed)
    # Keep original for now to see if first request works
    # gpu_worker.Worker.compile_or_warm_up_model = minimal_compile_warmup

    print("Patches applied.")


def test_no_warmup():
    """Test startup with warmup skipped."""
    print("\n" + "="*70)
    print("TEST: SKIP WARMUP")
    print("="*70)

    import torch
    torch.cuda.empty_cache()
    gc.collect()

    # Apply patches before importing LLM
    patch_skip_warmup()

    from vllm import LLM, SamplingParams

    kv_bytes = 4 * 1024 * 1024 * 1024

    print(f"\n[{time.strftime('%H:%M:%S')}] Starting LLM init (warmup skipped)...")
    start = time.perf_counter()

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        kv_cache_memory_bytes=kv_bytes,
        enforce_eager=True,
    )

    init_time = time.perf_counter() - start
    print(f"[{time.strftime('%H:%M:%S')}] LLM init complete: {init_time:.2f}s")

    # First inference - this will be the "warmup"
    print(f"\n[{time.strftime('%H:%M:%S')}] First inference (acting as warmup)...")
    start = time.perf_counter()
    try:
        out = llm.generate(["Hello, how are you?"], SamplingParams(max_tokens=20))
        first_inf = time.perf_counter() - start
        print(f"[{time.strftime('%H:%M:%S')}] First inference: {first_inf:.2f}s")
        print(f"  Output: {out[0].outputs[0].text}")
    except Exception as e:
        first_inf = time.perf_counter() - start
        print(f"[{time.strftime('%H:%M:%S')}] First inference FAILED: {e}")
        first_inf = -1

    # Second inference
    print(f"\n[{time.strftime('%H:%M:%S')}] Second inference...")
    start = time.perf_counter()
    try:
        out = llm.generate(["Tell me a joke"], SamplingParams(max_tokens=20))
        second_inf = time.perf_counter() - start
        print(f"[{time.strftime('%H:%M:%S')}] Second inference: {second_inf:.2f}s")
        print(f"  Output: {out[0].outputs[0].text}")
    except Exception as e:
        second_inf = time.perf_counter() - start
        print(f"[{time.strftime('%H:%M:%S')}] Second inference FAILED: {e}")
        second_inf = -1

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Init time:        {init_time:.2f}s")
    print(f"  First inference:  {first_inf:.2f}s")
    print(f"  Second inference: {second_inf:.2f}s")
    print(f"  Total to first response: {init_time + first_inf:.2f}s")
    print("="*70)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return init_time, first_inf, second_inf


if __name__ == '__main__':
    test_no_warmup()
