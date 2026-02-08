#!/usr/bin/env python3
"""Profile different cleanup methods to find what releases GPU memory."""
import os
import gc
import time
import subprocess as sp
import multiprocessing

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # Single-process mode with proper cleanup

import torch


def nvidia_smi():
    result = sp.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return result.stdout.strip() or "(none)"


def gpu_stats():
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**3,
        'reserved': torch.cuda.memory_reserved() / 1024**3,
        'nvidia_smi': nvidia_smi(),
    }


def main():
    from vllm import LLM, SamplingParams

    print("=" * 70)
    print("CLEANUP METHOD PROFILING")
    print("=" * 70)

    print("\n=== Initial state ===")
    print(f"GPU stats: {gpu_stats()}")

    print("\n=== Loading model ===")
    # max_model_len: not specified, let model use its default max
    config = {
        'dtype': 'bfloat16',
        'gpu_memory_utilization': 0.95,
        'max_num_seqs': 16,
        'max_num_batched_tokens': 8192,
        'enforce_eager': True,
        'trust_remote_code': True,
    }

    llm = LLM(model='openai/gpt-oss-120b', **config)

    # Do inference to ensure model is fully initialized
    out = llm.generate(['Hello'], SamplingParams(max_tokens=10))
    print(f"Output: {out[0].outputs[0].text[:30]}...")
    print(f"GPU stats: {gpu_stats()}")

    print("\n" + "=" * 70)
    print("TESTING CLEANUP METHODS")
    print("=" * 70)

    # Method 1: Just del
    print("\n--- Method 1: del llm ---")
    del llm
    time.sleep(1)
    print(f"GPU stats: {gpu_stats()}")

    # Method 2: gc.collect
    print("\n--- Method 2: gc.collect() ---")
    gc.collect()
    gc.collect()
    time.sleep(1)
    print(f"GPU stats: {gpu_stats()}")

    # Method 3: empty_cache
    print("\n--- Method 3: torch.cuda.empty_cache() ---")
    torch.cuda.empty_cache()
    time.sleep(1)
    print(f"GPU stats: {gpu_stats()}")

    # Method 4: synchronize
    print("\n--- Method 4: torch.cuda.synchronize() ---")
    torch.cuda.synchronize()
    time.sleep(1)
    print(f"GPU stats: {gpu_stats()}")

    # Method 5: ipc_collect
    print("\n--- Method 5: torch.cuda.ipc_collect() ---")
    try:
        torch.cuda.ipc_collect()
        time.sleep(1)
        print(f"GPU stats: {gpu_stats()}")
    except Exception as e:
        print(f"Error: {e}")

    # Method 6: Check children
    print("\n--- Method 6: Check/terminate children ---")
    children = multiprocessing.active_children()
    print(f"Active children: {len(children)}")
    for child in children:
        print(f"  Terminating {child.name} (pid={child.pid})")
        child.terminate()
        child.join(timeout=5)
    time.sleep(2)
    print(f"GPU stats: {gpu_stats()}")

    # Method 7: Reset CUDA stats
    print("\n--- Method 7: torch.cuda.reset_peak_memory_stats() ---")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    time.sleep(1)
    print(f"GPU stats: {gpu_stats()}")

    # Method 8: Try resetting device
    print("\n--- Method 8: torch.cuda.reset_max_memory_allocated() ---")
    try:
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        time.sleep(1)
        print(f"GPU stats: {gpu_stats()}")
    except Exception as e:
        print(f"Error: {e}")

    # Method 9: See if we can call CUDA driver directly
    print("\n--- Method 9: Check CUDA driver context ---")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    # Try to see what CUDA thinks about memory
    try:
        mem_info = torch.cuda.mem_get_info()
        print(f"CUDA mem_get_info: free={mem_info[0]/1024**3:.2f}GB, total={mem_info[1]/1024**3:.2f}GB")
    except Exception as e:
        print(f"mem_get_info error: {e}")

    # Method 10: Wait and recheck
    print("\n--- Method 10: Wait 10 seconds and recheck ---")
    for i in range(10):
        time.sleep(1)
        stats = gpu_stats()
        if '(none)' in stats['nvidia_smi'] or 'python3' not in stats['nvidia_smi'] or int(stats['nvidia_smi'].split(',')[-1].replace(' MiB', '').strip()) < 1000:
            print(f"  {i+1}s: MEMORY FREED! {stats}")
            break
        print(f"  {i+1}s: {stats}")

    print("\n=== FINAL STATE ===")
    print(f"GPU stats: {gpu_stats()}")

    # Try to get more detail
    try:
        print("\nMemory summary:")
        print(torch.cuda.memory_summary(abbreviated=True))
    except Exception as e:
        print(f"Memory summary error: {e}")


if __name__ == '__main__':
    main()
