#!/usr/bin/env python3
"""Profile GPU memory to find what holds memory after vLLM unload."""
import os
import gc
import time
import subprocess
import multiprocessing

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'  # Single-process mode with proper cleanup

import torch

def gpu_mem():
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        'free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3,
    }

def nvidia_smi_processes():
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def main():
    print('=' * 60)
    print('GPU MEMORY PROFILING')
    print('=' * 60)

    print('\n=== INITIAL STATE ===')
    mem = gpu_mem()
    print(f'Allocated: {mem["allocated_gb"]:.2f} GB')
    print(f'Reserved:  {mem["reserved_gb"]:.2f} GB')
    print(f'Free:      {mem["free_gb"]:.2f} GB')
    print(f'\nProcesses using GPU:\n{nvidia_smi_processes() or "(none)"}')

    print('\n=== LOADING MODEL ===')
    from vllm import LLM, SamplingParams

    config = {
        'dtype': 'bfloat16',
        'max_model_len': 100000,
        'gpu_memory_utilization': 0.95,
        'max_num_seqs': 16,
        'max_num_batched_tokens': 8192,
        'enforce_eager': True,
        'trust_remote_code': True,
    }

    t0 = time.time()
    llm = LLM(model='openai/gpt-oss-120b', **config)
    load_time = time.time() - t0
    print(f'Model loaded in {load_time:.1f}s')

    mem = gpu_mem()
    print(f'Allocated: {mem["allocated_gb"]:.2f} GB')
    print(f'Reserved:  {mem["reserved_gb"]:.2f} GB')
    print(f'Free:      {mem["free_gb"]:.2f} GB')
    print(f'\nProcesses using GPU:\n{nvidia_smi_processes()}')

    # Do inference
    print('\n=== INFERENCE ===')
    out = llm.generate(['Hello, I am'], SamplingParams(max_tokens=20))
    print(f'Output: {out[0].outputs[0].text[:50]}...')

    mem = gpu_mem()
    print(f'Allocated: {mem["allocated_gb"]:.2f} GB')
    print(f'Reserved:  {mem["reserved_gb"]:.2f} GB')

    print('\n' + '=' * 60)
    print('UNLOAD SEQUENCE')
    print('=' * 60)

    # Step 1: Check children before unload
    print('\n=== STEP 1: Check children before unload ===')
    children = multiprocessing.active_children()
    print(f'Active children: {len(children)}')
    for child in children:
        print(f'  - {child.name} (pid={child.pid}, alive={child.is_alive()})')

    # Step 2: Try shutdown method
    print('\n=== STEP 2: Call llm.llm_engine.shutdown() ===')
    try:
        llm.llm_engine.shutdown()
        print('Shutdown called successfully')
    except Exception as e:
        print(f'Shutdown error: {e}')

    time.sleep(2)
    mem = gpu_mem()
    print(f'Allocated: {mem["allocated_gb"]:.2f} GB')
    print(f'Reserved:  {mem["reserved_gb"]:.2f} GB')
    print(f'Processes using GPU:\n{nvidia_smi_processes() or "(none)"}')

    # Step 3: Delete LLM object
    print('\n=== STEP 3: del llm ===')
    del llm
    gc.collect()
    time.sleep(2)

    mem = gpu_mem()
    print(f'Allocated: {mem["allocated_gb"]:.2f} GB')
    print(f'Reserved:  {mem["reserved_gb"]:.2f} GB')
    print(f'Processes using GPU:\n{nvidia_smi_processes() or "(none)"}')

    # Step 4: Check children after del
    print('\n=== STEP 4: Check children after del ===')
    children = multiprocessing.active_children()
    print(f'Active children: {len(children)}')
    for child in children:
        print(f'  - {child.name} (pid={child.pid}, alive={child.is_alive()})')

    # Step 5: Terminate children
    if children:
        print('\n=== STEP 5: Terminate remaining children ===')
        for child in children:
            print(f'Terminating {child.name} (pid={child.pid})...')
            child.terminate()
            child.join(timeout=10)
            if child.is_alive():
                print(f'  Force killing {child.pid}...')
                child.kill()
                child.join(timeout=5)

        time.sleep(2)
        mem = gpu_mem()
        print(f'Allocated: {mem["allocated_gb"]:.2f} GB')
        print(f'Reserved:  {mem["reserved_gb"]:.2f} GB')
        print(f'Processes using GPU:\n{nvidia_smi_processes() or "(none)"}')

    # Step 6: torch.cuda.empty_cache()
    print('\n=== STEP 6: torch.cuda.empty_cache() ===')
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    mem = gpu_mem()
    print(f'Allocated: {mem["allocated_gb"]:.2f} GB')
    print(f'Reserved:  {mem["reserved_gb"]:.2f} GB')
    print(f'Processes using GPU:\n{nvidia_smi_processes() or "(none)"}')

    # Step 7: Check what's holding memory
    print('\n=== STEP 7: Memory snapshot ===')
    if mem['allocated_gb'] > 1.0:
        print('WARNING: Significant memory still allocated!')
        print('Checking torch memory stats...')
        print(torch.cuda.memory_summary(abbreviated=True))

    print('\n' + '=' * 60)
    print('FINAL STATE')
    print('=' * 60)
    mem = gpu_mem()
    print(f'Allocated: {mem["allocated_gb"]:.2f} GB')
    print(f'Reserved:  {mem["reserved_gb"]:.2f} GB')
    print(f'Free:      {mem["free_gb"]:.2f} GB')
    print(f'\nProcesses using GPU:\n{nvidia_smi_processes() or "(none)"}')

    if mem['free_gb'] < 90:
        print(f'\n*** PROBLEM: Only {mem["free_gb"]:.1f} GB free, expected ~95 GB ***')
    else:
        print(f'\n*** SUCCESS: {mem["free_gb"]:.1f} GB free ***')


if __name__ == '__main__':
    main()
