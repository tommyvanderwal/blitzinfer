#!/usr/bin/env python3
"""Profile vLLM startup with hardware monitoring (rocm-smi + top)."""

import os
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import sys
import types
import subprocess
import threading
import time
import gc
from datetime import datetime

# Patch torchvision._meta_registrations before import
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta

sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')


class HardwareMonitor:
    """Monitor GPU and CPU usage during profiling."""

    def __init__(self, interval=2.0):
        self.interval = interval
        self.running = False
        self.thread = None
        self.samples = []

    def _sample(self):
        """Collect one sample of hardware metrics."""
        timestamp = datetime.now().strftime('%H:%M:%S')

        # ROCm-SMI data
        try:
            result = subprocess.run(
                ['rocm-smi', '--showuse', '--showmemuse', '--showpower'],
                capture_output=True, text=True, timeout=5
            )
            rocm_output = result.stdout
        except Exception as e:
            rocm_output = f"Error: {e}"

        # CPU/Memory from top (single snapshot)
        try:
            result = subprocess.run(
                ['top', '-bn1', '-w', '120'],
                capture_output=True, text=True, timeout=5
            )
            # Extract just the summary lines and python processes
            lines = result.stdout.split('\n')
            top_summary = '\n'.join(lines[:7])  # First 7 lines have summary
            python_procs = [l for l in lines if 'python' in l.lower()][:5]
            top_output = top_summary + '\n' + '\n'.join(python_procs)
        except Exception as e:
            top_output = f"Error: {e}"

        return {
            'timestamp': timestamp,
            'rocm_smi': rocm_output,
            'top': top_output
        }

    def _monitor_loop(self):
        """Background monitoring thread."""
        while self.running:
            sample = self._sample()
            self.samples.append(sample)
            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"[HW Monitor] Started (sampling every {self.interval}s)")

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(f"[HW Monitor] Stopped ({len(self.samples)} samples collected)")

    def print_report(self):
        """Print hardware usage report."""
        print("\n" + "="*80)
        print("HARDWARE USAGE TIMELINE")
        print("="*80)

        for i, sample in enumerate(self.samples):
            print(f"\n--- Sample {i+1} @ {sample['timestamp']} ---")

            # Parse ROCm-SMI output for key metrics
            rocm_lines = sample['rocm_smi'].split('\n')
            for line in rocm_lines:
                if any(k in line.lower() for k in ['gpu', 'use', 'mem', 'power', 'gfx']):
                    print(f"  GPU: {line.strip()}")

            # Extract CPU usage from top
            top_lines = sample['top'].split('\n')
            for line in top_lines:
                if line.startswith('%Cpu') or 'MiB Mem' in line or 'python' in line.lower():
                    print(f"  CPU: {line.strip()}")


def profile_with_monitoring():
    """Profile LLM startup with hardware monitoring."""
    print("\n" + "="*80)
    print("VLLM STARTUP PROFILING WITH HARDWARE MONITORING")
    print("="*80)

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    torch.cuda.empty_cache()
    gc.collect()

    # Start hardware monitor
    monitor = HardwareMonitor(interval=2.0)
    monitor.start()

    from vllm import LLM, SamplingParams

    print(f"\n[{time.strftime('%H:%M:%S')}] Starting LLM initialization...")
    init_start = time.perf_counter()

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
        gpu_memory_utilization=0.40,
        max_model_len=1024,
        enforce_eager=True,
    )

    init_time = time.perf_counter() - init_start
    print(f"[{time.strftime('%H:%M:%S')}] LLM init complete: {init_time:.2f}s")

    # First inference
    print(f"[{time.strftime('%H:%M:%S')}] First inference...")
    inf_start = time.perf_counter()
    out = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    inf_time = time.perf_counter() - inf_start
    print(f"[{time.strftime('%H:%M:%S')}] First inference: {inf_time:.2f}s")
    print(f"  Output: {out[0].outputs[0].text}")

    # Stop monitor
    monitor.stop()

    # Print hardware report
    monitor.print_report()

    # Summary
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    print(f"  LLM initialization: {init_time:.2f}s")
    print(f"  First inference:    {inf_time:.2f}s")
    print(f"  Total:              {init_time + inf_time:.2f}s")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return monitor.samples


def analyze_samples(samples):
    """Analyze hardware samples to identify bottlenecks."""
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)

    gpu_busy = 0
    cpu_busy = 0
    io_wait = 0

    for sample in samples:
        # Parse GPU utilization
        for line in sample['rocm_smi'].split('\n'):
            if 'GPU use' in line and '%' in line:
                try:
                    pct = int(line.split('%')[0].split()[-1])
                    if pct > 50:
                        gpu_busy += 1
                except:
                    pass

        # Parse CPU utilization from top
        for line in sample['top'].split('\n'):
            if line.startswith('%Cpu'):
                try:
                    # Parse: %Cpu(s): 12.5 us, 3.1 sy, 0.0 ni, 84.2 id, 0.2 wa
                    parts = line.split(',')
                    idle = float([p for p in parts if 'id' in p][0].split()[0])
                    wa = float([p for p in parts if 'wa' in p][0].split()[0])
                    if 100 - idle > 30:
                        cpu_busy += 1
                    if wa > 10:
                        io_wait += 1
                except:
                    pass

    total = len(samples)
    print(f"  Samples with GPU busy (>50%): {gpu_busy}/{total}")
    print(f"  Samples with CPU busy (>30%): {cpu_busy}/{total}")
    print(f"  Samples with I/O wait (>10%): {io_wait}/{total}")

    if io_wait > gpu_busy and io_wait > cpu_busy:
        print("\n  BOTTLENECK: I/O (disk or network)")
    elif gpu_busy >= cpu_busy:
        print("\n  BOTTLENECK: GPU computation")
    else:
        print("\n  BOTTLENECK: CPU computation or synchronization")


if __name__ == '__main__':
    samples = profile_with_monitoring()
    analyze_samples(samples)
