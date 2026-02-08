#!/usr/bin/env python3
"""
Large Model Swap Test - Profile swapping between 120B and 46B models.

Tests:
- openai/gpt-oss-120b (~112GB, MXFP4)
- cyankiwi/GLM-4.6V-AWQ-4bit (~19.5GB, compressed-tensors)

Profiles:
- GPU memory (allocated, reserved, used)
- System memory (RAM, swap)
- Disk I/O throughput during loading
- Load time breakdown
- Memory pressure effects (cold loads from SSD vs warm from RAM cache)
"""

import gc
import os
import sys
import time
import types
import json
import threading
from dataclasses import dataclass, field
from typing import Optional

# Environment setup
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

# Fake torchvision module
fake_meta = types.ModuleType('torchvision._meta_registrations')
sys.modules['torchvision._meta_registrations'] = fake_meta
sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

import torch
import psutil


@dataclass
class MemorySnapshot:
    """Snapshot of all memory metrics."""
    timestamp: float
    label: str
    # GPU
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_used_gb: float = 0.0
    gpu_free_gb: float = 0.0
    # System
    sys_used_gb: float = 0.0
    sys_available_gb: float = 0.0
    sys_cached_gb: float = 0.0
    sys_buffers_gb: float = 0.0
    # Swap
    swap_used_gb: float = 0.0
    # Process
    proc_rss_gb: float = 0.0
    proc_vms_gb: float = 0.0


@dataclass
class DiskIOSnapshot:
    """Snapshot of disk I/O counters."""
    timestamp: float
    read_bytes: int = 0
    write_bytes: int = 0
    read_count: int = 0
    write_count: int = 0


@dataclass
class LoadProfile:
    """Profile data for a model load."""
    switch_num: int
    model: str
    model_short: str
    success: bool
    error: Optional[str] = None
    # Timing
    total_load_time: float = 0.0
    weight_load_time: float = 0.0
    kv_cache_time: float = 0.0
    warmup_time: float = 0.0
    # Memory snapshots
    before_load: Optional[MemorySnapshot] = None
    after_load: Optional[MemorySnapshot] = None
    after_cleanup: Optional[MemorySnapshot] = None
    # Disk I/O
    disk_read_gb: float = 0.0
    disk_read_speed_gbps: float = 0.0
    # Inference
    inference_time: float = 0.0
    tokens_generated: int = 0


def take_memory_snapshot(label: str) -> MemorySnapshot:
    """Take a comprehensive memory snapshot."""
    snap = MemorySnapshot(timestamp=time.time(), label=label)

    # GPU memory
    if torch.cuda.is_available():
        snap.gpu_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        snap.gpu_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        free, total = torch.cuda.mem_get_info()
        snap.gpu_used_gb = (total - free) / (1024**3)
        snap.gpu_free_gb = free / (1024**3)

    # System memory
    vm = psutil.virtual_memory()
    snap.sys_used_gb = vm.used / (1024**3)
    snap.sys_available_gb = vm.available / (1024**3)
    snap.sys_cached_gb = getattr(vm, 'cached', 0) / (1024**3)
    snap.sys_buffers_gb = getattr(vm, 'buffers', 0) / (1024**3)

    # Swap
    swap = psutil.swap_memory()
    snap.swap_used_gb = swap.used / (1024**3)

    # Process
    proc = psutil.Process()
    mem_info = proc.memory_info()
    snap.proc_rss_gb = mem_info.rss / (1024**3)
    snap.proc_vms_gb = mem_info.vms / (1024**3)

    return snap


def take_disk_io_snapshot() -> DiskIOSnapshot:
    """Take a disk I/O snapshot."""
    io = psutil.disk_io_counters()
    return DiskIOSnapshot(
        timestamp=time.time(),
        read_bytes=io.read_bytes,
        write_bytes=io.write_bytes,
        read_count=io.read_count,
        write_count=io.write_count,
    )


class DiskIOMonitor:
    """Monitor disk I/O in a background thread."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples: list[DiskIOSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self.samples = []
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float]:
        """Stop monitoring and return (total_read_gb, avg_speed_gbps)."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        if len(self.samples) < 2:
            return 0.0, 0.0

        first, last = self.samples[0], self.samples[-1]
        total_read = (last.read_bytes - first.read_bytes) / (1024**3)
        duration = last.timestamp - first.timestamp
        speed = total_read / duration if duration > 0 else 0

        return total_read, speed

    def _monitor(self):
        while self._running:
            self.samples.append(take_disk_io_snapshot())
            time.sleep(self.interval)


def comprehensive_cleanup(llm):
    """Full cleanup after model unload."""
    import torch._dynamo
    import multiprocessing

    try:
        engine_core = llm.llm_engine.engine_core
        if hasattr(engine_core, 'engine_core'):
            core = engine_core.engine_core
        else:
            core = engine_core

        if hasattr(core, 'model_executor'):
            executor = core.model_executor
            if hasattr(executor, 'driver_worker'):
                worker = executor.driver_worker
                if hasattr(worker, 'worker') and worker.worker is not None:
                    model_runner = getattr(worker.worker, 'model_runner', None)
                    if model_runner is not None:
                        if hasattr(model_runner, 'model'):
                            for param in model_runner.model.parameters():
                                param.data = torch.empty(0, device='cpu')

                        if hasattr(model_runner, 'kv_caches'):
                            for i, cache in enumerate(model_runner.kv_caches):
                                if cache is not None and hasattr(cache, 'device') and cache.device.type == 'cuda':
                                    model_runner.kv_caches[i] = torch.empty(0, device='cpu')
                            model_runner.kv_caches.clear()

                        if hasattr(model_runner, 'compilation_config'):
                            sfc = getattr(model_runner.compilation_config, 'static_forward_context', None)
                            if sfc:
                                for layer in sfc.values():
                                    if hasattr(layer, 'kv_cache') and layer.kv_cache:
                                        for j, kv in enumerate(layer.kv_cache):
                                            if kv is not None and hasattr(kv, 'device') and kv.device.type == 'cuda':
                                                layer.kv_cache[j] = torch.empty(0, device='cpu')
                                        layer.kv_cache = []
                        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Cleanup error: {e}")

    del llm

    try:
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            vllm_config.compilation_config.static_forward_context.clear()
    except:
        pass

    try:
        import vllm.config.vllm as vllm_config_module
        vllm_config_module._current_vllm_config = None
        vllm_config_module._current_prefix = None
        vllm_config_module.get_cached_compilation_config.cache_clear()
    except:
        pass

    try:
        torch._dynamo.reset()
    except:
        pass

    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory(shutdown_ray=False)
    except:
        pass

    try:
        from vllm.v1.worker.workspace import reset_workspace_manager
        reset_workspace_manager()
    except:
        pass

    for child in multiprocessing.active_children():
        child.join(timeout=5.0)
        if child.is_alive():
            child.terminate()
            child.join(timeout=2.0)

    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def drop_caches():
    """Drop system page cache to force cold loads from SSD."""
    try:
        # Sync first
        os.sync()
        # Drop caches (requires root or appropriate permissions)
        with open('/proc/sys/vm/drop_caches', 'w') as f:
            f.write('3')
        print("  Dropped system caches")
        return True
    except PermissionError:
        print("  Could not drop caches (need sudo)")
        return False
    except Exception as e:
        print(f"  Could not drop caches: {e}")
        return False


def print_memory_summary(label: str, snap: MemorySnapshot):
    """Print a memory snapshot summary."""
    print(f"  {label}:")
    print(f"    GPU: alloc={snap.gpu_allocated_gb:.1f}GB resv={snap.gpu_reserved_gb:.1f}GB used={snap.gpu_used_gb:.1f}GB free={snap.gpu_free_gb:.1f}GB")
    print(f"    SYS: used={snap.sys_used_gb:.1f}GB avail={snap.sys_available_gb:.1f}GB cache={snap.sys_cached_gb:.1f}GB")
    print(f"    SWAP: {snap.swap_used_gb:.1f}GB | PROC: rss={snap.proc_rss_gb:.1f}GB")


def main():
    from vllm import LLM, SamplingParams

    print("="*80)
    print("LARGE MODEL SWAP TEST")
    print("="*80)
    print("Models:")
    print("  A: openai/gpt-oss-120b (~112GB, MXFP4)")
    print("  B: cyankiwi/GLM-4.6V-AWQ-4bit (~19.5GB, compressed-tensors)")
    print()

    # Initial state
    gc.collect()
    torch.randn(1000, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    initial_snap = take_memory_snapshot("initial")
    print_memory_summary("Initial state", initial_snap)

    # Model configs - use most of the 96GB VRAM
    # GPT-OSS-120B needs ~112GB for weights + KV cache
    # With 96GB unified memory, we need to be strategic

    models = {
        "gpt-oss-120b": {
            "name": "openai/gpt-oss-120b",
            "short": "GPT-OSS-120B",
            "config": {
                "dtype": "auto",  # MXFP4 will use its native dtype
                "gpu_memory_utilization": 0.90,  # Use 90% of available
                "max_model_len": 2048,
                "max_num_batched_tokens": 2048,
                "enforce_eager": True,
                "trust_remote_code": True,
                "compilation_config": {"custom_ops": ["none"]},
            }
        },
        "glm-4.6v": {
            "name": "cyankiwi/GLM-4.6V-AWQ-4bit",
            "short": "GLM-4.6V-AWQ",
            "config": {
                "dtype": "auto",
                "gpu_memory_utilization": 0.90,
                "max_model_len": 2048,
                "max_num_batched_tokens": 2048,
                "enforce_eager": True,
                "trust_remote_code": True,
                "compilation_config": {"custom_ops": ["none"]},
            }
        }
    }

    # Test sequence: alternate models for 10 swaps
    model_sequence = ["gpt-oss-120b", "glm-4.6v"] * 5

    profiles: list[LoadProfile] = []
    io_monitor = DiskIOMonitor(interval=0.05)

    for switch_num, model_key in enumerate(model_sequence, 1):
        model_info = models[model_key]
        model_name = model_info["name"]
        model_short = model_info["short"]
        config = model_info["config"]

        print(f"\n{'#'*80}")
        print(f"# SWITCH {switch_num}/10: {model_short}")
        print(f"{'#'*80}")

        profile = LoadProfile(
            switch_num=switch_num,
            model=model_name,
            model_short=model_short,
            success=False,
        )

        # Take pre-load snapshot
        profile.before_load = take_memory_snapshot("before_load")
        print_memory_summary("Before load", profile.before_load)

        # Optionally drop caches to force cold load (every other load)
        if switch_num % 2 == 0:
            drop_caches()

        try:
            # Start I/O monitoring
            io_monitor.start()

            # Load model
            print(f"\n  Loading {model_short}...")
            t0 = time.perf_counter()

            llm = LLM(model=model_name, **config)

            profile.total_load_time = time.perf_counter() - t0

            # Stop I/O monitoring
            profile.disk_read_gb, profile.disk_read_speed_gbps = io_monitor.stop()

            print(f"  Loaded in {profile.total_load_time:.1f}s")
            print(f"  Disk I/O: {profile.disk_read_gb:.1f}GB read at {profile.disk_read_speed_gbps:.2f}GB/s")

            # Take post-load snapshot
            profile.after_load = take_memory_snapshot("after_load")
            print_memory_summary("After load", profile.after_load)

            # Quick inference test
            print("\n  Running inference...")
            sampling_params = SamplingParams(max_tokens=20, temperature=0.7)
            t0 = time.perf_counter()

            try:
                outputs = llm.generate(["Hello, how are you today?"], sampling_params)
                profile.inference_time = time.perf_counter() - t0
                profile.tokens_generated = len(outputs[0].outputs[0].token_ids) if outputs else 0
                text = outputs[0].outputs[0].text[:50] if outputs else ""
                print(f"  Generated {profile.tokens_generated} tokens in {profile.inference_time:.1f}s")
                print(f"  Output: {text}...")
            except Exception as e:
                print(f"  Inference failed: {e}")
                profile.inference_time = time.perf_counter() - t0

            profile.success = True

            # Cleanup
            print("\n  Cleaning up...")
            comprehensive_cleanup(llm)

            # Take post-cleanup snapshot
            profile.after_cleanup = take_memory_snapshot("after_cleanup")
            print_memory_summary("After cleanup", profile.after_cleanup)

            # Wait for memory to settle
            time.sleep(2)

        except Exception as e:
            import traceback
            print(f"\n  *** FAILED: {e} ***")
            traceback.print_exc()
            profile.error = str(e)
            profile.disk_read_gb, profile.disk_read_speed_gbps = io_monitor.stop()

        profiles.append(profile)

        if not profile.success:
            print("\n  Stopping due to failure...")
            break

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n{'Switch':<8} {'Model':<15} {'Load(s)':<10} {'Disk(GB)':<10} {'Speed(GB/s)':<12} {'GPU After':<12} {'Status'}")
    print("-"*80)

    for p in profiles:
        status = "✓" if p.success else f"✗ {p.error[:20] if p.error else ''}"
        gpu_after = f"{p.after_cleanup.gpu_used_gb:.1f}GB" if p.after_cleanup else "-"
        print(f"{p.switch_num:<8} {p.model_short:<15} {p.total_load_time:<10.1f} {p.disk_read_gb:<10.1f} {p.disk_read_speed_gbps:<12.2f} {gpu_after:<12} {status}")

    successful = [p for p in profiles if p.success]
    if successful:
        print(f"\nSuccessful: {len(successful)}/{len(profiles)}")

        avg_load = sum(p.total_load_time for p in successful) / len(successful)
        avg_disk_speed = sum(p.disk_read_speed_gbps for p in successful) / len(successful)

        print(f"Average load time: {avg_load:.1f}s")
        print(f"Average disk read speed: {avg_disk_speed:.2f}GB/s")

        # Separate stats for each model
        for model_key in models:
            model_profiles = [p for p in successful if model_key in p.model.lower()]
            if model_profiles:
                avg = sum(p.total_load_time for p in model_profiles) / len(model_profiles)
                print(f"  {models[model_key]['short']}: avg load {avg:.1f}s")

    # Save detailed results
    results_file = "/tmp/large_swap_results.json"
    results = {
        "profiles": [
            {
                "switch": p.switch_num,
                "model": p.model,
                "success": p.success,
                "load_time": p.total_load_time,
                "disk_read_gb": p.disk_read_gb,
                "disk_speed_gbps": p.disk_read_speed_gbps,
                "inference_time": p.inference_time,
                "tokens": p.tokens_generated,
                "gpu_before": p.before_load.gpu_used_gb if p.before_load else None,
                "gpu_after_load": p.after_load.gpu_used_gb if p.after_load else None,
                "gpu_after_cleanup": p.after_cleanup.gpu_used_gb if p.after_cleanup else None,
                "sys_avail_before": p.before_load.sys_available_gb if p.before_load else None,
                "sys_avail_after": p.after_cleanup.sys_available_gb if p.after_cleanup else None,
            }
            for p in profiles
        ]
    }
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    # Final state
    final_snap = take_memory_snapshot("final")
    print_memory_summary("\nFinal state", final_snap)


if __name__ == "__main__":
    main()
