#!/usr/bin/env python3
"""Profile MXFP4 model to find all CUDA tensor locations.

Identifies what cleanup_vllm_model MISSES and what force_free catches.
"""
import os
import gc
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from vllm import LLM, SamplingParams


def get_active_blocks():
    """Get all active_allocated blocks from memory snapshot."""
    snapshot = torch.cuda.memory._snapshot()
    blocks = {}
    for seg in snapshot.get('segments', []):
        for block in seg.get('blocks', []):
            if block.get('state') == 'active_allocated':
                addr = block.get('addr', block.get('address', 0))
                size = block.get('size', 0)
                if addr and size > 0:
                    blocks[addr] = size
    return blocks


def get_gc_tensor_addresses():
    """Get all CUDA tensor storage addresses visible to gc."""
    addrs = {}
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            try:
                if obj.device.type == 'cuda' and obj.storage().size() > 0:
                    addr = obj.data_ptr()
                    size = obj.storage().size() * obj.element_size()
                    addrs[addr] = {
                        'size': size,
                        'shape': tuple(obj.shape),
                        'dtype': str(obj.dtype),
                    }
            except Exception:
                pass
    return addrs


def walk_model_tensors(model, max_depth=10):
    """Walk entire model object graph to find ALL CUDA tensors."""
    found = []
    visited = set()

    def _visit(obj, path, depth):
        if depth > max_depth or id(obj) in visited:
            return
        visited.add(id(obj))

        if isinstance(obj, torch.Tensor):
            if obj.device.type == 'cuda' and obj.storage().size() > 0:
                found.append({
                    'path': path,
                    'shape': tuple(obj.shape),
                    'dtype': str(obj.dtype),
                    'size_mb': obj.numel() * obj.element_size() / 1024**2,
                    'data_ptr': obj.data_ptr(),
                    'storage_size': obj.storage().size(),
                })
            return

        if isinstance(obj, torch.nn.Module):
            for name, child in obj.named_children():
                _visit(child, f"{path}.{name}", depth + 1)
            for attr_name in list(vars(obj).keys()):
                if attr_name.startswith('_') and attr_name not in ('_parameters', '_buffers', '_modules'):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                    _visit(attr, f"{path}.{attr_name}", depth + 1)
                except Exception:
                    pass
            return

        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _visit(item, f"{path}[{i}]", depth + 1)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _visit(v, f"{path}[{k!r}]", depth + 1)
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            for attr_name, attr_val in vars(obj).items():
                _visit(attr_val, f"{path}.{attr_name}", depth + 1)

    _visit(model, "model", 0)
    return found


def main():
    print("=" * 70)
    print("MXFP4 Tensor Location Profiler")
    print("=" * 70)

    _ = torch.zeros(1, device='cuda')
    del _
    gc.collect()
    torch.cuda.empty_cache()

    free0, total = torch.cuda.mem_get_info()
    print(f"Baseline: {(total-free0)/1024**3:.3f} GB used")

    # Load model
    llm = LLM(model="openai/gpt-oss-120b", dtype="auto",
              gpu_memory_utilization=0.90, max_model_len=4096,
              trust_remote_code=True, enforce_eager=True)

    free1, _ = torch.cuda.mem_get_info()
    print(f"After load: {(total-free1)/1024**3:.3f} GB used")

    # Quick inference
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    print(f"Output: {outputs[0].outputs[0].text[:30]}")

    # Get model reference
    engine = llm.llm_engine
    inproc = engine.engine_core
    engine_core = inproc.engine_core if hasattr(inproc, 'engine_core') else inproc
    executor = engine_core.model_executor
    worker = executor.driver_worker.worker
    model_runner = worker.model_runner
    model = model_runner.model

    # Walk model object graph
    print(f"\n{'='*70}")
    print("Walking model object graph for CUDA tensors...")
    model_tensors = walk_model_tensors(model)
    total_model_mb = sum(t['size_mb'] for t in model_tensors)
    print(f"Found {len(model_tensors)} CUDA tensors ({total_model_mb:.0f} MB)")

    # Show by path depth
    by_path = {}
    for t in model_tensors:
        # Get the top-level path component
        parts = t['path'].split('.')
        key = '.'.join(parts[:4]) if len(parts) >= 4 else t['path']
        if key not in by_path:
            by_path[key] = {'count': 0, 'total_mb': 0}
        by_path[key]['count'] += 1
        by_path[key]['total_mb'] += t['size_mb']

    print(f"\nTop tensor locations:")
    for path, info in sorted(by_path.items(), key=lambda x: -x[1]['total_mb'])[:20]:
        print(f"  {info['total_mb']:>8.1f} MB ({info['count']:>3} tensors) - {path}")

    # Compare with named_parameters
    param_ptrs = set()
    for name, param in model.named_parameters():
        if param.device.type == 'cuda':
            param_ptrs.add(param.data_ptr())
    for name, buf in model.named_buffers():
        if buf.device.type == 'cuda':
            param_ptrs.add(buf.data_ptr())

    model_tensor_ptrs = set(t['data_ptr'] for t in model_tensors)
    not_in_params = model_tensor_ptrs - param_ptrs

    print(f"\nTensors in named_parameters+buffers: {len(param_ptrs)}")
    print(f"Tensors from model walk: {len(model_tensor_ptrs)}")
    print(f"Tensors NOT in named_parameters/buffers: {len(not_in_params)}")

    # Show the non-parameter tensors
    if not_in_params:
        print(f"\nNon-parameter CUDA tensors:")
        for t in model_tensors:
            if t['data_ptr'] in not_in_params:
                print(f"  {t['size_mb']:>8.2f} MB {t['dtype']:>14} {str(t['shape']):>20} - {t['path']}")

    # Now run cleanup_vllm_model and check what survives
    print(f"\n{'='*70}")
    print("Running cleanup_vllm_model...")
    from blitzinfer.engine.cleanup import cleanup_vllm_model
    cleanup_vllm_model(llm)
    llm = None

    free2, _ = torch.cuda.mem_get_info()
    print(f"After cleanup: {(total-free2)/1024**3:.3f} GB used")

    # Check allocator blocks vs gc tensors
    blocks = get_active_blocks()
    gc_addrs = get_gc_tensor_addresses()

    total_block_gb = sum(blocks.values()) / 1024**3
    total_gc_gb = sum(a['size'] for a in gc_addrs.values()) / 1024**3
    print(f"\nAllocator active blocks: {len(blocks)} ({total_block_gb:.2f} GB)")
    print(f"GC-visible CUDA tensors: {len(gc_addrs)} ({total_gc_gb:.2f} GB)")

    # Find blocks with NO gc tensor
    orphan_blocks = 0
    orphan_bytes = 0
    for addr, size in blocks.items():
        if addr not in gc_addrs:
            orphan_blocks += 1
            orphan_bytes += size

    print(f"\nOrphan blocks (no gc tensor): {orphan_blocks} ({orphan_bytes/1024**3:.2f} GB)")
    print(f"GC-covered blocks: {len(blocks) - orphan_blocks}")

    # Show gc-visible tensors that ARE still allocated
    gc_still = {}
    for addr, info in gc_addrs.items():
        if addr in blocks:
            gc_still[addr] = info
    if gc_still:
        total_still_mb = sum(i['size'] for i in gc_still.values()) / 1024**2
        print(f"\nGC-visible tensors still in allocator: {len(gc_still)} ({total_still_mb:.0f} MB)")
        for addr, info in sorted(gc_still.items(), key=lambda x: -x[1]['size'])[:10]:
            print(f"  {info['size']/1024**2:>8.2f} MB {info['dtype']:>14} {str(info['shape']):>20}")


if __name__ == "__main__":
    main()
