#!/usr/bin/env python3
"""Debug: verify the model walker finds MXFP4 Triton Tensor objects."""
import os
import gc

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from vllm import LLM, SamplingParams


def walk_and_report(root, max_depth=12):
    """Walk model graph and report what's found."""
    found_tensors = []
    found_types = {}
    visited = set()

    def _visit(obj, path, depth):
        if depth > max_depth:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        type_name = type(obj).__name__
        if type_name not in found_types:
            found_types[type_name] = 0
        found_types[type_name] += 1

        if isinstance(obj, torch.Tensor):
            try:
                if obj.device.type == 'cuda' and obj.storage().size() > 0:
                    size_mb = obj.numel() * obj.element_size() / 1024**2
                    found_tensors.append({
                        'path': path,
                        'size_mb': size_mb,
                        'dtype': str(obj.dtype),
                        'shape': tuple(obj.shape),
                        'depth': depth,
                    })
            except Exception as e:
                found_tensors.append({
                    'path': path,
                    'size_mb': 0,
                    'dtype': 'ERROR',
                    'error': str(e),
                    'depth': depth,
                })
            return

        if isinstance(obj, torch.nn.Module):
            for name, child in obj.named_children():
                _visit(child, f"{path}.{name}", depth + 1)
            for attr_name, attr_val in vars(obj).items():
                if attr_name.startswith('_') and attr_name in ('_parameters', '_modules', '_buffers'):
                    continue  # Skip special dicts (already covered by named_children, etc.)
                _visit(attr_val, f"{path}.{attr_name}", depth + 1)
            # Also visit registered parameters and buffers explicitly
            for name, param in obj._parameters.items():
                if param is not None:
                    _visit(param, f"{path}._parameters[{name}]", depth + 1)
            for name, buf in obj._buffers.items():
                if buf is not None:
                    _visit(buf, f"{path}._buffers[{name}]", depth + 1)
            return

        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _visit(item, f"{path}[{i}]", depth + 1)
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                _visit(v, f"{path}[{k!r}]", depth + 1)
            return

        if isinstance(obj, (int, float, str, bytes, bool, type, type(None))):
            return

        obj_dict = getattr(obj, '__dict__', None)
        if obj_dict is not None:
            for attr_name, attr_val in obj_dict.items():
                _visit(attr_val, f"{path}.{attr_name}", depth + 1)

    _visit(root, "model", 0)
    return found_tensors, found_types


def main():
    print("=" * 70)
    print("DEBUG: Model Walker on gpt-oss-120b")
    print("=" * 70)

    _ = torch.zeros(1, device='cuda')
    del _
    gc.collect()
    torch.cuda.empty_cache()

    free0, total = torch.cuda.mem_get_info()
    print(f"Baseline: {(total-free0)/1024**3:.2f} GB used")

    llm = LLM(model="openai/gpt-oss-120b", dtype="auto",
              gpu_memory_utilization=0.90, max_model_len=4096,
              trust_remote_code=True, enforce_eager=True)

    free1, _ = torch.cuda.mem_get_info()
    print(f"After load: {(total-free1)/1024**3:.2f} GB used")

    # Quick inference
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    print(f"Output: {outputs[0].outputs[0].text[:30]}")

    # Get model reference
    engine = llm.llm_engine
    inproc = engine.engine_core
    engine_core = inproc.engine_core if hasattr(inproc, 'engine_core') else inproc
    executor = engine_core.model_executor
    worker = executor.driver_worker.worker
    model_runner = worker.model_runner
    model = model_runner.model

    # Walk and report
    print(f"\n{'='*70}")
    print("Walking model object graph...")
    found_tensors, found_types = walk_and_report(model)

    total_mb = sum(t['size_mb'] for t in found_tensors)
    print(f"\nFound {len(found_tensors)} CUDA tensors ({total_mb:.0f} MB = {total_mb/1024:.1f} GB)")

    # Show by depth
    by_depth = {}
    for t in found_tensors:
        d = t['depth']
        if d not in by_depth:
            by_depth[d] = {'count': 0, 'total_mb': 0}
        by_depth[d]['count'] += 1
        by_depth[d]['total_mb'] += t['size_mb']

    print("\nBy depth:")
    for d in sorted(by_depth):
        info = by_depth[d]
        print(f"  Depth {d}: {info['count']} tensors, {info['total_mb']:.0f} MB")

    # Show largest tensors
    print(f"\nTop 20 largest CUDA tensors:")
    for t in sorted(found_tensors, key=lambda x: -x['size_mb'])[:20]:
        print(f"  {t['size_mb']:>8.1f} MB d={t['depth']} {t['dtype']:>14} {str(t['shape']):>20} - {t['path']}")

    # Check for Triton Tensor, Storage, PrecisionConfig in found types
    print(f"\nObject types found:")
    for type_name in sorted(found_types, key=lambda x: -found_types[x])[:30]:
        print(f"  {found_types[type_name]:>5} - {type_name}")

    # Check what named_parameters covers
    param_ptrs = set()
    for name, param in model.named_parameters():
        if param.device.type == 'cuda':
            param_ptrs.add(param.data_ptr())
    for name, buf in model.named_buffers():
        if buf.device.type == 'cuda':
            param_ptrs.add(buf.data_ptr())

    walker_ptrs = set(t.get('data_ptr', None) for t in found_tensors)
    # Get data_ptrs properly
    walker_ptrs = set()
    not_in_params = []
    for t in found_tensors:
        if t.get('error'):
            continue
        # Check if this tensor's address is in named params/buffers
        # We can't easily get data_ptr from our report, but we can check by path
        if '_parameters[' in t['path']:
            continue
        if '_buffers[' in t['path']:
            continue
        not_in_params.append(t)

    print(f"\nTensors in named_parameters/buffers: {len(param_ptrs)}")
    print(f"Tensors NOT in named_parameters/buffers:")
    for t in sorted(not_in_params, key=lambda x: -x['size_mb'])[:20]:
        print(f"  {t['size_mb']:>8.1f} MB d={t['depth']} {t['dtype']:>14} {str(t['shape']):>20} - {t['path']}")

    # Now check: what does the actual walker free?
    print(f"\n{'='*70}")
    print("Running actual _walk_and_free_cuda_tensors...")
    from blitzinfer.engine.cleanup import _walk_and_free_cuda_tensors
    n_freed = _walk_and_free_cuda_tensors(model)
    print(f"Walker freed {n_freed} CUDA tensors")

    gc.collect()
    torch.cuda.empty_cache()
    free2, _ = torch.cuda.mem_get_info()
    print(f"After walker + gc + empty_cache: {(total-free2)/1024**3:.2f} GB used")

    # Check what's left
    snapshot = torch.cuda.memory._snapshot()
    remaining = 0
    remaining_count = 0
    for seg in snapshot.get('segments', []):
        for block in seg.get('blocks', []):
            if block.get('state') == 'active_allocated':
                remaining += block.get('size', 0)
                remaining_count += 1
    print(f"Remaining allocated blocks: {remaining_count} ({remaining/1024**3:.2f} GB)")

    # Now run full cleanup
    print(f"\n{'='*70}")
    print("Running full cleanup_vllm_model...")
    model_runner.model = None
    del model
    from blitzinfer.engine.cleanup import cleanup_vllm_model
    # Can't run cleanup_vllm_model since model is already detached
    # Just gc + empty_cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free3, _ = torch.cuda.mem_get_info()
    print(f"After full gc: {(total-free3)/1024**3:.2f} GB used")


if __name__ == "__main__":
    main()
