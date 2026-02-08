#!/usr/bin/env python3
"""Profile FLA tensor_cache and GPU memory in detail."""
import os
import gc
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch


def get_memory_snapshot_summary():
    """Get detailed memory snapshot."""
    snapshot = torch.cuda.memory._snapshot()
    segments = snapshot.get('segments', [])

    total_allocated = 0
    total_reserved = 0
    blocks_by_size = {}

    for seg in segments:
        seg_size = seg.get('total_size', 0)
        total_reserved += seg_size

        for block in seg.get('blocks', []):
            if block.get('state') == 'active_allocated':
                size = block.get('size', 0)
                total_allocated += size

                # Categorize by size
                size_mb = size / 1024**2
                if size_mb < 1:
                    cat = '<1MB'
                elif size_mb < 10:
                    cat = '1-10MB'
                elif size_mb < 100:
                    cat = '10-100MB'
                elif size_mb < 1000:
                    cat = '100MB-1GB'
                else:
                    cat = '>1GB'

                if cat not in blocks_by_size:
                    blocks_by_size[cat] = {'count': 0, 'total_mb': 0}
                blocks_by_size[cat]['count'] += 1
                blocks_by_size[cat]['total_mb'] += size_mb

    return {
        'allocated_gb': total_allocated / 1024**3,
        'reserved_gb': total_reserved / 1024**3,
        'blocks_by_size': blocks_by_size,
    }


def inspect_fla_tensor_cache():
    """Inspect what's in FLA tensor_cache closures."""
    fla_modules = {name: mod for name, mod in sys.modules.items()
                   if mod is not None and 'fla' in name.lower()}

    print(f"\n=== FLA Modules Loaded: {len(fla_modules)} ===")

    cache_info = []

    for name, mod in fla_modules.items():
        for attr_name in dir(mod):
            try:
                attr = getattr(mod, attr_name, None)
                if not callable(attr) or not hasattr(attr, '__closure__'):
                    continue

                closure = attr.__closure__
                if closure is None:
                    continue

                for i, cell in enumerate(closure):
                    try:
                        contents = cell.cell_contents
                        if isinstance(contents, list) and len(contents) > 0:
                            # Check if it looks like cache_entries
                            if isinstance(contents[0], tuple) and len(contents[0]) == 3:
                                # This is a tensor_cache!
                                tensors = []
                                for entry in contents:
                                    args, kwargs, result = entry
                                    for item in (args if isinstance(args, tuple) else ()):
                                        if isinstance(item, torch.Tensor):
                                            tensors.append({
                                                'shape': tuple(item.shape),
                                                'dtype': str(item.dtype),
                                                'device': str(item.device),
                                                'size_mb': item.numel() * item.element_size() / 1024**2,
                                                'storage_size': item.storage().size(),
                                            })
                                    if isinstance(result, torch.Tensor):
                                        tensors.append({
                                            'shape': tuple(result.shape),
                                            'dtype': str(result.dtype),
                                            'device': str(result.device),
                                            'size_mb': result.numel() * result.element_size() / 1024**2,
                                            'storage_size': result.storage().size(),
                                        })

                                cache_info.append({
                                    'module': name,
                                    'function': attr_name,
                                    'entries': len(contents),
                                    'tensors': tensors,
                                })
                    except ValueError:
                        pass
            except Exception:
                pass

    return cache_info


def clear_fla_caches_direct():
    """Clear FLA caches by directly resizing tensor storages."""
    fla_modules = {name: mod for name, mod in list(sys.modules.items())
                   if mod is not None and 'fla' in name.lower()}

    cleared = 0
    bytes_freed = 0

    for name, mod in fla_modules.items():
        for attr_name in dir(mod):
            try:
                attr = getattr(mod, attr_name, None)
                if not callable(attr) or not hasattr(attr, '__closure__'):
                    continue

                closure = attr.__closure__
                if closure is None:
                    continue

                for cell in closure:
                    try:
                        contents = cell.cell_contents
                        if isinstance(contents, list):
                            for entry in contents:
                                if isinstance(entry, tuple) and len(entry) == 3:
                                    args, kwargs, result = entry
                                    for item in (args if isinstance(args, tuple) else ()):
                                        if isinstance(item, torch.Tensor) and item.device.type == 'cuda':
                                            bytes_freed += item.storage().size() * item.element_size()
                                            item.data.storage().resize_(0)
                                            cleared += 1
                                    if isinstance(result, torch.Tensor) and result.device.type == 'cuda':
                                        bytes_freed += result.storage().size() * result.element_size()
                                        result.data.storage().resize_(0)
                                        cleared += 1
                            contents.clear()
                    except ValueError:
                        pass
            except Exception as e:
                pass

    return cleared, bytes_freed


def main():
    print("=" * 70)
    print("FLA Tensor Cache & GPU Memory Profiler")
    print("=" * 70)

    # Baseline memory
    free, total = torch.cuda.mem_get_info()
    print(f"\n1. BASELINE (no model loaded)")
    print(f"   Free: {free/1024**3:.2f} GB / {total/1024**3:.2f} GB")

    # Load qwen3-coder-next
    print(f"\n2. Loading qwen3-coder-next (FLA model)...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen3-Coder-Next-FP8",
        dtype="auto",
        gpu_memory_utilization=0.94,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        enforce_eager=True,
    )

    free, total = torch.cuda.mem_get_info()
    print(f"   After load - Free: {free/1024**3:.2f} GB")

    # Run inference to populate tensor_cache
    print(f"\n3. Running inference to populate tensor_cache...")
    outputs = llm.generate(["Write a hello world in Python"], SamplingParams(max_tokens=100))
    print(f"   Output: {outputs[0].outputs[0].text[:50]}...")

    free_after_infer, _ = torch.cuda.mem_get_info()
    print(f"   After inference - Free: {free_after_infer/1024**3:.2f} GB")

    # Inspect FLA caches
    print(f"\n4. Inspecting FLA tensor_cache contents...")
    cache_info = inspect_fla_tensor_cache()

    total_cached_mb = 0
    for info in cache_info:
        print(f"\n   {info['module']}.{info['function']}: {info['entries']} entries")
        for t in info['tensors']:
            print(f"      - {t['shape']} {t['dtype']} on {t['device']}: {t['size_mb']:.2f} MB (storage: {t['storage_size']})")
            if 'cuda' in t['device']:
                total_cached_mb += t['size_mb']

    print(f"\n   TOTAL cached on GPU: {total_cached_mb:.2f} MB")

    # Get memory snapshot
    print(f"\n5. Memory snapshot before cleanup...")
    snap = get_memory_snapshot_summary()
    print(f"   Allocated: {snap['allocated_gb']:.2f} GB")
    print(f"   Reserved: {snap['reserved_gb']:.2f} GB")
    print(f"   Blocks by size:")
    for cat, info in sorted(snap['blocks_by_size'].items()):
        print(f"      {cat}: {info['count']} blocks, {info['total_mb']:.1f} MB")

    # Clear FLA caches directly (without GC)
    print(f"\n6. Clearing FLA caches directly (no GC)...")
    cleared, bytes_freed = clear_fla_caches_direct()
    print(f"   Cleared {cleared} tensors, {bytes_freed/1024**2:.2f} MB")

    torch.cuda.empty_cache()
    free_after_clear, _ = torch.cuda.mem_get_info()
    print(f"   After clear - Free: {free_after_clear/1024**3:.2f} GB")
    print(f"   Memory recovered: {(free_after_clear - free_after_infer)/1024**2:.2f} MB")

    # Now do full cleanup
    print(f"\n7. Full model cleanup...")
    from blitzinfer.engine.cleanup import full_cleanup

    freed = full_cleanup(llm, nuclear=True, force_free=False)
    print(f"   Freed: {freed:.2f} GB")

    free_final, _ = torch.cuda.mem_get_info()
    print(f"   Final free: {free_final/1024**3:.2f} GB")

    # Check what's still allocated
    print(f"\n8. Final memory snapshot...")
    snap = get_memory_snapshot_summary()
    print(f"   Allocated: {snap['allocated_gb']:.2f} GB")
    print(f"   Reserved: {snap['reserved_gb']:.2f} GB")
    if snap['blocks_by_size']:
        print(f"   Remaining blocks:")
        for cat, info in sorted(snap['blocks_by_size'].items()):
            print(f"      {cat}: {info['count']} blocks, {info['total_mb']:.1f} MB")


if __name__ == "__main__":
    main()
