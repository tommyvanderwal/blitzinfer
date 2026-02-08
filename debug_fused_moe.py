#!/usr/bin/env python3
"""Debug: check exactly how FusedMoE stores MXFP4 weights."""
import os
import gc

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from vllm import LLM, SamplingParams


def main():
    print("=" * 70)
    print("DEBUG: FusedMoE weight storage for gpt-oss-120b")
    print("=" * 70)

    _ = torch.zeros(1, device='cuda')
    del _

    llm = LLM(model="openai/gpt-oss-120b", dtype="auto",
              gpu_memory_utilization=0.90, max_model_len=4096,
              trust_remote_code=True, enforce_eager=True)

    # Get model reference
    engine = llm.llm_engine
    inproc = engine.engine_core
    engine_core = inproc.engine_core if hasattr(inproc, 'engine_core') else inproc
    executor = engine_core.model_executor
    worker = executor.driver_worker.worker
    model_runner = worker.model_runner
    model = model_runner.model

    # Find a FusedMoE layer
    print("\n--- Checking FusedMoE layers ---")

    fused_moe_layers = []
    for name, module in model.named_modules():
        if type(module).__name__ == 'FusedMoE':
            fused_moe_layers.append((name, module))

    print(f"Found {len(fused_moe_layers)} FusedMoE layers")

    # Check first 3 layers in detail
    for i, (name, layer) in enumerate(fused_moe_layers[:3]):
        print(f"\n--- Layer {i}: {name} ---")
        print(f"  type: {type(layer).__name__}")
        print(f"  id: {id(layer)}")

        # Check __dict__ keys
        dict_keys = list(layer.__dict__.keys())
        print(f"  __dict__ keys ({len(dict_keys)}): {dict_keys[:20]}")

        # Check if w13_weight is in __dict__
        has_w13 = 'w13_weight' in layer.__dict__
        has_w2 = 'w2_weight' in layer.__dict__
        print(f"  w13_weight in __dict__: {has_w13}")
        print(f"  w2_weight in __dict__: {has_w2}")

        # Check _parameters
        param_keys = list(layer._parameters.keys())
        print(f"  _parameters keys: {param_keys}")

        # Check if w13_weight is accessible as attribute
        w13 = getattr(layer, 'w13_weight', 'NOT FOUND')
        print(f"  getattr(layer, 'w13_weight'): {type(w13).__name__}")

        if w13 != 'NOT FOUND':
            print(f"    id: {id(w13)}")
            if hasattr(w13, 'storage') and hasattr(w13.storage, 'data'):
                data = w13.storage.data
                print(f"    storage.data type: {type(data).__name__}")
                print(f"    storage.data.device: {data.device}")
                print(f"    storage.data.shape: {data.shape}")
                print(f"    storage.data size MB: {data.numel() * data.element_size() / 1024**2:.1f}")
                print(f"    storage.data id: {id(data)}")
            elif hasattr(w13, 'data'):
                print(f"    data type: {type(w13.data).__name__}")

        # Check quant_method
        qm = getattr(layer, 'quant_method', None)
        if qm is not None:
            print(f"  quant_method: {type(qm).__name__} (id: {id(qm)})")
            qm_w13 = getattr(qm, 'w13_weight', None)
            if qm_w13 is not None:
                print(f"    qm.w13_weight id: {id(qm_w13)}")
                print(f"    Same as layer.w13_weight? {id(qm_w13) == id(w13)}")

    # Check if quant_method is shared
    print("\n--- quant_method sharing ---")
    qm_ids = set()
    for name, layer in fused_moe_layers:
        qm = getattr(layer, 'quant_method', None)
        if qm is not None:
            qm_ids.add(id(qm))
    print(f"Unique quant_method instances: {len(qm_ids)} (out of {len(fused_moe_layers)} layers)")

    # Check w13_weight sharing across layers
    print("\n--- w13_weight sharing ---")
    w13_ids = set()
    for name, layer in fused_moe_layers:
        w13 = getattr(layer, 'w13_weight', None)
        if w13 is not None:
            w13_ids.add(id(w13))
    print(f"Unique w13_weight instances: {len(w13_ids)} (out of {len(fused_moe_layers)} layers)")

    w13_data_ids = set()
    for name, layer in fused_moe_layers:
        w13 = getattr(layer, 'w13_weight', None)
        if w13 is not None and hasattr(w13, 'storage') and hasattr(w13.storage, 'data'):
            w13_data_ids.add(id(w13.storage.data))
    print(f"Unique w13_weight.storage.data torch.Tensor instances: {len(w13_data_ids)}")

    # Count total MXFP4 CUDA memory
    print("\n--- Total MXFP4 weight memory ---")
    total_mxfp4_mb = 0
    for name, layer in fused_moe_layers:
        for attr_name in ['w13_weight', 'w2_weight']:
            w = getattr(layer, attr_name, None)
            if w is not None and hasattr(w, 'storage') and hasattr(w.storage, 'data'):
                data = w.storage.data
                if data.device.type == 'cuda':
                    mb = data.numel() * data.element_size() / 1024**2
                    total_mxfp4_mb += mb
    print(f"Total MXFP4 weight data: {total_mxfp4_mb:.0f} MB = {total_mxfp4_mb/1024:.1f} GB")

    # Test if walker can find these
    print("\n--- Testing walker on single FusedMoE ---")
    layer0 = fused_moe_layers[0][1]
    visited = set()
    found = []

    def _visit(obj, path, depth):
        if depth > 12 or id(obj) in visited:
            return
        visited.add(id(obj))

        if isinstance(obj, torch.Tensor):
            if obj.device.type == 'cuda':
                try:
                    size = obj.numel() * obj.element_size() / 1024**2
                    found.append((path, size, str(obj.dtype), tuple(obj.shape)))
                except Exception:
                    pass
            return

        if isinstance(obj, torch.nn.Module):
            for child_name, child in obj.named_children():
                _visit(child, f"{path}.{child_name}", depth + 1)
            for attr_name, attr_val in vars(obj).items():
                if attr_name in ('_parameters', '_modules', '_buffers'):
                    continue
                _visit(attr_val, f"{path}.{attr_name}", depth + 1)
            for pname, param in obj._parameters.items():
                if param is not None:
                    _visit(param, f"{path}._p[{pname}]", depth + 1)
            return

        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _visit(item, f"{path}[{i}]", depth + 1)
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                _visit(v, f"{path}[{k}]", depth + 1)
            return

        if isinstance(obj, (int, float, str, bytes, bool, type, type(None))):
            return

        obj_dict = getattr(obj, '__dict__', None)
        if obj_dict is not None:
            for attr_name, attr_val in obj_dict.items():
                _visit(attr_val, f"{path}.{attr_name}", depth + 1)

    _visit(layer0, "fused_moe_0", 0)

    print(f"Found {len(found)} CUDA tensors in FusedMoE layer 0:")
    for path, size, dtype, shape in sorted(found, key=lambda x: -x[1])[:15]:
        print(f"  {size:>8.1f} MB {dtype:>14} {str(shape):>20} - {path}")

    total_found_mb = sum(f[1] for f in found)
    print(f"Total: {total_found_mb:.0f} MB")

    # Check FusedMoE vars specifically
    print("\n--- FusedMoE __dict__ detail ---")
    for k, v in layer0.__dict__.items():
        t = type(v).__name__
        if t in ('Tensor', 'PrecisionConfig', 'Parameter', 'ModelWeightParameter'):
            print(f"  {k}: {t} (id={id(v)})")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: torch.Tensor shape={v.shape} device={v.device}")


if __name__ == "__main__":
    main()
