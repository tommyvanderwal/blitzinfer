#!/usr/bin/env python3
"""Debug: inspect what's on gpt-oss MoE layers after process_weights_after_loading."""

import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

import torch

def main():
    from sglang.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path="openai/gpt-oss-120b",
        mem_fraction_static=0.85,
        max_running_requests=4,
        disable_cuda_graph=True,
        attention_backend="triton",
        trust_remote_code=True,
        log_level="info",
        moe_runner_backend="triton_kernel",
        dtype="bfloat16",
    )

    # Access the model through the engine's internal structure
    model_runner = engine.tokenizer_manager.scheduler.tp_worker.model_runner
    model = model_runner.model

    print(f"\nModel type: {type(model).__name__}")
    print(f"GPU alloc: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # Find all modules
    module_types = {}
    for name, module in model.named_modules():
        t = type(module).__name__
        module_types[t] = module_types.get(t, 0) + 1
    print(f"\nModule types: {module_types}")

    # Find FusedMoE modules
    moe_count = 0
    total_moe_gpu_bytes = 0
    for name, module in model.named_modules():
        if 'FusedMoE' in type(module).__name__:
            moe_count += 1
            if moe_count <= 2:  # Print details for first 2
                print(f"\n=== {name} ({type(module).__name__}) ===")
                print(f"  vars keys: {list(vars(module).keys())[:20]}")
                qm = getattr(module, 'quant_method', None)
                if qm is not None:
                    print(f"  quant_method type: {type(qm).__name__}")
                    print(f"  quant_method has __dict__: {hasattr(qm, '__dict__')}")
                    if hasattr(qm, '__dict__'):
                        qm_dict = vars(qm)
                        print(f"  quant_method vars keys: {list(qm_dict.keys())}")
                        for k, v in qm_dict.items():
                            if isinstance(v, torch.Tensor):
                                print(f"    {k}: Tensor({v.shape}, {v.device}, storage={v.storage().size()} bytes)")
                                total_moe_gpu_bytes += v.storage().size() * v.element_size() if v.is_cuda else 0
                            elif hasattr(v, '__dict__'):
                                print(f"    {k}: {type(v).__name__}")
                                for k2, v2 in vars(v).items():
                                    if isinstance(v2, torch.Tensor):
                                        print(f"      {k2}: Tensor({v2.shape}, {v2.device}, storage={v2.storage().size()} bytes)")
                                        total_moe_gpu_bytes += v2.storage().size() * v2.element_size() if v2.is_cuda else 0

    print(f"\nTotal MoE modules: {moe_count}")
    print(f"Total MoE GPU bytes found on quant_method: {total_moe_gpu_bytes/1024**3:.1f}GB")

    # Now test freeing
    print(f"\n--- Testing cleanup ---")
    ga_before = torch.cuda.memory_allocated() / 1024**3
    print(f"Before: {ga_before:.1f}GB")

    n_params = 0
    for p in model.parameters():
        p.data.storage().resize_(0)
        n_params += 1

    n_bufs = 0
    for b in model.buffers():
        b.data.storage().resize_(0)
        n_bufs += 1

    ga_after_params = torch.cuda.memory_allocated() / 1024**3
    print(f"After params+bufs ({n_params}p+{n_bufs}b): {ga_after_params:.1f}GB (freed {ga_before - ga_after_params:.1f}GB)")

    # Now free quant_method tensors
    n_qm = 0
    for name, module in model.named_modules():
        qm = vars(module).get('quant_method')
        if qm is not None and hasattr(qm, '__dict__'):
            for k, v in list(vars(qm).items()):
                if isinstance(v, torch.Tensor) and v.is_cuda and v.storage().size() > 0:
                    v.storage().resize_(0)
                    n_qm += 1
                elif hasattr(v, '__dict__'):
                    for k2, v2 in list(vars(v).items()):
                        if isinstance(v2, torch.Tensor) and v2.is_cuda and v2.storage().size() > 0:
                            v2.storage().resize_(0)
                            n_qm += 1

    ga_after_qm = torch.cuda.memory_allocated() / 1024**3
    print(f"After quant_method ({n_qm} tensors): {ga_after_qm:.1f}GB (freed {ga_after_params - ga_after_qm:.1f}GB)")

    engine.shutdown()
    print("Done")

if __name__ == "__main__":
    main()
