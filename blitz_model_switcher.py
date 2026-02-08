#!/usr/bin/env python3
"""
BlitzInfer Model Switcher for ROCm 780M

Fast in-process model switching for vLLM on AMD Radeon 780M.
Achieves ~8-10s switch time vs ~30-40s with subprocess mode.

Key technique: Targeted cleanup of model weight dictionaries to release
GPU memory without corrupting shared caches.

Usage:
    from blitz_model_switcher import BlitzModelSwitcher

    switcher = BlitzModelSwitcher()
    llm = switcher.load_model("Qwen/Qwen2.5-7B-Instruct")
    output = llm.generate(["Hello"], sampling_params)
    llm = switcher.switch_model("mistralai/Mistral-7B-Instruct-v0.3")
"""

import gc
import sys
import time
import torch


class BlitzModelSwitcher:
    """
    Fast model switcher for vLLM on ROCm 780M.
    """

    def __init__(self, default_config=None):
        """
        Initialize the model switcher.

        Args:
            default_config: Default LLM config dict (can be overridden per-model)
        """
        self.current_model = None
        self.current_llm = None
        self.default_config = default_config or {
            "dtype": "float16",
            "gpu_memory_utilization": 0.50,  # Conservative for switching
            "max_model_len": 1024,
            "max_num_batched_tokens": 1024,
            "kv_cache_memory_bytes": 4 * 1024**3,  # 4GB KV cache
            "enforce_eager": True,
            "compilation_config": {"custom_ops": ["none"]},
        }

    def _get_memory_gb(self):
        """Get free GPU memory in GB."""
        free, total = torch.cuda.mem_get_info()
        return free / (1024**3)

    def _targeted_cleanup(self):
        """
        Release GPU memory by clearing model weight dictionaries.
        This preserves shared caches (rotary embeddings, etc).
        """
        # Step 1: Standard vLLM cleanup
        try:
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory(shutdown_ray=False)
        except:
            pass
        gc.collect()

        # Step 2: Clear weight dicts (from nn.Module._parameters)
        cleared = 0
        for obj in gc.get_objects():
            try:
                if isinstance(obj, dict) and 'weight' in obj:
                    val = obj.get('weight')
                    if torch.is_tensor(val) and val.is_cuda:
                        for key in list(obj.keys()):
                            v = obj.get(key)
                            if torch.is_tensor(v) and v.is_cuda:
                                obj[key] = None
                                cleared += 1
            except:
                pass
        gc.collect()

        # Step 3: Clear tensor storage dicts (from safetensors)
        for obj in gc.get_objects():
            try:
                if isinstance(obj, dict) and 'tensor' in obj:
                    val = obj.get('tensor')
                    if torch.is_tensor(val) and val.is_cuda:
                        for key in list(obj.keys()):
                            v = obj.get(key)
                            if torch.is_tensor(v) and v.is_cuda:
                                obj[key] = None
            except:
                pass
        gc.collect()

        # Step 4: Reset torch dynamo
        try:
            import torch._dynamo
            torch._dynamo.reset()
        except:
            pass

        # Step 5: Clear CUDA caches
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        return cleared

    def load_model(self, model_name, **config_overrides):
        """
        Load a model.

        Args:
            model_name: HuggingFace model name or path
            **config_overrides: Override default config options

        Returns:
            vLLM LLM instance
        """
        from vllm import LLM

        # Merge configs
        config = {**self.default_config, **config_overrides}
        config["model"] = model_name

        # Load
        start = time.perf_counter()
        self.current_llm = LLM(**config)
        load_time = time.perf_counter() - start

        self.current_model = model_name
        print(f"Loaded {model_name} in {load_time:.2f}s")
        print(f"Free memory: {self._get_memory_gb():.2f} GB")

        return self.current_llm

    def switch_model(self, model_name, **config_overrides):
        """
        Switch to a different model.

        Args:
            model_name: HuggingFace model name or path
            **config_overrides: Override default config options

        Returns:
            vLLM LLM instance
        """
        from vllm import LLM

        if self.current_llm is None:
            return self.load_model(model_name, **config_overrides)

        # Cleanup current model
        print(f"\nSwitching from {self.current_model} to {model_name}...")
        cleanup_start = time.perf_counter()

        del self.current_llm
        self.current_llm = None
        gc.collect()

        cleared = self._targeted_cleanup()
        cleanup_time = time.perf_counter() - cleanup_start
        print(f"Cleanup: {cleanup_time:.2f}s ({cleared} tensors released)")
        print(f"Free memory: {self._get_memory_gb():.2f} GB")

        # Load new model
        load_start = time.perf_counter()
        config = {**self.default_config, **config_overrides}
        config["model"] = model_name
        self.current_llm = LLM(**config)
        load_time = time.perf_counter() - load_start

        self.current_model = model_name
        total_time = cleanup_time + load_time

        print(f"Load: {load_time:.2f}s")
        print(f"TOTAL SWITCH: {total_time:.2f}s")
        print(f"Free memory: {self._get_memory_gb():.2f} GB")

        return self.current_llm

    def unload(self):
        """Unload current model and free memory."""
        if self.current_llm is not None:
            del self.current_llm
            self.current_llm = None
            gc.collect()
            self._targeted_cleanup()
            self.current_model = None
            print(f"Model unloaded. Free memory: {self._get_memory_gb():.2f} GB")


def demo():
    """Demo of model switching."""
    import os
    import types

    # ROCm setup
    os.environ['HIP_VISIBLE_DEVICES'] = '0'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_SKIP_WARMUP'] = '1'
    os.environ['VLLM_DEEP_GEMM_WARMUP'] = 'skip'

    # vLLM import workaround
    fake_meta = types.ModuleType('torchvision._meta_registrations')
    sys.modules['torchvision._meta_registrations'] = fake_meta
    sys.path.insert(0, '/home/tommy/pythonprojects/blitzinfer/vllm')

    from vllm import SamplingParams

    print("=" * 70)
    print("BlitzInfer Model Switcher Demo")
    print("=" * 70)

    switcher = BlitzModelSwitcher()

    # Load first model
    llm = switcher.load_model("Qwen/Qwen2.5-7B-Instruct")
    out = llm.generate(["Hello, how are you?"], SamplingParams(max_tokens=20))
    print(f"Response: {out[0].outputs[0].text}\n")

    # Switch to same model (or different model if available)
    llm = switcher.switch_model("Qwen/Qwen2.5-7B-Instruct")
    out = llm.generate(["Tell me a joke"], SamplingParams(max_tokens=50))
    print(f"Response: {out[0].outputs[0].text}\n")

    # Unload
    switcher.unload()

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
