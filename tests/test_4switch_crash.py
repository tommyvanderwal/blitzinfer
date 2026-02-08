#!/usr/bin/env python3
"""
Test 4-switch pattern that previously caused crashes.
Run on remote system: ./venv/bin/python tests/test_4switch_crash.py
"""
import os
import sys
import gc
import time

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

import torch

def get_gpu_mem():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_and_test(model_name, quant, short_name):
    """Load model, generate one token, return LLM object."""
    log(f"Loading {short_name} (quant={quant})...")
    log(f"  GPU before: {get_gpu_mem():.1f}GB")
    
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        quantization=quant,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.85,
    )
    
    log(f"  GPU after load: {get_gpu_mem():.1f}GB")
    
    # Generate one token to exercise the model
    out = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    log(f"  Output: {out[0].outputs[0].text[:30]}")
    log(f"  {short_name} OK!")
    
    return llm

def cleanup_llm(llm):
    """Clean up LLM and free GPU memory."""
    log("  Cleaning up...")
    
    try:
        # Navigate to model runner (InprocClient structure)
        engine_core = llm.llm_engine.engine_core
        if hasattr(engine_core, 'engine_core'):
            engine_core = engine_core.engine_core
        
        model_runner = engine_core.model_executor.driver_worker.worker.model_runner
        model = model_runner.model
        
        # Clear model weights
        for name, param in model.named_parameters():
            param.data.storage().resize_(0)
        for name, buf in model.named_buffers():
            buf.data.storage().resize_(0)
        
        # Clear KV cache
        if hasattr(model_runner, 'kv_caches'):
            for kv in model_runner.kv_caches:
                if hasattr(kv, 'storage'):
                    kv.storage().resize_(0)
            model_runner.kv_caches.clear()
        
        model_runner.model = None
        
        # Shutdown engine
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        log(f"  Cleanup error: {e}")
    
    del llm
    
    # Clear caches
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Clear vLLM state
    try:
        from vllm.distributed import parallel_state
        parallel_state.cleanup_dist_env_and_memory(shutdown_ray=False)
    except: pass
    
    try:
        from vllm.model_executor.layers import rotary_embedding
        rotary_embedding._ROPE_DICT.clear()
    except: pass
    
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
    except: pass
    
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"  GPU after cleanup: {get_gpu_mem():.1f}GB")

def main():
    log("=" * 60)
    log("4-SWITCH CRASH TEST")
    log("Pattern: gpt-oss → Qwen → gpt-oss → llama-AWQ-MARLIN")
    log("=" * 60)
    log(f"Initial GPU: {get_gpu_mem():.1f}GB")
    log("")
    
    # Models to test
    models = [
        ("openai/gpt-oss-120b", "mxfp4", "gpt-oss-1"),
        ("Qwen/Qwen3-VL-32B-Thinking-FP8", "fp8", "Qwen-32B"),
        ("openai/gpt-oss-120b", "mxfp4", "gpt-oss-2"),
        ("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "awq_marlin", "llama-AWQ-MARLIN"),
    ]
    
    for i, (model_name, quant, short_name) in enumerate(models, 1):
        log(f"\n=== SWITCH {i}/4: {short_name} ===")
        
        try:
            llm = load_and_test(model_name, quant, short_name)
            
            if i < len(models):
                # Not the last model, clean up for next
                cleanup_llm(llm)
                time.sleep(2)  # Brief pause between switches
            else:
                # Last model (llama-AWQ), keep it loaded to see if inference works
                log("\n*** llama-AWQ-MARLIN loaded successfully! ***")
                log("Running additional inference to stress test...")
                
                from vllm import SamplingParams
                for j in range(3):
                    out = llm.generate([f"Test {j}: Tell me about"], SamplingParams(max_tokens=20))
                    log(f"  Inference {j+1}: {out[0].outputs[0].text[:40]}...")
                
                log("\n*** ALL TESTS PASSED - NO CRASH ***")
                cleanup_llm(llm)
                
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    log(f"\nFinal GPU: {get_gpu_mem():.1f}GB")
    log("Test complete!")

if __name__ == "__main__":
    main()
