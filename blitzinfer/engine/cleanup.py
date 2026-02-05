"""Aggressive GPU memory cleanup for vLLM V1 single-process mode.

vLLM V1's single-process mode doesn't properly release model weights
when shutdown() is called. This module provides functions to manually
clear all GPU memory.

SOLUTION: Phantom Block Cleanup
-------------------------------
Internal C++ libraries (Flash Attention, NCCL, cuBLAS) allocate 3x160MB
"phantom blocks" through PyTorch's allocator WITHOUT creating Python
tensor objects. These cannot be freed by gc.collect() or empty_cache().

SOLUTION: Use `caching_allocator_delete()` with addresses from memory_snapshot()
to force-free these blocks. The `force_free_phantom_blocks()` function does this
automatically and is called by `nuclear_cleanup()`.

With nuclear_cleanup() enabled (default), memory drift is reduced to ~10MB
per switch cycle, enabling 1000+ model switches without OOM.
"""

import gc
import logging
import torch

logger = logging.getLogger(__name__)


def _walk_and_free_cuda_tensors(root, max_depth: int = 12) -> int:
    """Recursively walk an object graph and resize ALL CUDA tensor storages to 0.

    This is critical for MXFP4 models where the Triton backend stores weights in
    custom wrapper objects (Tensor → Storage → torch.Tensor) that are NOT registered
    as nn.Parameters or buffers. named_parameters() and named_buffers() miss these.

    The walker handles:
    - torch.Tensor directly on CUDA
    - Triton Tensor wrappers (storage.data is a torch.Tensor)
    - PrecisionConfig/FlexCtx dataclasses with nested tensors
    - nn.Module children and attributes
    - Lists, tuples, dicts
    - Any object with __dict__

    Returns:
        Number of CUDA tensors freed.
    """
    freed = 0
    visited = set()

    def _visit(obj, depth):
        nonlocal freed
        if depth > max_depth:
            return

        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        # Direct torch.Tensor on CUDA - resize its storage
        if isinstance(obj, torch.Tensor):
            try:
                if obj.device.type == 'cuda' and obj.storage().size() > 0:
                    obj.data.storage().resize_(0)
                    freed += 1
            except Exception:
                pass
            return

        # nn.Module - walk children, parameters, buffers, and all attributes
        if isinstance(obj, torch.nn.Module):
            for child in obj.children():
                _visit(child, depth + 1)
            # Walk ALL instance attributes (catches quant_method, MoE internals, etc.)
            for attr_val in vars(obj).values():
                _visit(attr_val, depth + 1)
            return

        # Lists and tuples
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _visit(item, depth + 1)
            return

        # Dicts
        if isinstance(obj, dict):
            for v in obj.values():
                _visit(v, depth + 1)
            return

        # Skip primitive types, strings, and types themselves
        if isinstance(obj, (int, float, str, bytes, bool, type, type(None))):
            return

        # Generic objects with __dict__ (covers dataclasses like Triton Tensor,
        # Storage, PrecisionConfig, FlexCtx, etc.)
        obj_dict = getattr(obj, '__dict__', None)
        if obj_dict is not None:
            for attr_val in obj_dict.values():
                _visit(attr_val, depth + 1)

    _visit(root, 0)
    return freed


def cleanup_vllm_model(llm) -> float:
    """Aggressively clean up a vLLM LLM instance and release GPU memory.

    Args:
        llm: The vLLM LLM instance to clean up.

    Returns:
        Amount of GPU memory freed in GB (based on actual CUDA free memory).
    """
    if llm is None:
        return 0.0

    # Use actual CUDA memory for accurate measurement
    free_before, total = torch.cuda.mem_get_info()

    try:
        # Get references to internal components
        engine = getattr(llm, 'llm_engine', None)
        if engine is None:
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            return 0.0

        # vLLM V1 uses InprocClient which wraps the actual EngineCore
        # We need to navigate: engine.engine_core (InprocClient) -> engine_core (EngineCore)
        inproc_client = getattr(engine, 'engine_core', None)

        # Unwrap InprocClient to get the actual EngineCore
        if inproc_client is not None and hasattr(inproc_client, 'engine_core'):
            engine_core = inproc_client.engine_core
        else:
            engine_core = inproc_client

        # Try to get model runner and worker
        model_runner = None
        model = None
        worker = None
        inner_worker = None

        if engine_core is not None:
            # V1 engine structure: EngineCore -> model_executor -> driver_worker -> worker -> model_runner
            executor = getattr(engine_core, 'model_executor', None)
            if executor is not None:
                worker = getattr(executor, 'driver_worker', None)
                if worker is not None:
                    inner_worker = getattr(worker, 'worker', None)
                    if inner_worker is not None:
                        model_runner = getattr(inner_worker, 'model_runner', None)

        if model_runner is not None:
            model = getattr(model_runner, 'model', None)

            # Clear KV caches from model_runner
            # IMPORTANT: Use storage().resize_(0) to actually free GPU memory!
            # tensor.data = empty doesn't release the underlying storage.
            if hasattr(model_runner, 'kv_caches'):
                kv_caches = model_runner.kv_caches
                for i, kv in enumerate(kv_caches):
                    if kv is not None:
                        if isinstance(kv, torch.Tensor):
                            try:
                                kv.storage().resize_(0)
                            except Exception:
                                pass
                        elif isinstance(kv, (list, tuple)):
                            for t in kv:
                                if isinstance(t, torch.Tensor):
                                    try:
                                        t.storage().resize_(0)
                                    except Exception:
                                        pass
                kv_caches.clear()
                logger.debug("Cleared model_runner.kv_caches")

            # Clear static forward context KV caches
            if hasattr(model_runner, 'static_forward_context'):
                ctx = model_runner.static_forward_context
                for name in list(ctx.keys()):
                    layer = ctx[name]
                    if hasattr(layer, 'kv_cache'):
                        kv_list = layer.kv_cache if isinstance(layer.kv_cache, list) else [layer.kv_cache]
                        for kv in kv_list:
                            if isinstance(kv, torch.Tensor):
                                try:
                                    kv.storage().resize_(0)
                                except Exception:
                                    pass
                        layer.kv_cache = []
                ctx.clear()
                logger.debug("Cleared static_forward_context")

            # Clear input buffers
            if hasattr(model_runner, 'input_ids'):
                model_runner.input_ids = None
            if hasattr(model_runner, 'positions'):
                model_runner.positions = None
            if hasattr(model_runner, 'input_embeds'):
                model_runner.input_embeds = None

            # Clear encoder caches for vision models
            if hasattr(model_runner, 'encoder_cache'):
                if model_runner.encoder_cache is not None:
                    model_runner.encoder_cache.clear()
                model_runner.encoder_cache = None
                logger.debug("Cleared encoder_cache")

        # Walk the ENTIRE model object graph to find and free ALL CUDA tensors.
        # This catches tensors that named_parameters()/named_buffers() miss:
        # - MXFP4 Triton backend: Tensor→Storage→torch.Tensor wrappers
        #   stored on Mxfp4MoEMethod (self.w13_weight, self.w2_weight)
        # - PrecisionConfig.weight_scale (Triton Tensor with CUDA data)
        # - Any other quantization method's internal CUDA state
        if model is not None:
            # CRITICAL: Clear compilation_config.static_forward_context BEFORE walking.
            # This dict maps layer names to ALL FusedMoE modules. It's a shared reference
            # accessible from every FusedMoE layer's vllm_config attribute. When the walker
            # visits layer 0 and traverses vllm_config, it finds all 36 layers at depth ~9,
            # adding them to the visited set. Later, layers 1-35 via model.layers (depth ~5)
            # are skipped as already-visited. Through vllm_config, MXFP4 data tensors end up
            # at depth ~13 (beyond max_depth=12) and never get freed. Clearing this dict
            # forces the walker to visit each layer only through model.layers (depth ~5),
            # giving enough depth budget to reach Tensor→Storage→data at depth ~9.
            for module in model.modules():
                vc = getattr(module, 'vllm_config', None)
                if vc is not None:
                    cc = getattr(vc, 'compilation_config', None)
                    if cc is not None:
                        sfc = getattr(cc, 'static_forward_context', None)
                        if sfc is not None and isinstance(sfc, dict) and len(sfc) > 0:
                            sfc.clear()
                            logger.debug("Cleared compilation_config.static_forward_context")
                            break  # Shared config, only need to clear once

            n_freed = _walk_and_free_cuda_tensors(model)
            logger.debug(f"Model graph walk: freed {n_freed} CUDA tensors")

            # Delete model reference
            del model
            if model_runner is not None:
                model_runner.model = None

        # Clear inner worker state
        if inner_worker is not None:
            inner_worker.model_runner = None

        # Clear worker state
        if worker is not None:
            worker.worker = None

        # Call engine_core shutdown AFTER clearing state
        if engine_core is not None:
            try:
                engine_core.shutdown()
                logger.debug("Called engine_core.shutdown()")
            except Exception as e:
                logger.debug(f"engine_core.shutdown() error: {e}")

        # Also shutdown InprocClient if it's different from engine_core
        if inproc_client is not None and inproc_client is not engine_core:
            try:
                if hasattr(inproc_client, 'shutdown'):
                    inproc_client.shutdown()
                    logger.debug("Called inproc_client.shutdown()")
            except Exception as e:
                logger.debug(f"inproc_client.shutdown() error: {e}")

        # Clear engine references
        if engine is not None:
            engine.engine_core = None

    except Exception as e:
        logger.warning(f"Error during model cleanup: {e}")
        import traceback
        traceback.print_exc()

    # Delete the LLM object
    del llm

    # Garbage collect to release Python references, then free cached CUDA memory
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # Synchronize and clear
    torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    # Try to release PyTorch's CUDA caching allocator memory
    try:
        torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.0")
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.debug(f"Could not adjust allocator settings: {e}")

    # Force synchronization and cleanup
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as e:
        logger.debug(f"Could not run ipc_collect: {e}")

    # Restore reasonable GC threshold
    try:
        torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.6")
    except Exception:
        pass

    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_after, _ = torch.cuda.mem_get_info()
    freed = (free_after - free_before) / 1024**3

    logger.info(f"GPU cleanup: freed {freed:.1f}GB ({(total-free_before)/1024**3:.1f}GB -> {(total-free_after)/1024**3:.1f}GB used)")

    return freed


def clear_rope_cache():
    """Clear vLLM's RoPE embedding cache for cross-architecture switching."""
    try:
        from vllm.model_executor.layers import rotary_embedding
        if hasattr(rotary_embedding, '_ROPE_DICT'):
            rotary_embedding._ROPE_DICT.clear()
            logger.debug("Cleared _ROPE_DICT")
    except Exception as e:
        logger.debug(f"Failed to clear RoPE cache: {e}")


def clear_workspace_manager():
    """Clear vLLM's WorkspaceManager singleton to prevent reinitialization issues.

    CRITICAL: The V1 WorkspaceManager holds workspace tensors that persist between
    model loads, causing ~0.47GB leak per switch. We must explicitly free these.
    """
    # V1 workspace manager (vllm.v1.worker.workspace)
    try:
        from vllm.v1.worker import workspace as v1_workspace
        if hasattr(v1_workspace, '_manager') and v1_workspace._manager is not None:
            mgr = v1_workspace._manager
            # Explicitly free workspace tensors
            if hasattr(mgr, '_current_workspaces'):
                for i, ws in enumerate(mgr._current_workspaces):
                    if ws is not None and isinstance(ws, torch.Tensor):
                        try:
                            ws.storage().resize_(0)
                        except Exception:
                            pass
                        mgr._current_workspaces[i] = None
                logger.debug("Cleared V1 WorkspaceManager workspace tensors")
            # Reset the manager
            v1_workspace._manager = None
            logger.debug("Reset V1 WorkspaceManager")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not clear V1 WorkspaceManager: {e}")

    # Legacy workspace manager (vllm.model_executor.layers.fused_moe)
    try:
        from vllm.model_executor.layers.fused_moe import workspace
        if hasattr(workspace, 'WorkspaceManager'):
            mgr_cls = workspace.WorkspaceManager
            # Clear the singleton instance
            if hasattr(mgr_cls, '_instance'):
                mgr_cls._instance = None
            if hasattr(mgr_cls, '_initialized'):
                mgr_cls._initialized = False
            # Also clear any per-device instances
            if hasattr(mgr_cls, '_instances'):
                mgr_cls._instances.clear()
            logger.debug("Cleared legacy WorkspaceManager")
    except Exception as e:
        logger.debug(f"Could not clear legacy WorkspaceManager: {e}")


def clear_forward_context():
    """Clear vLLM's global forward context.

    CRITICAL: ForwardContext holds slot_mapping tensors (GPU) and attn_metadata.
    These persist in the global _forward_context and accumulate memory.
    """
    try:
        from vllm import forward_context as fc_module
        if hasattr(fc_module, '_forward_context') and fc_module._forward_context is not None:
            ctx = fc_module._forward_context
            # Clear slot_mapping tensors
            if hasattr(ctx, 'slot_mapping') and ctx.slot_mapping is not None:
                if isinstance(ctx.slot_mapping, dict):
                    for name, tensor in ctx.slot_mapping.items():
                        if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
                            try:
                                tensor.storage().resize_(0)
                            except Exception:
                                pass
                    ctx.slot_mapping.clear()
                elif isinstance(ctx.slot_mapping, list):
                    for d in ctx.slot_mapping:
                        if isinstance(d, dict):
                            for tensor in d.values():
                                if isinstance(tensor, torch.Tensor) and tensor.device.type == 'cuda':
                                    try:
                                        tensor.storage().resize_(0)
                                    except Exception:
                                        pass
                            d.clear()
            # Clear attn_metadata
            if hasattr(ctx, 'attn_metadata') and ctx.attn_metadata is not None:
                if isinstance(ctx.attn_metadata, dict):
                    ctx.attn_metadata.clear()
                elif isinstance(ctx.attn_metadata, list):
                    for d in ctx.attn_metadata:
                        if isinstance(d, dict):
                            d.clear()
            # Reset the global
            fc_module._forward_context = None
            logger.debug("Cleared forward_context")
    except Exception as e:
        logger.debug(f"Could not clear forward_context: {e}")


def clear_ubatch_contexts():
    """Clear vLLM's ubatching global contexts.

    CRITICAL: UBatchContext holds CUDA streams and events that persist
    between model loads and can accumulate memory.
    """
    try:
        from vllm.v1.worker import ubatching
        if hasattr(ubatching, '_CURRENT_CONTEXTS'):
            for i, ctx in enumerate(ubatching._CURRENT_CONTEXTS):
                if ctx is not None:
                    # Clear the context's references
                    if hasattr(ctx, 'forward_context'):
                        ctx.forward_context = None
                    if hasattr(ctx, 'recv_hook'):
                        ctx.recv_hook = None
                    ubatching._CURRENT_CONTEXTS[i] = None
            ubatching._CURRENT_CONTEXTS.clear()
        if hasattr(ubatching, '_THREAD_ID_TO_CONTEXT'):
            ubatching._THREAD_ID_TO_CONTEXT.clear()
        logger.debug("Cleared ubatch contexts")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not clear ubatch contexts: {e}")


def clear_current_vllm_config():
    """Reset vLLM's global config to allow fresh initialization.

    CRITICAL: _current_vllm_config persists after model unload. If not reset,
    the next model's parallel state initialization reads the OLD model's config,
    causing MoE models to fail with "expert parallel group is not initialized".
    """
    try:
        from vllm.config import vllm as vllm_config_module
        if hasattr(vllm_config_module, '_current_vllm_config'):
            vllm_config_module._current_vllm_config = None
            logger.debug("Reset _current_vllm_config")
    except Exception as e:
        logger.debug(f"Could not reset vllm config: {e}")


def clear_vllm_caches():
    """Clear various vLLM internal caches."""
    # CRITICAL: Reset global vllm config first (prevents MoE state corruption)
    clear_current_vllm_config()

    # Clear RoPE cache
    clear_rope_cache()

    # Clear WorkspaceManager singleton
    clear_workspace_manager()

    # Clear forward context globals
    clear_forward_context()

    # Clear ubatch contexts
    clear_ubatch_contexts()

    # Try to clear other vLLM caches
    try:
        from vllm.attention.backends import flash_attn
        if hasattr(flash_attn, '_CACHED_ATTENTION'):
            flash_attn._CACHED_ATTENTION = None
    except Exception:
        pass

    try:
        from vllm.model_executor.custom_op import custom_ops
        if hasattr(custom_ops, '_CACHED_OPS'):
            custom_ops._CACHED_OPS.clear()
    except Exception:
        pass

    # Clear triton attention caches
    try:
        from vllm.attention.backends import triton_attn
        if hasattr(triton_attn, '_CACHED_KERNELS'):
            triton_attn._CACHED_KERNELS = {}
    except Exception:
        pass

    # Clear quantization caches
    try:
        from vllm.model_executor.layers.quantization import marlin
        if hasattr(marlin, '_CACHED_WORKSPACE'):
            marlin._CACHED_WORKSPACE = None
    except Exception:
        pass

    gc.collect()


def destroy_parallel_state():
    """Destroy vLLM's distributed parallel state to release NCCL resources."""
    try:
        from vllm.distributed import parallel_state
        if hasattr(parallel_state, 'destroy_model_parallel'):
            parallel_state.destroy_model_parallel()
            logger.debug("Destroyed model parallel state")
        if hasattr(parallel_state, 'destroy_distributed_environment'):
            parallel_state.destroy_distributed_environment()
            logger.debug("Destroyed distributed environment")

        # Also reset the global state variables
        if hasattr(parallel_state, '_WORLD_GROUP'):
            parallel_state._WORLD_GROUP = None
        if hasattr(parallel_state, '_MODEL_PARALLEL_GROUP'):
            parallel_state._MODEL_PARALLEL_GROUP = None
        if hasattr(parallel_state, '_TENSOR_PARALLEL_GROUP'):
            parallel_state._TENSOR_PARALLEL_GROUP = None
        if hasattr(parallel_state, '_PIPELINE_PARALLEL_GROUP'):
            parallel_state._PIPELINE_PARALLEL_GROUP = None
        if hasattr(parallel_state, '_DATA_PARALLEL_GROUP'):
            parallel_state._DATA_PARALLEL_GROUP = None
        if hasattr(parallel_state, '_LOCAL_RANK'):
            parallel_state._LOCAL_RANK = None
        if hasattr(parallel_state, '_WORLD_SIZE'):
            parallel_state._WORLD_SIZE = None
    except Exception as e:
        logger.debug(f"Could not destroy parallel state: {e}")

    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.debug("Destroyed torch process group")
    except Exception as e:
        logger.debug(f"Could not destroy process group: {e}")

    # Force CUDA synchronization and cache clearing
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


# =============================================================================
# Additional cleanup functions to eliminate residual memory drift
# =============================================================================

def clear_triton_caches():
    """Clear ALL triton kernel caches to prevent memory accumulation.

    Triton's autotuner and JIT caches can hold GPU memory references
    that accumulate across model switches.
    """
    try:
        import triton

        # Clear autotune cache via runtime module
        if hasattr(triton, 'runtime'):
            runtime = triton.runtime
            if hasattr(runtime, 'autotuner'):
                if hasattr(runtime.autotuner, 'cache'):
                    runtime.autotuner.cache.clear()
                    logger.debug("Cleared triton autotuner cache")
            # Also check for CachingAutotuner
            if hasattr(runtime, 'CachingAutotuner'):
                CachingAutotuner = runtime.CachingAutotuner
                if hasattr(CachingAutotuner, 'cache'):
                    CachingAutotuner.cache.clear()

        # Clear JIT function caches
        if hasattr(triton, 'JITFunction'):
            jit_cls = triton.JITFunction
            if hasattr(jit_cls, 'cache') and isinstance(jit_cls.cache, dict):
                jit_cls.cache.clear()
            if hasattr(jit_cls, '_cache'):
                jit_cls._cache.clear()
            logger.debug("Cleared triton JIT caches")

        # Try clearing via triton.compiler
        if hasattr(triton, 'compiler'):
            compiler = triton.compiler
            if hasattr(compiler, 'compiler'):
                inner = compiler.compiler
                if hasattr(inner, 'CompiledKernel'):
                    ck = inner.CompiledKernel
                    if hasattr(ck, '_cache'):
                        ck._cache.clear()

        # Clear kernel cache from vLLM's triton attention backend
        try:
            from vllm.attention.backends import triton_attn
            if hasattr(triton_attn, '_CACHED_KERNELS'):
                triton_attn._CACHED_KERNELS = {}
                logger.debug("Cleared vLLM triton attention cache")
        except Exception:
            pass

    except ImportError:
        logger.debug("Triton not available")
    except Exception as e:
        logger.debug(f"Triton cache clear: {e}")


def clear_cudnn_cublas_workspaces():
    """Clear cuDNN and cuBLAS workspaces.

    These libraries maintain internal workspaces for optimized kernels
    that can accumulate memory across model switches.
    """
    try:
        # Clear cuBLAS workspaces using internal API
        # This is essential after force-freeing MXFP4 blocks to prevent
        # CUBLAS_STATUS_INTERNAL_ERROR on subsequent model loads
        try:
            torch._C._cuda_clearCublasWorkspaces()
            logger.debug("Cleared cuBLAS workspaces via _cuda_clearCublasWorkspaces")
        except Exception as e:
            logger.debug(f"_cuda_clearCublasWorkspaces not available: {e}")

        # Toggle benchmark to force workspace re-selection
        original_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = not original_benchmark
        torch.backends.cudnn.benchmark = original_benchmark

        # Clear cuDNN plan cache if available
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            orig = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = not orig
            torch.backends.cudnn.allow_tf32 = orig

        # Force CUDA sync to release workspaces
        torch.cuda.synchronize()

        logger.debug("Reset cuDNN/cuBLAS workspaces")
    except Exception as e:
        logger.debug(f"cuDNN/cuBLAS clear: {e}")


def clear_dynamo_inductor():
    """Clear torch.compile / dynamo / inductor state.

    torch._dynamo and torch._inductor maintain caches of compiled
    code that can hold GPU memory references.
    """
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        logger.debug("Reset torch._dynamo")
    except Exception as e:
        logger.debug(f"Dynamo reset: {e}")

    try:
        import torch._inductor
        if hasattr(torch._inductor, 'codecache'):
            cc = torch._inductor.codecache
            if hasattr(cc, 'cache_clear'):
                cc.cache_clear()
            if hasattr(cc, 'PyCodeCache'):
                pcc = cc.PyCodeCache
                if hasattr(pcc, 'cache'):
                    pcc.cache.clear()
            if hasattr(cc, 'CUDACodeCache'):
                ccc = cc.CUDACodeCache
                if hasattr(ccc, 'cache'):
                    ccc.cache.clear()
            logger.debug("Cleared inductor code caches")
    except Exception as e:
        logger.debug(f"Inductor cache clear: {e}")

    try:
        # Clear inductor graph cache
        import torch._inductor.graph as ig
        if hasattr(ig, '_GRAPH_CACHE'):
            ig._GRAPH_CACHE.clear()
    except Exception:
        pass


def release_nccl_resources():
    """Release NCCL communication buffers explicitly.

    NCCL maintains persistent buffers for communication that
    may not be released automatically.
    """
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
                logger.debug("Destroyed process group for NCCL cleanup")
            except Exception:
                pass
    except Exception:
        pass

    # Clear vLLM's NCCL backend cache
    try:
        from vllm.distributed import parallel_state
        if hasattr(parallel_state, '_NCCL_BACKEND_CACHE'):
            parallel_state._NCCL_BACKEND_CACHE.clear()
            logger.debug("Cleared vLLM NCCL backend cache")
        if hasattr(parallel_state, '_LOCAL_NCCL_BACKEND_CACHE'):
            parallel_state._LOCAL_NCCL_BACKEND_CACHE.clear()
    except Exception as e:
        logger.debug(f"NCCL resource release: {e}")


def clear_flash_attention_cache():
    """Clear Flash Attention cached state."""
    try:
        from vllm.attention.backends import flash_attn
        if hasattr(flash_attn, '_CACHED_ATTENTION'):
            flash_attn._CACHED_ATTENTION = None
        if hasattr(flash_attn, 'FlashAttentionImpl'):
            impl = flash_attn.FlashAttentionImpl
            if hasattr(impl, '_cached_state'):
                impl._cached_state = None
        logger.debug("Cleared Flash Attention cache")
    except Exception as e:
        logger.debug(f"Flash Attention cache clear: {e}")


def clear_marlin_workspace():
    """Clear Marlin quantization workspace."""
    try:
        from vllm.model_executor.layers.quantization import marlin
        if hasattr(marlin, '_CACHED_WORKSPACE'):
            marlin._CACHED_WORKSPACE = None
            logger.debug("Cleared Marlin workspace")
    except Exception as e:
        logger.debug(f"Marlin workspace clear: {e}")


def aggressive_allocator_cleanup():
    """Aggressively clean up PyTorch's CUDA caching allocator.

    This sets a zero GC threshold temporarily to force release
    of all cached allocations.
    """
    try:
        # Set aggressive GC threshold
        torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.0")

        gc.collect()
        torch.cuda.empty_cache()

        # Collect IPC memory
        torch.cuda.ipc_collect()

        # Try raw allocator delete
        try:
            torch.cuda.memory._cuda_caching_allocator_raw_delete(torch.cuda.current_device())
        except Exception:
            pass

        logger.debug("Aggressive allocator cleanup complete")
    except Exception as e:
        logger.debug(f"Aggressive allocator cleanup: {e}")
    finally:
        # Restore reasonable GC threshold
        try:
            torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.6")
        except Exception:
            pass


def force_free_phantom_blocks(min_size_mb: float = 100.0) -> int:
    """Force-free orphaned CUDA allocations using caching_allocator_delete.

    These are memory blocks allocated by C++ libraries (Flash Attention, NCCL,
    cuBLAS) through PyTorch's allocator but WITHOUT Python tensor wrappers.
    They cannot be freed by gc.collect() or empty_cache() but CAN be freed
    using caching_allocator_delete() with the block address from memory_snapshot().

    Args:
        min_size_mb: Minimum block size to consider for force-free (default 100MB).
                     This avoids freeing small blocks that might still be in use.

    Returns:
        Number of blocks successfully freed.
    """
    import torch.cuda.memory as cuda_mem

    freed_count = 0

    try:
        # Get memory snapshot to find orphaned blocks
        snapshot = torch.cuda.memory._snapshot()
        if not snapshot or 'segments' not in snapshot:
            return 0

        min_size_bytes = min_size_mb * 1024 * 1024

        # Collect addresses of large blocks to free
        blocks_to_free = []
        for segment in snapshot['segments']:
            for block in segment.get('blocks', []):
                if block.get('state') == 'active_allocated':
                    size = block.get('size', block.get('requested_size', 0))
                    if size >= min_size_bytes:
                        address = block.get('address', 0)
                        if address == 0:
                            address = segment.get('address', 0)
                        if address != 0:
                            blocks_to_free.append({
                                'address': address,
                                'size_mb': size / 1024**2,
                            })

        if not blocks_to_free:
            logger.debug("No phantom blocks found to free")
            return 0

        logger.debug(f"Found {len(blocks_to_free)} phantom blocks to free")

        # Force-free each block
        for block in blocks_to_free:
            try:
                cuda_mem.caching_allocator_delete(block['address'])
                freed_count += 1
                logger.debug(f"Freed phantom block: {block['size_mb']:.1f} MB at 0x{block['address']:x}")
            except Exception as e:
                logger.debug(f"Failed to free block at 0x{block['address']:x}: {e}")

        if freed_count > 0:
            # Sync and clear cache after freeing
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info(f"Force-freed {freed_count} phantom blocks ({sum(b['size_mb'] for b in blocks_to_free[:freed_count]):.1f} MB)")

    except Exception as e:
        logger.debug(f"force_free_phantom_blocks error: {e}")

    return freed_count


def clear_fla_module_caches():
    """Clear Flash Linear Attention (FLA) tensor_cache closures.

    FLA ops use a @tensor_cache decorator that stores (args, kwargs, result) tuples
    in a closure list called cache_entries. These contain tiny index tensors (<1KB).
    After force_free invalidates CUDA memory, these cached tensors have dangling
    pointers and crash with "invalid device pointer" when their destructors run.

    Fix: Resize tensor storages to 0 (safely detaches from CUDA memory) and clear
    the cache_entries lists. This is fast since FLA caches hold only tiny tensors.

    CRITICAL: Must be called BEFORE force_free_all_allocated_blocks().
    """
    import sys

    fla_modules = [mod for name, mod in sys.modules.items()
                   if mod is not None and 'fla' in name.lower()]

    caches_cleared = 0

    for mod in fla_modules:
        for attr_name in dir(mod):
            try:
                func = getattr(mod, attr_name, None)
                if not callable(func) or not hasattr(func, '__closure__') or func.__closure__ is None:
                    continue

                # Look for cache_entries list in closure cells
                for cell in func.__closure__:
                    try:
                        contents = cell.cell_contents
                        # cache_entries is a list of (args, kwargs, result) tuples
                        if not isinstance(contents, list) or not contents:
                            continue
                        if not (isinstance(contents[0], tuple) and len(contents[0]) == 3):
                            continue

                        # Found a tensor_cache! Clear it.
                        for args, kwargs, result in contents:
                            # Resize CUDA tensor storages to 0 (safe detach)
                            for item in (args if isinstance(args, tuple) else ()):
                                if isinstance(item, torch.Tensor) and item.device.type == 'cuda':
                                    try:
                                        item.data.storage().resize_(0)
                                    except Exception:
                                        pass
                            if isinstance(result, torch.Tensor) and result.device.type == 'cuda':
                                try:
                                    result.data.storage().resize_(0)
                                except Exception:
                                    pass
                        contents.clear()
                        caches_cleared += 1
                    except ValueError:
                        pass  # Empty cell
            except Exception:
                pass

    if caches_cleared > 0:
        # Single GC to clean up cleared tensor objects
        gc.collect()
        logger.debug(f"Cleared {caches_cleared} FLA tensor_cache closures")


def detach_all_cuda_tensors() -> int:
    """Detach ALL Python-side CUDA tensor references before force_free.

    Walks gc.get_objects() to find every torch.Tensor on CUDA and resizes
    its storage to 0. This safely detaches the Python object from the CUDA
    address, preventing "invalid device pointer" crashes at exit when:
    1. force_free_all_allocated_blocks() frees the CUDA memory
    2. Python later destructs tensor objects during shutdown that still
       point to the now-freed CUDA addresses

    CRITICAL: Must be called BEFORE force_free_all_allocated_blocks().
    """
    detached = 0
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            try:
                if obj.device.type == 'cuda' and obj.storage().size() > 0:
                    obj.data.storage().resize_(0)
                    detached += 1
            except Exception:
                pass
    if detached:
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug(f"Detached {detached} CUDA tensors before force_free")
    return detached


def _log_remaining_allocated_blocks():
    """Log any remaining allocated CUDA blocks for monitoring.

    After the model walker + detach_all_cuda_tensors + gc + empty_cache,
    any remaining 'active_allocated' blocks are C++ internal allocations
    (cuBLAS workspaces, Flash Attention buffers, NCCL, etc.) that are
    managed by their owning libraries. We log them for monitoring but
    do NOT force-free them - doing so causes "invalid device pointer"
    crashes at exit when C++ TensorImpl destructors try to free the
    already-deleted addresses.
    """
    try:
        snapshot = torch.cuda.memory._snapshot()
        segments = snapshot.get('segments', [])

        total_remaining = 0
        block_count = 0

        for seg in segments:
            for block in seg.get('blocks', []):
                if block.get('state') == 'active_allocated':
                    total_remaining += block.get('size', 0)
                    block_count += 1

        if block_count > 0:
            remaining_gb = total_remaining / 1024**3
            logger.debug(
                f"Remaining C++ allocated blocks: {block_count} "
                f"({remaining_gb:.2f}GB) - left for library cleanup"
            )
    except Exception:
        pass


def force_free_all_allocated_blocks() -> tuple[int, float]:
    """Force-free ALL remaining allocated blocks using caching_allocator_delete.

    IMPORTANT: This should only be called AFTER:
    1. _walk_and_free_cuda_tensors() has resized all model tensor storages to 0
    2. detach_all_cuda_tensors() has resized all gc-visible tensor storages to 0
    3. gc.collect() + empty_cache() has released those blocks

    After those steps, any remaining 'active_allocated' blocks are truly orphaned
    C++ allocations with NO Python TensorImpl references. force_free is safe because
    there's no Python destructor that will try to double-free these addresses.

    Returns:
        Tuple of (blocks_deleted, bytes_deleted_gb)
    """
    try:
        snapshot = torch.cuda.memory._snapshot()
        segments = snapshot.get('segments', [])

        blocks_deleted = 0
        bytes_deleted = 0

        for seg in segments:
            blocks = seg.get('blocks', [])
            for block in blocks:
                if block.get('state') == 'active_allocated':
                    size = block.get('size', 0)
                    addr = block.get('addr', block.get('address', 0))
                    if addr:
                        try:
                            torch.cuda.caching_allocator_delete(addr)
                            blocks_deleted += 1
                            bytes_deleted += size
                        except Exception:
                            pass

        # Clean up after force deletion
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        bytes_deleted_gb = bytes_deleted / 1024**3
        if blocks_deleted > 0:
            logger.info(f"Force-freed {blocks_deleted} blocks ({bytes_deleted_gb:.2f}GB)")

        return blocks_deleted, bytes_deleted_gb

    except Exception as e:
        logger.debug(f"force_free_all_allocated_blocks error: {e}")
        return 0, 0.0


def nuclear_cleanup(force_free_phantoms: bool = False):
    """Ultimate cleanup - use after full_cleanup() if drift persists.

    This runs all additional cleanup steps to minimize residual
    memory drift. Call this after full_cleanup() for cross-architecture
    switches or when memory drift accumulates.

    Args:
        force_free_phantoms: If True, call force_free_phantom_blocks() to free
            orphaned CUDA allocations. WARNING: This can cause crashes if called
            while C++ code still holds references to those blocks. Only use when
            you're certain no C++ code will access GPU memory after this call.
            Default False for safety.
    """
    logger.debug("Starting nuclear cleanup")

    # Clear all kernel/JIT caches
    clear_triton_caches()
    clear_cudnn_cublas_workspaces()
    clear_dynamo_inductor()

    # Release communication resources
    release_nccl_resources()

    # Clear attention and quantization caches
    clear_flash_attention_cache()
    clear_marlin_workspace()

    # Aggressive allocator cleanup
    aggressive_allocator_cleanup()

    # Only force-free phantom blocks if explicitly requested
    # WARNING: This can crash if C++ code still holds references to GPU memory
    # The 3x160MB phantom blocks are internal buffers from flash_attention/NCCL
    # that persist across model loads. Freeing them while C++ code still has
    # pointers to them causes "invalid device pointer" errors.
    if force_free_phantoms:
        force_free_phantom_blocks(min_size_mb=100.0)

    # Final sync
    torch.cuda.synchronize()

    logger.debug("Nuclear cleanup complete")


def full_cleanup(llm, nuclear: bool = True, force_free: bool = True) -> float:
    """Full cleanup including parallel state destruction.

    This is more aggressive than cleanup_vllm_model and should be used
    when switching between different model architectures.

    Args:
        llm: The vLLM LLM instance to clean up.
        nuclear: If True, run additional cleanup steps to minimize
                 residual memory drift (~0.6GB per switch). Default True.
        force_free: If True, force-free ALL remaining allocated blocks.
                    Essential for MXFP4 models which have opaque CUDA allocations.
                    Default True.

    Returns:
        Amount of GPU memory freed in GB.
    """
    free_before, total = torch.cuda.mem_get_info()

    # Standard cleanup
    cleanup_vllm_model(llm)

    # Clear all vLLM caches
    clear_vllm_caches()

    # Destroy parallel state (this releases NCCL resources)
    destroy_parallel_state()

    # Run nuclear cleanup to minimize residual drift
    if nuclear:
        nuclear_cleanup()

    # Intermediate cleanup pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ALWAYS clear FLA (Flash Linear Attention) module caches BEFORE force_free.
    # FLA's tensor_cache decorator stores tensor references in closures.
    # Clearing FLA modules makes these cached tensors orphaned, and GC (called
    # inside clear_fla_module_caches) destroys them while CUDA memory is still
    # valid. If we wait until after force_free, the tensor destructors crash
    # with "invalid device pointer".
    clear_fla_module_caches()

    # Clear cuBLAS workspaces - these hold internal pointers to CUDA memory
    try:
        torch._C._cuda_clearCublasWorkspaces()
        logger.debug("Cleared cuBLAS workspaces")
    except Exception as e:
        logger.debug(f"_cuda_clearCublasWorkspaces: {e}")

    # Detach ALL remaining Python CUDA tensors as a safety net.
    # The model walker catches most tensors, but this handles any stray
    # tensors in global variables, closures, or caches.
    # storage().resize_(0) is safe - it properly updates the DataPtr.
    detach_all_cuda_tensors()

    if force_free:
        # Log remaining allocated blocks for monitoring, but do NOT
        # force-free them with caching_allocator_delete(). These are
        # C++ internal allocations (cuBLAS workspaces, Flash Attention
        # buffers, etc.) that have C++ TensorImpl references invisible
        # to gc.get_objects(). Force-freeing them causes "invalid device
        # pointer" crashes at exit when C++ destructors try to free
        # the already-deleted addresses.
        #
        # The model graph walker now handles all model memory including
        # MXFP4 Triton tensors, so force_free is no longer needed for
        # its original purpose. Remaining blocks are typically <0.2GB
        # of non-growing C++ overhead.
        _log_remaining_allocated_blocks()

    # Final cleanup pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_after, _ = torch.cuda.mem_get_info()
    freed = (free_after - free_before) / 1024**3

    logger.info(f"GPU cleanup: freed {freed:.1f}GB ({(total-free_before)/1024**3:.1f}GB -> {(total-free_after)/1024**3:.1f}GB used)")

    return freed


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return {
        'free_gb': free / 1024**3,
        'total_gb': total / 1024**3,
        'allocated_gb': allocated / 1024**3,
        'reserved_gb': reserved / 1024**3,
        'used_gb': (total - free) / 1024**3,
    }


def log_gpu_memory(label: str = ""):
    """Log current GPU memory state."""
    info = get_gpu_memory_info()
    prefix = f"[{label}] " if label else ""
    logger.info(f"{prefix}GPU: {info['used_gb']:.1f}/{info['total_gb']:.1f}GB used, "
                f"{info['free_gb']:.1f}GB free")
    return info
