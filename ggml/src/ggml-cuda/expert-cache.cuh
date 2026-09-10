#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

// Expert cache in device memory for MoE weights that stay in host memory.
//
// The pool tensor holds cap experts of this device's slice of the weight. MUL_MAT_ID finds the cache
// through the pool tensor's extra and, before the matmul, runs two kernels on the compute stream: one
// block walks the routed ids against a device-resident LRU and lists the misses, then a grid copies
// the missing experts straight from pinned host memory into their pool slots and the matmul reads the
// pool through the remapped ids. No host round trip, so the whole layer can stay in one CUDA graph.
// The LRU is deterministic, so every device of a tensor-parallel group assigns the same slots.

struct ggml_cuda_expert_cache;

// host_buf/host_buf_size: the ggml buffer holding the weight, registered with CUDA once (page-locked, mapped).
// host_ptr: the weight's data. chunk_full/chunk_dev/offset_dev/n_chunks describe this device's slice of one
// expert, see ggml_backend_meta_set_tensor_async_dev. Returns nullptr when the memory cannot be registered.
void * ggml_cuda_expert_cache_create(int device, const void * host_buf, size_t host_buf_size,
    const void * host_ptr, int64_t n_expert, int cap, size_t expert_bytes, size_t chunk_full, size_t chunk_dev,
    size_t offset_dev, int64_t n_chunks, int max_ids);
void ggml_cuda_expert_cache_free(void * cache);

bool ggml_cuda_expert_cache_is(const ggml_tensor * pool);

// Gathers the experts routed by ids into pool (src0 of the MUL_MAT_ID) and fills ids_remap with a tensor
// like ids whose values are pool slots.
void ggml_cuda_expert_cache_gather(ggml_backend_cuda_context & ctx, const ggml_tensor * pool, const ggml_tensor * ids, ggml_tensor * ids_remap);
