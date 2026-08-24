#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// One-shot peer-to-peer AllReduce for small messages.
//
// NCCL's ring AllReduce costs ~35 us of fixed overhead plus ~6.75 us per ring step, so a
// hidden-state sized message (tens of KB) is entirely dominated by the fixed term: measured
// 49 us at 2 ranks and 76 us at 4 on NVLink-connected V100s. Batch-1 tensor-parallel decode
// issues two AllReduces per layer, which made the collective ~35% of token generation time.
//
// This path trades bandwidth for latency: every rank pushes its whole payload straight into
// peer-mapped staging on all other ranks, synchronises once, then reduces locally. That is
// n_devices times the traffic of a ring, which is irrelevant at these sizes and removes all
// but one of the round trips. Above the size threshold the ring wins and this returns false
// so the caller falls back to NCCL.

struct ggml_cuda_p2p_ar;

// Returns nullptr if the path is unavailable: fewer than 2 devices, no peer access between
// some pair, or allocation failure. The caller keeps using NCCL in that case.
ggml_cuda_p2p_ar * ggml_cuda_p2p_ar_init(const int * devices, size_t n_devices);

void ggml_cuda_p2p_ar_free(ggml_cuda_p2p_ar * ar);

// In-place AllReduce (sum) over tensors[0..n_devices-1], tensors[i] living on devices[i].
// Ranks whose tensor lacks GGML_TENSOR_FLAG_COMPUTE contribute nothing but still receive the
// sum, matching the NCCL path's zero-fill semantics.
//
// Returns false without enqueueing anything if this call is not eligible (too large, not F32,
// misaligned, non-contiguous); the caller must then fall back. Returns true once the work is
// enqueued on each device's stream.
bool ggml_cuda_p2p_ar_allreduce(ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors);
