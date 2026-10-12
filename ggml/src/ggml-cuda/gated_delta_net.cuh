#pragma once

#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// q and k before their l2 norm, rms_norm(x, eps) * post_scale + post_bias per head (build_gdn_l2_norm), for the kernel
//   to apply itself
struct ggml_cuda_gated_delta_net_l2 {
    const ggml_tensor * q = nullptr;
    const ggml_tensor * k = nullptr;
    float eps        = 0.0f;
    float post_scale = 1.0f;
    float post_bias  = 0.0f;
};

// the start state read from the cache row of each sequence instead of the GET_ROWS that gathers it, and written to
//   the rollback slot a cpy would fill (see ggml_cuda_gdn_gather_fusion)
struct ggml_cuda_gated_delta_net_gather {
    const float   * states     = nullptr; // cache, one row of S_v*S_v*H floats per cell
    const int32_t * ids        = nullptr; // cache row of each sequence
    int64_t         row_stride = 0;       // floats between cache rows
    float         * pre        = nullptr; // rollback slot that gets the start state, or nullptr
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache,
                                              const ggml_cuda_gated_delta_net_l2 * l2 = nullptr,
                                              const ggml_cuda_gated_delta_net_gather * gather = nullptr);

// chunked prefill (gated_delta_net_chunked.cu): Volta tensor cores, scalar gate, S_v 128; writes the final state only
struct ggml_cuda_gated_delta_net_chunked_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * s;         // start state, or the cache when ids is set
    float       * dst;       // attention output
    float       * state;     // final state
    const int32_t * ids;     // gathered: cache row of each sequence, else nullptr
    int64_t       s_row_stride;
    float       * s_pre;     // nullptr, or where the start state goes
    int64_t H, n_tokens, n_seqs;
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    int64_t neqk1, rq3;
    float   scale;
};

bool   ggml_cuda_gated_delta_net_chunked_supported(int device, int64_t S_v, bool kda, int64_t n_tokens);
size_t ggml_cuda_gated_delta_net_chunked_scratch_size(int64_t H, int64_t n_tokens, int64_t n_seqs);
void   ggml_cuda_gated_delta_net_chunked(const ggml_cuda_gated_delta_net_chunked_args & args, char * scratch, cudaStream_t stream);
