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
