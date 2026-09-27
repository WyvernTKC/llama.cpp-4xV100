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

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache,
                                              const ggml_cuda_gated_delta_net_l2 * l2 = nullptr);
