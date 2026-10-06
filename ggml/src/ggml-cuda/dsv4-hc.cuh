#include "common.cuh"
#include "ggml.h"

#include "unary.cuh"

// the post weight of a hyper-connection combine: with gated, the SCALE -> SIGMOID -> SCALE that made it from raw, in the
//   order and form of those ops
template <bool gated>
static __device__ __forceinline__ float dsv4_hc_post_weight(float p, const float ps_in, const float pb_in,
        const float ps_out, const float pb_out) {
    if constexpr (gated) {
        p = ps_out * ggml_cuda_op_sigmoid_single(ps_in * p + pb_in) + pb_out;
    }
    return p;
}

// one value of an identity-comb hyper-connection combine: x * p + the stream's residual
static __device__ __forceinline__ float dsv4_hc_post_identity(const float x, const float p, const float r) {
    float sum = x * p;
    sum += r;
    return sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// dst = DSV4_HC_POST whose post weights are SCALE(sigmoid(SCALE(raw))): scale_in and scale_out are those SCALE nodes, the
//   weights are computed from raw in the kernel
void ggml_cuda_op_dsv4_hc_post_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * scale_in, const ggml_tensor * scale_out);

// DeepSeek-V4: MUL -> ADD -> SIGMOID -> SCALE of the mixes (the stream weights) and DSV4_HC_PRE (not gated) with them in
//   one launch, each step rounded as its own kernel would; dst is the DSV4_HC_PRE node
void ggml_cuda_op_dsv4_hc_pre_coef(ggml_backend_cuda_context & ctx, const ggml_tensor * mul_w, const ggml_tensor * add_w,
                                   const ggml_tensor * scale_w, ggml_tensor * dst);
