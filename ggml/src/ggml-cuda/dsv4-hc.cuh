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

static constexpr int DSV4_HC = 4;

// The comb math is written with the rounding intrinsics as dsv4_hc_comb_f32 was compiled (an FFMA for the two affine
//   steps, products and sums rounded apart elsewhere), so that it rounds the same in any kernel that inlines it.
static __device__ __forceinline__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum = __fadd_rn(sum, comb[idst + DSV4_HC*isrc]);
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] = __fmul_rn(comb[idst + DSV4_HC*isrc], inv_sum);
        }
    }
}

static __device__ __forceinline__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum = __fadd_rn(sum, comb[idst + DSV4_HC*isrc]);
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] = __fmul_rn(comb[idst + DSV4_HC*isrc], inv_sum);
        }
    }
}

// The DSV4_HC_COMB of one token: its mixes at stride sm0 (the comb from row 2*DSV4_HC), comb[idst + DSV4_HC*isrc]. The
//   op's kernel and the fused kernels that compute it in place share this.
static __device__ __forceinline__ void dsv4_hc_comb_one(const float * mixes, const int64_t sm0, const float scale_comb,
        const float * base, const int64_t sb0, const float eps, const int32_t n_iter, float * comb) {
    constexpr int comb_offset = 2*DSV4_HC;

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = __fmaf_rn(mixes[(comb_offset + idx)*sm0], scale_comb, base[(comb_offset + idx)*sb0]);
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(__fsub_rn(comb[idx], max));
            comb[idx] = v;
            sum = __fadd_rn(sum, v);
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = __fmaf_rn(comb[idx], inv_sum, eps);
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }
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

// qwen4exp, one token: SCALE -> SILU -> MUL_MAT (the q8_0 up projection of a hyper-connection mix) and the gated
//   DSV4_HC_PRE that reads it as one launch; pre is the DSV4_HC_PRE node, the views between are not computed
bool ggml_cuda_should_fuse_dsv4_hc_up_pre(const ggml_tensor * scale, const ggml_tensor * silu, const ggml_tensor * mm,
        const ggml_tensor * pre);
void ggml_cuda_op_dsv4_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, const ggml_tensor * mm,
        ggml_tensor * pre);
