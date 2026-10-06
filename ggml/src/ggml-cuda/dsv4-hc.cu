#include "common.cuh"
#include "unary.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

// q8_copy: one token, also the q8_1 copy of the output (padded to k_padded) for the MMVQ matmul that reads it
template <bool gated, bool q8_copy>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale,
        block_q8_1 * q8,
        int64_t k_padded) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        if constexpr (q8_copy) {
            // whole warps, nr being a multiple of 32
            if (ir < k_padded) {
                quantize_q8_1_lane(0.0f, q8 + ir/QK8_1, ir % QK8_1);
            }
        }
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    const float v = scale * sum;
    dst[i0*sd0 + it*sd1] = v;
    if constexpr (q8_copy) {
        quantize_q8_1_lane(v, q8 + ir/QK8_1, ir % QK8_1);
    }
}

// DSV4_HC_PRE (not gated) with its weights computed in place from the mixes: MUL, ADD, SIGMOID, SCALE rounded as
//   mul_add_sigmoid_scale_f32 does, then the sum as dsv4_hc_pre_f32 forms it. hc <= 4.
template <bool q8_copy>
static __global__ void dsv4_hc_pre_coef_f32(
        const float * x, const float * mixes, const float * m, const float * a, float * dst, const int64_t n_embd,
        const int hc, const int64_t n_tokens, const int64_t sx0, const int64_t sx1, const int64_t sx2,
        const int64_t smix1, const int ne_m, const int ne_a, const float ws, const float wb, const int64_t sd0,
        const int64_t sd1, const float scale, block_q8_1 * q8, const int64_t k_padded) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        if constexpr (q8_copy) {
            // whole warps, nr being a multiple of 32
            if (ir < k_padded) {
                quantize_q8_1_lane(0.0f, q8 + ir/QK8_1, ir % QK8_1);
            }
        }
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        const float sv = ggml_cuda_op_sigmoid_single(__fadd_rn(__fmul_rn(mixes[it*smix1 + ih], m[ih % ne_m]), a[ih % ne_a]));
        const float wv = ws * sv + wb;
        sum += xv * wv;
    }

    const float v = scale * sum;
    dst[i0*sd0 + it*sd1] = v;
    if constexpr (q8_copy) {
        quantize_q8_1_lane(v, q8 + ir/QK8_1, ir % QK8_1);
    }
}

template <bool has_comb, bool gated>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float ps_in, float pb_in, float ps_out, float pb_out) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    const float p = dsv4_hc_post_weight<gated>(post[idst*sp0 + it*sp1], ps_in, pb_in, ps_out, pb_out);
    float sum;
    if constexpr (has_comb) {
        sum = x[i0*sx0 + it*sx1] * p;
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum = dsv4_hc_post_identity(x[i0*sx0 + it*sx1], p, residual[i0*sr0 + idst*sr1 + it*sr2]);
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;

    // one token: the q8_1 copy of the output for the MMVQ matmul that reads it
    int64_t      ne_padded = 0;
    block_q8_1 * q8        = nullptr;
    if (n_tokens == 1 && n_embd % QK8_1 == 0 && nbd0 == sizeof(float)) {
        q8 = (block_q8_1 *) ctx.q8_1_emit_alloc(dst, ne_padded);
    }

    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((std::max(nr, ne_padded) + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = q8 ? (gated ? dsv4_hc_pre_f32<true, true> : dsv4_hc_pre_f32<false, true>) :
                       (gated ? dsv4_hc_pre_f32<true, false> : dsv4_hc_pre_f32<false, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale, q8, ne_padded);
}

void ggml_cuda_op_dsv4_hc_pre_coef(ggml_backend_cuda_context & ctx, const ggml_tensor * mul_w, const ggml_tensor * add_w,
                                   const ggml_tensor * scale_w, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * mixes = mul_w->src[0];
    const ggml_tensor * m     = mul_w->src[1];
    const ggml_tensor * a     = add_w->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && mixes->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[1] <= 4 && mixes->ne[0] == x->ne[1] && mixes->nb[0] == sizeof(float));

    float ws = 1.0f;
    float wb = 0.0f;
    memcpy(&ws, (const float *) scale_w->op_params + 0, sizeof(float));
    memcpy(&wb, (const float *) scale_w->op_params + 1, sizeof(float));
    const float scale = ggml_get_op_params_f32(dst, 0);

    const int64_t n_embd   = x->ne[0];
    const int     hc       = (int) x->ne[1];
    const int64_t n_tokens = x->ne[2];
    const int64_t nr       = n_embd * n_tokens;

    // one token: the q8_1 copy of the output for the MMVQ matmul that reads it
    int64_t      ne_padded = 0;
    block_q8_1 * q8        = nullptr;
    if (n_tokens == 1 && n_embd % QK8_1 == 0 && dst->nb[0] == sizeof(float)) {
        q8 = (block_q8_1 *) ctx.q8_1_emit_alloc(dst, ne_padded);
    }

    const int block_size = 256;
    const dim3 grid_dims((std::max(nr, ne_padded) + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, dim3(block_size, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(q8 ? dsv4_hc_pre_coef_f32<true> : dsv4_hc_pre_coef_f32<false>, launch_params,
        (const float *) x->data, (const float *) mixes->data, (const float *) m->data, (const float *) a->data,
        (float *) dst->data, n_embd, hc, n_tokens,
        (int64_t) (x->nb[0]/sizeof(float)), (int64_t) (x->nb[1]/sizeof(float)), (int64_t) (x->nb[2]/sizeof(float)),
        (int64_t) (mixes->nb[1]/sizeof(float)), (int) m->ne[0], (int) a->ne[0], ws, wb,
        (int64_t) (dst->nb[0]/sizeof(float)), (int64_t) (dst->nb[1]/sizeof(float)), scale, q8, ne_padded);
}

static void ggml_cuda_op_dsv4_hc_post_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * scale_in, const ggml_tensor * scale_out) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const bool          gated    = scale_in != nullptr;
    const ggml_tensor * post     = gated ? scale_in->src[0] : dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    float ps_in = 1.0f, pb_in = 0.0f, ps_out = 1.0f, pb_out = 0.0f;
    if (gated) {
        memcpy(&ps_in,  (const float *) scale_in->op_params  + 0, sizeof(float));
        memcpy(&pb_in,  (const float *) scale_in->op_params  + 1, sizeof(float));
        memcpy(&ps_out, (const float *) scale_out->op_params + 0, sizeof(float));
        memcpy(&pb_out, (const float *) scale_out->op_params + 1, sizeof(float));
    }

    auto kernel = comb ? (gated ? dsv4_hc_post_f32<true, true>  : dsv4_hc_post_f32<true, false>) :
                         (gated ? dsv4_hc_post_f32<false, true> : dsv4_hc_post_f32<false, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            ps_in, pb_in, ps_out, pb_out);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_dsv4_hc_post_impl(ctx, dst, nullptr, nullptr);
}

void ggml_cuda_op_dsv4_hc_post_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * scale_in, const ggml_tensor * scale_out) {
    ggml_cuda_op_dsv4_hc_post_impl(ctx, dst, scale_in, scale_out);
}

// qwen4exp, one token: SCALE -> SILU of the low-rank mix, its up projection (q8_0) and the gated DSV4_HC_PRE that reads
//   it in one launch. A block owns 32 columns of all hc streams: their rows go to shared memory with coalesced loads, a
//   lane sums one row, and warp 0 writes the 32 mixed values and their q8_1 block. The SILU input is quantized as
//   scale_unary_kernel does it, the mix is formed as in dsv4_hc_pre_f32; a row sums its blocks in order, which is not
//   the order of mul_mat_vec_q, so the result is not bitwise that of the three launches.
template <int hc, int k_blocks>
__launch_bounds__(WARP_SIZE*hc, 1)
static __global__ void dsv4_hc_up_pre_q8_0(
        const float * lo, const char * w_up, const float * x, float * dst, block_q8_1 * q8, const float lo_scale,
        const float lo_bias, const int n_embd, const int64_t sx1, const float scale) {
    static_assert(k_blocks % 2 == 0, "a row must be whole ints");
    constexpr int row_ints   = k_blocks*sizeof(block_q8_0)/sizeof(int);
    constexpr int chunk_int4 = WARP_SIZE*row_ints/4; // the 32 rows of one stream
    static_assert((WARP_SIZE*row_ints) % 4 == 0, "32 rows must be whole int4");

    __shared__ int4       rows_s4[hc*chunk_int4];
    __shared__ block_q8_1 y_s[k_blocks];
    __shared__ float      g_s[hc][WARP_SIZE];

    ggml_cuda_pdl_lc();
    const int lane = threadIdx.x;
    const int s    = threadIdx.y;
    const int j0   = blockIdx.x*WARP_SIZE;

    if (j0 >= n_embd) {
        // the padding blocks of the q8_1 copy
        if (q8 && s == 0) {
            quantize_q8_1_lane(0.0f, q8 + blockIdx.x, lane);
        }
        return;
    }

    // the weights are not written by an earlier kernel, so they load before the dependency wait
    for (int i = s*WARP_SIZE + lane; i < hc*chunk_int4; i += hc*WARP_SIZE) {
        const int is = i / chunk_int4;
        const int4 * src = (const int4 *) (w_up + ((int64_t) is*n_embd + j0)*row_ints*sizeof(int));
        rows_s4[i] = src[i % chunk_int4];
    }

    ggml_cuda_pdl_sync();
    for (int b = s; b < k_blocks; b += hc) {
        const float v = ggml_cuda_op_silu_single(lo_scale*lo[b*QK8_1 + lane] + lo_bias);
        quantize_q8_1_lane(v, y_s + b, lane);
    }
    __syncthreads();

    // a row starts on a word; a block of 34 bytes starts on one when its index is even
    const int * row = (const int *) rows_s4 + (s*WARP_SIZE + lane)*row_ints;
    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < k_blocks; ++b) {
        const int w0 = b*(int) sizeof(block_q8_0)/(int) sizeof(int);
        int   q[QI8_0];
        float d;
        if (b % 2 == 0) {
            d = __half2float(__ushort_as_half((unsigned short) (row[w0] & 0xFFFF)));
#pragma unroll
            for (int i = 0; i < QI8_0; ++i) {
                q[i] = __funnelshift_r(row[w0 + i], row[w0 + i + 1], 16);
            }
        } else {
            d = __half2float(__ushort_as_half((unsigned short) ((unsigned int) row[w0] >> 16)));
#pragma unroll
            for (int i = 0; i < QI8_0; ++i) {
                q[i] = row[w0 + 1 + i];
            }
        }
        const int * u = (const int *) y_s[b].qs;
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < QI8_0; ++i) {
            sumi = ggml_cuda_dp4a(q[i], u[i], sumi);
        }
        acc += d*__low2float(y_s[b].ds) * (float) sumi;
    }
    g_s[s][lane] = acc;
    __syncthreads();

    if (s == 0) {
        const int j = j0 + lane;
        float sum = 0.0f;
#pragma unroll
        for (int ih = 0; ih < hc; ++ih) {
            const float xv = x[j + ih*sx1];
            const float wv = 1.0f / (1.0f + expf(-g_s[ih][lane]));
            sum += xv * wv;
        }
        const float v = scale * sum;
        dst[j] = v;
        if (q8) {
            quantize_q8_1_lane(v, q8 + blockIdx.x, lane);
        }
    }
}

static constexpr int DSV4_HC_UP_PRE_HC       = 4;
static constexpr int DSV4_HC_UP_PRE_K_BLOCKS = 10;

bool ggml_cuda_should_fuse_dsv4_hc_up_pre(const ggml_tensor * scale, const ggml_tensor * silu, const ggml_tensor * mm,
        const ggml_tensor * pre) {
    if (scale->op != GGML_OP_SCALE || silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
            mm->op != GGML_OP_MUL_MAT || pre->op != GGML_OP_DSV4_HC_PRE || silu->src[0] != scale || mm->src[1] != silu ||
            ggml_get_op_params_i32(pre, 1) == 0) {
        return false;
    }
    const ggml_tensor * lo = scale->src[0];
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * x  = pre->src[0];
    const ggml_tensor * g  = pre->src[1];
    const int64_t n_embd = x->ne[0];

    // the gate is the matmul output read as n_embd x hc; the rows of 32 columns of a stream load as whole int4
    return (g == mm || (g->view_src == mm && g->view_offs == 0)) && ggml_is_contiguous(g) && ggml_is_contiguous(mm) &&
        lo->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32 && silu->type == GGML_TYPE_F32 &&
        mm->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && pre->type == GGML_TYPE_F32 &&
        w->type == GGML_TYPE_Q8_0 && w->ne[0] == DSV4_HC_UP_PRE_K_BLOCKS*QK8_0 && ggml_is_contiguous(w) &&
        x->ne[1] == DSV4_HC_UP_PRE_HC && x->ne[2] == 1 && x->ne[3] == 1 && w->ne[1] == n_embd*DSV4_HC_UP_PRE_HC &&
        mm->ne[1] == 1 && ggml_is_contiguous(lo) && ggml_nelements(lo) == w->ne[0] &&
        x->nb[0] == sizeof(float) && pre->nb[0] == sizeof(float) && n_embd % WARP_SIZE == 0 && n_embd <= INT_MAX &&
        (uintptr_t) w->data % 16 == 0 && (n_embd*w->nb[1]) % 16 == 0 && (WARP_SIZE*w->nb[1]) % 16 == 0 &&
        !ggml_cuda_tensors_overlap(pre, lo) && !ggml_cuda_tensors_overlap(pre, x) && !ggml_cuda_tensors_overlap(pre, w);
}

void ggml_cuda_op_dsv4_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, const ggml_tensor * mm,
        ggml_tensor * pre) {
    const ggml_tensor * lo = scale->src[0];
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * x  = pre->src[0];
    const int n_embd = (int) x->ne[0];

    float lo_scale;
    float lo_bias;
    memcpy(&lo_scale, (const float *) scale->op_params + 0, sizeof(float));
    memcpy(&lo_bias,  (const float *) scale->op_params + 1, sizeof(float));

    int64_t      ne_padded = 0;
    block_q8_1 * q8        = (block_q8_1 *) ctx.q8_1_emit_alloc(pre, ne_padded);

    const dim3 block_dims(WARP_SIZE, DSV4_HC_UP_PRE_HC, 1);
    const dim3 grid_dims(std::max<int64_t>(n_embd, ne_padded)/WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());
    ggml_cuda_kernel_launch(dsv4_hc_up_pre_q8_0<DSV4_HC_UP_PRE_HC, DSV4_HC_UP_PRE_K_BLOCKS>, launch_params,
            (const float *) lo->data, (const char *) w->data, (const float *) x->data, (float *) pre->data, q8,
            lo_scale, lo_bias, n_embd, (int64_t) (x->nb[1] / sizeof(float)), ggml_get_op_params_f32(pre, 0));
}
