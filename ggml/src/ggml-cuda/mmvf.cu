#include "ggml.h"
#include "common.cuh"
#include "allreduce-p2p.cuh"
#include "unary.cuh"
#include "mmvf.cuh"
#include "mmf.cuh"
#include "convert.cuh"
#include "spin.cuh"
#include "ssm-conv.cuh"
#include "topk-moe.cuh"

GGML_CUDA_SPIN_SETTER(mmvf)

// the channels of the MoE all-reduce fold (ar_moe of mul_mat_vec_f): the used experts and the sibling, as many as the q8_0
//   fold takes (MMVQ_SHORT_AR_MAX_EXPERTS + 1, mmvq.cuh)
#define MMVF_AR_MOE_MAX_CHANNELS 11

static __device__ __forceinline__ topk_moe_config mul_mat_vec_f_topk_moe_config(const ggml_cuda_mmvf_topk_args & topk) {
    topk_moe_config config;
    config.use_sigmoid       = topk.sigmoid;
    config.use_sqrt_softplus = topk.sqrt_softplus;
    config.with_norm         = topk.with_norm;
    config.delayed_softmax   = topk.delayed_softmax;
    config.n_groups          = 1;
    config.n_group_used      = 1;
    return config;
}

// The block of a router's launch that writes its logit last selects the experts of all of them (see
//   ggml_cuda_topk_moe_fold): every block counts itself on the ticket after its write, and the last one reads the logits
//   from L2 and runs the selection of topk_moe_cuda on them with one warp.
template <int n_experts>
static __device__ __forceinline__ void mul_mat_vec_f_topk_moe(const ggml_cuda_mmvf_topk_args & topk, const int lane) {
    unsigned int ticket = 0;
    if (lane == 0) {
        __threadfence();
        ticket = atomicAdd(topk.ticket, 1u);
    }
    ticket = __shfl_sync(0xFFFFFFFF, ticket, 0);
    if (ticket != gridDim.x*gridDim.y*gridDim.z - 1) {
        return;
    }
    __threadfence();

    constexpr int experts_per_thread = topk_moe_experts_per_thread(n_experts);
    static_assert(n_experts % WARP_SIZE == 0, "the logits are read in whole warps");
    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = __ldcg(topk.logits + i*WARP_SIZE + lane);
    }
    const float bias_v[1] = { 0.0f };
    topk_moe_select<n_experts, false, false>(wt, bias_v, topk.weights, topk.ids, topk.n_used, topk.clamp, topk.scale,
        mul_mat_vec_f_topk_moe_config(topk));

    if (lane == 0) {
        *topk.ticket = 0;
    }
}

// the arguments of the all-reduce folded into a launch, with the gated norm those in its norm arguments
static __device__ __forceinline__ const ggml_cuda_ar_fold_args & mul_mat_vec_f_ar_args(const ggml_cuda_ar_fold_args & a) {
    return a;
}
static __device__ __forceinline__ const ggml_cuda_ar_fold_args & mul_mat_vec_f_ar_args(const ggml_cuda_mmvf_norm_args & a) {
    return a.ar;
}

// The 8 values y[8*col8 + l] of the gated output norm of a gated delta net layer, rms_norm(x)*w*silu(z) over rows of 128
//   (16 lanes of a block, whole rows per warp), as rms_norm_gated_f32<256> computes them: its sum of squares is the
//   butterfly of a warp over 32 columns, here xor 16 and 8 across lanes (lanes 2 and 1 apart) and 4, 2, 1 within a lane's
//   8 columns, then (w0 + w2) + (w1 + w3) of the row's 4 warps (lanes 8 and 4 apart). All lanes of the warp take part.
static __device__ __forceinline__ void mul_mat_vec_f_norm_gated(const ggml_cuda_mmvf_norm_args & a, const float4 * x4,
        const int col8, float4 * y4) {
    const float4 * z4 = (const float4 *) a.z;
    const float4 * w4 = (const float4 *) a.w + 2*(col8 % 16);
    const float4 x_lo = x4[2*col8 + 0];
    const float4 x_hi = x4[2*col8 + 1];
    const float4 z_lo = z4[2*col8 + 0];
    const float4 z_hi = z4[2*col8 + 1];
    const float4 w_lo = w4[0];
    const float4 w_hi = w4[1];
    const float xv[8] = {x_lo.x, x_lo.y, x_lo.z, x_lo.w, x_hi.x, x_hi.y, x_hi.z, x_hi.w};
    const float zv[8] = {z_lo.x, z_lo.y, z_lo.z, z_lo.w, z_hi.x, z_hi.y, z_hi.z, z_hi.w};
    const float wv[8] = {w_lo.x, w_lo.y, w_lo.z, w_lo.w, w_hi.x, w_hi.y, w_hi.z, w_hi.w};

    // each square rounded on its own, as the norm's loop does: contracted into the sums it would not be
    float s[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        s[l] = __fmul_rn(xv[l], xv[l]);
    }
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        s[l] += __shfl_xor_sync(0xFFFFFFFF, s[l], 2);
    }
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        s[l] += __shfl_xor_sync(0xFFFFFFFF, s[l], 1);
    }
    float tmp = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
    tmp += __shfl_xor_sync(0xFFFFFFFF, tmp, 8);
    tmp += __shfl_xor_sync(0xFFFFFFFF, tmp, 4);

    const float mean  = tmp / a.ncols;
    const float scale = rsqrtf(mean + a.eps);

    float yv[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const float normed = scale * xv[l] * wv[l];
        yv[l] = ggml_cuda_op_silu_single(zv[l]) * normed;
    }
    y4[0] = make_float4(yv[0], yv[1], yv[2], yv[3]);
    y4[1] = make_float4(yv[4], yv[5], yv[6], yv[7]);
}

// rpb > 1 puts that many consecutive rows in a block: they share the y loads, and each row still sums in
// exactly the order it would with a block of its own. With several tokens it needs load16.
// load16 (2-byte weights) reads 8 weights per 16-byte load instead of 2 per 4-byte load, which changes the
// order in which a row sums, so its results can differ in the last bits.
// topk_n: the expert count of a topk_moe run in the last block (topk_args), see mul_mat_vec_f_topk_moe
// ar: the two rows of a block go to an LL line of an all-reduce folded into this launch (topk_args holds its
//   ggml_cuda_ar_fold_args), not to dst; see ggml_cuda_ar_fold_push. With norm 4 rows, the two lines of one float4
//   (threads 0 and 1 push one each)
// sib_ch: with ids and one token, the channel past the used experts (the last of the grid) is the sibling sib_x of one
//   expert's layout with its own input sib_y, written to sib_dst; see ggml_cuda_mul_mat_vec_f_moe_sibling. With
//   has_fusion (a gate) it is the pair sib_x/sib_gate of a GLU with the experts' input, see
//   ggml_cuda_mul_mat_vec_f_moe_glu_sibling
// gdn_qkv: one token, the projections of a gated delta net layer in one launch: the rows of x (qkv) with the convolution
//   of build_conv_state in their epilogue (topk_args holds its ggml_cuda_mmvq_conv_args), then those of sib_x (z), then
//   from gdn_block0 on those of gdn_x (alpha), the only ones with the gate (beta) and the gated delta net epilogue
// norm: one token, y is the gated output norm of a gated delta net layer of the launch's y (topk_args holds its
//   ggml_cuda_mmvf_norm_args, and those of the all-reduce with ar), see mul_mat_vec_f_norm_gated
// ar_moe (with sib_ch, one warp a channel, 4 rows): the MoE all-reduce folded into the expert down projection (topk_args
//   holds its ggml_cuda_ar_fold_moe_args) as mul_mat_vec_q_short_ar_moe does it: a block takes the 4 rows of one float4
//   line of every expert and of the sibling, the warp threadIdx.y being the channel, and pushes their weighted sum
template <typename T, typename type_acc, int ncols_dst, int block_size, bool has_fusion = false, bool is_multi_token_id = false, int rpb = 1, bool load16 = false,
          int topk_n = 0, bool ar = false, bool sib_ch = false, bool gdn_qkv = false, bool norm = false, bool ar_moe = false>
static __global__ void mul_mat_vec_f(
        const T * x_ptr, const float * y_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const int ncols2, const uint3 nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride,
        const std::conditional_t<ar_moe, ggml_cuda_ar_fold_moe_args, std::conditional_t<norm, ggml_cuda_mmvf_norm_args,
            std::conditional_t<(topk_n > 0), ggml_cuda_mmvf_topk_args, std::conditional_t<ar, ggml_cuda_ar_fold_args,
            std::conditional_t<gdn_qkv, ggml_cuda_mmvq_conv_args, ggml_cuda_no_args>>>>> topk_args) {
    const T       * GGML_CUDA_RESTRICT x   = x_ptr;
    const float   * GGML_CUDA_RESTRICT y   = y_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;
    static_assert(rpb == 1 || ((ncols_dst == 1 || load16) && !is_multi_token_id), "rows per block need a single token or the 16-byte loads");
    const int row         = rpb*blockIdx.x; // the first of the block's rows
    [[maybe_unused]] const int stride_row2 = stride_row/2;
    // for MUL_MAT_ID - blockIdx.y = n_expert_used, blockIdx.z = ncols_dst (tokens); with ar_moe the channels are the warps
    const int channel_dst = ar_moe ? int(threadIdx.y) : int(blockIdx.y);
    [[maybe_unused]] const int nchannels_grid = ar_moe ? int(blockDim.y) : int(gridDim.y);
    const int tid         = threadIdx.x;

    int token_idx;
    int channel_x;
    int channel_y;
    int sample_dst;

    ggml_cuda_pdl_sync();
    if constexpr (is_multi_token_id) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
        token_idx  = blockIdx.z;
        channel_x  = ids[channel_dst + token_idx * ids_stride];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        static_assert(!sib_ch || (ncols_dst == 1 && (!has_fusion || rpb == 1) && topk_n == 0 && !ar && !gdn_qkv),
            "the sibling channel is one token, nothing else fused but a gate");
        // the sibling channel has no id
        const bool sib_blk = sib_ch && channel_dst == nchannels_grid - 1;
        token_idx  = ids ? blockIdx.z                                          : 0;
        channel_x  = ids ? (sib_blk ? 0 : ids[channel_dst + token_idx * ids_stride]) : fastdiv((uint32_t) channel_dst, channel_ratio);
        channel_y  = ids ? fastmodulo(channel_dst, nchannels_y)                : channel_dst;
        sample_dst = ids ? 0                                                   : blockIdx.z;
    }

    const int sample_x    = fastdiv((uint32_t) sample_dst, sample_ratio);
    const int sample_y    = sample_dst;

    // ar: the sequence number of the call, loaded while the dot product runs
    static_assert(!ar || (ncols_dst == 1 && (rpb == 2 || (norm && rpb == 4)) && !has_fusion && !is_multi_token_id && topk_n == 0),
        "the fold pushes the rows of one column of a block, two to a line");
    [[maybe_unused]] uint32_t ar_seq = 0;
    if constexpr (ar && rpb == 2) {
        if (tid == 0) {
            ar_seq = ggml_cuda_ar_seq(mul_mat_vec_f_ar_args(topk_args), row/4);
        }
    } else if constexpr (ar) {
        if (tid < 2) {
            ar_seq = ggml_cuda_ar_seq(mul_mat_vec_f_ar_args(topk_args), row/4);
        }
    }
    static_assert(!norm || (load16 && ncols_dst == 1 && (rpb == 1 || ar) && !has_fusion && !is_multi_token_id && topk_n == 0 &&
        !sib_ch && !gdn_qkv && std::is_same_v<T, half>), "the gated norm of one token's y, nothing else fused but an all-reduce");

    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    y   += int64_t(sample_y)  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;
    if constexpr (is_multi_token_id) {
        y   += token_idx*stride_col_y2*2;
        dst += token_idx*stride_col_dst;
    }
    if constexpr (sib_ch) {
        if (channel_dst == nchannels_grid - 1) {
            x   = (const T *) fusion.sib_x + row*stride_row;
            y   = fusion.sib_y;
            dst = fusion.sib_dst;
        }
    }
    // the column stride of the dst this block writes
    int stride_col_dst_blk = stride_col_dst;
    if constexpr (std::is_same_v<T, float> && (has_fusion || (ncols_dst > 1 && !is_multi_token_id))) {
        // the row of a sibling matrix with the same y is the last block's (one row per block; several tokens in the plain
        //   kernel, the sibling's dst holding them one after the other)
        if (fusion.sib_x != nullptr && fusion.sib_block0 == 0 && blockIdx.x == gridDim.x - 1) {
            x   = (const T *) fusion.sib_x;
            dst = fusion.sib_dst - row;
            stride_col_dst_blk = 1;
        }
    }
    // (the plain kernel of several tokens takes sibling rows too: the fused one holds the gate's sums in registers)
    if constexpr ((has_fusion && !sib_ch) || (load16 && ncols_dst > 1)) {
        // sibling rows (one channel and sample): the rows from sib_block0 on are the rows of a sibling matrix of the same
        //   type, width and y, written to its own dst ([rows, tokens], contiguous), those from sib2_block0 on a second's
        //   and those from gdn_block0 on a third's (gdn_x, the gated delta net rows of mmvq). One token with one row per
        //   block, or several with 4 rows per block and the 16-byte loads; the bounds are multiples of rpb.
        if (fusion.sib_block0 > 0 && uint32_t(row) >= fusion.sib_block0) {
            const int k = fusion.gdn_x  != nullptr && uint32_t(row) >= fusion.gdn_block0  ? 2 :
                          fusion.sib2_x != nullptr && uint32_t(row) >= fusion.sib2_block0 ? 1 : 0;
            const int r = row - int(k == 2 ? fusion.gdn_block0 : k == 1 ? fusion.sib2_block0 : fusion.sib_block0);
            x   = (const T *) (k == 2 ? fusion.gdn_x : k == 1 ? fusion.sib2_x : fusion.sib_x) + int64_t(r)*stride_row;
            dst = (k == 2 ? fusion.gdn_dst : k == 1 ? fusion.sib2_dst : fusion.sib_dst) + (r - row);
            stride_col_dst_blk = int(k == 2 ? fusion.gdn_nrows : k == 1 ? fusion.sib2_nrows : fusion.sib_nrows);
        }
    }

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    ggml_glu_op glu_op = ggml_glu_op::GGML_GLU_OP_SWIGLU;
    float glu_limit = 0.0f;
    const T * gate_x = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * gdn_dt = nullptr;
    const float * gdn_a = nullptr;
    float * gdn_beta = nullptr;

    if constexpr (has_fusion) {
        gdn_dt   = fusion.gdn_dt;
        gdn_a    = fusion.gdn_a;
        gdn_beta = fusion.gdn_beta;
        use_gate = fusion.gate != nullptr;
        use_bias = fusion.x_bias != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr;
        glu_op = fusion.glu_op;
        glu_limit = fusion.glu_limit;

        if (use_gate) {
            gate_x = static_cast<const T *>(fusion.gate);
        }
        if (use_bias) {
            x_bias = static_cast<const float *>(fusion.x_bias);
        }
        if (use_gate_bias) {
            gate_bias = static_cast<const float *>(fusion.gate_bias);
            use_gate_bias = use_gate;
        } else {
            use_gate_bias = false;
        }
    }

    if (use_gate) {
        gate_x += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    }
    if constexpr (sib_ch && has_fusion) {
        // the sibling's gate
        if (channel_dst == nchannels_grid - 1) {
            gate_x = (const T *) fusion.sib_gate + row*stride_row;
        }
    }

    // gdn_qkv: the rows past gdn_block0 are gdn_x's, those of the gate the same rows of fusion.gate; only they have the
    //   gate and the gated delta net epilogue, indexed by the row in gdn_x (gdn_row). The rows of x have the
    //   convolution: the state columns and weights of the row's channel, loaded while the dot product runs.
    [[maybe_unused]] int  gdn_row  = 0;
    [[maybe_unused]] bool conv_row = false;
    [[maybe_unused]] float conv_win[4];
    [[maybe_unused]] float conv_wv[4];
    if constexpr (gdn_qkv) {
        static_assert(has_fusion && ncols_dst == 1 && rpb == 1 && load16 && !is_multi_token_id && topk_n == 0 && !ar,
            "the projections of one token, one row per block");
        const bool gdn_blk = uint32_t(row) >= fusion.gdn_block0;
        gdn_row  = row - int(fusion.gdn_block0);
        use_gate = gdn_blk;
        use_gate_bias = false;
        if (gdn_blk) {
            gate_x = (const T *) fusion.gate + int64_t(gdn_row)*stride_row;
        } else {
            gdn_beta = nullptr;
        }
        conv_row = topk_args.w != nullptr && uint32_t(row) < fusion.sib_block0;
        if (conv_row && tid == 0) {
            const float * s = topk_args.states + int64_t(topk_args.ids[0])*topk_args.row_stride + int64_t(row)*3;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                conv_win[k] = s[k];
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                conv_wv[k] = topk_args.w[int64_t(row)*topk_args.w_stride + k];
            }
        }
    }

    if constexpr (has_fusion) {
        const int channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
        if (use_gate_bias) {
            gate_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
    }

    [[maybe_unused]] const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;
    [[maybe_unused]] float * buf_iw_gate = nullptr;
    if constexpr (has_fusion) {
        // past the partial sums of x: one per warp, or with the 16-byte loads one per warp, row and column
        constexpr int nsums_red = load16 ? ncols_dst*rpb*(block_size/warp_size) : 0;
        buf_iw_gate = (float *) (data_mmv + (nsums_red > warp_size ? nsums_red : warp_size)*sizeof(float));
    }

    if (!load16 && block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    float sumf[ncols_dst*rpb] = {0.0f};
    float sumf_gate[ncols_dst*rpb];
    if constexpr (has_fusion) {
#pragma unroll
        for (int j = 0; j < ncols_dst*rpb; ++j) {
            sumf_gate[j] = 0.0f;
        }
    }

    if constexpr (load16) {
        // a 16-byte load brings 8 weights of a row and two more bring the 8 matching y values of a column, and the
        // block uses each weight for every column and each y value for every row
        static_assert(sizeof(T) == 2 && !is_multi_token_id, "16-byte loads need 2-byte weights");
        const float4 * x4 = (const float4 *) x;
        [[maybe_unused]] const float4 * gate_x4 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x4 = (const float4 *) gate_x;
            }
        }
        const float4 * y4 = (const float4 *) y;
        const int ncols8        = ncols2/4;
        const int stride_row8   = stride_row/8;
        const int stride_col_y4 = stride_col_y2/2;

        if constexpr (std::is_same_v<T, half> && std::is_same_v<type_acc, half>) {
#ifdef FP16_AVAILABLE
            half2 sumh2[ncols_dst*rpb] = {{0.0f, 0.0f}};
            half2 sumh2_gate[ncols_dst*rpb] = {{0.0f, 0.0f}};

            for (int col8 = tid; col8 < ncols8; col8 += block_size) {
                float4 tmpx[rpb];
                [[maybe_unused]] float4 tmpx_gate[rpb];
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    tmpx[i] = x4[i*stride_row8 + col8];
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmpx_gate[i] = gate_x4[i*stride_row8 + col8];
                        }
                    }
                }

#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    [[maybe_unused]] float4 tmpy_norm[2];
                    if constexpr (norm) {
                        mul_mat_vec_f_norm_gated(topk_args, y4, col8, tmpy_norm);
                    }
                    const float4 tmpy_lo = norm ? tmpy_norm[0] : y4[j*stride_col_y4 + 2*col8 + 0];
                    const float4 tmpy_hi = norm ? tmpy_norm[1] : y4[j*stride_col_y4 + 2*col8 + 1];
                    const half2 tmpy[4] = {
                        make_half2(tmpy_lo.x, tmpy_lo.y), make_half2(tmpy_lo.z, tmpy_lo.w),
                        make_half2(tmpy_hi.x, tmpy_hi.y), make_half2(tmpy_hi.z, tmpy_hi.w)};
#pragma unroll
                    for (int i = 0; i < rpb; ++i) {
                        const half2 * tmpx_h2 = (const half2 *) &tmpx[i];
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            sumh2[i*ncols_dst + j] += tmpx_h2[l] * tmpy[l];
                        }

                        if constexpr (has_fusion) {
                            if (use_gate) {
                                const half2 * tmpx_gate_h2 = (const half2 *) &tmpx_gate[i];
#pragma unroll
                                for (int l = 0; l < 4; ++l) {
                                    sumh2_gate[i*ncols_dst + j] += tmpx_gate_h2[l] * tmpy[l];
                                }
                            }
                        }
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst*rpb; ++j) {
                sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        sumf_gate[j] = __low2float(sumh2_gate[j]) + __high2float(sumh2_gate[j]);
                    }
                }
            }
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        } else {
            for (int col8 = tid; col8 < ncols8; col8 += block_size) {
                float tmpx[rpb][8];
                [[maybe_unused]] float tmpx_gate[rpb][8];
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    const float4 tmpx4 = x4[i*stride_row8 + col8];
                    const T * tmpx_e = (const T *) &tmpx4;
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        tmpx[i][l] = ggml_cuda_cast<float>(tmpx_e[l]);
                    }
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            const float4 tmpx4_gate = gate_x4[i*stride_row8 + col8];
                            const T * tmpx_gate_e = (const T *) &tmpx4_gate;
#pragma unroll
                            for (int l = 0; l < 8; ++l) {
                                tmpx_gate[i][l] = ggml_cuda_cast<float>(tmpx_gate_e[l]);
                            }
                        }
                    }
                }

#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    [[maybe_unused]] float4 tmpy_norm[2];
                    if constexpr (norm) {
                        mul_mat_vec_f_norm_gated(topk_args, y4, col8, tmpy_norm);
                    }
                    const float4 tmpy_lo = norm ? tmpy_norm[0] : y4[j*stride_col_y4 + 2*col8 + 0];
                    const float4 tmpy_hi = norm ? tmpy_norm[1] : y4[j*stride_col_y4 + 2*col8 + 1];
                    const float tmpy[8] = {tmpy_lo.x, tmpy_lo.y, tmpy_lo.z, tmpy_lo.w, tmpy_hi.x, tmpy_hi.y, tmpy_hi.z, tmpy_hi.w};
#pragma unroll
                    for (int i = 0; i < rpb; ++i) {
#pragma unroll
                        for (int l = 0; l < 8; ++l) {
                            ggml_cuda_mad(sumf[i*ncols_dst + j], tmpx[i][l], tmpy[l]);
                        }

                        if constexpr (has_fusion) {
                            if (use_gate) {
#pragma unroll
                                for (int l = 0; l < 8; ++l) {
                                    ggml_cuda_mad(sumf_gate[i*ncols_dst + j], tmpx_gate[i][l], tmpy[l]);
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;
        [[maybe_unused]] const float2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const float2 *) gate_x;
            }
        }

        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                const float2 tmpx = x2[i*stride_row2 + col2];
                float2 tmpx_gate = make_float2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = gate_x2[i*stride_row2 + col2];
                    }
                }

#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    ggml_cuda_mad(sumf[i*ncols_dst + j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(sumf[i*ncols_dst + j], tmpx.y, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_mad(sumf_gate[i*ncols_dst + j], tmpx_gate.x, tmpy.x);
                            ggml_cuda_mad(sumf_gate[i*ncols_dst + j], tmpx_gate.y, tmpy.y);
                        }
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, half>) {
        const half2 * x2 = (const half2 *) x;
        [[maybe_unused]] const half2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const half2 *) gate_x;
            }
        }

        if (std::is_same_v<type_acc, float>) {
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    const float2 tmpx = __half22float2(x2[i*stride_row2 + col2]);
                    float2 tmpx_gate = make_float2(0.0f, 0.0f);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmpx_gate = __half22float2(gate_x2[i*stride_row2 + col2]);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const float2 tmpy = y2[j*stride_col_y2 + col2];
                        ggml_cuda_mad(sumf[i*ncols_dst + j], tmpx.x, tmpy.x);
                        ggml_cuda_mad(sumf[i*ncols_dst + j], tmpx.y, tmpy.y);

                        if constexpr (has_fusion) {
                            if (use_gate) {
                                ggml_cuda_mad(sumf_gate[i*ncols_dst + j], tmpx_gate.x, tmpy.x);
                                ggml_cuda_mad(sumf_gate[i*ncols_dst + j], tmpx_gate.y, tmpy.y);
                            }
                        }
                    }
                }
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2[ncols_dst*rpb] = {{0.0f, 0.0f}};
            half2 sumh2_gate[ncols_dst*rpb] = {{0.0f, 0.0f}};

            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
#pragma unroll
                for (int i = 0; i < rpb; ++i) {
                    const half2 tmpx = x2[i*stride_row2 + col2];
                    half2 tmpx_gate = make_half2(0.0f, 0.0f);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmpx_gate = gate_x2[i*stride_row2 + col2];
                        }
                    }
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const float2 tmpy = y2[j*stride_col_y2 + col2];
                        sumh2[i*ncols_dst + j] += tmpx * make_half2(tmpy.x, tmpy.y);

                        if constexpr (has_fusion) {
                            if (use_gate) {
                                sumh2_gate[i*ncols_dst + j] += tmpx_gate * make_half2(tmpy.x, tmpy.y);
                            }
                        }
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst*rpb; ++j) {
                sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
            }

            if constexpr (has_fusion) {
                if (use_gate) {
#pragma unroll
                    for (int j = 0; j < ncols_dst*rpb; ++j) {
                        sumf_gate[j] = __low2float(sumh2_gate[j]) + __high2float(sumh2_gate[j]);
                    }
                }
            }
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
//TODO: add support for ggml_cuda_mad for hip_bfloat162
#if defined(GGML_USE_HIP)
        const int * x2 = (const int *) x;
        const int * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const int *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                const int tmpx = x2[i*stride_row2 + col2];
                int tmpx_gate = 0;
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = gate_x2[i*stride_row2 + col2];
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
                    const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
                    ggml_cuda_mad(sumf[i*ncols_dst + j], tmpx0, tmpy.x);
                    ggml_cuda_mad(sumf[i*ncols_dst + j], tmpx1, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            const float tmpx0_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[0]);
                            const float tmpx1_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[1]);
                            ggml_cuda_mad(sumf_gate[i*ncols_dst + j], tmpx0_gate, tmpy.x);
                            ggml_cuda_mad(sumf_gate[i*ncols_dst + j], tmpx1_gate, tmpy.y);
                        }
                    }
                }
            }
        }
#else
        const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
        [[maybe_unused]] const nv_bfloat162 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const nv_bfloat162 *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
#pragma unroll
            for (int i = 0; i < rpb; ++i) {
                const nv_bfloat162 tmpx = x2[i*stride_row2 + col2];
                [[maybe_unused]] nv_bfloat162 tmpx_gate;
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = gate_x2[i*stride_row2 + col2];
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    ggml_cuda_mad(sumf[i*ncols_dst + j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(sumf[i*ncols_dst + j], tmpx.y, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_mad(sumf_gate[i*ncols_dst + j], tmpx_gate.x, tmpy.x);
                            ggml_cuda_mad(sumf_gate[i*ncols_dst + j], tmpx_gate.y, tmpy.y);
                        }
                    }
                }
            }
        }
#endif
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

    ggml_cuda_pdl_lc();
    if constexpr (load16) {
        // all rows and columns reduce together, through a single barrier
        constexpr int nwarps = block_size/warp_size;
        constexpr int nsums  = ncols_dst*rpb;
#pragma unroll
        for (int i = 0; i < nsums; ++i) {
            sumf[i] = warp_reduce_sum<warp_size>(sumf[i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    sumf_gate[i] = warp_reduce_sum<warp_size>(sumf_gate[i]);
                }
            }
        }

        if (block_size > warp_size) {
            if (tid % warp_size == 0) {
#pragma unroll
                for (int i = 0; i < nsums; ++i) {
                    buf_iw[i*nwarps + tid/warp_size] = sumf[i];
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            buf_iw_gate[i*nwarps + tid/warp_size] = sumf_gate[i];
                        }
                    }
                }
            }
            __syncthreads();
            if (tid >= nsums) {
                return;
            }
#pragma unroll
            for (int i = 0; i < nsums; ++i) {
                sumf[i] = 0.0f;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
                    sumf[i] += buf_iw[i*nwarps + w];
                }
                if constexpr (has_fusion) {
                    if (use_gate) {
                        sumf_gate[i] = 0.0f;
#pragma unroll
                        for (int w = 0; w < nwarps; ++w) {
                            sumf_gate[i] += buf_iw_gate[i*nwarps + w];
                        }
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < ncols_dst*rpb; ++j) {
            sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

            if constexpr (has_fusion) {
                if (use_gate) {
                    sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
                }
            }

            if (block_size > warp_size) {
                buf_iw[tid/warp_size] = sumf[j];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        buf_iw_gate[tid/warp_size] = sumf_gate[j];
                    }
                }
                __syncthreads();
                if (tid < warp_size) {
                    sumf[j] = buf_iw[tid];
                    sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            sumf_gate[j] = buf_iw_gate[tid];
                            sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
                        }
                    }
                }

                if (j < ncols_dst*rpb) {
                    __syncthreads();
                }
            }
        }
    }

    if constexpr (ar_moe) {
        // the products in expert order, then the gated sibling, as ggml_cuda_p2p_ar_ll_moe_reduce_kernel adds them; the even
        //   row's lane pushes the pair
        static_assert(sib_ch && load16 && !has_fusion && block_size == warp_size && rpb == 4 && ncols_dst == 1 && !ar,
            "the fold takes one warp per channel and the 4 rows of a line");
        const ggml_cuda_ar_fold_moe_args & m = topk_args;
        __shared__ float sums[MMVF_AR_MOE_MAX_CHANNELS][4];
        if (tid == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sums[channel_dst][i] = sumf[i];
            }
        }
        __syncthreads();
        if (channel_dst != 0) {
            return;
        }
        float v = 0.0f;
        if (tid < 4) {
            v = __fmul_rn(sums[0][tid], m.weights[0]);
            for (int e = 1; e < m.n_used; ++e) {
                v = __fadd_rn(v, __fmul_rn(sums[e][tid], m.weights[e]));
            }
            v = __fadd_rn(v, __fmul_rn(sums[m.n_used][tid], ggml_cuda_op_sigmoid_single(*m.gate)));
        }
        const float v1 = __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (tid < 4 && tid % 2 == 0) {
            const int r = row + tid;
            ggml_cuda_ar_fold_push(m.ar, ggml_cuda_ar_seq(m.ar, r/4), r/4, (r/2) % 2, v, v1);
        }
        return;
    }

    if constexpr (topk_n > 0) {
        static_assert(ncols_dst == 1 && rpb == 1 && !is_multi_token_id && !load16, "topk_moe needs one token and one row per block");
        if (tid >= warp_size) {
            return;
        }
        if (tid == 0) {
            dst[row] = sumf[0];
        }
        mul_mat_vec_f_topk_moe<topk_n>(topk_args, tid);
        return;
    }

    if constexpr (ar && rpb == 2) {
        if (tid == 0) {
            ggml_cuda_ar_fold_push(mul_mat_vec_f_ar_args(topk_args), ar_seq, row/4, (row/2) % 2, sumf[0], sumf[1]);
        }
        return;
    } else if constexpr (ar) {
        // the block's float4: thread 0 pushes its rows 0 and 1, thread 1 rows 2 and 3
        if (tid < 2) {
            ggml_cuda_ar_fold_push(mul_mat_vec_f_ar_args(topk_args), ar_seq, row/4, tid, tid == 0 ? sumf[0] : sumf[2],
                tid == 0 ? sumf[1] : sumf[3]);
        }
        return;
    }

    if (tid >= ncols_dst*rpb) {
        return;
    }

    // tid indexes the block's rows and dst's columns
    const int drow = row + tid/ncols_dst;
    const int dcol = tid % ncols_dst;

    // the sums of this thread, picked with constant indices: indexed by tid, sumf and sumf_gate would live in local
    //   memory
    float value = sumf[0];
#pragma unroll
    for (int i = 1; i < ncols_dst*rpb; ++i) {
        if (tid == i) {
            value = sumf[i];
        }
    }

    if constexpr (has_fusion) {
        if (use_bias) {
            value += x_bias[dcol*stride_col_dst + drow];
        }

        if (use_gate) {
            float gate_value = sumf_gate[0];
#pragma unroll
            for (int i = 1; i < ncols_dst*rpb; ++i) {
                if (tid == i) {
                    gate_value = sumf_gate[i];
                }
            }
            if (use_gate_bias) {
                gate_value += gate_bias[dcol*stride_col_dst + drow];
            }
            if (gdn_beta) {
                // the gate and beta of a gated delta net layer, x and gate being its alpha and beta projections
                const int grow = gdn_qkv ? gdn_row : drow;
                dst[dcol*stride_col_dst + drow]      = ggml_cuda_op_softplus_single(value + gdn_dt[grow]) * gdn_a[grow];
                gdn_beta[dcol*stride_col_dst + grow] = ggml_cuda_op_sigmoid_single(gate_value);
                return;
            }
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    value *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    value *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI: {
                    value = ggml_cuda_op_swiglu_oai_single(gate_value, value);
                    break;
                }
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    value = ggml_cuda_op_swiglu_clamp_single(gate_value, value, glu_limit);
                    break;
                default:
                    break;
            }
        }
    }

    dst[dcol*stride_col_dst_blk + drow] = value;

    if constexpr (gdn_qkv) {
        // as ssm_conv_state_f32 with one token; the convolution's dst may sit exactly on dst, so it is written last
        if (conv_row) {
            conv_win[3] = value;
            if (topk_args.slot0 != nullptr) {
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    topk_args.slot0[int64_t(row)*3 + k] = conv_win[k];
                }
            }
            if (topk_args.slot1 != nullptr) {
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    topk_args.slot1[int64_t(row)*3 + k] = conv_win[1 + k];
                }
            }
            topk_args.dst[row] = ggml_cuda_ssm_conv_silu<4>(conv_win, conv_wv, 0.0f);
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, glu_op, glu_limit, gate_x, x_bias, gate_bias, sumf_gate,
            gdn_dt, gdn_a, gdn_beta);
    }
}

template<typename T, typename type_acc, int ncols_dst, int block_size, bool is_multi_token_id = false>
static void mul_mat_vec_f_switch_fusion(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_launch & fusion, float * dst,
        const int64_t ncols, const uint3 nchannels_y,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const dim3 & block_dims, const dim3 & block_nums, const int nbytes_shared, const int ids_stride, const int rpb,
        const bool load16, const cudaStream_t stream) {

    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, nbytes_shared, stream};
    GGML_ASSERT(fusion.topk.ids == nullptr || (ncols_dst == 1 && !is_multi_token_id && sizeof(T) == 4 && rpb == 1 && !load16));

    const auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params,
            x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fusion.topk);
    };

    const bool sib_ch     = ids != nullptr && fusion.sib_y != nullptr;
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
        (fusion.sib_x != nullptr && !sib_ch);
    // the projections of a gated delta net layer, see ggml_cuda_mul_mat_vec_f
    const bool gdn_qkv    = fusion.gdn_x != nullptr && fusion.gdn_beta != nullptr;
    GGML_ASSERT(!gdn_qkv || (ncols_dst == 1 && !is_multi_token_id && sizeof(T) == 2 && load16 && rpb == 1 && !sib_ch));
    if constexpr (ncols_dst > 1 && !is_multi_token_id && sizeof(T) == 2) {
        if (load16) {
            if (fusion.sib_block0 > 0) {
                // sibling rows, nothing else fused
                GGML_ASSERT(rpb == 4 && !fusion.gate && !fusion.x_bias && !fusion.gate_bias && !fusion.gdn_beta);
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 4, true>);
            } else if (has_fusion) {
                // with several tokens only the gate of a gated delta net layer, which has no biases, fuses
                GGML_ASSERT(fusion.gdn_beta && rpb == 1);
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, false, 1, true>);
            } else if (rpb == 4) {
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 4, true>);
            } else if (rpb == 8) {
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 8, true>);
            } else {
                GGML_ASSERT(rpb == 1);
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 1, true>);
            }
            return;
        }
    }
    if constexpr (ncols_dst == 1 && !is_multi_token_id && sizeof(T) == 2) {
        if (load16 && sib_ch) {
            if (has_fusion) {
                // the sibling pair of a GLU past the used experts, nothing else fused
                GGML_ASSERT(rpb == 1 && fusion.sib_gate != nullptr && fusion.gate != nullptr && !fusion.x_bias && !fusion.gate_bias);
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, false, 1, true, 0, false, true>);
                return;
            }
            // the MoE all-reduce folded in (see ggml_cuda_ar_fold_moe_args): one warp per channel, a block per 4 rows; every
            //   row sums as it does with rpb rows per block
            if constexpr (block_size == WARP_SIZE) {
                const int nchannels = int(block_nums.y);
                const int nrows     = int(block_nums.x)*rpb;
                if (fusion.ar_moe && fusion.ar_pushed != nullptr && nchannels == fusion.ar_n_used + 1 &&
                        nchannels <= MMVF_AR_MOE_MAX_CHANNELS && nrows % 4 == 0 && block_nums.z == 1) {
                    ggml_cuda_ar_fold_moe_args m;
                    m.ar      = fusion.ar;
                    m.weights = fusion.ar_weights;
                    m.gate    = fusion.ar_gate;
                    m.n_used  = fusion.ar_n_used;
                    const ggml_cuda_kernel_launch_params fold_params = {dim3(nrows/4, 1, 1), dim3(WARP_SIZE, nchannels, 1), nbytes_shared, stream};
                    ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 4, true, 0, false, true, false, false, true>,
                        fold_params, x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, m);
                    *fusion.ar_pushed = true;
                    return;
                }
            }
            // the sibling channel past the used experts, nothing else fused
            switch (rpb) {
                case 1: launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 1, true, 0, false, true>); break;
                case 4: launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 4, true, 0, false, true>); break;
                case 8: launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 8, true, 0, false, true>); break;
                default: GGML_ABORT("fatal error");
            }
            return;
        }
        if (load16) {
            // the gated norm of y in the launch (see ggml_cuda_mul_mat_vec_f_norm_gated), with the fold's all-reduce
            const auto launch_norm = [&](auto kernel) {
                ggml_cuda_mmvf_norm_args norm_args = fusion.norm;
                norm_args.ar = fusion.ar;
                ggml_cuda_kernel_launch(kernel, launch_params,
                    x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, norm_args);
            };
            if (rpb == 2 || (fusion.norm.z != nullptr && rpb == 4)) {
                // the all-reduce of dst folded in, see launch_mul_mat_vec_f_cuda
                GGML_ASSERT(fusion.ar_pushed != nullptr && !has_fusion);
                if (fusion.norm.z != nullptr) {
                    if constexpr (std::is_same_v<T, half>) {
                        GGML_ASSERT(rpb == 4);
                        launch_norm(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 4, true, 0, true, false, false, true>);
                    } else {
                        GGML_ABORT("fatal error");
                    }
                } else {
                    ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 2, true, 0, true>, launch_params,
                        x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fusion.ar);
                }
                *fusion.ar_pushed = true;
                return;
            }
            if (fusion.norm.z != nullptr) {
                GGML_ASSERT(rpb == 1 && !has_fusion && !gdn_qkv);
                if constexpr (std::is_same_v<T, half>) {
                    launch_norm(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 1, true, 0, false, false, false, true>);
                } else {
                    GGML_ABORT("fatal error");
                }
                return;
            }
            if (rpb == 4) {
                if (has_fusion) {
                    launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, false, 4, true>);
                } else {
                    launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 4, true>);
                }
            } else if (rpb == 8) {
                GGML_ASSERT(!has_fusion);
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 8, true>);
            } else {
                GGML_ASSERT(rpb == 1);
                if (gdn_qkv) {
                    // the convolution's arguments in place of topk's
                    ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, false, 1, true, 0, false, false, true>,
                        launch_params, x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fusion.conv);
                } else if (has_fusion) {
                    launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, false, 1, true>);
                } else {
                    launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 1, true>);
                }
            }
            return;
        }
    }
    GGML_ASSERT(!load16 && !sib_ch && fusion.norm.z == nullptr);
    if constexpr (ncols_dst == 1 && !is_multi_token_id) {
        if (rpb == 4) {
            if (has_fusion) {
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, false, 4>);
            } else {
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 4>);
            }
            return;
        }
    }
    GGML_ASSERT(rpb == 1);

    if constexpr (ncols_dst == 1 && !is_multi_token_id && std::is_same_v<T, float>) {
        if (fusion.topk.ids != nullptr) {
            switch (fusion.topk.n_expert) {
                case 128: launch(mul_mat_vec_f<T, type_acc, 1, block_size, true, false, 1, false, 128>); return;
                case 256: launch(mul_mat_vec_f<T, type_acc, 1, block_size, true, false, 1, false, 256>); return;
                case 512: launch(mul_mat_vec_f<T, type_acc, 1, block_size, true, false, 1, false, 512>); return;
                default: GGML_ABORT("fatal error");
            }
        }
    }
    GGML_ASSERT(fusion.topk.ids == nullptr);

    if constexpr (ncols_dst == 1) {
        if (has_fusion) {
            launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, is_multi_token_id>);
            return;
       }
    }

    // several tokens of f32 rows: the plain kernel takes the one sibling row (a shared expert gate after its router)
    const bool sib_row_only = fusion.sib_x != nullptr && fusion.sib_block0 == 0 && !fusion.gate && !fusion.x_bias &&
        !fusion.gate_bias;
    GGML_ASSERT((!has_fusion || (std::is_same_v<T, float> && !is_multi_token_id && sib_row_only)) &&
        "fusion with several tokens needs the 16-byte loads");

    launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, is_multi_token_id>);
}

// On Volta 2-byte weights are read 8 per 16-byte load. Measured with a cold cache, one token: bf16 8192x1024
// -14%, 4096x512 -20%, 256x4096 -10%, f16 2560x5120 -4%, 62080x5120 and K = 4096 flat; 2-8 tokens: bf16
// 8192x1024 and 1280x17408 1.4-2.6x faster.
static bool mmvf_use_load16(
        const bool two_byte_weights, const int cc, const int64_t ncols_dst, const int64_t ncols, const int64_t stride_row,
        const int64_t stride_col_y, const int64_t stride_channel_x, const int64_t stride_sample_x,
        const int64_t stride_channel_y, const int64_t stride_sample_y, const void * x, const void * y, const void * gate) {
    const bool is_volta = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING;
    return two_byte_weights && is_volta && (ncols_dst == 1 || stride_col_y % 4 == 0) &&
        ncols % 8 == 0 && stride_row % 8 == 0 && stride_channel_x % 8 == 0 && stride_sample_x % 8 == 0 &&
        stride_channel_y % 4 == 0 && stride_sample_y % 4 == 0 &&
        (uintptr_t) x % 16 == 0 && (uintptr_t) y % 16 == 0 && (uintptr_t) gate % 16 == 0;
}

// the block size of a launch: the fewest iterations over a row, then the smallest block
static int64_t mmvf_block_size(const int64_t ncols, const int64_t ncols_thread, const int device) {
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + ncols_thread*warp_size - 1) / (ncols_thread*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + ncols_thread*block_size - 1) / (ncols_thread*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    return block_size_best;
}

// the sibling rows of a launch (see mul_mat_vec_f) read their own matrices
static bool mmvf_siblings_aligned16(const ggml_cuda_mm_fusion_args_device & fusion) {
    return (uintptr_t) fusion.sib_x % 16 == 0 && (uintptr_t) fusion.sib2_x % 16 == 0 && (uintptr_t) fusion.gdn_x % 16 == 0 &&
        (uintptr_t) fusion.sib_y % 16 == 0 && (uintptr_t) fusion.sib_gate % 16 == 0;
}

template <typename T, typename type_acc, int ncols_dst, bool is_multi_token_id = false>
void launch_mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_launch & fusion, float * dst,
        const int64_t ncols, const int64_t nrows,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t nsamples_or_ntokens, const int64_t ids_stride, cudaStream_t stream) {
    GGML_ASSERT(ncols        % 2 == 0);
    GGML_ASSERT(stride_row   % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    const int  cc        = ggml_cuda_info().devices[device].cc;
    const bool is_volta  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING;
    const bool load16 = !is_multi_token_id && mmvf_use_load16(sizeof(T) == 2, cc, ncols_dst, ncols, stride_row, stride_col_y,
        stride_channel_x, stride_sample_x, stride_channel_y, stride_sample_y, x, y, fusion.gate) && mmvf_siblings_aligned16(fusion);
    const int64_t ncols_thread = load16 ? 8 : 2; // columns a thread takes per load

    const int64_t block_size_best = mmvf_block_size(ncols, ncols_thread, device);

    // the sibling channel past the used experts (see ggml_cuda_mul_mat_vec_f_moe_sibling) is one more channel of the grid
    const bool sib_ch     = ids != nullptr && fusion.sib_y != nullptr;
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
        (fusion.sib_x != nullptr && !sib_ch);
    GGML_ASSERT(!sib_ch || (load16 && ncols_dst == 1 && !is_multi_token_id && (!has_fusion || fusion.sib_gate != nullptr)));

    // On Volta four rows per block share the y loads and keep more loads in flight. With the 16-byte loads
    // that pays for one token only for rows of <= 512 (bf16 4096x512 -10%, 8192x1024 +4%), and for 2-6 tokens for
    // rows of >= 1024, where each row would otherwise read every y column again (bf16 1280x17408 at 6 tokens
    // 125 -> 79 us, 4352x5120 118 -> 88 us; rows of 512 and 8 tokens are mixed). With the 4-byte loads, where every
    // row still sums in its own order, short rows gain the most (bf16 8192x1024 -22%, 4096x512 -23%) and so do
    // very tall matrices (f16 62080x5120 -10%), rows of 2k-17k are within +-3%, and below ~8 blocks per SM it
    // loses.
    // (the GLU of a sibling pair past the used experts keeps one row per block)
    int rpb = 1;
    if (ncols_dst == 1 && !is_multi_token_id && is_volta && (fusion.sib_x == nullptr || (sib_ch && !has_fusion)) && fusion.topk.ids == nullptr &&
            fusion.norm.z == nullptr &&
            (load16 ? ncols <= 512 : (ncols <= 1024 || nrows >= 16384)) &&
            nrows % 4 == 0 && (nrows/4)*nchannels_dst*nsamples_or_ntokens >= 8*ggml_cuda_info().devices[device].nsm) {
        rpb = 4;
        // and 8 for the experts' rows of <= 256 (32 lanes of 16-byte loads), which give each lane only one load per row:
        //   f16 down of 256 experts, 8 used, 2048 x 128, 9.9 -> 7.0 us (16 rows 7.4)
        if (ids != nullptr && load16 && !has_fusion && ncols <= 256 && nrows % 8 == 0 &&
                (nrows/8)*nchannels_dst*nsamples_or_ntokens >= 8*ggml_cuda_info().devices[device].nsm) {
            rpb = 8;
        }
    }
    // Only while that leaves a block per SM: with fewer, each block walks the whole K for every column on its own,
    //   and one row per block spreads that over more SMs (f16 12x5120 at 6 tokens 14.5 -> 9.5 us cold, 24x5120
    //   14.7 -> 9.9). Each row sums in the same order either way, so a fused gate keeps one row per block.
    if (load16 && ncols_dst > 1 && ncols_dst <= 6 && !has_fusion && ncols >= 1024 && nrows % 4 == 0 &&
            (nrows/4)*nchannels_dst*nsamples_or_ntokens >= ggml_cuda_info().devices[device].nsm) {
        rpb = 4;
        // and 8 for a matrix that still leaves 8 blocks per SM (an LM head: f16 6656 x 50512 at 4 tokens 902 -> 827 us;
        //   rows of 1024 lose, 26.5 vs 20.8 us)
        if (nrows % 8 == 0 && (nrows/8)*nchannels_dst*nsamples_or_ntokens >= 8*ggml_cuda_info().devices[device].nsm) {
            rpb = 8;
        }
    }

    // sibling rows of several tokens (see ggml_cuda_mul_mat_vec_f): 4 rows per block, which the row counts allow
    if (ncols_dst > 1 && fusion.sib_block0 > 0) {
        GGML_ASSERT(load16 && !is_multi_token_id && nrows % 4 == 0);
        rpb = 4;
    }

    // an all-reduce of dst folded into the launch (ggml_backend_cuda_context::ar_fold): one token, the whole of dst in
    //   its rows and nothing else fused; two rows per block make an LL line and sum as with one row per block
    if (fusion.ar_pushed != nullptr && ncols_dst == 1 && ids == nullptr && load16 && !has_fusion && nchannels_dst == 1 &&
            nsamples_or_ntokens == 1 && nrows % 4 == 0 && nrows == stride_col_dst) {
        rpb = 2;
        // with the gated norm of y every block normalizes all of y, so 4 rows share it (the two lines of a float4). tg128
        //   d0 / d8192 against 2 rows: Sage-A1.5 F16 (ssm_out 2048 x 1024 per GPU) -0.1% / +0.6%; Qwen3.8-27B F16 (5120 x
        //   1536, where at 2 rows the prologue cost as much as the norm's launch, +3.4 vs 3.1 us) +0.8% / +0.6%. 8 rows
        //   against 4: -0.1% / -0.1% and -0.0% / -0.2%
        if (fusion.norm.z != nullptr) {
            rpb = 4;
        }
    }
    // the gated norm of y (see ggml_cuda_mul_mat_vec_f_norm_gated): every lane of a block takes as many columns
    GGML_ASSERT(fusion.norm.z == nullptr || (load16 && ncols_dst == 1 && !has_fusion && !sib_ch && (ncols/8) % block_size_best == 0 &&
        (rpb == 1 || (rpb == 4 && fusion.ar_pushed != nullptr))));

    // the 16-byte path keeps one partial sum per warp, row and column
    const int nsums_red     = load16 ? int(ncols_dst*rpb*(block_size_best/warp_size)) : 0;
    const int nbytes_shared = std::max(warp_size, nsums_red)*(has_fusion ? 2 : 1)*sizeof(float);

    const dim3 block_nums(nrows/rpb, nchannels_dst + (sib_ch ? 1 : 0), nsamples_or_ntokens);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
        case   32: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 32, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, rpb, load16, stream);
        } break;
        case   64: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 64, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, rpb, load16, stream);
        } break;
        case   96: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 96, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, rpb, load16, stream);
        } break;
        case  128: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 128, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, rpb, load16, stream);
        } break;
        case  160: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 160, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, rpb, load16, stream);
        } break;
        case  192: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 192, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, rpb, load16, stream);
        } break;
        case  224: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 224, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, rpb, load16, stream);
        } break;
        case  256: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 256, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, rpb, load16, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}

template <typename T, typename type_acc>
static void mul_mat_vec_f_cuda_switch_ncols_dst(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_launch & fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, cudaStream_t stream) {

    const bool has_ids = ids != nullptr;

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path only - single-token goes through regular path below
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst, true>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    if (has_ids) {
        // Single-token MUL_MAT_ID path
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_f_cuda<T, type_acc, 1>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 2:
            launch_mul_mat_vec_f_cuda<T, type_acc, 2>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 3:
            launch_mul_mat_vec_f_cuda<T, type_acc, 3>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 4:
            launch_mul_mat_vec_f_cuda<T, type_acc, 4>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 5:
            launch_mul_mat_vec_f_cuda<T, type_acc, 5>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 6:
            launch_mul_mat_vec_f_cuda<T, type_acc, 6>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 7:
            launch_mul_mat_vec_f_cuda<T, type_acc, 7>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 8:
            launch_mul_mat_vec_f_cuda<T, type_acc, 8>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

template<typename T>
static void mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_launch & fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, enum ggml_prec prec, cudaStream_t stream) {

    if constexpr(std::is_same_v<T, half>) {
        if (prec == GGML_PREC_DEFAULT) {
            mul_mat_vec_f_cuda_switch_ncols_dst<T, half>
                (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            return;
        }
    }
    mul_mat_vec_f_cuda_switch_ncols_dst<T, float>
        (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
        nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
        stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
}

// The waiting block of mul_mat_vec_f_router (one warp): once every slot holds a logit with the tag of this launch, it writes the logits to dst and runs the selection of mul_mat_vec_f_topk_moe on them.
// A launch tags its logits with the epoch plus one and the waiter moves the epoch on after reading them, so a slot from an earlier launch never matches and nothing has to reset the slots.
// A first pass with one expert runs on whatever the slots hold while the rows are still read: it brings the selection code into this SM.
template <int n_experts>
static __device__ __forceinline__ void mul_mat_vec_f_router_wait(const ggml_cuda_mmvf_topk_args & topk, float * dst, const int lane) {
    constexpr int warp_size          = ggml_cuda_get_physical_warp_size();
    constexpr int experts_per_thread = topk_moe_experts_per_thread(n_experts);
    static_assert(n_experts % warp_size == 0 && n_experts <= GGML_CUDA_TOPK_SLOT_COUNT, "the slots are read in whole warps");
    static_assert(2*warp_size <= GGML_CUDA_BLOCK_FLAGS - GGML_CUDA_TOPK_WARM, "scratch of the first pass");
    volatile unsigned long long * slots = topk.slots;
    unsigned int * epoch        = (unsigned int *) (topk.slots + GGML_CUDA_TOPK_SLOT_COUNT);
    float        * warm_weights = (float *) (epoch + (GGML_CUDA_TOPK_WARM - GGML_CUDA_TOPK_EPOCH));
    int32_t      * warm_ids     = (int32_t *) (warm_weights + warp_size);
    const unsigned int tag = *(volatile unsigned int *) epoch + 1;

    const float bias_v[1] = { 0.0f };
    const topk_moe_config config = mul_mat_vec_f_topk_moe_config(topk);

    float wt[experts_per_thread];
    bool  warm = true;
#pragma unroll 1
    while (true) {
        // the first pass polls once
        const bool ready = ggml_cuda_spin_until([&]() {
            bool ready = true;
#pragma unroll
            for (int i = 0; i < experts_per_thread; i++) {
                const unsigned long long v = slots[i*warp_size + lane];
                wt[i] = __uint_as_float((unsigned int) v);
                ready = ready && (unsigned int) (v >> 32) == tag;
            }
            return __all_sync(0xFFFFFFFF, ready);
        }, warm);
        if (!ready && !warm) {
            ggml_cuda_spin_trap(GGML_CUDA_SPIN_SITE_ROUTER, -1, GGML_CUDA_SPIN_PEER_BLOCKS);
        }
        if (ready) {
#pragma unroll
            for (int i = 0; i < experts_per_thread; i++) {
                dst[i*warp_size + lane] = wt[i];
            }
            // every router block read the epoch before it stored its tag
            if (lane == 0) {
                *(volatile unsigned int *) epoch = tag;
            }
            ggml_cuda_syncwarp();
        }
        topk_moe_select<n_experts, false, false>(wt, bias_v, ready ? topk.weights : warm_weights, ready ? topk.ids : warm_ids,
            ready ? topk.n_used : 1, topk.clamp, topk.scale, config);
        if (ready) {
            return;
        }
        warm = false;
    }
}

// The one-token f32 router of n_experts rows, its one-row sibling and the topk_moe of its logits in one launch (ggml_cuda_topk_moe_fold on Volta).
// Block n_experts waits for the logits and the last block is the sibling; each router block stores its logit with the tag in one 64-bit store, so no ticket or fence is needed.
// The rows sum as in mul_mat_vec_f with 256 threads, so the logits are the same.
template <int n_experts>
static __global__ void mul_mat_vec_f_router(
        const float * x_ptr, const float * y_ptr, const float * sib_x, float * dst, float * sib_dst, const int ncols2,
        const int stride_row, const ggml_cuda_mmvf_topk_args topk) {
    constexpr int block_size = 256;
    constexpr int warp_size  = ggml_cuda_get_physical_warp_size();
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    if (blockIdx.x == n_experts) {
        if (tid < warp_size) {
            mul_mat_vec_f_router_wait<n_experts>(topk, dst, tid);
        }
        return;
    }
    const bool sib = blockIdx.x == n_experts + 1;
    const float2 * GGML_CUDA_RESTRICT x2 = (const float2 *) (sib ? sib_x : x_ptr + blockIdx.x*stride_row);
    const float2 * GGML_CUDA_RESTRICT y2 = (const float2 *) y_ptr;

    // read early, its latency hides behind the dot product
    unsigned int tag = 0;
    if (tid == 0 && !sib) {
        tag = *(volatile unsigned int *) (topk.slots + GGML_CUDA_TOPK_SLOT_COUNT) + 1;
    }

    __shared__ float buf_iw[warp_size];
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
        ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
    }

    sumf = warp_reduce_sum<warp_size>(sumf);
    buf_iw[tid/warp_size] = sumf;
    __syncthreads();
    if (tid >= warp_size) {
        return;
    }
    sumf = warp_reduce_sum<warp_size>(buf_iw[tid]);
    if (tid == 0) {
        if (sib) {
            sib_dst[0] = sumf;
        } else {
            *(volatile unsigned long long *) (topk.slots + blockIdx.x) = ((unsigned long long) tag << 32) | __float_as_uint(sumf);
        }
    }
}

// The one-token f32 router with its selection by ranks (mul_mat_vec_f_router_cuda takes it for up to GGML_CUDA_ROUTER_RANK_MAX_USED of 128 or 256 experts):
//   the same experts in the same order and the same weights, bit for bit, as the rounds of mul_mat_vec_f_router_wait.
// The rounds are latency bound, ~420 cycles each, ~4400 cycles for 8 of 256 after the last logit lands. Here warp 0 polls and gates the logits as
//   topk_moe_select does, then warp w ranks the experts 32w..32w+31 among themselves (31 shuffles); the n_used best of each group are its candidates, and
//   every expert of the top n_used is one of them, so a candidate's rank among all experts is its rank in its group plus, for every other group, how many of
//   that group's candidates beat it. Warp 0 then normalizes and scales the weights as topk_moe_select does: the k-th weight in lane k, each lane summing the
//   weights of the experts it would have owned there in the order they were picked. ~2800 cycles with the barriers.
// Ranks compare integer keys: key() orders as the float compares of topk_moe_select (FSETP.FTZ: -0 and +0 are one key; the values are never NaN, which is
//   sanitized, nor denormal, as outputs of FTZ ops), and "o beats v" (larger, or equal and a lower expert) is one compare against a threshold that carries
//   the tie-break: ko > kv - (o's expert < v's expert). An unused candidate slot holds INT_MIN, a candidate's own group compares against INT_MAX.
// The code is cold in the instruction cache at every launch (~10 cycles per instruction met for the first time), so a first pass runs all of it on made-up
//   logits while the rows are read: warp 0 the gating, warps 1.. the ranks, warp 1 the weights, at the same time and without barriers (that pass ends after
//   ~2 us; one warp running it all ended after the logits had landed). The waiting block is block 0, so it starts first, and the kernel keeps to 64
//   registers, so that 4 blocks of 256 fit an SM and the launch is one wave.
// rpb router rows per block share the y loads; each row sums as with a block of its own, so the logits are those of mul_mat_vec_f.
#define GGML_CUDA_ROUTER_RANK_MAX_USED 8

static __device__ __forceinline__ int mul_mat_vec_f_router_key(const float f) {
    const int b = __float_as_int(f);
    const int c = b == INT_MIN ? 0 : b;
    return c ^ ((c >> 31) & INT_MAX);
}

// the top bit of j: lane ^ j for j in [2^b, 2^(b+1)) is below lane iff bit b of lane is set
static constexpr __device__ int mul_mat_vec_f_router_msb(const int j) {
    return j >= 16 ? 4 : j >= 8 ? 3 : j >= 4 ? 2 : j >= 2 ? 1 : 0;
}

template <int n_experts>
static __device__ __forceinline__ void mul_mat_vec_f_router_rank_wait(const ggml_cuda_mmvf_topk_args & topk, float * dst, const int tid) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_groups  = topk_moe_experts_per_thread(n_experts);
    constexpr int max_used  = GGML_CUDA_ROUTER_RANK_MAX_USED;
    constexpr int parts     = warp_size/max_used;
    static_assert(warp_size == 32 && n_groups*warp_size == n_experts && n_groups <= 256/warp_size && n_groups % parts == 0,
        "one warp of the block per 32 experts, whole parts of a warp per group");
    static_assert(2*warp_size <= GGML_CUDA_BLOCK_FLAGS - GGML_CUDA_TOPK_WARM, "scratch of the first pass");
    __shared__ float                 s_val[n_experts];
    __shared__ __align__(16) int     s_ckey[n_groups][max_used];
    __shared__ int                   s_clane[n_groups][max_used];
    __shared__ float                 s_cval[n_groups][max_used];
    __shared__ float                 s_sel[max_used];
    __shared__ int                   s_own[max_used];
    __shared__ int                   s_ready;

    const int warp = tid / warp_size;
    const int lane = tid % warp_size;
    volatile unsigned long long * slots = topk.slots;
    unsigned int * epoch        = (unsigned int *) (topk.slots + GGML_CUDA_TOPK_SLOT_COUNT);
    float        * warm_weights = (float *) (epoch + (GGML_CUDA_TOPK_WARM - GGML_CUDA_TOPK_EPOCH));
    int32_t      * warm_ids     = (int32_t *) (warm_weights + warp_size);
    const topk_moe_config config = mul_mat_vec_f_topk_moe_config(topk);

    unsigned int tag = 0;
    if (warp == 0) {
        tag = *(volatile unsigned int *) epoch + 1;
    }
    bool warm = true;
#pragma unroll 1
    while (true) {
        if (warp == 0) {
            float wt[n_groups];
            bool  ready = false;
            if (warm) {
#pragma unroll
                for (int i = 0; i < n_groups; i++) {
                    wt[i] = (float) (i*warp_size + lane);
                }
            } else {
                ready = ggml_cuda_spin_until([&]() {
                    bool ready = true;
#pragma unroll
                    for (int i = 0; i < n_groups; i++) {
                        const unsigned long long v = slots[i*warp_size + lane];
                        wt[i] = __uint_as_float((unsigned int) v);
                        ready = ready && (unsigned int) (v >> 32) == tag;
                    }
                    return __all_sync(0xFFFFFFFF, ready);
                });
                if (!ready) {
                    ggml_cuda_spin_trap(GGML_CUDA_SPIN_SITE_ROUTER, -1, GGML_CUDA_SPIN_PEER_BLOCKS);
                }
#pragma unroll
                for (int i = 0; i < n_groups; i++) {
                    dst[i*warp_size + lane] = wt[i];
                }
                // every router block read the epoch before it stored its tag
                if (lane == 0) {
                    *(volatile unsigned int *) epoch = tag;
                }
            }
            if (!config.delayed_softmax) {
                if (config.use_sigmoid) {
                    sigmoid_warp_inplace<n_groups, false>(wt, n_experts, lane);
                } else if (config.use_sqrt_softplus) {
                    sqrt_softplus_warp_inplace<n_groups, false>(wt, n_experts, lane);
                } else {
                    softmax_warp_inplace<n_groups, false>(wt, n_experts, lane);
                }
            }
#pragma unroll
            for (int i = 0; i < n_groups; i++) {
                s_val[i*warp_size + lane] = __isnanf(wt[i]) ? -FLT_MAX : wt[i];
            }
            if (lane == 0) {
                s_ready = ready;
            }
        }
        // the first pass has no barriers: its stages run at the same time on whatever the shared memory holds
        if (!warm) {
            __syncthreads();
        }
        const bool ready  = !warm && s_ready;
        const int  n_used = ready ? topk.n_used : 1;

        if (warp < n_groups && (!warm || warp != 0)) {
            const float v  = s_val[warp*warp_size + lane];
            const int   kv = mul_mat_vec_f_router_key(v);
            int th[5];
#pragma unroll
            for (int b = 0; b < 5; b++) {
                th[b] = kv - ((lane >> b) & 1);
            }
            // four counts: the shuffles do not wait for the adds
            int r4[4] = { 0, 0, 0, 0 };
#pragma unroll
            for (int j = 1; j < warp_size; j++) {
                r4[j % 4] += __shfl_xor_sync(0xFFFFFFFF, kv, j) > th[mul_mat_vec_f_router_msb(j)];
            }
            const int r = (r4[0] + r4[1]) + (r4[2] + r4[3]);
            if (r < max_used) {
                s_ckey[warp][r]  = r < n_used ? kv : INT_MIN;
                s_clane[warp][r] = lane;
                s_cval[warp][r]  = v;
            }
        }
        if (!warm) {
            __syncthreads();
        }
        // lane max_used*part + t: candidate t of this warp against the groups part, part + parts, ..., a lower group wins ties
        if (warp < n_groups && (!warm || warp != 0)) {
            const int   t    = lane % max_used;
            const int   part = lane / max_used;
            const int   kc   = s_ckey[warp][t];
            const int   l    = s_clane[warp][t];
            const float cv   = s_cval[warp][t];
            int g = part == 0 ? t : 0;
#pragma unroll
            for (int w0 = 0; w0 < n_groups; w0 += parts) {
                const int  w  = w0 + part;
                const int4 a  = *(const int4 *) &s_ckey[w][0];
                const int4 b  = *(const int4 *) &s_ckey[w][4];
                const int  th = w == warp ? INT_MAX : (int) ((unsigned int) kc - (w < warp ? 1u : 0u));
                g += (a.x > th) + (a.y > th) + (a.z > th) + (a.w > th) + (b.x > th) + (b.y > th) + (b.z > th) + (b.w > th);
            }
#pragma unroll
            for (int offset = max_used; offset < warp_size; offset *= 2) {
                g += __shfl_xor_sync(0xFFFFFFFF, g, offset);
            }
            if (part == 0 && t < n_used && g < n_used) {
                (ready ? topk.ids : warm_ids)[g] = warp*warp_size + l;
                s_sel[g] = cv;
                s_own[g] = l;
            }
        }
        if (!warm) {
            __syncthreads();
        }
        if (warp == (warm ? 1 : 0)) {
            float ow[n_groups];
#pragma unroll
            for (int i = 0; i < n_groups; i++) {
                ow[i] = 0.0f;
            }
            ow[0] = lane < n_used ? s_sel[lane] : 0.0f;
            if (config.with_norm) {
                float wt_sum = 0.0f;
#pragma unroll
                for (int k = 0; k < max_used; k++) {
                    if (k < n_used && s_own[k] == lane) {
                        wt_sum += s_sel[k];
                    }
                }
                wt_sum              = warp_reduce_sum(wt_sum);
                wt_sum              = max(wt_sum, topk.clamp);
                const float inv_sum = 1.0f / wt_sum;
#pragma unroll
                for (int i = 0; i < n_groups; i++) {
                    ow[i] *= inv_sum;
                }
            }
            if (config.delayed_softmax) {
                softmax_warp_inplace<n_groups, true>(ow, n_used, lane);
            }
            if (lane < n_used) {
                (ready ? topk.weights : warm_weights)[lane] = ow[0] * topk.scale;
            }
        }
        if (ready) {
            return;
        }
        warm = false;
    }
}

// Block 0 waits for the logits and selects, blocks 1..n_experts/rpb are rpb router rows each, the last block is the sibling.
template <int n_experts, int rpb>
static __global__ void __launch_bounds__(256, 4) mul_mat_vec_f_router_rank(
        const float * x_ptr, const float * y_ptr, const float * sib_x, float * dst, float * sib_dst, const int ncols2,
        const int stride_row, const ggml_cuda_mmvf_topk_args topk) {
    constexpr int block_size = 256;
    constexpr int warp_size  = ggml_cuda_get_physical_warp_size();
    constexpr int n_blocks   = n_experts/rpb;
    static_assert(n_experts % rpb == 0 && rpb <= block_size/warp_size, "a warp per row for the last reduction");
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    if (blockIdx.x == 0) {
        mul_mat_vec_f_router_rank_wait<n_experts>(topk, dst, tid);
        return;
    }
    const bool sib  = blockIdx.x == n_blocks + 1;
    const int  nr   = sib ? 1 : rpb;
    const int  row0 = (blockIdx.x - 1)*rpb;
    const float2 * GGML_CUDA_RESTRICT x2 = (const float2 *) (sib ? sib_x : x_ptr + int64_t(row0)*stride_row);
    const float2 * GGML_CUDA_RESTRICT y2 = (const float2 *) y_ptr;
    const int stride_row2 = stride_row/2;

    // read early, its latency hides behind the dot product; warp r stores the logit of row r
    unsigned int tag = 0;
    if (tid % warp_size == 0 && tid / warp_size < nr && !sib) {
        tag = *(volatile unsigned int *) (topk.slots + GGML_CUDA_TOPK_SLOT_COUNT) + 1;
    }

    __shared__ float buf_iw[rpb][warp_size];
    if (tid < warp_size) {
#pragma unroll
        for (int r = 0; r < rpb; r++) {
            buf_iw[r][tid] = 0.0f;
        }
    }
    __syncthreads();

    float sumf[rpb];
#pragma unroll
    for (int r = 0; r < rpb; r++) {
        sumf[r] = 0.0f;
    }
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpy = y2[col2];
#pragma unroll
        for (int r = 0; r < rpb; r++) {
            if (r < nr) {
                const float2 tmpx = x2[r*stride_row2 + col2];
                ggml_cuda_mad(sumf[r], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[r], tmpx.y, tmpy.y);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < rpb; r++) {
        sumf[r] = warp_reduce_sum<warp_size>(sumf[r]);
        buf_iw[r][tid/warp_size] = sumf[r];
    }
    __syncthreads();
    const int r = tid/warp_size;
    if (r >= nr) {
        return;
    }
    const int   lane = tid % warp_size;
    const float sum  = warp_reduce_sum<warp_size>(buf_iw[r][lane]);
    if (lane == 0) {
        if (sib) {
            sib_dst[0] = sum;
        } else {
            *(volatile unsigned long long *) (topk.slots + row0 + r) = ((unsigned long long) tag << 32) | __float_as_uint(sum);
        }
    }
}

static void mul_mat_vec_f_router_cuda(const float * x, const float * y, const ggml_cuda_mm_fusion_args_launch & fusion,
        float * dst, const int64_t ncols, const int64_t stride_row, cudaStream_t stream) {
    // rows of a multiple of 512 make mul_mat_vec_f take 256 threads
    GGML_ASSERT(ncols % 512 == 0 && fusion.sib_x != nullptr && fusion.sib_block0 == 0 && !fusion.gate && !fusion.x_bias &&
        !fusion.gate_bias && !fusion.gdn_beta);
    // GGML_CUDA_NO_ROUTER_RANK=1 keeps the rounds
    static const bool no_rank = getenv("GGML_CUDA_NO_ROUTER_RANK") != nullptr;
    const bool rank = !no_rank && fusion.topk.n_used <= GGML_CUDA_ROUTER_RANK_MAX_USED && fusion.topk.n_expert <= 256;
    constexpr int rpb = 4;
    const int n_blocks = rank ? fusion.topk.n_expert/rpb + 2 : fusion.topk.n_expert + 2;
    const ggml_cuda_kernel_launch_params launch_params = {dim3(n_blocks, 1, 1), dim3(256, 1, 1), 0, stream};
    const auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params, x, y, (const float *) fusion.sib_x, dst, fusion.sib_dst, int(ncols/2),
            int(stride_row), fusion.topk);
    };
    switch (fusion.topk.n_expert) {
        case 128: rank ? launch(mul_mat_vec_f_router_rank<128, rpb>) : launch(mul_mat_vec_f_router<128>); break;
        case 256: rank ? launch(mul_mat_vec_f_router_rank<256, rpb>) : launch(mul_mat_vec_f_router<256>); break;
        case 512: launch(mul_mat_vec_f_router<512>); break;
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(!ids || ne12 <= MMVF_MAX_BATCH_SIZE);
    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_launch fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1 || ((fusion->gdn_beta || fusion->sib_rows) && ggml_cuda_mmvf_fuses_tokens(src0, src1, fusion->gate)) ||
            (fusion->sib && !fusion->sib_rows && !fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->gdn_beta));
        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, fusion->gdn_x ? fusion->gdn_x : src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
        if (fusion->gdn_beta && !fusion->gdn_x) {
            GGML_ASSERT(!ids && fusion->gate && !fusion->x_bias && !fusion->gate_bias);
            GGML_ASSERT(ggml_is_contiguous(dst) && dst->ne[2] == 1 && dst->ne[3] == 1);
            for (const ggml_tensor * t : { fusion->gdn_dt, fusion->gdn_a }) {
                GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == dst->ne[0] && ggml_nelements(t) == dst->ne[0]);
            }
            GGML_ASSERT(fusion->gdn_beta->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->gdn_beta) &&
                ggml_nelements(fusion->gdn_beta) == ggml_nelements(dst));
            fusion_local.gdn_dt   = (const float *) fusion->gdn_dt->data;
            fusion_local.gdn_a    = (const float *) fusion->gdn_a->data;
            fusion_local.gdn_beta = (float *) fusion->gdn_beta->data;
        }
        if (fusion->gdn_x) {
            // the projections of a gated delta net layer, one token (see mul_mat_vec_f, gdn_qkv): the rows of src0 (qkv)
            //   with the convolution in their epilogue, sib's (z) into sib_dst, and gdn_x's (alpha) with the gate (beta)
            //   and the gated delta net epilogue into gdn_dst
            GGML_ASSERT(ggml_cuda_mul_mat_vec_f_gdn_qkv(src0, src1, fusion->sib, fusion->gdn_x, fusion->gate) && !ids && fusion->sib &&
                fusion->sib_dst && fusion->gdn_dst && fusion->gdn_beta && !fusion->x_bias && !fusion->gate_bias && !fusion->sib_rows &&
                !fusion->sib2 && !fusion->sib_y && !fusion->sib_gate);
            GGML_ASSERT(ggml_nrows(dst) == 1 && ggml_is_contiguous(dst));
            const int64_t nrows_gdn = fusion->gdn_x->ne[1];
            for (const ggml_tensor * t : { fusion->gdn_dt, fusion->gdn_a, (const ggml_tensor *) fusion->gdn_beta, (const ggml_tensor *) fusion->gdn_dst }) {
                GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == nrows_gdn);
            }
            GGML_ASSERT(fusion->sib_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->sib_dst) &&
                ggml_nelements(fusion->sib_dst) == fusion->sib->ne[1]);
            fusion_local.sib_x      = fusion->sib->data;
            fusion_local.sib_dst    = (float *) fusion->sib_dst->data;
            fusion_local.sib_block0 = (uint32_t) ne01;
            fusion_local.sib_nrows  = (uint32_t) fusion->sib->ne[1];
            fusion_local.gdn_x      = fusion->gdn_x->data;
            fusion_local.gdn_dst    = (float *) fusion->gdn_dst->data;
            fusion_local.gdn_block0 = (uint32_t) (ne01 + fusion->sib->ne[1]);
            fusion_local.gdn_nrows  = (uint32_t) nrows_gdn;
            fusion_local.gdn_dt     = (const float *) fusion->gdn_dt->data;
            fusion_local.gdn_a      = (const float *) fusion->gdn_a->data;
            fusion_local.gdn_beta   = (float *) fusion->gdn_beta->data;
            if (fusion->conv_w != nullptr) {
                GGML_ASSERT(fusion->conv_states && fusion->conv_ids && fusion->conv_dst);
                fusion_local.conv.states     = fusion->conv_states;
                fusion_local.conv.ids        = fusion->conv_ids;
                fusion_local.conv.row_stride = fusion->conv_row_stride;
                fusion_local.conv.w          = fusion->conv_w;
                fusion_local.conv.w_stride   = fusion->conv_w_stride;
                fusion_local.conv.dst        = fusion->conv_dst;
                fusion_local.conv.slot0      = fusion->conv_slot[0];
                fusion_local.conv.slot1      = fusion->conv_slot[1];
            }
        }
        GGML_ASSERT(fusion->conv_w == nullptr || fusion->gdn_x != nullptr);
        if (fusion->sib && fusion->sib_gate) {
            // with ids and a gate: the GLU of the sibling pair sib/sib_gate past the used experts, nothing else fused
            //   (ggml_cuda_try_defer_glu)
            GGML_ASSERT(ggml_cuda_mul_mat_vec_f_moe_glu_sibling(src0, fusion->gate, src1, ids, fusion->sib, fusion->sib_gate) &&
                fusion->sib_dst && !fusion->x_bias && !fusion->gate_bias && !fusion->gdn_beta && !fusion->sib_rows && !fusion->sib_y);
            GGML_ASSERT(fusion->sib_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->sib_dst) &&
                ggml_nelements(fusion->sib_dst) == ne01);
            fusion_local.sib_x    = fusion->sib->data;
            fusion_local.sib_gate = fusion->sib_gate->data;
            fusion_local.sib_dst  = (float *) fusion->sib_dst->data;
            fusion_local.sib_y    = src1_d; // the experts' input, which makes the sibling a channel of the grid
        }
        if (fusion->sib && fusion->sib_y) {
            // with ids: the sibling channel past the used experts, nothing else fused
            GGML_ASSERT(ggml_cuda_mul_mat_vec_f_moe_sibling(src0, src1, ids, fusion->sib, fusion->sib_y) && fusion->sib_out &&
                !fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->gdn_beta && !fusion->sib_rows);
            fusion_local.sib_x   = fusion->sib->data;
            fusion_local.sib_y   = (const float *) fusion->sib_y->data;
            fusion_local.sib_dst = fusion->sib_out;
        }
        if (fusion->sib && !fusion->sib_rows && !fusion->sib_y && !fusion->sib_gate && !fusion->gdn_x) {
            GGML_ASSERT(!ids && !fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->gdn_beta);
            GGML_ASSERT(src0->type == GGML_TYPE_F32 && fusion->sib->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->sib->ne[0] == ne00 && ggml_nrows(fusion->sib) == 1 && ggml_is_contiguous(fusion->sib));
            GGML_ASSERT(ggml_nrows(dst) == dst->ne[1] && ggml_is_contiguous(dst) && fusion->sib_dst->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(fusion->sib_dst) && ggml_nelements(fusion->sib_dst) == dst->ne[1]);
            fusion_local.sib_x   = fusion->sib->data;
            fusion_local.sib_dst = (float *) fusion->sib_dst->data;
        }
        if (fusion->sib && fusion->sib_rows) {
            // the rows of up to three sibling matrices after those of src0, see mul_mat_vec_f; several tokens take
            //   whole blocks of 4 rows of each
            GGML_ASSERT(!ids && !fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->gdn_beta);
            GGML_ASSERT(ggml_nrows(src1) == src1->ne[1] && ggml_nrows(dst) == dst->ne[1] && ggml_is_contiguous(dst) &&
                ggml_nrows(src0) == ne01 && (dst->ne[1] == 1 || ne01 % 4 == 0));
            GGML_ASSERT(!fusion->sib3 || fusion->sib2);
            int64_t row0 = ne01;
            const ggml_tensor * sibs[3] = { fusion->sib,     fusion->sib2,     fusion->sib3     };
            ggml_tensor       * dsts[3] = { fusion->sib_dst, fusion->sib2_dst, fusion->sib3_dst };
            for (int k = 0; k < 3 && sibs[k] != nullptr; ++k) {
                const ggml_tensor * sib = sibs[k];
                GGML_ASSERT(sib->type == src0->type && sib->ne[0] == ne00 && sib->nb[1] == src0->nb[1] && ggml_nrows(sib) == sib->ne[1]);
                GGML_ASSERT(dsts[k]->type == GGML_TYPE_F32 && ggml_is_contiguous(dsts[k]) &&
                    ggml_nelements(dsts[k]) == sib->ne[1]*dst->ne[1] && (dst->ne[1] == 1 || sib->ne[1] % 4 == 0));
                if (k == 0) {
                    fusion_local.sib_x      = sib->data;
                    fusion_local.sib_dst    = (float *) dsts[k]->data;
                    fusion_local.sib_block0 = (uint32_t) row0;
                    fusion_local.sib_nrows  = (uint32_t) sib->ne[1];
                } else if (k == 1) {
                    fusion_local.sib2_x      = sib->data;
                    fusion_local.sib2_dst    = (float *) dsts[k]->data;
                    fusion_local.sib2_block0 = (uint32_t) row0;
                    fusion_local.sib2_nrows  = (uint32_t) sib->ne[1];
                } else {
                    fusion_local.gdn_x      = sib->data;
                    fusion_local.gdn_dst    = (float *) dsts[k]->data;
                    fusion_local.gdn_block0 = (uint32_t) row0;
                    fusion_local.gdn_nrows  = (uint32_t) sib->ne[1];
                }
                row0 += sib->ne[1];
            }
        }
        if (fusion->topk_ids) {
            GGML_ASSERT(!ids && !fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->gdn_beta && !fusion->sib_rows);
            GGML_ASSERT(src0->type == GGML_TYPE_F32 && ggml_nrows(dst) == 1 && ggml_is_contiguous(dst) &&
                dst->ne[0] == fusion->topk_n_expert && fusion->topk_weights && fusion->topk_ticket);
            fusion_local.topk.logits          = dst_d;
            fusion_local.topk.weights         = fusion->topk_weights;
            fusion_local.topk.ids             = fusion->topk_ids;
            fusion_local.topk.ticket          = fusion->topk_ticket;
            if (fusion->topk_slots) {
                fusion_local.topk.slots = fusion->topk_slots;
            }
            fusion_local.topk.n_expert        = fusion->topk_n_expert;
            fusion_local.topk.n_used          = fusion->topk_n_used;
            fusion_local.topk.clamp           = fusion->topk_clamp;
            fusion_local.topk.scale           = fusion->topk_scale;
            fusion_local.topk.sigmoid         = fusion->topk_sigmoid;
            fusion_local.topk.sqrt_softplus   = fusion->topk_sqrt_softplus;
            fusion_local.topk.with_norm       = fusion->topk_with_norm;
            fusion_local.topk.delayed_softmax = fusion->topk_delayed_softmax;
        }
    }

    if (fusion && fusion->norm_x) {
        // y is the gated norm of norm_x, which the launch computes as it reads it, nothing else fused
        GGML_ASSERT(ggml_cuda_mul_mat_vec_f_norm_gated(src0, src1, dst, fusion->norm_x, fusion->norm_z, fusion->norm_w) && !ids &&
            !fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->gdn_beta && !fusion->gdn_x && !fusion->sib &&
            !fusion->topk_ids);
        src1_d                  = (const float *) fusion->norm_x->data;
        fusion_local.norm.z     = (const float *) fusion->norm_z->data;
        fusion_local.norm.w     = (const float *) fusion->norm_w->data;
        fusion_local.norm.ncols = (int) fusion->norm_x->ne[0];
        fusion_local.norm.eps   = fusion->norm_eps;
    }

    // the all-reduce of the output folded into the launch, if it can take it (see ggml_backend_cuda_context::ar_fold),
    //   not with hyper-connections (see ggml_backend_cuda_session_allreduce_next)
    if (ctx.ar_fold.next == dst && !ctx.ar_fold.hc) {
        fusion_local.ar        = ctx.ar_fold.next_call.args;
        fusion_local.ar_pushed = &ctx.ar_fold.pushed;
    }
    // with ar_weights that of the MoE sum of the experts and the sibling this launch computes, which the caller checked
    //   is next (see ggml_cuda_try_fuse_moe_down_shexp)
    if (fusion && fusion->ar_weights && ctx.ar_fold.next != nullptr) {
        GGML_ASSERT(ids && fusion->sib_y && fusion->ar_gate && ggml_is_contiguous(fusion->ar_weights) &&
            ggml_nelements(fusion->ar_weights) == ne1);
        fusion_local.ar         = ctx.ar_fold.next_call.args;
        fusion_local.ar_pushed  = &ctx.ar_fold.pushed;
        fusion_local.ar_moe     = true;
        fusion_local.ar_weights = (const float *) fusion->ar_weights->data;
        fusion_local.ar_gate    = (const float *) fusion->ar_gate->data;
        fusion_local.ar_n_used  = (int) ne1;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            if (fusion && fusion->topk_ids && fusion->topk_slots) {
                mul_mat_vec_f_router_cuda(src0_d, src1_d, fusion_local, dst_d, ne00, s01, ctx.stream());
                break;
            }
            // a sibling row is one more block, sibling rows as many more
            const int64_t nrows = ne01 + (fusion_local.sib_block0 > 0 ? fusion_local.sib_nrows + fusion_local.sib2_nrows + fusion_local.gdn_nrows :
                                          fusion_local.sib_x != nullptr ? 1 : 0);
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, nrows, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0->data;
            const int64_t nrows = ne01 + fusion_local.sib_nrows + fusion_local.sib2_nrows + fusion_local.gdn_nrows;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, nrows, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
            const int64_t nrows = ne01 + fusion_local.sib_nrows + fusion_local.sib2_nrows + fusion_local.gdn_nrows;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, nrows, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne0  =  dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00;
    const int64_t stride_col_y       = ne10;
    const int64_t stride_col_dst     = id == ctx.device ? ne0 : row_diff; // main device has larger memory buffer
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t nchannels_dst      = 1;
    const int64_t stride_channel_x   = 0;
    const int64_t stride_channel_y   = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x         = 1;
    const int64_t nsamples_dst       = 1;
    const int64_t stride_sample_x    = 0;
    const int64_t stride_sample_y    = 0;
    const int64_t stride_sample_dst  = 0;

    ggml_cuda_mm_fusion_args_launch empty{};
    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED_VARS(ctx, src1, dst, src1_ddq_i, src1_ncols, src1_padded_row_size);
}

bool ggml_cuda_mmvf_fuses_tokens(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * gate) {
    if (src1->ne[1] == 1) {
        return true;
    }
    const int    cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    return src1->type == GGML_TYPE_F32 && src1->ne[1] <= MMVF_MAX_BATCH_SIZE && mmvf_use_load16(ts_src0 == 2, cc, src1->ne[1],
        src0->ne[0], src0->nb[1]/ts_src0, src1->nb[1]/ts_src1, src0->nb[2]/ts_src0, src0->nb[3]/ts_src0,
        src1->nb[2]/ts_src1, src1->nb[3]/ts_src1, src0->data, src1->data, gate ? gate->data : nullptr);
}

bool ggml_cuda_mul_mat_vec_f_moe_sibling(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * sib, const ggml_tensor * sib_y) {
    // one token of 2-byte experts with the 16-byte loads; the sibling has one expert's layout and its own f32 input
    if (!ids || (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) || src1->type != GGML_TYPE_F32 ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || ids->ne[1] != 1 || src1->ne[1] != ids->ne[0] || src0->ne[3] != 1 ||
            sib->type != src0->type || sib->ne[0] != src0->ne[0] || sib->ne[1] != src0->ne[1] || sib->nb[1] != src0->nb[1] ||
            ggml_nrows(sib) != sib->ne[1] || sib_y->type != GGML_TYPE_F32 || !ggml_is_contiguous(sib_y) ||
            ggml_nelements(sib_y) != src0->ne[0] || (uintptr_t) sib->data % 16 != 0 || (uintptr_t) sib_y->data % 16 != 0) {
        return false;
    }
    const int    cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    return src0->nb[0] == ts_src0 && src1->nb[0] == ts_src1 &&
        mmvf_use_load16(true, cc, 1, src0->ne[0], src0->nb[1]/ts_src0, src1->nb[2]/ts_src1, src0->nb[2]/ts_src0,
            src0->nb[3]/ts_src0, src1->nb[1]/ts_src1, src1->nb[3]/ts_src1, src0->data, src1->data, nullptr);
}

bool ggml_cuda_mul_mat_vec_f_moe_glu_sibling(const ggml_tensor * src0, const ggml_tensor * gate0, const ggml_tensor * src1,
        const ggml_tensor * ids, const ggml_tensor * sib, const ggml_tensor * sib_gate) {
    // one token of 2-byte experts with the 16-byte loads, one input for all of them; rows of > 512, which the experts'
    //   launch keeps one per block without the pair too (see launch_mul_mat_vec_f_cuda)
    if (!ids || (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) || src1->type != GGML_TYPE_F32 ||
            gate0 == nullptr || gate0->type != src0->type || !ggml_are_same_shape(gate0, src0) || !ggml_are_same_stride(gate0, src0) ||
            src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || ids->ne[1] != 1 || src0->ne[3] != 1 || src0->ne[0] <= 512) {
        return false;
    }
    // the pair has one expert's layout
    for (const ggml_tensor * s : { sib, sib_gate }) {
        if (s->type != src0->type || s->ne[0] != src0->ne[0] || s->ne[1] != src0->ne[1] || s->nb[1] != src0->nb[1] ||
                ggml_nrows(s) != s->ne[1] || (uintptr_t) s->data % 16 != 0) {
            return false;
        }
    }
    const int    cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    return src0->nb[0] == ts_src0 && src1->nb[0] == ts_src1 &&
        mmvf_use_load16(true, cc, 1, src0->ne[0], src0->nb[1]/ts_src0, src1->nb[2]/ts_src1, src0->nb[2]/ts_src0,
            src0->nb[3]/ts_src0, src1->nb[1]/ts_src1, src1->nb[3]/ts_src1, src0->data, src1->data, gate0->data);
}

bool ggml_cuda_mul_mat_vec_f_gdn_qkv(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * sib,
        const ggml_tensor * gdn_x, const ggml_tensor * gate) {
    // one token of 2-byte rows with the 16-byte loads, all matrices of the same type, K and row stride
    if ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) || src1->type != GGML_TYPE_F32 || ggml_nrows(src1) != 1 ||
            !ggml_is_contiguous(src1) || src1->ne[0] != src0->ne[0] || ggml_nrows(src0) != src0->ne[1] || gdn_x->ne[1] != gate->ne[1]) {
        return false;
    }
    for (const ggml_tensor * t : { sib, gdn_x, gate }) {
        if (t->type != src0->type || t->ne[0] != src0->ne[0] || t->nb[1] != src0->nb[1] || ggml_nrows(t) != t->ne[1] ||
                (uintptr_t) t->data % 16 != 0) {
            return false;
        }
    }
    const int    cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    return src0->nb[0] == ts_src0 && src1->nb[0] == ts_src1 &&
        mmvf_use_load16(true, cc, 1, src0->ne[0], src0->nb[1]/ts_src0, src1->nb[1]/ts_src1, src0->nb[2]/ts_src0,
            src0->nb[3]/ts_src0, src1->nb[2]/ts_src1, src1->nb[3]/ts_src1, src0->data, src1->data, gate->data);
}

bool ggml_cuda_mul_mat_vec_f_norm_gated(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
        const ggml_tensor * x, const ggml_tensor * z, const ggml_tensor * w) {
    // one token of f16 rows with the 16-byte loads and one matrix, y being the norm's rows of 128 one after the other
    const int64_t ncols = src0->ne[0];
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ggml_nrows(src1) != 1 ||
            !ggml_is_contiguous(src1) || src1->ne[0] != ncols || ggml_nrows(src0) != src0->ne[1] || ggml_nrows(dst) != 1 ||
            !ggml_is_contiguous(dst) || dst->ne[0] != src0->ne[1] || ncols % 128 != 0) {
        return false;
    }
    for (const ggml_tensor * t : { x, z }) {
        if (t->type != GGML_TYPE_F32 || t->ne[0] != 128 || ggml_nelements(t) != ncols || !ggml_is_contiguous(t) ||
                (uintptr_t) t->data % 16 != 0) {
            return false;
        }
    }
    if (w->type != GGML_TYPE_F32 || w->ne[0] != 128 || ggml_nelements(w) != 128 || !ggml_is_contiguous(w) || (uintptr_t) w->data % 16 != 0) {
        return false;
    }
    const int    device  = ggml_cuda_get_device();
    const int    cc      = ggml_cuda_info().devices[device].cc;
    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    // every lane of a block takes as many columns: the norm's shuffles need whole warps
    return src0->nb[0] == ts_src0 && (ncols/8) % mmvf_block_size(ncols, 8, device) == 0 &&
        mmvf_use_load16(true, cc, 1, ncols, src0->nb[1]/ts_src0, src1->nb[1]/ts_src1, src0->nb[2]/ts_src0,
            src0->nb[3]/ts_src0, src1->nb[2]/ts_src1, src1->nb[3]/ts_src1, src0->data, x->data, nullptr);
}

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }
                // Volta at 4 columns: a 2560 x 512 router 12.0 us vs cuBLAS 21.3 (30.7 with the f16 compute type,
                //   which converts the weights on every call), 4096 x 256 11.2 vs 16.5
                if (cc >= GGML_CUDA_CC_VOLTA) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        case GGML_TYPE_F16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                // MMF needs full row tiles, for other row counts MMVF still beats cuBLAS at small batch size
                if (src0_small && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                // On Volta the 16-byte loads beat mmf at 2 tokens, and up to 6 when mmf's 32-row blocks would leave
                // half of the SMs idle (f16 1280x17408 at 5 tokens 71 vs 83 us); from 3 tokens mmf is otherwise
                // faster (4352x5120 at 3 tokens 64 vs 68 us, and 65 vs 86 us at 6), and without the 16-byte loads
                // the old limit stays
                const bool load16 = src0_ne[0] % 8 == 0 && src0_nb[1] % 16 == 0 && src0_nb[2] % 16 == 0 && src0_nb[3] % 16 == 0;
                if (cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING && load16) {
                    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
                    return src0_small && (ne11 <= 2 || (ne11 <= 6 && src0_ne[1] <= 16*nsm));
                }
                if (fp16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (GGML_CUDA_CC_IS_RDNA(cc) && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (fp16_mma_hardware_available(cc)) {
                    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
                        return ne11 <= 3;
                    }
                    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
                        return ne11 <= 5;
                    }
                    return ne11 <= 2;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                // MMF needs full row tiles, for other row counts MMVF still beats cuBLAS at small batch size
                if (src0_small && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (GGML_CUDA_CC_IS_RDNA(cc) && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        default:
            return false;
    }
}
