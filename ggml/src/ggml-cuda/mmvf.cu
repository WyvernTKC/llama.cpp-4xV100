#include "ggml.h"
#include "common.cuh"
#include "unary.cuh"
#include "mmvf.cuh"
#include "mmf.cuh"
#include "convert.cuh"
#include "topk-moe.cuh"

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

    constexpr int experts_per_thread = n_experts / WARP_SIZE;
    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = __ldcg(topk.logits + i*WARP_SIZE + lane);
    }
    const float bias_v[1] = { 0.0f };

    topk_moe_config config;
    config.use_sigmoid       = topk.sigmoid;
    config.use_sqrt_softplus = topk.sqrt_softplus;
    config.with_norm         = topk.with_norm;
    config.delayed_softmax   = topk.delayed_softmax;
    config.n_groups          = 1;
    config.n_group_used      = 1;
    topk_moe_select<n_experts, false, false>(wt, bias_v, topk.weights, topk.ids, topk.n_used, topk.clamp, topk.scale, config);

    if (lane == 0) {
        *topk.ticket = 0;
    }
}

// rpb > 1 puts that many consecutive rows in a block: they share the y loads, and each row still sums in
// exactly the order it would with a block of its own. With several tokens it needs load16.
// load16 (2-byte weights) reads 8 weights per 16-byte load instead of 2 per 4-byte load, which changes the
// order in which a row sums, so its results can differ in the last bits.
// topk_n: the expert count of a topk_moe run in the last block (topk_args), see mul_mat_vec_f_topk_moe
template <typename T, typename type_acc, int ncols_dst, int block_size, bool has_fusion = false, bool is_multi_token_id = false, int rpb = 1, bool load16 = false,
          int topk_n = 0>
static __global__ void mul_mat_vec_f(
        const T * x_ptr, const float * y_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const int ncols2, const uint3 nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const std::conditional_t<(topk_n > 0), ggml_cuda_mmvf_topk_args, ggml_cuda_no_args> topk_args) {
    const T       * GGML_CUDA_RESTRICT x   = x_ptr;
    const float   * GGML_CUDA_RESTRICT y   = y_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;
    static_assert(rpb == 1 || ((ncols_dst == 1 || load16) && !is_multi_token_id), "rows per block need a single token or the 16-byte loads");
    const int row         = rpb*blockIdx.x; // the first of the block's rows
    [[maybe_unused]] const int stride_row2 = stride_row/2;
    // for MUL_MAT_ID - blockIdx.y = n_expert_used, blockIdx.z = ncols_dst (tokens)
    const int channel_dst = blockIdx.y;
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
        token_idx  = ids ? blockIdx.z                                          : 0;
        channel_x  = ids ? ids[blockIdx.y + token_idx * ids_stride]            : fastdiv((uint32_t) channel_dst, channel_ratio);
        channel_y  = ids ? fastmodulo(blockIdx.y, nchannels_y)                 : channel_dst;
        sample_dst = ids ? 0                                                   : blockIdx.z;
    }

    const int sample_x    = fastdiv((uint32_t) sample_dst, sample_ratio);
    const int sample_y    = sample_dst;

    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    y   += int64_t(sample_y)  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;
    if constexpr (is_multi_token_id) {
        y   += token_idx*stride_col_y2*2;
        dst += token_idx*stride_col_dst;
    }
    if constexpr (has_fusion && std::is_same_v<T, float>) {
        // the row of a sibling matrix with the same y is the last block's (one token, one row per block)
        if (fusion.sib_x != nullptr && fusion.sib_block0 == 0 && blockIdx.x == gridDim.x - 1) {
            x   = (const T *) fusion.sib_x;
            dst = fusion.sib_dst - row;
        }
    }
    if constexpr (has_fusion) {
        // sibling rows (one token, one row per block, one channel and sample): the blocks from sib_block0 on run the
        //   rows of a sibling matrix of the same type, width and y into its own dst, those from sib2_block0 on a second's
        if (fusion.sib_block0 > 0 && blockIdx.x >= fusion.sib_block0) {
            const bool second = fusion.sib2_x != nullptr && blockIdx.x >= fusion.sib2_block0;
            const int  r      = rpb*(blockIdx.x - (second ? fusion.sib2_block0 : fusion.sib_block0));
            x   = (const T *) (second ? fusion.sib2_x : fusion.sib_x) + int64_t(r)*stride_row;
            dst = (second ? fusion.sib2_dst : fusion.sib_dst) + (r - row);
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
                    const float4 tmpy_lo = y4[j*stride_col_y4 + 2*col8 + 0];
                    const float4 tmpy_hi = y4[j*stride_col_y4 + 2*col8 + 1];
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
                    const float4 tmpy_lo = y4[j*stride_col_y4 + 2*col8 + 0];
                    const float4 tmpy_hi = y4[j*stride_col_y4 + 2*col8 + 1];
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
                dst[dcol*stride_col_dst + drow]      = ggml_cuda_op_softplus_single(value + gdn_dt[drow]) * gdn_a[drow];
                gdn_beta[dcol*stride_col_dst + drow] = ggml_cuda_op_sigmoid_single(gate_value);
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

    dst[dcol*stride_col_dst + drow] = value;

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

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
        fusion.sib_x != nullptr;
    if constexpr (ncols_dst > 1 && !is_multi_token_id && sizeof(T) == 2) {
        if (load16) {
            if (has_fusion) {
                // with several tokens only the gate of a gated delta net layer, which has no biases, fuses
                GGML_ASSERT(fusion.gdn_beta && rpb == 1);
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, false, 1, true>);
            } else if (rpb == 4) {
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 4, true>);
            } else {
                GGML_ASSERT(rpb == 1);
                launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 1, true>);
            }
            return;
        }
    }
    if constexpr (ncols_dst == 1 && !is_multi_token_id && sizeof(T) == 2) {
        if (load16) {
            if (rpb == 4) {
                if (has_fusion) {
                    launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, false, 4, true>);
                } else {
                    launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 4, true>);
                }
            } else {
                GGML_ASSERT(rpb == 1);
                if (has_fusion) {
                    launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, false, 1, true>);
                } else {
                    launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, 1, true>);
                }
            }
            return;
        }
    }
    GGML_ASSERT(!load16);
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

    GGML_ASSERT(!has_fusion && "fusion with several tokens needs the 16-byte loads");

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

// the sibling rows of a launch (see mul_mat_vec_f) read their own matrices
static bool mmvf_siblings_aligned16(const ggml_cuda_mm_fusion_args_device & fusion) {
    return (uintptr_t) fusion.sib_x % 16 == 0 && (uintptr_t) fusion.sib2_x % 16 == 0;
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

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
        fusion.sib_x != nullptr;

    // On Volta four rows per block share the y loads and keep more loads in flight. With the 16-byte loads
    // that pays for one token only for rows of <= 512 (bf16 4096x512 -10%, 8192x1024 +4%), and for 2-6 tokens for
    // rows of >= 1024, where each row would otherwise read every y column again (bf16 1280x17408 at 6 tokens
    // 125 -> 79 us, 4352x5120 118 -> 88 us; rows of 512 and 8 tokens are mixed). With the 4-byte loads, where every
    // row still sums in its own order, short rows gain the most (bf16 8192x1024 -22%, 4096x512 -23%) and so do
    // very tall matrices (f16 62080x5120 -10%), rows of 2k-17k are within +-3%, and below ~8 blocks per SM it
    // loses.
    int rpb = 1;
    if (ncols_dst == 1 && !is_multi_token_id && is_volta && fusion.sib_x == nullptr && fusion.topk.ids == nullptr &&
            (load16 ? ncols <= 512 : (ncols <= 1024 || nrows >= 16384)) &&
            nrows % 4 == 0 && (nrows/4)*nchannels_dst*nsamples_or_ntokens >= 8*ggml_cuda_info().devices[device].nsm) {
        rpb = 4;
    }
    // Only while that leaves a block per SM: with fewer, each block walks the whole K for every column on its own,
    //   and one row per block spreads that over more SMs (f16 12x5120 at 6 tokens 14.5 -> 9.5 us cold, 24x5120
    //   14.7 -> 9.9). Each row sums in the same order either way, so a fused gate keeps one row per block.
    if (load16 && ncols_dst > 1 && ncols_dst <= 6 && !has_fusion && ncols >= 1024 && nrows % 4 == 0 &&
            (nrows/4)*nchannels_dst*nsamples_or_ntokens >= ggml_cuda_info().devices[device].nsm) {
        rpb = 4;
    }

    // the 16-byte path keeps one partial sum per warp, row and column
    const int nsums_red     = load16 ? int(ncols_dst*rpb*(block_size_best/warp_size)) : 0;
    const int nbytes_shared = std::max(warp_size, nsums_red)*(has_fusion ? 2 : 1)*sizeof(float);

    const dim3 block_nums(nrows/rpb, nchannels_dst, nsamples_or_ntokens);
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
        default: {
            GGML_ABORT("fatal error");
        } break;
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
            break;
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
        GGML_ASSERT(  ids || dst->ne[1] == 1 || (fusion->gdn_beta && ggml_cuda_mmvf_fuses_tokens(src0, src1, fusion->gate)));
        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
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
        if (fusion->gdn_beta) {
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
        if (fusion->sib && !fusion->sib_rows) {
            GGML_ASSERT(!ids && !fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->gdn_beta);
            GGML_ASSERT(src0->type == GGML_TYPE_F32 && fusion->sib->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->sib->ne[0] == ne00 && ggml_nrows(fusion->sib) == 1 && ggml_is_contiguous(fusion->sib));
            GGML_ASSERT(ggml_nrows(dst) == 1 && fusion->sib_dst->type == GGML_TYPE_F32 && ggml_nelements(fusion->sib_dst) == 1);
            fusion_local.sib_x   = fusion->sib->data;
            fusion_local.sib_dst = (float *) fusion->sib_dst->data;
        }
        if (fusion->sib && fusion->sib_rows) {
            // the rows of one or two sibling matrices after those of src0, see mul_mat_vec_f
            GGML_ASSERT(!ids && !fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->gdn_beta);
            GGML_ASSERT(ggml_nrows(src1) == 1 && ggml_nrows(dst) == 1 && ggml_nrows(src0) == ne01);
            int64_t row0 = ne01;
            const ggml_tensor * sibs[2] = { fusion->sib,     fusion->sib2     };
            ggml_tensor       * dsts[2] = { fusion->sib_dst, fusion->sib2_dst };
            for (int k = 0; k < 2 && sibs[k] != nullptr; ++k) {
                const ggml_tensor * sib = sibs[k];
                GGML_ASSERT(sib->type == src0->type && sib->ne[0] == ne00 && sib->nb[1] == src0->nb[1] && ggml_nrows(sib) == sib->ne[1]);
                GGML_ASSERT(dsts[k]->type == GGML_TYPE_F32 && ggml_is_contiguous(dsts[k]) && ggml_nelements(dsts[k]) == sib->ne[1]);
                if (k == 0) {
                    fusion_local.sib_x      = sib->data;
                    fusion_local.sib_dst    = (float *) dsts[k]->data;
                    fusion_local.sib_block0 = (uint32_t) row0;
                    fusion_local.sib_nrows  = (uint32_t) sib->ne[1];
                } else {
                    fusion_local.sib2_x      = sib->data;
                    fusion_local.sib2_dst    = (float *) dsts[k]->data;
                    fusion_local.sib2_block0 = (uint32_t) row0;
                    fusion_local.sib2_nrows  = (uint32_t) sib->ne[1];
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
            // a sibling row is one more block, sibling rows as many more
            const int64_t nrows = ne01 + (fusion_local.sib_block0 > 0 ? fusion_local.sib_nrows + fusion_local.sib2_nrows :
                                          fusion_local.sib_x != nullptr ? 1 : 0);
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, nrows, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0->data;
            const int64_t nrows = ne01 + fusion_local.sib_nrows + fusion_local.sib2_nrows;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, nrows, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
            const int64_t nrows = ne01 + fusion_local.sib_nrows + fusion_local.sib2_nrows;
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
                if (cc >= GGML_CUDA_CC_TURING) {
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
