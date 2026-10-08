#include "ggml.h"
#include "mmf.cuh"
#include "mmid.cuh"
#include "mmvf.cuh"

static __forceinline__ int mmf_get_rows_per_block(const int cc) {
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMF_ROWS_PER_BLOCK_CDNA;
    } else {
        return MMF_ROWS_PER_BLOCK;
    }
}

// the K slices of a split matmul added in slice order: part is [ksplit][ncols][nrows]
static __global__ void mul_mat_f_ksplit_sum(const float * part, float * dst, const int nrows, const int ncols, const int ksplit,
        const int64_t stride_col_dst) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nrows*ncols) {
        return;
    }
    float sum = part[i];
    for (int s = 1; s < ksplit; ++s) {
        sum += part[int64_t(s)*nrows*ncols + i];
    }
    dst[int64_t(i / nrows)*stride_col_dst + i % nrows] = sum;
}

// On Volta a block of rows walks all of K at ~14 GB/s per SM: a matrix of a few row blocks (the K and V projections of
//   Muse-Glimmer at 16 tokens: 256 x 6656 per device, 8 blocks, 30 us) runs on that many SMs. Split K into slices that
//   run as the samples of one launch, so each block walks only its slice, and add them up in a second kernel. F16 only:
//   Volta has no BF16 path in mul_mat_f.
static int mul_mat_f_ksplit(const int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const int rows_per_block, const int warp_size) {
    if (ids || !volta_mma_available(cc) || src0->type != GGML_TYPE_F16 ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[1] % rows_per_block != 0) {
        return 1;
    }
    const int64_t nblocks = src0->ne[1] / rows_per_block;
    const int64_t k_T     = src0->ne[0] / 2;
    const int     nsm     = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    int ksplit = 1;
    while (nblocks*ksplit*2 <= nsm && k_T % (ksplit*2*warp_size) == 0 && k_T/(ksplit*2) >= 8*warp_size) {
        ksplit *= 2;
    }
    return ksplit;
}

void ggml_cuda_mul_mat_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);


    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t ids_s0 = ids ? ids->nb[0] / ggml_type_size(ids->type) : 0;
    const int64_t ids_s1 = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mmf_ids_data ids_info{};
    mmf_ids_data * ids_info_ptr = nullptr;
    ggml_cuda_pool_alloc<int32_t> ids_src_compact_dev;
    ggml_cuda_pool_alloc<int32_t> ids_dst_compact_dev;
    ggml_cuda_pool_alloc<int32_t> expert_bounds_dev;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_dst      = ids ? ne1 : ne2;

    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1 : s2;

    int64_t stride_channel_y         = ids ? s11  : s12;
    int64_t nchannels_y              = ids ? ne11 : ne12;

    //mul_mat_id: handle broadcast
    if (ids && nchannels_y == 1) {
        stride_channel_y = 0;
        nchannels_y      = ids->ne[0];
    }

    if (ids && ncols_dst > 16) {
        const int64_t n_expert_used = ids->ne[0];
        const int64_t n_experts     = ne02;
        const int64_t n_tokens      = ne12;
        const int64_t ne_get_rows   = n_tokens * n_expert_used;

        ids_src_compact_dev.alloc(ctx.pool(), ne_get_rows);
        ids_dst_compact_dev.alloc(ctx.pool(), ne_get_rows);
        expert_bounds_dev.alloc(ctx.pool(), n_experts + 1);

        const int si1  = static_cast<int>(ids_s1);
        const int sis1 = static_cast<int>(src1->nb[2] / src1->nb[1]);

        GGML_ASSERT(sis1 > 0);

        ggml_cuda_launch_mm_ids_helper(ids_d, ids_src_compact_dev.get(), ids_dst_compact_dev.get(), expert_bounds_dev.get(),
            static_cast<int>(n_experts), static_cast<int>(n_tokens), static_cast<int>(n_expert_used), static_cast<int>(ne11), si1, sis1, /*write_inverse =*/ false, ctx.stream());
        CUDA_CHECK(cudaGetLastError());

        ids_info.ids_src_compact   = ids_src_compact_dev.get();
        ids_info.ids_dst_compact   = ids_dst_compact_dev.get();
        ids_info.expert_bounds_dev = expert_bounds_dev.get();
        ids_info.n_experts         = static_cast<int>(n_experts);
        ids_info.sis1              = sis1;
        ids_info_ptr = &ids_info;
    }

    const int device    = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int rows_per_block = mmf_get_rows_per_block(cc);

    static const bool no_ksplit = getenv("GGML_CUDA_MMF_NO_KSPLIT") != nullptr;
    const int ksplit = no_ksplit ? 1 : mul_mat_f_ksplit(cc, src0, src1, ids, rows_per_block, ggml_cuda_info().devices[device].warp_size);
    if (ksplit > 1) {
        const int64_t k_slice = ne00 / ksplit;
        ggml_cuda_pool_alloc<float> part(ctx.pool(), ksplit*ne01*ncols_dst);
        // the slices are the samples: x and y advance by k_slice values, each slice into its own partial dst
        mul_mat_f_switch_rows_per_block<half2>(
            rows_per_block, (const half2 *) src0->data, src1_d, nullptr, part.get(), k_slice/2, ne01, ncols_dst, s01/2,
            stride_col_y/2, ne01, 0, 0, 1, 1, 1, s02/2, stride_channel_y, ne01*ncols_dst,
            ksplit, ksplit, k_slice/2, k_slice, ne01*ncols_dst, ctx.stream(), nullptr);
        const int n = (int) (ne01*ncols_dst);
        mul_mat_f_ksplit_sum<<<(n + 255)/256, 256, 0, ctx.stream()>>>(part.get(), dst_d, (int) ne01, (int) ncols_dst, ksplit, stride_col_dst);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            constexpr int vals_per_T = 1;
            mul_mat_f_switch_rows_per_block<float>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        case GGML_TYPE_F16: {
            const half2 * src0_d = (const half2 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_rows_per_block<half2>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat162 * src0_d = (const nv_bfloat162 *) src0->data;
            constexpr int vals_per_T = 2;
            mul_mat_f_switch_rows_per_block<nv_bfloat162>(
                rows_per_block, src0_d, src1_d, ids_d, dst_d, ne00/vals_per_T, ne01, ncols_dst, s01/vals_per_T, stride_col_y/vals_per_T, stride_col_dst,
                ids_s0, ids_s1, ne02, nchannels_y, nchannels_dst, s02/vals_per_T, stride_channel_y, stride_channel_dst,
                ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream(), ids_info_ptr);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne,
        const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {
    if (ggml_is_quantized(type)) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_ne[0] % (warp_size * (4/ts)) != 0) {
        return false;
    }

    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }
    if (src0_ne[1] % mmf_get_rows_per_block(cc) != 0) {
        return false;
    }

    if (GGML_CUDA_CC_IS_CDNA3(cc) && type == GGML_TYPE_BF16) {
        return false;
    }

    if (mul_mat_id) {
        if (src0_ne[1] <= 1024 && src1_ncols > 512) {
            return false;
        } else if(src0_ne[1] > 1024 && src1_ncols > 128) {
            return false;
        }
    } else {
        if (GGML_CUDA_CC_IS_RDNA3_0(cc) && src1_ncols > 8) {
            return false;
        } else if (GGML_CUDA_CC_IS_CDNA2(cc) && (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16)) {
            //TODO: truse CDNA2 as CDNA1, tune the perf when CDNA2 is available.
            return false;
        } else if (GGML_CUDA_CC_IS_CDNA1(cc) && (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16)) {
            return false;
        } else if (src1_ncols > 16) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            return ampere_mma_available(cc) || amd_mfma_available(cc);
        case GGML_TYPE_F16:
            return volta_mma_available(cc) || turing_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc);
        case GGML_TYPE_BF16:
            return ampere_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc);
        default:
            return false;
    }
}

bool ggml_cuda_should_fuse_mul_mat_f_glu(const ggml_tensor * ffn_up, const ggml_tensor * ffn_gate) {
    const ggml_tensor * src0 = ffn_up->src[0];
    const ggml_tensor * src1 = ffn_up->src[1];

    const int device    = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    if (!volta_mma_available(cc) || ffn_up->op != GGML_OP_MUL_MAT || ffn_gate->op != GGML_OP_MUL_MAT) {
        return false;
    }
    if (src0->type != GGML_TYPE_F16 || ffn_gate->src[0]->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 ||
            ffn_up->type != GGML_TYPE_F32 || ffn_gate->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t ncols_dst = src1->ne[1];
    if (ncols_dst < 2 || ncols_dst > 8) {
        return false;
    }

    // only where the two matrices would each go to mmf
    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ncols_dst)) {
        return false;
    }
    return ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ncols_dst, /*mul_mat_id =*/ false);
}

template <int cols_per_block>
static void mul_mat_f_glu_cuda(
        const half2 * x, const float * y, float * dst, const ggml_cuda_mm_fusion_args_device & fusion,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_dst, const int64_t stride_channel_x, const int64_t stride_channel_y,
        const int64_t stride_channel_dst, const int64_t nsamples_x, const int64_t nsamples_dst, const int64_t stride_sample_x,
        const int64_t stride_sample_y, const int64_t stride_sample_dst, cudaStream_t stream) {
    constexpr int rows_per_block = MMF_ROWS_PER_BLOCK;

    const int device    = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    // the same block size as mul_mat_f_cuda picks, so that each sum runs in the same order as without the fusion
    int64_t nwarps_best = 1;
    int64_t niter_best  = (ncols_x + warp_size*2 - 1) / (warp_size*2);
    for (int64_t nwarps = 2; nwarps <= mmf_get_max_block_size(cc)/warp_size; nwarps++) {
        const int64_t niter = (ncols_x + nwarps*warp_size*2 - 1) / (nwarps*warp_size*2);
        if (niter < niter_best) {
            niter_best  = niter;
            nwarps_best = nwarps;
        }
    }

    // per warp a tile of x (or gate) and one of y, then the partial sums of every warp
    const int padding               = mmf_get_padding(cc);
    const int nbytes_shared_iter    = nwarps_best*(rows_per_block + 8)*(warp_size + padding)*4;
    const int nbytes_shared_combine = GGML_PAD(cols_per_block, 8)*(nwarps_best*rows_per_block + padding)*4;
    const int nbytes_shared         = std::max(nbytes_shared_iter, nbytes_shared_combine);

    const dim3 block_nums(nrows_x/rows_per_block, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, nwarps_best, 1);

    const int channel_ratio = nchannels_dst / nchannels_x;
    const int sample_ratio  = nsamples_dst  / nsamples_x;

#define MMF_GLU_CASE(nwarps)                                                                                                      \
        case nwarps:                                                                                                            \
            mul_mat_f<half2, rows_per_block, cols_per_block, nwarps, false, true><<<block_nums, block_dims, nbytes_shared, stream>>> \
                (x, y, nullptr, dst, ncols_x, cols_per_block, nchannels_dst, stride_row, stride_col_y, stride_col_dst, 0, 0,  \
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,                                       \
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, fusion);                                  \
            break;

    switch (nwarps_best) {
        MMF_GLU_CASE(1)
        MMF_GLU_CASE(2)
        MMF_GLU_CASE(3)
        MMF_GLU_CASE(4)
        MMF_GLU_CASE(5)
        MMF_GLU_CASE(6)
        MMF_GLU_CASE(7)
        MMF_GLU_CASE(8)
        default:
            GGML_ABORT("fatal error");
    }
#undef MMF_GLU_CASE
}

void ggml_cuda_mul_mat_f_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_cuda_mm_fusion_args_host * fusion, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 && fusion->gate->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_stride(src0, fusion->gate));

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(ne13 == ne3);
    GGML_ASSERT(nb00 == ts_src0);
    GGML_ASSERT(nb10 == ts_src1);
    GGML_ASSERT(nb0  == ts_dst);

    const int64_t s01 = nb01 / ts_src0;
    const int64_t s02 = nb02 / ts_src0;
    const int64_t s03 = nb03 / ts_src0;
    const int64_t s11 = nb11 / ts_src1;
    const int64_t s12 = nb12 / ts_src1;
    const int64_t s13 = nb13 / ts_src1;
    const int64_t s1  = nb1  / ts_dst;
    const int64_t s2  = nb2  / ts_dst;
    const int64_t s3  = nb3  / ts_dst;

    ggml_cuda_mm_fusion_args_device fusion_dev{};
    fusion_dev.gate      = fusion->gate->data;
    fusion_dev.glu_op    = fusion->glu_op;
    fusion_dev.glu_limit = fusion->glu_limit;

    const half2 * src0_d = (const half2 *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    // values of half2 per row and column of y, as in ggml_cuda_mul_mat_f
    constexpr int vals_per_T = 2;

#define MMF_GLU_NCOLS(n)                                                                                                    \
        case n:                                                                                                           \
            mul_mat_f_glu_cuda<n>(src0_d, src1_d, dst_d, fusion_dev, ne00/vals_per_T, ne01, s01/vals_per_T,               \
                s11/vals_per_T, s1, ne02, ne2, s02/vals_per_T, s12, s2, ne03, ne3, s03/vals_per_T, s13, s3, ctx.stream()); \
            break;

    switch (ne11) {
        MMF_GLU_NCOLS(2)
        MMF_GLU_NCOLS(3)
        MMF_GLU_NCOLS(4)
        MMF_GLU_NCOLS(5)
        MMF_GLU_NCOLS(6)
        MMF_GLU_NCOLS(7)
        MMF_GLU_NCOLS(8)
        default:
            GGML_ABORT("fatal error");
    }
#undef MMF_GLU_NCOLS
}
