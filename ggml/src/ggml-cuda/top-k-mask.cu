#include "top-k-mask.cuh"
#include "convert.cuh"

#define CUDA_TOP_K_MASK_BLOCK_SIZE 512
#define CUDA_TOP_K_MASK_FILL_CELLS 4096

// one block per row: fill the row with -inf, then copy the mask cells of the live picks and the extra cells
template <typename T>
static __global__ void top_k_mask_row_kernel(
        const T * mask, const int32_t * top_k, const float * score, const int32_t * blocks, const int32_t * extra, T * dst,
        const int64_t n_kv, const int64_t ne1, const int64_t ne2,
        const int64_t m1, const int64_t m2, const int64_t m3,
        const int64_t n_top, const int64_t n_pick, const int64_t n_blk, const int64_t n_extra,
        const int64_t sm1, const int64_t sm2, const int64_t sm3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3,
        const int64_t sk0, const int64_t sk1, const int64_t sk2, const int64_t sk3,
        const int64_t ss0, const int64_t ss1, const int64_t ss2, const int64_t ss3,
        const int64_t sb0, const int64_t sb1,
        const int64_t se0, const int64_t se1, const int64_t se2, const int64_t se3,
        const bool vec) {
    ggml_cuda_pdl_lc();

    const int64_t ir = blockIdx.x;
    const int64_t i1 = ir % ne1;
    const int64_t i2 = (ir / ne1) % ne2;
    const int64_t i3 = ir / (ne1*ne2);

    const T       * mask_row  = mask  + (i1 % m1)*sm1 + (i2 % m2)*sm2 + (i3 % m3)*sm3;
    T             * dst_row   = dst   + i1*sd1 + i2*sd2 + i3*sd3;
    const int32_t * top_k_row = top_k + i1*sk1 + i2*sk2 + i3*sk3;

    ggml_cuda_pdl_sync();

    const T ninf = ggml_cuda_cast<T>(-INFINITY);
    if (vec) {
        constexpr int nv = sizeof(int4)/sizeof(T);
        T fill[nv];
#pragma unroll
        for (int j = 0; j < nv; ++j) {
            fill[j] = ninf;
        }
        const int4 fill4 = *(const int4 *) fill;
        for (int64_t i = threadIdx.x; i < n_kv/nv; i += blockDim.x) {
            ((int4 *) dst_row)[i] = fill4;
        }
    } else {
        for (int64_t i = threadIdx.x; i < n_kv; i += blockDim.x) {
            dst_row[i] = ninf;
        }
    }

    __syncthreads();

    // one thread per member cell of a pick (a pick is its own cell without blocks, n_blk 1) or extra cell; cells kept
    //   twice get the same value
    const int64_t n_pc = n_top*n_blk;
    for (int64_t i = threadIdx.x; i < n_pc + n_extra; i += blockDim.x) {
        if (i >= n_pc) {
            const int32_t c = extra[(i - n_pc)*se0 + i1*se1 + i2*se2 + i3*se3];
            if (c >= 0 && c < n_kv) {
                dst_row[c] = mask_row[c];
            }
            continue;
        }

        const int64_t j = blocks ? i / n_blk : i;
        const int32_t p = top_k_row[j*sk0];
        if (p < 0 || p >= n_pick) {
            continue;
        }
        if (score && !(score[p*ss0 + i1*ss1 + i2*ss2 + i3*ss3] > -INFINITY)) {
            continue;
        }
        const int32_t c = blocks ? blocks[(i - j*n_blk)*sb0 + p*sb1] : p;
        if (c >= 0 && c < n_kv) {
            dst_row[c] = mask_row[c];
        }
    }
}

// pass 1: fill every row with -inf, many blocks per row
template <typename T>
static __global__ void top_k_mask_fill_kernel(T * dst, const int64_t n_kv, const int64_t ne1, const int64_t ne2,
        const int64_t sd1, const int64_t sd2, const int64_t sd3, const bool vec) {
    ggml_cuda_pdl_lc();

    const int64_t ir = blockIdx.x;
    const int64_t i1 = ir % ne1;
    const int64_t i2 = (ir / ne1) % ne2;
    const int64_t i3 = ir / (ne1*ne2);

    T * dst_row = dst + i1*sd1 + i2*sd2 + i3*sd3;
    const int64_t c0 = (int64_t) blockIdx.y*CUDA_TOP_K_MASK_FILL_CELLS;
    const int64_t c1 = min(n_kv, c0 + CUDA_TOP_K_MASK_FILL_CELLS);

    ggml_cuda_pdl_sync();

    const T ninf = ggml_cuda_cast<T>(-INFINITY);
    if (vec) {
        constexpr int nv = sizeof(int4)/sizeof(T);
        T fill[nv];
#pragma unroll
        for (int j = 0; j < nv; ++j) {
            fill[j] = ninf;
        }
        const int4 fill4 = *(const int4 *) fill;
        for (int64_t i = c0/nv + threadIdx.x; i < c1/nv; i += blockDim.x) {
            ((int4 *) dst_row)[i] = fill4;
        }
    } else {
        for (int64_t i = c0 + threadIdx.x; i < c1; i += blockDim.x) {
            dst_row[i] = ninf;
        }
    }
}

// pass 2: copy the mask cells of the live picks and the extra cells; a group of tpi threads per pick writes its block's cells
template <typename T>
static __global__ void top_k_mask_scatter_kernel(
        const T * mask, const int32_t * top_k, const float * score, const int32_t * blocks, const int32_t * extra, T * dst,
        const int64_t n_kv, const int64_t ne1, const int64_t ne2,
        const int64_t m1, const int64_t m2, const int64_t m3,
        const int64_t n_top, const int64_t n_pick, const int64_t n_blk, const int64_t n_extra,
        const int64_t sm1, const int64_t sm2, const int64_t sm3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3,
        const int64_t sk0, const int64_t sk1, const int64_t sk2, const int64_t sk3,
        const int64_t ss0, const int64_t ss1, const int64_t ss2, const int64_t ss3,
        const int64_t sb0, const int64_t sb1,
        const int64_t se0, const int64_t se1, const int64_t se2, const int64_t se3, const int tpi) {
    ggml_cuda_pdl_lc();

    const int64_t ir = blockIdx.x;
    const int64_t i1 = ir % ne1;
    const int64_t i2 = (ir / ne1) % ne2;
    const int64_t i3 = ir / (ne1*ne2);

    const T       * mask_row  = mask  + (i1 % m1)*sm1 + (i2 % m2)*sm2 + (i3 % m3)*sm3;
    T             * dst_row   = dst   + i1*sd1 + i2*sd2 + i3*sd3;
    const int32_t * top_k_row = top_k + i1*sk1 + i2*sk2 + i3*sk3;

    const int64_t i = (int64_t) blockIdx.y*(blockDim.x/tpi) + threadIdx.x/tpi;
    const int     t = threadIdx.x % tpi;

    ggml_cuda_pdl_sync();

    if (i >= n_top + n_extra) {
        return;
    }
    if (i >= n_top) {
        if (t == 0) {
            const int32_t c = extra[(i - n_top)*se0 + i1*se1 + i2*se2 + i3*se3];
            if (c >= 0 && c < n_kv) {
                dst_row[c] = mask_row[c];
            }
        }
        return;
    }

    const int32_t p = top_k_row[i*sk0];
    if (p < 0 || p >= n_pick) {
        return;
    }
    if (score && !(score[p*ss0 + i1*ss1 + i2*ss2 + i3*ss3] > -INFINITY)) {
        return;
    }
    if (!blocks) {
        if (t == 0 && p < n_kv) {
            dst_row[p] = mask_row[p];
        }
        return;
    }
    for (int64_t k = t; k < n_blk; k += tpi) {
        const int32_t c = blocks[k*sb0 + p*sb1];
        if (c >= 0 && c < n_kv) {
            dst_row[c] = mask_row[c];
        }
    }
}

template <typename T>
static void top_k_mask_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mask   = dst->src[0];
    const ggml_tensor * top_k  = dst->src[1];
    const ggml_tensor * score  = dst->src[2];
    const ggml_tensor * blocks = dst->src[3];
    const ggml_tensor * extra  = dst->src[4];

    const int64_t n_kv = mask->ne[0];

    GGML_ASSERT(dst->nb[0] == sizeof(T) && mask->nb[0] == sizeof(T));
    GGML_ASSERT(dst->nb[1] % sizeof(T) == 0 && mask->nb[1] % sizeof(T) == 0);

    // 16-byte stores need every dst row aligned
    const bool vec = (n_kv*sizeof(T)) % sizeof(int4) == 0 &&
        dst->nb[1] % sizeof(int4) == 0 && dst->nb[2] % sizeof(int4) == 0 && dst->nb[3] % sizeof(int4) == 0 &&
        (uintptr_t) dst->data % sizeof(int4) == 0;

    const int64_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    const int64_t n_top   = top_k->ne[0];
    const int64_t n_extra = extra ? extra->ne[0] : 0;
    const int64_t n_rows  = ggml_nrows(dst);

    const size_t si = sizeof(int32_t);
    const size_t sf = sizeof(float);

    // enough rows to fill the GPU with one block each (prefill): keep the single fused pass
    if (n_rows >= ggml_cuda_info().devices[ctx.device].nsm) {
        const ggml_cuda_kernel_launch_params row_params(dim3(n_rows, 1, 1), dim3(CUDA_TOP_K_MASK_BLOCK_SIZE, 1, 1), 0, ctx.stream());
        ggml_cuda_kernel_launch(top_k_mask_row_kernel<T>, row_params,
                (const T *) mask->data, (const int32_t *) top_k->data,
                score  ? (const float   *) score->data  : nullptr,
                blocks ? (const int32_t *) blocks->data : nullptr,
                extra  ? (const int32_t *) extra->data  : nullptr,
                (T *) dst->data,
                n_kv, dst->ne[1], dst->ne[2],
                mask->ne[1], mask->ne[2], mask->ne[3],
                n_top, n_pick, blocks ? blocks->ne[0] : 1, n_extra,
                mask->nb[1]/sizeof(T), mask->nb[2]/sizeof(T), mask->nb[3]/sizeof(T),
                dst->nb[1]/sizeof(T),  dst->nb[2]/sizeof(T),  dst->nb[3]/sizeof(T),
                top_k->nb[0]/si, top_k->nb[1]/si, top_k->nb[2]/si, top_k->nb[3]/si,
                score  ? score->nb[0]/sf  : 0, score ? score->nb[1]/sf : 0, score ? score->nb[2]/sf : 0, score ? score->nb[3]/sf : 0,
                blocks ? blocks->nb[0]/si : 0, blocks ? blocks->nb[1]/si : 0,
                extra  ? extra->nb[0]/si  : 0, extra ? extra->nb[1]/si : 0, extra ? extra->nb[2]/si : 0, extra ? extra->nb[3]/si : 0,
                vec);
        return;
    }

    // few rows (decode): fill and scatter with many blocks per row
    const dim3 fill_grid(n_rows, (n_kv + CUDA_TOP_K_MASK_FILL_CELLS - 1)/CUDA_TOP_K_MASK_FILL_CELLS, 1);
    const ggml_cuda_kernel_launch_params fill_params(fill_grid, dim3(256, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(top_k_mask_fill_kernel<T>, fill_params, (T *) dst->data, n_kv, dst->ne[1], dst->ne[2],
            dst->nb[1]/sizeof(T), dst->nb[2]/sizeof(T), dst->nb[3]/sizeof(T), vec);

    // threads per pick: one per block cell up to 64, so picks of single cells or small blocks do not idle most threads
    int tpi = 1;
    while (blocks && tpi < 64 && tpi < blocks->ne[0]) {
        tpi *= 2;
    }
    const int64_t items_per_block = CUDA_TOP_K_MASK_BLOCK_SIZE/tpi;
    const dim3 scatter_grid(n_rows, (n_top + n_extra + items_per_block - 1)/items_per_block, 1);
    const ggml_cuda_kernel_launch_params scatter_params(scatter_grid, dim3(CUDA_TOP_K_MASK_BLOCK_SIZE, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(top_k_mask_scatter_kernel<T>, scatter_params,
            (const T *) mask->data, (const int32_t *) top_k->data,
            score  ? (const float   *) score->data  : nullptr,
            blocks ? (const int32_t *) blocks->data : nullptr,
            extra  ? (const int32_t *) extra->data  : nullptr,
            (T *) dst->data,
            n_kv, dst->ne[1], dst->ne[2],
            mask->ne[1], mask->ne[2], mask->ne[3],
            n_top, n_pick, blocks ? blocks->ne[0] : 1, n_extra,
            mask->nb[1]/sizeof(T), mask->nb[2]/sizeof(T), mask->nb[3]/sizeof(T),
            dst->nb[1]/sizeof(T),  dst->nb[2]/sizeof(T),  dst->nb[3]/sizeof(T),
            top_k->nb[0]/si, top_k->nb[1]/si, top_k->nb[2]/si, top_k->nb[3]/si,
            score  ? score->nb[0]/sf  : 0, score ? score->nb[1]/sf : 0, score ? score->nb[2]/sf : 0, score ? score->nb[3]/sf : 0,
            blocks ? blocks->nb[0]/si : 0, blocks ? blocks->nb[1]/si : 0,
            extra  ? extra->nb[0]/si  : 0, extra ? extra->nb[1]/si : 0, extra ? extra->nb[2]/si : 0, extra ? extra->nb[3]/si : 0, tpi);
}

void ggml_cuda_op_top_k_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->type) {
        case GGML_TYPE_F16:
            top_k_mask_cuda<half>(ctx, dst);
            break;
        case GGML_TYPE_F32:
            top_k_mask_cuda<float>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
