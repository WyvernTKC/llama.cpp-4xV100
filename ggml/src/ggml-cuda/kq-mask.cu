#include "kq-mask.cuh"
#include "convert.cuh"

// see ggml_kq_block_bias
static __global__ void kq_block_bias(const int32_t * __restrict__ blk_pos, const int32_t * __restrict__ q_pos, float * __restrict__ dst,
        const int64_t n_blk, const int64_t n_q, const int ratio) {
    const int64_t j = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= n_blk) {
        return;
    }
    const int64_t i = blockIdx.y;
    const int64_t s = blockIdx.z;

    const int32_t p0         = blk_pos[s*n_blk + j];
    const int32_t tail_start = (q_pos[s*n_q + i] + 1)/ratio*ratio;

    dst[(s*n_q + i)*n_blk + j] = p0 == -2 ? 1e9f : p0 < 0 ? -INFINITY : p0 >= tail_start ? 1e9f : 0.0f;
}

// see ggml_kq_mask: 0 where the cell is visible to the query, -inf elsewhere
template <typename T>
static __global__ void kq_mask(const int32_t * __restrict__ kv_pos, const int32_t * __restrict__ q_pos, T * __restrict__ dst,
        const int64_t n_kv, const int64_t n_q, const int swa_type, const int n_swa) {
    const int64_t j = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= n_kv) {
        return;
    }
    const int64_t i = blockIdx.y;
    const int64_t s = blockIdx.z;

    const int32_t p0 = kv_pos[s*n_kv + j];
    const int32_t p1 = q_pos[s*n_q + i];

    bool masked = p0 < 0 || p0 > p1;
    switch (swa_type) {
        case GGML_KQ_MASK_SWA_STANDARD:  masked = masked || p1 - p0 >= n_swa;            break;
        case GGML_KQ_MASK_SWA_CHUNKED:   masked = masked || p0 < (p1/n_swa)*n_swa;       break;
        case GGML_KQ_MASK_SWA_SYMMETRIC: masked = masked || p1 - p0 > n_swa/2;           break;
        default: break;
    }

    dst[(s*n_q + i)*n_kv + j] = ggml_cuda_cast<T>(masked ? -INFINITY : 0.0f);
}

void ggml_cuda_op_kq_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst));

    const int64_t n_kv = dst->ne[0];
    const int64_t n_q  = dst->ne[1];
    const int64_t n_s  = dst->ne[3];
    GGML_ASSERT(n_q <= 65535 && n_s <= 65535);

    const int swa_type = ggml_get_op_params_i32(dst, 0);
    const int n_swa    = ggml_get_op_params_i32(dst, 1);
    const int ratio    = ggml_get_op_params_i32(dst, 2);

    const dim3 block_dims(CUDA_KQ_MASK_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((n_kv + CUDA_KQ_MASK_BLOCK_SIZE - 1)/CUDA_KQ_MASK_BLOCK_SIZE, n_q, n_s);

    const int32_t * kv_pos = (const int32_t *) src0->data;
    const int32_t * q_pos  = (const int32_t *) src1->data;

    cudaStream_t stream = ctx.stream();
    if (ratio > 0) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        kq_block_bias<<<block_nums, block_dims, 0, stream>>>(kv_pos, q_pos, (float *) dst->data, n_kv, n_q, ratio);
        return;
    }
    if (dst->type == GGML_TYPE_F16) {
        kq_mask<<<block_nums, block_dims, 0, stream>>>(kv_pos, q_pos, (half *) dst->data, n_kv, n_q, swa_type, n_swa);
    } else {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        kq_mask<<<block_nums, block_dims, 0, stream>>>(kv_pos, q_pos, (float *) dst->data, n_kv, n_q, swa_type, n_swa);
    }
}

bool ggml_cuda_kq_mask_supported(const ggml_tensor * op) {
    return (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) &&
        op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32 &&
        ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op) &&
        op->ne[1] <= 65535 && op->ne[3] <= 65535;
}
