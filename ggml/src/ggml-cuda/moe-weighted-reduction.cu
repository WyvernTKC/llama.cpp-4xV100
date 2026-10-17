#include "moe-weighted-reduction.cuh"
#include "unary.cuh"

#define MOE_WEIGHTED_REDUCTION_SMEM_EXPERTS 32

// blockIdx.x is the token and blockIdx.y a slice of its columns. A block reads the token's router weights (and expert
//   scales) before any of its threads writes, so with a single block per token dst may sit on them; each thread reads
//   the experts of its columns before it writes them, so dst may sit on the token's own expert rows. The products and
//   the sum are rounded like the MULs and the chain of ADDs apart. add_x (one token): dst = sum + sigmoid(*add_g)*add_x,
//   rounded like the MUL and the ADD after the sum. scale_ids: expert_scale is a table read at the token's expert ids.
static __global__ void moe_weighted_reduction_f32(const float *   experts,
                                                  const float *   expert_scale,
                                                  const float *   weights,
                                                  float *         dst,
                                                  const int64_t   n_embd,
                                                  const int       n_expert_used,
                                                  const float *   add_x,
                                                  const float *   add_g,
                                                  const int32_t * scale_ids,
                                                  const int64_t   ids_stride) {
    const int64_t  token     = blockIdx.x;
    const uint64_t first_row = (uint64_t) token * n_expert_used;

    __shared__ float s_weights[MOE_WEIGHTED_REDUCTION_SMEM_EXPERTS];
    __shared__ float s_scale[MOE_WEIGHTED_REDUCTION_SMEM_EXPERTS];
    if (threadIdx.x < n_expert_used) {
        s_weights[threadIdx.x] = weights[first_row + threadIdx.x];
        s_scale[threadIdx.x]   = scale_ids != nullptr ? expert_scale[scale_ids[token*ids_stride + threadIdx.x]] :
                                 expert_scale != nullptr ? expert_scale[first_row + threadIdx.x] : 1.0f;
    }
    __syncthreads();

    const float add_sig = add_x != nullptr ? ggml_cuda_op_sigmoid_single(*add_g) : 0.0f;

    for (int64_t col = (int64_t) blockIdx.y*blockDim.x + threadIdx.x; col < n_embd; col += (int64_t) gridDim.y*blockDim.x) {
        float sum = 0.0f;
#pragma unroll 8
        for (int expert = 0; expert < n_expert_used; ++expert) {
            float v = experts[(first_row + expert) * n_embd + col];
            if (expert_scale != nullptr) {
                v = __fmul_rn(v, s_scale[expert]);
            }
            v   = __fmul_rn(v, s_weights[expert]);
            sum = expert == 0 ? v : __fadd_rn(sum, v);
        }
        if (add_x != nullptr) {
            sum = __fadd_rn(sum, __fmul_rn(add_x[col], add_sig));
        }
        dst[token * n_embd + col] = sum;
    }
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const bool                  copy_weights,
                                         const float *               add_x,
                                         const ggml_tensor *         add_g,
                                         const ggml_tensor *         scale_ids) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();
    GGML_ASSERT(n_expert_used <= MOE_WEIGHTED_REDUCTION_SMEM_EXPERTS);
    GGML_ASSERT(add_x == nullptr || (n_tokens == 1 && add_g != nullptr && add_g->type == GGML_TYPE_F32 && ggml_nelements(add_g) == 1));
    GGML_ASSERT(scale_ids == nullptr || (expert_scale != nullptr && scale_ids->type == GGML_TYPE_I32 &&
                scale_ids->nb[0] == sizeof(int32_t) && scale_ids->ne[0] == n_expert_used && ggml_nrows(scale_ids) == n_tokens &&
                (n_tokens == 1 || !ggml_cuda_tensors_overlap(dst, scale_ids))));

    const float * weights_d = (const float *) weights->data;
    const float * scale_d   = expert_scale ? (const float *) expert_scale->data : nullptr;

    // dst sits on the weights of other tokens, which other blocks read: a copy of them
    ggml_cuda_pool_alloc<float> weights_copy(ctx.pool());
    ggml_cuda_pool_alloc<float> scale_copy(ctx.pool());
    if (copy_weights) {
        weights_copy.alloc(ggml_nelements(weights));
        CUDA_CHECK(cudaMemcpyAsync(weights_copy.get(), weights_d, ggml_nbytes(weights), cudaMemcpyDeviceToDevice, stream));
        weights_d = weights_copy.get();
        if (scale_d != nullptr && scale_ids == nullptr) {
            scale_copy.alloc(ggml_nelements(expert_scale));
            CUDA_CHECK(cudaMemcpyAsync(scale_copy.get(), scale_d, ggml_nbytes(expert_scale), cudaMemcpyDeviceToDevice, stream));
            scale_d = scale_copy.get();
        }
    }

    // a single block per token when dst sits on the weights it still has to read, else a block per 256 columns
    const bool one_block = !copy_weights && (ggml_cuda_tensors_overlap(dst, weights) ||
        (expert_scale != nullptr && ggml_cuda_tensors_overlap(dst, expert_scale)) ||
        (scale_ids != nullptr && ggml_cuda_tensors_overlap(dst, scale_ids)));
    GGML_ASSERT(!one_block || n_tokens == 1);
    const int  threads = one_block ? 1024 : 256;
    const dim3 blocks(n_tokens, one_block ? 1 : (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32<<<blocks, threads, 0, stream>>>(
        (const float *) experts->data, scale_d, weights_d, (float *) dst->data, n_embd, (int) n_expert_used,
        add_x, add_x != nullptr ? (const float *) add_g->data : nullptr,
        scale_ids != nullptr ? (const int32_t *) scale_ids->data : nullptr,
        scale_ids != nullptr ? (int64_t) (scale_ids->nb[1] / sizeof(int32_t)) : 0);
    CUDA_CHECK(cudaGetLastError());
}
