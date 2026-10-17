#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias, bool grouped>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = topk_moe_experts_per_thread(n_experts);

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    // the bias never aliases the outputs: read with the logits, not after the barrier
    [[maybe_unused]] float bias_v[has_bias ? experts_per_thread : 1];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
        if constexpr (has_bias) {
            bias_v[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? bias[expert] : 0.0f;
        }
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    topk_moe_select<n_experts, has_bias, grouped>(wt, bias_v, weights, ids, n_expert_used, clamp_val, scale_val, config);
}

template <int n_experts, bool has_bias, bool grouped>
static void launch_topk_moe_cuda_n(const ggml_cuda_kernel_launch_params & launch_params,
                                   const float *                          logits,
                                   float *                                weights,
                                   int32_t *                              ids,
                                   float *                                bias,
                                   const int                              n_rows,
                                   const int                              n_expert_used,
                                   const float                            clamp_val,
                                   const float                            scale_val,
                                   const topk_moe_config                  config) {
    ggml_cuda_kernel_launch(topk_moe_cuda<n_experts, has_bias, grouped>, launch_params,
        logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
}

template<bool has_bias, bool grouped>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

#define TOPK_MOE_CASE(n) \
        case n: \
            launch_topk_moe_cuda_n<n, has_bias, grouped>(launch_params, logits, weights, ids, bias, n_rows, n_expert_used, \
                clamp_val, scale_val, config); \
            break;

    if constexpr (grouped) {
        // see ggml_cuda_topk_moe_groups_supported
        switch (n_expert) {
            TOPK_MOE_CASE(64)
            TOPK_MOE_CASE(128)
            TOPK_MOE_CASE(256)
            TOPK_MOE_CASE(384)
            TOPK_MOE_CASE(512)
            default:
                GGML_ABORT("fatal error");
        }
    } else {
        switch (n_expert) {
            TOPK_MOE_CASE(1)
            TOPK_MOE_CASE(2)
            TOPK_MOE_CASE(4)
            TOPK_MOE_CASE(8)
            TOPK_MOE_CASE(16)
            TOPK_MOE_CASE(32)
            TOPK_MOE_CASE(64)
            TOPK_MOE_CASE(128)
            TOPK_MOE_CASE(240) // Nemotron 3.5 Bolt, per layer
            TOPK_MOE_CASE(256)
            TOPK_MOE_CASE(288) // StepFun 3.7
            TOPK_MOE_CASE(304) // Nemotron 3.5 Bolt
            TOPK_MOE_CASE(368) // Nemotron 3.5 Bolt
            TOPK_MOE_CASE(384) // DeepSeek-V4.1
            TOPK_MOE_CASE(432) // Nemotron 3.5 Bolt
            TOPK_MOE_CASE(496) // Nemotron 3.5 Bolt
            TOPK_MOE_CASE(512)
            TOPK_MOE_CASE(576)
            default:
                GGML_ABORT("fatal error");
        }
    }
#undef TOPK_MOE_CASE
}

bool ggml_cuda_topk_moe_groups_supported(const int n_expert, const int n_groups, const int n_group_used) {
    if (n_expert != 64 && n_expert != 128 && n_expert != 256 && n_expert != 384 && n_expert != 512) {
        return false;
    }
    return n_groups > 1 && n_groups <= WARP_SIZE && n_expert % (n_groups * WARP_SIZE) == 0 &&
           n_group_used >= 1 && n_group_used <= n_groups;
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;
    config.n_groups          = args.n_groups;
    config.n_group_used      = args.n_group_used;

    if (bias && args.n_groups > 1) {
        GGML_ASSERT(ggml_cuda_topk_moe_groups_supported(n_experts, args.n_groups, args.n_group_used));
        launch_topk_moe_cuda<true, true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used,
                                         clamp_val, scale_val, config);
    } else if (bias) {
        launch_topk_moe_cuda<true, false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used,
                                          clamp_val, scale_val, config);
    } else {
        GGML_ASSERT(args.n_groups <= 1);
        launch_topk_moe_cuda<false, false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used,
                                           clamp_val, scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models (StepFun 3.7, DeepSeek-V4.1, Nemotron 3.5 Bolt)
    const int n_expert = ids->nb[1] / ids->nb[0];
    switch (n_expert) {
        case 240: case 288: case 304: case 368: case 384: case 432: case 496: case 576:
            break;
        default:
            if ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
                return false;
            }
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}
