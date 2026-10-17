#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles.
#define TOPK_MOE_ROWS_PER_BLOCK 8

// experts per lane: n_experts in whole warps, the lanes past the last expert hold -INFINITY
static constexpr __host__ __device__ int topk_moe_experts_per_thread(const int n_experts) {
    return n_experts > WARP_SIZE ? (n_experts + WARP_SIZE - 1) / WARP_SIZE : 1;
}

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
    int  n_groups{1};     // grouped routing (DeepSeek-V3) when > 1
    int  n_group_used{1};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);

// whether the grouped kernel is built for n_expert experts in n_groups groups
bool ggml_cuda_topk_moe_groups_supported(int n_expert, int n_groups, int n_group_used);

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
    int  n_groups;     // grouped routing: groups of n_experts / n_groups experts, a multiple of WARP_SIZE
    int  n_group_used; //   groups kept
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

// Grouped routing (DeepSeek-V3): keeps the n_group_used groups with the largest sum of their top 2 selection values
//   (lowest group first on ties) and masks the experts of the others. A group is spg whole slots of sel.
template <int experts_per_thread>
__device__ __forceinline__ void topk_moe_mask_groups(float (&sel)[experts_per_thread], const int n_groups,
                                                     const int n_group_used, const int lane) {
    const int spg = experts_per_thread / n_groups;

    // top 2 of every slot over the warp; the first round has no second value on either side
    float s1[experts_per_thread];
    float s2[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const float o = __shfl_xor_sync(0xFFFFFFFF, sel[i], WARP_SIZE / 2, WARP_SIZE);
        s1[i]         = fmaxf(sel[i], o);
        s2[i]         = fminf(sel[i], o);
    }
#pragma unroll
    for (int offset = WARP_SIZE / 4; offset > 0; offset >>= 1) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            const float o1 = __shfl_xor_sync(0xFFFFFFFF, s1[i], offset, WARP_SIZE);
            const float o2 = __shfl_xor_sync(0xFFFFFFFF, s2[i], offset, WARP_SIZE);
            s2[i]          = fmaxf(fminf(s1[i], o1), fmaxf(s2[i], o2));
            s1[i]          = fmaxf(s1[i], o1);
        }
    }

    // lane g holds the score of group g
    float g1 = -INFINITY;
    float g2 = -INFINITY;
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const bool  mine = i >= lane * spg && i < (lane + 1) * spg;
        const float n2   = fmaxf(fminf(g1, s1[i]), fmaxf(g2, s2[i]));
        const float n1   = fmaxf(g1, s1[i]);
        g1               = mine ? n1 : g1;
        g2               = mine ? n2 : g2;
    }
    const float score = g1 + g2;

    int rank = 0;
    for (int h = 0; h < n_groups; h++) {
        const float sh = __shfl_sync(0xFFFFFFFF, score, h, WARP_SIZE);
        rank += sh > score || (sh == score && h < lane);
    }
    const unsigned keep = __ballot_sync(0xFFFFFFFF, lane < n_groups && rank < n_group_used);

    unsigned slot_keep = 0;
    for (int g = 0; g < n_groups; g++) {
        slot_keep |= (keep >> g) & 1 ? ((1u << spg) - 1) << (g * spg) : 0u;
    }
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        sel[i] = (slot_keep >> i) & 1 ? sel[i] : -INFINITY;
    }
}

// The expert selection of one row by one warp from its logits wt (and bias_v), as topk_moe_cuda does after it read them:
//   the gating, the top n_expert_used experts into ids and their weights
template <int n_experts, bool has_bias, bool grouped>
static __device__ __forceinline__ void topk_moe_select(
        float (&wt)[topk_moe_experts_per_thread(n_experts)],
        const float (&bias_v)[has_bias ? topk_moe_experts_per_thread(n_experts) : 1],
        float * weights, int32_t * ids, const int n_expert_used, const float clamp_val, const float scale_val,
        const topk_moe_config config) {
    constexpr int experts_per_thread = topk_moe_experts_per_thread(n_experts);

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias_v[i / WARP_SIZE] : -INFINITY;
        }
        if constexpr (grouped) {
            topk_moe_mask_groups<experts_per_thread>(selection_wt, config.n_groups, config.n_group_used, threadIdx.x);
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            const bool owner = (max_expert & (WARP_SIZE - 1)) == threadIdx.x;
#pragma unroll
            for (int i = 0; i < experts_per_thread; i++) {
                selection_wt[i] = owner && i == max_expert / WARP_SIZE ? -INFINITY : selection_wt[i];
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            const bool owner = (max_expert & (WARP_SIZE - 1)) == threadIdx.x;
#pragma unroll
            for (int i = 0; i < experts_per_thread; i++) {
                wt[i] = owner && i == max_expert / WARP_SIZE ? -INFINITY : wt[i];
            }
        }

        // every slot is written through a select: indexing wt, selection_wt or output_weights with max_expert / WARP_SIZE
        //   or k / WARP_SIZE, or a guarded write to the one slot (which the compiler turns back into that), puts the arrays
        //   in local memory, read again in every round
        const bool k_owner = (k & (WARP_SIZE - 1)) == threadIdx.x;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] = k_owner && i == k / WARP_SIZE ? max_val : output_weights[i];
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}
