#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

#define GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS 8

// the rollback slots of a convolution state, by the first column of the window they hold (0..n_t); nullptr where no
//   slot starts
struct ggml_cuda_ssm_conv_state_slots {
    float * dst[GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS + 1];
};

// silu(ssm_conv(concat(states[ids[0]], x^T))) of one sequence of at most GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS tokens,
//   with the windows of the concat stored to the slots: build_conv_state in one launch, reading the cache row itself
void ggml_cuda_op_ssm_conv_state(ggml_backend_cuda_context & ctx, const ggml_tensor * states, const ggml_tensor * ids,
                                 const ggml_tensor * x, const ggml_tensor * ssm_conv, ggml_tensor * silu,
                                 const ggml_cuda_ssm_conv_state_slots & slots);
