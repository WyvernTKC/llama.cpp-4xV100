#include "common.cuh"
#include "unary.cuh"

// one channel of one token: silu(window . w + b), as ssm_conv_state_f32 and the epilogue of mul_mat_vec_q compute it
template <int d_conv>
static __device__ __forceinline__ float ggml_cuda_ssm_conv_silu(const float * win, const float * wv, const float b) {
    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        sumf += win[j] * wv[j];
    }
    sumf += b;
    return ggml_cuda_op_silu_single(sumf);
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

#define GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS 8

// the rollback slots of a convolution state, by the first column of the window they hold (0..n_t); nullptr where no
//   slot starts
struct ggml_cuda_ssm_conv_state_slots {
    float * dst[GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS + 1];
};

// silu(ssm_conv(concat(states[ids[0]], x^T))) of one sequence of at most GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS tokens,
//   with the windows of the concat stored to the slots: build_conv_state in one launch, reading the cache row itself.
//   With a gate the output is gate*ssm_conv(...) instead (silu is then that MUL), and with x_a/x_b x is (x_a*x_b)^T.
void ggml_cuda_op_ssm_conv_state(ggml_backend_cuda_context & ctx, const ggml_tensor * states, const ggml_tensor * ids,
                                 const ggml_tensor * x, const ggml_tensor * ssm_conv, ggml_tensor * silu,
                                 const ggml_cuda_ssm_conv_state_slots & slots, const ggml_tensor * gate = nullptr,
                                 const ggml_tensor * x_a = nullptr, const ggml_tensor * x_b = nullptr);

// the rollback slots of a longer batch, at the first column of the window each holds in the concat
struct ggml_cuda_ssm_conv_state_long_slots {
    int     n;
    int     start[GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS + 1];
    float * dst  [GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS + 1];
};

// the same for one sequence of more tokens, without the concat in memory; silu may sit on x only with its layout
void ggml_cuda_op_ssm_conv_state_long(ggml_backend_cuda_context & ctx, const ggml_tensor * states, const ggml_tensor * ids,
                                      const ggml_tensor * x, const ggml_tensor * ssm_conv, ggml_tensor * silu,
                                      const ggml_cuda_ssm_conv_state_long_slots & slots);
