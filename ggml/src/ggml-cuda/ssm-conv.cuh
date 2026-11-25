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
//   With residual it is (ssm_conv(...) + x)*res_scale (silu is then that SCALE).
// the input-dependent taps of a dynamic short conv (Inkling 2): taps[lag + head*s_head + token*s_token], cph channels per head
struct ggml_cuda_ssm_conv_dyn {
    const float * taps    = nullptr;
    int64_t       s_head  = 0;
    int64_t       s_token = 0;
    int           cph     = 1;
};

// a norm of each head of 128 channels of the residual output (Inkling 2's K and V), written to out instead of the output:
//   mode GGML_CUDA_SSM_CONV_EPI_L2_CLIP scales a head to an L2 norm of at most t as ggml_cuda_op_l2_clip does, f =
//   clamp(t/(sqrt(sum v^2)*scale + bias), lo, hi); GGML_CUDA_SSM_CONV_EPI_RMS_NORM is ggml_rms_norm with eps
#define GGML_CUDA_SSM_CONV_EPI_NONE     0
#define GGML_CUDA_SSM_CONV_EPI_L2_CLIP  1
#define GGML_CUDA_SSM_CONV_EPI_RMS_NORM 2
#define GGML_CUDA_SSM_CONV_EPI_ROW      128
//   With cache the rows go into the rows of a KV cache instead (llama_kv_cache::cpy_k/cpy_v's SET_ROWS), cache[idx[t]*
//   cache_row + c] in f16 or f32, and out is not written.
struct ggml_cuda_ssm_conv_epi {
    int     mode  = GGML_CUDA_SSM_CONV_EPI_NONE;
    float * out   = nullptr;
    float   eps   = 0.0f;
    float   scale = 1.0f;
    float   bias  = 0.0f;
    float   t     = 0.0f;
    float   lo    = 0.0f;
    float   hi    = 0.0f;
    void *       cache     = nullptr;
    int64_t      cache_row = 0;
    bool         cache_f16 = false;
    const void * idx       = nullptr;
    bool         idx_i64   = true;
};

void ggml_cuda_op_ssm_conv_state(ggml_backend_cuda_context & ctx, const ggml_tensor * states, const ggml_tensor * ids,
                                 const ggml_tensor * x, const ggml_tensor * ssm_conv, ggml_tensor * silu,
                                 const ggml_cuda_ssm_conv_state_slots & slots, const ggml_tensor * gate = nullptr,
                                 const ggml_tensor * x_a = nullptr, const ggml_tensor * x_b = nullptr,
                                 bool residual = false, float res_scale = 1.0f, const ggml_tensor * res_r = nullptr,
                                 const ggml_cuda_ssm_conv_dyn dyn = {}, const ggml_cuda_ssm_conv_epi epi = {});

// the rollback slots of a longer batch, at the first column of the window each holds in the concat
struct ggml_cuda_ssm_conv_state_long_slots {
    int     n;
    int     start[GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS + 1];
    float * dst  [GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS + 1];
};

// the same for one sequence of more tokens, without the concat in memory; silu may sit on x only with its layout
void ggml_cuda_op_ssm_conv_state_long(ggml_backend_cuda_context & ctx, const ggml_tensor * states, const ggml_tensor * ids,
                                      const ggml_tensor * x, const ggml_tensor * ssm_conv, ggml_tensor * silu,
                                      const ggml_cuda_ssm_conv_state_long_slots & slots,
                                      bool residual = false, float res_scale = 1.0f, const ggml_tensor * res_r = nullptr);

// The replay form of the chain (llm_build_delta_net_base::build_conv_state_replay, one sequence): the window is the
//   d_conv - 1 columns of the channels-first conv history of the cell that win_cols names (after the kept pending
//   tokens), x holds the new tokens channels-first, and the last n_dst columns of [window | x] go back to the history
//   row of the cell, dst_hist[col*hist_stride + c]. silu(ssm_conv(cont(transpose(concat)))) in one launch (two with
//   more than one block of tokens), the history read from the cache row itself; silu may sit on x with its layout.
struct ggml_cuda_ssm_conv_replay_args {
    const float   * states;      // the history of this convolution in row ids[0], column k at k*col_stride
    const int32_t * ids;
    int64_t         row_stride;
    int64_t         col_stride;
    const int32_t * win_cols;    // [d_conv - 1]
    float         * dst_hist;
    int64_t         hist_stride;
    int             n_dst;
};

void ggml_cuda_op_ssm_conv_replay(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_replay_args & a,
                                  const ggml_tensor * x, const ggml_tensor * ssm_conv, ggml_tensor * silu);
