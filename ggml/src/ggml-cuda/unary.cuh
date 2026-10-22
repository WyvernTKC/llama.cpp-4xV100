#pragma once
#include "common.cuh"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// softplus(x + bias) * scale with bias and scale one row each
bool ggml_cuda_should_fuse_add_softplus_mul(const ggml_tensor * add, const ggml_tensor * unary, const ggml_tensor * mul);
void ggml_cuda_op_add_softplus_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * unary, ggml_tensor * mul);

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// a GLU of the gated ops above with an f32 output that also writes the output as f16 to dst_f16
bool ggml_cuda_op_glu_f16_copy_supported(const ggml_tensor * dst);

void ggml_cuda_op_glu_f16_copy(ggml_backend_cuda_context & ctx, ggml_tensor * dst, half * dst_f16);

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);

// CONT of a strided view, unary, mul: the gated kernel reads the view (the attention output gate of qwen35)
bool ggml_cuda_should_fuse_cont_unary_mul(const ggml_tensor * cont, const ggml_tensor * unary, const ggml_tensor * mul);
void ggml_cuda_op_cont_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * cont, ggml_tensor * unary, ggml_tensor * mul);

// SCALE -> UNARY(SILU) of contiguous f32 as one launch (qwen4exp's hyper-connection down projection), with the q8_1 copy
//   of a one-row output for the MMVQ matmul that reads it, see ggml_cuda_q8_1_request
bool ggml_cuda_should_fuse_scale_unary(const ggml_tensor * scale, const ggml_tensor * unary);
void ggml_cuda_op_scale_unary(ggml_backend_cuda_context & ctx, ggml_tensor * scale, ggml_tensor * unary);

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);

// scale(sigmoid(x * m + a)) with m and a broadcast along the rows, the DeepSeek V4 hyper-connection coefficients
void ggml_cuda_op_mul_add_sigmoid_scale(ggml_backend_cuda_context & ctx, ggml_tensor * mul_node, ggml_tensor * add_node,
        ggml_tensor * scale_node);

// base + x * sigmoid(g) with one g per row: the shared expert gate of qwen35moe / qwen4exp for more than one token
bool ggml_cuda_should_fuse_sigmoid_mul_add(const ggml_tensor * sigmoid, const ggml_tensor * mul, const ggml_tensor * add);
void ggml_cuda_op_sigmoid_mul_add(ggml_backend_cuda_context & ctx, ggml_tensor * sigmoid, ggml_tensor * mul, ggml_tensor * add);

// scale(sigmoid(x * m)) with m broadcast as by the MUL, the forget gate of bailingmoe3 (m per head)
// pre: a SCALE between the MUL and the SIGMOID, or nullptr
bool ggml_cuda_should_fuse_mul_sigmoid_scale(const ggml_tensor * mul, const ggml_tensor * pre, const ggml_tensor * sigmoid,
        const ggml_tensor * scale);
void ggml_cuda_op_mul_sigmoid_scale(ggml_backend_cuda_context & ctx, ggml_tensor * mul, const ggml_tensor * pre,
        ggml_tensor * scale);

// the op kernels use these too, so a fused kernel gets the values the op's own kernel would
__device__ __forceinline__ float ggml_cuda_op_sigmoid_single(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float ggml_cuda_op_softplus_single(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

__device__ __forceinline__ float ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float ggml_cuda_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_clamp_single(float gate, float up, float limit) {
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);

    return ggml_cuda_op_silu_single(gate) * up;
}
