#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

// rms_norm (with mul) or rope may be null; rot is a Hadamard MUL_MAT of their result (or of its own src[1] when both are
//   null), set_rows the store of the result into a cache; rows gives the row shape when there is no rope
void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows,
    ggml_tensor * rot = nullptr, const ggml_tensor * rows = nullptr);

// whether the fused kernel above can store into a SET_ROWS destination of this type
bool ggml_cuda_rms_norm_mul_rope_fused_dst_supported(ggml_type type);

// two or three fusions of ggml_cuda_op_rms_norm_mul_rope_fused of one token (the Q, K and V of an attention layer),
//   nodes[s] = { rms_norm, mul, rope, set_rows }, in one launch; false (nothing launched) where that does not apply
bool ggml_cuda_op_rms_norm_mul_rope_multi(ggml_backend_cuda_context & ctx, ggml_tensor * nodes[][4], int n);

void ggml_cuda_op_get_rows_mean_norm_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * get_rows_mean, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope);
