#include "common.cuh"

#define CUDA_KQ_MASK_BLOCK_SIZE 256

void ggml_cuda_op_kq_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_kq_mask_supported(const ggml_tensor * op);
