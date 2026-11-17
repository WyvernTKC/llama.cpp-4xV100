#include "common.cuh"

bool ggml_cuda_page_top_k_supported(const ggml_tensor * op);

void ggml_cuda_op_page_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
