#include "common.cuh"

bool ggml_cuda_page_stats_supported(const ggml_tensor * op);

void ggml_cuda_op_page_stats(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
