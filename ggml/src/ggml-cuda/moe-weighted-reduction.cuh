#include "common.cuh"

// copy_weights: dst overlaps the weights or expert scales of a ubatch of several tokens, so the kernel reads a copy
// add_x (n_embd floats, one token): dst = sum + sigmoid(add_g)*add_x
void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         bool                        copy_weights,
                                         const float *               add_x = nullptr,
                                         const ggml_tensor *         add_g = nullptr);
