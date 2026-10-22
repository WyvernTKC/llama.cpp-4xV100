#include "common.cuh"

#define CUDA_SOFT_MAX_BLOCK_SIZE 1024

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_soft_max_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// pooled[rep[p]] = sum_k softmax_k(gate_k + ape_k) * key_k over the cache rows members[p*kpool + k] (key | gate),
//   cache and pooled: views of one cache (F16 or F32) of the same row stride; ape F32 [n_embd, kpool]
void ggml_cuda_kpool_pool(ggml_backend_cuda_context & ctx, const ggml_tensor * cache, const ggml_tensor * members,
        const ggml_tensor * ape, const ggml_tensor * rep, const ggml_tensor * pooled, const int kpool);
