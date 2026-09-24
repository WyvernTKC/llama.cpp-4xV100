#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

#define GGML_CUDA_CPY_BATCH_MAX 8

// whether ggml_cuda_cpy copies src0 into src1 with the generic f32 scalar kernel
bool ggml_cuda_cpy_is_scalar_f32(const ggml_tensor * src0, const ggml_tensor * src1);

// n of those copies of one shape and stride in one launch, for runs of small copies such as the conv windows
//   a recurrent layer keeps for rollback (one per slot)
void ggml_cuda_cpy_scalar_f32_batch(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0, ggml_tensor * const * src1, int n);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
