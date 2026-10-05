#include "common.cuh"

#define CUDA_GET_ROWS_BLOCK_SIZE 256
#define CUDA_GET_ROWS_BACK_BLOCK_SIZE 256

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream);

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_get_rows_mean_supported(const ggml_tensor * op);

void ggml_cuda_op_get_rows_mean(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GET_ROWS of a float matrix (two dimensions, i32 indices, f32 rows) and the SET_ROWS of those rows into another, or the
//   ADD of them to an f32 matrix of the same shape, in one launch each; the conversions are those of the ops apart
bool ggml_cuda_should_fuse_get_rows_set_rows(const ggml_tensor * get_rows, const ggml_tensor * set_rows);
bool ggml_cuda_should_fuse_get_rows_add(const ggml_tensor * get_rows, const ggml_tensor * add);
void ggml_cuda_op_get_rows_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * get_rows, ggml_tensor * set_rows);
void ggml_cuda_op_get_rows_add(ggml_backend_cuda_context & ctx, const ggml_tensor * get_rows, ggml_tensor * add);
