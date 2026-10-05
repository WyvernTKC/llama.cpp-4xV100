#include "common.cuh"

// The largest pooled window ggml_cuda_dsv4_compress takes (entries per block of the softmax).
#define GGML_CUDA_DSV4_COMPRESS_MAX 256

// The DeepSeek-V4 compressor of a folded chain (ggml_cuda_dsv4_compress_fold): per block b and value d, the softmax over
//   the window's scores weights its values, summed. The window of an overlap compressor is the first half of the rows
//   of the previous r entries followed by the second half of the rows of the next r. Each step is computed as SOFT_MAX,
//   MUL and SUM_ROWS compute it (the same block reductions), so dst, [1, hd, nb], is bitwise theirs.
void ggml_cuda_dsv4_compress(ggml_backend_cuda_context & ctx, const ggml_cuda_dsv4_compress_fold & f, ggml_tensor * dst);
