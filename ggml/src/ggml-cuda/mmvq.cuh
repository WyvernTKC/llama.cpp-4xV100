#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// whether the shared expert can run as one more channel of a routed MUL_MAT_ID launch
bool ggml_cuda_mmvq_shared_expert_supported(int cc);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// Whether the gate of a one-token GLU can go into the mmvq kernel of its up projection without a loss. For a
// MUL_MAT_ID nchannels_dst are the experts used by the token.
bool ggml_cuda_should_fuse_mmvq_gate(enum ggml_type type, int cc, int64_t nrows_x, int64_t nchannels_dst, int64_t nsamples_dst,
    bool mul_mat_id);

// whether the one-token short q8_0 kernel takes this MUL_MAT_ID with a sibling of one more channel, sib*sib_y
bool ggml_cuda_mul_mat_vec_q_short_sibling(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * sib, const ggml_tensor * sib_y);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// ggml_cuda_mul_mat_vec_q of the Walsh-Hadamard transform of each src1_rot consecutive values of src1
void ggml_cuda_mul_mat_vec_q_rot(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int src1_rot);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
