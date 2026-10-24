#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11);

// whether ggml_cuda_mul_mat_vec_f can fuse a gate over src1->ne[1] tokens: at one token always, at several only
//   on the 16-byte loads
bool ggml_cuda_mmvf_fuses_tokens(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * gate);

// whether ggml_cuda_mul_mat_vec_f can run sib (the shared expert's down projection, of one expert's layout) on its own
//   input sib_y as one more channel past the used experts of a one-token MUL_MAT_ID (fusion sib, sib_y and sib_out)
bool ggml_cuda_mul_mat_vec_f_moe_sibling(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    const ggml_tensor * sib, const ggml_tensor * sib_y);

// whether ggml_cuda_mul_mat_vec_f can run the GLU of the pair sib (up) / sib_gate (gate), of one expert's layout, on
//   the experts' input as one more channel past the used experts of a one-token MUL_MAT_ID with the gate gate0 (fusion
//   gate, sib, sib_gate and sib_dst; see ggml_cuda_try_defer_glu)
bool ggml_cuda_mul_mat_vec_f_moe_glu_sibling(const ggml_tensor * src0, const ggml_tensor * gate0, const ggml_tensor * src1,
    const ggml_tensor * ids, const ggml_tensor * sib, const ggml_tensor * sib_gate);

// whether ggml_cuda_mul_mat_vec_f can run the projections of a gated delta net layer at one token in one launch: the rows
//   of src0 (qkv), then those of sib (z) and gdn_x (alpha) with the gate (beta), all of the same type, K and input
//   (fusion sib, gdn_x, gate and the gated delta net epilogue)
bool ggml_cuda_mul_mat_vec_f_gdn_qkv(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * sib,
    const ggml_tensor * gdn_x, const ggml_tensor * gate);

// whether ggml_cuda_mul_mat_vec_f can compute its y, src1, as the gated output norm of a gated delta net layer,
//   rms_norm(x)*w*silu(z) over rows of 128, from x as the launch reads it (fusion norm_x, norm_z, norm_w and norm_eps):
//   one token of f16 rows, every block normalizing all of y, which saves the norm's launch
bool ggml_cuda_mul_mat_vec_f_norm_gated(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
    const ggml_tensor * x, const ggml_tensor * z, const ggml_tensor * w);
