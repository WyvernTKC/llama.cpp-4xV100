#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

#define MMF_SMALL_K_MAX      16 // max. row length of src0 for ggml_cuda_mul_mat_small_k
#define MMF_SMALL_K_MAX_COLS  8 // max. src1 columns for ggml_cuda_mul_mat_small_k

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmvf(const ggml_tensor * src0, int cc, int warp_size, int64_t ne11);

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

// one token through f32 rows whose input is CONCAT(a, b, dim 0) of two vectors, the concat not computed
bool ggml_cuda_should_use_mul_mat_vec_f_concat(const ggml_tensor * w, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * dst);
void ggml_cuda_mul_mat_vec_f_concat(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * a, const ggml_tensor * b,
        ggml_tensor * dst, bool norm_a = false, float eps = 0.0f);

// f32 matrix products with rows of at most MMF_SMALL_K_MAX: one thread per row of src0, which can be a transposed view
bool ggml_cuda_should_use_mul_mat_small_k(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_mul_mat_small_k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Inkling 2's loop gate of one token in the launch of its gain and forget products (the short rows of
//   mul_mat_vec_f_short_rows): out = (c - tanh(wf*y)*c) + tanh(wg*y)*m per row, as the tanh gate after them computes it.
//   Whether it takes them, then the launch.
bool ggml_cuda_mul_mat_vec_f_gate_rows_supported(const ggml_backend_cuda_context & ctx, const ggml_tensor * wg, const ggml_tensor * wf,
        const ggml_tensor * y, const ggml_tensor * g, const ggml_tensor * f);
void ggml_cuda_mul_mat_vec_f_gate_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * wg, const ggml_tensor * wf,
        const ggml_tensor * y, const ggml_tensor * c, const ggml_tensor * m, ggml_tensor * out);
