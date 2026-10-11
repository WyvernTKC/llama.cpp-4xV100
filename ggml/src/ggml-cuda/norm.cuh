#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// with scale_tensor the mul reads a SCALE of the norm, rms_norm(x) * s * w
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                                 const ggml_tensor * scale_tensor = nullptr);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

// a residual add followed by rms_norm and the norm's mul: the add is kept as an output
bool ggml_cuda_should_fuse_rms_norm_pre_add(const ggml_tensor * add, const ggml_tensor * rms_norm, const ggml_tensor * mul);
void ggml_cuda_op_rms_norm_pre_add(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul);

// the same with the all-reduce of t, one of the add's inputs, folded in: t's sum is polled from the LL lines of ar
bool ggml_cuda_rms_norm_pre_add_ar_supported(const ggml_tensor * add, const ggml_tensor * rms_norm, const ggml_tensor * mul,
                                             const ggml_tensor * t);
void ggml_cuda_op_rms_norm_pre_add_ar(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul,
                                      const ggml_cuda_ar_fold_args & ar, ggml_tensor * t);

// the same with the sigmoid-gated shared expert of a MoE layer added first:
//   add = (moe + shexp * sigmoid(g)) + residual, mul = rms_norm(add) * w
bool ggml_cuda_should_fuse_rms_norm_pre_add_gated(const ggml_tensor * sigmoid, const ggml_tensor * mul_gate, const ggml_tensor * add_shexp,
                                                  const ggml_tensor * add, const ggml_tensor * rms_norm, const ggml_tensor * mul);
void ggml_cuda_op_rms_norm_pre_add_gated(ggml_backend_cuda_context & ctx, ggml_tensor * sigmoid, ggml_tensor * mul_gate,
                                         ggml_tensor * add_shexp, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul);

// y = rms_norm(x) * w1 + r (* s when scale is set) and mul2 = rms_norm(y) * w2: gemma's post-norm, residual add and the
//   next pre-norm of one row; y is the output of add or of scale
bool ggml_cuda_should_fuse_rms_norm_add_rms_norm(const ggml_tensor * norm1, const ggml_tensor * mul1, const ggml_tensor * add,
                                                 const ggml_tensor * scale, const ggml_tensor * norm2, const ggml_tensor * mul2,
                                                 const ggml_tensor * scale2 = nullptr);
// an identity-comb DSV4_HC_POST (its post weights made by SCALE -> SIGMOID -> SCALE when scale_in is set) and the
//   RMS_NORM -> MUL of each of its streams that follows, qwen4exp's hyper-connection combine and the next grouped norm
bool ggml_cuda_should_fuse_dsv4_hc_post_rms_norm(const ggml_tensor * post, const ggml_tensor * rms_norm, const ggml_tensor * mul);
void ggml_cuda_op_dsv4_hc_post_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post, const ggml_tensor * scale_in,
                                        const ggml_tensor * scale_out, ggml_tensor * rms_norm, ggml_tensor * mul);

// the two post-norms of a gemma4 MoE block summed (rms_norm(a) * wa + rms_norm(b) * wb) as the x of the above
bool ggml_cuda_should_fuse_rms_norm_pair_add(const ggml_tensor * norm_a, const ggml_tensor * mul_a, const ggml_tensor * norm_b,
                                             const ggml_tensor * mul_b, const ggml_tensor * add_ab, const ggml_tensor * norm1,
                                             const ggml_tensor * y, const ggml_tensor * mul2);
void ggml_cuda_op_rms_norm_add_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * norm1, ggml_tensor * mul1, ggml_tensor * add,
                                        ggml_tensor * scale, ggml_tensor * norm2, ggml_tensor * mul2,
                                        const ggml_tensor * norm_a = nullptr, const ggml_tensor * mul_a = nullptr,
                                        const ggml_tensor * norm_b = nullptr, const ggml_tensor * mul_b = nullptr,
                                        const ggml_tensor * scale2 = nullptr);

// rms_norm(x) * w * op(z) with op silu or sigmoid, the gated output norm of a gated delta net layer
bool ggml_cuda_should_fuse_rms_norm_gated(const ggml_tensor * rms_norm, const ggml_tensor * mul, const ggml_tensor * unary,
                                          const ggml_tensor * mul_gate);
void ggml_cuda_op_rms_norm_gated(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * unary,
                                 ggml_tensor * mul_gate);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// MUL_MAT(w, RMS_NORM(x)) of one token and up to 31 f32 rows of a long K (DeepSeek-V4's hyper-connection mixes), one launch
bool ggml_cuda_should_fuse_rms_norm_mul_mat(const ggml_tensor * rms_norm, const ggml_tensor * mul_mat);
void ggml_cuda_op_rms_norm_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul_mat);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
