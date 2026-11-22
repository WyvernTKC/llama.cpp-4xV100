#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// NORM -> MUL(weight row) -> ADD(bias row) of rows under 1024 columns, one launch with the values of the three ops
bool ggml_cuda_should_fuse_norm_mul_add(const ggml_tensor * norm, const ggml_tensor * mul, const ggml_tensor * add);
void ggml_cuda_op_norm_mul_add(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul, ggml_tensor * add);

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

// the weights of the matmuls after a kernel that waits, for it to prefetch into L2 while it waits
struct ggml_cuda_l2_prefetch_args {
    const char * ptr[2]   = {};
    int          bytes[2] = {};
};

// pf: weights to prefetch into L2 while the lines come (the one-token f32 router after the norm, maybe none); nullptr
//   takes the instance without the prefetch, see ggml_backend_cuda_session_begin
void ggml_cuda_op_rms_norm_pre_add_ar(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul,
                                      const ggml_cuda_ar_fold_args & ar, ggml_tensor * t,
                                      const ggml_cuda_l2_prefetch_args * pf = nullptr);

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
//   should_fuse: 0 no, 1 yes, 2 yes with sync (an output sits on an input other than its own row of the residual: the
//   blocks wait for each other before storing). raw: the inputs of the post weights' SCALE when gated
int  ggml_cuda_should_fuse_dsv4_hc_post_rms_norm(const ggml_tensor * post, const ggml_tensor * rms_norm, const ggml_tensor * mul,
                                                 const ggml_tensor * raw);
void ggml_cuda_op_dsv4_hc_post_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post, const ggml_tensor * scale_in,
                                        const ggml_tensor * scale_out, ggml_tensor * rms_norm, ggml_tensor * mul, bool sync);

// the two post-norms of a gemma4 MoE block summed (rms_norm(a) * wa + rms_norm(b) * wb) as the x of the above
bool ggml_cuda_should_fuse_rms_norm_pair_add(const ggml_tensor * norm_a, const ggml_tensor * mul_a, const ggml_tensor * norm_b,
                                             const ggml_tensor * mul_b, const ggml_tensor * add_ab, const ggml_tensor * norm1,
                                             const ggml_tensor * y, const ggml_tensor * mul2);
void ggml_cuda_op_rms_norm_add_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * norm1, ggml_tensor * mul1, ggml_tensor * add,
                                        ggml_tensor * scale, ggml_tensor * norm2, ggml_tensor * mul2,
                                        const ggml_tensor * norm_a = nullptr, const ggml_tensor * mul_a = nullptr,
                                        const ggml_tensor * norm_b = nullptr, const ggml_tensor * mul_b = nullptr,
                                        const ggml_tensor * scale2 = nullptr);

// the same with the all-reduce of t, the post-norm's input, folded in: t's sum is polled from the LL lines of ar and
//   written into t if t is not nullptr
bool ggml_cuda_rms_norm_add_rms_norm_ar_supported(const ggml_tensor * norm1, const ggml_tensor * mul1, const ggml_tensor * add,
                                                  const ggml_tensor * scale, const ggml_tensor * norm2, const ggml_tensor * mul2,
                                                  const ggml_tensor * t);
void ggml_cuda_op_rms_norm_add_rms_norm_ar(ggml_backend_cuda_context & ctx, ggml_tensor * norm1, ggml_tensor * mul1, ggml_tensor * add,
                                           ggml_tensor * scale, ggml_tensor * norm2, ggml_tensor * mul2,
                                           const ggml_cuda_ar_fold_args & ar, ggml_tensor * t);

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

// the same with x the DSV4_HC_POST (with comb, one token) right before the norm, computed and written in that launch;
//   fold: { MUL, ADD, SCALE, DSV4_HC_COMB } that make its post weights and comb, computed in the launch, or nullptr
bool ggml_cuda_should_fuse_dsv4_hc_post_rms_norm_mul_mat(const ggml_tensor * post, const ggml_tensor * rms_norm,
        const ggml_tensor * mul_mat);
bool ggml_cuda_should_fold_dsv4_hc_post(const ggml_tensor * post, const ggml_tensor * mul_p, const ggml_tensor * add_p,
        const ggml_tensor * scale_p, const ggml_tensor * comb);
void ggml_cuda_op_dsv4_hc_post_rms_norm_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * rms_norm,
        ggml_tensor * mul_mat, const ggml_tensor * const * fold);

// and the DSV4_HC_PRE of the next block (its weights MUL -> ADD -> SIGMOID -> SCALE of a slice of those mixes, as
//   ggml_cuda_op_dsv4_hc_pre_coef fuses them) -> RMS_NORM -> MUL, with the q8_1 copy of the MUL if requested; pre_out:
//   the DSV4_HC_PRE dst is read by more than the norm
bool ggml_cuda_should_fuse_dsv4_hc_post_pre(const ggml_tensor * post, const ggml_tensor * mul_mat, const ggml_tensor * mul_w,
        const ggml_tensor * add_w, const ggml_tensor * scale_w, const ggml_tensor * pre, const ggml_tensor * norm,
        const ggml_tensor * mul, bool pre_out);
void ggml_cuda_op_dsv4_hc_post_pre(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * rms_norm,
        ggml_tensor * mul_mat, const ggml_tensor * mul_w, const ggml_tensor * add_w, const ggml_tensor * scale_w,
        ggml_tensor * pre, bool pre_out, ggml_tensor * norm2, ggml_tensor * mul2, const ggml_tensor * const * fold);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// dst = y*clamp(t/(sqrt(sum(y*y))*scale + bias), lo, hi) per row, y = x*s (s: a vector per row of x, or none); with tau
//   ([1, 1, ne2, 1]) dst times tau of the row's dim-2 index after that, as a MUL apart
struct ggml_cuda_l2_clip_set {
    const ggml_tensor * x   = nullptr;
    const ggml_tensor * s   = nullptr;
    const ggml_tensor * tau = nullptr;
    ggml_tensor       * dst = nullptr;
    float scale = 1.0f, bias = 0.0f, t = 0.0f, lo = 0.0f, hi = 0.0f;
};

void ggml_cuda_op_l2_clip(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * s, ggml_tensor * dst,
        float scale, float bias, float t, float lo, float hi);

// one or two row sets of ggml_cuda_op_l2_clip in one launch (b.x nullptr: a alone)
void ggml_cuda_op_l2_clip_sets(ggml_backend_cuda_context & ctx, const ggml_cuda_l2_clip_set & a, const ggml_cuda_l2_clip_set & b);

// a depth mix (Inkling 2): states = the rows of x (already normed, [n, n_x, n_tokens]) and rms_norm(cur) ([n, 1, n_tokens]),
//   dst[:, q, t] = sum_s softmax_s(scale*<state_s, query[:, q]>)*state_s
bool ggml_cuda_should_fuse_depth_mix(const ggml_tensor * x, const ggml_tensor * cur, const ggml_tensor * query, const ggml_tensor * dst);
void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * cur, const ggml_tensor * query,
        ggml_tensor * dst, float eps, float scale);
