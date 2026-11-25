#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>

// two state columns per warp, not upstream's four: on V100 with 8-12 heads per device (-sm tensor) two are fastest at
//   prefill (kernel -7..9%) and four are slower than one at 8 heads; decode is the same for all three
constexpr int gdn_cols_per_warp = 2;

// rms_norm(x, eps) * post_scale + post_bias over one head, x[r] being row r * lanes_per_col + lane. rms_norm_f32<256>
//   takes one value per thread, sums each 32 of them in a butterfly and adds those sums as below. The butterfly steps
//   wider than lanes_per_col pair rows of one lane, so they run in registers first, the squares rounded as there (no
//   FMA): these are bitwise its values.
template <int lanes_per_col, int rows_per_lane>
static __device__ __forceinline__ void gdn_l2_norm_head(
        float * x, const int ncols, const float eps, const float post_scale, const float post_bias) {
    constexpr int head     = lanes_per_col * rows_per_lane;
    constexpr int group    = head < 32 ? head : 32;       // the values one warp of rms_norm_f32 sums
    constexpr int per_lane = group / lanes_per_col;       // of those, the rows in each lane
    constexpr int n_groups = rows_per_lane / per_lane;
    static_assert(n_groups == 1 || n_groups == 2 || n_groups == 4, "unsupported head size");
    float w[n_groups];
#pragma unroll
    for (int g = 0; g < n_groups; g++) {
        float p[per_lane];
#pragma unroll
        for (int m = 0; m < per_lane; m++) {
            p[m] = __fmul_rn(x[g * per_lane + m], x[g * per_lane + m]);
        }
#pragma unroll
        for (int d = per_lane / 2; d > 0; d /= 2) {
#pragma unroll
            for (int m = 0; m < d; m++) {
                p[m] += p[m + d];
            }
        }
        w[g] = warp_reduce_sum<lanes_per_col>(p[0]);
    }
    float sum = w[0];
    if constexpr (n_groups == 2) {
        sum = w[0] + w[1];
    } else if constexpr (n_groups == 4) {
        sum = (w[0] + w[2]) + (w[1] + w[3]);
    }
    const float mean  = sum / ncols;
    const float scale = rsqrtf(mean + eps);
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        x[r] = scale * x[r] * post_scale + post_bias;
    }
}

// replay: pending tokens to run before the new ones (see ggml_gated_delta_net_ext)
struct gdn_replay_args {
    const float *   pk;    // [S_v, H_k, n_pend, n_seqs]
    const float *   pv;    // [S_v, H, n_pend, n_seqs]
    const float *   pgb;   // [g0, H, 2*n_pend, n_seqs], gates then betas
    const int32_t * n_rep; // [n_seqs]
    int64_t         sk1, sk2, sk3; // head, token and seq strides in floats
    int64_t         sv1, sv2, sv3;
    int64_t         sg1, sg2, sg3;
    int             n_pend;
    bool            keep;  // write the state after the replay, not the final one

    // optional: store the k, v, g, beta inputs of the new tokens (fused cpys)
    float *         st[4];
    int64_t         st2[4]; // token stride in floats
    int64_t         st3[4]; // seq stride in floats
};

template <int S_v, bool KDA, bool keep_rs_t, bool l2_qk, bool replay, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     int           l2_ncols, // S_v, as a runtime value: rms_norm divides by one
                                     float         l2_eps,
                                     float         l2_post_scale,
                                     float         l2_post_bias,
                                     const int32_t * s_ids,         // gathered: cache row of each sequence, else nullptr
                                     int64_t       s_row_stride,
                                     float *       s_pre,           // nullptr, or where the start state goes
                                     const gdn_replay_args rep) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    // gathered, the start state is the cache row of this sequence
    curr_state += (s_ids ? int64_t(s_ids[sequence]) * s_row_stride + h_idx * S_v * S_v : state_in_offset) + col * S_v;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }
    if (s_pre) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            s_pre[state_out_offset + col * S_v + i] = s_shard[r];
        }
    }

    if constexpr (replay) {
        const int     n_r = rep.n_rep[sequence];
        const float * pk  = rep.pk  + sequence * rep.sk3 + iq1   * rep.sk1;
        const float * pv  = rep.pv  + sequence * rep.sv3 + h_idx * rep.sv1;
        const float * pgb = rep.pgb + sequence * rep.sg3 + h_idx * rep.sg1;

        // the pending k was stored after its l2 norm (the fused stores below store the normalized k_reg)
        for (int t = 0; t < n_r; t++) {
            const float * k_t      = pk + t * rep.sk2;
            const float   v_col    = pv[t * rep.sv2 + col];
            const float * g_t      = pgb + t * rep.sg2;
            const float   beta_val = pgb[(rep.n_pend + t) * rep.sg2];

            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = k_t[r * lanes_per_col + lane_in_col];
            }

            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                g_reg[r] = expf(KDA ? g_t[r * lanes_per_col + lane_in_col] : g_t[0]);
            }

            // same update as below, without the attention output
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += (KDA ? g_reg[r] : 1.0f) * s_shard[r] * k_reg[r];
            }
            const float kv_col    = warp_reduce_sum<lanes_per_col>(kv_shard);
            const float delta_col = (v_col - (KDA ? 1.0f : g_reg[0]) * kv_col) * beta_val;

#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] = g_reg[r] * s_shard[r] + k_reg[r] * delta_col;
            }
        }

        if (rep.keep) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[col * S_v + r * lanes_per_col + lane_in_col] = s_shard[r];
            }
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (l2_qk) {
            gdn_l2_norm_head<lanes_per_col, rows_per_lane>(k_reg, l2_ncols, l2_eps, l2_post_scale, l2_post_bias);
            gdn_l2_norm_head<lanes_per_col, rows_per_lane>(q_reg, l2_ncols, l2_eps, l2_post_scale, l2_post_bias);
        }

        if constexpr (replay && !KDA) {
            if (rep.st[0] != nullptr) {
                // keep the inputs of this token pending (k after its norm): one CTA per k head writes k, each column its v
                const bool first = blockIdx.z == 0 && threadIdx.y == 0 && col_in_warp == 0;
                if (first && h_idx < neqk1_magic.z) {
                    float * dk = rep.st[0] + sequence * rep.st3[0] + t * rep.st2[0] + h_idx * S_v;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        dk[r * lanes_per_col + lane_in_col] = k_reg[r];
                    }
                }
                if (lane_in_col == 0) {
                    rep.st[1][sequence * rep.st3[1] + t * rep.st2[1] + h_idx * S_v + col] = v_t[col];
                }
                if (first && lane_in_col == 0) {
                    rep.st[2][sequence * rep.st3[2] + t * rep.st2[2] + h_idx] = *g_t;
                    rep.st[3][sequence * rep.st3[3] + t * rep.st2[3] + h_idx] = beta_val;
                }
            }
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // the gates in registers as in the replay loop, so that both sum alike
            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                g_reg[r] = expf(g_t[r * lanes_per_col + lane_in_col]);
            }

            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += g_reg[r] * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_reg[r] * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        if (replay && rep.keep) {
            return;
        }
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t, bool l2_qk, bool replay>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const ggml_cuda_gated_delta_net_l2 & l2,
        const ggml_cuda_gated_delta_net_gather & gather, const gdn_replay_args & rep, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // gdn_cols_per_warp columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    const auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
            (int) S_v, l2.eps, l2.post_scale, l2.post_bias, gather.ids, gather.row_stride, gather.pre, rep);
    };
    switch (S_v) {
        case 16:
            launch(gated_delta_net_cuda<16, KDA, keep_rs_t, l2_qk, replay>);
            break;
        case 32:
            launch(gated_delta_net_cuda<32, KDA, keep_rs_t, l2_qk, replay>);
            break;
        case 64:
            launch(gated_delta_net_cuda<64, KDA, keep_rs_t, l2_qk, replay>);
            break;
        case 128:
            launch(gated_delta_net_cuda<128, KDA, keep_rs_t, l2_qk, replay>);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache,
        const ggml_cuda_gated_delta_net_l2 * l2, const ggml_cuda_gated_delta_net_gather * gather) {
    // with l2, q and k are the inputs of their norms, which the kernel applies
    const ggml_tensor * src_q     = l2 ? l2->q : dst->src[0];
    const ggml_tensor * src_k     = l2 ? l2->k : dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    // gathered, src_state is not computed: the kernel reads the cache rows itself
    const float * s_d   = gather ? gather->states : (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    const ggml_cuda_gated_delta_net_l2     l2_args     = l2     ? *l2     : ggml_cuda_gated_delta_net_l2{};
    const ggml_cuda_gated_delta_net_gather gather_args = gather ? *gather : ggml_cuda_gated_delta_net_gather{};

    // replay (ggml_gated_delta_net_ext): the kept pending tokens of the last ubatch run first, from src 6..9
    const ggml_tensor * src_pend = dst->src[6];

    gdn_replay_args rep = {};
    if (src_pend) {
        const ggml_tensor * src_pv  = dst->src[7];
        const ggml_tensor * src_pgb = dst->src[8];

        // no rollback slots under replay: one state out, and no start state stored for a slot
        GGML_ASSERT(K == 1);
        GGML_ASSERT(gather_args.pre == nullptr);
        rep.pk     = (const float *) src_pend->data;
        rep.pv     = (const float *) src_pv->data;
        rep.pgb    = (const float *) src_pgb->data;
        rep.n_rep  = (const int32_t *) dst->src[9]->data;
        rep.sk1    = src_pend->nb[1] / sizeof(float);
        rep.sk2    = src_pend->nb[2] / sizeof(float);
        rep.sk3    = src_pend->nb[3] / sizeof(float);
        rep.sv1    = src_pv->nb[1] / sizeof(float);
        rep.sv2    = src_pv->nb[2] / sizeof(float);
        rep.sv3    = src_pv->nb[3] / sizeof(float);
        rep.sg1    = src_pgb->nb[1] / sizeof(float);
        rep.sg2    = src_pgb->nb[2] / sizeof(float);
        rep.sg3    = src_pgb->nb[3] / sizeof(float);
        rep.n_pend = ggml_get_op_params_i32(dst, 1);
        rep.keep   = ggml_get_op_params_i32(dst, 2) != 0;
        if (cache != nullptr) {
            for (int i = 0; i < 4; ++i) {
                rep.st[i]  = cache->in_dst[i];
                rep.st2[i] = cache->in_nb2[i];
                rep.st3[i] = cache->in_nb3[i];
            }
        }
    }

    // the pending tokens alone (n_tokens 0, keep): their state into state_out, for a chunked kernel to start from
    const auto launch_replay_only = [&](float * state_out) {
        gdn_replay_args rep_only = rep;
        rep_only.keep = true;
        launch_gated_delta_net<false, false, false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out,
            S_v, H, 0, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ggml_cuda_gated_delta_net_l2{}, gather_args, rep_only,
            stream);
    };

    // prefill: the chunked kernels (gated_delta_net_chunked.cu), which read q, k and v as float4. With rollback snapshots
    //   they stop K-1 tokens early and leave their state in slot K-1, and the kernel below runs the last K-1 tokens from
    //   there; its attention output has the token count as sequence stride, so that needs one sequence. With replay they
    //   start from the state after the pending tokens (a ubatch this long never keeps its own tokens pending)
    static const bool chunked_enabled = [] {
        const char * env = getenv("GGML_CUDA_GDN_CHUNKED");
        return env == nullptr || atoi(env) != 0;
    }();
    const bool aligned16 = (uintptr_t) q_d % 16 == 0 && (uintptr_t) k_d % 16 == 0 && (uintptr_t) v_d % 16 == 0 &&
        sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0 && sv1 % 4 == 0 && sv2 % 4 == 0 && sv3 % 4 == 0;
    const int64_t n_tail    = K - 1;
    const int64_t n_chunked = n_tokens - n_tail;
    if (chunked_enabled && !l2 && aligned16 && (n_tail == 0 || n_seqs == 1) && !(src_pend && (rep.keep || rep.st[0])) &&
            ggml_cuda_gated_delta_net_chunked_supported(ctx.device, S_v, kda, n_chunked)) {
        ggml_cuda_pool_alloc<char> scratch(ctx.pool(), ggml_cuda_gated_delta_net_chunked_scratch_size(H, n_chunked, n_seqs));
        ggml_cuda_pool_alloc<float> rep_state(ctx.pool()); // after scratch: the pool frees in reverse order
        ggml_cuda_gated_delta_net_chunked_args args;
        args.q            = q_d;
        args.k            = k_d;
        args.v            = v_d;
        args.g            = g_d;
        args.beta         = b_d;
        args.s            = s_d;
        args.dst          = dst_d;
        args.state        = state_d + n_tail*state_slot_stride;
        args.ids          = gather_args.ids;
        args.s_row_stride = gather_args.row_stride;
        args.s_pre        = gather_args.pre;
        args.H            = H;
        args.n_tokens     = n_chunked;
        args.n_seqs       = n_seqs;
        args.sq1 = sq1; args.sq2 = sq2; args.sq3 = sq3;
        args.sv1 = sv1; args.sv2 = sv2; args.sv3 = sv3;
        args.sb1 = sb1; args.sb2 = sb2; args.sb3 = sb3;
        args.neqk1 = neqk1;
        args.rq3   = rq3;
        args.scale = scale;
        if (src_pend) {
            // a separate state, not the output in place: the scan reads s and writes state through restrict pointers
            launch_replay_only(rep_state.alloc(S_v*S_v*H*n_seqs));
            args.s            = rep_state.get();
            args.ids          = nullptr;
            args.s_row_stride = 0;
        }
        ggml_cuda_gated_delta_net_chunked(args, scratch.get(), stream);
        if (n_tail > 0) {
            launch_gated_delta_net<false, true, false, false>(
                q_d + n_chunked*sq2, k_d + n_chunked*sq2, v_d + n_chunked*sv2, g_d + n_chunked*sb2, b_d + n_chunked*sb2,
                args.state, dst_d + n_chunked*S_v*H, state_d,
                S_v, H, n_tail, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, ggml_cuda_gated_delta_net_l2{},
                ggml_cuda_gated_delta_net_gather{}, gdn_replay_args{}, stream);
        }
        return;
    }

    // upstream's BF16 tensor-core prefill (gated-delta-net-mma.cu): Ampere-class only (a runtime gate to cc 860/890,
    //   Blackwell and DGX Spark), S_v 128, H_k 16 and H in {16,32,48,64} per device, never on V100. It reads q, k and the
    //   start state from dst->src, which the l2 and gather folds do not compute, and writes only the final state, which
    //   would leave rollback slots 1..K-1 stale after a long ubatch (seq_rm rolls back after any ubatch); nor does it
    //   write the fused cache rows at their slot stride. It does not replay: the pending tokens run first into its state
    //   output, which it then updates in place, as upstream
    if (!l2 && !gather && !keep_rs && cache == nullptr && !(src_pend && rep.keep)) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        if (!src_pend) {
            args.state_out = state_d;
            if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
                return;
            }
        } else {
            args.state     = state_d;
            args.state_out = state_d;
            if (ggml_cuda_gdn_mma_supported(ctx.device, args)) {
                launch_replay_only(state_d);
                ggml_cuda_gdn_mma_launch(ctx.device, args, stream);
                return;
            }
        }
    }

    const auto launch = [&](auto kda_c, auto keep_rs_c, auto l2_c, auto replay_c) {
        launch_gated_delta_net<decltype(kda_c)::value, decltype(keep_rs_c)::value, decltype(l2_c)::value,
                               decltype(replay_c)::value>(
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, l2_args, gather_args, rep, stream);
    };
    const auto launch_l2 = [&](auto kda_c, auto keep_rs_c, auto replay_c) {
        if (l2) {
            launch(kda_c, keep_rs_c, std::true_type{}, replay_c);
        } else {
            launch(kda_c, keep_rs_c, std::false_type{}, replay_c);
        }
    };
    // snapshots (K > 1, the deprecated ggml_gated_delta_net) and replay do not mix (ggml.c asserts it)
    const auto launch_mode = [&](auto kda_c) {
        if (keep_rs) {
            launch_l2(kda_c, std::true_type{}, std::false_type{});
        } else if (src_pend) {
            launch_l2(kda_c, std::false_type{}, std::true_type{});
        } else {
            launch_l2(kda_c, std::false_type{}, std::false_type{});
        }
    };
    if (kda) {
        launch_mode(std::true_type{});
    } else {
        launch_mode(std::false_type{});
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr, nullptr, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache,
        const ggml_cuda_gated_delta_net_l2 * l2, const ggml_cuda_gated_delta_net_gather * gather) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache, l2, gather);
}
