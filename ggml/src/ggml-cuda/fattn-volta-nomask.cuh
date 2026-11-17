#pragma once

#include "common.cuh"

// Flash attention on Volta tensor cores (mma.m8n8k4) for FLASH_ATTN_EXT without mask, sinks, softcap or ALiBi:
// F32 Q, F16 K/V, F32 dst, head size 64, 72 or 128. Vision towers run this case (unmasked self-attention).
// Ported from the stable-diffusion.cpp Volta kernel (volta_fattn.cuh).
//
// A block owns 128 queries of one head and 8 warps. Warps 2i and 2i+1 own the same 32 queries and split each
// 64-key tile (32 keys each); their softmax states are merged at the end. Volta has no cp.async, so the next tile
// goes through registers: the loads of V(j) are issued before S(j), those of K(j+1) before P.V(j).
// Scores are F32, the output accumulators F16 (as in the Volta path of fattn-mma-f16).
//
// mma.m8n8k4 fragments (quad pair = (lane % 16) / 4 does its own 8x8x4 product):
//   A .row (8 x 4):  row (lane / 16) * 4 + lane % 4 of the quad pair, k 0..3.
//   B .col (4 x 8):  column (lane / 16) * 4 + lane % 4, k 0..3.
//   B .row (4 x 8):  row lane % 4, columns (lane / 16) * 4 + 0..3.
//   C f32 (8 x 8):   rows (lane / 16) * 4 + lane % 2 + {0, 2}, columns (lane & 2) + {0, 1, 4, 5}.
//   C f16 (8 x 8):   row (lane / 16) * 4 + lane % 4, columns 0..7.
// Query rows are permuted so that the A row of a thread is query `lane`; its f32 C rows are then queries
// (lane & ~2) and (lane & ~2) + 2.

namespace fattn_volta_nomask {

constexpr int BQ      = 128; // queries per block
constexpr int BK      = 64;  // keys per tile
constexpr int WARPS   = 8;
constexpr int THREADS = WARPS * 32;
constexpr float SUM_LIMIT = 64.0f; // a warp's share of the sum of P of a row stays below SUM_LIMIT + 32
constexpr float SUM_SHIFT = 6.0f;  // log2 of the step down

template <int D>
struct shape {
    static_assert(D == 64 || D == 72 || D == 128, "head size 64, 72 or 128");
    static constexpr int PITCH    = D + 8;  // halves per shared row: conflict-free 16-byte fragment loads
    static constexpr int SMEM     = 2 * BK * PITCH * 2 + 4 * 64 * 4; // K and V tiles (Q tile and merge reuse them) + merge stats
    static constexpr int PIECES   = D / 8;  // 16-byte pieces of a K or V row
    static constexpr int N_PIECES = BK * PIECES;
    static constexpr int CHUNKS   = (N_PIECES + THREADS - 1) / THREADS;
    static constexpr int ROW_F4   = D / 4;  // float4 per output row
    // 64 dims fit two blocks per SM in 128 registers (a few spills). At 72 dims two blocks spill more and are slower
    // (V100, 16 heads x 4096: 2.37 vs 2.08 ms), so 72 and 128 run one block.
    static constexpr int MIN_BLOCKS = D == 64 ? 2 : 1;
};

struct params {
    const float * q;
    const char  * k;
    const char  * v;
    float       * dst;
    int n_q, n_kv, n_head, head_ratio; // head_ratio = q heads per K/V head
    int64_t q_nb1, q_nb2, q_nb3;       // bytes
    int64_t k_nb1, k_nb2, k_nb3;
    int64_t v_nb1, v_nb2, v_nb3;
    float scale_log2;                  // scale * log2(e)
};

#ifdef VOLTA_MMA_AVAILABLE
static __device__ __forceinline__ void mma_f32(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ void mma_f16(uint32_t (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t *>(&h);
}

static __device__ __forceinline__ uint32_t mul_half2(uint32_t x, half2 f) {
    half2 h = __hmul2(*reinterpret_cast<half2 *>(&x), f);
    return *reinterpret_cast<uint32_t *>(&h);
}

// Piece c of a tile is key c / PIECES, dims (c % PIECES) * 8 .. + 7.
template <int D>
static __device__ __forceinline__ void load_tile(uint4 (&stage)[shape<D>::CHUNKS], const char * rows, int64_t nb1, int key0, int n_kv) {
    constexpr int PIECES = shape<D>::PIECES;
#pragma unroll
    for (int i = 0; i < shape<D>::CHUNKS; ++i) {
        const int c   = threadIdx.x + i * THREADS;
        const int key = key0 + c / PIECES;
        const bool ok = (shape<D>::N_PIECES % THREADS == 0 || c < shape<D>::N_PIECES) && key < n_kv;
        stage[i] = ok ? *reinterpret_cast<const uint4 *>(rows + key * nb1 + (c % PIECES) * 16) : make_uint4(0, 0, 0, 0);
    }
}

template <int D>
static __device__ __forceinline__ void store_k(half * tile, const uint4 (&stage)[shape<D>::CHUNKS]) {
    constexpr int PIECES = shape<D>::PIECES;
#pragma unroll
    for (int i = 0; i < shape<D>::CHUNKS; ++i) {
        const int c = threadIdx.x + i * THREADS;
        if (shape<D>::N_PIECES % THREADS == 0 || c < shape<D>::N_PIECES) {
            *reinterpret_cast<uint4 *>(tile + (c / PIECES) * shape<D>::PITCH + (c % PIECES) * 8) = stage[i];
        }
    }
}

// V rows keep each 16-dim chunk as dims 0-3, 8-11, 4-7, 12-15: one 16-byte load gives a thread its four B-row
// dims of two neighbouring 8-dim output blocks. The last chunk of D = 72 holds dims 64-67 and 68-71 at 0-3 and 8-11.
template <int D>
static __device__ __forceinline__ void store_v(half * tile, const uint4 (&stage)[shape<D>::CHUNKS]) {
    constexpr int PIECES = shape<D>::PIECES;
#pragma unroll
    for (int i = 0; i < shape<D>::CHUNKS; ++i) {
        const int c = threadIdx.x + i * THREADS;
        if (shape<D>::N_PIECES % THREADS == 0 || c < shape<D>::N_PIECES) {
            const int piece = c % PIECES;
            half * row      = tile + (c / PIECES) * shape<D>::PITCH + (piece / 2) * 16;
            const int first = (piece % 2) * 4;
            *reinterpret_cast<uint2 *>(row + first)     = make_uint2(stage[i].x, stage[i].y);
            *reinterpret_cast<uint2 *>(row + first + 8) = make_uint2(stage[i].z, stage[i].w);
        }
    }
}
#endif // VOLTA_MMA_AVAILABLE

template <int D, int MIN_BLOCKS>
static __global__ void __launch_bounds__(THREADS, MIN_BLOCKS) flash_attn_ext_volta_nomask(const params p) {
#ifdef VOLTA_MMA_AVAILABLE
    constexpr int PITCH  = shape<D>::PITCH;
    constexpr int CHUNKS = shape<D>::CHUNKS;
    constexpr int NB     = D / 8; // 8-dim output blocks
    extern __shared__ __align__(16) half smem_volta_nomask[];
    half * smem   = smem_volta_nomask;
    half * tile_k = smem;
    half * tile_v = smem + BK * PITCH;

    const int lane  = threadIdx.x % 32;
    const int warp  = threadIdx.x / 32;
    const int group = warp / 2; // 32 queries
    const int kh    = warp % 2; // which 32 keys of each tile
    const int h     = blockIdx.y;
    const int batch = blockIdx.z;
    const int q0    = blockIdx.x * BQ;

    const char * q = reinterpret_cast<const char *>(p.q) + h * p.q_nb2 + batch * p.q_nb3;
    const char * k = p.k + (h / p.head_ratio) * p.k_nb2 + batch * p.k_nb3;
    const char * v = p.v + (h / p.head_ratio) * p.v_nb2 + batch * p.v_nb3;

    // Q tile, scaled into log2 units, F16, then A fragments: query `lane` of the group.
    {
        half * tile_q = smem; // BQ rows of PITCH halves: exactly the K and V tiles
#pragma unroll 4
        for (int i = threadIdx.x; i < BQ * D / 4; i += THREADS) {
            const int row = i / (D / 4);
            const int col = (i % (D / 4)) * 4;
            const int qi  = q0 + row;
            float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (qi < p.n_q) {
                x = *reinterpret_cast<const float4 *>(q + qi * p.q_nb1 + col * 4);
            }
            *reinterpret_cast<uint2 *>(tile_q + row * PITCH + col) =
                make_uint2(pack_half2(x.x * p.scale_log2, x.y * p.scale_log2), pack_half2(x.z * p.scale_log2, x.w * p.scale_log2));
        }
    }
    __syncthreads();
    uint32_t qa[NB][4];
#pragma unroll
    for (int s = 0; s < NB; ++s) {
        const uint4 x = *reinterpret_cast<const uint4 *>(smem + (group * 32 + lane) * PITCH + s * 8);
        qa[s][0] = x.x;
        qa[s][1] = x.y;
        qa[s][2] = x.z;
        qa[s][3] = x.w;
    }
    __syncthreads();

    uint4 stage[CHUNKS];
    load_tile<D>(stage, k, p.k_nb1, 0, p.n_kv);
    store_k<D>(tile_k, stage);
    __syncthreads();

    uint32_t o[NB][4];
#pragma unroll
    for (int d = 0; d < NB; ++d) {
        o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0u;
    }
    float m_a = -FLT_MAX / 2, m_b = -FLT_MAX / 2; // running max of rows (lane & ~2) and (lane & ~2) + 2
    float l_a = 0.0f, l_b = 0.0f;                 // this thread's part of the running sums

    const int tiles   = (p.n_kv + BK - 1) / BK;
    const int key_row = kh * 32 + (lane / 16) * 4 + lane % 4; // B .col: key of this thread within a group
    for (int j = 0; j < tiles; ++j) {
        load_tile<D>(stage, v, p.v_nb1, j * BK, p.n_kv);

        // S = Q K^T for 32 queries x 32 keys, in four groups of 8 keys.
        float s[4][8];
#pragma unroll
        for (int g = 0; g < 4; ++g) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                s[g][l] = 0.0f;
            }
        }
        // consecutive MMAs go to different groups: an MMA waits for the previous one on its accumulator
#pragma unroll
        for (int st = 0; st < NB; ++st) {
            uint4 b[4];
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                b[g] = *reinterpret_cast<const uint4 *>(tile_k + (key_row + g * 8) * PITCH + st * 8);
            }
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                mma_f32(s[g], qa[st][0], qa[st][1], b[g].x, b[g].y);
            }
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                mma_f32(s[g], qa[st][2], qa[st][3], b[g].z, b[g].w);
            }
        }
        if ((j + 1) * BK > p.n_kv) {
#pragma unroll
            for (int g = 0; g < 4; ++g) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int key = j * BK + kh * 32 + g * 8 + (lane & 2) + (l & 5);
                    if (key >= p.n_kv) {
                        s[g][l] = -INFINITY;
                    }
                }
            }
        }

        // Online softmax. l & 2 selects row b. Lane ^ 2 holds the other half of each row.
        float mx_a = -FLT_MAX / 2, mx_b = -FLT_MAX / 2;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                if (l & 2) {
                    mx_b = fmaxf(mx_b, s[g][l]);
                } else {
                    mx_a = fmaxf(mx_a, s[g][l]);
                }
            }
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xFFFFFFFF, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xFFFFFFFF, mx_b, 2));
        // The F16 accumulators hold sum(P.V), up to the row sum of P times max |V|: a long flat row overflows.
        // Once a row sum passes SUM_LIMIT, raise its reference level: sum and accumulators drop by 2^-SUM_SHIFT.
        const float row_a   = l_a + __shfl_xor_sync(0xFFFFFFFF, l_a, 2);
        const float row_b   = l_b + __shfl_xor_sync(0xFFFFFFFF, l_b, 2);
        const float new_a   = fmaxf(m_a + (row_a > SUM_LIMIT ? SUM_SHIFT : 0.0f), mx_a);
        const float new_b   = fmaxf(m_b + (row_b > SUM_LIMIT ? SUM_SHIFT : 0.0f), mx_b);
        const float alpha_a = exp2f(m_a - new_a);
        const float alpha_b = exp2f(m_b - new_b);
        m_a = new_a;
        m_b = new_b;
        float sum_a = 0.0f, sum_b = 0.0f;
        uint32_t pa[4][4]; // P as A fragments: query `lane`, keys 8g .. 8g + 7
#pragma unroll
        for (int g = 0; g < 4; ++g) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                s[g][l] = exp2f(s[g][l] - ((l & 2) ? new_b : new_a));
                if (l & 2) {
                    sum_b += s[g][l];
                } else {
                    sum_a += s[g][l];
                }
            }
            // C layout -> A layout: lanes with (lane & 2) == 0 keep row a and get its other keys from lane ^ 2,
            // the others keep row b. x[0..1] are keys 0-3, x[2..3] keys 4-7.
            const uint32_t x[4] = {pack_half2(s[g][0], s[g][1]), pack_half2(s[g][2], s[g][3]),
                                   pack_half2(s[g][4], s[g][5]), pack_half2(s[g][6], s[g][7])};
            const bool is_b = lane & 2;
            const uint32_t give0 = is_b ? x[0] : x[1];
            const uint32_t give1 = is_b ? x[2] : x[3];
            const uint32_t got0  = __shfl_xor_sync(0xFFFFFFFF, give0, 2);
            const uint32_t got1  = __shfl_xor_sync(0xFFFFFFFF, give1, 2);
            pa[g][0] = is_b ? got0 : x[0];
            pa[g][1] = is_b ? x[1] : got0;
            pa[g][2] = is_b ? got1 : x[2];
            pa[g][3] = is_b ? x[3] : got1;
        }
        l_a = l_a * alpha_a + sum_a;
        l_b = l_b * alpha_b + sum_b;
        // once the running maxima settle, most tiles change none of the 32 rows of the warp: skip the rescale
        const float alpha = (lane & 2) ? alpha_b : alpha_a;
        if (__any_sync(0xFFFFFFFF, alpha != 1.0f)) {
            const half2 alpha2 = __floats2half2_rn(alpha, alpha);
#pragma unroll
            for (int d = 0; d < NB; ++d) {
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    o[d][x] = mul_half2(o[d][x], alpha2);
                }
            }
        }

        store_v<D>(tile_v, stage);
        __syncthreads(); // V(j) visible, every warp is done with K(j)
        if (j + 1 < tiles) {
            load_tile<D>(stage, k, p.k_nb1, (j + 1) * BK, p.n_kv);
        }

        // O += P V. For each 4 keys one 16-byte load gives the B rows of 8-dim blocks 2b and 2b + 1.
#pragma unroll
        for (int g = 0; g < 4; ++g) {
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const half * row = tile_v + (kh * 32 + g * 8 + kk * 4 + lane % 4) * PITCH + (lane / 16) * 8;
#pragma unroll
                for (int b = 0; b < D / 16; ++b) {
                    const uint4 vb = *reinterpret_cast<const uint4 *>(row + b * 16);
                    mma_f16(o[2 * b],     pa[g][2 * kk], pa[g][2 * kk + 1], vb.x, vb.y);
                    mma_f16(o[2 * b + 1], pa[g][2 * kk], pa[g][2 * kk + 1], vb.z, vb.w);
                }
                if constexpr (NB % 2 != 0) {
                    const uint2 vb = *reinterpret_cast<const uint2 *>(row + (D / 16) * 16);
                    mma_f16(o[NB - 1], pa[g][2 * kk], pa[g][2 * kk + 1], vb.x, vb.y);
                }
            }
        }
        if (j + 1 < tiles) {
            store_k<D>(tile_k, stage);
        }
        __syncthreads(); // K(j + 1) visible, every warp is done with V(j)
    }

    // row sums over both halves of each row, then the state of output row `lane` of this thread
    l_a += __shfl_xor_sync(0xFFFFFFFF, l_a, 2);
    l_b += __shfl_xor_sync(0xFFFFFFFF, l_b, 2);
    const float m_row = (lane & 2) ? m_b : m_a;
    const float l_row = (lane & 2) ? l_b : l_a;

    // merge the two key halves of each query group through shared memory (the tiles are free now)
    half  * merge_o = smem + group * 32 * PITCH;
    float * merge_s = reinterpret_cast<float *>(smem + 2 * BK * PITCH) + group * 64;
    if (kh == 1) {
#pragma unroll
        for (int d = 0; d < NB; ++d) {
            *reinterpret_cast<uint4 *>(merge_o + lane * PITCH + d * 8) = make_uint4(o[d][0], o[d][1], o[d][2], o[d][3]);
        }
        merge_s[lane]      = m_row;
        merge_s[32 + lane] = l_row;
    }
    __syncthreads();
    if (kh == 0) {
        // the merged, normalized row `lane` replaces the partner row in shared memory, then the warp writes whole rows
        const float m1  = merge_s[lane];
        const float l1  = merge_s[32 + lane];
        const float m   = fmaxf(m_row, m1);
        const float f0  = exp2f(m_row - m);
        const float f1  = exp2f(m1 - m);
        const float inv = 1.0f / (l_row * f0 + l1 * f1);
        const half2 w0  = __floats2half2_rn(f0 * inv, f0 * inv);
        const half2 w1  = __floats2half2_rn(f1 * inv, f1 * inv);
#pragma unroll
        for (int d = 0; d < NB; ++d) {
            uint4 other = *reinterpret_cast<const uint4 *>(merge_o + lane * PITCH + d * 8);
            uint32_t * t = reinterpret_cast<uint32_t *>(&other);
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const half2 r = __hfma2(*reinterpret_cast<const half2 *>(&o[d][x]), w0,
                                        __hmul2(*reinterpret_cast<const half2 *>(&t[x]), w1));
                t[x] = *reinterpret_cast<const uint32_t *>(&r);
            }
            *reinterpret_cast<uint4 *>(merge_o + lane * PITCH + d * 8) = other;
        }
        __syncwarp();
        constexpr int ROW_F4 = shape<D>::ROW_F4;
        for (int i = lane; i < 32 * ROW_F4; i += 32) {
            const int r  = i / ROW_F4;
            const int qi = q0 + group * 32 + r;
            if (qi >= p.n_q) {
                break;
            }
            const int dims = (i % ROW_F4) * 4;
            const uint2  x = *reinterpret_cast<const uint2 *>(merge_o + r * PITCH + dims);
            const float2 a = __half22float2(*reinterpret_cast<const half2 *>(&x.x));
            const float2 c = __half22float2(*reinterpret_cast<const half2 *>(&x.y));
            float * out = p.dst + ((int64_t(batch) * p.n_q + qi) * p.n_head + h) * D;
            *reinterpret_cast<float4 *>(out + dims) = make_float4(a.x, a.y, c.x, c.y);
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // VOLTA_MMA_AVAILABLE
}

template <int D>
static void launch(const params & p, int n_batch, cudaStream_t stream) {
    constexpr int PITCH = shape<D>::PITCH;
    static_assert(BQ * PITCH <= 2 * BK * PITCH, "the Q tile must fit in the K and V tiles");
    static_assert(shape<D>::SMEM <= 48 * 1024, "more than the default shared memory per block");
    const dim3 grid((p.n_q + BQ - 1) / BQ, p.n_head, n_batch);
    flash_attn_ext_volta_nomask<D, shape<D>::MIN_BLOCKS><<<grid, THREADS, shape<D>::SMEM, stream>>>(p);
}

} // namespace fattn_volta_nomask

// FLASH_ATTN_EXT cases the kernel above takes: Volta, no mask/sinks/softcap/ALiBi, F32 Q, F16 K/V, head size 64/72/128,
// contiguous 16-byte aligned rows, at least 64 queries and one block per SM (it walks every key in one block, so few
// queries x many keys are faster in the stream-k kernels). GGML_CUDA_FATTN_VOLTA_NOMASK=0 turns it off.
static bool ggml_cuda_fattn_volta_nomask_applies(const int cc, const int nsm, const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_FATTN_VOLTA_NOMASK");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled || !volta_mma_available(cc)) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (dst->src[3] || dst->src[4] || dst->src[5] || dst->src[6] || dst->src[7]) {
        return false;
    }
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;
    }
    const int64_t D = Q->ne[0];
    if ((D != 64 && D != 72 && D != 128) || K->ne[0] != D || V->ne[0] != D) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0 || K->ne[2] != V->ne[2] || K->ne[1] != V->ne[1] || K->ne[1] == 0) {
        return false;
    }
    for (const ggml_tensor * t : {Q, K, V}) {
        if ((uintptr_t) t->data % 16 != 0 || t->nb[0] != ggml_type_size(t->type)) {
            return false;
        }
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }
    if (Q->ne[1] < fattn_volta_nomask::BQ / 2 || Q->ne[1] > INT_MAX || K->ne[1] > INT_MAX) {
        return false;
    }
    const int64_t blocks = (Q->ne[1] + fattn_volta_nomask::BQ - 1) / fattn_volta_nomask::BQ * Q->ne[2] * Q->ne[3];
    return blocks >= nsm;
}

static void ggml_cuda_flash_attn_ext_volta_nomask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    fattn_volta_nomask::params p;
    p.q          = (const float *) Q->data;
    p.k          = (const char *) K->data;
    p.v          = (const char *) V->data;
    p.dst        = (float *) dst->data;
    p.n_q        = Q->ne[1];
    p.n_kv       = K->ne[1];
    p.n_head     = Q->ne[2];
    p.head_ratio = Q->ne[2] / K->ne[2];
    p.q_nb1 = Q->nb[1]; p.q_nb2 = Q->nb[2]; p.q_nb3 = Q->nb[3];
    p.k_nb1 = K->nb[1]; p.k_nb2 = K->nb[2]; p.k_nb3 = K->nb[3];
    p.v_nb1 = V->nb[1]; p.v_nb2 = V->nb[2]; p.v_nb3 = V->nb[3];
    p.scale_log2 = scale * 1.4426950408889634f;

    const int n_batch = Q->ne[3];
    cudaStream_t stream = ctx.stream();
    switch (Q->ne[0]) {
        case  64: fattn_volta_nomask::launch< 64>(p, n_batch, stream); break;
        case  72: fattn_volta_nomask::launch< 72>(p, n_batch, stream); break;
        case 128: fattn_volta_nomask::launch<128>(p, n_batch, stream); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());
}
