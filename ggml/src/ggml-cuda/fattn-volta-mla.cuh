#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// One token of MLA's absorbed attention on Volta tensor cores (mma.m8n8k4): K rows of 576 halves (the 512-dim latent
// and the 64-dim rope part), V the first 512 dims of the same rows, 8 Q heads on a K/V head. LongCat-Next under
// -sm tensor has exactly this per device: 8 of its 32 heads on the one latent head. The tile kernel does the same work
// on the FP16 FMA units at ~13% of their peak (compute/latency bound), well above the time to read the cache.
//
// A block of 8 warps takes the 8 heads and walks 32-cell tiles of its share of the cells; the shares of the parallel
// blocks are merged by flash_attn_combine_results (launch_fattn). Per tile:
//   1. The tile (32 x 576 halves) is stored to shared memory; the loads of the next tile are issued right after and
//      land in registers while this one is computed (Volta has no cp.async).
//   2. S = Q K^T (8 heads x 32 cells): warp w takes 72 of the 576 dims (64 of the latent, 8 of the rope part); quad
//      pair q takes half of them (q & 1) for 16 of the cells (q >> 1), in two rounds of 8, and keeps its Q fragments in
//      registers. The two halves are summed with a shuffle, the 8 warps in shared memory. Scores are F32.
//   3. Online softmax: warp w takes head w, a lane a cell. P (F16) and the rescale factor of each head go to shared
//      memory.
//   4. O += P V (8 heads x 512 dims, F32 accumulators): warp w owns dims 64w .. 64w + 63, 16 per quad pair.
// Two blocks of 8 warps fit an SM (shared memory, 128 registers a thread).
//
// mma.m8n8k4 fragments (quad pair = (lane % 16) / 4 does its own 8x8x4 product), as in fattn-volta-nomask.cuh:
//   A .row (8 x 4):  row (lane / 16) * 4 + lane % 4, k 0..3.
//   B .col (4 x 8):  column (lane / 16) * 4 + lane % 4, k 0..3.
//   B .row (4 x 8):  row lane % 4, columns (lane / 16) * 4 + 0..3.
//   C f32 (8 x 8):   element i: row (lane / 16) * 4 + lane % 2 + (i & 2), column (lane & 2) + (i & 4) + (i & 1).

namespace fattn_volta_mla {

constexpr int DKQ     = 576;
constexpr int DV      = 512;
constexpr int NH      = 8;                     // Q heads per block, the M of the MMAs
constexpr int NC      = 32;                    // cells per tile
constexpr int NW      = 8;                     // warps per block
constexpr int THREADS = NW*32;
constexpr int PITCH   = DKQ + 8;               // halves per shared K row: 4 banks apart, conflict-free fragment loads
constexpr int PIECES  = DKQ/8;                 // 16-byte pieces of a K row
constexpr int CHUNKS  = NC*PIECES/THREADS;     // pieces of a tile per thread
constexpr int SPITCH  = NC + 2;                // floats per (head, warp) row of the partial scores: conflict-free stores
constexpr int PPITCH  = NC + 8;                // halves per row of P
static_assert(NC*PIECES % THREADS == 0, "tile pieces must split evenly");

#ifdef VOLTA_MMA_AVAILABLE
static __device__ __forceinline__ void mma_row_col(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ void mma_row_row(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, {%0, %1, %2, %3, %4, %5, %6, %7};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ uint32_t pack_half2(const float lo, const float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&h);
}
#endif // VOLTA_MMA_AVAILABLE

// The template parameter only keeps the kernel out of translation units that never launch it.
template <int dummy>
__launch_bounds__(THREADS, 2)
static __global__ void flash_attn_volta_mla(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        const int  * __restrict__ KV_rows,
        const int  * __restrict__ KV_ranges,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int64_t nb32, const int64_t nb33) {
#ifdef VOLTA_MMA_AVAILABLE
    GGML_UNUSED_VARS(V, sinks, KV_rows, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, nb01, ne10, ne13,
        nb21, nb22, nb23, ne32, nb31, nb32);

    __shared__ __align__(16) half  tile_k[NC*PITCH];
    __shared__ __align__(16) float s_part[NW*NH*SPITCH];
    __shared__ __align__(16) half  p_tile[NH*PPITCH];
    __shared__ float alpha_s[NH];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*32 + lane;
    const int qp   = (lane % 16) / 4; // quad pair
    const int t    = lane % 4;
    const int hi   = lane / 16;

    // a block takes NH Q heads of one K/V head
    const int gqa_ratio    = ne02 / ne12;
    const int ntiles_z_gqa = gqa_ratio / NH;
    const int sequence     = blockIdx.z / (ne12*ntiles_z_gqa);
    const int z_KV         = blockIdx.z / ntiles_z_gqa - sequence*ne12;
    const int zt_gqa       = blockIdx.z % ntiles_z_gqa;
    const int head0        = z_KV*gqa_ratio + zt_gqa*NH;

    const char * K_rows = K + nb13*sequence + nb12*z_KV;
    const half * maskh  = (const half *) (mask + nb33*(sequence % ne33)); // one token: mask row 0, one mask head

    ggml_cuda_pdl_sync();

    // the KV range of the token, as in the tile kernel: its end, then a fully masked hole [lo, hi) to skip
    int k_end   = ne11;
    int hole_lo = 0;
    int hole_hi = 0;
    if (KV_ranges) {
        ggml_cuda_fattn_tile_range(KV_ranges, sequence % ne33, 0, 1, ne31, ne11, k_end, hole_lo, hole_hi);
    } else if (KV_max) {
        const int i_KV = sequence*gridDim.x + blockIdx.x;
        const int n_KV = ne03*gridDim.x;
        k_end   = KV_max[i_KV];
        hole_lo = KV_max[1*n_KV + i_KV];
        hole_hi = KV_max[2*n_KV + i_KV];
    }

    const int stride = gridDim.y*NC;
    const auto skip_hole = [&](const int k) {
        if (k < hole_lo || k >= hole_hi) {
            return k;
        }
        return k + (hole_hi - k + stride - 1) / stride * stride;
    };

    // Q as A fragments, scaled, F16: head t + 4*hi of the block, the 9 k-steps of this quad pair's half kh of the
    //     warp's dims (8 of the latent at 64*warp + 32*kh, 1 of the rope part at DV + 8*warp + 4*kh)
    const int kh   = qp & 1;
    const int d_lt = 64*warp + 32*kh;
    const int d_rp = DV + 8*warp + 4*kh;
    uint32_t qa[9][2];
    {
        const float * Qh = (const float *) (Q + nb03*sequence + nb02*(head0 + t + 4*hi));
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            const float4 x = *(const float4 *) (Qh + (j < 8 ? d_lt + 4*j : d_rp));
            qa[j][0] = pack_half2(x.x*scale, x.y*scale);
            qa[j][1] = pack_half2(x.z*scale, x.w*scale);
        }
    }

    // thread tid loads row tid/8 of a tile, 16-byte pieces tid%8 + 8*i: one base address, constant offsets, and the 8
    //     threads of a row read (and store) 128 contiguous bytes per piece
    static_assert(THREADS == 8*NC && CHUNKS*8 == PIECES, "one row per 8 threads");
    uint4 stage[CHUNKS];
    const int    ld_row = tid / 8;
    const int    ld_col = (tid % 8)*8; // halves
    half * const st_ptr = tile_k + ld_row*PITCH + ld_col;
    const auto load_tile = [&](const int k0) {
        const uint4 * src = (const uint4 *) (K_rows + int64_t(k0 + ld_row)*nb11 + ld_col*sizeof(half));
#pragma unroll
        for (int i = 0; i < CHUNKS; ++i) {
            stage[i] = src[8*i];
        }
    };
    const auto store_tile = [&]() {
#pragma unroll
        for (int i = 0; i < CHUNKS; ++i) {
            *(uint4 *) (st_ptr + 64*i) = stage[i];
        }
    };

    float o[2][8];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[jj][i] = 0.0f;
        }
    }
    // the softmax state of head `warp`, the same in every lane of the warp
    float m_run = -FLT_MAX/2.0f;
    float l_run = 0.0f;

    // B .col column n = t + 4*hi of round r of quad pair q is cell 4*(2*r + (q >> 1)) + t + 16*hi: the quad pairs
    //     of a 16-byte shared load read 4 rows 4 banks apart at dims 32 apart, of an 8-byte load rows 16 banks apart
    const half * k_row  = tile_k + (4*(qp >> 1) + t + 16*hi)*PITCH;
    // B .row: the 8 V dims of this thread, two output blocks of 4; the quad pairs of a 16-byte load are 32 dims apart
    const int    dims_o = 64*warp + 32*(qp & 1) + 8*(qp >> 1) + 16*hi;

    int k0 = skip_hole(blockIdx.y*NC);
    if (k0 < k_end) {
        load_tile(k0);
    }
    while (k0 < k_end) {
        store_tile();
        __syncthreads(); // the tile is visible

        const int k_next = skip_hole(k0 + stride);
        if (k_next < k_end) {
            load_tile(k_next);
        }

        // S = Q K^T over the half of the warp's dims of this quad pair, for its two groups of 8 cells
        float s[2][8];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s[r][i] = 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const uint4 kb = *(const uint4 *) (k_row + 8*r*PITCH + d_lt + 8*u);
                mma_row_col(s[r], qa[2*u][0],     qa[2*u][1],     kb.x, kb.y);
                mma_row_col(s[r], qa[2*u + 1][0], qa[2*u + 1][1], kb.z, kb.w);
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint2 kb = *(const uint2 *) (k_row + 8*r*PITCH + d_rp);
            mma_row_col(s[r], qa[8][0], qa[8][1], kb.x, kb.y);
        }
        // the other half of the dims is in lane ^ 4: half kh keeps round kh, so quad pair q ends with the cells of
        //     group 2*kh + (q >> 1)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float give = kh ? s[0][i] : s[1][i];
            const float keep = kh ? s[1][i] : s[0][i];
            s[0][i] = keep + __shfl_xor_sync(0xFFFFFFFF, give, 4);
        }
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            // C column n = (i & 4) + (t & 2) + (i & 1) is cell 4*group + (n & 3) + 16*(n >> 2)
            const int h = (t & 1) + (i & 2) + 4*hi;
            const int c = 4*(2*kh + (qp >> 1)) + (t & 2) + 16*((i >> 2) & 1);
            *(float2 *) (s_part + (h*NW + warp)*SPITCH + c) = make_float2(s[0][i], s[0][i + 1]);
        }
        __syncthreads(); // the partial scores of all warps are visible

        // online softmax of head `warp` over the 32 cells of the tile, a lane a cell
        {
            float x = __half2float(maskh[k0 + lane]);
#pragma unroll
            for (int w = 0; w < NW; ++w) {
                x += s_part[(warp*NW + w)*SPITCH + lane];
            }
            const float m_new = fmaxf(m_run, warp_reduce_max(x));
            const float alpha = expf(m_run - m_new);
            const float p     = expf(x - m_new);
            l_run = l_run*alpha + warp_reduce_sum(p);
            m_run = m_new;
            p_tile[warp*PPITCH + lane] = __float2half(p);
            if (lane == 0) {
                alpha_s[warp] = alpha;
            }
        }
        __syncthreads(); // P and the rescale factors are visible

        // O += P V: rescale the rows of this thread, then 8 steps of 4 cells
        {
            const float a0 = alpha_s[(t & 1) + 4*hi];
            const float a1 = alpha_s[(t & 1) + 4*hi + 2];
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    o[jj][i] *= (i & 2) ? a1 : a0;
                }
            }
#pragma unroll
            for (int s4 = 0; s4 < NC/4; ++s4) {
                const uint2 pa = *(const uint2 *) (p_tile + (t + 4*hi)*PPITCH + 4*s4);
                const uint4 v  = *(const uint4 *) (tile_k + (4*s4 + t)*PITCH + dims_o);
                mma_row_row(o[0], pa.x, pa.y, v.x, v.y);
                mma_row_row(o[1], pa.x, pa.y, v.z, v.w);
            }
        }
        __syncthreads(); // every warp is done with the tile

        k0 = k_next;
    }

    // the share of this block: unnormalized O and (max, sum) of each head, or the normalized O for one block
    if (lane == 0) {
        alpha_s[warp] = gridDim.y == 1 ? 1.0f/l_run : 1.0f;
        if (gridDim.y != 1) {
            const int j_dst = ((sequence*int(ne01.z))*ne02 + head0 + warp)*gridDim.y + blockIdx.y;
            dst_meta[j_dst] = make_float2(m_run, l_run);
        }
    }
    __syncthreads();

    // C element i of output block jj: head (t & 1) + (i & 2) + 4*hi, column n = (i & 4) + (t & 2) + (i & 1), which is
    //     the V dim dims_o + 4*jj + (n & 3) of the thread with hi = n >> 2
    const int dims_c = 64*warp + 32*(qp & 1) + 8*(qp >> 1) + (t & 2);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const int   h     = (t & 1) + (i & 2) + 4*hi;
        const float f     = alpha_s[h];
        const int   j_dst = ((sequence*int(ne01.z))*ne02 + head0 + h)*gridDim.y + blockIdx.y;
        float * dst_h = dst + int64_t(j_dst)*DV + dims_c + 16*((i >> 2) & 1);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            *(float2 *) (dst_h + 4*jj) = make_float2(o[jj][i]*f, o[jj][i + 1]*f);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, KV_rows, KV_ranges, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // VOLTA_MMA_AVAILABLE
}

} // namespace fattn_volta_mla

// The FLASH_ATTN_EXT cases the kernel above takes: Volta, one token, 576/512 with V a view of K, F16 cache, 8 Q heads on
//     each K/V head, a mask without a head dimension, no sinks, softcap or ALiBi. GGML_CUDA_FATTN_VOLTA_MLA = the fewest
//     cells for it, 0 = off.
static bool ggml_cuda_fattn_volta_mla_applies(const int cc, const ggml_tensor * dst) {
    static const int64_t min_kv = [] {
        const char * env = getenv("GGML_CUDA_FATTN_VOLTA_MLA");
        return env != nullptr ? (int64_t) atoll(env) : (int64_t) 1;
    }();
    if (min_kv <= 0 || !volta_mma_available(cc)) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (dst->src[4] || dst->src[5] || dst->src[7] || !mask || mask->ne[2] != 1 || mask->type != GGML_TYPE_F16) {
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
    if (Q->ne[0] != fattn_volta_mla::DKQ || K->ne[0] != fattn_volta_mla::DKQ || V->ne[0] != fattn_volta_mla::DV) {
        return false;
    }
    if (Q->ne[1] != 1 || Q->ne[2] != fattn_volta_mla::NH*K->ne[2] || K->ne[1] < min_kv || K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return false;
    }
    // V is read from the K tile
    if (V->data != K->data || V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2] || V->ne[3] != K->ne[3] ||
            V->nb[1] != K->nb[1] || V->nb[2] != K->nb[2] || V->nb[3] != K->nb[3]) {
        return false;
    }
    for (const ggml_tensor * t : {Q, K}) {
        if ((uintptr_t) t->data % 16 != 0 || t->nb[0] != ggml_type_size(t->type)) {
            return false;
        }
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }
    return K->ne[1]*K->nb[1] <= INT_MAX;
}

static void ggml_cuda_flash_attn_ext_volta_mla(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace fattn_volta_mla;
    fattn_kernel_t kernel = flash_attn_volta_mla<0>;

    // two blocks of ~46 KiB per SM need the largest shared memory carveout
    static bool carveout_set[GGML_CUDA_MAX_DEVICES] = {false};
    const int id = ggml_cuda_get_device();
    if (!carveout_set[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        carveout_set[id] = true;
    }

    launch_fattn<DV, 1, NH>(ctx, dst, kernel, NW, 0, NC, false, false, false, false, 32);
}
