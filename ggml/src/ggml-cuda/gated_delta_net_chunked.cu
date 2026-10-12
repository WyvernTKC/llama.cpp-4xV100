#include "gated_delta_net.cuh"
#include "mma.cuh"

// Chunked gated delta net for prefill (same math as build_delta_net_chunking), per chunk of C tokens with start state S:
//   gamma = cumsum(g), G = exp(gamma), A[r][s] = beta_r exp(gamma_r - gamma_s) k_r.k_s (s < r), T = (I + A)^-1
//   U = T beta V, W = T beta G K, Delta = U - W S
//   O = scale (G Q S + (Q K^T exp(gamma_r - gamma_s))_{s <= r} Delta)
//   S = G_C S + (exp(gamma_C - gamma) K)^T Delta
// The prep kernel computes U, W, P and K~^T for all chunks in parallel, the scan kernel steps over the chunks and keeps
//   the start state of each chunk and Delta, the out kernel computes O for all chunks in parallel.
// f16 is used only for the tensor core operands, everything that accumulates is fp32.

using namespace ggml_cuda_mma;

typedef tile<32, 4, half2>                               gdn_tile_A;  // 32 rows x 8 k
typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> gdn_tile_B;  // 8 cols x 8 k, memory [n][k]
typedef tile< 8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> gdn_tile_Bt; // 8 k x 8 cols, memory [k][n]
typedef tile<32, 8, float>                               gdn_tile_C;

static constexpr int gdn_ch_D = 128; // S_k == S_v
static constexpr int gdn_ch_C = 64;  // tokens per chunk

// scratch of one chunk: U f32 [C][D], W f16 [C][D], Q f16 [C][D], K~^T f16 [D][C], P f16 [C][C], G f32 [C]; Q and P are stored
//   as A fragments: 16 B per lane, the 32 lanes of a fragment (32 rows x 8 columns) next to each other, and so is K~^T
static constexpr size_t gdn_ch_off_W  = gdn_ch_C*gdn_ch_D*sizeof(float);
static constexpr size_t gdn_ch_off_Q  = gdn_ch_off_W + gdn_ch_C*gdn_ch_D*sizeof(half);
static constexpr size_t gdn_ch_off_K  = gdn_ch_off_Q + gdn_ch_C*gdn_ch_D*sizeof(half);
static constexpr size_t gdn_ch_off_P  = gdn_ch_off_K + gdn_ch_D*gdn_ch_C*sizeof(half);
static constexpr size_t gdn_ch_off_G  = gdn_ch_off_P + gdn_ch_C*gdn_ch_C*sizeof(half);
static constexpr size_t gdn_ch_off_H  = gdn_ch_off_G + gdn_ch_C*sizeof(float);        // the scan: start state f16 [k][v]
static constexpr size_t gdn_ch_off_DL = gdn_ch_off_H + gdn_ch_D*gdn_ch_D*sizeof(half); //           Delta f16 [C][D]
static constexpr size_t gdn_ch_rec    = gdn_ch_off_DL + gdn_ch_C*gdn_ch_D*sizeof(half);

// padded row strides: rows of 16 B loads start on different banks
static constexpr int gdn_ch_SH = gdn_ch_D/2 + 4; // half2, rows of D
static constexpr int gdn_ch_SC = gdn_ch_C/2 + 4; // half2, rows of C
static constexpr int gdn_ch_SA = gdn_ch_C + 4;   // float, rows of A and T

static constexpr int gdn_ch_prep_threads = 512;
static constexpr size_t gdn_ch_prep_smem =
    2*gdn_ch_C*gdn_ch_SH*sizeof(half2) + gdn_ch_D*gdn_ch_SC*sizeof(half2) + gdn_ch_C*gdn_ch_SA*sizeof(float) + 4*gdn_ch_C*sizeof(float);
static_assert(gdn_ch_D*gdn_ch_SC*sizeof(half2) >= gdn_ch_C*gdn_ch_SA*sizeof(float), "K~^T region holds A");
static_assert(2*gdn_ch_C*gdn_ch_SH*sizeof(half2) >= gdn_ch_C*(gdn_ch_D + 4)*sizeof(float), "K and Q regions hold U");

template <int BV>
static constexpr size_t gdn_ch_scan_smem() {
    return (gdn_ch_C*gdn_ch_SH + (gdn_ch_D + gdn_ch_C)*(BV/2 + 4))*sizeof(half2) +
        (gdn_ch_C*(BV + 4) + 4)*sizeof(float);
}

static __global__ void __launch_bounds__(gdn_ch_prep_threads, 1) gated_delta_net_chunk_prep(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta, char * __restrict__ scratch,
        const int64_t H, const int64_t n_tokens,
        const int64_t sq1, const int64_t sq2, const int64_t sq3, const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int64_t sb1, const int64_t sb2, const int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic, const float scale) {
#if defined(VOLTA_MMA_AVAILABLE)
    constexpr int C  = gdn_ch_C;
    constexpr int D  = gdn_ch_D;
    constexpr int SH = gdn_ch_SH;
    constexpr int SC = gdn_ch_SC;
    constexpr int SA = gdn_ch_SA;
    constexpr int nt = gdn_ch_prep_threads;
    constexpr int nw = nt/WARP_SIZE;
    static_assert(nw >= 14, "warps 0-11 make K K^T, then 0-1 the diagonal blocks of T and 2-13 Q K^T");

    extern __shared__ float4 gdn_ch_smem_prep[];
    half2 * sK    = (half2 *) gdn_ch_smem_prep;  // [C][SH] K
    half2 * sQV   = sK + C*SH;                   // [C][SH] Q, then V
    half2 * sKt   = sQV + C*SH;                  // [D][SC] K~^T, then A as f32 [C][SA]
    float * sA    = (float *) sKt;
    float * sT    = (float *) (sKt + D*SC);      // [C][SA] T
    float * sgam  = sT + C*SA;                   // [C] cumsum of g
    float * sbeta = sgam + C;                    // [C] beta
    float * sbg   = sbeta + C;                   // [C] beta G
    float * sG    = sbg + C;                     // [C] G

    const int c    = blockIdx.x;
    const int h    = blockIdx.y;
    const int seq  = blockIdx.z;
    const int lane = threadIdx.x; // mma.cuh takes the lane from threadIdx.x
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;

    const int64_t t0 = int64_t(c)*C;
    const int     L  = (int) min(int64_t(C), n_tokens - t0); // the tail of the last chunk is zero

    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const uint32_t iq3 = fastdiv(seq, rq3_magic);

    const float * q_c = q + iq3*sq3 + t0*sq2 + iq1*sq1;
    const float * k_c = k + iq3*sq3 + t0*sq2 + iq1*sq1;
    const float * v_c = v + seq*sv3 + t0*sv2 + h*sv1;
    const int64_t gb0 = seq*sb3 + t0*sb2 + h*sb1;

    char * rec = scratch + ((int64_t(seq)*H + h)*gridDim.x + c)*gdn_ch_rec;

    // the gates first, so that their cumsum runs while K and Q arrive
    float gv = 0.0f;
    float bv = 0.0f;
    if (tid < L) {
        gv = g   [gb0 + tid*sb2];
        bv = beta[gb0 + tid*sb2];
    }

    // K and Q rows, all loads first; 8 lanes per row, 4 rows per warp (the transposed store of K~ below has 4-way conflicts)
    constexpr int nkq = C*D/4/nt;
    float4 kf[nkq];
    float4 qf[nkq];
    const auto kq_row = [&](const int e) { return (e / 8) % C; };
    const auto kq_c4  = [&](const int e) { return e % 8 + 8*(e / (8*C)); };
#pragma unroll
    for (int i = 0; i < nkq; ++i) {
        const int r  = min(kq_row(tid + i*nt), L - 1);
        const int c4 = kq_c4(tid + i*nt);
        kf[i] = *(const float4 *) (k_c + r*sq2 + 4*c4);
        qf[i] = *(const float4 *) (q_c + r*sq2 + 4*c4);
    }

    // cumsum of the log gates over the chunk
    if (tid < C) {
#pragma unroll
        for (int off = 1; off < WARP_SIZE; off *= 2) {
            const float y = __shfl_up_sync(0xFFFFFFFF, gv, off, WARP_SIZE);
            if (lane >= off) {
                gv += y;
            }
        }
        sgam[tid]  = gv;
        sbeta[tid] = bv;
    }
    __syncthreads();
    if (tid < C) {
        const float gam = sgam[tid] + (tid >= WARP_SIZE ? sgam[WARP_SIZE - 1] : 0.0f);
        const float G   = expf(gam);
        if (tid >= WARP_SIZE) {
            sgam[tid] = gam; // the first warp has its final values
        }
        sG[tid]  = G;
        sbg[tid] = sbeta[tid]*G;
    }
    __syncthreads();

    // K and Q as f16, and K~^T = (exp(gamma_C - gamma) K)^T
    {
        const float gam_C = sgam[C - 1];
        half * sKt_h = (half *) sKt;
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < nkq; ++i) {
            const int r  = kq_row(tid + i*nt);
            const int c4 = kq_c4(tid + i*nt);
            const float4 kv = r < L ? kf[i] : zero;
            const float4 qv = r < L ? qf[i] : zero;
            sK [r*SH + 2*c4 + 0] = make_half2(kv.x, kv.y);
            sK [r*SH + 2*c4 + 1] = make_half2(kv.z, kv.w);
            sQV[r*SH + 2*c4 + 0] = make_half2(qv.x, qv.y);
            sQV[r*SH + 2*c4 + 1] = make_half2(qv.z, qv.w);
            const float dk = expf(gam_C - sgam[r]);
            sKt_h[(4*c4 + 0)*2*SC + r] = __float2half(dk*kv.x);
            sKt_h[(4*c4 + 1)*2*SC + r] = __float2half(dk*kv.y);
            sKt_h[(4*c4 + 2)*2*SC + r] = __float2half(dk*kv.z);
            sKt_h[(4*c4 + 3)*2*SC + r] = __float2half(dk*kv.w);
        }
    }
    // V, which goes into the Q buffer after Q K^T
    constexpr int nv = C*D/4/nt;
    float4 vf[nv];
#pragma unroll
    for (int i = 0; i < nv; ++i) {
        const int r  = min((tid + i*nt) / (D/4), L - 1);
        const int c4 = (tid + i*nt) % (D/4);
        vf[i] = *(const float4 *) (v_c + r*sv2 + 4*c4);
    }
    __syncthreads();
    {
        int4 * gQ = (int4 *) (rec + gdn_ch_off_Q);
        int4 * gK = (int4 *) (rec + gdn_ch_off_K);
        for (int u = tid; u < C*D/8; u += nt) {
            gQ[u] = *(const int4 *) (sQV + (32*(u / (32*D/8)) + u % 32)*SH + 4*((u / 32) % (D/8)));
            gK[u] = *(const int4 *) (sKt + (32*(u / (32*C/8)) + u % 32)*SC + 4*((u / 32) % (C/8)));
        }
    }
    __syncthreads(); // A overwrites K~^T

    // K K^T and Q K^T: the 12 tiles of 32x8 on or below the diagonal
    const auto product = [&](const half2 * src_A, const int p) {
        const int rt = p < 4 ? 0 : 1;
        const int ct = p < 4 ? p : p - 4;
        gdn_tile_C acc;
#pragma unroll
        for (int ks = 0; ks < D/8; ++ks) {
            gdn_tile_A a;
            gdn_tile_B b;
            load_ldmatrix(a, src_A + rt*32*SH + 4*ks, SH);
            load_ldmatrix(b, sK    + ct*8*SH  + 4*ks, SH);
            mma(acc, a, b);
        }
        return acc;
    };
    // A (f32, smem), a tile per warp
    if (warp < 12) {
        const gdn_tile_C acc = product(sK, warp);
#pragma unroll
        for (int l = 0; l < gdn_tile_C::ne; l += 2) {
            const int   r  = (warp < 4 ? 0 : 32) + gdn_tile_C::get_i(l);
            const int   s  = (warp < 4 ? warp : warp - 4)*8 + gdn_tile_C::get_j(l);
            const float gr = sgam[r];
            sA[r*SA + s + 0] = s     < r ? sbeta[r]*expf(gr - sgam[s    ])*acc.x[l + 0] : 0.0f;
            sA[r*SA + s + 1] = s + 1 < r ? sbeta[r]*expf(gr - sgam[s + 1])*acc.x[l + 1] : 0.0f;
        }
    }
    __syncthreads();

    // T = (I + A)^-1 by blocks of 16. The diagonal blocks by forward substitution in the first two warps, thread j of
    //   block b owns column j; meanwhile Q K^T -> P (f16, scratch) in the next 12 warps
    half2 * gP = (half2 *) (rec + gdn_ch_off_P);
    if (tid < C) {
        const int b0 = 16*(tid / 16);
        const int j  = tid % 16;
        float tc[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            tc[r] = r == j ? 1.0f : 0.0f;
        }
#pragma unroll
        for (int r = 1; r < 16; ++r) {
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < r; ++s) {
                acc += sA[(b0 + r)*SA + b0 + s]*tc[s];
            }
            tc[r] = r > j ? -acc : tc[r];
        }
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            sT[(b0 + r)*SA + b0 + j] = tc[r];
        }
    } else if (warp < 14) {
        const int p = warp - 2;
        const gdn_tile_C acc = product(sQV, p);
#pragma unroll
        for (int l = 0; l < gdn_tile_C::ne; l += 2) {
            const int   r  = (p < 4 ? 0 : 32) + gdn_tile_C::get_i(l);
            const int   s  = (p < 4 ? p : p - 4)*8 + gdn_tile_C::get_j(l);
            const float gr = sgam[r];
            const float p0 = s     <= r ? scale*expf(gr - sgam[s    ])*acc.x[l + 0] : 0.0f;
            const float p1 = s + 1 <= r ? scale*expf(gr - sgam[s + 1])*acc.x[l + 1] : 0.0f;
            gP[(((r/32)*(C/8) + s/8)*32 + r % 32)*4 + (s % 8)/2] = make_half2(p0, p1);
        }
    }
    for (int e = tid; e < 32*16; e += nt) {
        gP[(C/16)*32*4 + e] = make_half2(0.0f, 0.0f); // the fragments of rows 0..31 above the diagonal
    }
    // the blocks above the diagonal are zero
    for (int e = tid; e < 6*256; e += nt) {
        const int b  = e / 256;
        const int bi = b < 3 ? 0 : (b < 5 ? 1 : 2);
        const int bk = b < 3 ? b + 1 : (b < 5 ? b - 1 : 3);
        sT[(16*bi + (e % 256) / 16)*SA + 16*bk + e % 16] = 0.0f;
    }
    __syncthreads();
    // V into the Q buffer; the next barrier is in the first stage below
#pragma unroll
    for (int i = 0; i < nv; ++i) {
        const int r  = (tid + i*nt) / (D/4);
        const int c4 = (tid + i*nt) % (D/4);
        const float4 vv = r < L ? vf[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        sQV[r*SH + 2*c4 + 0] = make_half2(vv.x, vv.y);
        sQV[r*SH + 2*c4 + 1] = make_half2(vv.z, vv.w);
    }

    // the blocks below the diagonal by distance d: T_ij = -T_ii sum_{k=j}^{i-1} A_ik T_kj, i = j + d
#pragma unroll
    for (int d = 1; d < 4; ++d) {
        constexpr int nm = (3*256 + nt - 1)/nt;
        float y[nm];
#pragma unroll
        for (int m = 0; m < nm; ++m) {
            const int e = tid + m*nt;
            if (e < 256*(4 - d)) {
                const int jb = e / 256;
                const int ib = jb + d;
                const int r  = (e % 256) / 16;
                const int cc = e % 16;
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int kb = jb; kb < ib; ++kb) {
#pragma unroll
                    for (int s = 0; s < 16; ++s) {
                        acc[s % 4] += sA[(16*ib + r)*SA + 16*kb + s]*sT[(16*kb + s)*SA + 16*jb + cc];
                    }
                }
                y[m] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            }
        }
#pragma unroll
        for (int m = 0; m < nm; ++m) {
            const int e = tid + m*nt;
            if (e < 256*(4 - d)) {
                const int jb = e / 256;
                sT[(16*(jb + d) + (e % 256) / 16)*SA + 16*jb + e % 16] = y[m];
            }
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < nm; ++m) {
            const int e = tid + m*nt;
            if (e < 256*(4 - d)) {
                const int jb = e / 256;
                const int ib = jb + d;
                const int r  = (e % 256) / 16;
                const int cc = e % 16;
                float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int s = 0; s < 16; ++s) { // T_ii is 0 above the diagonal
                    acc[s % 4] += sT[(16*ib + r)*SA + 16*ib + s]*sT[(16*ib + s)*SA + 16*jb + cc];
                }
                y[m] = -((acc[0] + acc[1]) + (acc[2] + acc[3]));
            }
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < nm; ++m) {
            const int e = tid + m*nt;
            if (e < 256*(4 - d)) {
                const int jb = e / 256;
                sT[(16*(jb + d) + (e % 256) / 16)*SA + 16*jb + e % 16] = y[m];
            }
        }
        __syncthreads();
    }

    // U = (T beta) V and W = (T beta G) K, warp: one row tile, nu column tiles of each; T is lower triangular
    {
        constexpr int nu = 32/nw;
        const int rt  = warp / (nw/2);
        const int ct0 = (warp % (nw/2))*nu;
        gdn_tile_C accU[nu];
        gdn_tile_C accW[nu];
#pragma unroll
        for (int ks = 0; ks < C/8; ++ks) {
            if (rt == 0 && ks >= C/16) {
                break;
            }
            const float * Tr = sT + (rt*32 + lane)*SA + 8*ks;
            const float4 ta = *(const float4 *) (Tr + 0);
            const float4 tb = *(const float4 *) (Tr + 4);
            const float  t8[8] = {ta.x, ta.y, ta.z, ta.w, tb.x, tb.y, tb.z, tb.w};
            gdn_tile_A aU;
            gdn_tile_A aW;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int s = 8*ks + 2*m;
                aU.x[m] = make_half2(t8[2*m]*sbeta[s], t8[2*m + 1]*sbeta[s + 1]);
                aW.x[m] = make_half2(t8[2*m]*sbg  [s], t8[2*m + 1]*sbg  [s + 1]);
            }
#pragma unroll
            for (int n = 0; n < nu; ++n) {
                gdn_tile_Bt bV;
                gdn_tile_Bt bK;
                load_ldmatrix(bV, sQV + 8*ks*SH + 4*(ct0 + n), SH);
                load_ldmatrix(bK, sK  + 8*ks*SH + 4*(ct0 + n), SH);
                mma(accU[n], aU, bV);
                mma(accW[n], aW, bK);
            }
        }
        // through shared memory, for whole rows of U and W
        constexpr int SUW = D + 4;
        float * sUo = (float *) sK;  // [C][SUW] over K and V
        half2 * sWo = sKt;           // [C][SH]
        __syncthreads();
#pragma unroll
        for (int n = 0; n < nu; ++n) {
#pragma unroll
            for (int l = 0; l < gdn_tile_C::ne; l += 2) {
                const int r   = rt*32 + gdn_tile_C::get_i(l);
                const int col = (ct0 + n)*8 + gdn_tile_C::get_j(l);
                *(float2 *) (sUo + r*SUW + col) = make_float2(accU[n].x[l], accU[n].x[l + 1]);
                sWo[r*SH + col/2] = make_half2(accW[n].x[l], accW[n].x[l + 1]);
            }
        }
        __syncthreads();
        for (int u = tid; u < C*D/4; u += nt) {
            ((float4 *) rec)[u] = *(const float4 *) (sUo + (u / (D/4))*SUW + 4*(u % (D/4)));
        }
        for (int u = tid; u < C*D/8; u += nt) {
            ((int4 *) (rec + gdn_ch_off_W))[u] = *(const int4 *) (sWo + (u / (D/8))*SH + 4*(u % (D/8)));
        }
    }
    if (tid < C) {
        ((float *) (rec + gdn_ch_off_G))[tid] = sG[tid];
    }
#else
    GGML_UNUSED_VARS(q, k, v, g, beta, scratch, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
        neqk1_magic, rq3_magic, scale);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// one block owns BV columns of the state of one head, in f32 accumulator tiles, and steps over the chunks.
// It writes the state at the start of each chunk and Delta, the output is computed by gated_delta_net_chunk_out.
template <int BV>
static __global__ void __launch_bounds__(BV*8, 1) gated_delta_net_chunk_scan(
        char * __restrict__ scratch, const float * __restrict__ s, float * __restrict__ state,
        const int64_t H, const int n_chunks, const int32_t * __restrict__ ids, const int64_t s_row_stride, float * __restrict__ s_pre) {
#if defined(VOLTA_MMA_AVAILABLE)
    constexpr int C  = gdn_ch_C;
    constexpr int D  = gdn_ch_D;
    constexpr int SH = gdn_ch_SH;
    constexpr int SS = BV/2 + 4; // half2, rows of BV
    constexpr int SU = BV + 4;   // float, rows of BV
    constexpr int nt = BV*8;     // 4 state rows of 32 x BV/8 columns of 8 = 2 tiles per warp
    static_assert(BV == 16 || BV == 32, "bad BV");

    extern __shared__ float4 gdn_ch_smem_scan[];
    half2 * sW  = (half2 *) gdn_ch_smem_scan; // [C][SH]
    half2 * sS  = sW  + C*SH;                 // [D][SS] the state as f16
    half2 * sD  = sS  + D*SS;                 // [C][SS] Delta as f16
    float * sU  = (float *) (sD + C*SS);      // [C][SU]
    float * sGC = sU + C*SU;                  // G at the end of the chunk

    const int v0   = blockIdx.x*BV;
    const int h    = blockIdx.y;
    const int seq  = blockIdx.z;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + threadIdx.x;

    const int rto = warp % 2;              // W S tile: rows of the chunk
    const int cto = warp / 2;              //           columns of the slice
    const int rts = warp / (BV/16);        // state tiles: rows of the state
    const int cts = 2*(warp % (BV/16));    //              first of two columns

    const int64_t state_off = (int64_t(seq)*H + h)*D*D;
    const float * S0 = ids ? s + int64_t(ids[seq])*s_row_stride + int64_t(h)*D*D : s + state_off;

    // the next chunk is loaded into registers while the current one is computed; K~^T as the A fragments of this warp
    constexpr int nDD = C*D/8/nt;  // 16 B pieces per thread of W
    constexpr int nUU = C*BV/4/nt; // of the U slice
    int4   pW[nDD];
    int4   pK[C/8];
    int4   cK[C/8];
    float4 pU[nUU];
    float  pGC = 0.0f;
    const auto load_chunk = [&](const char * rec) {
#pragma unroll
        for (int i = 0; i < nDD; ++i) {
            pW[i] = ((const int4 *) (rec + gdn_ch_off_W))[tid + i*nt];
        }
#pragma unroll
        for (int ks = 0; ks < C/8; ++ks) {
            pK[ks] = ((const int4 *) (rec + gdn_ch_off_K))[(rts*(C/8) + ks)*WARP_SIZE + threadIdx.x];
        }
#pragma unroll
        for (int i = 0; i < nUU; ++i) {
            const int u = tid + i*nt;
            pU[i] = ((const float4 *) rec)[(u / (BV/4))*(D/4) + v0/4 + u % (BV/4)];
        }
        if (tid == 0) {
            pGC = ((const float *) (rec + gdn_ch_off_G))[C - 1];
        }
    };
    const auto store_chunk = [&]() {
#pragma unroll
        for (int i = 0; i < nDD; ++i) {
            const int u = tid + i*nt;
            *(int4 *) (sW  + (u / (D/8))*SH + 4*(u % (D/8))) = pW[i];
        }
#pragma unroll
        for (int ks = 0; ks < C/8; ++ks) {
            cK[ks] = pK[ks];
        }
#pragma unroll
        for (int i = 0; i < nUU; ++i) {
            const int u = tid + i*nt;
            *(float4 *) (sU + (u / (BV/4))*SU + 4*(u % (BV/4))) = pU[i];
        }
        if (tid == 0) {
            *sGC = pGC;
        }
    };

    char * rec = scratch + (int64_t(seq)*H + h)*n_chunks*gdn_ch_rec;
    load_chunk(rec);

    // the state is stored transposed: element [k][v] at v*D + k
    gdn_tile_C accS[2];
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int l = 0; l < gdn_tile_C::ne; ++l) {
            const int kk = rts*32 + gdn_tile_C::get_i(l);
            const int vv = v0 + (cts + n)*8 + gdn_tile_C::get_j(l);
            accS[n].x[l] = S0[vv*D + kk];
        }
    }
    if (s_pre) {
        for (int e = tid; e < BV*D; e += nt) {
            s_pre[state_off + int64_t(v0)*D + e] = S0[int64_t(v0)*D + e];
        }
    }

    for (int c = 0; c < n_chunks; ++c) {
        // f16 copy of the state: the operand of W S, and the start state of the chunk for the output
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int l = 0; l < gdn_tile_C::ne; l += 2) {
                const int kk = rts*32 + gdn_tile_C::get_i(l);
                const int j  = (cts + n)*8 + gdn_tile_C::get_j(l);
                sS[kk*SS + j/2] = make_half2(accS[n].x[l], accS[n].x[l + 1]);
            }
        }
        store_chunk();
        __syncthreads();
        char * rec_c = rec;
        if (c + 1 < n_chunks) {
            rec += gdn_ch_rec;
            load_chunk(rec);
        }
        for (int u = tid; u < D*BV/8; u += nt) {
            const int kk = u / (BV/8);
            const int j8 = u % (BV/8);
            ((int4 *) (rec_c + gdn_ch_off_H))[kk*(D/8) + v0/8 + j8] = *(const int4 *) (sS + kk*SS + 4*j8);
        }

        // Delta = U - W S, in two chains of mma
        gdn_tile_C accW[2];
#pragma unroll
        for (int ks = 0; ks < D/8; ++ks) {
            gdn_tile_A  a;
            gdn_tile_Bt b;
            load_ldmatrix(a, sW + rto*32*SH + 4*ks, SH);
            load_ldmatrix(b, sS + 8*ks*SS   + 4*cto, SS);
            mma(accW[ks % 2], a, b);
        }
#pragma unroll
        for (int l = 0; l < gdn_tile_C::ne; l += 2) {
            const int r = rto*32 + gdn_tile_C::get_i(l);
            const int j = cto*8  + gdn_tile_C::get_j(l);
            const float2 u = *(const float2 *) (sU + r*SU + j);
            sD[r*SS + j/2] = make_half2(u.x - (accW[0].x[l] + accW[1].x[l]), u.y - (accW[0].x[l + 1] + accW[1].x[l + 1]));
        }
        __syncthreads();
        for (int u = tid; u < C*BV/8; u += nt) {
            const int r  = u / (BV/8);
            const int j8 = u % (BV/8);
            ((int4 *) (rec_c + gdn_ch_off_DL))[r*(D/8) + v0/8 + j8] = *(const int4 *) (sD + r*SS + 4*j8);
        }

        // S = G_C S + K~^T Delta
        const float gC = *sGC;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int l = 0; l < gdn_tile_C::ne; ++l) {
                accS[n].x[l] *= gC;
            }
        }
        gdn_tile_C accS1[2]; // the odd k steps, a second chain
#pragma unroll
        for (int ks = 0; ks < C/8; ++ks) {
            gdn_tile_A a;
            memcpy(a.x, &cK[ks], sizeof(int4));
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                gdn_tile_Bt b;
                load_ldmatrix(b, sD + 8*ks*SS + 4*(cts + n), SS);
                mma(ks % 2 ? accS1[n] : accS[n], a, b);
            }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int l = 0; l < gdn_tile_C::ne; ++l) {
                accS[n].x[l] += accS1[n].x[l];
            }
        }
        __syncthreads();
    }

    float * S1 = state + state_off;
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int l = 0; l < gdn_tile_C::ne; ++l) {
            const int kk = rts*32 + gdn_tile_C::get_i(l);
            const int vv = v0 + (cts + n)*8 + gdn_tile_C::get_j(l);
            S1[vv*D + kk] = accS[n].x[l];
        }
    }
#else
    GGML_UNUSED_VARS(scratch, s, state, H, n_chunks, ids, s_row_stride, s_pre);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

// O = scale G Q S + P Delta for each chunk, S being the state at the start of the chunk; a block makes half of the columns
static constexpr int gdn_ch_out_threads = 128;
static constexpr int gdn_ch_out_SV      = gdn_ch_D/4 + 4; // half2, rows of D/2
static constexpr size_t gdn_ch_out_smem = (gdn_ch_D + gdn_ch_C)*gdn_ch_out_SV*sizeof(half2) + gdn_ch_C*sizeof(float);

static __global__ void __launch_bounds__(gdn_ch_out_threads, 3) gated_delta_net_chunk_out(
        const char * __restrict__ scratch, float * __restrict__ dst, const int64_t H, const int64_t n_tokens, const float scale) {
#if defined(VOLTA_MMA_AVAILABLE)
    constexpr int C  = gdn_ch_C;
    constexpr int D  = gdn_ch_D;
    constexpr int SV = gdn_ch_out_SV;
    constexpr int nt = gdn_ch_out_threads;

    extern __shared__ float4 gdn_ch_smem_out[];
    half2 * sH = (half2 *) gdn_ch_smem_out; // [D][SV] the state, columns v0..v0+D/2
    half2 * sD = sH + D*SV;                 // [C][SV] Delta
    float * sG = (float *) (sD + C*SV);     // [C]

    const int c    = blockIdx.x / 2;
    const int v0   = (blockIdx.x % 2)*(D/2);
    const int h    = blockIdx.y;
    const int seq  = blockIdx.z;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;

    const char * rec = scratch + ((int64_t(seq)*H + h)*(gridDim.x/2) + c)*gdn_ch_rec;
    for (int u = tid; u < D*D/16; u += nt) {
        const int r = u / (D/16);
        const int c8 = u % (D/16);
        *(int4 *) (sH + r*SV + 4*c8) = ((const int4 *) (rec + gdn_ch_off_H))[r*(D/8) + v0/8 + c8];
    }
    for (int u = tid; u < C*D/16; u += nt) {
        const int r = u / (D/16);
        const int c8 = u % (D/16);
        *(int4 *) (sD + r*SV + 4*c8) = ((const int4 *) (rec + gdn_ch_off_DL))[r*(D/8) + v0/8 + c8];
    }
    if (tid < C) {
        sG[tid] = ((const float *) (rec + gdn_ch_off_G))[tid];
    }
    __syncthreads();

    // warp: one row tile, 4 column tiles; the A operands Q and P are read from the scratch directly
    const int rt  = warp / 2;
    const int ct0 = (warp % 2)*4;
    const half2 * gQ = (const half2 *) (rec + gdn_ch_off_Q) + rt*(D/8)*32*4;
    const half2 * gP = (const half2 *) (rec + gdn_ch_off_P) + rt*(C/8)*32*4;
    gdn_tile_C acc[4];
#pragma unroll
    for (int ks = 0; ks < D/8; ++ks) {
        gdn_tile_A a;
        load_ldmatrix(a, gQ + ks*32*4, 4);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            gdn_tile_Bt b;
            load_ldmatrix(b, sH + 8*ks*SV + 4*(ct0 + n), SV);
            mma(acc[n], a, b);
        }
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int l = 0; l < gdn_tile_C::ne; ++l) {
            acc[n].x[l] *= scale*sG[rt*32 + gdn_tile_C::get_i(l)];
        }
    }
#pragma unroll
    for (int ks = 0; ks < C/8; ++ks) {
        if (rt == 0 && ks >= C/16) {
            break; // P is lower triangular
        }
        gdn_tile_A a;
        load_ldmatrix(a, gP + ks*32*4, 4);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            gdn_tile_Bt b;
            load_ldmatrix(b, sD + 8*ks*SV + 4*(ct0 + n), SV);
            mma(acc[n], a, b);
        }
    }

    // through shared memory, for whole rows of the output
    constexpr int SO = D/2 + 4;
    float * sO = (float *) gdn_ch_smem_out; // [C][SO]
    __syncthreads();
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int l = 0; l < gdn_tile_C::ne; l += 2) {
            const int r = rt*32 + gdn_tile_C::get_i(l);
            const int j = (ct0 + n)*8 + gdn_tile_C::get_j(l);
            *(float2 *) (sO + r*SO + j) = make_float2(acc[n].x[l], acc[n].x[l + 1]);
        }
    }
    __syncthreads();
    float * attn = dst + (int64_t(seq)*n_tokens*H + h)*D + v0;
    for (int u = tid; u < C*D/8; u += nt) {
        const int     r  = u / (D/8);
        const int     c4 = u % (D/8);
        const int64_t t  = int64_t(c)*C + r;
        if (t < n_tokens) {
            *(float4 *) (attn + t*D*H + 4*c4) = *(const float4 *) (sO + r*SO + 4*c4);
        }
    }
#else
    GGML_UNUSED_VARS(scratch, dst, H, n_tokens, scale);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE)
}

bool ggml_cuda_gated_delta_net_chunked_supported(const int device, const int64_t S_v, const bool kda, const int64_t n_tokens) {
    const ggml_cuda_device_info::cuda_device_info & info = ggml_cuda_info().devices[device];
    return volta_mma_available(info.cc) && S_v == gdn_ch_D && !kda && n_tokens >= 2*gdn_ch_C &&
        info.smpbo >= std::max(std::max(gdn_ch_prep_smem, gdn_ch_scan_smem<32>()), gdn_ch_out_smem);
}

size_t ggml_cuda_gated_delta_net_chunked_scratch_size(const int64_t H, const int64_t n_tokens, const int64_t n_seqs) {
    return n_seqs*H*((n_tokens + gdn_ch_C - 1)/gdn_ch_C)*gdn_ch_rec;
}

template <int BV>
static void gated_delta_net_chunk_scan_launch(const ggml_cuda_gated_delta_net_chunked_args & a, char * scratch, const int n_chunks,
        cudaStream_t stream) {
    constexpr size_t smem = gdn_ch_scan_smem<BV>();
    CUDA_SET_SHARED_MEMORY_LIMIT(gated_delta_net_chunk_scan<BV>, smem);
    const dim3 grid(gdn_ch_D/BV, a.H, a.n_seqs);
    gated_delta_net_chunk_scan<BV><<<grid, dim3(WARP_SIZE, BV/4), smem, stream>>>(
        scratch, a.s, a.state, a.H, n_chunks, a.ids, a.s_row_stride, a.s_pre);
}

void ggml_cuda_gated_delta_net_chunked(const ggml_cuda_gated_delta_net_chunked_args & a, char * scratch, cudaStream_t stream) {
    const int  n_chunks = (a.n_tokens + gdn_ch_C - 1)/gdn_ch_C;
    const dim3 grid_chunks(n_chunks, a.H, a.n_seqs);

    CUDA_SET_SHARED_MEMORY_LIMIT(gated_delta_net_chunk_prep, gdn_ch_prep_smem);
    gated_delta_net_chunk_prep<<<grid_chunks, dim3(WARP_SIZE, gdn_ch_prep_threads/WARP_SIZE), gdn_ch_prep_smem, stream>>>(
        a.q, a.k, a.v, a.g, a.beta, scratch, a.H, a.n_tokens, a.sq1, a.sq2, a.sq3, a.sv1, a.sv2, a.sv3, a.sb1, a.sb2, a.sb3,
        init_fastdiv_values(a.neqk1), init_fastdiv_values(a.rq3), a.scale);

    // the scan is sequential over the chunks: the narrower slices keep the grid within one wave of SMs
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    if (a.H*a.n_seqs*(gdn_ch_D/16) <= nsm) {
        gated_delta_net_chunk_scan_launch<16>(a, scratch, n_chunks, stream);
    } else {
        gated_delta_net_chunk_scan_launch<32>(a, scratch, n_chunks, stream);
    }

    CUDA_SET_SHARED_MEMORY_LIMIT(gated_delta_net_chunk_out, gdn_ch_out_smem);
    gated_delta_net_chunk_out<<<dim3(2*n_chunks, a.H, a.n_seqs), dim3(WARP_SIZE, gdn_ch_out_threads/WARP_SIZE), gdn_ch_out_smem, stream>>>(
        scratch, a.dst, a.H, a.n_tokens, a.scale);
}
