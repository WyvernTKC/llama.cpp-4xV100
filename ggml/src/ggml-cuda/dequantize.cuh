#include "common.cuh"
#include "convert.cuh"

static __device__ __forceinline__ uint32_t dequantize_bits(const half        v) { return __half_as_ushort(v); }
static __device__ __forceinline__ uint32_t dequantize_bits(const nv_bfloat16 v) { return __bfloat16_as_ushort(v); }
static __device__ __forceinline__ uint32_t dequantize_bits(const float       v) { return __float_as_uint(v); }

// n consecutive values of global memory as stores of up to 16 bytes, or one by one if y is not aligned for that.
// The vector stores are PTX: NVVM splits a uint2/uint4 store of values in registers into 32-bit stores.
template <int n, typename dst_t>
static __device__ __forceinline__ void dequantize_store(dst_t * __restrict__ y, const dst_t * __restrict__ v) {
    constexpr int per_word = 4/sizeof(dst_t);
    constexpr int nw = n/per_word;
    constexpr int align = nw >= 4 ? 16 : 4*nw;
    static_assert(n % per_word == 0 && (nw == 1 || nw == 2 || nw % 4 == 0), "bad n");

    if ((uintptr_t) y % align != 0) {
#pragma unroll
        for (int j = 0; j < n; ++j) {
            y[j] = v[j];
        }
        return;
    }

    uint32_t w[nw];
#pragma unroll
    for (int i = 0; i < nw; ++i) {
        w[i] = dequantize_bits(v[per_word*i]);
        if constexpr (per_word == 2) {
            w[i] |= dequantize_bits(v[per_word*i + 1]) << 16;
        }
    }

    if constexpr (nw == 1) {
        *(uint32_t *) y = w[0];
    } else if constexpr (nw == 2) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        asm volatile("st.global.v2.b32 [%0], {%1, %2};" :: "l"(y), "r"(w[0]), "r"(w[1]) : "memory");
#else
        *(uint2 *) y = make_uint2(w[0], w[1]);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    } else {
#pragma unroll
        for (int i = 0; i < nw; i += 4) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(y + per_word*i),
                "r"(w[i + 0]), "r"(w[i + 1]), "r"(w[i + 2]), "r"(w[i + 3]) : "memory");
#else
            ((uint4 *) y)[i/4] = make_uint4(w[i + 0], w[i + 1], w[i + 2], w[i + 3]);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        }
    }
}

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q2_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_0 * x = (const block_q2_0 *) vx;

    const float d = x[ib].d;

    // Q2_0: 2 bits per element, 4 elements per byte.
    // Stored code c in {0,1,2,3} maps to symbol s = c - 1 in {-1, 0, +1, +2}.
    const int byte_index_0 = iqs / 4;
    const int bit_offset_0 = (iqs % 4) * 2;

    const int byte_index_1 = (iqs + 1) / 4;
    const int bit_offset_1 = ((iqs + 1) % 4) * 2;

    const int c0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 0x3;
    const int c1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 0x3;

    v.x = (c0 - 1) * d;
    v.y = (c1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    char2 q; // iqs is even
    ggml_cuda_memcpy_1<sizeof(q)>(&q, x[ib].qs + iqs);

    v.x = q.x;
    v.y = q.y;

    v.x *= d;
    v.y *= d;
}

//================================== k-quants

// Each call dequantizes one super-block of QK_K values into y using the
// thread layout of the caller: 32 threads for q4_K, 64 threads otherwise.

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q2_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[ib].qs[32*n + l];
    dst_t * y = yy + 128*n;

    float dall = __low2half(x[ib].dm);
    float dmin = __high2half(x[ib].dm);
    y[l+ 0] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[ib].scales[is+0] >> 4));
    y[l+32] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[ib].scales[is+2] >> 4));
    y[l+64] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[ib].scales[is+4] >> 4));
    y[l+96] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[ib].scales[is+6] >> 4));
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q3_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = tid/4;
    const int64_t t = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(tid%4);
    const int64_t n = t / 4;
    const int64_t j = t - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is+0] >> 4) & 3) << 4) :
                          (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[ib].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + 128*n + 32*j;

    uint32_t q; // the block is only 2-byte aligned
    uint32_t hm;
    ggml_cuda_memcpy_1<sizeof(q),  2>(&q,  x[ib].qs + 32*n + l0);
    ggml_cuda_memcpy_1<sizeof(hm), 2>(&hm, x[ib].hmask + l0);

    dst_t v[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint8_t ql = q  >> 8*l;
        const uint8_t hl = hm >> 8*l;
        v[l] = ggml_cuda_cast<dst_t>(dl * ((int8_t)((ql >> shift) & 3) - ((hl & m) ? 0 : 4)));
    }
    dequantize_store<4>(y + l0, v);
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q4_K * x = (const block_q4_K *) vx;

    // assume 32 threads
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + 64*il + n*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    uint32_t q;
    ggml_cuda_memcpy_1<sizeof(q)>(&q, x[ib].qs + 32*il + n*ir);

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    dst_t lo[n];
    dst_t hi[n];
#pragma unroll
    for (int l = 0; l < n; ++l) {
        const int ql = (q >> 8*l) & 0xFF;
        lo[l] = ggml_cuda_cast<dst_t>(d1 * (ql & 0xF) - m1);
        hi[l] = ggml_cuda_cast<dst_t>(d2 * (ql >>  4) - m2);
    }
    dequantize_store<n>(y +  0, lo);
    dequantize_store<n>(y + 32, hi);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q5_K * x = (const block_q5_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + 64*il + 2*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    uchar2 ql;
    uchar2 qh;
    ggml_cuda_memcpy_1<sizeof(ql)>(&ql, x[ib].qs + 32*il + 2*ir);
    ggml_cuda_memcpy_1<sizeof(qh)>(&qh, x[ib].qh + 2*ir);

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    dst_t lo[2];
    dst_t hi[2];
    lo[0] = ggml_cuda_cast<dst_t>(d1 * ((ql.x & 0xF) + (qh.x & hm ? 16 : 0)) - m1);
    lo[1] = ggml_cuda_cast<dst_t>(d1 * ((ql.y & 0xF) + (qh.y & hm ? 16 : 0)) - m1);
    hm <<= 1;
    hi[0] = ggml_cuda_cast<dst_t>(d2 * ((ql.x >>  4) + (qh.x & hm ? 16 : 0)) - m2);
    hi[1] = ggml_cuda_cast<dst_t>(d2 * ((ql.y >>  4) + (qh.y & hm ? 16 : 0)) - m2);
    dequantize_store<2>(y +  0, lo);
    dequantize_store<2>(y + 32, hi);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q6_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q6_K * x = (const block_q6_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + 128*ip + il;

    const float d = x[ib].d;

    const uint8_t * ql = x[ib].ql + 64*ip + il;
    const uint8_t   qh = x[ib].qh[32*ip + il];
    const int8_t  * sc = x[ib].scales + is;

    y[ 0] = ggml_cuda_cast<dst_t>(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = ggml_cuda_cast<dst_t>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = ggml_cuda_cast<dst_t>(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = ggml_cuda_cast<dst_t>(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

//================================== i-quants

// Each call dequantizes one super-block of QK_K values into y with 32
// threads; iq4_nl packs QK_K/QK4_NL sub-blocks per super-block.

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xxs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint64_t grid = iq2xxs_grid[aux8[il]];
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    dst_t v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = ggml_cuda_cast<dst_t>(d * (uint8_t)(grid >> 8*j) * (signs & (1 << j) ? -1.f : 1.f));
    }
    dequantize_store<8>(y, v);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint64_t grid = iq2xs_grid[q2[il] & 511];
    const float d = (float)x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    dst_t v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = ggml_cuda_cast<dst_t>(d * (uint8_t)(grid >> 8*j) * (signs & (1 << j) ? -1.f : 1.f));
    }
    dequantize_store<8>(y, v);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint64_t grid = iq2s_grid[x[ibs].qs[4*ib+il] | ((x[ibs].qh[ib] << (8-2*il)) & 0x300)];
    const float d = (float)x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K/8+4*ib+il];
    dst_t v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = ggml_cuda_cast<dst_t>(d * (uint8_t)(grid >> 8*j) * (signs & (1 << j) ? -1.f : 1.f));
    }
    dequantize_store<8>(y, v);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_xxs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t  * q3 = x[ibs].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[ibs].qs + QK_K/4) + 2*ib;
    const uint32_t grid1 = iq3xxs_grid[q3[2*il+0]];
    const uint32_t grid2 = iq3xxs_grid[q3[2*il+1]];
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    dst_t v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j+0] = ggml_cuda_cast<dst_t>(d * (uint8_t)(grid1 >> 8*j) * (signs & (1 << (j+0)) ? -1.f : 1.f));
        v[j+4] = ggml_cuda_cast<dst_t>(d * (uint8_t)(grid2 >> 8*j) * (signs & (1 << (j+4)) ? -1.f : 1.f));
    }
    dequantize_store<8>(y, v);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t * qs = x[ibs].qs + 8*ib;
    const uint32_t grid1 = iq3s_grid[qs[2*il+0] | ((x[ibs].qh[ib] << (8-2*il)) & 256)];
    const uint32_t grid2 = iq3s_grid[qs[2*il+1] | ((x[ibs].qh[ib] << (7-2*il)) & 256)];
    const float d = (float)x[ibs].d * (1 + 2*((x[ibs].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4*ib + il];
    dst_t v[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j+0] = ggml_cuda_cast<dst_t>(d * (uint8_t)(grid1 >> 8*j) * (signs & (1 << (j+0)) ? -1.f : 1.f));
        v[j+4] = ggml_cuda_cast<dst_t>(d * (uint8_t)(grid2 >> 8*j) * (signs & (1 << (j+4)) ? -1.f : 1.f));
    }
    dequantize_store<8>(y, v);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const float delta = x[ibs].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[ibs].d * (2*((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4*ib+il] | (((x[ibs].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    dst_t v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
    dequantize_store<8>(y, v);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_m(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4*ib+il] | (((x[ibs].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    dst_t v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
    dequantize_store<8>(y, v);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq4_nl * x = (const block_iq4_nl *) vx + ibs*(QK_K/QK4_NL);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    uint32_t q4; // qs is only 2-byte aligned
    ggml_cuda_memcpy_1<sizeof(q4), 2>(&q4, x[ib].qs + 4*il);
    const float d = (float)x[ib].d;
    dst_t lo[4];
    dst_t hi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint8_t q = q4 >> 8*j;
        lo[j] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q & 0xf]);
        hi[j] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q >>  4]);
    }
    dequantize_store<4>(y +  0, lo);
    dequantize_store<4>(y + 16, hi);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_xs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    uint32_t q4;
    ggml_cuda_memcpy_1<sizeof(q4)>(&q4, x[ibs].qs + 16*ib + 4*il);
    const float d = (float)x[ibs].d * ((((x[ibs].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[ibs].scales_h >> 2*ib) & 3) << 4)) - 32);
    dst_t lo[4];
    dst_t hi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint8_t q = q4 >> 8*j;
        lo[j] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q & 0xf]);
        hi[j] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q >>  4]);
    }
    dequantize_store<4>(y +  0, lo);
    dequantize_store<4>(y + 16, hi);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_mxfp4(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_mxfp4 * x = (const block_mxfp4 *) vx + ibs*(QK_K/QK_MXFP4);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    dst_t lo[4];
    dst_t hi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo[j] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] & 0xf]*0.5f);
        hi[j] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] >>  4]*0.5f);
    }
    dequantize_store<4>(y +  0, lo);
    dequantize_store<4>(y + 16, hi);
}
