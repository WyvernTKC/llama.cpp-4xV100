#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// q8_0 and q4_0 blocks are quantized by a group of 8 consecutive threads: each thread loads 4 of the 32 values and
//   stores its own quants, the maximum is reduced with shuffles. One thread per block, as for the other types above,
//   loads and stores at the block stride, so every warp instruction touches up to 32 memory sectors.
// The formulas are those of quantize_row_q8_0_ref/q4_0_ref, and the q4_0 search keeps the first value of the largest
//   magnitude like them. All 8 threads of a group must call it, t is the thread index within the group.
#define QUANTIZE_GROUP_SIZE 8

static __device__ __forceinline__ unsigned int quantize_group_mask() {
    return 0xFFu << ((threadIdx.x % WARP_SIZE) & ~(QUANTIZE_GROUP_SIZE - 1));
}

// n consecutive floats, one load if they are aligned for it
template <int n>
static __device__ __forceinline__ void quantize_group_load(const float * __restrict__ x, float * __restrict__ v) {
    if ((uintptr_t) x % (n*sizeof(float)) == 0) {
        ggml_cuda_memcpy_1<n*sizeof(float)>(v, x);
    } else {
#pragma unroll
        for (int k = 0; k < n; ++k) {
            v[k] = x[k];
        }
    }
}

static __device__ __forceinline__ void quantize_f32_q8_0_group(const float * __restrict__ x, block_q8_0 * __restrict__ y, const int t) {
    // values 4t .. 4t+3
    __align__(16) float v[4];
    quantize_group_load<4>(x + 4*t, v);

    float amax = 0.0f; // absolute max
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        amax = fmaxf(amax, fabsf(v[k]));
    }
    const unsigned int mask = quantize_group_mask();
#pragma unroll
    for (int offset = QUANTIZE_GROUP_SIZE/2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, QUANTIZE_GROUP_SIZE));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    __align__(4) int8_t q[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x0 = v[k]*id;
        q[k] = roundf(x0);
    }
    ggml_cuda_memcpy_1<sizeof(q), 2>(y->qs + 4*t, q); // qs is only 2-byte aligned

    if (t == 0) {
        y->d = d;
    }
}

static __device__ __forceinline__ void quantize_f32_q4_0_group(const float * __restrict__ x, block_q4_0 * __restrict__ y, const int t) {
    // values 2t, 2t+1 and 16+2t, 17+2t: the low and high nibbles of qs[2t] and qs[2t+1]
    __align__(8) float v[4];
    quantize_group_load<2>(x +           2*t, v + 0);
    quantize_group_load<2>(x + QK4_0/2 + 2*t, v + 2);

    // the first value of the largest magnitude in the order of the block, j is its index
    float amax = 0.0f;
    float vmax = 0.0f;
    int   jmax = QK4_0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (amax < fabsf(v[k])) {
            amax = fabsf(v[k]);
            vmax = v[k];
            jmax = (k/2)*(QK4_0/2) + 2*t + k%2;
        }
    }
    const unsigned int mask = quantize_group_mask();
#pragma unroll
    for (int offset = QUANTIZE_GROUP_SIZE/2; offset > 0; offset >>= 1) {
        const float a = __shfl_xor_sync(mask, amax, offset, QUANTIZE_GROUP_SIZE);
        const float b = __shfl_xor_sync(mask, vmax, offset, QUANTIZE_GROUP_SIZE);
        const int   j = __shfl_xor_sync(mask, jmax, offset, QUANTIZE_GROUP_SIZE);
        if (a > amax || (a == amax && j < jmax)) {
            amax = a;
            vmax = b;
            jmax = j;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    __align__(2) uint8_t qs[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const float x0 = v[0 + k]*id;
        const float x1 = v[2 + k]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        qs[k]  = xi0;
        qs[k] |= xi1 << 4;
    }
    ggml_cuda_memcpy_1<sizeof(qs)>(y->qs + 2*t, qs);

    if (t == 0) {
        y->d = d;
    }
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q8_0_group(const char * cxi, char * cdsti, const int t) {
    quantize_f32_q8_0_group((const float *)cxi, (block_q8_0 *)cdsti, t);
}

static __device__ void cpy_blck_f32_q4_0_group(const char * cxi, char * cdsti, const int t) {
    quantize_f32_q4_0_group((const float *)cxi, (block_q4_0 *)cdsti, t);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
