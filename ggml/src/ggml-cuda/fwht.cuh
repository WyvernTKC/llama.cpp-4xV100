#pragma once

#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// The scale of an n-point transform, computed on the host
static inline float ggml_cuda_fwht_scale(const int n) {
    return 1 / sqrtf(n);
}

// The same transform by all threads of a block, in place on each n consecutive values of s[0 .. len) in shared memory
//   (n a power of 2 dividing len, len a multiple of 32, blockDim.x a multiple of the warp size). The values are scaled
//   first and the butterflies run with the stride doubling from 1 as (lo + hi, lo - hi), strides up to 16 in registers
//   across a warp, which is what fwht_cuda does, so the results are bitwise the same when scale is the host's 1/sqrtf(n)
//   (see ggml_cuda_fwht_scale; fast math on the device rounds it differently). Starts and ends with a barrier: s may be
//   written just before the call and read just after it by any thread.
static __device__ __forceinline__ void fwht_block_shared(float * s, const int len, const int n, const float scale) {
    const int lane   = threadIdx.x % 32;
    const int nwarps = blockDim.x / 32;

    __syncthreads();
    for (int g = threadIdx.x / 32; g < len/32; g += nwarps) {
        float v = __fmul_rn(s[32*g + lane], scale);
        for (int h = 1; h < 32 && h < n; h *= 2) {
            const float v2 = __shfl_xor_sync(0xFFFFFFFF, v, h, 32);
            v = (lane & h) == 0 ? v + v2 : v2 - v;
        }
        s[32*g + lane] = v;
    }
    __syncthreads();

    for (int lh = 5; (1 << lh) < n; ++lh) {
        // pair p is (i, i + h) with i the p-th index whose bit lh is clear; a pair never leaves its segment of n
        for (int p = threadIdx.x; p < len/2; p += blockDim.x) {
            const int i = ((p >> lh) << (lh + 1)) | (p & ((1 << lh) - 1));

            const float lo = s[i];
            const float hi = s[i + (1 << lh)];

            s[i]             = lo + hi;
            s[i + (1 << lh)] = lo - hi;
        }
        __syncthreads();
    }
}
