#include "convert.cuh"
#include "dequantize.cuh"

#include <cstdint>

#define CUDA_Q8_0_NE_ALIGN 2048

// Each thread dequantizes two consecutive pairs, so its values are adjacent in y:
//   four in a row for qr == 1, two at iqs and two at iqs + qk/2 for qr == 2.
template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    static_assert(qr == 1 || qr == 2, "bad qr");
    const int64_t i00 = 4 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    for (int64_t i01 = blockDim.y*blockIdx.y + threadIdx.y; i01 < ne01; i01 += blockDim.y*gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

            const int64_t ib = ibx0 + i00/qk; // block index
            const int64_t iqs = (i00%qk)/qr; // quant index
            const int64_t iybs = i00 - i00%qk; // y block start index

            // dequantize
            float2 v0;
            float2 v1;
            dequantize_kernel(vx, ib, iqs,        v0);
            dequantize_kernel(vx, ib, iqs + 2/qr, v1);

            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            if constexpr (qr == 1) {
                const dst_t vy[4] = {
                    ggml_cuda_cast<dst_t>(v0.x), ggml_cuda_cast<dst_t>(v0.y), ggml_cuda_cast<dst_t>(v1.x), ggml_cuda_cast<dst_t>(v1.y) };
                dequantize_store<4>(y + iy0, vy);
            } else {
                const dst_t lo[2] = { ggml_cuda_cast<dst_t>(v0.x), ggml_cuda_cast<dst_t>(v1.x) };
                const dst_t hi[2] = { ggml_cuda_cast<dst_t>(v0.y), ggml_cuda_cast<dst_t>(v1.y) };
                dequantize_store<2>(y + iy0,        lo);
                dequantize_store<2>(y + iy0 + qk/2, hi);
            }
        }
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

// The q4_0/q4_1 kernels use 32 threads per 256 values and blockDim.y groups of them per block:
//   a block of 32 threads each leaves the GPU bound by block launches.
template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    uint32_t q; // qs is only 2-byte aligned
    ggml_cuda_memcpy_1<sizeof(q), 2>(&q, x->qs + 4*il);

    dst_t lo[4];
    dst_t hi[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int ql = (q >> 8*l) & 0xFF;
        lo[l] = ggml_cuda_cast<dst_t>(d * (ql & 0xF) + dm);
        hi[l] = ggml_cuda_cast<dst_t>(d * (ql >>  4) + dm);
    }
    dequantize_store<4>(y +  0, lo);
    dequantize_store<4>(y + 16, hi);
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    uint32_t q;
    ggml_cuda_memcpy_1<sizeof(q)>(&q, x->qs + 4*il);

    dst_t lo[4];
    dst_t hi[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int ql = (q >> 8*l) & 0xFF;
        lo[l] = ggml_cuda_cast<dst_t>(d.x * (ql & 0xF) + d.y);
        hi[l] = ggml_cuda_cast<dst_t>(d.x * (ql >>  4) + d.y);
    }
    dequantize_store<4>(y +  0, lo);
    dequantize_store<4>(y + 16, hi);
}

//================================== k-quants and i-quants

// nth threads per super-block of QK_K values, blockDim.y super-blocks per block
template<typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static __global__ void dequantize_block_kq(const void * __restrict__ vx, dst_t * __restrict__ yy, const int64_t nb) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;
    if (i >= nb) {
        return;
    }

    dequantize_kq(vx, i, yy + i*QK_K, threadIdx.x);
}

template<int nth, typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static void dequantize_block_kq_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    constexpr int nsb = CUDA_DEQUANTIZE_BLOCK_SIZE/nth;
    const int64_t nb = (k + QK_K - 1) / QK_K;
    dequantize_block_kq<dst_t, dequantize_kq><<<(nb + nsb - 1)/nsb, dim3(nth, nsb, 1), 0, stream>>>(vx, y, nb);
}

// an IQ4_NL row is a multiple of QK4_NL only, so the sub-blocks past its end are skipped
template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int64_t i = blockIdx.x;

    const int64_t ib = 8*i + threadIdx.x%8;
    if (ib >= nb32) {
        return;
    }

    dequantize_iq4_nl(vx, i, yy + i*QK_K, threadIdx.x);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    // four values per thread; several rows per block if a row has fewer, e.g. one head of a K/V cache
    const int64_t nquads = (ne00 + 3)/4;
    int nx = CUDA_DEQUANTIZE_BLOCK_SIZE;
    while (nx > 1 && nx/2 >= nquads) {
        nx /= 2;
    }
    const int ny = CUDA_DEQUANTIZE_BLOCK_SIZE/nx;
    const dim3 block_dims(nx, ny, 1);
    const dim3 num_blocks((nquads + nx - 1)/nx, (int)std::min((ne01 + ny - 1)/ny, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, block_dims, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<64, dst_t, dequantize_q2_K<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<64, dst_t, dequantize_q3_K<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    constexpr int nsb = CUDA_DEQUANTIZE_BLOCK_SIZE/WARP_SIZE;
    dequantize_block_q4_0<<<(nb + nsb - 1)/nsb, dim3(WARP_SIZE, nsb, 1), 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    constexpr int nsb = CUDA_DEQUANTIZE_BLOCK_SIZE/WARP_SIZE;
    dequantize_block_q4_1<<<(nb + nsb - 1)/nsb, dim3(WARP_SIZE, nsb, 1), 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<32, dst_t, dequantize_q4_K<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<64, dst_t, dequantize_q5_K<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<64, dst_t, dequantize_q6_K<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<32, dst_t, dequantize_iq2_xxs<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<32, dst_t, dequantize_iq2_xs<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<32, dst_t, dequantize_iq2_s<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<32, dst_t, dequantize_iq3_xxs<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<32, dst_t, dequantize_iq3_s<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<32, dst_t, dequantize_iq1_s<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / QK4_NL;
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<32, dst_t, dequantize_iq1_m<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<32, dst_t, dequantize_iq4_xs<dst_t>>(vx, y, k, stream);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    dequantize_block_kq_cuda<32, dst_t, dequantize_mxfp4<dst_t>>(vx, y, k, stream);
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.y + threadIdx.y;
    const int     tid = threadIdx.x;

    const int64_t base = i * QK_NVFP4;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    constexpr int nsb = CUDA_DEQUANTIZE_BLOCK_SIZE/WARP_SIZE;
    dequantize_block_nvfp4<<<(nb + nsb - 1)/nsb, dim3(WARP_SIZE, nsb, 1), 0, stream>>>(vx, y, k);
}
template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const src_t * x = (const src_t *) vx;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
        }
    }
}

template <typename T> struct alignas(sizeof(T)*4) cvt_vec4 { T v[4]; };

// four elements per thread, so a warp moves 512B (RDNA) / 1k (CDNA) per load
template <typename src_t, typename dst_t>
static __global__ void convert_unary_cont_vec4(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k4) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k4) {
        return;
    }

    const cvt_vec4<src_t> xv = ((const cvt_vec4<src_t> *) vx)[i];

    cvt_vec4<dst_t> yv;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        yv.v[j] = ggml_cuda_cast<dst_t>(xv.v[j]);
    }

    ((cvt_vec4<dst_t> *) y)[i] = yv;
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    if (k % 4 == 0 &&
        (uintptr_t) vx % alignof(cvt_vec4<src_t>) == 0 &&
        (uintptr_t) y  % alignof(cvt_vec4<dst_t>) == 0) {
        const int64_t k4 = k/4;
        const int64_t num_blocks = (k4 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
        convert_unary_cont_vec4<src_t, dst_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k4);
        return;
    }

    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}
