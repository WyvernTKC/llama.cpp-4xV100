#include "getrows.cuh"
#include "dequantize.cuh"
#include "convert.cuh"

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 =  blockIdx.x;
            const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
            const int i11 =  dm.x;
            const int i12 =  dm.y;

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

            const int ib   =  i00/qk;      // block index
            const int iqs  = (i00%qk)/qr;  // quant index
            const int iybs = i00 - i00%qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(src0_row, ib, iqs, v);

            dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(v.x);
            dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template<typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static __global__ void k_get_rows_kq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    const int64_t nsb = ne00/QK_K; // super-blocks per row
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

        for (int64_t ib = blockIdx.y; ib < nsb; ib += gridDim.y) {
            dequantize_kq(src0_row, ib, dst_row + ib*QK_K, threadIdx.x);
        }
    }
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
        const src0_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_lc();
    const src0_t  * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const int32_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * GGML_CUDA_RESTRICT dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const src0_t * GGML_CUDA_RESTRICT src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) {
            dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
        }
    }
}

// rows narrower than a block: one thread per element over every gathered row, so a block covers many rows
template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float_narrow(
        const src0_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t n, const uint3 ne00_fdv, const uint3 ne10_fdv, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    ggml_cuda_pdl_lc();
    const src0_t  * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const int32_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        const uint2 r00 = fast_div_modulo((uint32_t) i, ne00_fdv); // row, i00
        const uint2 r10 = fast_div_modulo(r00.x, ne10_fdv);         // z,   i10
        const uint2 r12 = fast_div_modulo(r10.x, ne12_fdv);         // i11, i12
        const int i00 = r00.y;
        const int i10 = r10.y;
        const int i11 = r12.x;
        const int i12 = r12.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        const src0_t * src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);
        dst[i10*s1 + i11*s2 + i12*s3 + i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
    }
}

template<typename dst_t>
static __global__ void k_get_rows_float_vec(
        const dst_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00v,
        const int64_t ne11, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1_ptr[i10*s10 + i11*s11 + i12*s12];

        int4       * GGML_CUDA_RESTRICT dst_row  = (int4 *)      (dst_ptr + i10*s1 + i11*s2 + i12*s3);
        const int4 * GGML_CUDA_RESTRICT src0_row = (const int4 *)((const char *) src0_ptr + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i = blockIdx.y*blockDim.x + threadIdx.x; i < ne00v; i += gridDim.y*blockDim.x) {
            dst_row[i] = src0_row[i];
        }
    }
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst,
        const int64_t ncols, const int64_t nrows_grad, const int64_t nrows_dst) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    ggml_cuda_pdl_sync();

    // grid.y is clamped to the CUDA grid limit, so stride over the destination rows
    for (int64_t dst_row = blockIdx.y; dst_row < nrows_dst; dst_row += gridDim.y) {
        float sum = 0.0f;

        for (int64_t i = 0; i < nrows_grad; ++i) {
            if (rows[i] != dst_row) {
                continue;
            }
            sum += grad[i*ncols + col];
        }

        dst[dst_row*ncols + col] = sum;
    }
}

template<int qk, int qr, dequantize_kernel_t dq, typename dst_t>
static void get_rows_cuda_q(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne00 % 2 == 0);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<int block_dim, typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static void get_rows_cuda_kq(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t nsb = ne00/QK_K;

    const dim3 block_dims(block_dim, 1, 1);
    const dim3 block_nums(ne10, MIN(nsb, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows_kq<dst_t, dequantize_kq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<typename src0_t, typename dst_t>
static void get_rows_cuda_float(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    // a row narrower than a block leaves most of a block-per-row launch idle
    // fastdiv sums two 32-bit terms, so the flat index must stay below 2^31
    const int64_t n_narrow = ne00*ne10*ne11*ne12;
    if (ne00 < CUDA_GET_ROWS_BLOCK_SIZE && n_narrow > 0 && n_narrow <= std::numeric_limits<int32_t>::max()) {
        const dim3 block_nums((n_narrow + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
        ggml_cuda_kernel_launch(k_get_rows_float_narrow<src0_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d,
            n_narrow, init_fastdiv_values(ne00), init_fastdiv_values(ne10), ne12_fdv,
            s1, s2, s3,
            nb01, nb02, nb03,
            s10, s11, s12);
        return;
    }

    if constexpr (std::is_same<src0_t, dst_t>::value) {
        constexpr int VEC = 16 / sizeof(dst_t);
        const int64_t ne00v = ne00 / VEC;
        const int64_t vec_block_num_y = (ne00v + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
        const bool enough_blocks = vec_block_num_y * ne10 * ne11 * ne12 >= 128;
        const bool can_vec = VEC > 1 && enough_blocks &&
            (ne00 % VEC == 0) &&
            (nb01 % 16 == 0) && (nb02 % 16 == 0) && (nb03 % 16 == 0) &&
            (nb1  % 16 == 0) && (nb2  % 16 == 0) && (nb3  % 16 == 0) &&
            (((uintptr_t) src0_d) % 16 == 0) && (((uintptr_t) dst_d) % 16 == 0);

        if (can_vec) {
            const int block_num_y = vec_block_num_y;
            const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
            ggml_cuda_kernel_launch(k_get_rows_float_vec<dst_t>, launch_params,
                (const dst_t *) src0_d, src1_d, dst_d,
                ne00v, ne11, ne12_fdv,
                s1, s2, s3,
                nb01, nb02, nb03,
                s10, s11, s12);
            return;
        }
    }

    const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
    ggml_cuda_kernel_launch(k_get_rows_float<src0_t, dst_t>, launch_params,
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(
        const void * src0_d, const ggml_type src0_type, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    switch (src0_type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float((const half *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float((const float *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            get_rows_cuda_float((const int32_t *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            get_rows_cuda_float((const nv_bfloat16 *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q1_0:
            get_rows_cuda_q<QK1_0, QR1_0, dequantize_q1_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_0:
            get_rows_cuda_q<QK2_0, QR2_0, dequantize_q2_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda_q<QK4_0, QR4_0, dequantize_q4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda_q<QK4_1, QR4_1, dequantize_q4_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda_q<QK5_0, QR5_0, dequantize_q5_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda_q<QK5_1, QR5_1, dequantize_q5_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda_q<QK8_0, QR8_0, dequantize_q8_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q2_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q3_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q3_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_K:
            get_rows_cuda_kq<32, dst_t, dequantize_q4_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q5_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q6_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q6_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_M:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_m<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_nl<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_MXFP4:
            get_rows_cuda_kq<32, dst_t, dequantize_mxfp4<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0_type));
            break;
    }
}

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (float *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (int32_t *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (half *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (nv_bfloat16 *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

// rows copied unchanged in units of T, for a dst of the same (quantized) type as src0
template <typename T>
static __global__ void k_get_rows_bytes(
        const char * src0, const int32_t * src1, char * dst,
        const int64_t n_units, const int64_t ne11, const uint3 ne12_fdv,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t s10, const size_t s11, const size_t s12) {
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        const int   i10 = blockIdx.x;
        const uint2 dm  = fast_div_modulo((uint32_t) z, ne12_fdv);
        const int   i11 = dm.x;
        const int   i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        const T * src_row = (const T *) (src0 + i01*nb01 + i11*nb02 + i12*nb03);
        T       * dst_row = (T       *) (dst  + i10*nb1  + i11*nb2  + i12*nb3);

        for (int64_t i = blockIdx.y*blockDim.x + threadIdx.x; i < n_units; i += gridDim.y*blockDim.x) {
            dst_row[i] = src_row[i];
        }
    }
}

// rows narrower than a block: one thread per unit over every gathered row, so a block covers many rows
template <typename T>
static __global__ void k_get_rows_bytes_narrow(
        const char * src0, const int32_t * src1, char * dst,
        const int64_t n, const uint3 n_units_fdv, const uint3 ne10_fdv, const uint3 ne12_fdv,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const size_t s10, const size_t s11, const size_t s12) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        const uint2 ru  = fast_div_modulo((uint32_t) i, n_units_fdv); // row, unit
        const uint2 r10 = fast_div_modulo(ru.x, ne10_fdv);             // z,   i10
        const uint2 r12 = fast_div_modulo(r10.x, ne12_fdv);            // i11, i12
        const int i10 = r10.y;
        const int i11 = r12.x;
        const int i12 = r12.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        const T * src_row = (const T *) (src0 + i01*nb01 + i11*nb02 + i12*nb03);
        T       * dst_row = (T       *) (dst  + i10*nb1  + i11*nb2  + i12*nb3);

        dst_row[ru.y] = src_row[ru.y];
    }
}

static void get_rows_cuda_bytes(
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const size_t row_size = ggml_row_size(src0->type, ne00);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    // the widest unit that divides the row and every stride and address
    auto fits = [&](size_t u) {
        return row_size % u == 0 && nb01 % u == 0 && nb02 % u == 0 && nb03 % u == 0 &&
            nb1 % u == 0 && nb2 % u == 0 && nb3 % u == 0 &&
            ((uintptr_t) src0->data) % u == 0 && ((uintptr_t) dst->data) % u == 0;
    };
    const size_t unit = fits(16) ? 16 : fits(8) ? 8 : fits(4) ? 4 : 1;

    const int64_t n_units = row_size / unit;
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int   block_num_y = (n_units + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    const char    * src0_d = (const char    *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    char          * dst_d  = (char          *) dst->data;

    // a row narrower than a block leaves most of a block-per-row launch idle
    // fastdiv sums two 32-bit terms, so the flat index must stay below 2^31
    const int64_t n_narrow = n_units*ne10*ne11*ne12;
    const bool    narrow   = n_units < CUDA_GET_ROWS_BLOCK_SIZE && n_narrow > 0 && n_narrow <= std::numeric_limits<int32_t>::max();

    auto launch = [&](auto unit_t) {
        using T = decltype(unit_t);
        if (narrow) {
            const dim3 block_nums_narrow((n_narrow + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
            k_get_rows_bytes_narrow<T><<<block_nums_narrow, block_dims, 0, stream>>>(src0_d, src1_d, dst_d,
                n_narrow, init_fastdiv_values(n_units), init_fastdiv_values(ne10), ne12_fdv,
                nb01, nb02, nb03, nb1, nb2, nb3, s10, s11, s12);
        } else {
            k_get_rows_bytes<T><<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d,
                n_units, ne11, ne12_fdv, nb01, nb02, nb03, nb1, nb2, nb3, s10, s11, s12);
        }
    };

    switch (unit) {
        case 16: launch(int4{});    break;
        case  8: launch(int2{});    break;
        case  4: launch(int32_t{}); break;
        default: launch(int8_t{});  break;
    }
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    if (dst->type == src0->type && ggml_is_quantized(src0->type)) {
        get_rows_cuda_bytes(src0, src1, dst, stream);
        return;
    }

    get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

// a pair of consecutive elements of a float row, in the shape of the dequantize_kernel_t with qk = qr = 1
// aligned: the pair is one half2/float2 load, which the caller checked the addresses allow
template<typename src_t, bool aligned>
static __device__ __forceinline__ void dequantize_float_pair(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    GGML_UNUSED(iqs);
    const src_t * x = (const src_t *) vx;
    if constexpr (aligned && std::is_same_v<src_t, half>) {
        const half2 h = *(const half2 *) (x + ib);
        v.x = __low2float(h);
        v.y = __high2float(h);
    } else if constexpr (aligned && std::is_same_v<src_t, float>) {
        v = *(const float2 *) (x + ib);
    } else {
        v.x = ggml_cuda_cast<float>(x[ib + 0]);
        v.y = ggml_cuda_cast<float>(x[ib + 1]);
    }
}

// a q8_0 pair in one 16-bit load: blocks are 34 bytes with the quants at offset 2, and iqs is even
static __device__ __forceinline__ void dequantize_q8_0_pair(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;
    const char2 q = *(const char2 *) (x[ib].qs + iqs);

    v.x = q.x;
    v.y = q.y;

    v.x *= d;
    v.y *= d;
}

// one thread per pair of output elements: the n_group gathered members are added in index order and the
// sum is scaled, the same arithmetic as get_rows + add + scale (__fadd_rn keeps the dequant multiply out of an FMA)
// n_group_fixed > 0 unrolls the member loop, so every member load is in flight at once
template<int qk, int qr, dequantize_kernel_t dequantize_kernel, int n_group_fixed>
static __global__ void k_get_rows_mean(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, float * __restrict__ dst,
        const int64_t n, const int n_group, const float scale,
        const uint3 ne00h_fdv, const uint3 ne1_fdv, const uint3 ne2_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        const uint2 r00 = fast_div_modulo((uint32_t) i, ne00h_fdv); // row, pair
        const uint2 r1  = fast_div_modulo(r00.x, ne1_fdv);          // z,   i1
        const uint2 r2  = fast_div_modulo(r1.x,  ne2_fdv);          // i12, i11
        const int i00 = 2*r00.y;
        const int i1  = r1.y;
        const int i11 = r2.y;
        const int i12 = r2.x;

        const int ib  =  i00/qk;
        const int iqs = (i00%qk)/qr;

        const int ng = n_group_fixed > 0 ? n_group_fixed : n_group;

        float2 acc = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int g = 0; g < ng; ++g) {
            const int i01 = src1[(i1*ng + g)*s10 + i11*s11 + i12*s12];

            float2 v;
            dequantize_kernel((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03, ib, iqs, v);

            acc.x = g == 0 ? v.x : __fadd_rn(acc.x, v.x);
            acc.y = g == 0 ? v.y : __fadd_rn(acc.y, v.y);
        }

        float * dst_row = dst + i1*s1 + i11*s2 + i12*s3;
        if constexpr (qr == 1) {
            // dst is contiguous with an even row length, so an even element is 8-byte aligned
            *(float2 *) (dst_row + i00) = make_float2(scale*acc.x, scale*acc.y);
        } else {
            const int iybs = i00 - i00%qk;
            dst_row[iybs + iqs + 0]    = scale*acc.x;
            dst_row[iybs + iqs + qk/2] = scale*acc.y;
        }
    }
}

template<int qk, int qr, dequantize_kernel_t dq>
static void get_rows_mean_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const int n_group, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // fastdiv sums two 32-bit terms, so the flat index must stay below 2^31
    const int64_t n = (ne00/2)*ne1*ne2*ne3;
    GGML_ASSERT(n <= std::numeric_limits<int32_t>::max());
    if (n == 0) {
        return;
    }

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((n + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);

    // 4 is the qwen4exp compress ratio
    auto kernel = n_group == 4 ? k_get_rows_mean<qk, qr, dq, 4> : k_get_rows_mean<qk, qr, dq, 0>;
    kernel<<<block_nums, block_dims, 0, stream>>>(
        src0->data, (const int32_t *) src1->data, (float *) dst->data,
        n, n_group, 1.0f/n_group,
        init_fastdiv_values(ne00/2), init_fastdiv_values(ne1), init_fastdiv_values(ne2),
        nb1/sizeof(float), nb2/sizeof(float), nb3/sizeof(float),
        nb01, nb02, nb03,
        nb10/sizeof(int32_t), nb11/sizeof(int32_t), nb12/sizeof(int32_t));
}

bool ggml_cuda_get_rows_mean_supported(const ggml_tensor * op) {
    if (op->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_I32 || op->src[0]->ne[0] % 2 != 0) {
        return false;
    }
    switch (op->src[0]->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

void ggml_cuda_op_get_rows_mean(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int n_group = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_cuda_get_rows_mean_supported(dst));
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == sizeof(int32_t));
    GGML_ASSERT(dst->nb[0]  == sizeof(float));

    cudaStream_t stream = ctx.stream();

    // a pair load needs the pair's alignment at every row start
    auto aligned = [&](size_t a) {
        return (uintptr_t) src0->data % a == 0 && src0->nb[1] % a == 0 && src0->nb[2] % a == 0 && src0->nb[3] % a == 0;
    };

    switch (src0->type) {
        case GGML_TYPE_F32:
            if (aligned(8)) {
                get_rows_mean_cuda<1, 1, dequantize_float_pair<float, true>>(src0, src1, dst, n_group, stream);
            } else {
                get_rows_mean_cuda<1, 1, dequantize_float_pair<float, false>>(src0, src1, dst, n_group, stream);
            }
            break;
        case GGML_TYPE_F16:
            if (aligned(4)) {
                get_rows_mean_cuda<1, 1, dequantize_float_pair<half, true>>(src0, src1, dst, n_group, stream);
            } else {
                get_rows_mean_cuda<1, 1, dequantize_float_pair<half, false>>(src0, src1, dst, n_group, stream);
            }
            break;
        case GGML_TYPE_BF16: get_rows_mean_cuda<1, 1, dequantize_float_pair<nv_bfloat16, false>>(src0, src1, dst, n_group, stream); break;
        case GGML_TYPE_Q4_0: get_rows_mean_cuda<QK4_0, QR4_0, dequantize_q4_0>(src0, src1, dst, n_group, stream); break;
        case GGML_TYPE_Q4_1: get_rows_mean_cuda<QK4_1, QR4_1, dequantize_q4_1>(src0, src1, dst, n_group, stream); break;
        case GGML_TYPE_Q5_0: get_rows_mean_cuda<QK5_0, QR5_0, dequantize_q5_0>(src0, src1, dst, n_group, stream); break;
        case GGML_TYPE_Q5_1: get_rows_mean_cuda<QK5_1, QR5_1, dequantize_q5_1>(src0, src1, dst, n_group, stream); break;
        case GGML_TYPE_Q8_0:
            if (aligned(2)) {
                get_rows_mean_cuda<QK8_0, QR8_0, dequantize_q8_0_pair>(src0, src1, dst, n_group, stream);
            } else {
                get_rows_mean_cuda<QK8_0, QR8_0, dequantize_q8_0>(src0, src1, dst, n_group, stream);
            }
            break;
        default: GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0->type));
    }
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, MIN(ne1, (int64_t)UINT16_MAX), 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10, ne1);
}

// GET_ROWS of float rows followed by an op of the gathered rows: a block row per gathered row, each element converted to
//   f32 as GET_ROWS writes it and on as the next op takes it
template <typename src_t>
static __device__ __forceinline__ float get_rows_fused_load(const char * src, const int64_t row, const int64_t nb1, const int64_t c) {
    return ggml_cuda_cast<float>(((const src_t *) (src + row*nb1))[c]);
}

template <typename src_t, typename dst_t, typename idx_t>
static __global__ void k_get_rows_set_rows(const char * src, const int32_t * gidx, const idx_t * sidx, char * dst,
        const int64_t ne0, const int64_t src_nb1, const int64_t dst_nb1) {
    const int64_t k = blockIdx.y;
    const int64_t c = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= ne0) {
        return;
    }
    const float v = get_rows_fused_load<src_t>(src, gidx[k], src_nb1, c);
    ((dst_t *) (dst + (int64_t) sidx[k]*dst_nb1))[c] = ggml_cuda_cast<dst_t>(v);
}

template <typename src_t>
static __global__ void k_get_rows_add(const char * src, const int32_t * gidx, const float * a, float * dst,
        const int64_t ne0, const int64_t src_nb1, const int64_t a_s1, const int64_t dst_s1) {
    const int64_t r = blockIdx.y;
    const int64_t c = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= ne0) {
        return;
    }
    dst[r*dst_s1 + c] = a[r*a_s1 + c] + get_rows_fused_load<src_t>(src, gidx[r], src_nb1, c);
}

// rows of a float matrix gathered by i32 indices into an f32 matrix, two dimensions
static bool ggml_cuda_get_rows_fusable(const ggml_tensor * get_rows) {
    const ggml_tensor * src = get_rows->src[0];
    const ggml_tensor * idx = get_rows->src[1];
    return (src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_BF16) &&
        src->nb[0] == ggml_type_size(src->type) && src->ne[2] == 1 && src->ne[3] == 1 &&
        idx->type == GGML_TYPE_I32 && ggml_is_contiguous(idx) && ggml_nelements(idx) == get_rows->ne[1] &&
        get_rows->type == GGML_TYPE_F32 && get_rows->ne[2] == 1 && get_rows->ne[3] == 1 && get_rows->ne[1] <= 65535;
}

bool ggml_cuda_should_fuse_get_rows_set_rows(const ggml_tensor * get_rows, const ggml_tensor * set_rows) {
    const ggml_tensor * sidx = set_rows->src[1];
    const ggml_tensor * dst  = set_rows;
    return ggml_cuda_get_rows_fusable(get_rows) && set_rows->src[0] == get_rows &&
        (sidx->type == GGML_TYPE_I32 || sidx->type == GGML_TYPE_I64) && ggml_is_contiguous(sidx) &&
        ggml_nelements(sidx) == get_rows->ne[1] &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16) &&
        dst->nb[0] == ggml_type_size(dst->type) && dst->ne[0] == get_rows->ne[0] && dst->ne[2] == 1 && dst->ne[3] == 1 &&
        !ggml_cuda_tensors_overlap(dst, get_rows->src[0]) && !ggml_cuda_tensors_overlap(dst, get_rows->src[1]) &&
        !ggml_cuda_tensors_overlap(dst, sidx);
}

bool ggml_cuda_should_fuse_get_rows_add(const ggml_tensor * get_rows, const ggml_tensor * add) {
    const ggml_tensor * a = add->src[0] == get_rows ? add->src[1] : add->src[0];
    return ggml_cuda_get_rows_fusable(get_rows) && (add->src[0] == get_rows) != (add->src[1] == get_rows) &&
        add->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 && ggml_are_same_shape(add, get_rows) &&
        ggml_are_same_shape(add, a) && a->nb[0] == sizeof(float) && add->nb[0] == sizeof(float) &&
        ggml_cuda_same_or_disjoint(add, a) && !ggml_cuda_tensors_overlap(add, get_rows->src[0]) &&
        !ggml_cuda_tensors_overlap(add, get_rows->src[1]);
}

template <typename src_t, typename dst_t>
static void get_rows_set_rows_launch(const ggml_tensor * get_rows, ggml_tensor * set_rows, cudaStream_t stream) {
    const ggml_tensor * src  = get_rows->src[0];
    const ggml_tensor * gidx = get_rows->src[1];
    const ggml_tensor * sidx = set_rows->src[1];
    const int64_t ne0 = get_rows->ne[0];
    const dim3 grid((unsigned) ((ne0 + CUDA_GET_ROWS_BLOCK_SIZE - 1)/CUDA_GET_ROWS_BLOCK_SIZE), (unsigned) get_rows->ne[1], 1);
    if (sidx->type == GGML_TYPE_I64) {
        k_get_rows_set_rows<src_t, dst_t, int64_t><<<grid, CUDA_GET_ROWS_BLOCK_SIZE, 0, stream>>>((const char *) src->data,
            (const int32_t *) gidx->data, (const int64_t *) sidx->data, (char *) set_rows->data, ne0, src->nb[1], set_rows->nb[1]);
    } else {
        k_get_rows_set_rows<src_t, dst_t, int32_t><<<grid, CUDA_GET_ROWS_BLOCK_SIZE, 0, stream>>>((const char *) src->data,
            (const int32_t *) gidx->data, (const int32_t *) sidx->data, (char *) set_rows->data, ne0, src->nb[1], set_rows->nb[1]);
    }
}

template <typename src_t>
static void get_rows_set_rows_dst(const ggml_tensor * get_rows, ggml_tensor * set_rows, cudaStream_t stream) {
    switch (set_rows->type) {
        case GGML_TYPE_F32:  get_rows_set_rows_launch<src_t, float>      (get_rows, set_rows, stream); break;
        case GGML_TYPE_F16:  get_rows_set_rows_launch<src_t, half>       (get_rows, set_rows, stream); break;
        case GGML_TYPE_BF16: get_rows_set_rows_launch<src_t, nv_bfloat16>(get_rows, set_rows, stream); break;
        default: GGML_ABORT("unsupported type");
    }
}

void ggml_cuda_op_get_rows_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * get_rows, ggml_tensor * set_rows) {
    cudaStream_t stream = ctx.stream();
    switch (get_rows->src[0]->type) {
        case GGML_TYPE_F32:  get_rows_set_rows_dst<float>      (get_rows, set_rows, stream); break;
        case GGML_TYPE_F16:  get_rows_set_rows_dst<half>       (get_rows, set_rows, stream); break;
        case GGML_TYPE_BF16: get_rows_set_rows_dst<nv_bfloat16>(get_rows, set_rows, stream); break;
        default: GGML_ABORT("unsupported type");
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_get_rows_add(ggml_backend_cuda_context & ctx, const ggml_tensor * get_rows, ggml_tensor * add) {
    const ggml_tensor * src  = get_rows->src[0];
    const ggml_tensor * gidx = get_rows->src[1];
    const ggml_tensor * a    = add->src[0] == get_rows ? add->src[1] : add->src[0];
    const int64_t ne0 = get_rows->ne[0];
    const dim3 grid((unsigned) ((ne0 + CUDA_GET_ROWS_BLOCK_SIZE - 1)/CUDA_GET_ROWS_BLOCK_SIZE), (unsigned) get_rows->ne[1], 1);
    cudaStream_t stream = ctx.stream();
    const int64_t a_s1   = a->nb[1]/sizeof(float);
    const int64_t dst_s1 = add->nb[1]/sizeof(float);

    switch (src->type) {
        case GGML_TYPE_F32:
            k_get_rows_add<float><<<grid, CUDA_GET_ROWS_BLOCK_SIZE, 0, stream>>>((const char *) src->data,
                (const int32_t *) gidx->data, (const float *) a->data, (float *) add->data, ne0, src->nb[1], a_s1, dst_s1);
            break;
        case GGML_TYPE_F16:
            k_get_rows_add<half><<<grid, CUDA_GET_ROWS_BLOCK_SIZE, 0, stream>>>((const char *) src->data,
                (const int32_t *) gidx->data, (const float *) a->data, (float *) add->data, ne0, src->nb[1], a_s1, dst_s1);
            break;
        case GGML_TYPE_BF16:
            k_get_rows_add<nv_bfloat16><<<grid, CUDA_GET_ROWS_BLOCK_SIZE, 0, stream>>>((const char *) src->data,
                (const int32_t *) gidx->data, (const float *) a->data, (float *) add->data, ne0, src->nb[1], a_s1, dst_s1);
            break;
        default: GGML_ABORT("unsupported type");
    }
    CUDA_CHECK(cudaGetLastError());
}
