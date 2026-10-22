#include "unary.cuh"
#include "convert.cuh"

static __device__ __forceinline__ float op_abs(float x) {
    return fabsf(x);
}

static __device__ __forceinline__ float op_sgn(float x) {
    return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f)));
}

static __device__ __forceinline__ float op_neg(float x) {
    return -x;
}

static __device__ __forceinline__ float op_step(float x) {
    return x > 0.0f;
}

static __device__ __forceinline__ float op_gelu(float x) {
    return ggml_cuda_op_gelu_single(x);
}

static __device__ __forceinline__ float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;

    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}

static __device__ __forceinline__ float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;

    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __device__ __forceinline__ float op_silu(float x) {
    return ggml_cuda_op_silu_single(x);
}

static __device__ __forceinline__ float op_tanh(float x) {
    return tanhf(x);
}

static __device__ __forceinline__ float op_relu(float x) {
    return fmaxf(x, 0);
}

static __device__ __forceinline__ float op_sigmoid(float x) {
    return ggml_cuda_op_sigmoid_single(x);
}

static __device__ __forceinline__ float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_exp(float x) {
    return expf(x);
}

static __device__ __forceinline__ float op_sqr(float x) {
    return x * x;
}

static __device__ __forceinline__ float op_relu_sqr(float x) {
    const float r = fmaxf(x, 0.0f);
    return r * r;
}

static __device__ __forceinline__ float op_sqrt(float x) {
    return sqrtf(x);
}

static __device__ __forceinline__ float op_sin(float x) {
    return sinf(x);
}

static __device__ __forceinline__ float op_cos(float x) {
    return cosf(x);
}

static __device__ __forceinline__ float op_log(float x) {
    return logf(x);
}

static __device__ __forceinline__ float op_expm1(float x) {
    return expm1f(x);
}

static __device__ __forceinline__ float op_softplus(float x) {
    return ggml_cuda_op_softplus_single(x);
}

static __device__ __forceinline__ float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static __device__ __forceinline__ float op_floor(float x) {
    return floorf(x);
}

static __device__ __forceinline__ float op_ceil(float x) {
    return ceilf(x);
}

static __device__ __forceinline__ float op_round(float x) {
    return roundf(x);
}

static __device__ __forceinline__ float op_trunc(float x) {
    return trunc(x);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[i])));
}

template <float (*op)(float), typename T>
static void unary_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_op_kernel<op, T>, launch_params, x, dst, k);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel_strided(const T * x, T * dst, const int k,const int64_t ne00,const int64_t ne01,const int64_t ne02,const size_t nb00,const size_t nb01,const size_t nb02,const size_t nb03) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    int64_t rem = i;
    const int64_t i0 = rem % ne00; rem /= ne00;
    const int64_t i1 = rem % ne01; rem /= ne01;
    const int64_t i2 = rem % ne02;
    const int64_t i3 = rem / ne02;
    const size_t src_byte_offset = i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    const T * src_ptr = (const T *)((const char *)x + src_byte_offset);

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(*src_ptr)));
}

template <float (*op)(float), typename T>
static void unary_cuda_strided(const T * x, T * dst, const int k,const int64_t ne00,const int64_t ne01,const int64_t ne02,const size_t nb00,const size_t nb01,const size_t nb02,const size_t nb03, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_op_kernel_strided<op, T>, launch_params, x, dst, k, ne00,ne01,ne02,nb00,nb01,nb02,nb03);
}

template <float (*op)(float)>
void ggml_cuda_op_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);

    if (ggml_is_contiguous(src0)) {
        if (src0->type == GGML_TYPE_F16) {
            unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
        } else if (src0->type == GGML_TYPE_BF16) {
            unary_cuda<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), stream);
        } else {
            unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
        }
    } else {
        if (src0->type == GGML_TYPE_F16) {
            unary_cuda_strided<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        } else if (src0->type == GGML_TYPE_BF16) {
            unary_cuda_strided<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        } else {
            unary_cuda_strided<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        }
    }
}

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_abs>(ctx, dst);
}

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sgn>(ctx, dst);
}

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_neg>(ctx, dst);
}

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_step>(ctx, dst);
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu>(ctx, dst);
}

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_quick>(ctx, dst);
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_silu>(ctx, dst);
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_tanh>(ctx, dst);
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_relu>(ctx, dst);
}

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sigmoid>(ctx, dst);
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardsigmoid>(ctx, dst);
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardswish>(ctx, dst);
}

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_exp>(ctx, dst);
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqr>(ctx, dst);
}

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqrt>(ctx, dst);
}

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sin>(ctx, dst);
}

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_cos>(ctx, dst);
}

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_log>(ctx, dst);
}

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_elu>(ctx, dst);
}

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_floor>(ctx, dst);
}

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_ceil>(ctx, dst);
}

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_round>(ctx, dst);
}

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_trunc>(ctx, dst);
}

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_expm1>(ctx, dst);
}

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_softplus>(ctx, dst);
}
/* gated ops */

// f16_copy also writes the output as f16 to dst_f16, the input a cuBLAS matmul after it would convert; q8_copy the q8_1
//   copy of one row (k a multiple of 32) to q8 for an MMVQ matmul, with zero blocks up to k_padded
// the fast division by n of the indices of k elements, or none (z == 0) when they do not fit 31 bits
static uint3 unary_gated_n_fastdiv(const int64_t k, const int64_t n) {
    return k <= INT_MAX ? init_fastdiv_values((uint64_t) n) : make_uint3(0, 0, 0);
}

template <float (*op)(float), typename T, bool f16_copy = false, bool q8_copy = false>
static __global__ void unary_gated_op_kernel(const T * x, const T * g, T * dst, half * dst_f16, const int64_t k, const int64_t n,
        const uint3 n_fd, const int64_t o0, const int64_t o1, block_q8_1 * q8, const int64_t k_padded) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        if constexpr (q8_copy) {
            // whole warps, k being a multiple of 32
            if (i < k_padded) {
                quantize_q8_1_lane(0.0f, q8 + i/QK8_1, i % QK8_1);
            }
        }
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one); a 64-bit division is a long
    //   software routine, so the row and column come from a 32-bit fast division where they fit
    int64_t row;
    int64_t col;
    if (n_fd.z != 0) {
        const uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        row = rc.x;
        col = rc.y;
    } else {
        row = i / n;
        col = i % n;
    }
    const int64_t j0 = row * o0 + col;
    const int64_t j1 = o0 == o1 ? j0 : row * o1 + col;

    ggml_cuda_pdl_sync();
    const T v = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[j0])) * ggml_cuda_cast<float>(g[j1]));
    dst[i] = v;
    if constexpr (f16_copy) {
        dst_f16[i] = ggml_cuda_cast<half>(v);
    }
    if constexpr (q8_copy) {
        quantize_q8_1_lane(ggml_cuda_cast<float>(v), q8 + i/QK8_1, i % QK8_1);
    }
}

// f32 rows whose q8_1 copy the next matmul reads at its padded stride (several tokens, rows needing padding, e.g. the
//   160 columns of a split shared expert): thread i of the padded rows writes the value of its column (as
//   unary_gated_op_kernel computes it), or 0 past the row, into the copy. Whole warps: n and n_padded are multiples of 32.
template <float (*op)(float)>
static __global__ void unary_gated_q8_rows_kernel(const float * x, const float * g, float * dst, const uint32_t k_padded,
        const uint32_t n, const uint32_t n_padded, const int64_t o0, const int64_t o1, block_q8_1 * q8) {
    ggml_cuda_pdl_lc();
    const uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k_padded) {
        return;
    }
    const uint32_t row = i / n_padded;
    const uint32_t col = i - row*n_padded;

    ggml_cuda_pdl_sync();
    float v = 0.0f;
    if (col < n) {
        v = op(x[row*o0 + col]) * g[row*o1 + col];
        dst[int64_t(row)*n + col] = v;
    }
    quantize_q8_1_lane(v, q8 + i/QK8_1, i % QK8_1);
}

template <float (*op)(float), typename T>
static void unary_gated_cuda(const T * x, const T * g, T * dst, half * dst_f16, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
    const uint3 n_fd = unary_gated_n_fastdiv(k, n);
    if constexpr (std::is_same_v<T, float>) {
        if (dst_f16 != nullptr) {
            ggml_cuda_kernel_launch(unary_gated_op_kernel<op, T, true>, launch_params, x, g, dst, dst_f16, k, n, n_fd, o0, o1, nullptr, 0);
            return;
        }
    }
    GGML_ASSERT(dst_f16 == nullptr);
    ggml_cuda_kernel_launch(unary_gated_op_kernel<op, T>, launch_params, x, g, dst, dst_f16, k, n, n_fd, o0, o1, nullptr, 0);
}

template <float (*op)(float)>
void ggml_cuda_op_unary_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst, half * dst_f16 = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ((const int32_t *) dst->op_params)[1];

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (half *)dst_d, dst_f16, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        nv_bfloat16 * src0_p = (nv_bfloat16 *) src0_d;
        nv_bfloat16 * src1_p = (nv_bfloat16 *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (nv_bfloat16 *)dst_d, nullptr, ggml_nelements(dst), nc, src0_o / sizeof(nv_bfloat16), src1_o / sizeof(nv_bfloat16), stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        // the q8_1 copy for the matmul that reads the rows (at its padded stride), see ggml_cuda_q8_1_request
        if (ctx.q8_1_emit == dst && ctx.q8_1_emit_as != nullptr && ctx.q8_1_emit_as->ne[0] == nc && dst_f16 == nullptr &&
                nc % QK8_1 == 0 && ggml_nrows(dst)*GGML_PAD(nc, MATRIX_ROW_PADDING) <= INT_MAX) {
            int64_t ne_padded = 0;
            block_q8_1 * q8 = (block_q8_1 *) ctx.q8_1_emit_alloc(dst, ne_padded, ggml_nrows(dst));
            if (q8 != nullptr) {
                const int64_t k_padded = ggml_nrows(dst)*ne_padded;
                const int64_t num_blocks = (k_padded + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
                ggml_cuda_kernel_launch(unary_gated_q8_rows_kernel<op>, launch_params, src0_p, src1_p, (float *) dst_d,
                    (uint32_t) k_padded, (uint32_t) nc, (uint32_t) ne_padded, src0_o / (int64_t) sizeof(float),
                    src1_o / (int64_t) sizeof(float), q8);
                return;
            }
        }
        unary_gated_cuda<op>(src0_p, src1_p, (float *)dst_d, dst_f16, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), stream);
    }
}

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_relu>(ctx, dst);
}

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu>(ctx, dst);
}

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_silu>(ctx, dst);
}

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_quick>(ctx, dst);
}

bool ggml_cuda_op_glu_f16_copy_supported(const ggml_tensor * dst) {
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) {
        return false;
    }
    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
            return true;
        default:
            return false;
    }
}

void ggml_cuda_op_glu_f16_copy(ggml_backend_cuda_context & ctx, ggml_tensor * dst, half * dst_f16) {
    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_REGLU:
            ggml_cuda_op_unary_gated<op_relu>(ctx, dst, dst_f16);
            break;
        case GGML_GLU_OP_GEGLU:
            ggml_cuda_op_unary_gated<op_gelu>(ctx, dst, dst_f16);
            break;
        case GGML_GLU_OP_SWIGLU:
            ggml_cuda_op_unary_gated<op_silu>(ctx, dst, dst_f16);
            break;
        case GGML_GLU_OP_GEGLU_ERF:
            ggml_cuda_op_unary_gated<op_gelu_erf>(ctx, dst, dst_f16);
            break;
        case GGML_GLU_OP_GEGLU_QUICK:
            ggml_cuda_op_unary_gated<op_gelu_quick>(ctx, dst, dst_f16);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// swiglu_oai

template <typename T>
static __global__ void swiglu_oai_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float alpha, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    float xi = x[j0];
    float gi = g[j1];

    dst[i] = ggml_cuda_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
static void swiglu_oai_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float alpha, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_oai_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(x, g, dst, k, n, o0, o1, alpha, limit);
}

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    float * src0_p = (float *) src0_d;
    float * src1_p = (float *) src1_d;

    if (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_cuda(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), alpha, limit, stream);
}

// swiglu_clamp

template <typename T>
static __global__ void swiglu_clamp_kernel(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    dst[i] = (T) ggml_cuda_op_swiglu_clamp_single((float) gate[j0], (float) up[j1], limit);
}

template <typename T>
static void swiglu_clamp_cuda(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_clamp_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(gate, up, dst, k, n, o0, o1, limit);
}

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float limit = ggml_get_op_params_f32(dst, 3);

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (half *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), limit, stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (float *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), limit, stream);
    }
}

/* CUDA kernel + launcher for xIELU */

template <typename T>
static __global__ void xielu_kernel(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float xi = ggml_cuda_cast<float>(x[i]);

    const float gate_pos = (xi > 0.0f);
    const float y_pos = alpha_p * xi * xi + beta * xi;
    const float min_v_eps = fminf(xi, eps);
    const float y_neg = (expm1f(min_v_eps) - xi) * alpha_n + beta * xi;
    const float out = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;

    dst[i] = ggml_cuda_cast<T>(out);
}

template <typename T>
static void xielu_cuda(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_XIELU_BLOCK_SIZE) / CUDA_XIELU_BLOCK_SIZE;
    xielu_kernel<<<num_blocks, CUDA_XIELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, alpha_n, alpha_p, beta, eps);
}

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16 ||  dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);

    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta    = ggml_get_op_params_f32(dst, 3);
    const float eps     = ggml_get_op_params_f32(dst, 4);

    if (src0->type == GGML_TYPE_F16) {
        xielu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        xielu_cuda((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else {
        xielu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    }
}



/* silu_back */

static __device__ __forceinline__ float op_silu_back(float grad, float x) {
    const float s = 1.0f / (1.0f + expf(-x));
    return grad * s * (1.0f + x * (1.0f - s));
}

template <class T>
static __global__ void silu_back_kernel(const T * grad, const T * xf, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_silu_back((float)grad[i], (float)xf[i]);
}

template <class T>
static void silu_back_cuda(const T * grad, const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back_kernel<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_back_cuda((const half *)src0_d, (const half *)src1_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_back_cuda((const float*)src0_d, (const float*)src1_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

/* leaky relu */

static __device__ __forceinline__ float op_leaky_relu(float x, const float negative_slope) {
    return fmaxf(x, 0) + fminf(x, 0.0f) * negative_slope;
}

template <class T>
static __global__ void leaky_relu_kernel(const T * x, T * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_leaky_relu((float)x[i], negative_slope);
}

template <class T>
static void leaky_relu_cuda(const T * x, T * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_kernel<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        leaky_relu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), negative_slope, stream);
    } else {
        leaky_relu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), negative_slope, stream);
    }
}

/* fused unary + mul */

template <float (*op)(float)>
static void ggml_cuda_op_unary_mul_impl(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    // unary_node: UNARY op applied to unary_node->src[0]
    // mul_node:   MUL(a, b) where one of a/b is unary_node
    // Output goes to mul_node->data

    const ggml_tensor * unary_src = unary_node->src[0];  // input to the unary op
    const ggml_tensor * other_src = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(unary_src));
    GGML_ASSERT(unary_src->nb[0] == ggml_element_size(unary_src));
    GGML_ASSERT(ggml_is_contiguous_1(other_src));
    GGML_ASSERT(other_src->nb[0] == ggml_element_size(other_src));
    GGML_ASSERT(ggml_are_same_shape(unary_src, other_src));

    GGML_ASSERT(unary_src->type == GGML_TYPE_F32 || unary_src->type == GGML_TYPE_F16 || unary_src->type == GGML_TYPE_BF16);
    GGML_ASSERT(unary_src->type == other_src->type);
    GGML_ASSERT(unary_src->type == mul_node->type);

    cudaStream_t stream = ctx.stream();

    const int64_t k  = ggml_nelements(mul_node);
    const int64_t nc = unary_src->ne[0];
    const int64_t unary_stride = unary_src->nb[1];
    const int64_t other_stride = other_src->nb[1];

    if (unary_src->type == GGML_TYPE_F16) {
        unary_gated_cuda<op>((const half *) unary_src->data, (const half *) other_src->data,
                             (half *) mul_node->data, nullptr, k, nc,
                             unary_stride / sizeof(half), other_stride / sizeof(half), stream);
    } else if (unary_src->type == GGML_TYPE_BF16) {
        unary_gated_cuda<op>((const nv_bfloat16 *) unary_src->data, (const nv_bfloat16 *) other_src->data,
                             (nv_bfloat16 *) mul_node->data, nullptr, k, nc,
                             unary_stride / sizeof(nv_bfloat16), other_stride / sizeof(nv_bfloat16), stream);
    } else {
        unary_gated_cuda<op>((const float *) unary_src->data, (const float *) other_src->data,
                             (float *) mul_node->data, nullptr, k, nc,
                             unary_stride / sizeof(float), other_stride / sizeof(float), stream);
    }
}

// CONT of a view with rows at one stride, then unary and mul: the gated kernel reads the strided rows of the view
//   itself, so the copy is not made. The values are those of the three ops apart.
bool ggml_cuda_should_fuse_cont_unary_mul(const ggml_tensor * cont, const ggml_tensor * unary, const ggml_tensor * mul) {
    if (cont->op != GGML_OP_CONT || unary->op != GGML_OP_UNARY || mul->op != GGML_OP_MUL || unary->src[0] != cont) {
        return false;
    }
    const ggml_unary_op uop = ggml_get_unary_op(unary);
    if (uop != GGML_UNARY_OP_SILU && uop != GGML_UNARY_OP_SIGMOID && uop != GGML_UNARY_OP_SOFTPLUS) {
        return false;
    }
    const ggml_tensor * v     = cont->src[0];
    const ggml_tensor * other = mul->src[0] == unary ? mul->src[1] : (mul->src[1] == unary ? mul->src[0] : nullptr);
    if (other == nullptr || other == unary) {
        return false;
    }
    for (const ggml_tensor * t : { v, cont, unary, mul, other }) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    // the rows of v follow each other at nb1 across all of its dims, so row r of the copy is at r*nb1; the output may
    //   sit on other, which each thread reads at its own element, but not on v, read at a stride
    return v->nb[0] == sizeof(float) && v->nb[1] % sizeof(float) == 0 && v->nb[2] == v->ne[1]*v->nb[1] &&
        v->nb[3] == v->ne[2]*v->nb[2] && ggml_is_contiguous(cont) && ggml_is_contiguous(mul) && ggml_is_contiguous(other) &&
        ggml_are_same_shape(other, cont) && ggml_are_same_shape(mul, cont) &&
        !ggml_cuda_tensors_overlap(mul, v) && ggml_cuda_same_or_disjoint(mul, other);
}

// op(scale*x + bias), the form of the SCALE and UNARY kernels, so the result is bitwise theirs
template <float (*op)(float), bool q8_copy>
static __global__ void scale_unary_kernel(const float * x, float * dst, const float scale, const float bias, const int64_t k,
        block_q8_1 * q8, const int64_t k_padded) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        if constexpr (q8_copy) {
            // whole warps, k being a multiple of 32
            if (i < k_padded) {
                quantize_q8_1_lane(0.0f, q8 + i/QK8_1, i % QK8_1);
            }
        }
        return;
    }

    ggml_cuda_pdl_sync();
    const float v = op(scale * x[i] + bias);
    dst[i] = v;
    if constexpr (q8_copy) {
        quantize_q8_1_lane(v, q8 + i/QK8_1, i % QK8_1);
    }
}

bool ggml_cuda_should_fuse_scale_unary(const ggml_tensor * scale, const ggml_tensor * unary) {
    const ggml_tensor * x = scale->src[0];
    return ggml_get_unary_op(unary) == GGML_UNARY_OP_SILU && unary->src[0] == scale &&
        x->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32 && unary->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(x) && ggml_is_contiguous(unary) && ggml_are_same_shape(x, unary) &&
        ggml_cuda_same_or_disjoint(unary, x);
}

void ggml_cuda_op_scale_unary(ggml_backend_cuda_context & ctx, ggml_tensor * scale_node, ggml_tensor * unary) {
    const ggml_tensor * x = scale_node->src[0];

    float scale;
    float bias;
    memcpy(&scale, (const float *) scale_node->op_params + 0, sizeof(float));
    memcpy(&bias,  (const float *) scale_node->op_params + 1, sizeof(float));

    const int64_t k = ggml_nelements(unary);
    const float * x_d = (const float *) x->data;
    float *       d   = (float *) unary->data;

    // one token: the q8_1 copy of the output for the MMVQ matmul that reads it
    if (ggml_nrows(unary) == 1 && k % QK8_1 == 0) {
        int64_t ne_padded = 0;
        block_q8_1 * q8 = (block_q8_1 *) ctx.q8_1_emit_alloc(unary, ne_padded);
        if (q8 != nullptr) {
            const int64_t num_blocks = (ne_padded + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, ctx.stream());
            ggml_cuda_kernel_launch(scale_unary_kernel<op_silu, true>, launch_params, x_d, d, scale, bias, k, q8, ne_padded);
            return;
        }
    }

    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(scale_unary_kernel<op_silu, false>, launch_params, x_d, d, scale, bias, k, nullptr, (int64_t) 0);
}

void ggml_cuda_op_cont_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * cont, ggml_tensor * unary, ggml_tensor * mul) {
    const ggml_tensor * v     = cont->src[0];
    const ggml_tensor * other = mul->src[0] == unary ? mul->src[1] : mul->src[0];

    const int64_t k  = ggml_nelements(mul);
    const int64_t nc = v->ne[0];
    const int64_t o0 = v->nb[1] / sizeof(float);
    const float * x  = (const float *) v->data;
    const float * g  = (const float *) other->data;
    float *       d  = (float *) mul->data;

    // one token: the q8_1 copy of the output for the MMVQ matmul that reads it, see ggml_cuda_q8_1_request
    if (ggml_nrows(mul) == 1 && k % QK8_1 == 0 && ggml_get_unary_op(unary) == GGML_UNARY_OP_SIGMOID) {
        int64_t ne_padded = 0;
        block_q8_1 * q8 = (block_q8_1 *) ctx.q8_1_emit_alloc(mul, ne_padded);
        if (q8 != nullptr) {
            const int64_t num_blocks = (ne_padded + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, ctx.stream());
            ggml_cuda_kernel_launch(unary_gated_op_kernel<op_sigmoid, float, false, true>, launch_params,
                x, g, d, nullptr, k, nc, unary_gated_n_fastdiv(ne_padded, nc), o0, nc, q8, ne_padded);
            return;
        }
    }

    switch (ggml_get_unary_op(unary)) {
        case GGML_UNARY_OP_SILU:
            unary_gated_cuda<op_silu>(x, g, d, nullptr, k, nc, o0, nc, ctx.stream());
            break;
        case GGML_UNARY_OP_SIGMOID:
            unary_gated_cuda<op_sigmoid>(x, g, d, nullptr, k, nc, o0, nc, ctx.stream());
            break;
        case GGML_UNARY_OP_SOFTPLUS:
            unary_gated_cuda<op_softplus>(x, g, d, nullptr, k, nc, o0, nc, ctx.stream());
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SILU:
            ggml_cuda_op_unary_mul_impl<op_silu>(ctx, unary_node, mul_node);
            break;
        case GGML_UNARY_OP_SIGMOID:
            ggml_cuda_op_unary_mul_impl<op_sigmoid>(ctx, unary_node, mul_node);
            break;
        case GGML_UNARY_OP_SOFTPLUS:
            ggml_cuda_op_unary_mul_impl<op_softplus>(ctx, unary_node, mul_node);
            break;
        default:
            GGML_ABORT("Unsupported unary op for fused unary+mul");
    }
}

/* fused add + softplus + mul, both by a row: the gate of a gated delta net layer, softplus(alpha + dt) * a,
   three launches per layer and token on a few dozen values. Each op is the one its own kernel runs. */

static __global__ void add_softplus_mul_f32(const float * x, const float * bias, const float * scale, float * dst,
                                            const int k, const uint3 ne0_packed) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const uint32_t i0 = fastmodulo((uint32_t) i, ne0_packed);
    dst[i] = op_softplus(x[i] + bias[i0]) * scale[i0];
}

static bool ggml_cuda_is_f32_row(const ggml_tensor * t, int64_t ne0) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == ne0 && ggml_nelements(t) == ne0;
}

bool ggml_cuda_should_fuse_add_softplus_mul(const ggml_tensor * add, const ggml_tensor * unary, const ggml_tensor * mul) {
    if (add->op != GGML_OP_ADD || unary->op != GGML_OP_UNARY || ggml_get_unary_op(unary) != GGML_UNARY_OP_SOFTPLUS ||
            mul->op != GGML_OP_MUL || unary->src[0] != add || (mul->src[0] != unary && mul->src[1] != unary)) {
        return false;
    }
    const int64_t ne0 = add->ne[0];
    const ggml_tensor * scale = mul->src[0] == unary ? mul->src[1] : mul->src[0];
    // the add takes the full tensor and a row, in either order
    const ggml_tensor * x = ggml_are_same_shape(add->src[0], add) ? add->src[0] : add->src[1];
    const ggml_tensor * b = x == add->src[0] ? add->src[1] : add->src[0];
    // dst[i] is written by the thread that reads x[i], and every thread reads the rows b and scale
    return add->type == GGML_TYPE_F32 && unary->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
        x->type == GGML_TYPE_F32 && ggml_are_same_shape(x, add) && ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_are_same_shape(mul, add) && ggml_is_contiguous(mul) &&
        ggml_cuda_is_f32_row(b, ne0) && ggml_cuda_is_f32_row(scale, ne0) && ggml_nelements(add) <= INT_MAX &&
        ggml_cuda_same_or_disjoint(mul, x) && !ggml_cuda_tensors_overlap(mul, b) && !ggml_cuda_tensors_overlap(mul, scale);
}

void ggml_cuda_op_add_softplus_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * unary, ggml_tensor * mul) {
    const ggml_tensor * scale = mul->src[0] == unary ? mul->src[1] : mul->src[0];
    const ggml_tensor * x     = ggml_are_same_shape(add->src[0], add) ? add->src[0] : add->src[1];
    const ggml_tensor * b     = x == add->src[0] ? add->src[1] : add->src[0];

    const int k = (int) ggml_nelements(mul);
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    add_softplus_mul_f32<<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) b->data, (const float *) scale->data, (float *) mul->data,
        k, init_fastdiv_values((uint32_t) add->ne[0]));
}

/* fused relu + sqr */

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node) {
    const ggml_tensor * src = relu_node->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(src->type == sqr_node->type);

    const int k = ggml_nelements(src);
    if (src->type == GGML_TYPE_F16) {
        unary_cuda<op_relu_sqr>((const half *)src->data, (half *)sqr_node->data, k, stream);
    } else {
        unary_cuda<op_relu_sqr>((const float *)src->data, (float *)sqr_node->data, k, stream);
    }
}

/* fused mul + add + sigmoid + scale */

// every step rounds like its own kernel would, so the result is bit-identical to the unfused chain
static __global__ void mul_add_sigmoid_scale_f32(const float * x, const float * m, const float * a, float * dst,
        const int ne0, const int k, const int64_t s1, const int ne_m, const int ne_a, const float scale, const float bias) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    const int i0 = i % ne0;
    const int i1 = i / ne0;
    const float v = op_sigmoid(__fadd_rn(__fmul_rn(x[i1*s1 + i0], m[i0 % ne_m]), a[i0 % ne_a]));
    dst[i] = scale * v + bias;
}

void ggml_cuda_op_mul_add_sigmoid_scale(ggml_backend_cuda_context & ctx, ggml_tensor * mul_node, ggml_tensor * add_node,
        ggml_tensor * scale_node) {
    const ggml_tensor * x = mul_node->src[0];
    const ggml_tensor * m = mul_node->src[1];
    const ggml_tensor * a = add_node->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && m->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_node->type == GGML_TYPE_F32 && ggml_is_contiguous(scale_node));
    GGML_ASSERT(x->nb[0] == sizeof(float) && ggml_nrows(x) == x->ne[1]);
    GGML_ASSERT(ggml_is_contiguous(m) && ggml_nrows(m) == 1 && x->ne[0] % m->ne[0] == 0);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_nrows(a) == 1 && x->ne[0] % a->ne[0] == 0);

    float scale;
    float bias;
    memcpy(&scale, (const float *) scale_node->op_params + 0, sizeof(float));
    memcpy(&bias,  (const float *) scale_node->op_params + 1, sizeof(float));

    const int k = ggml_nelements(scale_node);
    const int num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(mul_add_sigmoid_scale_f32, launch_params,
        (const float *) x->data, (const float *) m->data, (const float *) a->data, (float *) scale_node->data,
        (int) x->ne[0], k, (int64_t) (x->nb[1] / sizeof(float)), (int) m->ne[0], (int) a->ne[0], scale, bias);
}

/* fused sigmoid + mul + add */

// dst = base + x * sigmoid(g[row]), each step rounded like its own kernel
static __global__ void sigmoid_mul_add_f32(const float * g, const float * x, const float * base, float * dst,
        const int ncols, const uint3 ncols_fd, const int64_t k) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // the row without a 64-bit division where the indices fit 31 bits, see unary_gated_op_kernel
    const int64_t row = ncols_fd.z != 0 ? (int64_t) fastdiv((uint32_t) i, ncols_fd) : i / ncols;

    ggml_cuda_pdl_sync();
    const float sg = op_sigmoid(g[row]);
    dst[i] = __fadd_rn(base[i], __fmul_rn(x[i], sg));
}

bool ggml_cuda_should_fuse_sigmoid_mul_add(const ggml_tensor * sigmoid, const ggml_tensor * mul, const ggml_tensor * add) {
    if (sigmoid->op != GGML_OP_UNARY || ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID || mul->op != GGML_OP_MUL ||
            add->op != GGML_OP_ADD) {
        return false;
    }
    const ggml_tensor * g    = sigmoid->src[0];
    const ggml_tensor * x    = mul->src[0] == sigmoid ? mul->src[1] : (mul->src[1] == sigmoid ? mul->src[0] : nullptr);
    const ggml_tensor * base = add->src[0] == mul ? add->src[1] : (add->src[1] == mul ? add->src[0] : nullptr);
    if (x == nullptr || base == nullptr || x == sigmoid || base == mul) {
        return false;
    }
    for (const ggml_tensor * t : { g, x, base, sigmoid, mul, add }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            return false;
        }
    }
    // one gate per row; the output may sit exactly on x or base (each element is read before it is written), not on g
    return g->ne[0] == 1 && ggml_nelements(g) == ggml_nrows(x) && ggml_are_same_shape(x, mul) &&
        ggml_are_same_shape(base, add) && ggml_are_same_shape(mul, add) && ggml_nelements(add) <= INT64_MAX/2 &&
        ggml_cuda_same_or_disjoint(add, x) && ggml_cuda_same_or_disjoint(add, base) && !ggml_cuda_tensors_overlap(add, g);
}

void ggml_cuda_op_sigmoid_mul_add(ggml_backend_cuda_context & ctx, ggml_tensor * sigmoid, ggml_tensor * mul, ggml_tensor * add) {
    const ggml_tensor * g    = sigmoid->src[0];
    const ggml_tensor * x    = mul->src[0] == sigmoid ? mul->src[1] : mul->src[0];
    const ggml_tensor * base = add->src[0] == mul ? add->src[1] : add->src[0];

    const int64_t k = ggml_nelements(add);
    const int64_t num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3) num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(sigmoid_mul_add_f32, launch_params,
        (const float *) g->data, (const float *) x->data, (const float *) base->data, (float *) add->data, (int) x->ne[0],
        unary_gated_n_fastdiv(k, x->ne[0]), k);
}

/* fused mul + sigmoid + scale */

// each step rounds like its own kernel: m is read as the MUL broadcasts it. pre: a SCALE between the MUL and the
//   SIGMOID (the decay gate of a KDA layer, lb * sigmoid(-(g * A))), an FFMA as scale_f32 rounds it.
template <bool pre>
static __global__ void mul_sigmoid_scale_f32(const float * x, const float * m, float * dst, const int k,
        const int ne0, const int ne1, const int ne2, const int m0, const int m1, const int m2, const int m3,
        const float scale, const float bias, const float pre_scale, const float pre_bias) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    const int i0 = i % ne0;
    const int i1 = (i / ne0) % ne1;
    const int i2 = (i / (ne0*ne1)) % ne2;
    const int i3 = i / (ne0*ne1*ne2);
    const float mv = m[((i3 % m3)*m2 + i2 % m2)*m1*m0 + (i1 % m1)*m0 + i0 % m0];
    float t = __fmul_rn(x[i], mv);
    if constexpr (pre) {
        t = __fmaf_rn(pre_scale, t, pre_bias);
    }
    const float v  = op_sigmoid(t);
    dst[i] = scale * v + bias;
}

bool ggml_cuda_should_fuse_mul_sigmoid_scale(const ggml_tensor * mul, const ggml_tensor * pre, const ggml_tensor * sigmoid,
        const ggml_tensor * scale) {
    const ggml_tensor * x = mul->src[0];
    const ggml_tensor * m = mul->src[1];
    static const bool no_pre = getenv("GGML_CUDA_NO_MUL_SCALE_SIGMOID_SCALE") != nullptr;
    if (pre != nullptr && (no_pre || pre->op != GGML_OP_SCALE || pre->src[0] != mul || pre->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(pre, mul))) {
        return false;
    }
    if (sigmoid->src[0] != (pre ? pre : mul) || scale->src[0] != sigmoid || ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID ||
            x->type != GGML_TYPE_F32 || m->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
            sigmoid->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(x) || !ggml_is_contiguous(m) || !ggml_is_contiguous(scale) ||
            !ggml_are_same_shape(x, mul) || !ggml_are_same_shape(mul, scale) || !ggml_can_repeat(m, x) ||
            ggml_nelements(scale) > INT_MAX) {
        return false;
    }
    // the output may sit on x (each element is read before it is written), not on m
    return !ggml_cuda_tensors_overlap(scale, m) && (!ggml_cuda_tensors_overlap(scale, x) || scale->data == x->data);
}

void ggml_cuda_op_mul_sigmoid_scale(ggml_backend_cuda_context & ctx, ggml_tensor * mul, const ggml_tensor * pre,
        ggml_tensor * scale_node) {
    const ggml_tensor * x = mul->src[0];
    const ggml_tensor * m = mul->src[1];

    float scale;
    float bias;
    memcpy(&scale, (const float *) scale_node->op_params + 0, sizeof(float));
    memcpy(&bias,  (const float *) scale_node->op_params + 1, sizeof(float));
    float pre_scale = 1.0f;
    float pre_bias  = 0.0f;
    if (pre) {
        memcpy(&pre_scale, (const float *) pre->op_params + 0, sizeof(float));
        memcpy(&pre_bias,  (const float *) pre->op_params + 1, sizeof(float));
    }

    const int k = ggml_nelements(scale_node);
    const int num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(pre ? mul_sigmoid_scale_f32<true> : mul_sigmoid_scale_f32<false>, launch_params,
        (const float *) x->data, (const float *) m->data, (float *) scale_node->data, k,
        (int) x->ne[0], (int) x->ne[1], (int) x->ne[2], (int) m->ne[0], (int) m->ne[1], (int) m->ne[2], (int) m->ne[3],
        scale, bias, pre_scale, pre_bias);
}
