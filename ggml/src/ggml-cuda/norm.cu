#include "norm.cuh"
#include "unary.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

// With cols_per_thread > 0 a thread holds its columns of the row in registers and issues every load before the first
//   use: a decode row is one block on one SM, and the column loop waits on each of its loads in turn (1.7x the time at
//   5120 columns). A thread sums its columns in the same order either way, so the result does not change.
template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false, int cols_per_thread = 0>
static __global__ void __launch_bounds__(block_size) rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_add, "fusing scale is not supported with adding");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp
    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();
    if constexpr (cols_per_thread > 0) {
        float xv[cols_per_thread];
        [[maybe_unused]] float mv[cols_per_thread];
        [[maybe_unused]] float av[cols_per_thread];
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = tid + i*block_size;
            xv[i] = col < ncols ? x[col] : 0.0f;
        }
        if constexpr (do_multiply) {
#pragma unroll
            for (int i = 0; i < cols_per_thread; ++i) {
                const int col = tid + i*block_size;
                mv[i] = col < ncols ? mul[fastmodulo(col, mul_ncols_packed)] : 0.0f;
                if constexpr (do_add) {
                    av[i] = col < ncols ? add[fastmodulo(col, add_ncols_packed)] : 0.0f;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            if (tid + i*block_size < ncols) {
                tmp += xv[i] * xv[i];
            }
        }

        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

        const float mean = tmp / ncols;
        const float scale = rsqrtf(mean + eps);

#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = tid + i*block_size;
            if (col >= ncols) {
                continue;
            }
            if constexpr (do_multiply && do_add) {
                dst[col] = scale * xv[i] * mv[i] + av[i];
            } else if constexpr (do_multiply && do_scale) {
                dst[col] = (scale_out * (scale * xv[i])) * mv[i];
            } else if constexpr (do_multiply) {
                dst[col] = scale * xv[i] * mv[i];
            } else if constexpr (do_scale) {
                dst[col] = scale_out * (scale * xv[i]);
            } else {
                dst[col] = scale * xv[i];
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = x[col];
            tmp += xi * xi;
        }

        // sum up partial sums
        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

        const float mean = tmp / ncols;
        const float scale = rsqrtf(mean + eps);

        for (int col = tid; col < ncols; col += block_size) {
            if constexpr (do_multiply && do_add) {
                const int mul_col = fastmodulo(col, mul_ncols_packed);
                const int add_col = fastmodulo(col, add_ncols_packed);
                dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
            } else if constexpr (do_multiply && do_scale) {
                const int mul_col = fastmodulo(col, mul_ncols_packed);
                dst[col]          = (scale_out * (scale * x[col])) * mul[mul_col];
            } else if constexpr (do_multiply) {
                const int mul_col = fastmodulo(col, mul_ncols_packed);
                dst[col]          = scale * x[col] * mul[mul_col];
            } else if constexpr (do_scale) {
                dst[col] = scale_out * (scale * x[col]);
            } else {
                dst[col] = scale * x[col];
            }
        }
    }
}

// sum = a + b and dst = rms_norm(sum) * mul in one pass: a residual add and the norm that follows it, which also
//   leaves the sum for the next residual. It reduces exactly like rms_norm_f32 with the same block size, so both
//   results are those of the three ops apart. a, b, sum and dst are contiguous. cols_per_thread as in rms_norm_f32.
//   With gated, sum = (a + c * sigmoid(g[row])) + b: the sigmoid-gated shared expert c of a MoE layer is added to its
//   routed output a before the residual b, rounded like the SIGMOID, MUL and ADDs apart.
template <int block_size, int cols_per_thread = 0, bool gated = false>
static __global__ void __launch_bounds__(block_size) rms_norm_pre_add_f32(const float * a, const float * b, float * sum, float * dst, const int ncols,
                                            const float eps, const float * mul, const int64_t mul_stride_row,
                                            const int64_t mul_stride_channel, const int64_t mul_stride_sample,
                                            const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
                                            const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
                                            const float * c = nullptr, const float * g = nullptr) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    const int64_t row_flat = (int64_t(sample)*nchannels + channel)*nrows + row;
    const int64_t offset   = row_flat*ncols;
    a   += offset;
    b   += offset;
    sum += offset;
    dst += offset;
    if constexpr (gated) {
        c += offset;
    }

    const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
    mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

    float tmp = 0.0f;
    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();
    [[maybe_unused]] float gate = 0.0f;
    if constexpr (gated) {
        gate = ggml_cuda_op_sigmoid_single(g[row_flat]);
    }
    const auto x_at = [&](const int col) {
        if constexpr (gated) {
            return (a[col] + __fmul_rn(c[col], gate)) + b[col];
        } else {
            return a[col] + b[col];
        }
    };

    if constexpr (cols_per_thread > 0) {
        float xv[cols_per_thread];
        float mv[cols_per_thread];
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = tid + i*block_size;
            xv[i] = col < ncols ? x_at(col) : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = tid + i*block_size;
            mv[i] = col < ncols ? mul[fastmodulo(col, mul_ncols_packed)] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = tid + i*block_size;
            if (col < ncols) {
                sum[col] = xv[i];
                tmp += xv[i] * xv[i];
            }
        }

        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

        const float mean  = tmp / ncols;
        const float scale = rsqrtf(mean + eps);

#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = tid + i*block_size;
            if (col < ncols) {
                dst[col] = scale * xv[i] * mv[i];
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = x_at(col);
            sum[col] = xi;
            tmp += xi * xi;
        }

        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

        const float mean  = tmp / ncols;
        const float scale = rsqrtf(mean + eps);

        for (int col = tid; col < ncols; col += block_size) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col] = scale * sum[col] * mul[mul_col];
        }
    }
}

// y = rms_norm(x) * w1 + r, with has_scale y *= s (one value or one per column), and dst = rms_norm(y) * w2: the
//   post-norm of a gemma attention or FFN block, its residual add (and the layer output scale) and the pre-norm of the
//   next block on the same row in one launch. Both norms reduce like rms_norm_f32 with the same block size and the y
//   of the separate kernels is written, so both outputs are those of the ops apart. Every input is read before the
//   first reduction, so an output may sit exactly on an input; the second reduction has its own shared slots.
template <int block_size, int cols_per_thread, bool has_scale>
static __global__ void __launch_bounds__(block_size) rms_norm_add_rms_norm_f32(
        const float * x, const float * w1, const float * r, const float * s, const float * w2, float * y_out, float * dst,
        const int ncols, const float eps1, const float eps2, const bool s_per_col) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;
    const int tid       = threadIdx.x;

    const int64_t offset = ((int64_t(blockIdx.z)*nchannels + blockIdx.y)*nrows + blockIdx.x)*ncols;
    x     += offset;
    r     += offset;
    y_out += offset;
    dst   += offset;

    float xv[cols_per_thread];
    float w1v[cols_per_thread];
    float rv[cols_per_thread];
    float w2v[cols_per_thread];
    [[maybe_unused]] float sv[cols_per_thread];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        const int col = tid + i*block_size;
        xv[i] = col < ncols ? x[col] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        const int col = tid + i*block_size;
        w1v[i] = col < ncols ? w1[col] : 0.0f;
        rv[i]  = col < ncols ? r[col]  : 0.0f;
        w2v[i] = col < ncols ? w2[col] : 0.0f;
        if constexpr (has_scale) {
            sv[i] = col < ncols ? s[s_per_col ? col : 0] : 0.0f;
        }
    }

    extern __shared__ float s_sum[];

    float tmp = 0.0f;
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        if (tid + i*block_size < ncols) {
            tmp += xv[i] * xv[i];
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale1 = rsqrtf(tmp / ncols + eps1);

    float tmp2 = 0.0f;
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        const int col = tid + i*block_size;
        if (col >= ncols) {
            continue;
        }
        float y = scale1 * xv[i] * w1v[i] + rv[i];
        if constexpr (has_scale) {
            y = __fmul_rn(y, sv[i]);
        }
        y_out[col] = y;
        xv[i]      = y;
        tmp2 += y * y;
    }
    tmp2 = block_reduce<block_reduce_method::SUM, block_size>(tmp2, s_sum + 32);
    const float scale2 = rsqrtf(tmp2 / ncols + eps2);

#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        const int col = tid + i*block_size;
        if (col < ncols) {
            dst[col] = scale2 * xv[i] * w2v[i];
        }
    }
}

// dst = op(gate) * (rms_norm(x) * mul): the gated output norm of a gated delta net layer (build_norm_gated) in one launch
//   instead of two. The norm reduces exactly like rms_norm_f32 with the same block size and the gate is applied like in
//   unary_gated_op_kernel, so the values are those of the separate kernels. x and gate have contiguous rows, dst is
//   contiguous.
template <int block_size, float (*op)(float)>
static __global__ void rms_norm_gated_f32(const float * x, const float * gate, float * dst, const int ncols,
                                          const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
                                          const int64_t gate_stride_row, const int64_t gate_stride_channel,
                                          const int64_t gate_stride_sample, const float eps, const float * mul,
                                          const int64_t mul_stride_row, const int64_t mul_stride_channel,
                                          const int64_t mul_stride_sample, const uint3 mul_ncols_packed,
                                          const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed,
                                          const uint3 mul_nsamples_packed) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    x    += sample*stride_sample + channel*stride_channel + row*stride_row;
    gate += sample*gate_stride_sample + channel*gate_stride_channel + row*gate_stride_row;
    dst  += ((int64_t(sample)*nchannels + channel)*nrows + row)*ncols;

    const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
    mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const int   mul_col = fastmodulo(col, mul_ncols_packed);
        const float normed  = scale * x[col] * mul[mul_col];
        dst[col] = op(gate[col]) * normed;
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

// calls launch(std::integral_constant<int, n>()) with the cols_per_thread of the rms_norm kernels, n = ceil(ncols /
//   block_size) up to n_max and 0 (the column loop) above it. n fits the row exactly: unused register slots still
//   issue their predicated loads, ~0.5 us on a one-block row.
template <int block_size, int n_max, int n = 1, typename F>
static void rms_norm_launch_cols(const int ncols, F && launch) {
    if constexpr (n > n_max) {
        launch(std::integral_constant<int, 0>());
    } else {
        if (ncols <= n*block_size) {
            launch(std::integral_constant<int, n>());
        } else {
            rms_norm_launch_cols<block_size, n_max, n + 1>(ncols, launch);
        }
    }
}

template <bool do_scale = false>
static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream,
        const float scale_out = 1.0f) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        rms_norm_launch_cols<256, 4>(ncols, [&](auto cols) {
            ggml_cuda_kernel_launch(rms_norm_f32<256, false, false, do_scale, decltype(cols)::value>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
            // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out);
        });
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        rms_norm_launch_cols<1024, 8>(ncols, [&](auto cols) {
            ggml_cuda_kernel_launch(rms_norm_f32<1024, false, false, do_scale, decltype(cols)::value>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
            // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out);
        });
    }
}

// do_scale multiplies the norm by scale_out before the mul, as a SCALE between them
template <bool do_scale = false>
static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream,
                                  const float    scale_out = 1.0f) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    GGML_ASSERT(!do_scale || (mul != nullptr && add == nullptr));
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            rms_norm_launch_cols<256, 4>(ncols, [&](auto cols) {
                ggml_cuda_kernel_launch(rms_norm_f32<256, true, false, do_scale, decltype(cols)::value>, launch_params,
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                    mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    // underlying cudaLaunchKernelEx does not support default params
                nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out);
            });
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            rms_norm_launch_cols<1024, 8>(ncols, [&](auto cols) {
                ggml_cuda_kernel_launch(rms_norm_f32<1024, true, false, do_scale, decltype(cols)::value>, launch_params,
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                    mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    // underlying cudaLaunchKernelEx does not support default params
                nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), scale_out);
            });
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            rms_norm_launch_cols<256, 4>(ncols, [&](auto cols) {
                ggml_cuda_kernel_launch(rms_norm_f32<256, true, true, false, decltype(cols)::value>, launch_params,
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                    mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                    add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                    add_nchannels_packed, add_nsamples_packed, 1.0f);
            });
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            rms_norm_launch_cols<1024, 8>(ncols, [&](auto cols) {
                ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true, false, decltype(cols)::value>, launch_params,
                    x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                    mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                    add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                    add_nchannels_packed, add_nsamples_packed, 1.0f);
            });
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) scale_tensor->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    float scale;
    memcpy(&scale, (const float *) scale_tensor->op_params + 0, sizeof(float));

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda<true>(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream, scale);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                                 const ggml_tensor * scale_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    // the mul reads the scaled norm when a SCALE sits between them
    const ggml_tensor * mul_in = scale_tensor ? scale_tensor : dst;
    float scale_out = 1.0f;
    if (scale_tensor) {
        GGML_ASSERT(scale_tensor->src[0] == dst && ggml_get_op_params_f32(scale_tensor, 1) == 0.0f);
        scale_out = ggml_get_op_params_f32(scale_tensor, 0);
    }

    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == mul_in) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == mul_in) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    if (scale_tensor) {
        rms_norm_mul_f32_cuda<true>(src0_d, mul_d, nullptr, dst_d,
                                    ne00, ne01, ne02, ne03,
                                    /*s00*/ s01, s02, s03,
                                    /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                                    mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                                    /*add_s00*/ 0, 0, 0,
                                    0, 0, 0, 0,
                                    eps, stream, scale_out);
        return;
    }
    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

bool ggml_cuda_should_fuse_rms_norm_pre_add(const ggml_tensor * add, const ggml_tensor * rms_norm, const ggml_tensor * mul) {
    if (add->op != GGML_OP_ADD || rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rms_norm->src[0] != add) {
        return false;
    }
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : (mul->src[1] == rms_norm ? mul->src[0] : nullptr);
    if (w == nullptr || w == rms_norm) {
        return false;
    }
    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];
    for (const ggml_tensor * t : { a, b, add, rms_norm, mul, w }) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    // no broadcast in the add, contiguous rows everywhere, and a weight the norm's mul can broadcast. A block reads its
    //   rows of a and b and writes its row of dst only after the reduction, so an output may sit exactly on an input
    //   but not partly over one: ggml-alloc frees b once the add is allocated, and the elided norm (with the mul in
    //   place over it) can then start a few rows before b - one block overwrites rows another has not read yet
    return ggml_are_same_shape(a, add) && ggml_are_same_shape(b, add) && ggml_are_same_shape(mul, add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(add) && ggml_is_contiguous(mul) &&
        w->nb[0] == sizeof(float) && ggml_can_repeat(w, add) &&
        add->ne[1] <= INT_MAX && add->ne[2] <= 65535 && add->ne[3] <= 65535 &&
        ggml_cuda_same_or_disjoint(add, a) && ggml_cuda_same_or_disjoint(add, b) && ggml_cuda_same_or_disjoint(mul, a) &&
        ggml_cuda_same_or_disjoint(mul, b) && ggml_cuda_same_or_disjoint(mul, add) &&
        !ggml_cuda_tensors_overlap(add, w) && !ggml_cuda_tensors_overlap(mul, w);
}

template <bool gated>
static void rms_norm_pre_add_cuda(ggml_backend_cuda_context & ctx, const float * a_d, const float * b_d, const float * c_d,
                                  const float * g_d, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul) {
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const int ncols = add->ne[0];
    const dim3 blocks_num(add->ne[1], add->ne[2], add->ne[3]);

    const size_t ts_w = ggml_type_size(w->type);
    const uint3 mul_ncols_packed     = init_fastdiv_values(w->ne[0]);
    const uint3 mul_nrows_packed     = init_fastdiv_values(w->ne[1]);
    const uint3 mul_nchannels_packed = init_fastdiv_values(w->ne[2]);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(w->ne[3]);

    float       * sum_d = (float *) add->data;
    float       * dst_d = (float *) mul->data;
    const float * w_d   = (const float *) w->data;

    // the block size rms_norm_f32 would pick, which fixes the order of the reduction
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, ctx.stream()};
        rms_norm_launch_cols<256, 4>(ncols, [&](auto cols) {
            ggml_cuda_kernel_launch(rms_norm_pre_add_f32<256, decltype(cols)::value, gated>, launch_params, a_d, b_d, sum_d, dst_d, ncols, eps, w_d,
                (int64_t) (w->nb[1]/ts_w), (int64_t) (w->nb[2]/ts_w), (int64_t) (w->nb[3]/ts_w),
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, c_d, g_d);
        });
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, ctx.stream()};
        rms_norm_launch_cols<1024, 8>(ncols, [&](auto cols) {
            ggml_cuda_kernel_launch(rms_norm_pre_add_f32<1024, decltype(cols)::value, gated>, launch_params, a_d, b_d, sum_d, dst_d, ncols, eps, w_d,
                (int64_t) (w->nb[1]/ts_w), (int64_t) (w->nb[2]/ts_w), (int64_t) (w->nb[3]/ts_w),
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, c_d, g_d);
        });
    }
}

void ggml_cuda_op_rms_norm_pre_add(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul) {
    rms_norm_pre_add_cuda<false>(ctx, (const float *) add->src[0]->data, (const float *) add->src[1]->data, nullptr, nullptr,
                                 add, rms_norm, mul);
}

// the inputs of the gated chain: a the routed experts, c the shared expert, g its gate, b the residual
static bool ggml_cuda_rms_norm_pre_add_gated_srcs(const ggml_tensor * sigmoid, const ggml_tensor * mul_gate,
                                                  const ggml_tensor * add_shexp, const ggml_tensor * add,
                                                  const ggml_tensor *& a, const ggml_tensor *& b, const ggml_tensor *& c,
                                                  const ggml_tensor *& g) {
    if (sigmoid->op != GGML_OP_UNARY || ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID || mul_gate->op != GGML_OP_MUL ||
            add_shexp->op != GGML_OP_ADD || add->op != GGML_OP_ADD) {
        return false;
    }
    c = mul_gate->src[0] == sigmoid ? mul_gate->src[1] : (mul_gate->src[1] == sigmoid ? mul_gate->src[0] : nullptr);
    a = add_shexp->src[0] == mul_gate ? add_shexp->src[1] : (add_shexp->src[1] == mul_gate ? add_shexp->src[0] : nullptr);
    b = add->src[0] == add_shexp ? add->src[1] : (add->src[1] == add_shexp ? add->src[0] : nullptr);
    g = sigmoid->src[0];
    return c != nullptr && c != sigmoid && a != nullptr && a != mul_gate && b != nullptr && b != add_shexp;
}

bool ggml_cuda_should_fuse_rms_norm_pre_add_gated(const ggml_tensor * sigmoid, const ggml_tensor * mul_gate, const ggml_tensor * add_shexp,
                                                  const ggml_tensor * add, const ggml_tensor * rms_norm, const ggml_tensor * mul) {
    const ggml_tensor * a;
    const ggml_tensor * b;
    const ggml_tensor * c;
    const ggml_tensor * g;
    if (!ggml_cuda_rms_norm_pre_add_gated_srcs(sigmoid, mul_gate, add_shexp, add, a, b, c, g) ||
            rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rms_norm->src[0] != add) {
        return false;
    }
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : (mul->src[1] == rms_norm ? mul->src[0] : nullptr);
    if (w == nullptr || w == rms_norm) {
        return false;
    }
    for (const ggml_tensor * t : { a, b, c, g, w, sigmoid, mul_gate, add_shexp, add, rms_norm, mul }) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    // one gate value per row; the outputs as in ggml_cuda_should_fuse_rms_norm_pre_add, and neither may touch g or w,
    //   which every thread of the row reads
    return ggml_are_same_shape(a, add) && ggml_are_same_shape(b, add) && ggml_are_same_shape(c, add) &&
        ggml_are_same_shape(mul, add) && g->ne[0] == 1 && ggml_nelements(g) == ggml_nrows(add) && ggml_is_contiguous(g) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(c) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(mul) && w->nb[0] == sizeof(float) && ggml_can_repeat(w, add) &&
        add->ne[1] <= INT_MAX && add->ne[2] <= 65535 && add->ne[3] <= 65535 &&
        ggml_cuda_same_or_disjoint(add, a) && ggml_cuda_same_or_disjoint(add, b) && ggml_cuda_same_or_disjoint(add, c) &&
        ggml_cuda_same_or_disjoint(mul, a) && ggml_cuda_same_or_disjoint(mul, b) && ggml_cuda_same_or_disjoint(mul, c) &&
        ggml_cuda_same_or_disjoint(mul, add) && !ggml_cuda_tensors_overlap(add, g) && !ggml_cuda_tensors_overlap(mul, g) &&
        !ggml_cuda_tensors_overlap(add, w) && !ggml_cuda_tensors_overlap(mul, w);
}

void ggml_cuda_op_rms_norm_pre_add_gated(ggml_backend_cuda_context & ctx, ggml_tensor * sigmoid, ggml_tensor * mul_gate,
                                         ggml_tensor * add_shexp, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul) {
    const ggml_tensor * a;
    const ggml_tensor * b;
    const ggml_tensor * c;
    const ggml_tensor * g;
    GGML_ASSERT(ggml_cuda_rms_norm_pre_add_gated_srcs(sigmoid, mul_gate, add_shexp, add, a, b, c, g));
    rms_norm_pre_add_cuda<true>(ctx, (const float *) a->data, (const float *) b->data, (const float *) c->data,
                                (const float *) g->data, add, rms_norm, mul);
}

// the other operand of a binary op whose one operand is t, or nullptr
static const ggml_tensor * ggml_cuda_other_src(const ggml_tensor * op, const ggml_tensor * t) {
    return op->src[0] == t ? op->src[1] : (op->src[1] == t ? op->src[0] : nullptr);
}

bool ggml_cuda_should_fuse_rms_norm_add_rms_norm(const ggml_tensor * norm1, const ggml_tensor * mul1, const ggml_tensor * add,
                                                 const ggml_tensor * scale, const ggml_tensor * norm2, const ggml_tensor * mul2) {
    if (norm1->op != GGML_OP_RMS_NORM || mul1->op != GGML_OP_MUL || add->op != GGML_OP_ADD ||
            (scale != nullptr && scale->op != GGML_OP_MUL) || norm2->op != GGML_OP_RMS_NORM || mul2->op != GGML_OP_MUL) {
        return false;
    }
    const ggml_tensor * y  = scale != nullptr ? scale : add;
    const ggml_tensor * x  = norm1->src[0];
    const ggml_tensor * w1 = ggml_cuda_other_src(mul1, norm1);
    const ggml_tensor * r  = ggml_cuda_other_src(add, mul1);
    const ggml_tensor * s  = scale != nullptr ? ggml_cuda_other_src(scale, add) : nullptr;
    const ggml_tensor * w2 = ggml_cuda_other_src(mul2, norm2);
    if (w1 == nullptr || w1 == norm1 || r == nullptr || r == mul1 || (scale != nullptr && (s == nullptr || s == add)) ||
            norm2->src[0] != y || w2 == nullptr || w2 == norm2) {
        return false;
    }
    for (const ggml_tensor * t : { x, w1, r, w2, norm1, mul1, add, norm2, mul2 }) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    const int64_t ncols = x->ne[0];
    if (scale != nullptr && (s->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(s) ||
            !(ggml_nelements(s) == 1 || (s->ne[0] == ncols && ggml_nelements(s) == ncols)) || !ggml_are_same_shape(scale, x))) {
        return false;
    }
    // rows held in registers (at most 8 columns per thread), weights of one row, and outputs that sit exactly on an
    //   input or apart from it: blocks of other rows still read their rows when this one writes
    return ncols <= 8*1024 && ggml_is_contiguous(x) && ggml_is_contiguous(r) && ggml_are_same_shape(r, x) &&
        ggml_are_same_shape(add, x) && ggml_are_same_shape(mul2, x) && ggml_is_contiguous(y) && ggml_is_contiguous(mul2) &&
        ggml_is_contiguous(w1) && ggml_nelements(w1) == ncols && ggml_is_contiguous(w2) && ggml_nelements(w2) == ncols &&
        x->ne[1] <= INT_MAX && x->ne[2] <= 65535 && x->ne[3] <= 65535 &&
        ggml_cuda_same_or_disjoint(y, x) && ggml_cuda_same_or_disjoint(y, r) && ggml_cuda_same_or_disjoint(mul2, x) &&
        ggml_cuda_same_or_disjoint(mul2, r) && !ggml_cuda_tensors_overlap(y, mul2) &&
        !ggml_cuda_tensors_overlap(y, w1) && !ggml_cuda_tensors_overlap(y, w2) &&
        !ggml_cuda_tensors_overlap(mul2, w1) && !ggml_cuda_tensors_overlap(mul2, w2) &&
        (s == nullptr || (!ggml_cuda_tensors_overlap(y, s) && !ggml_cuda_tensors_overlap(mul2, s)));
}

void ggml_cuda_op_rms_norm_add_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * norm1, ggml_tensor * mul1, ggml_tensor * add,
                                        ggml_tensor * scale, ggml_tensor * norm2, ggml_tensor * mul2) {
    const ggml_tensor * x  = norm1->src[0];
    const ggml_tensor * w1 = ggml_cuda_other_src(mul1, norm1);
    const ggml_tensor * r  = ggml_cuda_other_src(add, mul1);
    const ggml_tensor * s  = scale != nullptr ? ggml_cuda_other_src(scale, add) : nullptr;
    const ggml_tensor * w2 = ggml_cuda_other_src(mul2, norm2);
    ggml_tensor       * y  = scale != nullptr ? scale : add;

    float eps1 = 0.0f;
    float eps2 = 0.0f;
    memcpy(&eps1, norm1->op_params, sizeof(float));
    memcpy(&eps2, norm2->op_params, sizeof(float));
    GGML_ASSERT(eps1 >= 0.0f && eps2 >= 0.0f);

    const int  ncols     = x->ne[0];
    const bool s_per_col = s != nullptr && ggml_nelements(s) > 1;
    const dim3 blocks_num(x->ne[1], x->ne[2], x->ne[3]);

    const auto launch = [&](auto has_scale, auto block_size, auto cols) {
        constexpr int nc = decltype(cols)::value;
        if constexpr (nc > 0) {
            constexpr int bs = decltype(block_size)::value;
            const ggml_cuda_kernel_launch_params launch_params{blocks_num, dim3(bs, 1, 1), 64 * sizeof(float), ctx.stream()};
            ggml_cuda_kernel_launch(rms_norm_add_rms_norm_f32<bs, nc, decltype(has_scale)::value>, launch_params,
                (const float *) x->data, (const float *) w1->data, (const float *) r->data,
                s != nullptr ? (const float *) s->data : nullptr, (const float *) w2->data, (float *) y->data,
                (float *) mul2->data, ncols, eps1, eps2, s_per_col);
        } else {
            GGML_ABORT("fatal error");
        }
    };
    // the block size rms_norm_f32 would pick, which fixes the order of the reductions
    const auto with_scale = [&](auto has_scale) {
        if (ncols < 1024) {
            rms_norm_launch_cols<256, 4>(ncols, [&](auto cols) { launch(has_scale, std::integral_constant<int, 256>(), cols); });
        } else {
            rms_norm_launch_cols<1024, 8>(ncols, [&](auto cols) { launch(has_scale, std::integral_constant<int, 1024>(), cols); });
        }
    };
    if (s != nullptr) {
        with_scale(std::true_type());
    } else {
        with_scale(std::false_type());
    }
}

// the same functions as op_silu and op_sigmoid of unary_gated_op_kernel
static __device__ __forceinline__ float rms_norm_gate_silu(float x) {
    return ggml_cuda_op_silu_single(x);
}

static __device__ __forceinline__ float rms_norm_gate_sigmoid(float x) {
    return ggml_cuda_op_sigmoid_single(x);
}

bool ggml_cuda_should_fuse_rms_norm_gated(const ggml_tensor * rms_norm, const ggml_tensor * mul, const ggml_tensor * unary,
                                          const ggml_tensor * mul_gate) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || unary->op != GGML_OP_UNARY || mul_gate->op != GGML_OP_MUL) {
        return false;
    }
    const ggml_unary_op uop = ggml_get_unary_op(unary);
    if (uop != GGML_UNARY_OP_SILU && uop != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : (mul->src[1] == rms_norm ? mul->src[0] : nullptr);
    if (w == nullptr || w == rms_norm ||
            !((mul_gate->src[0] == mul && mul_gate->src[1] == unary) || (mul_gate->src[0] == unary && mul_gate->src[1] == mul))) {
        return false;
    }
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * z = unary->src[0];
    for (const ggml_tensor * t : { x, z, w, rms_norm, mul, unary, mul_gate }) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    // a block reads its rows of x twice and writes its row of dst after the reduction, so dst may sit exactly on a
    //   contiguous x or z (ggml-alloc can reuse them once the elided nodes are done) but not partly over one
    return ggml_are_same_shape(mul, rms_norm) && ggml_are_same_shape(z, rms_norm) && ggml_are_same_shape(mul_gate, rms_norm) &&
        x->nb[0] == sizeof(float) && z->nb[0] == sizeof(float) && w->nb[0] == sizeof(float) && ggml_can_repeat(w, rms_norm) &&
        ggml_is_contiguous(mul_gate) && rms_norm->ne[1] <= INT_MAX && rms_norm->ne[2] <= 65535 && rms_norm->ne[3] <= 65535 &&
        ggml_cuda_same_or_disjoint(mul_gate, x) && (mul_gate->data != x->data || ggml_is_contiguous(x)) &&
        ggml_cuda_same_or_disjoint(mul_gate, z) && (mul_gate->data != z->data || ggml_is_contiguous(z)) &&
        !ggml_cuda_tensors_overlap(mul_gate, w);
}

void ggml_cuda_op_rms_norm_gated(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * unary,
                                 ggml_tensor * mul_gate) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * z = unary->src[0];
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const int ncols = x->ne[0];
    const dim3 blocks_num(x->ne[1], x->ne[2], x->ne[3]);

    const size_t ts = sizeof(float);
    const uint3 mul_ncols_packed     = init_fastdiv_values(w->ne[0]);
    const uint3 mul_nrows_packed     = init_fastdiv_values(w->ne[1]);
    const uint3 mul_nchannels_packed = init_fastdiv_values(w->ne[2]);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(w->ne[3]);

    const auto launch = [&](auto kernel, const int block_size) {
        const dim3 block_dims(block_size, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, ctx.stream()};
        ggml_cuda_kernel_launch(kernel, launch_params, (const float *) x->data, (const float *) z->data, (float *) mul_gate->data,
            ncols, (int64_t) (x->nb[1]/ts), (int64_t) (x->nb[2]/ts), (int64_t) (x->nb[3]/ts),
            (int64_t) (z->nb[1]/ts), (int64_t) (z->nb[2]/ts), (int64_t) (z->nb[3]/ts), eps, (const float *) w->data,
            (int64_t) (w->nb[1]/ts), (int64_t) (w->nb[2]/ts), (int64_t) (w->nb[3]/ts),
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
    };
    // the block size rms_norm_f32 would pick, which fixes the order of the reduction
    const bool silu = ggml_get_unary_op(unary) == GGML_UNARY_OP_SILU;
    if (ncols < 1024) {
        if (silu) {
            launch(rms_norm_gated_f32<256, rms_norm_gate_silu>, 256);
        } else {
            launch(rms_norm_gated_f32<256, rms_norm_gate_sigmoid>, 256);
        }
    } else {
        if (silu) {
            launch(rms_norm_gated_f32<1024, rms_norm_gate_silu>, 1024);
        } else {
            launch(rms_norm_gated_f32<1024, rms_norm_gate_sigmoid>, 1024);
        }
    }
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_d  = (const float *) rms_norm_src->data;
    const float *       mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d,add_d,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}
