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

// rms_norm_f32<block_size, true, false, false, cols_per_thread> (dst = rms_norm(x) * mul, contiguous) that also writes the
//   q8_1 copy of dst for the MMVQ matmul that reads all of it, one row after the other: a warp holds 32 consecutive
//   columns of a register slot and quantizes them as one q8_1 block. The reduction and products are rms_norm_f32's and
//   the quantization quantize_q8_1's, so both outputs are those of the ops apart. ncols % 32 == 0, so a warp is in the
//   row or past it as a whole. The grouped norm of qwen4exp's hyper-connections, hc rows read as one by their matmul.
template <int block_size, int cols_per_thread>
static __global__ void __launch_bounds__(block_size) rms_norm_mul_q8_f32(const float * x, float * dst, const int ncols,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        const float * mul, const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed,
        const uint3 mul_nsamples_packed, block_q8_1 * q8_dst) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    const int64_t row_flat = (int64_t(sample)*nchannels + channel)*nrows + row;
    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += row_flat*ncols;
    block_q8_1 * q8 = q8_dst + row_flat*(ncols/QK8_1);

    const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
    mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

    float tmp = 0.0f;
    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();
    float xv[cols_per_thread];
    float mv[cols_per_thread];
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        const int col = tid + i*block_size;
        xv[i] = col < ncols ? x[col] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        const int col = tid + i*block_size;
        mv[i] = col < ncols ? mul[fastmodulo(col, mul_ncols_packed)] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        if (tid + i*block_size < ncols) {
            tmp += xv[i] * xv[i];
        }
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        const int col = tid + i*block_size;
        if (col >= ncols) {
            continue;
        }
        const float v = scale * xv[i] * mv[i];
        dst[col] = v;
        quantize_q8_1_lane(v, q8 + col/QK8_1, col % QK8_1);
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
                                            const float * c = nullptr, const float * g = nullptr,
                                            block_q8_1 * q8_dst = nullptr, const int nq8 = 0, unsigned int * flags = nullptr) {
    ggml_cuda_pdl_lc();
    // With q8_dst there is one row, and block 1 + i writes the q8_1 copy of the columns in register slot i of block 0
    //   (one block of 32 per warp). It repeats block 0's reduction, in the same order, so it gets the same values. The
    //   outputs may sit on the inputs: with flags, block 0 writes nothing before every other block has read them.
    const bool emitter = q8_dst != nullptr && blockIdx.x > 0;
    const int nrows     = q8_dst != nullptr ? 1 : gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = q8_dst != nullptr ? 0 : blockIdx.x;
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
                tmp += xv[i] * xv[i];
            }
        }

        // the inputs have been read (tmp used them): tell block 0
        if (emitter && flags != nullptr) {
            __syncthreads();
            if (tid == 0) {
                __threadfence();
                atomicAdd(flags, 1u);
            }
        }

        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

        const float mean  = tmp / ncols;
        const float scale = rsqrtf(mean + eps);

        if (emitter) {
            // the slot of this block, the padding blocks of the q8_1 row included (zeros)
            const int slot = blockIdx.x - 1;
            const int ib   = slot*(block_size/WARP_SIZE) + tid/WARP_SIZE;
            float v = 0.0f;
#pragma unroll
            for (int i = 0; i < cols_per_thread; ++i) {
                if (i == slot && tid + i*block_size < ncols) {
                    v = scale * xv[i] * mv[i];
                }
            }
            if (ib < nq8) {
                quantize_q8_1_lane(v, q8_dst + ib, tid % WARP_SIZE);
            }
            return;
        }

        // the other blocks have read the inputs, which the outputs may sit on; lower the flag for the next launch
        if (q8_dst != nullptr && flags != nullptr) {
            if (tid == 0) {
                while (atomicAdd(flags, 0u) < gridDim.x - 1) {
                }
                atomicExch(flags, 0u);
            }
            __syncthreads();
        }

#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = tid + i*block_size;
            if (col < ncols) {
                sum[col] = xv[i];
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
// has_pre: x is not read but made as rms_norm(a) * wa + rms_norm(b) * wb, the two post-norms of a gemma4 MoE block (the
//   dense MLP and the experts) summed, rounded like the RMS_NORM -> MUL and RMS_NORM -> MUL -> ADD kernels apart.
struct rms_norm_pre_pair {
    const float * a  = nullptr;
    const float * wa = nullptr;
    const float * b  = nullptr;
    const float * wb = nullptr;
    float eps_a = 0.0f;
    float eps_b = 0.0f;
};

template <int block_size, int cols_per_thread, bool has_scale, bool has_pre = false>
static __global__ void __launch_bounds__(block_size) rms_norm_add_rms_norm_f32(
        const float * x, const float * w1, const float * r, const float * s, const float * w2, float * y_out, float * dst,
        const int ncols, const float eps1, const float eps2, const bool s_per_col,
        block_q8_1 * q8_dst = nullptr, const int nq8 = 0, unsigned int * flags = nullptr, const rms_norm_pre_pair pre = {}) {
    ggml_cuda_pdl_lc();
    // the q8_1 copy of dst as in rms_norm_pre_add_f32: blocks after the first, one row
    const bool emitter  = q8_dst != nullptr && blockIdx.x > 0;
    const int nrows     = q8_dst != nullptr ? 1 : gridDim.x;
    const int nchannels = gridDim.y;
    const int tid       = threadIdx.x;

    const int64_t offset = q8_dst != nullptr ? 0 : ((int64_t(blockIdx.z)*nchannels + blockIdx.y)*nrows + blockIdx.x)*ncols;
    r     += offset;
    y_out += offset;
    dst   += offset;

    float xv[cols_per_thread];
    float w1v[cols_per_thread];
    float rv[cols_per_thread];
    float w2v[cols_per_thread];
    [[maybe_unused]] float sv[cols_per_thread];
    [[maybe_unused]] float av[cols_per_thread];
    [[maybe_unused]] float bv[cols_per_thread];

    ggml_cuda_pdl_sync();
    if constexpr (has_pre) {
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = tid + i*block_size;
            av[i] = col < ncols ? pre.a[offset + col] : 0.0f;
            bv[i] = col < ncols ? pre.b[offset + col] : 0.0f;
        }
    } else {
        x += offset;
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = tid + i*block_size;
            xv[i] = col < ncols ? x[col] : 0.0f;
        }
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

    // the inputs the outputs may sit on have been read (the predicate waits for them): tell block 0
    if (emitter && flags != nullptr) {
        bool nan = false;
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            if constexpr (has_pre) {
                nan = nan || av[i] != av[i] || bv[i] != bv[i] || rv[i] != rv[i];
            } else {
                nan = nan || xv[i] != xv[i] || rv[i] != rv[i];
            }
            if constexpr (has_scale) {
                nan = nan || sv[i] != sv[i];
            }
        }
        __syncthreads_or(nan);
        if (tid == 0) {
            __threadfence();
            atomicAdd(flags, 1u);
        }
    }

    extern __shared__ float s_sum[];

    if constexpr (has_pre) {
        float ta = 0.0f;
        float tb = 0.0f;
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            if (tid + i*block_size < ncols) {
                ta += av[i] * av[i];
                tb += bv[i] * bv[i];
            }
        }
        ta = block_reduce<block_reduce_method::SUM, block_size>(ta, s_sum + 64);
        tb = block_reduce<block_reduce_method::SUM, block_size>(tb, s_sum + 96);
        const float scale_a = rsqrtf(ta / ncols + pre.eps_a);
        const float scale_b = rsqrtf(tb / ncols + pre.eps_b);
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int   col = tid + i*block_size;
            const float wa  = col < ncols ? pre.wa[col] : 0.0f;
            const float wb  = col < ncols ? pre.wb[col] : 0.0f;
            const float ya  = __fmul_rn(__fmul_rn(scale_a, av[i]), wa);
            xv[i] = col < ncols ? scale_b * bv[i] * wb + ya : 0.0f;
        }
    }

    float tmp = 0.0f;
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        if (tid + i*block_size < ncols) {
            tmp += xv[i] * xv[i];
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale1 = rsqrtf(tmp / ncols + eps1);

    // the other blocks have read the inputs; lower the flag for the next launch
    if (q8_dst != nullptr && !emitter && flags != nullptr) {
        if (tid == 0) {
            while (atomicAdd(flags, 0u) < gridDim.x - 1) {
            }
            atomicExch(flags, 0u);
        }
        __syncthreads();
    }

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
        if (!emitter) {
            y_out[col] = y;
        }
        xv[i] = y;
        tmp2 += y * y;
    }
    tmp2 = block_reduce<block_reduce_method::SUM, block_size>(tmp2, s_sum + 32);
    const float scale2 = rsqrtf(tmp2 / ncols + eps2);

    if (emitter) {
        const int slot = blockIdx.x - 1;
        const int ib   = slot*(block_size/WARP_SIZE) + tid/WARP_SIZE;
        float v = 0.0f;
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            if (i == slot && tid + i*block_size < ncols) {
                v = scale2 * xv[i] * w2v[i];
            }
        }
        if (ib < nq8) {
            quantize_q8_1_lane(v, q8_dst + ib, tid % WARP_SIZE);
        }
        return;
    }

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
                                          const uint3 mul_nsamples_packed, block_q8_1 * q8_dst) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    const int64_t row_flat = (int64_t(sample)*nchannels + channel)*nrows + row;
    x    += sample*stride_sample + channel*stride_channel + row*stride_row;
    gate += sample*gate_stride_sample + channel*gate_stride_channel + row*gate_stride_row;
    dst  += row_flat*ncols;

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

    float v = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const int   mul_col = fastmodulo(col, mul_ncols_packed);
        const float normed  = scale * x[col] * mul[mul_col];
        v = op(gate[col]) * normed;
        dst[col] = v;
    }

    // the q8_1 copy of the output as one row, as quantize_q8_1 writes it: the launcher asks for it only with one value
    //   per thread and whole warps per block of 32, the lanes of which are those of quantize_q8_1
    if (q8_dst != nullptr && tid < ncols) {
        quantize_q8_1_lane(v, q8_dst + (row_flat*ncols + tid)/QK8_1, tid % QK8_1);
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

    // the q8_1 copy of the output for the MMVQ matmul that reads all of it, see ggml_cuda_q8_1_request; whole q8_1
    //   blocks per row and no padding blocks past the last row
    if (!scale_tensor && ctx.q8_1_emit == mul_tensor && ggml_is_contiguous(mul_tensor) && ne00 % QK8_1 == 0 &&
            ne00 <= 8*1024 && ggml_nelements(mul_tensor) % MATRIX_ROW_PADDING == 0) {
        int64_t ne_padded = 0;
        block_q8_1 * q8 = (block_q8_1 *) ctx.q8_1_emit_alloc(mul_tensor, ne_padded);
        if (q8 != nullptr) {
            GGML_ASSERT(ne_padded == ggml_nelements(mul_tensor));
            const dim3 blocks_num(ne01, ne02, ne03);
            const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
            const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
            const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
            const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
            const auto launch = [&](auto block, auto cols) {
                constexpr int bs = decltype(block)::value;
                constexpr int nc = decltype(cols)::value;
                if constexpr (nc > 0) {
                    const dim3 block_dims(bs, 1, 1);
                    const ggml_cuda_kernel_launch_params launch_params{blocks_num, block_dims, 32 * sizeof(float), stream};
                    ggml_cuda_kernel_launch(rms_norm_mul_q8_f32<bs, nc>, launch_params, src0_d, dst_d, (int) ne00, s01, s02, s03,
                        eps, mul_d, mul_s01, mul_s02, mul_s03, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed,
                        mul_nsamples_packed, q8);
                } else {
                    GGML_ABORT("fatal error");
                }
            };
            if (ne00 < 1024) {
                rms_norm_launch_cols<256, 4>(ne00, [&](auto cols) { launch(std::integral_constant<int, 256>(), cols); });
            } else {
                rms_norm_launch_cols<1024, 8>(ne00, [&](auto cols) { launch(std::integral_constant<int, 1024>(), cols); });
            }
            return;
        }
    }

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
    const int block_size = ncols < 1024 ? 256 : 1024;

    // the q8_1 copy for the matmul that reads the output, see ggml_cuda_q8_1_request and rms_norm_pre_add_f32
    block_q8_1   * q8_d  = nullptr;
    int            nq8   = 0;
    unsigned int * flags = nullptr;
    dim3 grid = blocks_num;
    if (ctx.q8_1_emit == mul && ggml_nrows(mul) == 1 && ggml_is_contiguous(mul) && ncols <= 8*block_size) {
        int64_t ne_padded = 0;
        q8_d   = (block_q8_1 *) ctx.q8_1_emit_alloc(mul, ne_padded);
        nq8    = (int) (ne_padded / QK8_1);
        grid.x = 1 + (nq8 + block_size/WARP_SIZE - 1) / (block_size/WARP_SIZE);
        // whether an output sits on an input, checked over the row bytes (g is one value)
        const auto overlaps = [&](const void * p) {
            const char * p0 = (const char *) p;
            const char * s0 = (const char *) sum_d;
            return p0 != nullptr && p0 < s0 + ggml_nbytes(add) && s0 < p0 + ggml_nbytes(add);
        };
        const auto overlaps_dst = [&](const void * p) {
            const char * p0 = (const char *) p;
            const char * d0 = (const char *) dst_d;
            return p0 != nullptr && p0 < d0 + ggml_nbytes(mul) && d0 < p0 + ggml_nbytes(add);
        };
        if (overlaps(a_d) || overlaps(b_d) || overlaps(c_d) || overlaps(g_d) || overlaps_dst(a_d) || overlaps_dst(b_d) ||
                overlaps_dst(c_d) || overlaps_dst(g_d)) {
            flags = ctx.block_flags_get();
        }
    }

    if (block_size == 256) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params{grid, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, ctx.stream()};
        rms_norm_launch_cols<256, 4>(ncols, [&](auto cols) {
            ggml_cuda_kernel_launch(rms_norm_pre_add_f32<256, decltype(cols)::value, gated>, launch_params, a_d, b_d, sum_d, dst_d, ncols, eps, w_d,
                (int64_t) (w->nb[1]/ts_w), (int64_t) (w->nb[2]/ts_w), (int64_t) (w->nb[3]/ts_w),
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, c_d, g_d, q8_d, nq8, flags);
        });
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params{grid, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, ctx.stream()};
        rms_norm_launch_cols<1024, 8>(ncols, [&](auto cols) {
            ggml_cuda_kernel_launch(rms_norm_pre_add_f32<1024, decltype(cols)::value, gated>, launch_params, a_d, b_d, sum_d, dst_d, ncols, eps, w_d,
                (int64_t) (w->nb[1]/ts_w), (int64_t) (w->nb[2]/ts_w), (int64_t) (w->nb[3]/ts_w),
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, c_d, g_d, q8_d, nq8, flags);
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

bool ggml_cuda_should_fuse_rms_norm_pair_add(const ggml_tensor * norm_a, const ggml_tensor * mul_a, const ggml_tensor * norm_b,
                                             const ggml_tensor * mul_b, const ggml_tensor * add_ab, const ggml_tensor * norm1,
                                             const ggml_tensor * y, const ggml_tensor * mul2) {
    if (norm_a->op != GGML_OP_RMS_NORM || mul_a->op != GGML_OP_MUL || norm_b->op != GGML_OP_RMS_NORM ||
            mul_b->op != GGML_OP_MUL || add_ab->op != GGML_OP_ADD || norm1->src[0] != add_ab) {
        return false;
    }
    const ggml_tensor * a  = norm_a->src[0];
    const ggml_tensor * b  = norm_b->src[0];
    const ggml_tensor * wa = ggml_cuda_other_src(mul_a, norm_a);
    const ggml_tensor * wb = ggml_cuda_other_src(mul_b, norm_b);
    if (wa == nullptr || wa == norm_a || wb == nullptr || wb == norm_b ||
            !((add_ab->src[0] == mul_a && add_ab->src[1] == mul_b) || (add_ab->src[0] == mul_b && add_ab->src[1] == mul_a))) {
        return false;
    }
    const int64_t ncols = add_ab->ne[0];
    for (const ggml_tensor * t : { a, b, wa, wb, norm_a, mul_a, norm_b, mul_b, add_ab }) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    // a and b are read before the first reduction, as x is: an output may sit exactly on them
    return ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_are_same_shape(a, add_ab) && ggml_are_same_shape(b, add_ab) &&
        ggml_are_same_shape(mul_a, add_ab) && ggml_are_same_shape(mul_b, add_ab) &&
        ggml_is_contiguous(wa) && ggml_nelements(wa) == ncols && ggml_is_contiguous(wb) && ggml_nelements(wb) == ncols &&
        ggml_cuda_same_or_disjoint(y, a) && ggml_cuda_same_or_disjoint(y, b) && ggml_cuda_same_or_disjoint(mul2, a) &&
        ggml_cuda_same_or_disjoint(mul2, b) && !ggml_cuda_tensors_overlap(y, wa) && !ggml_cuda_tensors_overlap(y, wb) &&
        !ggml_cuda_tensors_overlap(mul2, wa) && !ggml_cuda_tensors_overlap(mul2, wb);
}

void ggml_cuda_op_rms_norm_add_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * norm1, ggml_tensor * mul1, ggml_tensor * add,
                                        ggml_tensor * scale, ggml_tensor * norm2, ggml_tensor * mul2,
                                        const ggml_tensor * norm_a, const ggml_tensor * mul_a,
                                        const ggml_tensor * norm_b, const ggml_tensor * mul_b) {
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

    const bool has_pre = norm_a != nullptr;
    rms_norm_pre_pair pre;
    if (has_pre) {
        pre.a  = (const float *) norm_a->src[0]->data;
        pre.b  = (const float *) norm_b->src[0]->data;
        pre.wa = (const float *) ggml_cuda_other_src(mul_a, norm_a)->data;
        pre.wb = (const float *) ggml_cuda_other_src(mul_b, norm_b)->data;
        memcpy(&pre.eps_a, norm_a->op_params, sizeof(float));
        memcpy(&pre.eps_b, norm_b->op_params, sizeof(float));
    }

    const int  ncols     = x->ne[0];
    const bool s_per_col = s != nullptr && ggml_nelements(s) > 1;
    const dim3 blocks_num(x->ne[1], x->ne[2], x->ne[3]);

    // the q8_1 copy for the matmul that reads the output, see ggml_cuda_q8_1_request and rms_norm_pre_add_f32
    const int      block_size = ncols < 1024 ? 256 : 1024;
    block_q8_1   * q8_d  = nullptr;
    int            nq8   = 0;
    unsigned int * flags = nullptr;
    dim3 grid = blocks_num;
    if (ctx.q8_1_emit == mul2 && ggml_nrows(mul2) == 1 && ggml_is_contiguous(mul2) && ncols <= 8*block_size) {
        int64_t ne_padded = 0;
        q8_d   = (block_q8_1 *) ctx.q8_1_emit_alloc(mul2, ne_padded);
        nq8    = (int) (ne_padded / QK8_1);
        grid.x = 1 + (nq8 + block_size/WARP_SIZE - 1) / (block_size/WARP_SIZE);
        const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
            const char * a0 = (const char *) a->data;
            const char * b0 = (const char *) b->data;
            return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
        };
        const ggml_tensor * ins[4] = { has_pre ? norm_a->src[0] : x, has_pre ? norm_b->src[0] : nullptr, r, s };
        for (const ggml_tensor * in : ins) {
            if (in != nullptr && (overlaps(in, y) || overlaps(in, mul2))) {
                flags = ctx.block_flags_get();
            }
        }
    }

    const auto launch = [&](auto has_scale, auto block_size, auto cols) {
        constexpr int nc = decltype(cols)::value;
        if constexpr (nc > 0) {
            constexpr int bs = decltype(block_size)::value;
            const ggml_cuda_kernel_launch_params launch_params{grid, dim3(bs, 1, 1), 128 * sizeof(float), ctx.stream()};
            const auto kernel = has_pre ? rms_norm_add_rms_norm_f32<bs, nc, decltype(has_scale)::value, true> :
                                          rms_norm_add_rms_norm_f32<bs, nc, decltype(has_scale)::value, false>;
            ggml_cuda_kernel_launch(kernel, launch_params,
                (const float *) x->data, (const float *) w1->data, (const float *) r->data,
                s != nullptr ? (const float *) s->data : nullptr, (const float *) w2->data, (float *) y->data,
                (float *) mul2->data, ncols, eps1, eps2, s_per_col, q8_d, nq8, flags, pre);
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
        // the q8_1 copy for the matmul that reads the output, see ggml_cuda_q8_1_request
        block_q8_1 * q8_dst = nullptr;
        if (ctx.q8_1_emit == mul_gate && ncols <= block_size && ncols % QK8_1 == 0 &&
                ggml_nelements(mul_gate) % MATRIX_ROW_PADDING == 0 && ggml_is_contiguous(mul_gate)) {
            int64_t ne_padded = 0;
            q8_dst = (block_q8_1 *) ctx.q8_1_emit_alloc(mul_gate, ne_padded);
        }
        const dim3 block_dims(block_size, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, ctx.stream()};
        ggml_cuda_kernel_launch(kernel, launch_params, (const float *) x->data, (const float *) z->data, (float *) mul_gate->data,
            ncols, (int64_t) (x->nb[1]/ts), (int64_t) (x->nb[2]/ts), (int64_t) (x->nb[3]/ts),
            (int64_t) (z->nb[1]/ts), (int64_t) (z->nb[2]/ts), (int64_t) (z->nb[3]/ts), eps, (const float *) w->data,
            (int64_t) (w->nb[1]/ts), (int64_t) (w->nb[2]/ts), (int64_t) (w->nb[3]/ts),
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, q8_dst);
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
