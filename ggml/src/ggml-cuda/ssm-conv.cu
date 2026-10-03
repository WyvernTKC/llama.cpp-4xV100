#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

// bounds_check covers a row count that the block size does not divide, which a tensor split can produce
template <bool apply_silu, size_t split_d_inner, size_t d_conv, bool bounds_check>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t nr, const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();

    if (bounds_check && (int64_t) bidy * split_d_inner + tid >= nr) {
        return;
    }

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t, bool bounds_check>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t nr,
                                               const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    // rows this block owns; a partial block still has to reach the __syncthreads() below
    const int n_rows = bounds_check ? (int) min((int64_t) split_d_inner, nr - (int64_t) bidy * split_d_inner)
                                    : (int) split_d_inner;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < n_rows) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    if (bounds_check && tid >= n_rows) {
        return;
    }

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;

    // a tensor split hands each device a slice of the rows, which the block size need not divide
    auto launch_kernel_bc = [&](auto NC, auto BC) {
        constexpr int  kNC = decltype(NC)::value;
        constexpr bool kBC = decltype(BC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC, kBC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, nr, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t, kBC><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, nr, n_t);
        }
    };

    auto launch_kernel = [&](auto NC) {
        if (nr % threads == 0) {
            launch_kernel_bc(NC, std::false_type{});
        } else {
            launch_kernel_bc(NC, std::true_type{});
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

// One thread per channel: the d_conv - 1 state columns from the cache row and the n_t new tokens make the window,
//   which goes to the rollback slots and is convolved as in ssm_conv_f32, so the output is that of the four ops apart.
//   The thread reads all of its inputs before it writes, so dst may sit exactly on x.
template <int d_conv, int n_t>
static __global__ void ssm_conv_state_f32(const float * states, const int32_t * ids, const int64_t row_stride,
                                          const float * x, const int64_t x_stride_t, const int64_t x_stride_c,
                                          const float * w, const int64_t w_stride, const float * bias,
                                          float * dst, const int64_t dst_stride_t,
                                          const ggml_cuda_ssm_conv_state_slots slots, const int nr) {
    ggml_cuda_pdl_lc();
    const int c = blockIdx.x*blockDim.x + threadIdx.x;

    ggml_cuda_pdl_sync();
    if (c >= nr) {
        return;
    }

    float win[d_conv - 1 + n_t];
    const float * s = states + ids[0]*row_stride + (int64_t) c*(d_conv - 1);
#pragma unroll
    for (int k = 0; k < d_conv - 1; ++k) {
        win[k] = s[k];
    }
#pragma unroll
    for (int t = 0; t < n_t; ++t) {
        win[d_conv - 1 + t] = x[t*x_stride_t + c*x_stride_c];
    }

    float wv[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wv[j] = w[c*w_stride + j];
    }
    const float b = bias != nullptr ? bias[c] : 0.0f;

#pragma unroll
    for (int i = 0; i <= n_t; ++i) {
        float * slot = slots.dst[i];
        if (slot != nullptr) {
#pragma unroll
            for (int k = 0; k < d_conv - 1; ++k) {
                slot[(int64_t) c*(d_conv - 1) + k] = win[i + k];
            }
        }
    }

#pragma unroll
    for (int t = 0; t < n_t; ++t) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sumf += win[t + j] * wv[j];
        }
        sumf += b;
        dst[t*dst_stride_t + c] = ggml_cuda_op_silu_single(sumf);
    }
}

void ggml_cuda_op_ssm_conv_state(ggml_backend_cuda_context & ctx, const ggml_tensor * states, const ggml_tensor * ids,
                                 const ggml_tensor * x, const ggml_tensor * ssm_conv, ggml_tensor * silu,
                                 const ggml_cuda_ssm_conv_state_slots & slots) {
    const ggml_tensor * w = ssm_conv->src[1];

    const int nr  = (int) w->ne[1];
    const int n_t = (int) silu->ne[1];
    GGML_ASSERT(w->ne[0] == 4 && n_t >= 1 && n_t <= GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS);

    const int threads = 128;
    const ggml_cuda_kernel_launch_params launch_params((nr + threads - 1) / threads, threads, 0, ctx.stream());

    const size_t ts = sizeof(float);
    auto launch = [&](auto NT) {
        ggml_cuda_kernel_launch(ssm_conv_state_f32<4, decltype(NT)::value>, launch_params,
            (const float *) states->data, (const int32_t *) ids->data, (int64_t) (states->nb[1]/ts),
            (const float *) x->data, (int64_t) (x->nb[0]/ts), (int64_t) (x->nb[1]/ts),
            (const float *) w->data, (int64_t) (w->nb[1]/ts), (const float *) nullptr,
            (float *) silu->data, (int64_t) (silu->nb[1]/ts), slots, nr);
    };
    switch (n_t) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 6: launch(std::integral_constant<int, 6>{}); break;
        case 7: launch(std::integral_constant<int, 7>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        default: GGML_ABORT("fatal error");
    }
}

// The same for a longer batch, convolved as ssm_conv_long_token_f32 does: a thread takes one channel of split_n_t tokens
//   and loads its window of the concat, the columns before the first token from the cache row and the tokens of x after
//   them; the threads of the first tokens also store the windows of the rollback slots. A window reaches into the tokens
//   of the block before, which with dst on x may already be overwritten: the first d_conv - 1 columns of every block
//   then come from ssm_conv_state_halo_f32, run before, which also stores the slot windows, after it has read the start
//   state (a slot is usually the cache row the state comes from), and a thread writes only tokens it has read.
template <int d_conv, int split_n_t>
static __global__ void ssm_conv_state_halo_f32(const float * states, const int32_t * ids, const int64_t row_stride,
                                               const float * x, const int64_t x_stride_t, const int64_t x_stride_c,
                                               float * halo, const ggml_cuda_ssm_conv_state_long_slots slots,
                                               const int nr, const int n_blocks_t) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nr) {
        return;
    }
    const float * s = states + ids[0]*row_stride + (int64_t) c*(d_conv - 1);

    // the first columns of block b: the start state for the first, the last tokens of the block before for the others
    for (int b = blockIdx.y; b < n_blocks_t; b += gridDim.y) {
#pragma unroll
        for (int k = 0; k < d_conv - 1; ++k) {
            const int t = b*split_n_t - (d_conv - 1) + k;
            halo[((int64_t) b*(d_conv - 1) + k)*nr + c] = b == 0 ? s[k] : x[(int64_t) t*x_stride_t + (int64_t) c*x_stride_c];
        }
    }

    if (blockIdx.y == 0) {
        // constant indices keep the slot table among the kernel parameters instead of a copy in local memory
#pragma unroll
        for (int k = 0; k < GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS + 1; ++k) {
            if (k >= slots.n) {
                break;
            }
#pragma unroll
            for (int kk = 0; kk < d_conv - 1; ++kk) {
                const int p = slots.start[k] + kk;
                slots.dst[k][(int64_t) c*(d_conv - 1) + kk] =
                    p < d_conv - 1 ? s[p] : x[(int64_t) (p - (d_conv - 1))*x_stride_t + (int64_t) c*x_stride_c];
            }
        }
    }
}

template <int d_conv, int split_n_t>
static __global__ void ssm_conv_state_long_f32(const float * states, const int32_t * ids, const int64_t row_stride,
                                               const float * x, const int64_t x_stride_t, const int64_t x_stride_c,
                                               const float * w, const int64_t w_stride, const float * bias,
                                               float * dst, const int64_t dst_stride_t, const float * halo,
                                               const ggml_cuda_ssm_conv_state_long_slots slots, const int nr, const int n_t) {
    ggml_cuda_pdl_lc();
    const int c  = blockIdx.x*blockDim.x + threadIdx.x;
    const int t0 = blockIdx.y*split_n_t;

    ggml_cuda_pdl_sync();
    if (c >= nr) {
        return;
    }

    const float * s = states + ids[0]*row_stride + (int64_t) c*(d_conv - 1);

    // column j of the window is column t0 + j of the concat
    constexpr int n_cols = d_conv - 1 + split_n_t;
    float win[n_cols];
#pragma unroll
    for (int j = 0; j < n_cols; ++j) {
        const int p = t0 + j;
        if (halo != nullptr && j < d_conv - 1) {
            win[j] = halo[((int64_t) blockIdx.y*(d_conv - 1) + j)*nr + c];
        } else {
            win[j] = p < d_conv - 1 ? s[p] : p - (d_conv - 1) < n_t ? x[(int64_t) (p - (d_conv - 1))*x_stride_t + (int64_t) c*x_stride_c] : 0.0f;
        }
    }

    if (blockIdx.y == 0) {
        // constant indices keep the slot table among the kernel parameters instead of a copy in local memory
#pragma unroll
        for (int k = 0; k < GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS + 1; ++k) {
            if (k >= slots.n) {
                break;
            }
#pragma unroll
            for (int kk = 0; kk < d_conv - 1; ++kk) {
                const int p = slots.start[k] + kk;
                slots.dst[k][(int64_t) c*(d_conv - 1) + kk] =
                    p < d_conv - 1 ? s[p] : x[(int64_t) (p - (d_conv - 1))*x_stride_t + (int64_t) c*x_stride_c];
            }
        }
    }

    float wv[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wv[j] = w[c*w_stride + j];
    }
    const float b = bias != nullptr ? bias[c] : 0.0f;

#pragma unroll
    for (int i = 0; i < split_n_t; ++i) {
        if (t0 + i >= n_t) {
            break;
        }
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sumf += win[i + j] * wv[j];
        }
        sumf += b;
        dst[(int64_t) (t0 + i)*dst_stride_t + c] = ggml_cuda_op_silu_single(sumf);
    }
}

void ggml_cuda_op_ssm_conv_state_long(ggml_backend_cuda_context & ctx, const ggml_tensor * states, const ggml_tensor * ids,
                                      const ggml_tensor * x, const ggml_tensor * ssm_conv, ggml_tensor * silu,
                                      const ggml_cuda_ssm_conv_state_long_slots & slots) {
    const ggml_tensor * w = ssm_conv->src[1];

    const int nr  = (int) w->ne[1];
    const int n_t = (int) silu->ne[1];
    GGML_ASSERT(w->ne[0] == 4 && silu->ne[2] == 1);

    constexpr int threads   = 128;
    constexpr int split_n_t = 32;
    const int     n_blocks_t = (n_t + split_n_t - 1) / split_n_t;
    const size_t  ts         = sizeof(float);

    // silu on x: the columns each block reads from the block before, and the slot windows, go first
    ggml_cuda_pool_alloc<float> halo(ctx.pool());
    ggml_cuda_ssm_conv_state_long_slots slots_long = slots;
    if (silu->data == x->data) {
        halo.alloc((size_t) n_blocks_t*3*nr);
        const dim3 blocks_halo((nr + threads - 1) / threads, std::min(n_blocks_t, 16), 1);
        ssm_conv_state_halo_f32<4, split_n_t><<<blocks_halo, threads, 0, ctx.stream()>>>(
            (const float *) states->data, (const int32_t *) ids->data, (int64_t) (states->nb[1]/ts),
            (const float *) x->data, (int64_t) (x->nb[0]/ts), (int64_t) (x->nb[1]/ts), halo.get(), slots, nr, n_blocks_t);
        slots_long.n = 0;
    }

    const ggml_cuda_kernel_launch_params launch_params(dim3((nr + threads - 1) / threads, n_blocks_t, 1), threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(ssm_conv_state_long_f32<4, split_n_t>, launch_params,
        (const float *) states->data, (const int32_t *) ids->data, (int64_t) (states->nb[1]/ts),
        (const float *) x->data, (int64_t) (x->nb[0]/ts), (int64_t) (x->nb[1]/ts),
        (const float *) w->data, (int64_t) (w->nb[1]/ts), (const float *) nullptr,
        (float *) silu->data, (int64_t) (silu->nb[1]/ts), (const float *) halo.ptr, slots_long, nr, n_t);
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
