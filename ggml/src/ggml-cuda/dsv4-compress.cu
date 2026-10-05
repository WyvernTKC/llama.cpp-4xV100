#include "dsv4-compress.cuh"

struct dsv4_compress_src {
    const char * state;
    const char * cur;
    int64_t      state_nb1;
    int64_t      cur_nb1;
    int          ns;          // rows of the state
    int          nt;          // rows of cur
    float        extra_scale; // the row past them: extra_scale*row0 + extra_bias (as SCALE computes it)
    float        extra_bias;
    const int32_t * idx;
};

struct dsv4_compress_args {
    dsv4_compress_src kv;
    dsv4_compress_src score;
    int   n;       // entries per window
    int   r;
    int   n_read;  // overlap: the rows of the second half start here
    int   hd;
    bool  overlap;
    float sm_scale;
};

static __device__ __forceinline__ float dsv4_compress_load(const dsv4_compress_src & s, const int row, const int col) {
    if (row < s.ns) {
        return ((const float *) (s.state + (int64_t) row*s.state_nb1))[col];
    }
    if (row < s.ns + s.nt) {
        return ((const float *) (s.cur + (int64_t) (row - s.ns)*s.cur_nb1))[col];
    }
    const float x = s.ns > 0 ? ((const float *) s.state)[col] : ((const float *) s.cur)[col];
    return s.extra_scale * x + s.extra_bias;
}

// a block per (d, b) with the softmax's block size: thread t holds entry t of the window
template <int block_size>
static __global__ void dsv4_compress_f32(const dsv4_compress_args a, float * dst) {
    __shared__ float buf_iw[WARP_SIZE];
    __shared__ float prod[block_size];

    const int d = blockIdx.x;
    const int b = blockIdx.y;
    const int t = threadIdx.x;

    float val     = 0.0f;
    float v       = 0.0f;
    float max_val = -INFINITY;
    if (t < a.n) {
        const bool second = a.overlap && t >= a.r;
        const int  k      = second ? a.n_read + b*a.r + (t - a.r) : b*a.r + t;
        const int  col    = second ? a.hd + d : d;
        val = dsv4_compress_load(a.score, a.score.idx[k], col)*a.sm_scale + 0.0f;
        v   = dsv4_compress_load(a.kv,    a.kv.idx[k],    col);
        max_val = max(max_val, val);
    }
    max_val = block_reduce<block_reduce_method::MAX, block_size>(max_val, buf_iw);

    float tmp = 0.0f;
    float e   = 0.0f;
    if (t < a.n) {
        e = expf(val - max_val);
        tmp += e;
    }
    if (block_size > WARP_SIZE) {
        __syncthreads();
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, buf_iw);
    const float inv_sum = 1.0f / tmp;

    if (t < a.n) {
        prod[t] = v * (e * inv_sum);
    }
    __syncthreads();

    // the row sum as reduce_row_f32 forms it with a block of one warp
    if (t < WARP_SIZE) {
        constexpr int num_unroll = 8;
        float temp[num_unroll];
        float sum_temp[num_unroll] = { 0.0f };
        for (int i = t; i < a.n;) {
            for (int j = 0; j < num_unroll; ++j) {
                temp[j] = i < a.n ? prod[i] : 0.0f;
                i += WARP_SIZE;
            }
            for (int j = 0; j < num_unroll; ++j) {
                sum_temp[j] += temp[j];
            }
        }
        float sum = 0.0f;
        for (int j = 0; j < num_unroll; ++j) {
            sum += sum_temp[j];
        }
        sum = warp_reduce_sum(sum);
        if (t == 0) {
            dst[(int64_t) b*a.hd + d] = sum;
        }
    }
}

static dsv4_compress_src dsv4_compress_src_of(const ggml_cuda_dsv4_compress_half & h) {
    dsv4_compress_src s;
    s.state       = (const char *) h.state->data;
    s.cur         = (const char *) h.cur->data;
    s.state_nb1   = h.state->nb[1];
    s.cur_nb1     = h.cur->nb[1];
    s.ns          = (int) h.state->ne[1];
    s.nt          = (int) h.cur->ne[1];
    s.extra_scale = 0.0f;
    s.extra_bias  = 0.0f;
    if (h.extra) {
        memcpy(&s.extra_scale, (const float *) h.extra->op_params + 0, sizeof(float));
        memcpy(&s.extra_bias,  (const float *) h.extra->op_params + 1, sizeof(float));
    }
    s.idx = (const int32_t *) h.idx->data;
    return s;
}

void ggml_cuda_dsv4_compress(ggml_backend_cuda_context & ctx, const ggml_cuda_dsv4_compress_fold & f, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && ggml_nelements(dst) == f.hd*f.nb);
    GGML_ASSERT(f.nb <= 65535 && f.hd <= INT32_MAX);

    dsv4_compress_args a;
    a.kv       = dsv4_compress_src_of(f.kv);
    a.score    = dsv4_compress_src_of(f.score);
    a.r        = (int) f.r;
    a.n        = (int) (f.overlap ? 2*f.r : f.r);
    a.n_read   = (int) (f.r*f.nb);
    a.hd       = (int) f.hd;
    a.overlap  = f.overlap;
    a.sm_scale = f.sm_scale;
    GGML_ASSERT(a.n <= GGML_CUDA_DSV4_COMPRESS_MAX);

    // the block size SOFT_MAX launches for a row of n
    int nth = WARP_SIZE;
    while (nth < a.n) {
        nth *= 2;
    }
    const dim3 grid((unsigned) f.hd, (unsigned) f.nb, 1);
    cudaStream_t stream = ctx.stream();
    switch (nth) {
        case  32: dsv4_compress_f32< 32><<<grid,  32, 0, stream>>>(a, (float *) dst->data); break;
        case  64: dsv4_compress_f32< 64><<<grid,  64, 0, stream>>>(a, (float *) dst->data); break;
        case 128: dsv4_compress_f32<128><<<grid, 128, 0, stream>>>(a, (float *) dst->data); break;
        case 256: dsv4_compress_f32<256><<<grid, 256, 0, stream>>>(a, (float *) dst->data); break;
        default: GGML_ABORT("unsupported window");
    }
    CUDA_CHECK(cudaGetLastError());
}
