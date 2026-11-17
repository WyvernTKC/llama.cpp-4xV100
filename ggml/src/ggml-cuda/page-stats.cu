#include "page-stats.cuh"
#include "convert.cuh"

// one block per (refreshed page, KV head), one thread per channel: [mean, var, max, min] over the page's 64 cells

template <typename TK>
static __device__ __forceinline__ float page_stats_load(const char * row, const int j);

template <>
__device__ __forceinline__ float page_stats_load<float>(const char * row, const int j) {
    return ((const float *) row)[j];
}

template <>
__device__ __forceinline__ float page_stats_load<half>(const char * row, const int j) {
    return __half2float(((const half *) row)[j]);
}

template <>
__device__ __forceinline__ float page_stats_load<block_q8_0>(const char * row, const int j) {
    const block_q8_0 * b = (const block_q8_0 *) row + j/QK8_0;
    return __half2float(b->d) * (float) b->qs[j % QK8_0];
}

template <typename TK, typename TS>
static __global__ void page_stats_kernel(
        const char * k, const int32_t * upd, TS * ps, const int d, const int n_kv_h,
        const int64_t nbk1, const int64_t nbk2, const int64_t sps1, const int64_t sps2) {
    ggml_cuda_pdl_lc();

    const int     h = blockIdx.y;
    ggml_cuda_pdl_sync();
    const int64_t p = upd[blockIdx.x];

    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float x[64];
        float sum = 0.0f;
        float mx  = -INFINITY;
        float mn  =  INFINITY;
#pragma unroll
        for (int c = 0; c < 64; ++c) {
            x[c] = page_stats_load<TK>(k + h*nbk1 + (64*p + c)*nbk2, j);
            sum += x[c];
            mx = fmaxf(mx, x[c]);
            mn = fminf(mn, x[c]);
        }
        const float mu = sum/64.0f;
        float var = 0.0f;
#pragma unroll
        for (int c = 0; c < 64; ++c) {
            const float e = x[c] - mu;
            var += e*e;
        }
        TS * out = ps + h*sps1 + p*sps2;
        out[j]       = ggml_cuda_cast<TS>(mu);
        out[d + j]   = ggml_cuda_cast<TS>(var/64.0f);
        out[2*d + j] = ggml_cuda_cast<TS>(mx);
        out[3*d + j] = ggml_cuda_cast<TS>(mn);
    }
}

bool ggml_cuda_page_stats_supported(const ggml_tensor * op) {
    const ggml_tensor * k = op->src[0];
    return (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) &&
        (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && (k->type != GGML_TYPE_Q8_0 || k->ne[0] % QK8_0 == 0);
}

template <typename TK, typename TS>
static void page_stats_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k   = dst->src[0];
    const ggml_tensor * upd = dst->src[2];

    const int d      = k->ne[0];
    const int n_kv_h = k->ne[1];
    const int n_upd  = ggml_nelements(upd);
    if (n_upd == 0) {
        return;
    }

    const ggml_cuda_kernel_launch_params params(dim3(n_upd, n_kv_h, 1), dim3(std::min(d, 256), 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(page_stats_kernel<TK, TS>, params,
            (const char *) k->data, (const int32_t *) upd->data, (TS *) dst->data, d, n_kv_h,
            (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) (dst->nb[1]/sizeof(TS)), (int64_t) (dst->nb[2]/sizeof(TS)));
}

template <typename TS>
static void page_stats_cuda_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:  page_stats_cuda<float,      TS>(ctx, dst); break;
        case GGML_TYPE_F16:  page_stats_cuda<half,       TS>(ctx, dst); break;
        case GGML_TYPE_Q8_0: page_stats_cuda<block_q8_0, TS>(ctx, dst); break;
        default: GGML_ABORT("unsupported type");
    }
}

void ggml_cuda_op_page_stats(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->type == GGML_TYPE_F16) {
        page_stats_cuda_k<half>(ctx, dst);
    } else {
        page_stats_cuda_k<float>(ctx, dst);
    }
}
