#include "common.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// the wmma path needs only fp16 wmma fragments (m8n32k16), which Volta has - keep the body for sm_70 as well
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)

typedef union {
    int2 i2;
    half2 h2[2];
} half4;

// TODO add support for AMD cards via rocWMMA
#include <mma.h>
namespace wmma = nvcuda::wmma;

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEADS_PER_INNER_LOOP = 8;
    constexpr int K_EMBD_PER_INNER_LOOP = 16;
    constexpr int N_EMBD_PADDED = N_EMBD + 8;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each block processes K_VECS_PER_BLOCK K vectors
    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load weights and first Q tile to shared memory

    __shared__ float w_shared[N_HEAD];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][N_EMBD_PADDED / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }

    // total number of half4 elements in HEADS_PER_INNER_LOOP x N_EMBD Q tile
    constexpr int N_Q_TILE = HEADS_PER_INNER_LOOP * (N_EMBD / 4);
    // number of registers needed in each thread to store Q tile in thread block
    constexpr int N_Q_NEXT = (N_Q_TILE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma unroll
    for (int i_q = tid; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (N_EMBD / 4);
        const int i_embd = i_q % (N_EMBD / 4);
        const float4 q = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        half4 q_packed;
        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
        q_shared_h[i_head][i_embd] = q_packed.i2;
    }

    // phase 2 - load (and dequantize if needed) K to shared mem

    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][N_EMBD_PADDED / 4][2];

    constexpr int n_k = K_VECS_PER_BLOCK * (N_EMBD / 4);

    if constexpr (TYPE_K == GGML_TYPE_F16) {
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    }

    __syncthreads();

    // phase 3 - calculate lightning indexer scores

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    // load K fragment
    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        // we don't use accumulator for anything, fill it with zeros
        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
        wmma::fill_fragment(frag_acc, 0.0f);

        // load Q fragment
        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

        // preload next Q tile to registers during matrix multiplication
        float4 q_next[N_Q_NEXT];

        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (N_EMBD / 4);
                const int i_embd =               i_q % (N_EMBD / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
            }
        }

        // perform matrix multiplication
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        // make sure all threads finished using q_shared_h so we can store next tile
        __syncthreads();

        // write preloaded Q tile to shared memory
        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_q / (N_EMBD / 4);
                const int i_embd = i_q % (N_EMBD / 4);
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
                q_shared_h[i_head][i_embd] = q_packed.i2;
                ++i_q_next;
            }
        }

        // accumulate QK multiplication results from all block warps
        // (there are 256 threads in block and 256 matmul outputs)
        // TODO it will break if WARP_SIZE is not 32
        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += qk_shared[w][h][k];
        }

        // ReLU, weight
        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        // wait until qk_shared[0] is no longer used
        __syncthreads();

        // reuse qk_shared[0] for storing partial results
        qk_shared[0][h][k] = sum;

        // wait until all threads write their results
        __syncthreads();

        // accumulate result over heads
        if (tid < K_VECS_PER_BLOCK) {
#pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        // make sure all threads finished using qk_shared
        __syncthreads();
    }

    // phase 4 - store output to VRAM

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = start_kv + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k + __half2float(m_base[i_kv]);
        }
    }
}

#else // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {
    GGML_UNUSED_VARS(Q, K, W, M, dst,
        n_stream, n_batch, n_kv,
        nb1, nb2, nb3,
        nbq1, nbq2, nbq3,
        nbk1, nbk2, nbk3,
        nbw1, nbw2, nbw3,
        nem3);
    NO_DEVICE_CODE;
}

#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// tokens scored per block by the tile kernel
#define LIGHTNING_INDEXER_TILE_TOKENS 8

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (N_EMBD)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_vec(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load (and dequantize if needed) K to registers

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
        // direct copy of float4
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const float4 * k_base = (const float4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else {
        // dequantize remaining types to float
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    // load weights and Q only for N_HEAD_INNER heads at once to reduce shared memory usage
    constexpr int N_HEAD_INNER = N_HEAD / 4;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // phase 2 - load weights and Q to shared memory

        __shared__ float  w_shared[N_HEAD_INNER];
        __shared__ float4 q_shared_f[N_HEAD_INNER][N_EMBD / 4];

        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = N_HEAD_INNER * (N_EMBD / 4);
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_head_inner][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        }

        __syncthreads();

        // phase 3 - calculate lightning indexer scores

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            // dot product of floats
            const float4 q_vec = q_shared_f[i_head_inner][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                // ReLU, weight
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid] + __half2float(m_base[i_kv]);
        }
    }
}

// one block scores a tile of K_VECS_PER_BLOCK keys against TOKENS_PER_BLOCK tokens: the keys are
// staged in half precision and the queries of every head in float, each thread owns KEYS_PER_THREAD
// keys for one token, a warp shares its token so the query reads are broadcasts, and every key
// element is widened once for all heads, so no dot product needs a cross thread reduction
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_tile(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int TOKENS_PER_BLOCK  = LIGHTNING_INDEXER_TILE_TOKENS;
    constexpr int KEY_LANES         = THREADS_PER_BLOCK / TOKENS_PER_BLOCK;
    constexpr int KEYS_PER_THREAD   = K_VECS_PER_BLOCK / KEY_LANES;
    constexpr int N_EMBD_H2         = N_EMBD / 2;

    static_assert(THREADS_PER_BLOCK % TOKENS_PER_BLOCK == 0, "threads must cover the token tile");
    static_assert(K_VECS_PER_BLOCK % KEY_LANES == 0, "key lanes must cover the key tile");

    const int tid         = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int start_kv    = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_batch = blockIdx.y * TOKENS_PER_BLOCK;
    const int i_stream    = blockIdx.z;

    // the row padding keeps the keys of consecutive threads in distinct banks
    __shared__ half2 k_shared[K_VECS_PER_BLOCK][N_EMBD_H2 + 1];
    __shared__ float2 q_shared[N_HEAD][TOKENS_PER_BLOCK][N_EMBD_H2];
    __shared__ float w_shared[N_HEAD][TOKENS_PER_BLOCK];

    // phase 1 - stage the key tile four elements at a time, rows past n_kv are zero

#pragma unroll
    for (int i = tid; i < K_VECS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
        const int r  = i / (N_EMBD / 4);
        const int c4 = i % (N_EMBD / 4);

        half2 lo = make_half2(0.0f, 0.0f);
        half2 hi = lo;
        if (start_kv + r < n_kv) {
            const char * k_row = K + (start_kv + r)*nbk2 + i_stream*nbk3;
            if constexpr (TYPE_K == GGML_TYPE_F16) {
                lo = ((const half2 *) k_row)[2*c4 + 0];
                hi = ((const half2 *) k_row)[2*c4 + 1];
            } else {
                float4 v;
                if constexpr (TYPE_K == GGML_TYPE_F32) {
                    v = ((const float4 *) k_row)[c4];
                } else {
                    constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
                    dequantize_k(k_row, &v, c4 * 4);
                }
                lo = make_half2(v.x, v.y);
                hi = make_half2(v.z, v.w);
            }
        }

        k_shared[r][2*c4 + 0] = lo;
        k_shared[r][2*c4 + 1] = hi;
    }

    // phase 2 - stage the queries and weights of every head, tokens past n_batch are zero

#pragma unroll
    for (int i = tid; i < N_HEAD * TOKENS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
        const int h  = i / (TOKENS_PER_BLOCK * (N_EMBD / 4));
        const int r  = i / (N_EMBD / 4) % TOKENS_PER_BLOCK;
        const int c4 = i % (N_EMBD / 4);

        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (start_batch + r < n_batch) {
            v = *(const float4 *) ((const char *) Q + h*nbq1 + (start_batch + r)*nbq2 + i_stream*nbq3 + c4*sizeof(float4));
        }

        q_shared[h][r][2*c4 + 0] = make_float2(v.x, v.y);
        q_shared[h][r][2*c4 + 1] = make_float2(v.z, v.w);
    }

    if (tid < N_HEAD * TOKENS_PER_BLOCK) {
        const int h = tid / TOKENS_PER_BLOCK;
        const int r = tid % TOKENS_PER_BLOCK;
        w_shared[h][r] = start_batch + r < n_batch ?
            ((const float *) ((const char *) W + (start_batch + r)*nbw1 + i_stream*nbw3))[h] : 0.0f;
    }

    __syncthreads();

    // phase 3 - float products of the widened keys for every head, ReLU, weight

    const int kl = tid % KEY_LANES;
    const int tl = tid / KEY_LANES;

    float qk[N_HEAD][KEYS_PER_THREAD] = { { 0.0f } };

#pragma unroll 8
    for (int c = 0; c < N_EMBD_H2; ++c) {
        float2 k_val[KEYS_PER_THREAD];
#pragma unroll
        for (int j = 0; j < KEYS_PER_THREAD; ++j) {
            k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);
        }
#pragma unroll
        for (int h = 0; h < N_HEAD; ++h) {
            const float2 q_val = q_shared[h][tl][c];
#pragma unroll
            for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                qk[h][j] = fmaf(k_val[j].x, q_val.x, qk[h][j]);
                qk[h][j] = fmaf(k_val[j].y, q_val.y, qk[h][j]);
            }
        }
    }

    float score[KEYS_PER_THREAD] = { 0.0f };

#pragma unroll
    for (int h = 0; h < N_HEAD; ++h) {
#pragma unroll
        for (int j = 0; j < KEYS_PER_THREAD; ++j) {
            score[j] += fmaxf(qk[h][j], 0.0f) * w_shared[h][tl];
        }
    }

    // phase 4 - add the mask and write, consecutive threads write consecutive keys

    const int i_batch = start_batch + tl;
    if (i_batch >= n_batch) {
        return;
    }

    const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
    float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);

#pragma unroll
    for (int j = 0; j < KEYS_PER_THREAD; ++j) {
        const int i_kv = start_kv + kl + j*KEY_LANES;
        if (i_kv < n_kv) {
            dst_base[i_kv] = score[j] + __half2float(m_base[i_kv]);
        }
    }
}

#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)         \
    if (K->type == (type_K)) {                                                              \
        lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K> \
            <<<grid, block, 0, ctx.stream()>>>(                                             \
            q_d, k_d, w_d, m_d, dst_d,                                                      \
            n_stream, n_batch, n_kv,                                                        \
            nb1, nb2, nb3,                                                                  \
            nbq1, nbq2, nbq3,                                                               \
            nbk1, nbk2, nbk3,                                                               \
            nbw1, nbw2, nbw3,                                                               \
            nbm1, nbm2, nbm3,                                                               \
            nem3                                                                            \
        );                                                                                  \
    } else

// ---- chunked cuBLAS path for NVIDIA GPUs without the wmma kernel -----------------------------
// score[b,kv] = sum_h w[b,h] * relu(q[b,h] . k[kv]) + mask[b,kv]
//
// The vector kernel above is the only option on pre-Turing hardware and reaches ~4.3 TFLOPS on a V100,
// while cuBLAS does the same multiply-accumulate at ~20 TFLOPS there. Compute K^T Q with cuBLAS one
// chunk of query rows at a time (C is n_kv x n_head*nb, column-major, so one head is one contiguous
// stream and neighbouring lanes of the reduce kernel read neighbouring kv rows), then do relu, the
// per-head weighting, the sum and the mask add in one fused kernel. The scratch buffer is bounded and
// does not grow with the context length.

#define LID_GEMM_MAX_SCRATCH (128ll << 20)

// query rows from which the GEMM beats the wmma kernel, see the gate in ggml_cuda_lightning_indexer()
#define LID_GEMM_MIN_BATCH 8

static __global__ void lightning_indexer_gemm_reduce(
        const float * __restrict__ G, const float * __restrict__ W, const half * __restrict__ M,
        float * __restrict__ dst, const int64_t n_kv, const int64_t n_b, const int n_head,
        const int64_t sw, const int64_t sm, const int64_t sd) {
    const int64_t i_kv = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    const int64_t ib   = blockIdx.y;

    if (i_kv >= n_kv || ib >= n_b) {
        return;
    }

    const float * g = G + ib*n_head*n_kv + i_kv;
    const float * w = W + ib*sw;

    float s = 0.0f;
    for (int h = 0; h < n_head; ++h) {
        const float v = g[h*n_kv];
        s += (v > 0.0f ? v : 0.0f) * w[h];
    }

    dst[ib*sd + i_kv] = s + __half2float(M[ib*sm + i_kv]);
}

// the strides the cuBLAS path can read; no constraint on the head count
static bool ggml_cuda_lightning_indexer_gemm_ok(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];

    const int64_t n_embd = q->ne[0];
    const int64_t n_head = q->ne[1];
    const size_t  tsk    = ggml_type_size(k->type);

    // Q columns (one per head of each token) and K rows must be on a regular stride
    if (q->nb[1] % sizeof(float) != 0 || q->nb[1]/sizeof(float) < (size_t) n_embd || q->nb[2] != (size_t) n_head*q->nb[1]) {
        return false;
    }
    if (k->nb[2] % tsk != 0 || k->nb[2]/tsk < (size_t) n_embd) {
        return false;
    }
    // K that is not F32 is converted first, which needs packed rows
    const bool k_convert = k->type != GGML_TYPE_F32;
    if (k_convert && (k->nb[2] != n_embd*tsk || ggml_get_to_fp32_cuda(k->type) == nullptr)) {
        return false;
    }
    return true;
}

static bool ggml_cuda_lightning_indexer_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_cuda_lightning_indexer_gemm_ok(dst)) {
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    const int64_t n_embd   = neq0;
    const int     n_head   = (int) neq1;
    const int64_t n_batch  = neq2;
    const int64_t n_stream = neq3;
    const int64_t n_kv     = nek2;

    const size_t tsk = ggml_type_size(k->type);

    const bool k_convert = k->type != GGML_TYPE_F32;

    const int64_t nb_chunk = std::max<int64_t>(1, std::min<int64_t>(
        n_batch, LID_GEMM_MAX_SCRATCH / ((int64_t) n_head*n_kv*(int64_t) sizeof(float))));

    // the scores pick the top-k, so a caller may ask for the full f32 product instead of the tensor-core one
    const bool prec_f32 = ggml_get_op_params_i32(dst, 0) == GGML_PREC_F32;
    const cublasComputeType_t compute = prec_f32 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_16F;
    const cublasGemmAlgo_t    algo    = prec_f32 ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    cudaStream_t   stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    ggml_cuda_pool_alloc<float> g_alloc(ctx.pool(), nb_chunk*n_head*n_kv);
    ggml_cuda_pool_alloc<float> k_alloc(ctx.pool());
    if (k_convert) {
        k_alloc.alloc(n_kv*n_embd);
    }

    for (int64_t s = 0; s < n_stream; ++s) {
        const float * k_ptr = (const float *) ((const char *) k->data + s*nbk3);
        int           ldk   = (int) (nbk2/tsk);
        if (k_convert) {
            ggml_get_to_fp32_cuda(k->type)((const char *) k->data + s*nbk3, k_alloc.get(), n_kv*n_embd, stream);
            k_ptr = k_alloc.get();
            ldk   = (int) n_embd;
        }

        const float * w_s = (const float *) ((const char *)   w->data + s*nbw3);
        const half  * m_s = (const half  *) ((const char *)   m->data + (s % nem3)*nbm3);
        float       * d_s = (float       *) ((char       *) dst->data + s*nb3);

        for (int64_t b0 = 0; b0 < n_batch; b0 += nb_chunk) {
            const int64_t nb     = std::min(nb_chunk, n_batch - b0);
            const int     n_rows = (int) (nb*n_head);
            const float * q_b    = (const float *) ((const char *) q->data + b0*nbq2 + s*nbq3);

            const float alpha = 1.0f;
            const float beta  = 0.0f;
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    (int) n_kv, n_rows, (int) n_embd,
                    &alpha, k_ptr, CUDA_R_32F, ldk,
                            q_b,   CUDA_R_32F, (int) (nbq1/sizeof(float)),
                    &beta,  g_alloc.get(), CUDA_R_32F, (int) n_kv,
                    compute, algo));

            const dim3 block(256, 1, 1);
            const dim3 grid((n_kv + 255)/256, nb, 1);
            lightning_indexer_gemm_reduce<<<grid, block, 0, stream>>>(
                g_alloc.get(), w_s + b0*(nbw1/sizeof(float)), m_s + b0*(nbm1/sizeof(half)), d_s + b0*(nb1/sizeof(float)),
                n_kv, nb, n_head, (int64_t) (nbw1/sizeof(float)), (int64_t) (nbm1/sizeof(half)), (int64_t) (nb1/sizeof(float)));
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return true;
}

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];

    const float *   q_d = (const float *)   q->data;
    const char  *   k_d = (const char  *)   k->data;
    const float *   w_d = (const float *)   w->data;
    const half  *   m_d = (const half  *)   m->data;
    float       * dst_d = (      float *) dst->data;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;

    // the wmma/vec kernels exist for the DeepSeek indexer shapes only; other head counts (the qwen4exp QSA indexer
    //   has 4) take the cuBLAS path, which ggml_cuda_lightning_indexer_supported admitted them for. The 4-head
    //   kernels below take only what the GEMM declines (quantized keys): they round the keys to f16.
    const bool kernel_shape = n_embd == 128 && (n_head == 64 || n_head == 32);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (!kernel_shape) {
        if (ggml_cuda_lightning_indexer_gemm(ctx, dst)) {
            return;
        }
        GGML_ASSERT(n_embd == 128 && n_head == 4);
    }

    // On Volta the wmma kernel below is the faster of the two for few query rows and the GEMM wins from about
    //   8 rows on - measured on a V100, cuBLAS/wmma is 0.5x at 1 row, 1.0x at 4 and 1.5-1.6x from 8 rows up,
    //   nearly flat in the kv size. F32 and BF16 keys have no wmma path at all and would run the scalar kernel
    //   at a third of the speed, so they take the GEMM at any batch size. The GEMM declines the shapes it
    //   cannot handle, quantized keys among them, and then falls through.
    static const bool no_gemm = getenv("GGML_CUDA_LID_NO_GEMM") != nullptr;
    const bool k_has_wmma = k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16;
    if (kernel_shape && !no_gemm && GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA &&
            (!k_has_wmma || n_batch >= LID_GEMM_MIN_BATCH) && ggml_cuda_lightning_indexer_gemm(ctx, dst)) {
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if (n_embd == 128 && n_head == 64) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        // wmma fp16 fragments (m8n32k16) exist on Volta; nothing in the kernel needs Turing's mma.sync/ldmatrix,
        // and the scalar fallback is the depth-scaling term of DeepSeek-V4 prefill on V100
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && fp16_mma_hardware_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 32) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        // wmma fp16 fragments (m8n32k16) exist on Volta; nothing in the kernel needs Turing's mma.sync/ldmatrix,
        // and the scalar fallback is the depth-scaling term of DeepSeek-V4 prefill on V100
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && fp16_mma_hardware_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
#ifndef GGML_USE_MUSA
    // MUSA archs 21 and 22 cap static shared memory at 28 KB, below what the tile kernel stages
    } else if (n_embd == 128 && n_head == 4 && n_batch >= LIGHTNING_INDEXER_TILE_TOKENS) {
        // too few heads for a wmma tile, the tile kernel shares the keys across the tokens
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = 64;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        int num_batch_blocks = (n_batch + LIGHTNING_INDEXER_TILE_TOKENS - 1) / LIGHTNING_INDEXER_TILE_TOKENS;
        dim3 grid(num_kv_blocks, num_batch_blocks, n_stream);

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
#endif // GGML_USE_MUSA
    } else if (n_embd == 128 && n_head == 4) {
        // a batch smaller than a token tile, or MUSA, use vector kernel
        constexpr int K_VECS_PER_WARP = 8;
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        dim3 grid(num_kv_blocks, n_batch, n_stream);

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    } else {
        GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    if (neq0 != 128 || (neq1 != 64 && neq1 != 32)) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        // no wmma/vec kernel for this shape: the cuBLAS path takes any head count, the 4-head kernels the rest
        if (ggml_cuda_lightning_indexer_gemm_ok(dst)) {
            return true;
        }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (neq0 != 128 || neq1 != 4) {
            return false;
        }
    }

    // alignment checks
    for (const ggml_tensor * t : {q, k}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch(k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}

// qwen4exp QSA indexer scores, see ggml_cuda_qsa_cell_scores
static __device__ __forceinline__ float qsa_mask_value(const half v)  { return __half2float(v); }
static __device__ __forceinline__ float qsa_mask_value(const float v) { return v; }

// the block score of block b for token t of stream s: the heads' rectified scores summed in order, then the bias
static __global__ void qsa_block_scores_f32(const float * x, const float * bias, float * dst, const int n_blocks,
        const int n_h, const int64_t sx1, const int64_t sx2, const int64_t sx3, const int64_t sb1, const int64_t sb2) {
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int s = blockIdx.z;
    if (b >= n_blocks) {
        return;
    }
    const float * xb = x + s*sx3 + t*sx2 + b;
    float v = fmaxf(xb[0], 0.0f);
    for (int h = 1; h < n_h; ++h) {
        v = v + fmaxf(xb[h*sx1], 0.0f);
    }
    dst[((int64_t) s*gridDim.y + t)*n_blocks + b] = v + bias[s*sb2 + t*sb1 + b];
}

// the block scores, [n_blocks, n_tps, n_stream] contiguous, expanded to the cells and masked
template <typename mask_t>
static __global__ void qsa_cell_scores_f32(const float * scores, const int32_t * cell_blk, const mask_t * mask,
        float * dst, const int n_kv, const int n_blocks, const int64_t sc1, const int64_t sm1, const int64_t sm2,
        const int64_t sd1, const int64_t sd2) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int s = blockIdx.z;
    if (c >= n_kv) {
        return;
    }
    const int b = cell_blk[s*sc1 + c];
    const float v = scores[((int64_t) s*gridDim.y + t)*n_blocks + b];
    dst[s*sd2 + t*sd1 + c] = v + qsa_mask_value(mask[s*sm2 + t*sm1 + c]);
}

void ggml_cuda_qsa_cell_scores(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * bias,
        const ggml_tensor * cell_blk, const ggml_tensor * mask, const size_t mask_nb_stream, ggml_tensor * dst) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32 && cell_blk->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(x->nb[0] == sizeof(float) && bias->nb[0] == sizeof(float) && cell_blk->nb[0] == sizeof(int32_t));
    GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type) && dst->nb[0] == sizeof(float));

    const int n_blocks = (int) x->ne[0];
    const int n_h      = (int) x->ne[1];
    const int n_tps    = (int) x->ne[2];
    const int n_stream = (int) x->ne[3];
    const int n_kv     = (int) cell_blk->ne[0];
    GGML_ASSERT(dst->ne[0] == n_kv && dst->ne[1] == n_tps && dst->ne[2] == n_stream);
    GGML_ASSERT(n_tps <= 65535 && n_stream <= 65535);

    cudaStream_t stream = ctx.stream();
    const int block_size = 256;

    // the block scores first: dst often sits on x (ggml-alloc freed it after the RELU), and a cell reads its block's
    //   sum instead of summing the heads again
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), (size_t) n_blocks*n_tps*n_stream);
    {
        const dim3 grid((n_blocks + block_size - 1)/block_size, n_tps, n_stream);
        qsa_block_scores_f32<<<grid, block_size, 0, stream>>>((const float *) x->data, (const float *) bias->data,
            scores.get(), n_blocks, n_h, x->nb[1]/sizeof(float), x->nb[2]/sizeof(float), x->nb[3]/sizeof(float),
            bias->nb[1]/sizeof(float), bias->nb[2]/sizeof(float));
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t  ts  = ggml_type_size(mask->type);
    const int64_t sc1 = cell_blk->nb[1]/sizeof(int32_t);
    const int64_t sm1 = mask->nb[1]/ts;
    const int64_t sm2 = mask_nb_stream/ts;
    const int64_t sd1 = dst->nb[1]/sizeof(float);
    const int64_t sd2 = dst->nb[2]/sizeof(float);
    const dim3 grid((n_kv + block_size - 1)/block_size, n_tps, n_stream);
    if (mask->type == GGML_TYPE_F16) {
        qsa_cell_scores_f32<<<grid, block_size, 0, stream>>>(scores.get(), (const int32_t *) cell_blk->data,
            (const half *) mask->data, (float *) dst->data, n_kv, n_blocks, sc1, sm1, sm2, sd1, sd2);
    } else {
        qsa_cell_scores_f32<<<grid, block_size, 0, stream>>>(scores.get(), (const int32_t *) cell_blk->data,
            (const float *) mask->data, (float *) dst->data, n_kv, n_blocks, sc1, sm1, sm2, sd1, sd2);
    }
    CUDA_CHECK(cudaGetLastError());
}

// the QSA attention mask, see ggml_cuda_qsa_topk_mask; a block owns chunk cells of one row and keeps their kq values,
//   so dst may sit exactly on kq
template <typename dst_t, typename val_t, int chunk>
static __global__ void qsa_topk_mask(const dst_t fill, const val_t * vals, const int32_t * idx, const dst_t * kq,
        dst_t * dst, const int n_kv, const int n_top_k, const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int64_t si1, const int64_t si2, const int64_t sk1, const int64_t sk3, const int64_t sd1, const int64_t sd3) {
    __shared__ dst_t kq_chunk[chunk];

    const int t  = blockIdx.y;
    const int s  = blockIdx.z;
    const int c0 = blockIdx.x*chunk;
    const int c1 = min(c0 + chunk, n_kv);

    const dst_t * kq_row  = kq  + s*sk3 + t*sk1;
    dst_t       * dst_row = dst + s*sd3 + t*sd1;
    for (int c = c0 + threadIdx.x; c < c1; c += blockDim.x) {
        const dst_t m = kq_row[c];
        kq_chunk[c - c0] = m;
        dst_row[c] = ggml_cuda_cast<dst_t>(ggml_cuda_cast<float>(fill) + ggml_cuda_cast<float>(m));
    }
    __syncthreads();

    const int32_t * idx_row  = idx  + s*si2 + t*si1;
    const val_t   * vals_row = vals + s*sv3 + t*sv2;
    for (int k = threadIdx.x; k < n_top_k; k += blockDim.x) {
        const int c = idx_row[k];
        if (c >= c0 && c < c1) {
            const dst_t v = ggml_cuda_cast<dst_t>(ggml_cuda_cast<float>(vals_row[k*sv1]));
            dst_row[c] = ggml_cuda_cast<dst_t>(ggml_cuda_cast<float>(v) + ggml_cuda_cast<float>(kq_chunk[c - c0]));
        }
    }
}

void ggml_cuda_qsa_topk_mask(ggml_backend_cuda_context & ctx, const float fill, const ggml_tensor * vals,
        const ggml_tensor * idx, const ggml_tensor * kq, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == kq->type && (dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32));
    GGML_ASSERT(vals->type == GGML_TYPE_F16 || vals->type == GGML_TYPE_F32);
    GGML_ASSERT(idx->type == GGML_TYPE_I32 && idx->nb[0] == sizeof(int32_t));
    GGML_ASSERT(ggml_are_same_shape(dst, kq) && dst->ne[2] == 1);
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type) && kq->nb[0] == ggml_type_size(kq->type));
    GGML_ASSERT(dst->data == kq->data ? ggml_are_same_stride(dst, kq) : !ggml_cuda_tensors_overlap(dst, kq));

    const int n_kv     = (int) dst->ne[0];
    const int n_tps    = (int) dst->ne[1];
    const int n_stream = (int) dst->ne[3];
    const int n_top_k  = (int) idx->ne[0];
    GGML_ASSERT(idx->ne[1] == n_tps && idx->ne[2] == n_stream);
    GGML_ASSERT(vals->ne[0] == 1 && vals->ne[1] == n_top_k && vals->ne[2] == n_tps && vals->ne[3] == n_stream);
    GGML_ASSERT(n_tps <= 65535 && n_stream <= 65535);

    cudaStream_t stream = ctx.stream();

    // the picked cells and their values are read after other blocks write their rows: copied first if dst covers them
    const char * idx_d  = (const char *) idx->data;
    const char * vals_d = (const char *) vals->data;
    ggml_cuda_pool_alloc<char> idx_copy(ctx.pool());
    ggml_cuda_pool_alloc<char> vals_copy(ctx.pool());
    for (int k = 0; k < 2; k++) {
        const ggml_tensor * t = k == 0 ? idx : vals;
        if (!ggml_cuda_tensors_overlap(dst, t)) {
            continue;
        }
        GGML_ASSERT(ggml_is_contiguous(t));
        char * copy = (k == 0 ? idx_copy : vals_copy).alloc(ggml_nbytes(t));
        CUDA_CHECK(cudaMemcpyAsync(copy, t->data, ggml_nbytes(t), cudaMemcpyDeviceToDevice, stream));
        (k == 0 ? idx_d : vals_d) = copy;
    }

    const size_t  tv  = ggml_type_size(vals->type);
    const size_t  td  = ggml_type_size(dst->type);
    const int64_t sv1 = vals->nb[1]/tv;
    const int64_t sv2 = vals->nb[2]/tv;
    const int64_t sv3 = vals->nb[3]/tv;
    const int64_t si1 = idx->nb[1]/sizeof(int32_t);
    const int64_t si2 = idx->nb[2]/sizeof(int32_t);
    const int64_t sk1 = kq->nb[1]/td;
    const int64_t sk3 = kq->nb[3]/td;
    const int64_t sd1 = dst->nb[1]/td;
    const int64_t sd3 = dst->nb[3]/td;

    // every block reads the row's whole top-k list, so a chunk is several cells per thread
    const int     block_size = 256;
    constexpr int chunk      = 2048;
    const dim3 grid((n_kv + chunk - 1)/chunk, n_tps, n_stream);

#define QSA_TOPK_MASK(dst_t, val_t) \
    qsa_topk_mask<dst_t, val_t, chunk><<<grid, block_size, 0, stream>>>(ggml_cuda_cast<dst_t>(fill), (const val_t *) vals_d, \
        (const int32_t *) idx_d, (const dst_t *) kq->data, (dst_t *) dst->data, n_kv, n_top_k, sv1, sv2, sv3, si1, si2, \
        sk1, sk3, sd1, sd3)
    if (dst->type == GGML_TYPE_F16) {
        if (vals->type == GGML_TYPE_F32) {
            QSA_TOPK_MASK(half, float);
        } else {
            QSA_TOPK_MASK(half, half);
        }
    } else {
        if (vals->type == GGML_TYPE_F32) {
            QSA_TOPK_MASK(float, float);
        } else {
            QSA_TOPK_MASK(float, half);
        }
    }
#undef QSA_TOPK_MASK
    CUDA_CHECK(cudaGetLastError());
}
