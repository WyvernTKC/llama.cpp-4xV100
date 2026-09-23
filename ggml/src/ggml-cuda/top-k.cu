#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(CUB_TOP_K_AVAILABLE)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift,
        bool first) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    for (int bin = tid; bin < NBINS; bin += BLOCK_SIZE) {
        histogram[bin] = 0;
    }
    __syncthreads();

    // the first pass counts every key, so the state is not read before a select has written it
    const uint32_t prefix      = first ? 0 : states[row].prefix;
    const uint32_t prefix_mask = first ? 0 : states[row].prefix_mask;
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & prefix_mask) == prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    for (int bin = tid; bin < NBINS; bin += BLOCK_SIZE) {
        block_histograms[histogram_offset + bin] = histogram[bin];
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift,
        int k,
        bool first) {
    constexpr int NBINS = 1 << RADIX_BITS;
    constexpr int BINS_PER_THREAD = NBINS / BLOCK_SIZE;
    static_assert(NBINS % BLOCK_SIZE == 0, "NBINS must be a multiple of BLOCK_SIZE");

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];
    __shared__ int suffix[BLOCK_SIZE];

    // read before the barriers below, since one thread overwrites the state at the end
    const top_k_radix_state prev = first ? top_k_radix_state{0, 0, k, 0, 0} : states[row];

    for (int bin = tid; bin < NBINS; bin += BLOCK_SIZE) {
        int count = 0;
        for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
            const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
            count += block_histograms[offset + bin];
        }
        histogram[bin] = count;
    }
    __syncthreads();

    // thread tid owns a contiguous run of bins; suffix[tid] counts the keys in its run and every run above it
    int own = 0;
    for (int i = 0; i < BINS_PER_THREAD; ++i) {
        own += histogram[tid*BINS_PER_THREAD + i];
    }
    suffix[tid] = own;
    __syncthreads();
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        const int add = tid + offset < BLOCK_SIZE ? suffix[tid + offset] : 0;
        __syncthreads();
        suffix[tid] += add;
        __syncthreads();
    }

    // exactly one run holds the key of the wanted rank
    const int above = suffix[tid] - own;
    if (above < prev.rank && prev.rank <= suffix[tid]) {
        int rank = prev.rank - above;
        int bin = (tid + 1)*BINS_PER_THREAD - 1;
        while (histogram[bin] < rank) {
            rank -= histogram[bin--];
        }

        top_k_radix_state state;
        state.prefix        = prev.prefix | (uint32_t) bin << shift;
        state.prefix_mask   = prev.prefix_mask | (uint32_t) (NBINS - 1) << shift;
        state.rank          = rank;
        state.greater_count = 0;
        state.equal_count   = 0;
        states[row] = state;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

template<int RADIX_BITS>
static void top_k_radix_cuda_impl(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, int blocks_per_row, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int NBINS = 1 << RADIX_BITS;
    constexpr int NPASSES = (32 + RADIX_BITS - 1) / RADIX_BITS;

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    const dim3 row_grid(blocks_per_row * nrows);
    for (int pass = 0; pass < NPASSES; ++pass) {
        // when RADIX_BITS does not divide 32, the last digit overlaps bits the prefix already holds, which is harmless
        const int shift = std::max(32 - (pass + 1)*RADIX_BITS, 0);
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift, pass == 0);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift, k, pass == 0);
    }

    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    // 11-bit digits need 3 passes instead of 4, but the select block reads 8x the bins per histogram block,
    //   so cap the histogram blocks at 16 and take them only for a few rows (decode). Past 64k columns the
    //   16 blocks get too long and 8-bit digits over 64 blocks are faster.
    const int blocks_per_row_11 = std::min((ncols + 1023) / 1024, 16);
    if (ncols <= 65536 && (int64_t) nrows * blocks_per_row_11 <= 256) {
        top_k_radix_cuda_impl<11>(pool, src, dst, ncols, nrows, k, blocks_per_row_11, stream);
    } else {
        const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);
        top_k_radix_cuda_impl<8>(pool, src, dst, ncols, nrows, k, blocks_per_row, stream);
    }
}

#endif // !defined(CUB_TOP_K_AVAILABLE)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // CUB has a fast single-tile sort up to 4096; above that it falls back to a much slower
    //   multi-pass sort of the whole row just to keep k elements
    if (ncols > 4096) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
