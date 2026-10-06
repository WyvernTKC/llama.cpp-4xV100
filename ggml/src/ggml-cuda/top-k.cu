#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
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

static constexpr int TOP_K_RADIX_MAX_BLOCKS_PER_ROW = 64;

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
};

// block-wide exclusive scan of v, and the block total
template<int BLOCK_SIZE>
static __device__ __forceinline__ int top_k_block_exclusive_scan(const int v, int & total, int * warp_sums) {
    constexpr int NWARPS = BLOCK_SIZE / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    int x = v;
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        const int y = __shfl_up_sync(0xffffffff, x, offset, WARP_SIZE);
        x += lane >= offset ? y : 0;
    }
    if (lane == WARP_SIZE - 1) {
        warp_sums[warp] = x;
    }
    __syncthreads();

    int before = 0;
    total = 0;
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) {
        before += w < warp ? warp_sums[w] : 0;
        total  += warp_sums[w];
    }
    __syncthreads();
    return before + x - v;
}

// Block row_block of a row takes the columns [row_block*chunk, (row_block + 1)*chunk). With the blocks and the
//   columns of each block taken in order, the gather writes the indices in column order, like top_k_radix_small.
template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int * __restrict__ block_greater,
        int ncols,
        int blocks_per_row,
        int chunk,
        int shift,
        bool first,
        bool last) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];
    __shared__ int n_greater;

    for (int bin = tid; bin < NBINS; bin += BLOCK_SIZE) {
        histogram[bin] = 0;
    }
    if (tid == 0) {
        n_greater = 0;
    }
    __syncthreads();

    // the first pass counts every key, so the state is not read before a select has written it
    const uint32_t prefix      = first ? 0 : states[row].prefix;
    const uint32_t prefix_mask = first ? 0 : states[row].prefix_mask;
    const int      col_end     = min(ncols, (row_block + 1)*chunk);
    int greater = 0;
    for (int col = row_block*chunk + tid; col < col_end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & prefix_mask) == prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        } else {
            greater += (key & prefix_mask) > prefix;
        }
    }
    // the keys above the digits taken so far: with the last histogram they give each block its place in the output
    if (last && greater > 0) {
        atomicAdd(&n_greater, greater);
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    for (int bin = tid; bin < NBINS; bin += BLOCK_SIZE) {
        block_histograms[histogram_offset + bin] = histogram[bin];
    }
    if (last && tid == 0) {
        block_greater[(size_t) row * blocks_per_row + row_block] = n_greater;
    }
}

template<int BLOCK_SIZE, int RADIX_BITS, bool LAST>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        const int * __restrict__ block_greater,
        top_k_radix_state * __restrict__ states,
        int2 * __restrict__ block_offsets,
        int blocks_per_row,
        int shift,
        int k,
        bool first,
        bool runs_fit) {
    constexpr int NBINS = 1 << RADIX_BITS;
    constexpr int BINS_PER_THREAD = NBINS / BLOCK_SIZE;
    static_assert(NBINS % BLOCK_SIZE == 0, "NBINS must be a multiple of BLOCK_SIZE");

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];
    __shared__ int suffix[BLOCK_SIZE];
    [[maybe_unused]] __shared__ int sel_bin;
    [[maybe_unused]] __shared__ int2 counts[LAST ? TOP_K_RADIX_MAX_BLOCKS_PER_ROW : 1];
    // the last pass: the keys of each block in the run of bins of each thread, at most one chunk
    [[maybe_unused]] __shared__ uint16_t runs[LAST ? TOP_K_RADIX_MAX_BLOCKS_PER_ROW : 1][BLOCK_SIZE];

    // read before the barriers below, since one thread overwrites the state at the end
    const top_k_radix_state prev = first ? top_k_radix_state{0, 0, k} : states[row];
    const bool keep_runs = LAST && runs_fit;

    // thread tid owns a contiguous run of bins; suffix[tid] counts the keys in its run and every run above it
    int own = 0;
    if (keep_runs) {
        int acc[BINS_PER_THREAD] = { 0 };
        for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
            const int * h = block_histograms + ((size_t) row * blocks_per_row + row_block) * NBINS + tid*BINS_PER_THREAD;
            int run = 0;
            if constexpr (BINS_PER_THREAD % 4 == 0) {
#pragma unroll
                for (int i = 0; i < BINS_PER_THREAD; i += 4) {
                    const int4 c = *(const int4 *) (h + i);
                    acc[i + 0] += c.x;
                    acc[i + 1] += c.y;
                    acc[i + 2] += c.z;
                    acc[i + 3] += c.w;
                    run += c.x + c.y + c.z + c.w;
                }
            } else {
#pragma unroll
                for (int i = 0; i < BINS_PER_THREAD; ++i) {
                    acc[i] += h[i];
                    run    += h[i];
                }
            }
            own += run;
            if constexpr (LAST) {
                runs[row_block][tid] = (uint16_t) run;
            }
        }
#pragma unroll
        for (int i = 0; i < BINS_PER_THREAD; ++i) {
            histogram[tid*BINS_PER_THREAD + i] = acc[i];
        }
    } else {
        for (int bin = tid; bin < NBINS; bin += BLOCK_SIZE) {
            int count = 0;
            for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
                const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
                count += block_histograms[offset + bin];
            }
            histogram[bin] = count;
        }
        __syncthreads();
        for (int i = 0; i < BINS_PER_THREAD; ++i) {
            own += histogram[tid*BINS_PER_THREAD + i];
        }
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
        state.prefix      = prev.prefix | (uint32_t) bin << shift;
        state.prefix_mask = prev.prefix_mask | (uint32_t) (NBINS - 1) << shift;
        state.rank        = rank;
        states[row] = state;
        if constexpr (LAST) {
            sel_bin = bin;
        }
    }

    if constexpr (LAST) {
        __syncthreads();

        // where each block writes: the keys above the k-th of the blocks before it, and the keys equal to it
        const int bin     = sel_bin;
        const int run_end = (bin / BINS_PER_THREAD + 1)*BINS_PER_THREAD;
        const int lane    = tid % WARP_SIZE;
        for (int b = tid / WARP_SIZE; b < blocks_per_row; b += BLOCK_SIZE / WARP_SIZE) {
            const size_t block = (size_t) row * blocks_per_row + b;
            const int * h = block_histograms + block * NBINS;
            int greater = 0;
            if (keep_runs) {
                // the runs above the one that holds bin, then the bins of that run above bin
                for (int t = bin / BINS_PER_THREAD + 1 + lane; t < BLOCK_SIZE; t += WARP_SIZE) {
                    greater += runs[b][t];
                }
                for (int d = bin + 1 + lane; d < run_end; d += WARP_SIZE) {
                    greater += h[d];
                }
            } else {
                for (int d = bin + 1 + lane; d < NBINS; d += WARP_SIZE) {
                    greater += h[d];
                }
            }
            greater = warp_reduce_sum(greater);
            if (lane == 0) {
                counts[b] = make_int2(greater + block_greater[block], h[bin]);
            }
        }
        __syncthreads();
        if (tid == 0) {
            int2 acc = make_int2(0, 0);
            for (int b = 0; b < blocks_per_row; ++b) {
                block_offsets[(size_t) row * blocks_per_row + b] = acc;
                acc.x += counts[b].x;
                acc.y += counts[b].y;
            }
        }
    }
}

// The keys above the k-th first, then the first rank keys equal to it, each in column order. No atomics, so the
//   indices and their order do not change from run to run, ties included.
template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        const int2 * __restrict__ block_offsets,
        int ncols,
        int k,
        int blocks_per_row,
        int chunk) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    __shared__ int warp_sums[BLOCK_SIZE / WARP_SIZE];

    const uint32_t prefix = states[row].prefix;
    const int      rank   = states[row].rank;
    const int2     offset = block_offsets[(size_t) row * blocks_per_row + row_block];
    int n_greater = offset.x;
    int n_equal   = offset.y;

    const int col_begin = row_block*chunk;
    const int col_end   = min(ncols, col_begin + chunk);
    for (int base = col_begin; base < col_end; base += BLOCK_SIZE) {
        const int      col     = base + tid;
        const bool     valid   = col < col_end;
        const uint32_t key     = valid ? top_k_float_to_ordered(row_src[col]) : 0;
        const bool     greater = valid && key > prefix;
        const bool     equal   = valid && key == prefix;

        // both counts are at most BLOCK_SIZE, so they share one scan
        int total;
        const int pos = top_k_block_exclusive_scan<BLOCK_SIZE>((int) greater | (int) equal << 16, total, warp_sums);
        if (greater) {
            row_dst[n_greater + (pos & 0xFFFF)] = col;
        }
        if (equal && n_equal + (pos >> 16) < rank) {
            row_dst[k - rank + n_equal + (pos >> 16)] = col;
        }
        n_greater += total & 0xFFFF;
        n_equal   += total >> 16;
    }
}

template<int RADIX_BITS>
static void top_k_radix_cuda_impl(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, int blocks_per_row, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int NBINS = 1 << RADIX_BITS;
    constexpr int NPASSES = (32 + RADIX_BITS - 1) / RADIX_BITS;
    GGML_ASSERT(blocks_per_row <= TOP_K_RADIX_MAX_BLOCKS_PER_ROW);

    // whole block steps per chunk
    const int chunk = ((ncols + blocks_per_row - 1) / blocks_per_row + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    ggml_cuda_pool_alloc<int> greater_alloc(pool, (size_t) nrows * blocks_per_row);
    ggml_cuda_pool_alloc<int2> offsets_alloc(pool, (size_t) nrows * blocks_per_row);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    // the last select keeps per-block counts of at most one chunk in 16 bits
    const bool runs_fit = chunk <= UINT16_MAX;

    const dim3 row_grid(blocks_per_row * nrows);
    for (int pass = 0; pass < NPASSES; ++pass) {
        // when RADIX_BITS does not divide 32, the last digit overlaps bits the prefix already holds, which is harmless
        const int  shift = std::max(32 - (pass + 1)*RADIX_BITS, 0);
        const bool first = pass == 0;
        const bool last  = pass == NPASSES - 1;
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, greater_alloc.get(), ncols, blocks_per_row, chunk, shift, first, last);
        if (last) {
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS, true>
                <<<nrows, BLOCK_SIZE, 0, stream>>>(
                    histograms, greater_alloc.get(), states, offsets_alloc.get(), blocks_per_row, shift, k, first, runs_fit);
        } else {
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS, false>
                <<<nrows, BLOCK_SIZE, 0, stream>>>(
                    histograms, greater_alloc.get(), states, offsets_alloc.get(), blocks_per_row, shift, k, first, runs_fit);
        }
    }

    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, offsets_alloc.get(), ncols, k, blocks_per_row, chunk);
}

// rows of at most NPER*BLOCK_SIZE columns: one block per row runs the whole radix select on keys held in
//   registers, in one launch. The indices come out in column order, the keys above the k-th first.
template<int BLOCK_SIZE, int NPER>
static __global__ void top_k_radix_small(
        const float * __restrict__ src,
        int * __restrict__ dst,
        int ncols,
        int k) {
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    static_assert(NBINS == BLOCK_SIZE, "one bin per thread");

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const float * row_src = src + (size_t) blockIdx.x * ncols;
    int * row_dst = dst + (size_t) blockIdx.x * k;

    __shared__ int histogram[NBINS];
    __shared__ int warp_sums[BLOCK_SIZE / WARP_SIZE];
    __shared__ uint32_t sel_prefix;
    __shared__ int sel_rank;

    uint32_t key[NPER];
#pragma unroll
    for (int i = 0; i < NPER; ++i) {
        const int col = i*BLOCK_SIZE + tid;
        key[i] = col < ncols ? top_k_float_to_ordered(row_src[col]) : 0;
    }

    uint32_t prefix = 0;
    uint32_t prefix_mask = 0;
    int rank = k;
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        histogram[tid] = 0;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NPER; ++i) {
            if (i*BLOCK_SIZE + tid < ncols && (key[i] & prefix_mask) == prefix) {
                atomicAdd(&histogram[(key[i] >> shift) & (NBINS - 1)], 1);
            }
        }
        __syncthreads();

        // keys whose digit is >= this thread's bin: a suffix sum within the warp, plus the warps above
        const int own = histogram[tid];
        int ge = own;
#pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            const int y = __shfl_down_sync(0xffffffff, ge, offset);
            ge += lane + offset < WARP_SIZE ? y : 0;
        }
        if (lane == 0) {
            warp_sums[warp] = ge;
        }
        __syncthreads();
#pragma unroll
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) {
            ge += w > warp ? warp_sums[w] : 0;
        }

        // exactly one bin holds the key of the wanted rank
        if (ge - own < rank && rank <= ge) {
            sel_prefix = prefix | (uint32_t) tid << shift;
            sel_rank   = rank - (ge - own);
        }
        __syncthreads();
        prefix       = sel_prefix;
        rank         = sel_rank;
        prefix_mask |= (uint32_t) (NBINS - 1) << shift;
    }

    // prefix is now the k-th key and rank the number of keys equal to it that are taken
    int n_greater = 0;
    int n_equal   = 0;
#pragma unroll
    for (int i = 0; i < NPER; ++i) {
        const int  col     = i*BLOCK_SIZE + tid;
        const bool valid   = col < ncols;
        const bool greater = valid && key[i] > prefix;
        const bool equal   = valid && key[i] == prefix;

        // both counts are at most BLOCK_SIZE, so they share one scan
        int total;
        const int pos = top_k_block_exclusive_scan<BLOCK_SIZE>((int) greater | (int) equal << 16, total, warp_sums);
        if (greater) {
            row_dst[n_greater + (pos & 0xFFFF)] = col;
        }
        if (equal && n_equal + (pos >> 16) < rank) {
            row_dst[k - rank + n_equal + (pos >> 16)] = col;
        }
        n_greater += total & 0xFFFF;
        n_equal   += total >> 16;
    }
}

static void top_k_radix_small_cuda(const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    const int nper = (ncols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GGML_ASSERT(nper <= 16);
    auto launch = [&](auto kernel) {
        kernel<<<nrows, BLOCK_SIZE, 0, stream>>>(src, dst, ncols, k);
    };
    if (nper <= 1) {
        launch(top_k_radix_small<BLOCK_SIZE, 1>);
    } else if (nper <= 2) {
        launch(top_k_radix_small<BLOCK_SIZE, 2>);
    } else if (nper <= 4) {
        launch(top_k_radix_small<BLOCK_SIZE, 4>);
    } else if (nper <= 8) {
        launch(top_k_radix_small<BLOCK_SIZE, 8>);
    } else {
        launch(top_k_radix_small<BLOCK_SIZE, 16>);
    }
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
    // above 4096 CUB falls back to a multi-pass sort of the whole row just to keep k elements
    if (ncols > 4096) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
    // up to 4096 keys a row fits one block, so the select is a single launch: on a V100 2-6x the argsort for one
    //   row and 10x for a thousand rows of 1024. Only many rows of a few keys pack tighter into the bitonic sort.
    if (ncols > 64 || nrows <= 64) {
        top_k_radix_small_cuda(src0_d, dst_d, ncols, nrows, k, stream);
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
