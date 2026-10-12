#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
//
// A block is one expert. Its warps each scan a contiguous range of tokens, a lane per id, and keep their matches in shared
// memory from their first token on: the ids of a token follow each other and its experts differ, so a token matches at
// most once and the order of the ids is that of the tokens. The matches are written after the counts of the warps
// before are known, so the result does not depend on the number of warps. One warp per expert ranking 4 tokens of 8 ids
// with 7 shuffles a step took 80-90 us per call for 2048 tokens.
//
// MUL_MAT_ID also allows an expert to repeat within a token's row (top-k routing never does), and every such slot needs a
// compact row of its own. A warp can then match more often than it has tokens, or a vec4 lane twice in its 4 ids: the
// warp stops storing past its room and the block makes an exact second pass instead, a lane per id, that counts the
// matches of each warp and then writes them straight to global memory in (token, slot) order.
#define MM_IDS_HELPER_MAX_WARPS 8

// the number of lanes below this one for which pred holds, and in total
static __device__ __forceinline__ int mm_ids_rank(const bool pred, int & total) {
#if defined(GGML_USE_HIP)
    const uint64_t mask = __ballot(pred);
    total = __popcll(mask);
    return __popcll(mask & ((uint64_t(1) << threadIdx.x) - 1));
#else
    const uint32_t mask = __ballot_sync(0xFFFFFFFF, pred);
    total = __popc(mask);
    return __popc(mask & ((1u << threadIdx.x) - 1));
#endif // defined(GGML_USE_HIP)
}

// The second pass for an expert that repeats within a row: count the matches of each warp exactly, a lane per id, then
// write them in order, i being the (token, slot) index it*n_expert_used + iex. Returns the column count of the expert.
// Out of line: inlined it made the kernel 1.5x longer and a 32-token call 9% slower (5.5 -> 6.0 us on a V100).
static __device__ __noinline__ int mm_ids_helper_repeated(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int * warp_count,
        const int expert, const int i_begin, const int i_end, const int n_expert_used, const int nex_prev,
        const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int warp   = threadIdx.y;
    const int nwarps = blockDim.y;

    int n_match = 0;
    for (int i0 = i_begin; i0 < i_end; i0 += warp_size) {
        const int i  = i0 + threadIdx.x;
        const int it = i / n_expert_used;
        int n_add;
        mm_ids_rank(i < i_end && ids[it*si1 + i - it*n_expert_used] == expert, n_add);
        n_match += n_add;
    }

    __syncthreads(); // all warps have read the counts of the first pass
    if (threadIdx.x == 0) {
        warp_count[warp] = n_match;
    }
    __syncthreads();

    int it_offset = 0;
    int n_compact = 0;
    for (int w = 0; w < nwarps; ++w) {
        it_offset += w < warp ? warp_count[w] : 0;
        n_compact += warp_count[w];
    }

    int itc = 0;
    for (int i0 = i_begin; i0 < i_end; i0 += warp_size) {
        const int  i     = i0 + threadIdx.x;
        const int  it    = i / n_expert_used;
        const int  iex   = i - it*n_expert_used;
        const bool match = i < i_end && ids[it*si1 + iex] == expert;
        int n_add;
        const int rank = mm_ids_rank(match, n_add);
        if (match) {
            const int ic = nex_prev + it_offset + itc + rank;
            ids_dst[ic] = i;
            if (write_inverse) {
                ids_src1[i] = ic;
            } else {
                ids_src1[ic] = it*sis1 + iex % nchannels_y;
            }
        }
        itc += n_add;
    }
    return n_compact;
}

// vec4: n_expert_used_template and the row stride of the ids are multiples of 4, and the ids are 16-byte aligned. A lane
//   then reads 4 ids of one token at once, of which at most one is the expert unless the ids repeat: one ballot ranks 4
//   ids a lane.
template <int n_expert_used_template, bool vec4>
__launch_bounds__(MM_IDS_HELPER_MAX_WARPS*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;
    const int warp   = threadIdx.y;
    const int nwarps = blockDim.y;

    const int tokens_per_warp = (n_tokens + nwarps - 1) / nwarps;
    const int it_begin        = min(warp*tokens_per_warp, n_tokens);
    const int it_end          = min(it_begin + tokens_per_warp, n_tokens);
    const int n_store         = it_end - it_begin; // room of the warp in shared memory

    // at most one match per token, so the matches of a warp fit from its first token on; the counts of the warps
    //   follow (dynamic too, as the launch gives the dynamic part all the shared memory a block can opt in to)
    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store         = (mm_ids_helper_store *) data_mm_ids_helper + it_begin;
    int                 * warp_count    = (int *) ((mm_ids_helper_store *) data_mm_ids_helper + n_tokens);
    int                 * warp_nex_prev = warp_count + MM_IDS_HELPER_MAX_WARPS;

    int  nex_prev   = 0;     // Number of columns for experts with a lower index.
    int  it_compact = 0;     // Running index for the compact slice of this expert.
    bool repeat     = false; // vec4: the expert more than once in the 4 ids of a lane

    // the ids of a few iterations are loaded first, each iteration waited on its own load otherwise
    constexpr int ids_per_lane = 4;

    const int i_end = it_end*n_expert_used;
    if constexpr (vec4) {
        static_assert(n_expert_used_template % 4 == 0, "vec4 needs whole groups of 4 ids per token");
        const int4 * ids4 = (const int4 *) ids;
        for (int i0 = it_begin*n_expert_used; i0 < i_end; i0 += ids_per_lane*4*warp_size) {
            int4 v[ids_per_lane];
#pragma unroll
            for (int k = 0; k < ids_per_lane; ++k) {
                const int i   = i0 + 4*(k*warp_size + (int) threadIdx.x);
                const int it  = i / n_expert_used;
                const int iex = i - it*n_expert_used;
                v[k] = i < i_end ? ids4[(it*si1 + iex)/4] : make_int4(INT_MAX, INT_MAX, INT_MAX, INT_MAX);
            }
#pragma unroll
            for (int k = 0; k < ids_per_lane; ++k) {
                const int i  = i0 + 4*(k*warp_size + (int) threadIdx.x);
                const int lt = int(v[k].x < expert) + int(v[k].y < expert) + int(v[k].z < expert) + int(v[k].w < expert);
                nex_prev += lt;
                const bool ex = v[k].x == expert;
                const bool ey = v[k].y == expert;
                const bool ez = v[k].z == expert;
                const bool ew = v[k].w == expert;
                repeat = repeat || (ex && ey) || (ez && ew) || ((ex || ey) && (ez || ew)); // at least two
                const int j  = ex ? 0 : ey ? 1 : ez ? 2 : ew ? 3 : -1;

                int n_add;
                const int rank = mm_ids_rank(j >= 0, n_add);
                if (j >= 0 && it_compact + rank < n_store) {
                    const int it = i / n_expert_used;
                    store[it_compact + rank] = mm_ids_helper_store(it, i - it*n_expert_used + j);
                }
                it_compact += n_add;
            }
        }
    } else {
        for (int i0 = it_begin*n_expert_used; i0 < i_end; i0 += ids_per_lane*warp_size) {
            int expert_used[ids_per_lane];
#pragma unroll
            for (int k = 0; k < ids_per_lane; ++k) {
                const int i   = i0 + k*warp_size + (int) threadIdx.x;
                const int it  = i / n_expert_used;
                const int iex = i - it*n_expert_used;
                expert_used[k] = i < i_end ? ids[it*si1 + iex] : INT_MAX;
            }
#pragma unroll
            for (int k = 0; k < ids_per_lane; ++k) {
                const int i   = i0 + k*warp_size + (int) threadIdx.x;
                const int it  = i / n_expert_used;
                const int iex = i - it*n_expert_used;
                nex_prev += expert_used[k] < expert;

                int n_add;
                const int rank = mm_ids_rank(expert_used[k] == expert, n_add);
                if (expert_used[k] == expert && it_compact + rank < n_store) {
                    store[it_compact + rank] = mm_ids_helper_store(it, iex);
                }
                it_compact += n_add;
            }
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    // a count of -1 marks a warp whose matches did not fit, or that missed some in its vec4 ids
    bool repeated = it_compact > n_store;
    if constexpr (vec4) {
        int n_repeat;
        mm_ids_rank(repeat, n_repeat);
        repeated = repeated || n_repeat > 0;
    }

    if (threadIdx.x == 0) {
        warp_count   [warp] = repeated ? -1 : it_compact;
        warp_nex_prev[warp] = nex_prev;
    }
    __syncthreads();

    int it_offset  = 0; // the matches of the warps before
    int n_compact  = 0;
    nex_prev       = 0;
    repeated       = false;
    for (int w = 0; w < nwarps; ++w) {
        it_offset += w < warp ? warp_count[w] : 0;
        n_compact += warp_count[w];
        nex_prev  += warp_nex_prev[w];
        repeated   = repeated || warp_count[w] < 0;
    }

    if (!repeated) {
        for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
            const mm_ids_helper_store store_it = store[itc];
            const int it       = store_it.it();
            const int iex_used = store_it.iex_used();
            const int ic       = nex_prev + it_offset + itc;
            ids_dst[ic] = it*n_expert_used + iex_used;
            // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex_used] = ic;
            } else {
                ids_src1[ic] = it*sis1 + iex_used % nchannels_y;
            }
        }
    } else {
        n_compact = mm_ids_helper_repeated(ids, ids_src1, ids_dst, warp_count, expert, it_begin*n_expert_used, i_end,
            n_expert_used, nex_prev, nchannels_y, si1, sis1, write_inverse);
    }

    if (warp != 0 || threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + n_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    // whole groups of 4 ids per token, aligned: the top-k view of an argsort has rows of n_expert ids
    constexpr bool vec4_ok = n_expert_used_template != 0 && n_expert_used_template % 4 == 0;
    const bool vec4 = vec4_ok && si1 % 4 == 0 && (uintptr_t) ids % sizeof(int4) == 0;

    // a warp per 8 iterations of 4 ids a lane, a single one for short batches. 2048 tokens of 8 experts, us per call
    //   with 128 / 256 experts: 4 warps 17.7 / 17.4, 8 warps 13.6 / 17.0, 16 warps 13.5 / 20.2
    const int n_iter = (n_tokens*n_expert_used_var + 4*warp_size - 1) / (4*warp_size);
    const int nwarps = std::max(1, std::min(MM_IDS_HELPER_MAX_WARPS, n_iter/8));

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, nwarps, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store) + 2*MM_IDS_HELPER_MAX_WARPS*sizeof(int);
    GGML_ASSERT(nbytes_shared <= smpbo);
    if (vec4) {
        if constexpr (vec4_ok) {
            CUDA_SET_SHARED_MEMORY_LIMIT((mm_ids_helper<n_expert_used_template, true>), smpbo);
            mm_ids_helper<n_expert_used_template, true><<<num_blocks, block_size, nbytes_shared, stream>>>
                (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
        }
    } else {
        CUDA_SET_SHARED_MEMORY_LIMIT((mm_ids_helper<n_expert_used_template, false>), smpbo);
        mm_ids_helper<n_expert_used_template, false><<<num_blocks, block_size, nbytes_shared, stream>>>
            (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
    }
}

#define MMQ_IDS_TILE_MAP_BLOCK 256

// One block scans the per-expert row counts into a dense list of the (expert, tile column) pairs
// that hold work. Entries past the end get jt_empty, which is beyond any expert's column count, so
// the callers' existing empty-tile test skips them without a special case.
static __global__ void mmq_ids_tile_map(
        const int32_t * __restrict__ expert_bounds, int2 * __restrict__ tile_map,
        const int n_experts, const int J, const int n_tiles_max, const int jt_empty) {
    __shared__ int scan[MMQ_IDS_TILE_MAP_BLOCK];

    int base = 0; // tiles written by previous chunks

    for (int e0 = 0; e0 < n_experts; e0 += MMQ_IDS_TILE_MAP_BLOCK) {
        const int e  = e0 + threadIdx.x;
        const int nt = e < n_experts ? (expert_bounds[e + 1] - expert_bounds[e] + J - 1) / J : 0;

        scan[threadIdx.x] = nt;
        __syncthreads();
        for (int offset = 1; offset < MMQ_IDS_TILE_MAP_BLOCK; offset *= 2) {
            const int add = threadIdx.x >= offset ? scan[threadIdx.x - offset] : 0;
            __syncthreads();
            scan[threadIdx.x] += add;
            __syncthreads();
        }

        for (int j = 0, t = base + scan[threadIdx.x] - nt; j < nt; ++j, ++t) {
            if (t < n_tiles_max) {
                tile_map[t] = make_int2(e, j);
            }
        }

        base += scan[MMQ_IDS_TILE_MAP_BLOCK - 1];
        __syncthreads();
    }

    for (int t = base + threadIdx.x; t < n_tiles_max; t += MMQ_IDS_TILE_MAP_BLOCK) {
        tile_map[t] = make_int2(0, jt_empty);
    }
}

void ggml_cuda_launch_mmq_ids_tile_map(
        const int32_t * expert_bounds, int2 * tile_map,
        const int n_experts, const int J, const int n_tiles_max, const int jt_empty, cudaStream_t stream) {
    mmq_ids_tile_map<<<1, MMQ_IDS_TILE_MAP_BLOCK, 0, stream>>>
        (expert_bounds, tile_map, n_experts, J, n_tiles_max, jt_empty);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
