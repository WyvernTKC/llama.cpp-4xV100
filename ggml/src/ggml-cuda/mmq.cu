#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// overrides the src1 precision requested by the graph, "auto" keeps the requested one
static ggml_prec ggml_cuda_mmq_get_prec_env() {
    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
    if (env_c == nullptr) {
        return GGML_PREC_UNDEFINED;
    }
    std::string env_cpp = env_c;
    for (char & c : env_cpp) {
        c = std::tolower(c);
    }
    if (env_cpp == "q4") {
        return GGML_PREC_Q4;
    }
    if (env_cpp == "q8") {
        return GGML_PREC_Q8;
    }
    if (env_cpp != "auto") {
        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
    }
    return GGML_PREC_UNDEFINED;
}

// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
// default to the native W4A4 instructions on Blackwell.
static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_tensor * src0, const ggml_tensor * dst, const int cc) {
    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();

    ggml_prec prec = prec_env;
    if (prec == GGML_PREC_UNDEFINED) {
        prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    }

    // Q4 only for the FP4 types on Blackwell
    GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
    const bool can_use_q4 = (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
    if (prec == GGML_PREC_Q8 || !can_use_q4) {
        return GGML_PREC_Q8;
    }
    return GGML_PREC_Q4;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ggml_cuda_mmq_needs_fallback(ne01);

    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0, dst, cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    int J_best        = 0;
    int nthreads_best = 0;
    {
        int64_t ncols_opt = ne11;
        if (ids) {
            const int64_t n_expert_used = ids->ne[0];
            ncols_opt = ne12;

            // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
            // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
            if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
                ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
            }
        }

        int ntiles_J_best = INT_MAX;

        for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J, fallback, cc, prec_src1);
            if (config.type == GGML_TYPE_COUNT) {
                continue;
            }

            if (mmq_get_nbytes_shared(config, cc) > smpbo) {
                continue;
            }

            const int ntiles_x = (ncols_opt + config.J - 1) / config.J;

            if (ntiles_x < ntiles_J_best) {
                J_best = J;
                nthreads_best = config.nthreads;
                ntiles_J_best = ntiles_x;
            }
        }
    }
    GGML_ASSERT(J_best > 0);

    // ncols_max has to survive one expert taking every token, so for a MoE launch the J above is the
    // worst case. An expert typically holds only ncols_dst/n_expert columns, and a tile computes all
    // J of them regardless - only the write back is masked - so the surplus is wasted dp4a work: at
    // batch 512 with 256 experts, 89% of it. Derive J from the expected per-expert column count
    // instead. Every J from 8 to 128 is a valid tiling and the host-side tile bound
    // sum(ceil(rows/J)) <= ncols_dst/J + n_expert still holds, so there is still no readback and the
    // dense tile map and CUDA graphs survive. Never raise J above the worst-case value: a batch
    // smaller than one tile must keep its own smaller J.
    //
    // A dense launch has no expert dispersion, so its J above is already the exact column count and
    // there is no surplus to reclaim - shrinking it only multiplies x-tile re-reads, which measures
    // up to 2.3x slower. Dense is therefore left alone.
    //
    // Only 16 and 32 are candidates. A tile width that is not a power of two divides badly into the
    // threads - at I=128 and 256 threads J=24 leaves 12 outputs per thread instead of 16 - and
    // measures worse than both neighbours: on nemotron-h-moe, whose 24 columns per expert make J=24
    // the natural choice, J=24 is 1.26x over the worst-case J where J=32 is 1.57x.
    //
    // Measured on 4x V100 (sm_70, dp4a) against the worst-case J: 1.46-1.57x prefill across five MoE
    // models under -sm layer (Q8_0/Q4_K/Q6_K, 128-512 experts) and 1.30-1.40x under -sm tensor,
    // where row splitting has already cut each device's MMQ share. Below 16 the x-tile re-reads
    // start to bite. Not measured on the Turing+ MMA path, which shares this function but tiles
    // differently. GGML_CUDA_MMQ_J_AUTO=0 restores the worst-case J.
    static const bool J_auto = getenv("GGML_CUDA_MMQ_J_AUTO") ? atoi(getenv("GGML_CUDA_MMQ_J_AUTO")) != 0 : true;
    if (J_auto && ids && ne02 > 0) {
        const int64_t cols_per_expert = ne12*ids->ne[0] / ne02;

        int J_moe = cols_per_expert <= 16 ? 16 : 32;
        if (J_moe > J_best) J_moe = J_best;

        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J_moe, fallback, cc, prec_src1);
        if (config.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(config, cc) <= smpbo) {
            J_best        = J_moe;
            nthreads_best = config.nthreads;
        }
    }

    // GGML_CUDA_MMQ_J pins J for MoE launches and GGML_CUDA_MMQ_J_DENSE for dense ones, so the two
    // regimes can be swept apart. Both override the choices above.
    static const int J_env       = getenv("GGML_CUDA_MMQ_J")       ? atoi(getenv("GGML_CUDA_MMQ_J"))       : 0;
    static const int J_env_dense = getenv("GGML_CUDA_MMQ_J_DENSE") ? atoi(getenv("GGML_CUDA_MMQ_J_DENSE")) : 0;

    const int J_pin = ids ? J_env : J_env_dense;
    if (J_pin >= 8 && J_pin <= 128 && J_pin % 8 == 0) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J_pin, fallback, cc, prec_src1);
        if (config.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(config, cc) <= smpbo) {
            J_best        = J_pin;
            nthreads_best = config.nthreads;
        }
    }

    // A tile of size J can read in at most J - 1 extra columns.
    // For simplicity, round up the padding of a full tile to a multiple of the number of bytes that nthreads can load in parallel.
    const size_t src1_load_chunk_size = nthreads_best * sizeof(int);
    const size_t src1_q8_1_padding = ((J_best * sizeof(block_q8_1_mmq) + src1_load_chunk_size - 1) / src1_load_chunk_size)
        * src1_load_chunk_size;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, J_best};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block + src1_q8_1_padding;
    const bool with_scale = src0->type == GGML_TYPE_NVFP4 && use_native_fp4;

    // the gate of an expert FFN may have made the expert lists and the quantized activations of its up projection. Only
    //   the memory of graph tensors is followed by the invalidation, as for the cuBLAS input in ggml-cuda.cu
    const auto in_buffer = [](const ggml_tensor * t) {
        const char * base = t->buffer ? (const char *) ggml_backend_buffer_get_base(t->buffer) : nullptr;
        return base != nullptr && (const char *) t->data >= base &&
            (const char *) t->data + ggml_nbytes(t) <= base + ggml_backend_buffer_get_size(t->buffer);
    };
    auto & cached = ctx.mmq_ids;
    const ggml_tensor * src1_owner = src1->view_src ? src1->view_src : src1;
    const bool keep = !ggml_cuda_info().devices[ctx.device].vmm && in_buffer(src1) && in_buffer(ids);
    const bool reuse = keep && cached.q8_1 && cached.src1_owner == src1_owner && cached.src1_data == src1->data &&
        std::equal(std::begin(cached.src1_ne), std::end(cached.src1_ne), src1->ne) &&
        std::equal(std::begin(cached.src1_nb), std::end(cached.src1_nb), src1->nb) &&
        cached.ids_data == ids->data && std::equal(std::begin(cached.ids_ne), std::end(cached.ids_ne), ids->ne) &&
        std::equal(std::begin(cached.ids_nb), std::end(cached.ids_nb), ids->nb) && cached.type == src0->type &&
        cached.prec == (int) prec_src1 && cached.n_experts == ne02 && cached.nbytes == nbytes_src1_q8_1 &&
        cached.stream_no == ctx.curr_stream_no;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool());
    ggml_cuda_pool_alloc<char>    src1_q8_1(ctx.pool());
    ggml_cuda_pool_alloc<float>   src1_scale(ctx.pool());

    int32_t * ids_src1_p      = nullptr;
    int32_t * ids_dst_p       = nullptr;
    int32_t * expert_bounds_p = nullptr;
    char    * src1_q8_1_p     = nullptr;
    float   * src1_scale_p    = nullptr;

    if (reuse) {
        ids_dst_p       = (int32_t *) cached.ids_dst->get();
        expert_bounds_p = (int32_t *) cached.expert_bounds->get();
        src1_q8_1_p     = cached.q8_1->get();
        src1_scale_p    = with_scale ? (float *) cached.scale->get() : nullptr;
    } else if (keep) {
        cached.reset();
        cached.ids_src1      = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), ne_get_rows*sizeof(int32_t));
        // padded for the unconditional reads of a tile, for the largest J: a later launch with another J may reuse it
        cached.ids_dst       = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), (ne_get_rows + 127)*sizeof(int32_t));
        cached.expert_bounds = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), (ne02 + 1)*sizeof(int32_t));
        cached.q8_1          = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), nbytes_src1_q8_1);
        if (with_scale) {
            cached.scale = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), ne12*n_expert_used*sizeof(float));
        }
        ids_src1_p      = (int32_t *) cached.ids_src1->get();
        ids_dst_p       = (int32_t *) cached.ids_dst->get();
        expert_bounds_p = (int32_t *) cached.expert_bounds->get();
        src1_q8_1_p     = cached.q8_1->get();
        src1_scale_p    = with_scale ? (float *) cached.scale->get() : nullptr;

        cached.src1_owner  = src1_owner;
        cached.src1_data   = src1->data;
        cached.src1_nbytes = ggml_nbytes(src1);
        std::copy_n(src1->ne, GGML_MAX_DIMS, cached.src1_ne);
        std::copy_n(src1->nb, GGML_MAX_DIMS, cached.src1_nb);
        cached.ids_data    = ids->data;
        cached.ids_nbytes  = ggml_nbytes(ids);
        std::copy_n(ids->ne, GGML_MAX_DIMS, cached.ids_ne);
        std::copy_n(ids->nb, GGML_MAX_DIMS, cached.ids_nb);
        cached.type        = src0->type;
        cached.prec        = (int) prec_src1;
        cached.n_experts   = ne02;
        cached.nbytes      = nbytes_src1_q8_1;
        cached.stream_no   = ctx.curr_stream_no;
    } else {
        ids_src1_p      = ids_src1.alloc(ne_get_rows);
        ids_dst_p       = ids_dst.alloc(ne_get_rows + J_best-1); // Needs to be padded for unconditional memory access.
        expert_bounds_p = expert_bounds.alloc(ne02 + 1);
        src1_q8_1_p     = src1_q8_1.alloc(nbytes_src1_q8_1);
        src1_scale_p    = with_scale ? src1_scale.alloc(ne12*n_expert_used) : nullptr;
    }

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    if (!reuse) {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1_p, ids_dst_p, expert_bounds_p,
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    if (!reuse) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1_p, src1_q8_1_p, src1_scale_p, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1_p, src1_q8_1_p, src1_scale_p, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1_p, src1_q8_1_p, src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1_p, src1_q8_1_p, src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1_p, ids_dst_p, expert_bounds_p, dst_d,
        src1_scale_p,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, J_best};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        // High expert counts are almost always better on MMQ due to
        //     the synchronization overhead in the cuBLAS path: that fallback loops over every
        //     expert, synchronizes the stream twice per call and disables CUDA graph capture.
        if (n_experts > 64) {
            return true;
        }
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    // MUSA: the MMQ kernels compute wrong values on PH1 (MTT S5000).
    if (cc == GGML_CUDA_CC_PH1) {
        return false;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
