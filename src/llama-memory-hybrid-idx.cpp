#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <type_traits>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                     bool   no_alloc,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        no_alloc,
        type_k, type_v, v_trans, kv_size, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&]() -> llama_kv_cache * {
        // a model whose layers all run dense attention leaves its indexer unused, and keeps no cache for it
        bool any = false;
        for (uint32_t il = 0; il < model.hparams.n_layer_all && !any; ++il) {
            any = filter_idx(il);
        }
        if (!any) {
            return nullptr;
        }

        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // a k-pool indexer caches its per-token rows and the pooled key side by side
        // (glm5-next: key | gate | pooled, qwen4exp: key | pooled)
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size * (model.hparams.indexer_kpool > 0 ? model.hparams.indexer_kpool_row : 1);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, no_alloc, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()),
    has_indexer(filter_idx != nullptr),
    qsa_pool_ctxs_bufs(no_alloc),
    kpool_ramp_ctxs_bufs(no_alloc) {
    // see get_kpool_ramp_cells, LLAMA_KPOOL_IDENT=0 keeps the per-pool host inputs
    const char * env_ident = getenv("LLAMA_KPOOL_IDENT");
    if (mem_idx && model.arch == LLM_ARCH_GLM5_NEXT && model.hparams.indexer_kpool > 1 && mem_idx->get_n_stream() == 1 &&
            !(env_ident && atoi(env_ident) == 0)) {
        const uint32_t kpool   = model.hparams.indexer_kpool;
        const uint32_t kv_size = mem_idx->get_size();
        // kpool_pad of the most pools the cache can hold
        const uint32_t n_pool  = kv_size/kpool + 64;

        const uint32_t il0 = mem_idx->get_layer_ids()[0];
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(mem_idx->get_k_storage(il0)->buffer);

        ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for the k-pool ramps");
        }

        kpool_ramp_cells = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, (int64_t) kpool*n_pool);
        kpool_ramp_reps  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_pool);
        ggml_format_name(kpool_ramp_cells, "cache_idx_r_l%u", il0);
        ggml_format_name(kpool_ramp_reps,  "cache_idx_c_l%u", il0);

        ggml_backend_buffer_t buf = kpool_ramp_ctxs_bufs.emplace(std::move(ctx), buft, "the k-pool ramps");

        if (!no_alloc) {
            std::vector<int32_t> v((size_t) kpool*n_pool);
            for (size_t i = 0; i < v.size(); ++i) {
                v[i] = (int32_t) i;
            }
            ggml_backend_tensor_set(kpool_ramp_cells, v.data(), 0, ggml_nbytes(kpool_ramp_cells));

            // the padded pools past the cache read its last cell, finite like every pooled slot
            v.resize(n_pool);
            for (uint32_t b = 0; b < n_pool; ++b) {
                v[b] = (int32_t) std::min<uint64_t>((uint64_t) b*kpool + kpool - 1, kv_size - 1);
            }
            ggml_backend_tensor_set(kpool_ramp_reps, v.data(), 0, ggml_nbytes(kpool_ramp_reps));
        }

        LLAMA_LOG_INFO("%s: %10s k-pool ramps buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf),
                ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
    }

    // LLAMA_QSA_POOL_CACHE=0 pools every block of every ubatch
    const char * env = getenv("LLAMA_QSA_POOL_CACHE");
    if (!mem_idx || model.arch != LLM_ARCH_QWEN4EXP || model.hparams.indexer_kpool > 0 || mem_idx->get_n_stream() != 1 ||
            (env && atoi(env) == 0)) {
        return;
    }

    const auto layer_ids = mem_idx->get_layer_ids();
    for (const uint32_t il : layer_ids) {
        const uint32_t r = model.hparams.dsv4_compress_ratios[il];
        if (r == 0 || (qsa_pool_ratio != 0 && r != qsa_pool_ratio)) {
            // one set of rows per ubatch is kept up to date, so the layers must share one block grouping
            return;
        }
        qsa_pool_ratio = r;
    }
    if (qsa_pool_ratio == 0) {
        return;
    }

    const uint32_t r     = qsa_pool_ratio;
    const uint32_t spare = (QSA_POOL_MAX_TOKENS - 1 + r - 1)/r + 2;
    qsa_pool_rows = (mem_idx->get_size() + r - 1)/r;

    std::map<ggml_backend_buffer_type_t, std::vector<uint32_t>> by_buft;
    for (const uint32_t il : layer_ids) {
        by_buft[ggml_backend_buffer_get_type(mem_idx->get_k_storage(il)->buffer)].push_back(il);
    }

    for (const auto & [buft, ils] : by_buft) {
        ggml_init_params params = {
            /*.mem_size   =*/ (ils.size() + 1)*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for the qsa pool");
        }

        ggml_tensor * ramp = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, qsa_pool_rows);
        ggml_format_name(ramp, "cache_idx_r_l%d", ils[0]);

        for (const uint32_t il : ils) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, model.hparams.indexer_head_size, qsa_pool_rows + spare);
            ggml_format_name(t, "cache_idx_p_l%d", il);
            qsa_pool[il]      = t;
            qsa_pool_ramp[il] = ramp;
        }

        // with no_alloc a dummy buffer is attached; emplace clears the buffer: stale rows are scored and dropped by
        //   the bias, so they must not hold a nan
        ggml_backend_buffer_t buf = qsa_pool_ctxs_bufs.emplace(std::move(ctx), buft, "the qsa pool");

        if (!no_alloc) {
            std::vector<int32_t> v(qsa_pool_rows);
            for (uint32_t b = 0; b < qsa_pool_rows; ++b) {
                v[b] = (int32_t) (b*r);
            }
            ggml_backend_tensor_set(ramp, v.data(), 0, ggml_nbytes(ramp));
        }

        LLAMA_LOG_INFO("%s: %10s QSA pool buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf),
                ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
        mem_idx_stale_set(-1, 0);
    }
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void llama_memory_hybrid_idx::mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0) {
    p0 = std::max<llama_pos>(p0, 0);

    if (seq_id < 0 || seq_id == qsa_pool_seq) {
        qsa_pool_valid = std::min(qsa_pool_valid, p0);
        qsa_pool_ident = false;
    }

    if (seq_id < 0) {
        for (auto & p : mem_idx_stale) {
            p = std::min(p, p0);
        }

        return;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id] = std::min(mem_idx_stale[seq_id], p0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
llama_pos llama_memory_hybrid_idx::mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || p0 <= mem_idx->seq_pos_min(seq_id)) {
        return 0;
    }

    return p0;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        // dropping the tail of a sequence keeps every other cell where it was
        const bool ident = qsa_pool_ident && seq_id == qsa_pool_seq && (p1 < 0 || p1 > mem_idx->seq_pos_max(seq_id));
        mem_idx->seq_rm(seq_id, p0, p1);
        mem_idx_stale_set(seq_id, stale);
        qsa_pool_ident = qsa_pool_ident || ident;
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // only whole sequences are copied: the recurrent state ignores the range, and a shared cell holds a single pool grouping
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "partial seq_cp is not supported");

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        // a whole sequence copy gives the destination the source's pools, rep rows included: the source keeps its
        // pooled keys, the destination rebuilds its layout and re-pools into the same rows
        mem_idx_stale_set(seq_id_dst, 0);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        // every other sequence loses its cells, so their layouts must rebuild
        mem_idx_stale_set(-1, 0);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // a negative shift moves the cells below p0, so they regroup as well
        const llama_pos stale = mem_idx_stale_pos(seq_id, shift < 0 ? p0 + shift : p0);
        mem_idx->seq_add(seq_id, p0, p1, shift);
        mem_idx_stale_set(seq_id, stale);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
        mem_idx_stale_set(seq_id, 0);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    // with no_alloc, the simulated sizes
    for (const auto & [buft, size] : qsa_pool_ctxs_bufs.memory_breakdown()) {
        mb[buft] += size;
    }
    for (const auto & [buft, size] : kpool_ramp_ctxs_bufs.memory_breakdown()) {
        mb[buft] += size;
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    if (mem_idx) {
        mem_idx->state_write(io, seq_id, flags);
    }
}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        // the indexer cache has the SWA type of the attention cache, so both read the same partial state
        get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if (mem_idx) {
            mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            // a restore that rewrites cells does it behind the pool layout's back
            if (!sinfos_attn.empty()) {
                mem_idx_stale_set(seq_id, 0);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->state_clear(seq_id);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->state_clear(seq_id);
        mem_idx_stale_set(seq_id, 0);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

// the indexer cache must match the attention cache cell for cell and the QSA inputs are sized by its n_kv: with an
//   indexer, no graph is built for more cells than its ubatch needs (the attention cache alone did, and qwen4exp
//   asserted at the first graph of a few tokens)
void llama_memory_hybrid_idx::set_graph_reshape(bool enable) {
    // a model whose indexer is unused keeps the graphs it had with the cache
    llama_memory_hybrid::set_graph_reshape(enable && mem_idx == nullptr && !has_indexer);
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * bias_blk,
        ggml_tensor * bias_q,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        bool causal_attn,
        bool pool_write) {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;

    // a device builds the bias from the first index of each block and the index of each query
    const bool bias_dev = bias_blk != nullptr;
    GGML_ASSERT(!bias_dev || (blk_bias && causal_attn));

    float   * dst_bias     = bias_dev ? nullptr : (float *) bias->data;
    int32_t * dst_bias_blk = bias_dev ? (int32_t *) bias_blk->data : nullptr;
    int32_t * dst_bias_q   = bias_dev ? (int32_t *) bias_q->data   : nullptr;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    // whether the single stream of this ubatch is in the identity layout, see qsa_pool_n_new
    bool         pool_ok  = false;
    llama_seq_id pool_seq = -1;

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        // the stream holds one sequence here, so a block is seen by all of its tokens or by none
        if (bias_dev) {
            const llama_seq_id seq_id = ubatch->seq_id[s*n_tps][0];
            int32_t * cur = dst_bias_blk + s*n_blocks;
            for (int64_t b = 0; b < n_blocks; ++b) {
                cur[b] = b < n_bid && cells.seq_has((uint32_t) bid_cell[b], seq_id) ? bid_idx[b] : -1;
            }
            if (have_dead) {
                cur[dead_bid] = -2;
            }
        }

        // the identity layout (cell j holds position j, for every position of the one sequence) lets later short
        //   ubatches skip this scan, see qsa_pool_n_new
        bool    ident  = n_ns == 1 && one_seq && !ranked && !dup;
        int64_t n_used = 0;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];

            if (!cells.is_empty(j)) {
                ident = ident && cells.pos_get(j) == j;
                n_used++;
            }
        }

        pool_ok  = ident && n_used == cells.seq_pos_max(seq_of_stream) + 1;
        pool_seq = seq_of_stream;

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            if (bias_dev) {
                dst_bias_q[i] = (int32_t) q;
                continue;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and, when causal, future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = (causal_attn && bid_idx[b] >= tail_start) ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (!causal_attn) {
                        // every visible block competes on score and the unpooled cells are always selected
                        v = blk_of[j] < 0 ? 1e9f : 0.0f;
                    } else if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }

    if (!qsa_pool.empty()) {
        if (pool_write && pool_ok) {
            qsa_pool_seq   = pool_seq;
            qsa_pool_valid = get_mem_idx()->get_cells(pool_seq).seq_pos_max(pool_seq) + 1;
            qsa_pool_ident = true;
        } else {
            qsa_pool_seq   = -1;
            qsa_pool_ident = false;
        }
    }
}

bool llama_memory_hybrid_idx::qsa_pool_on() const {
    return !qsa_pool.empty();
}

ggml_tensor * llama_memory_hybrid_idx::get_qsa_pool(int32_t il) const {
    const auto it = qsa_pool.find(il);

    return it == qsa_pool.end() ? nullptr : it->second;
}

ggml_tensor * llama_memory_hybrid_idx::get_qsa_pool_ramp(int32_t il) const {
    const auto it = qsa_pool_ramp.find(il);

    return it == qsa_pool_ramp.end() ? nullptr : it->second;
}

// A short ubatch may skip the full lists when the stream holds just the sequence the pool rows were written for, in the
//   identity layout, and the ubatch keeps it: each token lands in the cell of its position. The rows of the blocks
//   from the first stale or new position on are pooled again, with the spare block.
uint32_t llama_memory_hybrid_idx::qsa_pool_n_new(const llama_ubatch & ubatch, uint32_t ratio) const {
    if (!qsa_pool_on() || ratio != qsa_pool_ratio || qsa_pool_seq < 0 || !qsa_pool_ident ||
            ubatch.n_tokens == 0 || ubatch.n_tokens > QSA_POOL_MAX_TOKENS) {
        return 0;
    }

    const llama_seq_id s = qsa_pool_seq;

    const auto & cells = mem_idx->get_cells(s);
    for (llama_seq_id sq = 0; sq < LLAMA_MAX_SEQ; ++sq) {
        if (sq != s && cells.seq_pos_min(sq) >= 0) {
            return 0;
        }
    }

    // cells 0..pos_max all hold a position of the sequence, so the one in cell p is p
    const llama_pos p_max = cells.seq_pos_max(s);
    if (cells.seq_pos_min(s) != 0 || (int64_t) cells.seq_pos_get(s).size() != (int64_t) p_max + 1) {
        return 0;
    }

    llama_pos p_lo = qsa_pool_valid;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i][0] != s || p < 0 || p >= (llama_pos) cells.size() ||
                cells.is_empty(p) || cells.pos_get(p) != p || !cells.seq_has(p, s)) {
            return 0;
        }
        p_lo = std::min(p_lo, p);
    }

    const int64_t r = ratio;

    // full blocks from p_lo on, at most, plus the spare block; the list length depends on the ubatch size only
    const int64_t n_upd = (p_max + 1)/r - p_lo/r + 1;
    const int64_t n_new = (ubatch.n_tokens - 1 + r - 1)/r + 2;

    return n_upd <= n_new ? (uint32_t) n_new : 0;
}

void llama_memory_hybrid_idx::set_input_qsa_pool(
        ggml_tensor * pool_cells,
        ggml_tensor * pool_pos,
        ggml_tensor * pool_rows,
        ggml_tensor * bias_q,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        int64_t n_kv) {
    const llama_seq_id s = qsa_pool_seq;
    GGML_ASSERT(s >= 0 && qsa_pool_ident && ratio == qsa_pool_ratio);

    const int64_t r        = ratio;
    const int64_t n_new    = pool_rows->ne[0];
    const int64_t n_blocks = n_kv/r;

    llama_pos p_lo = qsa_pool_valid;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        p_lo = std::min(p_lo, ubatch->pos[i]);
        ((int32_t *) bias_q->data)[i] = ubatch->pos[i];
    }

    const llama_pos p_max = get_mem_idx()->get_cells(s).seq_pos_max(s);
    const int64_t   n_bid = (p_max + 1)/r;

    int32_t * pc = (int32_t *) pool_cells->data;
    int32_t * pp = (int32_t *) pool_pos->data;
    int64_t * pr = (int64_t *) pool_rows->data;

    std::fill(pc, pc + r*n_new, 0);
    std::fill(pp, pp + 4*n_new, 0);

    // full blocks from p_lo on, their cells are their positions
    int64_t k = 0;
    for (int64_t b = p_lo/r; b < n_bid; ++b, ++k) {
        GGML_ASSERT(k < n_new);
        for (int64_t i = 0; i < r; ++i) {
            pc[k*r + i] = (int32_t) (b*r + i);
        }
        for (int64_t sec = 0; sec < 4; ++sec) {
            pp[sec*n_new + k] = (int32_t) (b*r);
        }
        pr[k] = b;
    }

    // the spare block pools cell 0 at position 0, as the full lists leave it
    if (n_bid < n_blocks) {
        GGML_ASSERT(k < n_new);
        pr[k++] = n_bid;
    }

    // padding writes rows past the blocks, one each
    for (int64_t i = 0; k < n_new; ++i) {
        pr[k++] = qsa_pool_rows + i;
    }

    qsa_pool_valid = p_max + 1;
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// Which cells of a sequence make up which pool, for the whole cache.
struct llama_memory_hybrid_idx::kpool_layout {
    struct seq {
        llama_pos pos_min = 0;
        uint32_t  strm    = 0; // Stream holding this sequence's cells
        std::vector<std::pair<llama_pos, uint32_t>> cells; // Position and stream local cell pairs, sorted by position.
        std::vector<uint32_t> pools;

        // Where the pool scan stopped, so an append resumes instead of starting over.
        size_t j_next = 0;

        // Leading cells that hold their own position (cell j, position j), see kpool_ident
        size_t n_ident = 0;
    };

    std::array<seq, LLAMA_MAX_SEQ> seqs;

    uint32_t n_pool_real = 0;
};

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct llama_memory_hybrid_idx_context::kpool_state {
    std::vector<uint32_t> is_new;
    std::vector<uint32_t> rep_gen; // per global cell, the generation that last marked a pool with that rep
    uint32_t generation = 0;

    uint32_t n_pool_real = 0;
    uint32_t n_new       = 0;
    uint32_t n_new_g     = 1; // graph size of the new pool list, stable across decode steps
    uint32_t new_lo      = UINT32_MAX; // first pool marked new, the identity input scans from it
};

namespace {

// The last padded pool is always unused.
uint32_t kpool_pad(uint32_t n_pool) {
    return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u));
}

// Rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent.
// In order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks.
int64_t kpool_rank(const std::vector<std::pair<llama_pos, uint32_t>> & cells, llama_pos pos, uint32_t cell) {
    auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(pos, cell));
    return it != cells.end() && it->second == cell && it->first == pos ? it - cells.begin() : -1;
}

}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() = default;

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_get() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return *kpool_lay;
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them, and mem_idx_stale tells us it happened.
const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_update() {
    GGML_ASSERT(mem_idx != nullptr);

    if (!kpool_lay) {
        kpool_lay = std::make_unique<kpool_layout>();
    }

    auto & lay = *kpool_lay;

    const uint32_t kpool       = get_kpool();
    const uint32_t n_stream_kv = mem_idx->get_n_stream();
    const bool     unified     = n_stream_kv == 1;

    lay.n_pool_real = 0;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        auto & sq = lay.seqs[s];

        // a non unified cache gives each sequence its own stream, with stream local cell indices
        if (!unified && s >= (llama_seq_id) n_stream_kv) {
            sq = kpool_layout::seq();
            continue;
        }

        const auto & cells = mem_idx->get_cells(unified ? 0 : s);
        const auto & sp    = cells.seq_pos_get(s);

        sq.strm = unified ? 0 : mem_idx->get_stream(s);

        if (mem_idx_stale[s] == POS_CLEAN && !sq.cells.empty() && !sp.empty() &&
                sq.pos_min == sp.begin()->first) {
            for (auto it = sp.upper_bound(sq.cells.back()); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
        }

        // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
        // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
        if (sq.cells.size() != sp.size() || mem_idx_stale[s] != POS_CLEAN) {
            sq.cells.assign(sp.begin(), sp.end());
            sq.pools.clear();
            sq.j_next  = 0;
            sq.n_ident = 0;
            sq.pos_min = sp.empty() ? 0 : sp.begin()->first;
        }

        // an append only extends the prefix, so the scan resumes where it stopped
        while (sq.n_ident < sq.cells.size() && sq.cells[sq.n_ident].first == (llama_pos) sq.n_ident &&
                sq.cells[sq.n_ident].second == sq.n_ident) {
            ++sq.n_ident;
        }

        // Pools start at the first valid token
        size_t j = sq.j_next;
        if (hparams_idx.indexer_kpool_by_order) {
            // consecutive cells in sequence order, whatever their positions
            for (; j + kpool <= sq.cells.size(); j += kpool) {
                sq.pools.push_back((uint32_t) j);
            }
        } else {
            while (j + kpool <= sq.cells.size()) {
                const llama_pos p0 = sq.cells[j].first;
                if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                    ++j;
                    continue;
                }
                bool ok = true;
                for (uint32_t k = 1; k < kpool; ++k) {
                    if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    sq.pools.push_back((uint32_t) j);
                    j += kpool;
                } else {
                    ++j;
                }
            }
        }
        sq.j_next = j;

        lay.n_pool_real += (uint32_t) sq.pools.size();
    }

    return lay;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {
    if (kpool_track()) {
        mem->kpool_layout_update();
        auto st = kpool_build_sizes();
        const auto * idx = mem->get_mem_idx();
        const uint64_t n_pool_max = uint64_t(idx->get_size() / mem->get_kpool()) * idx->get_n_seq_max();
        GGML_ASSERT(n_pool_max <= UINT32_MAX - 64);
        st.n_pool_real = std::max(st.n_pool_real, uint32_t(n_pool_max));
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        kpool_st = std::make_unique<kpool_state>(std::move(st));
        i_kpool  = 0;
    }
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    sinfos_kpool(mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && mem->get_kpool_by_order() ? sinfos_idx : slot_info_vec_t()),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {
    // Sequence edits force the touched positions to re-pool.
    mem_idx_stale_batch = mem->mem_idx_stale_get();
}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    // Clear only after a successful ubatch.
    if (i_cur == 0 && mem != nullptr) {
        mem->mem_idx_stale_clear();
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // Extend the pool layout with this ubatch's cells, then pick what it must re-pool.
    if (res && kpool_track()) {
        mem->kpool_layout_update();
        if (!kpool_st) {
            kpool_st = std::make_unique<kpool_state>();
        }
        kpool_build_state(get_ubatch());
        i_kpool  = i_cur;
    }

    return res;
}

bool llama_memory_hybrid_idx_context::kpool_track() const {
    // Derived from mem instead of being cached.
    return mem != nullptr && mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && !ns_ubatch.empty();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * bias_blk,
        ggml_tensor * bias_q,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        bool causal_attn,
        bool pool_write) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, bias_blk, bias_q, ubatch, ratio, blk_bias, causal_attn, pool_write);
}

bool llama_memory_hybrid_idx_context::qsa_pool_on() const {
    return mem != nullptr && mem->qsa_pool_on();
}

ggml_tensor * llama_memory_hybrid_idx_context::get_qsa_pool(int32_t il) const {
    return mem != nullptr ? mem->get_qsa_pool(il) : nullptr;
}

ggml_tensor * llama_memory_hybrid_idx_context::get_qsa_pool_ramp(int32_t il) const {
    return mem != nullptr ? mem->get_qsa_pool_ramp(il) : nullptr;
}

uint32_t llama_memory_hybrid_idx_context::qsa_pool_n_new(const llama_ubatch & ubatch, uint32_t ratio) const {
    return mem != nullptr ? mem->qsa_pool_n_new(ubatch, ratio) : 0;
}

void llama_memory_hybrid_idx_context::set_input_qsa_pool(ggml_tensor * pool_cells, ggml_tensor * pool_pos, ggml_tensor * pool_rows,
        ggml_tensor * bias_q, const llama_ubatch * ubatch, uint32_t ratio) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa_pool(pool_cells, pool_pos, pool_rows, bias_q, ubatch, ratio, get_idx()->get_n_kv());
}

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd) : ctx(ctx) {
    // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
    const int64_t n_tok = k->ne[0] - n_embd;
    GGML_ASSERT(n_tok > 0 && n_tok % n_embd == 0);

    const int64_t n_cells = k->ne[1]*k->ne[2];

    // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
    // https://github.com/ggml-org/llama.cpp/pull/27773#discussion_r4130905603
    key_gate = ggml_view_2d(ctx, k, n_tok,  n_cells, k->nb[1], 0);
    pooled   = ggml_view_2d(ctx, k, n_embd, n_cells, k->nb[1], ggml_row_size(k->type, n_tok));
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_key_gate(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, key_gate, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const {
    return ggml_set_rows(ctx, pooled, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_pooled(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, pooled, idxs);
}

llama_memory_hybrid_idx_context::kpool_access llama_memory_hybrid_idx_context::get_kpool_access(
        ggml_context * ctx, int32_t il, int64_t n_embd) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    return kpool_access(ctx, mem->get_mem_idx()->get_k_storage(il), n_embd);
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
llama_memory_hybrid_idx_context::kpool_state llama_memory_hybrid_idx_context::kpool_build_sizes() const {
    const auto & lay = mem->kpool_layout_get();

    kpool_state st;
    st.n_pool_real = lay.n_pool_real;

    return st;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch last wrote them.
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
void llama_memory_hybrid_idx_context::kpool_build_state(const llama_ubatch & ubatch) {
    const auto & lay = mem->kpool_layout_get();
    auto & st = *kpool_st;

    const auto *   idx     = mem->get_mem_idx();
    const uint32_t kv_size = idx->get_size();
    const uint32_t kpool   = mem->get_kpool();

    st.n_pool_real = lay.n_pool_real;
    st.n_new       = 0;
    st.new_lo      = UINT32_MAX;
    if (++st.generation == 0) {
        std::fill(st.is_new.begin(),  st.is_new.end(),  0);
        std::fill(st.rep_gen.begin(), st.rep_gen.end(), 0);
        st.generation = 1;
    }
    st.is_new.resize(lay.n_pool_real, 0);
    st.rep_gen.resize((size_t) kv_size*idx->get_n_stream(), 0);

    std::array<uint32_t, LLAMA_MAX_SEQ> pool_start;

    // a pool is marked once per rep: sequences sharing cells (a seq_cp, or tokens decoded for several sequences)
    // share their pools, whose single pooled row they all read through pool_cells, so the scatter rows stay unique
    auto mark = [&](llama_seq_id s, size_t k) {
        const auto & sq  = lay.seqs[s];
        const size_t rep = (size_t) sq.strm*kv_size + sq.cells[sq.pools[k] + kpool - 1].second;
        if (st.rep_gen[rep] != st.generation) {
            st.rep_gen[rep] = st.generation;
            st.is_new[pool_start[s] + k] = st.generation;
            st.new_lo = std::min(st.new_lo, pool_start[s] + (uint32_t) k);
            ++st.n_new;
        }
    };

    uint32_t ip = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        pool_start[s] = ip;
        ip += (uint32_t) sq.pools.size();

        // A sequence edit invalidates only pools ending after the edited position.
        const llama_pos stale_from = i_cur == 0 ?
            mem_idx_stale_batch[s] : llama_memory_hybrid_idx::POS_CLEAN;
        if (stale_from == llama_memory_hybrid_idx::POS_CLEAN) {
            continue;
        }

        auto first = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale_from,
                [&](uint32_t j, llama_pos p) { return sq.cells[j + kpool - 1].first < p; });
        for (auto it = first; it != sq.pools.end(); ++it) {
            mark(s, it - sq.pools.begin());
        }
    }
    GGML_ASSERT(ip == st.is_new.size());

    // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
    const bool by_order = mem->get_kpool_by_order();
    const auto *   sinfo = by_order ? &sinfos_kpool[i_cur] : nullptr;
    const uint32_t n_tps = by_order ? (uint32_t) sinfo->size() : 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            const auto & sq = lay.seqs[s];
            if (by_order) {
                const int64_t r = kpool_rank(sq.cells, p, sinfo->idxs[i / n_tps][i % n_tps]);
                GGML_ASSERT(r >= 0);
                if ((size_t) r / kpool < sq.pools.size()) {
                    mark(s, (size_t) r / kpool);
                }
                continue;
            }
            auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), p,
                    [&](llama_pos pos, uint32_t j) { return pos < sq.cells[j].first; });
            if (it == sq.pools.begin()) {
                continue;
            }
            --it;
            if (p <= sq.cells[*it + kpool - 1].first) {
                mark(s, it - sq.pools.begin());
            }
        }
    }

    // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
    // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
    const uint32_t n_pool_max = kv_size / kpool * idx->get_n_seq_max();
    const uint32_t bound = ubatch.n_tokens/kpool + ubatch.n_seqs_unq;
    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, kpool_pad(st.n_pool_real) - 1, n_pool_max})});
}

const llama_memory_hybrid_idx_context::kpool_state & llama_memory_hybrid_idx_context::kpool_cur() const {
    GGML_ASSERT(kpool_st != nullptr && i_kpool == i_cur && "k-pool state read before apply()");

    return *kpool_st;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool() const {
    return kpool_pad(kpool_cur().n_pool_real);
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_new() const {
    return kpool_cur().n_new_g;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_visible(const llama_ubatch & ubatch, uint32_t i) const {
    if (!kpool_track() || mem->get_kpool_by_order() || i >= ubatch.n_tokens) {
        return 0;
    }

    const uint32_t kpool = mem->get_kpool();
    const auto &   sq    = mem->kpool_layout_get().seqs[ubatch.seq_id[i][0]];

    // the same count as set_input_kpool's mask: pools are sorted by position, the visible ones are a prefix
    const auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), ubatch.pos[i],
            [&](llama_pos p, uint32_t j) { return p < sq.cells[j + kpool - 1].first; });
    return (uint32_t) (it - sq.pools.begin());
}

void llama_memory_hybrid_idx_context::set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * sel_mask, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_mask->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_idxs->buffer));

    const uint32_t kpool = mem->get_kpool();
    const uint32_t n_kv  = get_idx()->get_n_kv();

    const auto & st  = kpool_cur();
    const auto & lay = mem->kpool_layout_get();

    const uint32_t n_tokens = ubatch->n_tokens;
    const uint32_t n_pool   = (uint32_t) pool_cells->ne[0];
    const uint32_t n_new    = st.n_new;
    // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
    const uint32_t n_new_g  = st.n_new_g;

    const bool by_order = mem->get_kpool_by_order();

    GGML_ASSERT(n_pool == kpool_pad(st.n_pool_real));
    GGML_ASSERT(st.is_new.size() == st.n_pool_real);
    GGML_ASSERT(pool_mask->ne[0] == (int64_t) n_pool && pool_mask->ne[1] == (int64_t) n_tokens);
    // a wider tail row is the caller's padding, left alone here
    GGML_ASSERT(tail_idxs->ne[0] >= (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool);
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(ggml_nelements(new_pool_idxs) == (int64_t) kpool*n_new_g);
    // the graph always scatters the fresh pooled keys back into the cache, see build_qsa_sel
    GGML_ASSERT(new_pool_rep != nullptr && ggml_backend_buffer_is_host(new_pool_rep->buffer));
    GGML_ASSERT(new_pool_rep->ne[0] == (int64_t) n_new_g);
    if (new_pool_pos != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_pos->buffer));
        GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    }

    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const uint32_t n_stream_kv = mem->get_mem_idx()->get_n_stream();

    auto gcell = [&](const llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t cell) {
        return (int64_t) sq.strm*kv_size + cell;
    };

    // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            seq_in_ub[ubatch->seq_id[i][k]] = 1;
        }
    }

    // a cell of this ubatch, written before any read, so the padded pools read a finite K row
    int64_t dummy_cell = 0;
    {
        const llama_seq_id s = ubatch->seq_id[0][0];
        const auto & sq = lay.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch->pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch->pos[0]);
        dummy_cell = gcell(sq, it->second);
    }

    // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
    std::vector<int64_t> rank;
    if (by_order) {
        const auto &   sinfo = sinfos_kpool[i_cur];
        const uint32_t n_tps = (uint32_t) sinfo.size();

        rank.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            rank[i] = kpool_rank(lay.seqs[ubatch->seq_id[i][0]].cells, ubatch->pos[i], sinfo.idxs[i / n_tps][i % n_tps]);
            GGML_ASSERT(rank[i] >= 0);
        }
    }

    // padding and absent cells point at the n_kv sentinel row, one past the live cells
    const int32_t sentinel = (int32_t) n_kv;

    float *  gm    = nullptr;
    uint32_t n_sel = 0;
    uint32_t n_top = 0; // Pools per token in the selection.
    if (sel_mask != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask->buffer));
        GGML_ASSERT(sel_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(sel_mask->ne[3] == (int64_t) n_tokens && sel_mask->ne[1] == 1 && sel_mask->ne[2] == 1);
        n_sel = (uint32_t) sel_mask->ne[0];
        // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
        n_top = n_sel / kpool;
        GGML_ASSERT(n_sel % kpool == 0 || n_sel % kpool == kpool - 1);
        gm = (float *) sel_mask->data;
    }

    // pools are laid out per sequence
    std::vector<uint32_t>  seq_pool_start(LLAMA_MAX_SEQ, 0);
    std::vector<llama_pos> pool_end;
    pool_end.reserve(n_pool);

    int32_t * pcell = (int32_t *) pool_cells->data;
    int32_t * pidx  = (int32_t *) pool_idxs->data;
    int32_t * nidx  = (int32_t *) new_pool_idxs->data;
    int64_t * nrep  = (int64_t *) new_pool_rep->data;
    int32_t * npos  = new_pool_pos != nullptr ? (int32_t *) new_pool_pos->data : nullptr;

    if (npos != nullptr) {
        std::fill(npos, npos + 4*n_new_g, 0);
    }

    uint32_t i_new = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        seq_pool_start[s] = (uint32_t) pool_end.size();

        const bool inert = n_stream_kv > 1 && !seq_in_ub[s];

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const uint32_t j  = sq.pools[pi];
            const uint32_t ip = (uint32_t) pool_end.size();
            GGML_ASSERT(ip + 1 < n_pool);

            // The pooled key lives in the last member's row.
            const uint32_t rep = sq.cells[j + kpool - 1].second;
            pcell[ip] = (int32_t) gcell(sq, rep);

            for (uint32_t k = 0; k < kpool; ++k) {
                pidx[(size_t) ip*kpool + k] = inert ? sentinel : (int32_t) sq.cells[j + k].second;
            }

            if (st.is_new[ip] == st.generation) {
                GGML_ASSERT(i_new < n_new);
                for (uint32_t k = 0; k < kpool; ++k) {
                    nidx[(size_t) i_new*kpool + k] = (int32_t) gcell(sq, sq.cells[j + k].second);
                }
                nrep[i_new] = gcell(sq, rep);
                if (npos != nullptr) {
                    // a pooled key is rotated to the M-RoPE position of its first member
                    const uint32_t c = sq.cells[j].second;
                    const auto &   e = mem->get_mem_idx()->get_cells(s).ext_get(c);
                    npos[0*n_new_g + i_new] = sq.cells[j].first;
                    npos[1*n_new_g + i_new] = e.y;
                    npos[2*n_new_g + i_new] = e.x;
                    npos[3*n_new_g + i_new] = sq.cells[j].first;
                }
                ++i_new;
            }

            pool_end.push_back(sq.cells[j + kpool - 1].first);
        }
    }
    GGML_ASSERT(i_new == n_new);

    // Padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read.
    // Each entry takes its own cell, entries sharing one would write it from several threads in the scatter.
    if (n_new_g > n_new) {
        std::vector<int64_t> reps(pcell, pcell + pool_end.size());
        std::sort(reps.begin(), reps.end());

        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (std::binary_search(reps.begin(), reps.end(), pad_cell)) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < (int64_t) kv_size*n_stream_kv);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            nrep[i] = pad_cell;
        }
    }

    const uint32_t n_pool_real = (uint32_t) pool_end.size();
    for (uint32_t ip = n_pool_real; ip < n_pool; ++ip) {
        pcell[ip] = (int32_t) dummy_cell; // pool_cells always addresses the K storage
        for (uint32_t k = 0; k < kpool; ++k) {
            pidx[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible when it belongs to the token's sequence and ends at or before it
    auto fill_mask = [&](auto * data) {
        using T = std::remove_pointer_t<decltype(data)>;
        const T keep = llama_cast<T>(0.0f);
        const T drop = llama_cast<T>(-INFINITY);

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];

            T * row = data + (size_t) i*n_pool;
            std::fill(row, row + n_pool, drop);

            const uint32_t p0 = seq_pool_start[s];
            const uint32_t p1 = p0 + (uint32_t) lay.seqs[s].pools.size();
            const uint32_t nv = by_order ? std::min(p1 - p0, (uint32_t) ((rank[i] + 1)/kpool)) :
                (uint32_t) (std::upper_bound(pool_end.begin() + p0, pool_end.begin() + p1, p) - (pool_end.begin() + p0));
            std::fill(row + p0, row + p0 + nv, keep);

            // Finite visible pools occupy the first min(nv, n_top) ranked slots.
            if (gm != nullptr) {
                const uint32_t nvc = std::min(nv, n_top);
                float * grow = gm + (size_t) i*n_sel;
                std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
            }
        }
    };
    if (pool_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) pool_mask->data);
    } else {
        fill_mask((float *) pool_mask->data);
    }

    int32_t * tidx = (int32_t *) tail_idxs->data;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & sq = lay.seqs[s];

        const uint32_t n_tail = by_order ?
            (uint32_t) ((rank[i] + 1) % kpool) :
            (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail && by_order) {
                const uint32_t c = sq.cells[rank[i] - k].second;
                cell = (int32_t) c;
                real = true;
            } else if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) it->second;
                    real = true;
                }
            }
            tidx[(size_t) i*tail_idxs->ne[0] + k] = cell;

            if (gm != nullptr && n_sel % kpool != 0) {
                gm[(size_t) i*n_sel + (size_t) n_top*kpool + k] = real ? 0.0f : -INFINITY;
            }
        }
    }
}

// The identity layout: one sequence in a single stream whose cell j holds position j for each of its positions. Pool b is
//   then the cells [b*kpool, (b + 1)*kpool) with its pooled key in the last one, as the ramps hold.
bool llama_memory_hybrid_idx_context::kpool_ident(const llama_ubatch & ubatch) const {
    if (!kpool_track() || mem->get_kpool_ramp_cells() == nullptr || mem->get_kpool_by_order() ||
            ubatch.n_tokens != 1 || ubatch.n_seq_id[0] != 1 || mem->get_mem_idx()->get_n_stream() != 1) {
        return false;
    }

    const llama_seq_id s   = ubatch.seq_id[0][0];
    const auto &       lay = mem->kpool_layout_get();
    for (llama_seq_id q = 0; q < LLAMA_MAX_SEQ; ++q) {
        if (q != s && !lay.seqs[q].cells.empty()) {
            return false;
        }
    }

    const auto & sq = lay.seqs[s];

    return !sq.cells.empty() && sq.n_ident == sq.cells.size();
}

void llama_memory_hybrid_idx_context::set_input_kpool_ident(ggml_tensor * vis, ggml_tensor * tail_idxs,
        ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep, const llama_ubatch * ubatch) const {
    GGML_ASSERT(kpool_ident(*ubatch));
    GGML_ASSERT(ggml_backend_buffer_is_host(vis->buffer) && vis->type == GGML_TYPE_I32 && ggml_nelements(vis) == 2);
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_rep->buffer));

    const uint32_t kpool   = mem->get_kpool();
    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const auto &   st      = kpool_cur();

    GGML_ASSERT(tail_idxs->ne[0] >= (int64_t) kpool - 1 && tail_idxs->ne[1] == 1);
    GGML_ASSERT(ggml_nelements(new_pool_idxs) == (int64_t) kpool*st.n_new_g && new_pool_rep->ne[0] == (int64_t) st.n_new_g);

    const llama_pos p = ubatch->pos[0];

    // the pools ending at or before the token see the query, as in set_input_kpool's mask
    int32_t * v = (int32_t *) vis->data;
    v[0] = (int32_t) get_n_kpool_visible(*ubatch, 0);
    v[1] = 0;

    // the pools this ubatch completes or rebuilds, from the first one marked
    int32_t * nidx = (int32_t *) new_pool_idxs->data;
    int64_t * nrep = (int64_t *) new_pool_rep->data;

    uint32_t i_new = 0;
    for (uint32_t b = st.new_lo; b < st.n_pool_real && i_new < st.n_new; ++b) {
        if (st.is_new[b] != st.generation) {
            continue;
        }
        for (uint32_t k = 0; k < kpool; ++k) {
            nidx[(size_t) i_new*kpool + k] = (int32_t) (b*kpool + k);
        }
        nrep[i_new] = (int64_t) b*kpool + kpool - 1;
        ++i_new;
    }
    GGML_ASSERT(i_new == st.n_new);

    // the padding re-pools the first cell of a pool, never a pool's rep, one cell per entry
    for (uint32_t i = st.n_new; i < st.n_new_g; ++i) {
        const int64_t c = (int64_t) (i - st.n_new)*kpool;
        GGML_ASSERT(c < (int64_t) kv_size);
        for (uint32_t k = 0; k < kpool; ++k) {
            nidx[(size_t) i*kpool + k] = (int32_t) c;
        }
        nrep[i] = c;
    }

    // the incomplete pool the token ends: position p - k is cell p - k, missing cells point at the n_kv sentinel row
    const int32_t  sentinel = (int32_t) get_idx()->get_n_kv();
    const uint32_t n_tail   = (uint32_t) ((p + 1) % (llama_pos) kpool);

    int32_t * tidx = (int32_t *) tail_idxs->data;
    for (uint32_t k = 0; k < kpool - 1; ++k) {
        tidx[k] = k < n_tail ? (int32_t) (p - (llama_pos) k) : sentinel;
    }
}
