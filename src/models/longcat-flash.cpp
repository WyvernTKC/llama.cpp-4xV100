#include "models.h"

#include "llama-kv-cache.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// LongCat-Flash (ref: transformers modeling_longcat_flash.py)
//
// Each HF layer holds two blocks of MLA attention + dense FFN, stored as two llama.cpp layers, and one
// MoE ("shortcut MoE") that reads the input of the first dense FFN and is added at the end of the
// second block. The router picks top-k among the real experts and n_zero_experts identity experts: an
// identity expert returns its input times its weight. The router weights are the unnormalized softmax
// probabilities times routed_scaling_factor; e_score_correction_bias only steers the selection.
//
// LongCat-Ngram (LongCat-Flash-Lite, LongCat-Next text): the input embedding also adds hashed n-gram
// embeddings, see build_inp_ngram.

void llama_model_longcat_flash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm,  false);
    ml.get_key(LLM_KV_N_ZERO_EXPERTS,              hparams.n_zero_experts,       false);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_SCALE,      hparams.f_attn_q_lora_scale,  false);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_SCALE,     hparams.f_attn_kv_lora_scale, false);

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false)) {
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
        // cancel the factor from the convert script
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    if (hparams.n_layer() % 2 != 0) {
        throw std::runtime_error("longcat-flash needs an even block count (2 blocks per HF layer)");
    }
    if (hparams.n_lora_q == 0 || hparams.n_lora_kv == 0) {
        throw std::runtime_error("longcat-flash needs q_lora_rank and kv_lora_rank");
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_longcat_flash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    GGML_ASSERT(hparams.is_mla());

    // the head sizes after decompression with wv_b
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    GGML_ASSERT(n_embd_head_qk_nope >= 1);

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t n_ff_exp     = hparams.n_ff_exp();

    if (n_expert == 0 || n_expert_used == 0) {
        throw std::runtime_error("longcat-flash needs n_expert and n_expert_used");
    }

    // the router also scores the identity experts
    const int64_t n_expert_router = n_expert + hparams.n_zero_experts;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

        layer.wq_a      = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.wq_b      = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);
        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, 0);
        layer.wk_b      = create_tensor(tn(LLM_TENSOR_ATTN_K_B,      "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
        layer.wv_b      = create_tensor(tn(LLM_TENSOR_ATTN_V_B,      "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        // the MoE of an HF layer is stored with its first block
        if (i % 2 == 0) {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert_router}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert_router}, TENSOR_NOT_REQUIRED);

            // +1: an all-zero expert, the slots of the identity experts point at it
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert + 1}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert + 1}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert + 1}, 0);
        }
    }
}

void llama_model_longcat_ngram::load_arch_hparams(llama_model_loader & ml) {
    llama_model_longcat_flash::load_arch_hparams(ml);

    uint32_t eos_id = 0;

    ml.get_key(LLM_KV_NGRAM_NEIGHBOR_COUNT, ngram_n_neighbor);
    ml.get_key(LLM_KV_NGRAM_SPLIT_COUNT,    ngram_n_split);
    ml.get_key(LLM_KV_NGRAM_EOS_TOKEN_ID,   eos_id);
    ml.get_key(LLM_KV_NGRAM_HASH_BASE,      ngram_hash_base, false);
    ml.get_arr(LLM_KV_NGRAM_VOCAB_SIZES,    ngram_vocab_sizes);

    ngram_eos_id = (llama_token) eos_id;

    if (ngram_n_neighbor < 2 || ngram_n_split == 0) {
        throw std::runtime_error("longcat-ngram needs ngram.neighbor_count > 1 and ngram.split_count > 0");
    }

    hparams.n_ngram_table = ngram_n_split * (ngram_n_neighbor - 1);

    if (ngram_vocab_sizes.size() != hparams.n_ngram_table) {
        throw std::runtime_error(format("longcat-ngram expects %u n-gram vocab sizes, got %zu",
                    hparams.n_ngram_table, ngram_vocab_sizes.size()));
    }
    for (uint32_t m : ngram_vocab_sizes) {
        if (m == 0) {
            throw std::runtime_error("longcat-ngram: n-gram vocab size of zero");
        }
    }
}

void llama_model_longcat_ngram::load_arch_tensors(llama_model_loader & ml) {
    llama_model_longcat_flash::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const uint32_t n_table = hparams.n_ngram_table;
    const uint32_t n_prev  = ngram_n_neighbor - 1;

    if (n_embd % n_table != 0) {
        throw std::runtime_error("longcat-ngram needs n_embd to be a multiple of the n-gram table count");
    }
    const int64_t n_embd_ngram = n_embd / n_table;

    // without a key the hash runs over the whole vocab (LongCat-Flash-Lite)
    if (ngram_hash_base == 0) {
        ngram_hash_base = (uint32_t) n_vocab;
    }

    // table t belongs to n-gram order t/n_split + 2, so it hashes t/n_split + 1 predecessors
    ngram_pow_mods.assign((size_t) n_table * n_prev, 0);
    for (uint32_t t = 0; t < n_table; ++t) {
        const uint64_t m = ngram_vocab_sizes[t];

        uint64_t pow_mod = 1;
        for (uint32_t s = 0; s < t/ngram_n_split + 1; ++s) {
            pow_mod = (pow_mod * ngram_hash_base) % m;
            ngram_pow_mods[t*n_prev + s] = pow_mod;
        }
    }

    ngram_ignored.assign(n_vocab, 0);
    {
        std::vector<int32_t> ignored;
        ml.get_arr(LLM_KV_NGRAM_IGNORED_TOKEN_IDS, ignored, false);
        for (int32_t id : ignored) {
            if (id >= 0 && id < n_vocab) {
                ngram_ignored[id] = 1;
            }
        }
    }

    // the tables are input tensors, their index goes into the name suffix
    ngram_suffixes.resize(n_table);
    ngram_embd.resize(n_table);
    for (uint32_t t = 0; t < n_table; ++t) {
        ngram_suffixes[t] = std::to_string(t) + ".weight";
        ngram_embd[t] = create_tensor(tn(LLM_TENSOR_NGRAM_EMBD, ngram_suffixes[t].c_str()), {n_embd_ngram, ngram_vocab_sizes[t]}, 0);
    }

    ngram_proj = create_tensor(tn(LLM_TENSOR_NGRAM_PROJ, "weight"), {n_embd_ngram * n_table, n_embd}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_longcat_flash::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// The hashed n-gram rows of every token. The hash needs 64 bit integers, so it runs on the host, and so
// does the gather when the tables are in host memory (the usual case, they are big input tensors).
//
//   hash_t = tok + sum_{s=1..t/n_split+1} prev_s * base^s     row_t = hash_t mod vocab_sizes[t]
//
// prev_s is the token s positions back. The look-back stops at the sequence start, at an EOS and at a
// token that reads as 0 (an ignored token or an embedding input). Row 0 reads as a zero vector. A token
// that reads as 0 gets no n-gram rows; an ignored token and an embedding input are also not scaled.
// ref: NgramEmbedding and _shift_right_ignore_eos in modeling_longcat_ngram.py
class llm_graph_input_longcat_ngram : public llm_graph_input_i {
public:
    llm_graph_input_longcat_ngram(const llama_model_longcat_ngram & pmodel, const llama_kv_cache_context * mctx) :
        pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_longcat_ngram() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_context *>(params.mctx);
        return scale->ne[1] == params.ubatch.n_tokens;
    }

    // host gather (the tables are in host memory)
    ggml_tensor * emb   = nullptr; // F32 [n_embd_ngram*n_table, n_tokens]

    // otherwise the graph gathers the rows
    ggml_tensor * ids   = nullptr; // I32 [n_tokens*n_table], table-major
    ggml_tensor * mask  = nullptr; // F32 [1, n_table, n_tokens], 0 for row 0

    ggml_tensor * scale = nullptr; // F32 [1, n_tokens]

    const llama_model_longcat_ngram & pmodel;

    // the predecessor tokens live in the KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
    std::vector<float>       rows;
    std::vector<int32_t>     row_ids;
    std::vector<float>       row_mask;
    std::vector<float>       scales;
};

void llm_graph_input_longcat_ngram::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(mctx != nullptr);

    const int64_t  n_tokens     = ubatch->n_tokens;
    const uint32_t n_table      = pmodel.ngram_embd.size();
    const uint32_t n_prev       = pmodel.ngram_n_neighbor - 1;
    const uint32_t n_split      = pmodel.ngram_n_split;
    const int64_t  n_embd_ngram = pmodel.ngram_embd[0]->ne[0];

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "n-gram input embeddings do not support tokens shared by multiple sequences");
    }

    // apply_ubatch() already stored this ubatch in the cells
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    if (emb) {
        rows.assign((size_t) n_tokens * n_table * n_embd_ngram, 0.0f);
    } else {
        row_ids.assign((size_t) n_tokens * n_table, 0);
        row_mask.assign((size_t) n_tokens * n_table, 0.0f);
    }
    scales.resize(n_tokens);

    const float s_mix = 1.0f/(1 + n_table);

    std::vector<uint64_t> shifted(n_prev + 1, 0);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an embedding input (multimodal) replaces the whole input embedding in the reference
        if (!ubatch->token || (ubatch->is_mixed() && ubatch->type[i])) {
            scales[i] = 1.0f;
            continue;
        }

        const llama_token tok = ubatch->token[i];
        const bool ignored = tok >= 0 && (size_t) tok < pmodel.ngram_ignored.size() && pmodel.ngram_ignored[tok];

        scales[i] = ignored ? 1.0f : s_mix;

        const uint64_t cur = pmodel.ngram_tok(tok);
        if (cur == 0) {
            continue;
        }

        bool cut = false;
        for (uint32_t s = 1; s <= n_prev; ++s) {
            // prev[] is oldest first, a missing entry is LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            const int64_t     v = pmodel.ngram_tok(t);

            cut = cut || v == 0 || t == pmodel.ngram_eos_id;

            shifted[s] = cut ? 0 : (uint64_t) v;
        }

        for (uint32_t t = 0; t < n_table; ++t) {
            const uint64_t * pow_mods = pmodel.ngram_pow_mods.data() + (size_t) t*n_prev;

            uint64_t hash = cur;
            for (uint32_t s = 0; s < t/n_split + 1; ++s) {
                hash += shifted[s + 1] * pow_mods[s];
            }

            const uint64_t row = hash % pmodel.ngram_vocab_sizes[t];
            if (row == 0) {
                continue;
            }

            if (!emb) {
                row_ids [t*n_tokens + i] = (int32_t) row;
                row_mask[i*n_table  + t] = 1.0f;
                continue;
            }

            const ggml_tensor * table = pmodel.ngram_embd[t];
            const uint8_t     * src   = (const uint8_t *) table->data + row*table->nb[1];

            ggml_get_type_traits(table->type)->to_float(src, rows.data() + (i*n_table + t)*n_embd_ngram, n_embd_ngram);
        }
    }

    if (emb) {
        ggml_backend_tensor_set(emb, rows.data(), 0, rows.size()*sizeof(float));
    } else {
        ggml_backend_tensor_set(ids,  row_ids.data(),  0, row_ids.size()*sizeof(int32_t));
        ggml_backend_tensor_set(mask, row_mask.data(), 0, row_mask.size()*sizeof(float));
    }
    ggml_backend_tensor_set(scale, scales.data(), 0, scales.size()*sizeof(float));
}

ggml_tensor * llama_model_longcat_flash::graph::build_inp_ngram(ggml_tensor * inp_embd) const {
    const auto & pmodel = static_cast<const llama_model_longcat_ngram &>(model);

    bool host_gather = true;
    for (const ggml_tensor * table : pmodel.ngram_embd) {
        host_gather = host_gather && table->data != nullptr && table->buffer != nullptr &&
                ggml_backend_buffer_is_host(table->buffer) && ggml_get_type_traits(table->type)->to_float != nullptr;
    }

    const int64_t n_table      = pmodel.ngram_embd.size();
    const int64_t n_embd_ngram = pmodel.ngram_embd[0]->ne[0];

    auto inp = std::make_unique<llm_graph_input_longcat_ngram>(pmodel, static_cast<const llama_kv_cache_context *>(mctx));

    ggml_tensor * emb = nullptr;

    if (host_gather) {
        inp->emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_ngram*n_table, n_tokens);
        ggml_set_input(inp->emb);
        emb = inp->emb;
    } else {
        inp->ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens*n_table);
        ggml_set_input(inp->ids);

        inp->mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_table, n_tokens);
        ggml_set_input(inp->mask);

        for (int64_t t = 0; t < n_table; ++t) {
            ggml_tensor * ids_t = ggml_view_1d(ctx0, inp->ids, n_tokens, t*n_tokens*ggml_element_size(inp->ids));
            ggml_tensor * rows  = ggml_get_rows(ctx0, pmodel.ngram_embd[t], ids_t);

            rows = ggml_reshape_3d(ctx0, rows, n_embd_ngram, 1, n_tokens);
            emb  = emb ? ggml_concat(ctx0, emb, rows, 1) : rows;
        }

        emb = ggml_mul(ctx0, emb, inp->mask);
        emb = ggml_reshape_2d(ctx0, emb, n_embd_ngram*n_table, n_tokens);
    }

    inp->scale = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    ggml_set_input(inp->scale);

    ggml_tensor * scale = inp->scale;

    res->add_input(std::move(inp));

    cb(emb, "inp_ngram_rows", -1);

    // sum_t proj_t @ e_t == [proj_0 | ... | proj_{n_table-1}] @ concat(e_t)
    ggml_tensor * cur = ggml_mul_mat(ctx0, pmodel.ngram_proj, emb);
    cb(cur, "inp_ngram_proj", -1);

    // the mean also scales the plain token embedding
    cur = ggml_add(ctx0, inp_embd, cur);
    cur = ggml_mul(ctx0, cur, scale);
    cb(cur, "inp_embd_ngram", -1);

    return cur;
}

llama_model_longcat_flash::graph::graph(const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.is_mla());
    GGML_ASSERT(n_layer % 2 == 0);

    // the head sizes after decompression with wv_b
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t kv_lora_rank = hparams.n_lora_kv;

    // YaRN pre-scales kq_scale, see deepseek2 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    if (hparams.n_ngram_table > 0) {
        inpL = build_inp_ngram(inpL);
    }

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn_k = build_attn_inp_k();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // the MoE of the current HF layer, routed in its first block
    moe_route route;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention, MLA as in deepseek2 plus the fixed lora scales
        {
            ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
            cb(q, "q", il);

            q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(q, "q", il);

            q = ggml_mul_mat(ctx0, layer.wq_b, q);
            cb(q, "q", il);

            // kq_scale goes in here too: with the lora scales the raw q.k sums overflow F16 in the Volta flash attention
            q = ggml_scale(ctx0, q, (hparams.f_attn_q_lora_scale != 0.0f ? hparams.f_attn_q_lora_scale : 1.0f) * kq_scale);
            cb(q, "q_scaled", il);

            // {n_embd_head_k, n_head, n_tokens}
            q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // and {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                    ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            k_pe = build_rope(k_pe, inp_pos, il);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            // the reference scales the latent only, not k_pe
            if (hparams.f_attn_kv_lora_scale != 0.0f) {
                kv_cmpr = ggml_scale(ctx0, kv_cmpr, hparams.f_attn_kv_lora_scale);
                cb(kv_cmpr, "kv_cmpr_scaled", il);
            }

            // split into {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                    q->nb[1], q->nb[2], 0);
            cb(q_nope, "q_nope", il);

            // and {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                    q->nb[1], q->nb[2], ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            q_pe = build_rope(q_pe, inp_pos, il);
            cb(q_pe, "q_pe", il);

            // {n_embd_head_qk_nope, n_tokens, n_head}
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            cb(q_nope, "q_nope_perm", il);

            // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
            ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
            cb(q_nope_absorbed, "q_nope_absorbed", il);

            // {kv_lora_rank, n_head, n_tokens}
            q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
            cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

            // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
            // note: rope must go first for in-place context shifting in build_rope_shift()
            ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
            cb(Qcur, "Qcur", il);

            kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
            cb(kv_cmpr, "kv_cmpr_reshape", il);

            // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
            ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
            cb(Kcur, "Kcur", il);

            // {kv_lora_rank, 1, n_tokens}
            ggml_tensor * Vcur = kv_cmpr;
            cb(Vcur, "Vcur", il);

            // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
            cur = build_attn(inp_attn_k,
                    layer.wo, NULL, layer.wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, 1.0f, il);
        }

        const bool out_rows = il == n_layer - 1 && inp_out_ids;
        if (out_rows) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // the residual add and the norm first after the all-reduce of the attention: under -sm tensor the norm polls it
        ggml_build_forward_expand(gf, cur);

        // the MoE of the HF layer reads the input of its first dense FFN
        if (layer.ffn_gate_inp) {
            route = build_moe_route(cur, layer, il);
        }

        // and lands after its second block: the identity part goes to the residual here, the routed experts run before
        //   the dense FFN, so that the two partial sums meet in one add, one all-reduce under -sm tensor, and the add of
        //   the residual is the first node after it
        const bool    moe_here = il % 2 == 1 && route.il >= 0;
        ggml_tensor * residual = ffn_inp;
        ggml_tensor * moe_sum  = nullptr;
        if (moe_here) {
            residual = ggml_add(ctx0, ffn_inp, out_rows ? ggml_get_rows(ctx0, route.identity, inp_out_ids) : route.identity);
            cb(residual, "ffn_inp_moe_identity", il);
            ggml_build_forward_expand(gf, residual);

            moe_sum = build_moe_experts(route, il);
            if (out_rows) {
                moe_sum = ggml_get_rows(ctx0, moe_sum, inp_out_ids);
            }
            route = moe_route();
        }

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (moe_sum) {
            cur = ggml_add(ctx0, moe_sum, cur);
            cb(cur, "ffn_out_moe", il);
        }

        cur = ggml_add(ctx0, cur, residual);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// The router scores n_expert real and n_zero_experts identity experts. The picks of an identity expert run the extra
//   all-zero expert n_expert, so mul_mat_id gives 0 for them, and the identity part is cur times the sum of their
//   weights. The router is built in the order of build_moe_ffn (softmax, its reshape, the bias, top-k, the weights),
//   which the CUDA backend runs as one kernel (topk_moe), and the ids and identity mask are rows of two tables.
llama_model_longcat_flash::graph::moe_route llama_model_longcat_flash::graph::build_moe_route(
        ggml_tensor * cur, const llama_layer & layer, int il) const {
    const int64_t n_expert_router = layer.ffn_gate_inp->ne[1];

    ggml_tensor * logits = ggml_mul_mat(ctx0, layer.ffn_gate_inp, cur); // [n_expert_router, n_tokens]
    cb(logits, "ffn_moe_logits", il);

    ggml_tensor * probs = ggml_soft_max(ctx0, logits);
    cb(probs, "ffn_moe_probs", il);

    ggml_tensor * probs_rows = ggml_reshape_3d(ctx0, probs, 1, n_expert_router, n_tokens);

    // the bias only steers the selection, the weights come from the plain probabilities
    ggml_tensor * selection_probs = probs;
    if (layer.ffn_exp_probs_b) {
        selection_probs = ggml_add(ctx0, probs, layer.ffn_exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    ggml_tensor * selected = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected, "ffn_moe_topk", il);

    ggml_tensor * weights = ggml_get_rows(ctx0, probs_rows, selected); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);

    if (hparams.expert_weights_norm) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx0, weights, weights_sum);
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_norm", il);
    }
    if (hparams.expert_weights_scale != 0.0f && hparams.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }
    // the router's nodes in that order, before anything else reads them
    ggml_build_forward_expand(gf, weights);

    if (lut_ids == nullptr) {
        // [1, n_expert_router, n_tokens]: get_rows takes a table per token
        ggml_tensor * e = ggml_arange(ctx0, 0.0f, float(n_expert_router), 1.0f);
        e = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, e, 1, n_expert_router, 1), 1, n_expert_router, n_tokens, 1);
        // identity experts -> the all-zero expert n_expert
        lut_ids  = ggml_cast(ctx0, ggml_clamp(ctx0, e, 0.0f, float(n_expert)), GGML_TYPE_I32);
        // 1 for an identity expert, 0 for a real one: clamp(e - (n_expert - 1), 0, 1)
        lut_zero = ggml_clamp(ctx0, ggml_scale_bias(ctx0, e, 1.0f, -float(n_expert - 1)), 0.0f, 1.0f);
    }

    moe_route route;
    route.cur     = cur;
    route.weights = weights;
    route.il      = il;

    route.ids = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, lut_ids, selected), n_expert_used, n_tokens);
    cb(route.ids, "ffn_moe_ids", il);

    ggml_tensor * zero_mask = ggml_get_rows(ctx0, lut_zero, selected); // [1, n_expert_used, n_tokens]
    cb(zero_mask, "ffn_moe_zero_mask", il);

    ggml_tensor * w_zero = ggml_sum_rows(ctx0, ggml_reshape_2d(ctx0, ggml_mul(ctx0, weights, zero_mask), n_expert_used, n_tokens)); // [1, n_tokens]
    cb(w_zero, "ffn_moe_weights_zero", il);

    route.identity = ggml_mul(ctx0, cur, w_zero);
    cb(route.identity, "ffn_moe_identity", il);

    ggml_build_forward_expand(gf, route.ids);
    ggml_build_forward_expand(gf, route.identity);

    return route;
}

// The weighted sum of the routed experts, views before adds as in build_moe_ffn: the CUDA backend sums them in one kernel,
//   and under -sm tensor the meta backend all-reduces the sum (with the next partial sum it is added to)
ggml_tensor * llama_model_longcat_flash::graph::build_moe_experts(const moe_route & route, int il) const {
    const llama_layer & layer = model.layers[route.il];

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx0, route.cur, n_embd, 1, n_tokens);

    ggml_tensor * up   = ggml_mul_mat_id(ctx0, layer.ffn_up_exps,   cur_3d, route.ids); // [n_ff_exp, n_expert_used, n_tokens]
    ggml_tensor * gate = ggml_mul_mat_id(ctx0, layer.ffn_gate_exps, cur_3d, route.ids);
    cb(up,   "ffn_moe_up",   il);
    cb(gate, "ffn_moe_gate", il);

    ggml_tensor * experts = ggml_swiglu_split(ctx0, gate, up);
    cb(experts, "ffn_moe_swiglu", il);

    experts = ggml_mul_mat_id(ctx0, layer.ffn_down_exps, experts, route.ids); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    experts = ggml_mul(ctx0, experts, route.weights);
    cb(experts, "ffn_moe_weighted", il);
    ggml_build_forward_expand(gf, experts);

    ggml_tensor * views[LLAMA_MAX_EXPERTS] = { nullptr };
    for (int64_t i = 0; i < n_expert_used; ++i) {
        views[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
        ggml_build_forward_expand(gf, views[i]);
    }

    ggml_tensor * moe_sum = views[0];
    for (int64_t i = 1; i < n_expert_used; ++i) {
        moe_sum = ggml_add(ctx0, moe_sum, views[i]);
        ggml_build_forward_expand(gf, moe_sum);
    }
    if (n_expert_used == 1) {
        moe_sum = ggml_cont(ctx0, moe_sum);
    }
    cb(moe_sum, "ffn_moe_out", il);

    return moe_sum;
}
