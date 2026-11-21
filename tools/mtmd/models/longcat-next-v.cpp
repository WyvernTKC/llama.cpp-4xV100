#include "models.h"

// ref: modular_longcat_next_visual.py (VisualQuantizer, VisualEmbeddingBridge) and get_visual_embeddings in modeling_longcat_next.py
ggml_cgraph * mtmd_graph_longcat_next_v::build() {
    ggml_tensor * cur = build_qwen2vl(FFN_GELU_ERF);

    // quant_conv: LayerNorm, Linear, GELU, Linear
    cur = build_norm(cur, model.v_rvq_in_norm_w, model.v_rvq_in_norm_b, NORM_TYPE_NORMAL, norm_eps, -1);
    cur = build_ffn(cur,
            model.v_rvq_in_fc1_w, model.v_rvq_in_fc1_b,
            nullptr, nullptr,
            model.v_rvq_in_fc2_w, model.v_rvq_in_fc2_b,
            FFN_GELU_ERF, -1);
    cb(cur, "rvq_in", -1);

    // residual quantizer with one codebook for all levels; each level's code picks a row of the LLM embedding
    ggml_tensor * codebook = model.v_rvq_codebook;
    ggml_tensor * code_norm = ggml_sum_rows(ctx0, ggml_sqr(ctx0, codebook));      // [1, n_code]
    code_norm = ggml_cont(ctx0, ggml_transpose(ctx0, code_norm));                 // [n_code, 1]

    ggml_tensor * table = model.mm_code_embd;
    const int64_t n_levels = table->ne[2];

    ggml_tensor * residual = cur;
    ggml_tensor * embd = nullptr;
    for (int64_t l = 0; l < n_levels; l++) {
        // nearest entry: argmin |r - c|^2 = argmax (2 r.c - |c|^2)
        ggml_tensor * scores = ggml_sub(ctx0, ggml_scale(ctx0, ggml_mul_mat(ctx0, codebook, residual), 2.0f), code_norm);
        ggml_tensor * idx = ggml_argmax(ctx0, scores);
        cb(idx, "rvq_code", l);

        if (l + 1 < n_levels) {
            residual = ggml_sub(ctx0, residual, ggml_get_rows(ctx0, codebook, idx));
        }

        ggml_tensor * rows = ggml_view_2d(ctx0, table, table->ne[0], table->ne[1], table->nb[1], l * table->nb[2]);
        ggml_tensor * e = ggml_get_rows(ctx0, rows, idx);
        embd = embd ? ggml_add(ctx0, embd, e) : e;
    }
    cb(embd, "code_embd_sum", -1);

    // visual_embedding_layer: x + MLP(LayerNorm(x))
    cur = build_norm(embd, model.mm_input_norm_w, model.mm_input_norm_b, NORM_TYPE_NORMAL, norm_eps, -1);
    cur = build_ffn(cur,
            model.mm_ffn_up_w, nullptr,
            model.mm_ffn_gate_w, nullptr,
            model.mm_ffn_down_w, nullptr,
            FFN_SILU, -1);
    cur = ggml_add(ctx0, embd, cur);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
