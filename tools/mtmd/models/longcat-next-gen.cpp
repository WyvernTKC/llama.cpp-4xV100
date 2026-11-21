#include "models.h"

// ref: CasualDepthTransformerHead (modular_longcat_next.py), LongcatNextAudioTokenizer.decode (modular_longcat_next_audio.py)

ggml_cgraph * mtmd_graph_longcat_next_gen::build() {
    if (gen_process == MTMD_SUBCTX_GEN_PROCESS_TTS_VOCODE) {
        return build_vocoder(vocode_n_mel, vocode_n_stft);
    }
    if (gen_process == MTMD_SUBCTX_GEN_PROCESS_TTS) {
        return build_mel();
    }
    return build_depth_head();
}

// whisper-style layer of the decoder and the prenet: pre-norm, causal attention (no k bias), GELU FFN
ggml_tensor * mtmd_graph_longcat_next_gen::dec_layer(const mtmd_layer & l, ggml_tensor * x, norm_type nt, float eps, int n_head,
                                                     ggml_tensor * mask) {
    const int64_t d = x->ne[0];
    const int64_t T = x->ne[1];
    const int64_t d_head = d / n_head;

    ggml_tensor * cur = build_norm(x, l.ln_1_w, l.ln_1_b, nt, eps, -1);
    ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_add(ctx0, build_mm(l.q_w, cur), l.q_b), d_head, n_head, T);
    ggml_tensor * k = ggml_reshape_3d(ctx0, build_mm(l.k_w, cur), d_head, n_head, T);
    ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_add(ctx0, build_mm(l.v_w, cur), l.v_b), d_head, n_head, T);
    x = ggml_add(ctx0, x, build_attn(l.o_w, l.o_b, q, k, v, mask, 1.0f / sqrtf((float) d_head), -1));

    cur = build_norm(x, l.ln_2_w, l.ln_2_b, nt, eps, -1);
    cur = build_ffn(cur, l.ff_up_w, l.ff_up_b, nullptr, nullptr, l.ff_down_w, l.ff_down_b, FFN_GELU_ERF, -1);
    return ggml_add(ctx0, x, cur);
}

std::vector<c2w_state_slot> list_longcat_next_gen_state_slots(const mtmd_hparams & hparams, const mtmd_model & model) {
    std::vector<c2w_state_slot> slots;
    const int64_t d     = model.lcn.hidden_proj_w->ne[1];
    const int64_t depth = (int64_t) hparams.rvq_codebook_size.size();
    for (size_t il = 0; il < model.lcn.head_layers.size(); il++) {
        const std::string prefix = mtmd_graph_longcat_next_gen::head_kv_slot;
        slots.push_back({ prefix + "k_" + std::to_string(il), d, depth, true });
        slots.push_back({ prefix + "v_" + std::to_string(il), d, depth, true });
    }
    return slots;
}

// one level of the depth head: the logits of level n_codes, given the codes of the levels before it. Only position n_codes
// runs: its key and value go to the slots, its query reads the slots up to it (the earlier positions of this frame).
ggml_cgraph * mtmd_graph_longcat_next_gen::build_depth_head() {
    const auto & g     = model.lcn;
    const auto & sizes = hparams.rvq_codebook_size;
    const int depth    = (int) sizes.size();
    const int n_codes  = n_tokens;
    GGML_ASSERT(n_codes >= 0 && n_codes <= depth);
    const int64_t n_embd = g.code_embd->ne[0];

    ggml_tensor * h = ggml_reshape_2d(ctx0, build_inp_raw(1), n_embd, 1);

    // the sum of the LLM rows of the given codes: the input of position n_codes, and the input row of the next LLM step
    ggml_tensor * acc = nullptr;
    if (n_codes > 0) {
        ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_codes);
        ggml_set_name(ids, "inp_codes");
        ggml_set_input(ids);
        ggml_tensor * rows = ggml_get_rows(ctx0, g.code_embd, ids); // [n_embd, n_codes]
        for (int j = 0; j < n_codes; j++) {
            ggml_tensor * r = ggml_view_2d(ctx0, rows, n_embd, 1, rows->nb[1], (size_t) j * rows->nb[1]);
            acc = acc ? ggml_add(ctx0, acc, r) : r;
        }
    }

    if (n_codes < depth) {
        const int pos    = n_codes;
        const int n_head = hparams.n_head;
        const int64_t d  = g.hidden_proj_w->ne[1];
        const int64_t d_head = d / n_head;

        std::vector<ggml_tensor *> slots;
        if (gen_state) {
            slots = gen_state->slots;
        } else {
            // buffer reserve: stand-ins of the slots
            for (size_t i = 0; i < 2 * g.head_layers.size(); i++) {
                ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, depth);
                ggml_set_input(t);
                slots.push_back(t);
            }
        }
        GGML_ASSERT(slots.size() == 2 * g.head_layers.size());

        ggml_tensor * x = build_norm(pos == 0 ? h : acc, g.hidden_norm_w, nullptr, NORM_TYPE_RMS, hparams.eps, -1);
        x = build_mm(g.hidden_proj_w, x); // [d, 1]
        for (size_t il = 0; il < g.head_layers.size(); il++) {
            const auto & l = g.head_layers[il];
            ggml_tensor * k_slot = slots[2 * il];
            ggml_tensor * v_slot = slots[2 * il + 1];

            ggml_tensor * cur = build_norm(x, l.ln_1_w, nullptr, NORM_TYPE_RMS, hparams.eps, (int) il);
            ggml_tensor * q = ggml_add(ctx0, build_mm(l.q_w, cur), l.q_b);
            ggml_tensor * k = build_mm(l.k_w, cur);
            ggml_tensor * v = ggml_add(ctx0, build_mm(l.v_w, cur), l.v_b);

            // this position's key and value for the calls of the later levels; the earlier ones are read from columns
            // before pos, so the store and the read never touch the same bytes
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, k, ggml_view_2d(ctx0, k_slot, d, 1, k_slot->nb[1], (size_t) pos * k_slot->nb[1])));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, v, ggml_view_2d(ctx0, v_slot, d, 1, v_slot->nb[1], (size_t) pos * v_slot->nb[1])));
            if (pos > 0) {
                k = ggml_concat(ctx0, ggml_view_2d(ctx0, k_slot, d, pos, k_slot->nb[1], 0), k, 1);
                v = ggml_concat(ctx0, ggml_view_2d(ctx0, v_slot, d, pos, v_slot->nb[1], 0), v, 1);
            }
            q = ggml_reshape_3d(ctx0, q, d_head, n_head, 1);
            k = ggml_reshape_3d(ctx0, k, d_head, n_head, pos + 1);
            v = ggml_reshape_3d(ctx0, v, d_head, n_head, pos + 1);
            x = ggml_add(ctx0, x, build_attn(l.o_w, l.o_b, q, k, v, nullptr, 1.0f / sqrtf((float) d_head), (int) il));

            // the FFN of this depth position: page pos of the 3D weights
            cur = build_norm(x, l.ln_2_w, nullptr, NORM_TYPE_RMS, hparams.eps, (int) il);
            ggml_tensor * up = ggml_view_2d(ctx0, l.ff_up_w, l.ff_up_w->ne[0], l.ff_up_w->ne[1], l.ff_up_w->nb[1],
                                            (size_t) pos * l.ff_up_w->nb[2]);
            ggml_tensor * down = ggml_view_2d(ctx0, l.ff_down_w, l.ff_down_w->ne[0], l.ff_down_w->ne[1], l.ff_down_w->nb[1],
                                              (size_t) pos * l.ff_down_w->nb[2]);
            cur = ggml_gelu_erf(ctx0, ggml_mul_mat(ctx0, up, cur));
            x = ggml_add(ctx0, x, ggml_mul_mat(ctx0, down, cur));
        }

        // the head of level n_codes
        int64_t off = 0;
        for (int j = 0; j < n_codes; j++) {
            off += sizes[j] + 1;
        }
        const int64_t n_out = sizes[n_codes] + 1;
        ggml_tensor * last = build_norm(x, g.out_norm_w, nullptr, NORM_TYPE_RMS, hparams.eps, -1);
        ggml_tensor * w = ggml_view_2d(ctx0, g.out_w, g.out_w->ne[0], n_out, g.out_w->nb[1], (size_t) off * g.out_w->nb[1]);
        ggml_tensor * b = ggml_view_1d(ctx0, g.out_b, n_out, (size_t) off * ggml_element_size(g.out_b));
        ggml_tensor * logits = ggml_add(ctx0, build_mm(w, last), b);
        ggml_set_name(logits, "out_feats");
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    }

    // last node: the sum of the rows of the given codes, the input row of the next LLM step once all levels are known
    ggml_tensor * fb = acc ? ggml_cont(ctx0, acc) : ggml_scale(ctx0, h, 0.0f);
    ggml_build_forward_expand(gf, fb);
    return gf;
}

// codes of T frames -> mel [80, 4T - 1] and f0
ggml_cgraph * mtmd_graph_longcat_next_gen::build_mel() {
    const auto & g     = model.lcn;
    const auto & c     = model.cbx;
    const auto & sizes = hparams.rvq_codebook_size;
    const int depth    = (int) sizes.size();
    GGML_ASSERT(n_tokens > 0 && n_tokens % depth == 0);
    const int T  = n_tokens / depth;
    const int T2 = n_mel_frames(T);
    GGML_ASSERT(T2 <= g.dec_pos->ne[1]);

    // ids of the codebook rows, level-major: row offset of the level + code
    ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) T * depth);
    ggml_set_name(ids, "inp_codes");
    ggml_set_input(ids);
    ggml_tensor * x = nullptr;
    for (int l = 0; l < depth; l++) {
        ggml_tensor * il = ggml_view_1d(ctx0, ids, T, (size_t) l * T * ggml_element_size(ids));
        ggml_tensor * e = ggml_get_rows(ctx0, g.rvq_codebook, il);
        x = x ? ggml_add(ctx0, x, e) : e;
    }
    x = cbx_linear(g.proj_w, g.proj_b, x); // [1280, T]

    // causal transposed conv (stride 4, kernel 3: no right trim) and GroupNorm with one group
    ggml_tensor * y = ggml_conv_transpose_1d(ctx0, g.up_conv_w, ggml_cont(ctx0, ggml_transpose(ctx0, x)), 4, 0, 1);
    y = ggml_add(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, y)), g.up_conv_b); // [1280, T2]
    const int64_t n_dec = y->ne[0];
    y = ggml_reshape_2d(ctx0, ggml_norm(ctx0, ggml_reshape_2d(ctx0, y, n_dec * T2, 1), 1e-5f), n_dec, T2);
    y = ggml_add(ctx0, ggml_mul(ctx0, y, g.up_norm_w), g.up_norm_b);
    y = ggml_add(ctx0, y, ggml_view_2d(ctx0, g.dec_pos, n_dec, T2, g.dec_pos->nb[1], 0));
    cb(y, "dec_in", -1);

    ggml_tensor * causal = build_inp_attn_mask(T2, T2, 0);
    for (const auto & l : g.dec_layers) {
        y = dec_layer(l, y, NORM_TYPE_NORMAL, 1e-5f, (int) (n_dec / dec_d_head), causal);
    }
    y = build_norm(y, g.dec_post_ln_w, g.dec_post_ln_b, NORM_TYPE_NORMAL, 1e-5f, -1);
    cb(y, "dec_out", -1);

    // prenet: MLP, causal RMSNorm layers, projection to the mel bins
    y = cbx_linear(g.pre_in2_w, g.pre_in2_b, ggml_silu(ctx0, cbx_linear(g.pre_in1_w, g.pre_in1_b, y)));
    const int64_t n_pre = y->ne[0];
    for (const auto & l : g.pre_layers) {
        y = dec_layer(l, y, NORM_TYPE_RMS, 1e-6f, (int) (n_pre / dec_d_head), causal);
    }
    y = build_norm(y, g.pre_out_norm_w, nullptr, NORM_TYPE_RMS, 1e-6f, -1);
    ggml_tensor * mu = build_mm(g.pre_out_w, y); // [80, T2]
    cb(mu, "flow_mu", -1);

    // flow matching: euler steps on the cosine schedule with classifier-free guidance, no speaker or prompt condition
    ggml_tensor * noise = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mu->ne[0], T2);
    ggml_set_name(noise, "inp_noise");
    ggml_set_input(noise);
    const int64_t n_temb = c.time_mlp_1_w->ne[0];
    ggml_tensor * temb_sin = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_temb, n_steps + 1);
    ggml_set_name(temb_sin, "inp_temb");
    ggml_set_input(temb_sin);
    est_mask = build_inp_attn_mask(T2, T2, 1);

    ggml_tensor * mu_zero = ggml_scale(ctx0, mu, 0.0f);
    ggml_tensor * mx = noise;
    for (int i = 0; i < n_steps; i++) {
        const float t0 = 1.0f - cosf((float) i / n_steps * (float) M_PI / 2.0f);
        const float t1 = 1.0f - cosf((float) (i + 1) / n_steps * (float) M_PI / 2.0f);
        ggml_tensor * temb = ggml_view_2d(ctx0, temb_sin, n_temb, 1, temb_sin->nb[1], (size_t) i * temb_sin->nb[1]);
        temb = cbx_linear(c.time_mlp_2_w, c.time_mlp_2_b, ggml_silu(ctx0, cbx_linear(c.time_mlp_1_w, c.time_mlp_1_b, temb)));
        ggml_tensor * d  = estimator(mx, mu, nullptr, nullptr, temb, T2);
        ggml_tensor * du = estimator(mx, mu_zero, nullptr, nullptr, temb, T2);
        d = ggml_add(ctx0, ggml_scale(ctx0, d, 1.0f + cfg), ggml_scale(ctx0, du, -cfg));
        mx = ggml_add(ctx0, mx, ggml_scale(ctx0, d, t1 - t0));
        cb(mx, "cfm_step", i);
    }

    ggml_tensor * mel = mx;
    ggml_set_name(mel, "out_mel");
    ggml_set_output(mel);
    ggml_build_forward_expand(gf, mel);
    build_f0(mel);
    return gf;
}
