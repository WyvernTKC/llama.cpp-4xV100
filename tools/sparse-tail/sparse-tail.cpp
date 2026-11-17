// sparse-tail: dense prefill of T - n_tail tokens, then n_tail teacher-forced single-token decode steps per config,
// KL(dense || config) of the next-token distribution per row. Configs set LLAMA_SPARSE_DECODE / LLAMA_SPARSE_DECODE_SCORE,
// and score:budget:f0|f1|f2|f3 also LLAMA_SPARSE_DECODE_FUSED (without the suffix the variable is left as it is).

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
// MSVC has no setenv/unsetenv; _putenv_s with an empty value removes the variable
static int setenv(const char * name, const char * value, int overwrite) {
    if (!overwrite && getenv(name)) {
        return 0;
    }
    return _putenv_s(name, value);
}
static int unsetenv(const char * name) {
    return _putenv_s(name, "");
}
#endif

struct tail_config {
    std::string name;
    std::string score;
    int         budget = 0;
    std::string fused;
};

struct dump_state {
    std::string    dir;
    std::set<int>  layers;
    std::set<int>  steps;
    bool           active = false;
    bool           with_k = false;
    std::string    prefix;
};

static bool dump_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * ds = (dump_state *) ud;
    if (!ds->active) {
        return false;
    }
    const char * name = t->name;
    const char * dash = strrchr(name, '-');
    if (strncmp(name, "sp_", 3) != 0 || !dash || !dash[1] || strspn(dash + 1, "0123456789") != strlen(dash + 1)) {
        return false;
    }
    const std::string base(name, dash - name);
    if (base != "sp_q" && base != "sp_stats" && base != "sp_gscore" && base != "sp_picks" && base != "sp_picks_ref" && base != "sp_mask" && !(ds->with_k && base == "sp_k")) {
        return false;
    }
    if (!ds->layers.count(atoi(dash + 1))) {
        return false;
    }
    if (ask) {
        return true;
    }
    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    const std::string path = ds->dir + "/" + ds->prefix + "_" + name + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    const int32_t hdr[5] = { (int32_t) t->type, (int32_t) t->ne[0], (int32_t) t->ne[1], (int32_t) t->ne[2], (int32_t) t->ne[3] };
    fwrite(hdr, sizeof(hdr), 1, f);
    // a non-contiguous view is dumped as the bytes it spans, with its strides
    const int64_t nb[4] = { (int64_t) t->nb[0], (int64_t) t->nb[1], (int64_t) t->nb[2], (int64_t) t->nb[3] };
    fwrite(nb, sizeof(nb), 1, f);
    fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);
    return true;
}

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        const size_t b = s.find(sep, a);
        out.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) {
            break;
        }
        a = b + 1;
    }
    return out;
}

static void log_softmax(const float * logits, int n, std::vector<double> & out) {
    out.resize(n);
    double mx = -INFINITY;
    for (int i = 0; i < n; ++i) {
        mx = std::max(mx, (double) logits[i]);
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::exp((double) logits[i] - mx);
    }
    const double lse = mx + std::log(sum);
    for (int i = 0; i < n; ++i) {
        out[i] = (double) logits[i] - lse;
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string tokens_path;
    std::string configs_arg = "dense";
    std::string out_path;
    dump_state  ds;
    std::string dump_cfgs;
    std::string ref_path;
    std::string ret_path;
    std::string gen_path;
    std::map<std::string, std::string> save_paths;
    ggml_type type_kv = GGML_TYPE_F16;
    int T       = 8192;
    int n_books = 1;
    int n_tail  = 256;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") { model_path = next(); }
        else if (a == "--tokens") { tokens_path = next(); }
        else if (a == "--T") { T = std::stoi(next()); }
        else if (a == "--n-books") { n_books = std::stoi(next()); }
        else if (a == "--n-tail") { n_tail = std::stoi(next()); }
        else if (a == "--configs") { configs_arg = next(); }
        else if (a == "--out") { out_path = next(); }
        else if (a == "--dump-dir") { ds.dir = next(); }
        else if (a == "--dump-layers") { for (auto & x : split(next(), ',')) ds.layers.insert(std::stoi(x)); }
        else if (a == "--dump-steps") { for (auto & x : split(next(), ',')) ds.steps.insert(std::stoi(x)); }
        else if (a == "--dump-configs") { dump_cfgs = next(); }
        else if (a == "--ref-logits") { ref_path = next(); }
        else if (a == "--ret") { ret_path = next(); }
        else if (a == "--gen") { gen_path = next(); }
        else if (a == "--kv-type") {
            const std::string t = next();
            if (t == "f16") type_kv = GGML_TYPE_F16;
            else if (t == "q8_0") type_kv = GGML_TYPE_Q8_0;
            else { fprintf(stderr, "unsupported --kv-type %s\n", t.c_str()); return 1; }
        }
        else if (a == "--save-logits") { auto p = split(next(), '='); save_paths[p[0]] = p[1]; }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    // configs: dense first, then score:budget
    std::vector<tail_config> configs;
    for (auto & c : split(configs_arg, ',')) {
        tail_config tc;
        tc.name = c;
        if (c != "dense") {
            auto p = split(c, ':');
            tc.score  = p[0];
            tc.budget = std::stoi(p[1]);
            if (p.size() > 2) {
                if (p[2] != "f0" && p[2] != "f1" && p[2] != "f2" && p[2] != "f3") { fprintf(stderr, "bad config %s\n", c.c_str()); return 1; }
                tc.fused = p[2].substr(1);
            }
        }
        configs.push_back(tc);
    }
    if (configs.empty() || (ref_path.empty() && configs[0].name != "dense")) {
        fprintf(stderr, "the first config must be dense unless --ref-logits is given\n");
        return 1;
    }
    std::set<std::string> dump_set;
    for (auto & c : split(dump_cfgs, ',')) {
        if (!c.empty()) dump_set.insert(c);
    }

    // --ret: retrieval trials [n | per trial: task, book, depth_pct, len, n_ans, ids[len]]
    struct ret_trial { int32_t task, book, depth, n_ans; std::vector<int32_t> ids; };
    std::vector<ret_trial> trials;
    // --gen: prompts [n | per prompt: id, max_new, len, ids[len]], greedy generation under every config
    struct gen_prompt { int32_t id, max_new; std::vector<int32_t> ids; };
    std::vector<gen_prompt> prompts;
    if (!gen_path.empty()) {
        FILE * f = fopen(gen_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", gen_path.c_str()); return 1; }
        int32_t n = 0;
        if (fread(&n, sizeof(n), 1, f) != 1) { fprintf(stderr, "bad --gen file\n"); return 1; }
        int max_len = 0;
        for (int i = 0; i < n; ++i) {
            int32_t h[3];
            if (fread(h, sizeof(int32_t), 3, f) != 3) { fprintf(stderr, "bad --gen file\n"); return 1; }
            gen_prompt gp{h[0], h[1], std::vector<int32_t>(h[2])};
            if (fread(gp.ids.data(), sizeof(int32_t), h[2], f) != (size_t) h[2]) { fprintf(stderr, "bad --gen file\n"); return 1; }
            max_len = std::max(max_len, h[2] + h[1]);
            prompts.push_back(std::move(gp));
        }
        fclose(f);
        T = (max_len + 255)/256*256;
    }
    if (!ret_path.empty()) {
        FILE * f = fopen(ret_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", ret_path.c_str()); return 1; }
        int32_t n = 0;
        if (fread(&n, sizeof(n), 1, f) != 1) { fprintf(stderr, "bad --ret file\n"); return 1; }
        int max_len = 0;
        for (int i = 0; i < n; ++i) {
            int32_t h[5];
            if (fread(h, sizeof(int32_t), 5, f) != 5) { fprintf(stderr, "bad --ret file\n"); return 1; }
            ret_trial tr{h[0], h[1], h[2], h[4], std::vector<int32_t>(h[3])};
            if (fread(tr.ids.data(), sizeof(int32_t), h[3], f) != (size_t) h[3]) { fprintf(stderr, "bad --ret file\n"); return 1; }
            max_len = std::max(max_len, h[3]);
            trials.push_back(std::move(tr));
        }
        fclose(f);
        T = (max_len + 255)/256*256;
    }

    std::vector<int32_t> all;
    if (ret_path.empty() && gen_path.empty()) {
        FILE * f = fopen(tokens_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", tokens_path.c_str()); return 1; }
        int32_t x;
        while (fread(&x, sizeof(x), 1, f) == 1) all.push_back(x);
        fclose(f);
    }
    const int book_len = (int) all.size()/n_books;
    if (ret_path.empty() && gen_path.empty() && book_len < T) {
        fprintf(stderr, "books have %d tokens < T = %d\n", book_len, T);
        return 1;
    }

    // the tool compares configs at any cache length
    setenv("LLAMA_SPARSE_DECODE_MIN_CTX", "0", 0);

    // set while the context is created so the graph budget covers the sparse ops
    for (auto & c : configs) {
        if (c.budget > 0) {
            setenv("LLAMA_SPARSE_DECODE", std::to_string(c.budget).c_str(), 1);
            setenv("LLAMA_SPARSE_DECODE_SCORE", c.score.c_str(), 1);
            if (!c.fused.empty()) {
                setenv("LLAMA_SPARSE_DECODE_FUSED", c.fused.c_str(), 1);
            }
            break;
        }
    }
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = T;
    cp.n_batch         = 2048;
    cp.n_ubatch        = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k          = type_kv;
    cp.type_v          = type_kv;
    cp.no_perf         = true;
    if (!ds.dir.empty()) {
        cp.cb_eval           = dump_cb;
        cp.cb_eval_user_data = &ds;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);
    unsetenv("LLAMA_SPARSE_DECODE");

    FILE * out = out_path.empty() ? stdout : fopen(out_path.c_str(), "w");
    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    const int P = T - n_tail;
    std::vector<std::vector<float>> ref(n_tail, std::vector<float>(n_vocab));
    FILE * fref  = ref_path.empty()  ? nullptr : fopen(ref_path.c_str(), "rb");
    std::map<std::string, FILE *> fsave;
    for (auto & [c, path] : save_paths) {
        fsave[c] = fopen(path.c_str(), "wb");
    }
    std::vector<double> lp, lq;

    const llama_token eos = llama_vocab_eos(vocab);
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        const gen_prompt & gp = prompts[pi];
        const int Pt = (int) gp.ids.size() - 1;

        unsetenv("LLAMA_SPARSE_DECODE");
        llama_memory_clear(mem, true);
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < Pt; i += cp.n_batch) {
            const int n = std::min<int>(cp.n_batch, Pt - i);
            llama_batch_ext_clear(batch);
            for (int j = 0; j < n; ++j) {
                const int32_t idx = llama_batch_ext_add_token(batch, 0, gp.ids[i + j]);
                const llama_pos pos = i + j;
                llama_batch_ext_set_pos(batch, idx, &pos);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { fprintf(stderr, "prefill failed\n"); return 1; }
        }
        const double t_pre = (ggml_time_us() - t0)/1e6;

        for (const auto & c : configs) {
            if (c.budget > 0) {
                setenv("LLAMA_SPARSE_DECODE", std::to_string(c.budget).c_str(), 1);
                setenv("LLAMA_SPARSE_DECODE_SCORE", c.score.c_str(), 1);
                if (!c.fused.empty()) {
                    setenv("LLAMA_SPARSE_DECODE_FUSED", c.fused.c_str(), 1);
                }
            } else {
                unsetenv("LLAMA_SPARSE_DECODE");
            }
            llama_memory_seq_rm(mem, 0, Pt, -1);

            std::vector<int32_t> outv;
            int32_t tok = gp.ids[Pt];
            const int64_t t1 = ggml_time_us();
            for (int s = 0; s < gp.max_new; ++s) {
                const llama_pos pos = Pt + s;
                ds.active = dump_set.count(c.name) && ds.steps.count(s);
                ds.with_k = false;
                ds.prefix = "p" + std::to_string(gp.id) + "_" + c.name + "_s" + std::to_string(s);
                std::replace(ds.prefix.begin(), ds.prefix.end(), ':', '-');
                llama_batch_ext_clear(batch);
                const int32_t idx = llama_batch_ext_add_token(batch, 0, tok);
                llama_batch_ext_set_pos(batch, idx, &pos);
                llama_batch_ext_set_output_logits(batch, idx, true);
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
                const float * logits = llama_get_logits_ith(ctx, -1);
                int am = 0;
                for (int v = 1; v < n_vocab; ++v) {
                    if (logits[v] > logits[am]) am = v;
                }
                if (am == eos) break;
                outv.push_back(am);
                tok = am;
            }
            ds.active = false;
            const double t_gen = (ggml_time_us() - t1)/1e6;
            fprintf(out, "{\"id\": %d, \"config\": \"%s\", \"len\": %d, \"prefill_s\": %.3f, \"gen_s\": %.3f, \"out\": [", gp.id, c.name.c_str(), Pt + 1, t_pre, t_gen);
            for (size_t i = 0; i < outv.size(); ++i) fprintf(out, "%s%d", i ? ", " : "", outv[i]);
            fprintf(out, "]}\n");
            fflush(out);
        }
        fprintf(stderr, "prompt %zu (%d tokens): prefill %.1f s\n", pi, Pt + 1, t_pre);
    }

    for (size_t t = 0; t < trials.size(); ++t) {
        const ret_trial & tr = trials[t];
        const int L  = (int) tr.ids.size();
        const int Pt = L - 1 - tr.n_ans;

        unsetenv("LLAMA_SPARSE_DECODE");
        llama_memory_clear(mem, true);
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < Pt; i += cp.n_batch) {
            const int n = std::min<int>(cp.n_batch, Pt - i);
            llama_batch_ext_clear(batch);
            for (int j = 0; j < n; ++j) {
                const int32_t idx = llama_batch_ext_add_token(batch, 0, tr.ids[i + j]);
                const llama_pos pos = i + j;
                llama_batch_ext_set_pos(batch, idx, &pos);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { fprintf(stderr, "prefill failed\n"); return 1; }
        }
        fprintf(stderr, "trial %zu: prefill %d tokens in %.1f s\n", t, Pt, (ggml_time_us() - t0)/1e6);
        // recurrent / hybrid memories cannot drop the answer tokens of the previous config: restore the prompt state
        const bool restore = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
        std::vector<uint8_t> state;
        if (restore) {
            state.resize(llama_state_seq_get_size(ctx, 0));
            state.resize(llama_state_seq_get_data(ctx, state.data(), state.size(), 0));
        }

        for (const auto & c : configs) {
            if (c.budget > 0) {
                setenv("LLAMA_SPARSE_DECODE", std::to_string(c.budget).c_str(), 1);
                setenv("LLAMA_SPARSE_DECODE_SCORE", c.score.c_str(), 1);
                if (!c.fused.empty()) {
                    setenv("LLAMA_SPARSE_DECODE_FUSED", c.fused.c_str(), 1);
                }
            } else {
                unsetenv("LLAMA_SPARSE_DECODE");
            }
            if (restore) {
                llama_memory_seq_rm(mem, 0, -1, -1);
                if (llama_state_seq_set_data(ctx, state.data(), state.size(), 0) != state.size()) {
                    fprintf(stderr, "state restore failed\n");
                    return 1;
                }
            } else if (!llama_memory_seq_rm(mem, 0, Pt, -1)) {
                fprintf(stderr, "seq_rm failed\n");
                return 1;
            }

            int correct = 1;
            double logp = 0.0;
            double reads = 0.0;
            for (int s = 0; s < tr.n_ans; ++s) {
                const llama_pos pos = Pt + s;
                llama_batch_ext_clear(batch);
                const int32_t idx = llama_batch_ext_add_token(batch, 0, tr.ids[pos]);
                llama_batch_ext_set_pos(batch, idx, &pos);
                llama_batch_ext_set_output_logits(batch, idx, true);
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
                const float * logits = llama_get_logits_ith(ctx, -1);
                log_softmax(logits, n_vocab, lq);
                int am = 0;
                for (int v = 1; v < n_vocab; ++v) {
                    if (lq[v] > lq[am]) am = v;
                }
                const int32_t target = tr.ids[pos + 1];
                if (getenv("SPARSE_TAIL_VERBOSE")) {
                    fprintf(stderr, "  %s s%d target %d lp %.3f argmax %d lp %.3f\n", c.name.c_str(), s, target, lq[target], am, lq[am]);
                }
                correct &= am == target;
                logp += lq[target];
                if (c.budget > 0) {
                    const int64_t win_blk = std::max<int64_t>(0, (pos - 63)/64);
                    const int64_t n_top   = (c.budget - 80)/64;
                    const int64_t n_elig  = std::max<int64_t>(0, win_blk - 1);
                    reads += 64 + 64*std::min(n_top, n_elig) + (pos - std::max<int64_t>(64, win_blk*64) + 1);
                } else {
                    reads += pos + 1;
                }
            }
            fprintf(out, "{\"trial\": %zu, \"task\": %d, \"book\": %d, \"depth_pct\": %d, \"len\": %d, \"config\": \"%s\", \"correct\": %d, \"logp\": %.6f, \"reads\": %.1f}\n",
                    t, tr.task, tr.book, tr.depth, L, c.name.c_str(), correct, logp, reads/tr.n_ans);
            fflush(out);
            fprintf(stderr, "trial %zu task %d depth %d %-14s correct=%d logp=%.3f\n", t, tr.task, tr.depth, c.name.c_str(), correct, logp);
        }
    }

    for (int b = 0; ret_path.empty() && gen_path.empty() && b < n_books; ++b) {
        const int32_t * toks = all.data() + (size_t) b*book_len;

        unsetenv("LLAMA_SPARSE_DECODE");
        llama_memory_clear(mem, true);
        const int64_t t0 = ggml_time_us();
        for (int i = 0; i < P; i += cp.n_batch) {
            const int n = std::min<int>(cp.n_batch, P - i);
            llama_batch_ext_clear(batch);
            for (int j = 0; j < n; ++j) {
                const int32_t idx = llama_batch_ext_add_token(batch, 0, toks[i + j]);
                const llama_pos pos = i + j;
                llama_batch_ext_set_pos(batch, idx, &pos);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { fprintf(stderr, "prefill failed\n"); return 1; }
        }
        fprintf(stderr, "book %d: prefill %d tokens in %.1f s\n", b, P, (ggml_time_us() - t0)/1e6);
        // a recurrent or hybrid memory cannot drop the tail of a sequence (seq_rm fails), so every config starts
        // from a copy of the state after the prefill
        const bool restore = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
        std::vector<uint8_t> state;
        if (restore) {
            state.resize(llama_state_seq_get_size(ctx, 0));
            state.resize(llama_state_seq_get_data(ctx, state.data(), state.size(), 0));
            fprintf(stderr, "book %d: saved the prefix state, %.1f MiB\n", b, state.size()/1048576.0);
        }
        if (fref) {
            for (int s = 0; s < n_tail; ++s) {
                if (fread(ref[s].data(), sizeof(float), n_vocab, fref) != (size_t) n_vocab) { fprintf(stderr, "short --ref-logits file\n"); return 1; }
            }
        }

        for (size_t ci = 0; ci < configs.size(); ++ci) {
            const auto & c = configs[ci];
            if (c.budget > 0) {
                setenv("LLAMA_SPARSE_DECODE", std::to_string(c.budget).c_str(), 1);
                setenv("LLAMA_SPARSE_DECODE_SCORE", c.score.c_str(), 1);
                if (!c.fused.empty()) {
                    setenv("LLAMA_SPARSE_DECODE_FUSED", c.fused.c_str(), 1);
                }
            } else {
                unsetenv("LLAMA_SPARSE_DECODE");
            }
            if (restore) {
                llama_memory_seq_rm(mem, 0, -1, -1);
                if (llama_state_seq_set_data(ctx, state.data(), state.size(), 0) != state.size()) {
                    fprintf(stderr, "state restore failed\n");
                    return 1;
                }
            } else if (!llama_memory_seq_rm(mem, 0, P, -1)) {
                fprintf(stderr, "seq_rm failed\n");
                return 1;
            }

            std::vector<double> kl(n_tail);
            int    flips = 0;
            double reads = 0.0;
            double max_diff = 0.0;
            const int64_t t1 = ggml_time_us();
            for (int s = 0; s < n_tail; ++s) {
                const llama_pos pos = P + s;
                ds.active = b == 0 && dump_set.count(c.name) && ds.steps.count(s);
                ds.with_k = ds.active;
                ds.prefix = "b0_" + c.name + "_s" + std::to_string(s);
                std::replace(ds.prefix.begin(), ds.prefix.end(), ':', '-');

                llama_batch_ext_clear(batch);
                const int32_t idx = llama_batch_ext_add_token(batch, 0, toks[pos]);
                llama_batch_ext_set_pos(batch, idx, &pos);
                llama_batch_ext_set_output_logits(batch, idx, true);
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
                const float * logits = llama_get_logits_ith(ctx, -1);
                if (fsave.count(c.name)) {
                    fwrite(logits, sizeof(float), n_vocab, fsave[c.name]);
                }

                if (c.budget == 0) {
                    ref[s].assign(logits, logits + n_vocab);
                    reads += pos + 1;
                    continue;
                }
                log_softmax(ref[s].data(), n_vocab, lp);
                log_softmax(logits, n_vocab, lq);
                double k = 0.0;
                int am_p = 0, am_q = 0;
                for (int v = 0; v < n_vocab; ++v) {
                    max_diff = std::max(max_diff, (double) std::fabs(logits[v] - ref[s][v]));
                    k += std::exp(lp[v])*(lp[v] - lq[v]);
                    if (lp[v] > lp[am_p]) am_p = v;
                    if (lq[v] > lq[am_q]) am_q = v;
                }
                kl[s] = k;
                flips += am_p != am_q;
                const int64_t win_blk = std::max<int64_t>(0, (pos - 63)/64);
                const int64_t n_top   = (c.budget - 80)/64;
                const int64_t n_elig  = std::max<int64_t>(0, win_blk - 1);
                reads += 64 + 64*std::min(n_top, n_elig) + (pos - std::max<int64_t>(64, win_blk*64) + 1);
            }
            ds.active = false;
            const double sec = (ggml_time_us() - t1)/1e6;

            std::vector<double> srt(kl);
            std::sort(srt.begin(), srt.end());
            double mean = 0.0;
            for (double x : kl) mean += x;
            mean /= n_tail;
            fprintf(out, "{\"book\": %d, \"T\": %d, \"config\": \"%s\", \"kl_mean\": %.9g, \"kl_p99\": %.9g, \"kl_max\": %.9g, \"flip\": %.6g, \"reads\": %.3f, \"max_abs_logit_diff\": %.9g, \"tail_s\": %.3f, \"kl_rows\": [",
                    b, T, c.name.c_str(), mean, srt[(size_t) std::floor(0.99*(n_tail - 1))], srt.back(), (double) flips/n_tail, reads/n_tail, max_diff, sec);
            for (int s = 0; s < n_tail; ++s) fprintf(out, "%s%.6g", s ? ", " : "", kl[s]);
            fprintf(out, "]}\n");
            fflush(out);
            fprintf(stderr, "book %d %-12s kl=%.3g flip=%.3f reads=%.0f (%.1f s)\n", b, c.name.c_str(), mean, (double) flips/n_tail, reads/n_tail, sec);
        }
    }

    if (fref) fclose(fref);
    for (auto & [c, f] : fsave) fclose(f);
    llama_batch_ext_free(batch);
    if (out != stdout) fclose(out);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
