// The device expert cache (pools of cached experts, see ggml_backend_expert_cache_create_t) against the same graph on
//   whole weights: the MoE FFN of one Qwen3.5-MoE layer in the order the model builds it (optionally with its sigmoid-gated
//   shared expert) over pools that evict, with the gate pool's LRU history apart from the others so that its slots
//   differ from theirs. The fused paths must read each pool through its own remapped ids.
//   Runs on the first device of the first backend that has an expert cache; skipped without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct moe_weight {
    ggml_tensor * host = nullptr; // the weight in host memory the cache reads from
    ggml_tensor * full = nullptr; // a device copy, the reference
    ggml_tensor * pool = nullptr; // cap experts on the device, its extra the cache
    void        * cache = nullptr;
};

struct test_case {
    ggml_type type;
    int64_t   n_embd;
    int64_t   n_ff;
    int       n_expert;
    int       n_used;
    int       cap;
    int       n_tokens;
    bool      shexp;
};

struct layer {
    test_case     tc;
    moe_weight    gate, up, down;
    ggml_tensor * gate_sh = nullptr;
    ggml_tensor * up_sh   = nullptr;
    ggml_tensor * down_sh = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * g       = nullptr;
    bool          use_pool = false;
    bool          gate_only = false;
};

void fill_quantized(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(ggml_nelements(t));
    for (float & v : f) {
        v = u(rng);
    }
    std::vector<uint8_t> q(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

void fill_f32(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(ggml_nelements(t));
    for (float & v : f) {
        v = u(rng);
    }
    ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
}

// distinct experts per token
std::vector<int32_t> random_ids(std::mt19937 & rng, int n_expert, int n_used, int n_tokens) {
    std::vector<int32_t> ids;
    std::vector<int32_t> perm(n_expert);
    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_expert; e++) {
            perm[e] = e;
        }
        std::shuffle(perm.begin(), perm.end(), rng);
        ids.insert(ids.end(), perm.begin(), perm.begin() + n_used);
    }
    return ids;
}

// computes the layer (or the gate pool alone, so that its LRU moves on without the others) for x and ids
std::vector<float> run(ggml_backend_t backend, const layer & l, const std::vector<float> & x, const std::vector<int32_t> & ids) {
    const test_case & tc = l.tc;
    ggml_init_params params = { ggml_tensor_overhead()*256 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph  * gf  = ggml_new_graph(ctx);
    ggml_tensor  * tx   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.n_embd, tc.n_tokens);
    ggml_tensor  * tids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, tc.n_used, tc.n_tokens);
    ggml_tensor  * xr   = ggml_reshape_3d(ctx, tx, tc.n_embd, 1, tc.n_tokens);

    ggml_tensor * out;
    if (l.gate_only) {
        out = ggml_mul_mat_id(ctx, l.gate.pool, xr, tids);
    } else {
        ggml_tensor * act_sh = nullptr;
        if (tc.shexp) {
            act_sh = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, l.gate_sh, tx), ggml_mul_mat(ctx, l.up_sh, tx));
            ggml_build_forward_expand(gf, act_sh);
        }
        ggml_tensor * g = ggml_mul_mat_id(ctx, l.use_pool ? l.gate.pool : l.gate.full, xr, tids);
        ggml_tensor * u = ggml_mul_mat_id(ctx, l.use_pool ? l.up.pool   : l.up.full,   xr, tids);
        ggml_tensor * h = ggml_swiglu_split(ctx, g, u);
        ggml_tensor * experts  = ggml_mul_mat_id(ctx, l.use_pool ? l.down.pool : l.down.full, h, tids);
        ggml_tensor * weighted = ggml_mul(ctx, experts, l.weights);
        ggml_tensor * sum = nullptr;
        for (int e = 0; e < tc.n_used; e++) {
            ggml_tensor * v = ggml_view_2d(ctx, weighted, tc.n_embd, tc.n_tokens, weighted->nb[2], e*weighted->nb[1]);
            sum = sum ? ggml_add(ctx, sum, v) : v;
        }
        out = sum;
        if (tc.shexp) {
            out = ggml_add(ctx, sum, ggml_mul(ctx, ggml_mul_mat(ctx, l.down_sh, act_sh), ggml_sigmoid(ctx, l.g)));
        }
    }
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    ggml_backend_tensor_set(tx, x.data(), 0, ggml_nbytes(tx));
    ggml_backend_tensor_set(tids, ids.data(), 0, ggml_nbytes(tids));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return res;
}

bool test_one(ggml_backend_reg_t reg, ggml_backend_dev_t dev, ggml_backend_t backend, const test_case & tc, std::mt19937 & rng) {
    auto cache_create = (ggml_backend_expert_cache_create_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_expert_cache_create");
    auto cache_free   = (ggml_backend_expert_cache_free_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_expert_cache_free");

    ggml_init_params params = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_context * ctx_host = ggml_init(params);
    ggml_context * ctx_full = ggml_init(params);
    ggml_context * ctx_pool = ggml_init(params);

    layer l;
    l.tc = tc;
    moe_weight * ws[3] = { &l.gate, &l.up, &l.down };
    for (int k = 0; k < 3; k++) {
        const int64_t ne0 = k < 2 ? tc.n_embd : tc.n_ff;
        const int64_t ne1 = k < 2 ? tc.n_ff   : tc.n_embd;
        ws[k]->host = ggml_new_tensor_3d(ctx_host, tc.type, ne0, ne1, tc.n_expert);
        ws[k]->full = ggml_new_tensor_3d(ctx_full, tc.type, ne0, ne1, tc.n_expert);
        ws[k]->pool = ggml_new_tensor_3d(ctx_pool, tc.type, ne0, ne1, tc.cap);
    }
    l.gate_sh = ggml_new_tensor_2d(ctx_full, tc.type, tc.n_embd, tc.n_ff);
    l.up_sh   = ggml_new_tensor_2d(ctx_full, tc.type, tc.n_embd, tc.n_ff);
    l.down_sh = ggml_new_tensor_2d(ctx_full, tc.type, tc.n_ff, tc.n_embd);
    l.weights = ggml_new_tensor_3d(ctx_full, GGML_TYPE_F32, 1, tc.n_used, tc.n_tokens);
    l.g       = ggml_new_tensor_2d(ctx_full, GGML_TYPE_F32, 1, tc.n_tokens);

    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    ggml_backend_buffer_t buf_host = ggml_backend_alloc_ctx_tensors_from_buft(ctx_host,
        host_buft ? host_buft : ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t buf_full = ggml_backend_alloc_ctx_tensors(ctx_full, backend);
    ggml_backend_buffer_t buf_pool = ggml_backend_alloc_ctx_tensors(ctx_pool, backend);
    ggml_backend_buffer_clear(buf_pool, 0);
    for (ggml_tensor * t : { l.gate_sh, l.up_sh, l.down_sh }) {
        fill_quantized(t, rng);
    }
    fill_f32(l.weights, rng);
    fill_f32(l.g, rng);

    bool ok = true;
    for (int k = 0; k < 3 && ok; k++) {
        moe_weight & w = *ws[k];
        fill_quantized(w.host, rng);
        std::vector<uint8_t> tmp(ggml_nbytes(w.host));
        ggml_backend_tensor_get(w.host, tmp.data(), 0, tmp.size());
        ggml_backend_tensor_set(w.full, tmp.data(), 0, tmp.size());
        // the whole weight on one device: one chunk per expert
        w.cache = cache_create(dev, ggml_backend_buffer_get_base(buf_host), ggml_backend_buffer_get_size(buf_host),
            w.host->data, tc.n_expert, tc.cap, w.host->nb[2], w.host->nb[2], w.host->nb[2], 0, 1, tc.cap);
        if (w.cache == nullptr) {
            printf("  could not create the cache\n");
            ok = false;
            break;
        }
        w.pool->extra = w.cache;
    }

    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    double max_err = 0.0;
    for (int it = 0; ok && it < 24; it++) {
        std::vector<float> x(tc.n_embd*tc.n_tokens);
        for (float & v : x) {
            v = u(rng);
        }
        if (it % 3 == 1) {
            l.gate_only = true;
            run(backend, l, x, random_ids(rng, tc.n_expert, tc.n_used, tc.n_tokens));
            l.gate_only = false;
        }
        const std::vector<int32_t> ids = random_ids(rng, tc.n_expert, tc.n_used, tc.n_tokens);
        l.use_pool = true;
        const std::vector<float> got = run(backend, l, x, ids);
        l.use_pool = false;
        const std::vector<float> ref = run(backend, l, x, ids);

        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < ref.size(); i++) {
            if (!std::isfinite(got[i])) {
                num = INFINITY;
                break;
            }
            num += (got[i] - ref[i])*(got[i] - ref[i]);
            den += ref[i]*ref[i];
        }
        const double err = den > 0.0 ? num/den : num;
        max_err = std::max(max_err, err);
        if (!(err <= 1e-12)) {
            printf("  iteration %d: NMSE %g\n", it, err);
            ok = false;
        }
    }
    printf("  %-6s embd %4lld ff %4lld experts %3d used %d cap %2d tokens %d shexp %d: max NMSE %g %s\n",
        ggml_type_name(tc.type), (long long) tc.n_embd, (long long) tc.n_ff, tc.n_expert, tc.n_used, tc.cap, tc.n_tokens,
        tc.shexp ? 1 : 0, max_err, ok ? "OK" : "FAIL");

    for (moe_weight * w : ws) {
        if (w->cache) {
            cache_free(w->cache);
        }
    }
    ggml_backend_buffer_free(buf_pool);
    ggml_backend_buffer_free(buf_full);
    ggml_backend_buffer_free(buf_host);
    ggml_free(ctx_pool);
    ggml_free(ctx_full);
    ggml_free(ctx_host);
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();

    for (size_t r = 0; r < ggml_backend_reg_count(); r++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(r);
        if (ggml_backend_reg_get_proc_address(reg, "ggml_backend_expert_cache_create") == nullptr ||
                ggml_backend_reg_dev_count(reg) == 0) {
            continue;
        }
        ggml_backend_dev_t dev     = ggml_backend_reg_dev_get(reg, 0);
        ggml_backend_t     backend = ggml_backend_dev_init(dev, nullptr);
        printf("%s: %s\n", ggml_backend_reg_name(reg), ggml_backend_dev_description(dev));

        std::mt19937 rng(1234);
        const test_case cases[] = {
            { GGML_TYPE_Q8_0,  512, 256, 64, 8, 24, 1, false },
            { GGML_TYPE_Q8_0,  512, 256, 64, 8, 24, 1, true  },
            { GGML_TYPE_Q8_0,  512, 256, 64, 8, 24, 3, false },
            { GGML_TYPE_Q8_0,  512, 256, 64, 8, 24, 3, true  },
            { GGML_TYPE_Q8_0,  512, 128, 64, 8, 24, 1, true  }, // short rows: the shared expert in the down launch
            { GGML_TYPE_MXFP4, 512, 256, 64, 6, 16, 1, false },
            { GGML_TYPE_MXFP4, 512, 256, 64, 6, 16, 1, true  },
            { GGML_TYPE_MXFP4, 512, 256, 64, 6, 16, 2, false },
            { GGML_TYPE_Q4_K,  512, 256, 32, 4, 12, 1, true  },
            { GGML_TYPE_Q2_K,  512, 256, 32, 4, 12, 1, false },
        };
        bool ok = true;
        for (const test_case & tc : cases) {
            ok = test_one(reg, dev, backend, tc, rng) && ok;
        }
        ggml_backend_free(backend);
        printf("%s\n", ok ? "all passed" : "FAILED");
        return ok ? 0 : 1;
    }
    printf("no backend with an expert cache, skipped\n");
    return 0;
}
