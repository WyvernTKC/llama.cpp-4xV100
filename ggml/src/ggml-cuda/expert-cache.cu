#include "expert-cache.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <vector>

#define GGML_CUDA_EXPERT_CACHE_MAGIC      0x6578706572746373ull
#define GGML_CUDA_EXPERT_CACHE_WARM_MAGIC 0x6578707761726d73ull

struct ggml_cuda_expert_cache;

// In the extra of a whole staged copy of a pooled weight (this device's slice of every expert): its MUL_MAT_ID also
//   brings the experts the ubatch routed to most into the pool, see expert_cache_warm_plan
struct expert_cache_warm_tag {
    uint64_t                 magic   = GGML_CUDA_EXPERT_CACHE_WARM_MAGIC;
    ggml_cuda_expert_cache * cache   = nullptr;
    char *                   pool    = nullptr;
    int                      n_evict = 0;
};

struct ggml_cuda_expert_cache_dev {
    const char * host;         // device pointer to expert 0 in host memory
    size_t       expert_bytes; // host stride between experts
    size_t       chunk_full;   // bytes of one chunk of an expert in the full tensor
    size_t       chunk_dev;    // this device's share of a chunk
    size_t       offset_dev;   // where that share starts inside the chunk
    size_t       slot_bytes;   // pool stride between slots, n_chunks*chunk_dev
    int          n_chunks;
    int          n_expert;
    int          cap;
    int          max_ids;
    int          cpu_slot;  // what the ids of an expert the CPU computes read: cap, the zeroed slot past the pool, when the
                            //   pool has one (a whole slot of headroom: split by rows), else 0 (its input must be zero)

    int32_t  * slot_of;     // [n_expert] pool slot of an expert, -1 when absent
    int32_t  * expert_of;   // [cap]
    uint32_t * last_used;   // [cap]
    uint32_t * clock;       // [1]
    int32_t  * remap;       // [max_ids]
    int32_t  * miss_expert; // [max_ids]
    int32_t  * miss_slot;   // [max_ids]
    int32_t  * n_miss;      // [1]
    uint32_t * stats;       // [7] calls, misses, warms, warm misses, CPU experts, one-row calls, their misses
                            //   - GGML_META_EXPERT_CACHE_STATS

    // GGML_META_EXPERT_CACHE_POLICY=fr, see expert_cache_policy; nullptr with the LRU
    float    * freq;        // [n_expert] routings, decayed with a half-life in tokens up to tok_of
    uint32_t * tok_of;      // [n_expert] the token count when freq was last brought up to date
    uint32_t * win_of;      // [cap] the window (plan or warm call) that last used each slot, then the window count and
                            //   the token count
    float      inv_hl;      // 1 / half-life in tokens
    float      inv_tau;     // 1 / tau in windows, 0: frequency alone
};

struct ggml_cuda_expert_cache {
    uint64_t                   magic = GGML_CUDA_EXPERT_CACHE_MAGIC;
    int                        device;
    int                        align;  // largest of 16/8/4/2/1 that divides every address and stride the copy uses
    bool                       covering; // align < 4, copied in the 8-byte words that cover each chunk
    ggml_cuda_expert_cache_dev d;
    void *                     state = nullptr;
    expert_cache_warm_tag      warm;
    const char *               host_cpu = nullptr; // host address of expert 0, the same for the caches of every device
};

// the live caches, so the CPU can find every device's slice of a weight (ggml_cuda_expert_cache_cpu_mat)
static std::vector<const ggml_cuda_expert_cache *> g_caches;
static std::mutex                                  g_caches_mutex;

// The caches of one launch: the experts of a MoE layer routed by the same ids (gate, up, down), each with its own
//   LRU, planned by a block each and copied by one grid.
#define GGML_CUDA_EXPERT_CACHE_GROUP_MAX 3
struct ggml_cuda_expert_cache_group {
    ggml_cuda_expert_cache_dev d[GGML_CUDA_EXPERT_CACHE_GROUP_MAX];
    char *                     pool[GGML_CUDA_EXPERT_CACHE_GROUP_MAX];
    int                        align[GGML_CUDA_EXPERT_CACHE_GROUP_MAX];
    int                        n;
};

// fast: at most this many ids, and no more than the capacity of any cache of the group, see expert_cache_plan
#define GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS 64
#define GGML_CUDA_EXPERT_CACHE_PLAN_FAST_CAP 4096
#define GGML_CUDA_EXPERT_CACHE_PLAN_FAST_CAP_FR 2048 // a third array of cap in shared memory

// Which resident a miss replaces. Every routed expert has to be in the pool, so a policy only chooses victims.
//   lru (default): the least recently used.
//   fr: the lowest freq*2^(-age/tau), age = windows (plan or warm calls of this pool) since the slot was last used,
//     freq = routings decayed with a half-life in tokens, ties to the least recently used; the experts of the
//     current window are never victims. Frequency x recency from capy-flash (github.com/ValerioDolci/capy-flash, T61);
//     a warm (wide ubatch) adds its routings weighted by their mean recency (its T69) and keeps LRU victims.
//   GGML_META_EXPERT_CACHE_POLICY=lru|fr, GGML_META_EXPERT_CACHE_FR_TAU=<windows> (0: frequency alone),
//   GGML_META_EXPERT_CACHE_FR_HALF_LIFE=<tokens>.
struct expert_cache_policy {
    bool  fr;
    float half_life;
    float tau;
};

static expert_cache_policy expert_cache_policy_get() {
    static const expert_cache_policy p = [] {
        expert_cache_policy r = {false, 128.0f, 32.0f};
        const char * pol = getenv("GGML_META_EXPERT_CACHE_POLICY");
        r.fr = pol != nullptr && strcmp(pol, "fr") == 0;
        const char * hl  = getenv("GGML_META_EXPERT_CACHE_FR_HALF_LIFE");
        const char * tau = getenv("GGML_META_EXPERT_CACHE_FR_TAU");
        if (hl != nullptr && atof(hl) > 0.0) {
            r.half_life = (float) atof(hl);
        }
        if (tau != nullptr && atof(tau) >= 0.0) {
            r.tau = (float) atof(tau);
        }
        return r;
    }();
    return p;
}

// The fr key of a slot holding expert e, last used in window win_s, at token count tok and window w: an empty slot
//   first. The frequency decays lazily, from the token count of its last update.
static __device__ __forceinline__ float expert_cache_fr_key(const ggml_cuda_expert_cache_dev & d, const int e,
        const uint32_t win_s, const uint32_t tok, const uint32_t w) {
    if (e < 0) {
        return -1.0f;
    }
    return d.freq[e]*exp2f(-((float) (tok - d.tok_of[e])*d.inv_hl + (float) (w - win_s)*d.inv_tau));
}

// (key, last_used, slot) lexicographically
static __device__ __forceinline__ bool expert_cache_fr_less(const float sa, const uint32_t la, const int ia,
        const float sb, const uint32_t lb, const int ib) {
    return sa < sb || (sa == sb && (la < lb || (la == lb && ia < ib)));
}

// fr: expert e was routed c times (weighted) in the window that brings the token count to tok
static __device__ __forceinline__ void expert_cache_fr_add(const ggml_cuda_expert_cache_dev & d, const int e,
        const float c, const uint32_t tok) {
    d.freq[e]   = d.freq[e]*exp2f(-(float) (tok - d.tok_of[e])*d.inv_hl) + c;
    d.tok_of[e] = tok;
}

// fr: the routings of id i, once per expert (at its first position), with how often the ids route to it
static __device__ __forceinline__ void expert_cache_fr_add_id(const ggml_cuda_expert_cache_dev & d, const int32_t * ids,
        const int n_ids, const int n_per_row, const int row_stride, const int i, const uint32_t tok) {
    const int id = ids[(i / n_per_row)*row_stride + (i % n_per_row)];
    int c = 0;
    for (int j = 0; j < n_ids; j++) {
        const int idj = ids[(j / n_per_row)*row_stride + (j % n_per_row)];
        if (idj == id) {
            if (j < i) {
                return;
            }
            c++;
        }
    }
    expert_cache_fr_add(d, id, (float) c, tok);
}

// The misses of a one-token window the CPU computes (see expert-cpu.cuh), lowest priority (decayed frequency before this
//   window, then id) first: as few as keep the layer time within the slack of the best split. s_cpu[i] = the place of
//   miss i in the CPU's list, -1 = copied. Returns how many. A repeated miss in the row turns it off.
static __device__ int expert_cache_cpu_choose(const ggml_cuda_expert_cpu_args & cpu, const int * s_remap, const int * s_id,
        const float * s_pri, int * s_cpu, const int n_ids) {
    int m[GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS];
    int nm = 0;
    bool repeat = false;
    for (int i = 0; i < n_ids; i++) {
        s_cpu[i] = -1;
        if (s_remap[i] >= 0) {
            continue;
        }
        int q = nm;
        for (int a = 0; a < nm; a++) {
            repeat = repeat || s_id[m[a]] == s_id[i];
        }
        while (q > 0 && (s_pri[m[q - 1]] > s_pri[i] || (s_pri[m[q - 1]] == s_pri[i] && s_id[m[q - 1]] > s_id[i]))) {
            m[q] = m[q - 1];
            q--;
        }
        m[q] = i;
        nm++;
    }
    if (repeat || nm == 0) {
        return 0;
    }
    auto t = [&](const int k) {
        const float tc = (float) (nm - k)*cpu.copy_us;
        const float tk = k > 0 ? cpu.cpu_a_us + (float) k*cpu.cpu_b_us : 0.0f;
        return fmaxf(tc, tk) + (k > 0 ? cpu.post_us : 0.0f);
    };
    float best = INFINITY;
    for (int k = 0; k <= nm; k++) {
        best = fminf(best, t(k));
    }
    int n_cpu = nm;
    for (int k = 0; k <= nm; k++) {
        if (t(k) <= best*(1.0f + cpu.slack_rel) + cpu.slack_us) {
            n_cpu = k;
            break;
        }
    }
    n_cpu = min(n_cpu, max(0, nm - cpu.min_copy));
    n_cpu = min(n_cpu, GGML_CUDA_EXPERT_CPU_MAX_K);
    for (int q = 0; q < n_cpu; q++) {
        s_cpu[m[q]] = q;
    }
    return n_cpu;
}

// The publishing device's request to the CPU, by warp 0 of the plan's first block: the ids and routing weights of the
//   misses it takes and the layer's input; state[1] = the request's sequence number, 0 = none; the time of the request in
//   state[4..5] (ggml_cuda_expert_cpu_args).
static __device__ void expert_cache_cpu_publish(const ggml_cuda_expert_cpu_args & cpu, const int * s_id, const int * s_cpu,
        const int n_ids, const int n_cpu) {
    const int lane = threadIdx.x;
    ggml_cuda_expert_cpu_mbox * mb = cpu.mbox;
    if (n_cpu == 0) {
        if (lane == 0) {
            cpu.state[1] = 0;
        }
        return;
    }
    for (int i = lane; i < n_ids; i += WARP_SIZE) {
        if (s_cpu[i] >= 0) {
            mb->ids[s_cpu[i]] = s_id[i];
            mb->w[s_cpu[i]]   = cpu.w[i];
        }
    }
    const float4 * x4 = (const float4 *) cpu.x;
    float4 *       d4 = (float4 *) mb->x;
    for (int j = lane; j < cpu.n_embd/4; j += WARP_SIZE) {
        d4[j] = x4[j];
    }
    if (lane == 0) {
        mb->group = cpu.group;
        mb->k     = n_cpu;
    }
    __threadfence_system();
    __syncwarp();
    if (lane == 0) {
        uint32_t seq = cpu.state[0] + 1;
        seq = seq == 0 ? 1 : seq;
        cpu.state[0] = seq;
        cpu.state[1] = seq;
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        ((unsigned long long *) (cpu.state + 4))[0] = t;
        mb->request  = seq;
    }
}

// The scheduler and the LRU run in one block. Hits are marked in parallel, misses are then taken one by one
// so the victim of one is never the slot another still needs: the argmin over last_used cannot return a
// slot touched by this batch, its stamp is the newest. Same ids on every device give the same slots.
// fast (a few ids, decode): the same loop by warp 0 alone on a shared copy of last_used and expert_of, with the misses
//   so far in shared memory instead of slot_of: no block barrier and no global round trip per miss, same victims.
// cpu.on: a one-token window (fast, fr) may send misses to the CPU, see expert_cache_cpu_choose; their ids read
//   cpu_slot and they do not enter the pool.
static __global__ void expert_cache_plan(const ggml_cuda_expert_cache_group g, const int32_t * ids,
        const int n_rows, const int n_per_row, const int row_stride, const bool fast, const ggml_cuda_expert_cpu_args cpu) {
    const ggml_cuda_expert_cache_dev & d = g.d[blockIdx.x];
    __shared__ int      s_n_miss;
    __shared__ uint32_t s_min_val[32];
    __shared__ int      s_min_idx[32];
    __shared__ int      s_remap[GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS];
    __shared__ int      s_id[GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS];
    __shared__ int      s_miss_expert[GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS];
    __shared__ int      s_miss_slot[GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS];
    __shared__ float    s_pri[GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS];
    __shared__ int      s_cpu[GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS];
    __shared__ float    s_min_sc[32];
    extern __shared__ uint32_t s_lru[]; // fast: last_used [cap], then expert_of [cap], then with fr the keys [cap]

    const int      n_ids = n_rows*n_per_row;
    const uint32_t clock = *d.clock;
    const bool     fr    = d.freq != nullptr;
    const uint32_t win   = fr ? d.win_of[d.cap] + 1 : 0;           // this window
    const uint32_t tok   = fr ? d.win_of[d.cap + 1] + n_rows : 0;  // the token count after it
    const bool     use_cpu = cpu.on && fast && fr && n_rows == 1;
    const bool     publish = cpu.on && cpu.mbox != nullptr && blockIdx.x == 0;
    if (publish && !use_cpu && threadIdx.x == 0) {
        cpu.state[1] = 0;
    }

    uint32_t * s_last  = s_lru;
    int32_t  * s_exp   = (int32_t *) (s_lru + d.cap);
    float    * s_score = (float *) (s_lru + 2*d.cap);
    if (fast) {
        // it does not depend on the ids, so its round trip overlaps theirs and that of slot_of; with fr the keys too:
        //   the only slots whose key changes in this window are the ones it uses, and those are never victims
        for (int s = threadIdx.x; s < d.cap; s += blockDim.x) {
            s_last[s] = d.last_used[s];
            s_exp[s]  = d.expert_of[s];
            if (fr) {
                s_score[s] = expert_cache_fr_key(d, s_exp[s], d.win_of[s], tok, win);
            }
        }
    }

    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int id = ids[(i / n_per_row)*row_stride + (i % n_per_row)];
        const int sl = d.slot_of[id];
        d.remap[i] = sl;
        if (fast) {
            s_remap[i] = sl;
            s_id[i]    = id;
            if (use_cpu) {
                // a miss's priority: its frequency before this window, decayed to its end (warps 1+ add the window later)
                s_pri[i] = sl < 0 ? d.freq[id]*exp2f(-(float) (tok - d.tok_of[id])*d.inv_hl) : 0.0f;
            }
        }
        if (sl >= 0) {
            // several tokens of one ubatch may route to the same expert; max keeps the stamp deterministic
            atomicMax(&d.last_used[sl], clock + 1 + i);
            if (fr) {
                d.win_of[sl] = win;
            }
        }
    }
    if (threadIdx.x == 0) {
        s_n_miss = 0;
    }
    __syncthreads();

    if (fast) {
        // the stamps of the hits, as in last_used: the copy may or may not have seen some of them, max is the same
        for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
            if (s_remap[i] >= 0) {
                atomicMax(&s_last[s_remap[i]], clock + 1 + i);
                if (fr) {
                    s_score[s_remap[i]] = INFINITY; // used in this window: not a victim
                }
            }
        }
        __syncthreads();
        if (threadIdx.x >= WARP_SIZE) {
            // fr: the frequencies of the routed experts, beside the misses of warp 0, which never reads them
            const int i = threadIdx.x - WARP_SIZE;
            if (fr && i < n_ids) {
                expert_cache_fr_add_id(d, s_id, n_ids, n_ids, n_ids, i, tok);
            }
            return;
        }
        const int lane   = threadIdx.x;
        int       n_miss = 0;
        int       n_cpu  = 0;
        if (use_cpu) {
            if (lane == 0) {
                n_cpu = expert_cache_cpu_choose(cpu, s_remap, s_id, s_pri, s_cpu, n_ids);
                d.stats[4] += n_cpu;
            }
            n_cpu = __shfl_sync(0xffffffff, n_cpu, 0);
            __syncwarp();
            if (publish) {
                expert_cache_cpu_publish(cpu, s_id, s_cpu, n_ids, n_cpu);
            }
        }
        for (int i = 0; i < n_ids; i++) {
            if (s_remap[i] >= 0) {
                continue;
            }
            if (n_cpu > 0 && s_cpu[i] >= 0) {
                // the CPU computes it: nothing copied or evicted, see cpu_slot
                if (lane == 0) {
                    d.remap[i] = d.cpu_slot;
                }
                continue;
            }
            const int id = s_id[i];
            int sl_now = -1; // a second occurrence of a miss handled earlier in this loop
            for (int k = 0; k < n_miss; k++) {
                sl_now = s_miss_expert[k] == id ? s_miss_slot[k] : sl_now;
            }
            if (sl_now >= 0) {
                if (lane == 0) {
                    d.remap[i] = sl_now;
                }
                continue;
            }

            uint32_t best   = 0xffffffffu;
            int      best_i = 0x7fffffff;
            if (fr) {
                float best_s = INFINITY;
                for (int s = lane; s < d.cap; s += WARP_SIZE) {
                    const float sc = s_score[s];
                    if (sc != INFINITY && expert_cache_fr_less(sc, s_last[s], s, best_s, best, best_i)) {
                        best_s = sc;
                        best   = s_last[s];
                        best_i = s;
                    }
                }
                for (int off = 16; off > 0; off >>= 1) {
                    const float    os = __shfl_xor_sync(0xffffffff, best_s, off);
                    const uint32_t ob = __shfl_xor_sync(0xffffffff, best,   off);
                    const int      oi = __shfl_xor_sync(0xffffffff, best_i, off);
                    if (expert_cache_fr_less(os, ob, oi, best_s, best, best_i)) {
                        best_s = os;
                        best   = ob;
                        best_i = oi;
                    }
                }
            } else {
                for (int s = lane; s < d.cap; s += WARP_SIZE) {
                    const uint32_t v = s_last[s];
                    if (v < best) {
                        best   = v;
                        best_i = s;
                    }
                }
                for (int off = 16; off > 0; off >>= 1) {
                    const uint32_t ob = __shfl_xor_sync(0xffffffff, best,   off);
                    const int      oi = __shfl_xor_sync(0xffffffff, best_i, off);
                    if (ob < best || (ob == best && oi < best_i)) {
                        best   = ob;
                        best_i = oi;
                    }
                }
            }
            if (lane == 0) {
                const int victim = best_i;
                const int old    = s_exp[victim];
                if (old >= 0) {
                    d.slot_of[old] = -1;
                }
                if (fr) {
                    s_score[victim]  = INFINITY;
                    d.win_of[victim] = win;
                }
                s_exp[victim]       = id;
                d.expert_of[victim] = id;
                d.slot_of[id]       = victim;
                s_last[victim]      = clock + 1 + i;
                d.last_used[victim] = clock + 1 + i;
                d.remap[i]          = victim;
                d.miss_expert[n_miss] = id;
                d.miss_slot[n_miss]   = victim;
                s_miss_expert[n_miss] = id;
                s_miss_slot[n_miss]   = victim;
            }
            n_miss++;
            __syncwarp();
        }
        if (lane == 0) {
            *d.n_miss = n_miss;
            *d.clock  = clock + 1 + n_ids;
            d.stats[0] += 1;
            d.stats[1] += n_miss;
            if (n_rows == 1) {
                d.stats[5] += 1;
                d.stats[6] += n_miss;
            }
            if (fr) {
                d.win_of[d.cap]     = win;
                d.win_of[d.cap + 1] = tok;
            }
        }
        return;
    }

    if (fr) {
        // the keys below read only slots this window has not used, whose experts these do not touch
        for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
            expert_cache_fr_add_id(d, ids, n_ids, n_per_row, row_stride, i, tok);
        }
    }

    for (int i = 0; i < n_ids; i++) {
        if (d.remap[i] >= 0) {
            continue;
        }
        const int id     = ids[(i / n_per_row)*row_stride + (i % n_per_row)];
        const int sl_now = d.slot_of[id]; // a second occurrence of a miss handled earlier in this loop
        if (sl_now >= 0) {
            if (threadIdx.x == 0) {
                d.remap[i] = sl_now;
            }
            __syncthreads();
            continue;
        }

        uint32_t best   = 0xffffffffu;
        int      best_i = 0x7fffffff;
        float    best_s = INFINITY;
        if (fr) {
            for (int s = threadIdx.x; s < d.cap; s += blockDim.x) {
                if (d.win_of[s] == win) {
                    continue; // used in this window
                }
                const float sc = expert_cache_fr_key(d, d.expert_of[s], d.win_of[s], tok, win);
                if (expert_cache_fr_less(sc, d.last_used[s], s, best_s, best, best_i)) {
                    best_s = sc;
                    best   = d.last_used[s];
                    best_i = s;
                }
            }
            for (int off = 16; off > 0; off >>= 1) {
                const float    os = __shfl_xor_sync(0xffffffff, best_s, off);
                const uint32_t ob = __shfl_xor_sync(0xffffffff, best,   off);
                const int      oi = __shfl_xor_sync(0xffffffff, best_i, off);
                if (expert_cache_fr_less(os, ob, oi, best_s, best, best_i)) {
                    best_s = os;
                    best   = ob;
                    best_i = oi;
                }
            }
        } else {
            for (int s = threadIdx.x; s < d.cap; s += blockDim.x) {
                const uint32_t v = d.last_used[s];
                if (v < best) {
                    best   = v;
                    best_i = s;
                }
            }
            for (int off = 16; off > 0; off >>= 1) {
                const uint32_t ob = __shfl_xor_sync(0xffffffff, best,   off);
                const int      oi = __shfl_xor_sync(0xffffffff, best_i, off);
                if (ob < best || (ob == best && oi < best_i)) {
                    best   = ob;
                    best_i = oi;
                }
            }
        }
        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        if (lane == 0) {
            s_min_val[warp] = best;
            s_min_idx[warp] = best_i;
            s_min_sc[warp]  = best_s;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int nw = blockDim.x / 32;
            for (int w = 1; w < nw; w++) {
                const bool better = fr ?
                    expert_cache_fr_less(s_min_sc[w], s_min_val[w], s_min_idx[w], best_s, best, best_i) :
                    s_min_val[w] < best || (s_min_val[w] == best && s_min_idx[w] < best_i);
                if (better) {
                    best_s = s_min_sc[w];
                    best   = s_min_val[w];
                    best_i = s_min_idx[w];
                }
            }
            const int victim = best_i;
            const int old    = d.expert_of[victim];
            if (old >= 0) {
                d.slot_of[old] = -1;
            }
            if (fr) {
                d.win_of[victim] = win;
            }
            d.expert_of[victim] = id;
            d.slot_of[id]       = victim;
            d.last_used[victim] = clock + 1 + i;
            d.remap[i]          = victim;
            d.miss_expert[s_n_miss] = id;
            d.miss_slot[s_n_miss]   = victim;
            s_n_miss++;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *d.n_miss = s_n_miss;
        *d.clock  = clock + 1 + n_ids;
        d.stats[0] += 1;
        d.stats[1] += s_n_miss;
        if (fr) {
            d.win_of[d.cap]     = win;
            d.win_of[d.cap + 1] = tok;
        }
    }
}

// One row of blocks per possible miss; rows past n_miss leave at once. Threads copy consecutive units of T,
// so a warp reads a contiguous span of host memory per instruction.
template <typename T>
static __device__ __forceinline__ void expert_cache_copy_slot(const ggml_cuda_expert_cache_dev & d, char * pool, const int m) {
    const int    e     = d.miss_expert[m];
    const int    s     = d.miss_slot[m];
    const char * src_e = d.host + (size_t) e*d.expert_bytes + d.offset_dev;
    char       * dst_s = pool + (size_t) s*d.slot_bytes;

    const size_t n = d.slot_bytes / sizeof(T);
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) {
        const size_t off = i*sizeof(T);
        const size_t c   = off / d.chunk_dev;
        const size_t w   = off - c*d.chunk_dev;
        ((T *) dst_s)[i] = *(const T *) (src_e + c*d.chunk_full + w);
    }
}

// A slice that is only 2-byte aligned (a K slice of q8_0 rows, 34-byte blocks) is read in the aligned 8-byte words that
//   cover each chunk instead of 2-byte units, which are slow over PCIe, and the units of TS inside the chunk are stored.
//   The words stay in the chunk's row: rows start 8-byte aligned (see covering in ggml_cuda_expert_cache_create).
template <typename TS>
static __device__ __forceinline__ void expert_cache_copy_slot_covering(const ggml_cuda_expert_cache_dev & d, char * pool, const int m) {
    const int    e     = d.miss_expert[m];
    const int    s     = d.miss_slot[m];
    const char * src_e = d.host + (size_t) e*d.expert_bytes + d.offset_dev;
    char       * dst_s = pool + (size_t) s*d.slot_bytes;

    const size_t nw = (d.chunk_dev + 14) / 8; // the words a chunk can touch
    const size_t n  = (size_t) d.n_chunks * nw;
    for (size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) {
        const size_t c     = i / nw;
        const char * src_c = src_e + c*d.chunk_full;
        const char * end   = src_c + d.chunk_dev;
        const char * word  = (const char *) ((uintptr_t) src_c & ~(uintptr_t) 7) + 8*(i - c*nw);
        if (word >= end) {
            continue;
        }
        const uint2    v = *(const uint2 *) word;
        const uint64_t u = (uint64_t) v.x | ((uint64_t) v.y << 32);
        char * dst_w = dst_s + c*d.chunk_dev + (word - src_c); // where the word's first byte goes
#pragma unroll
        for (int k = 0; k < 8; k += sizeof(TS)) {
            if (word + k >= src_c && word + k < end) {
                *(TS *) (dst_w + k) = (TS) (u >> (8*k));
            }
        }
    }
}

// blockIdx.z is the cache of the group, each copied with the widest unit its addresses allow (negative: covering);
//   blockIdx.y steps through the misses
static __global__ void expert_cache_copy(const ggml_cuda_expert_cache_group g) {
    const int z = blockIdx.z;
    const ggml_cuda_expert_cache_dev & d = g.d[z];
    const int n_miss = *d.n_miss;
    for (int m = blockIdx.y; m < n_miss; m += gridDim.y) {
        switch (g.align[z]) {
            case 16: expert_cache_copy_slot<int4>   (d, g.pool[z], m); break;
            case  8: expert_cache_copy_slot<int2>   (d, g.pool[z], m); break;
            case  4: expert_cache_copy_slot<int32_t>(d, g.pool[z], m); break;
            case  2: expert_cache_copy_slot<int16_t>(d, g.pool[z], m); break;
            case -2: expert_cache_copy_slot_covering<uint16_t>(d, g.pool[z], m); break;
            case -1: expert_cache_copy_slot_covering<uint8_t> (d, g.pool[z], m); break;
            default: expert_cache_copy_slot<int8_t> (d, g.pool[z], m); break;
        }
    }
}

// The experts the pool holds, copied from their slots into a whole staged copy of this device's slice of the
// weight (one slot-sized stride per expert). The caller copies the other experts from host memory.
template <typename T>
static __global__ void expert_cache_fill(const ggml_cuda_expert_cache_dev d, const char * pool, char * dst) {
    const size_t n = d.slot_bytes / sizeof(T);
    for (int e = blockIdx.x; e < d.n_expert; e += gridDim.x) {
        const int s = d.slot_of[e];
        if (s < 0) {
            continue;
        }
        const T * src_s = (const T *) (pool + (size_t) s*d.slot_bytes);
        T *       dst_e = (T *) (dst + (size_t) e*d.slot_bytes);
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
            dst_e[i] = src_s[i];
        }
    }
}

// After a wide ubatch read a whole staged copy of the weight, the experts it routed to most (ties to the lower id) come
//   into the pool: the first n_evict of them as if the LRU had been handed them coldest first (resident ones take a new
//   stamp, the others the least recently used slots no selected expert holds), the next ones only into slots that hold
//   no expert yet. A full pool so keeps all its residents but n_evict, the most recently used ones. The misses are listed
//   for expert_cache_copy. One block, the same decision on every device and in every pool of the layer, as for a gather.
static __global__ void expert_cache_warm_plan(const ggml_cuda_expert_cache_dev d, const int32_t * ids,
        const int n_rows, const int n_per_row, const int row_stride, const int n_evict) {
    extern __shared__ int s_warm[]; // count [n_expert], expert by rank [cap], resident [cap], last_used [cap], expert_of [cap]
    int      * s_cnt  = s_warm;
    int      * s_sel  = s_cnt + d.n_expert;
    int      * s_res  = s_sel + d.cap;
    uint32_t * s_last = (uint32_t *) (s_res + d.cap);
    int      * s_exp  = (int *) (s_last + d.cap);
    __shared__ int s_n_sel;
    __shared__ int s_n_empty;

    // every stamp so far is below clock, see expert_cache_plan
    const uint32_t clock = *d.clock;
    if (threadIdx.x == 0) {
        s_n_sel   = 0;
        s_n_empty = 0;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < d.n_expert; e += blockDim.x) {
        s_cnt[e] = 0;
    }
    for (int s = threadIdx.x; s < d.cap; s += blockDim.x) {
        s_last[s] = d.last_used[s];
        s_exp[s]  = d.expert_of[s];
        if (s_exp[s] < 0) {
            atomicAdd(&s_n_empty, 1);
        }
    }
    __syncthreads();
    const bool fr = d.freq != nullptr;
    if (n_evict == 0 && s_n_empty == 0 && !fr) {
        if (threadIdx.x == 0) {
            *d.n_miss = 0;
        }
        return;
    }

    const int n_ids = n_rows*n_per_row;
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        atomicAdd(&s_cnt[ids[(i / n_per_row)*row_stride + (i % n_per_row)]], 1);
    }
    __syncthreads();

    // fr: the routings of a wide ubatch count with the mean recency of its tokens, hl*(1 - 2^(-T/hl))/(T ln 2), as if
    //   they had come one by one (capy-flash T69), so a long prompt does not outweigh the decode after it
    const uint32_t win = fr ? d.win_of[d.cap] + 1 : 0;
    const uint32_t tok = fr ? d.win_of[d.cap + 1] + n_rows : 0;
    if (fr) {
        const float x      = (float) n_rows*d.inv_hl;
        const float weight = x > 0.0f ? (1.0f - exp2f(-x))/(x*0.69314718f) : 1.0f;
        for (int e = threadIdx.x; e < d.n_expert; e += blockDim.x) {
            if (s_cnt[e] > 0) {
                expert_cache_fr_add(d, e, weight*(float) s_cnt[e], tok);
            }
        }
        if (n_evict == 0 && s_n_empty == 0) {
            if (threadIdx.x == 0) {
                *d.n_miss           = 0;
                d.win_of[d.cap]     = win;
                d.win_of[d.cap + 1] = tok;
            }
            return;
        }
    }

    // the rank of a routed expert: how many were routed more often, or as often with a lower id
    for (int e = threadIdx.x; e < d.n_expert; e += blockDim.x) {
        const int c = s_cnt[e];
        if (c == 0) {
            continue;
        }
        int r = 0;
        for (int e2 = 0; e2 < d.n_expert; e2++) {
            const int c2 = s_cnt[e2];
            r += c2 > c || (c2 == c && e2 < e);
        }
        if (r < d.cap) {
            s_sel[r] = e;
            atomicAdd(&s_n_sel, 1);
        }
    }
    __syncthreads();
    const int n_sel = s_n_sel;
    const int n_top = min(n_evict, n_sel);

    // rank r is stamped clock + n_sel - r: the first n_top are above every other slot, so none of them is a victim below
    for (int r = threadIdx.x; r < n_sel; r += blockDim.x) {
        const int sl = d.slot_of[s_sel[r]];
        s_res[r] = sl >= 0;
        if (sl >= 0 && r < n_top) {
            s_last[sl]      = clock + (uint32_t) (n_sel - r);
            d.last_used[sl] = clock + (uint32_t) (n_sel - r);
            if (fr) {
                d.win_of[sl] = win;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x >= WARP_SIZE) {
        return;
    }

    const int lane   = threadIdx.x;
    int       n_miss = 0;
    for (int i = 0; i < n_sel; i++) {
        const int  r   = i < n_top ? n_top - 1 - i : i; // the first n_top coldest first, then the others hottest first
        const bool top = i < n_top;
        if (s_res[r]) {
            continue;
        }
        uint32_t best   = 0xffffffffu;
        int      best_i = 0x7fffffff;
        for (int s = lane; s < d.cap; s += WARP_SIZE) {
            const uint32_t v = top || s_exp[s] < 0 ? s_last[s] : 0xffffffffu;
            if (v < best) {
                best   = v;
                best_i = s;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const uint32_t ob = __shfl_xor_sync(0xffffffff, best,   off);
            const int      oi = __shfl_xor_sync(0xffffffff, best_i, off);
            if (ob < best || (ob == best && oi < best_i)) {
                best   = ob;
                best_i = oi;
            }
        }
        if (best_i == 0x7fffffff) {
            break; // past the first n_top: no empty slot left
        }
        if (lane == 0) {
            const int e      = s_sel[r];
            const int victim = best_i;
            const int old    = s_exp[victim];
            if (old >= 0) {
                d.slot_of[old] = -1;
            }
            s_exp[victim]         = e;
            d.expert_of[victim]   = e;
            d.slot_of[e]          = victim;
            s_last[victim]        = clock + (uint32_t) (n_sel - r);
            d.last_used[victim]   = clock + (uint32_t) (n_sel - r);
            if (fr) {
                d.win_of[victim] = win;
            }
            d.miss_expert[n_miss] = e;
            d.miss_slot[n_miss]   = victim;
        }
        n_miss++;
        __syncwarp();
    }
    if (lane == 0) {
        *d.n_miss = n_miss;
        *d.clock  = clock + (uint32_t) n_sel + 1;
        d.stats[2] += 1;
        d.stats[3] += n_miss;
        if (fr) {
            d.win_of[d.cap]     = win;
            d.win_of[d.cap + 1] = tok;
        }
    }
}

// GGML_META_EXPERT_CACHE_CHECK=1 (validation): every resident slot of the pool compared byte by byte with its expert's
//   slice in host memory. One block per slot; bad[0] counts the slots checked, bad[1] the bytes that differ.
static __global__ void expert_cache_check(const ggml_cuda_expert_cache_dev d, const char * pool, unsigned int * bad) {
    const int e = d.expert_of[blockIdx.x];
    if (e < 0) {
        return;
    }
    const char * src_e = d.host + (size_t) e*d.expert_bytes + d.offset_dev;
    const char * dst_s = pool + (size_t) blockIdx.x*d.slot_bytes;
    unsigned int n_bad = 0;
    for (size_t off = threadIdx.x; off < d.slot_bytes; off += blockDim.x) {
        const size_t c = off / d.chunk_dev;
        n_bad += dst_s[off] != src_e[c*d.chunk_full + (off - c*d.chunk_dev)];
    }
    if (threadIdx.x == 0) {
        atomicAdd(&bad[0], 1u);
    }
    if (n_bad > 0) {
        atomicAdd(&bad[1], n_bad);
    }
}

// Registered host ranges, so several weights in one buffer register it once.
struct expert_cache_host_reg {
    const char * base;
    size_t       size;
    const char * dev_base;
    bool         owned; // registered here
};
static std::vector<expert_cache_host_reg> g_host_regs;
static std::mutex                         g_host_regs_mutex;

// cudaHostRegister of [base, base + len), page aligned; its device address or nullptr
static const char * expert_cache_register_range(const char * base, size_t len) {
    cudaError_t err = cudaHostRegister((void *) base, len, cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: could not register %.0f MiB of host memory at %p for device access: %s\n",
            __func__, len/1024.0/1024.0, (const void *) base, cudaGetErrorString(err));
        return nullptr;
    }
    void * dev_base = nullptr;
    err = cudaHostGetDevicePointer(&dev_base, (void *) base, 0);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        cudaHostUnregister((void *) base);
        return nullptr;
    }
    g_host_regs.push_back({base, len, (const char *) dev_base, true});
    return (const char *) dev_base;
}

// The device address of the weight at ptr (ptr_size bytes) in the host buffer buf: the whole buffer is registered once,
//   or if that fails (a large pageable buffer, see ggml_cuda_expert_cpu_host_alloc) the weight's own range.
static const char * expert_cache_register_host(const void * buf, size_t size, const void * ptr, size_t ptr_size) {
    static const size_t page = 4096;
    const char * base  = (const char *) ((uintptr_t) buf & ~(page - 1));
    size_t       len   = (((const char *) buf + size) - base + page - 1) & ~(page - 1);
    const char * p     = (const char *) ptr;

    std::lock_guard<std::mutex> lock(g_host_regs_mutex);
    for (const expert_cache_host_reg & r : g_host_regs) {
        if (r.base <= p && p + ptr_size <= r.base + r.size) {
            return r.dev_base + (p - r.base);
        }
    }

    // memory that is already page-locked (a cudaMallocHost buffer, or registered by someone else) cannot be
    // registered again, but already has a device address
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, buf) == cudaSuccess && attr.type == cudaMemoryTypeHost && attr.devicePointer != nullptr) {
        const char * dev_base = (const char *) attr.devicePointer - ((const char *) buf - base);
        g_host_regs.push_back({base, len, dev_base, false});
        return dev_base + (p - base);
    }
    (void) cudaGetLastError();

    const char * dev_base = expert_cache_register_range(base, len);
    if (dev_base != nullptr) {
        return dev_base + (p - base);
    }
    const char * wbase = (const char *) ((uintptr_t) p & ~(page - 1));
    const size_t wlen  = ((p + ptr_size) - wbase + page - 1) & ~(page - 1);
    dev_base = wlen < len ? expert_cache_register_range(wbase, wlen) : nullptr;
    return dev_base != nullptr ? dev_base + (p - wbase) : nullptr;
}

bool ggml_cuda_expert_cache_unregister_host(const void * base, size_t size) {
    std::lock_guard<std::mutex> lock(g_host_regs_mutex);
    bool any = false;
    for (size_t i = 0; i < g_host_regs.size(); ) {
        const expert_cache_host_reg & r = g_host_regs[i];
        if (r.base >= (const char *) base && r.base < (const char *) base + size && r.owned) {
            CUDA_CHECK(cudaHostUnregister((void *) r.base));
            g_host_regs.erase(g_host_regs.begin() + i);
            any = true;
        } else {
            i++;
        }
    }
    return any;
}

static int expert_cache_align(std::initializer_list<size_t> values) {
    int align = 16;
    for (size_t v : values) {
        while (align > 1 && v % align != 0) {
            align /= 2;
        }
    }
    return align;
}

void * ggml_cuda_expert_cache_create(int device, const void * host_buf, size_t host_buf_size,
        const void * host_ptr, int64_t n_expert, int cap, size_t expert_bytes, size_t chunk_full, size_t chunk_dev,
        size_t offset_dev, int64_t n_chunks, int max_ids) {
    if (cap <= 0 || max_ids <= 0 || chunk_dev == 0) {
        return nullptr;
    }

    const char * dev_ptr = expert_cache_register_host(host_buf, host_buf_size, host_ptr, (size_t) n_expert*expert_bytes);
    if (dev_ptr == nullptr) {
        return nullptr;
    }

    ggml_cuda_expert_cache * c = new ggml_cuda_expert_cache();
    c->device = device;
    c->d.host         = dev_ptr;
    c->d.expert_bytes = expert_bytes;
    c->d.chunk_full   = chunk_full;
    c->d.chunk_dev    = chunk_dev;
    c->d.offset_dev   = offset_dev;
    c->d.slot_bytes   = (size_t) n_chunks*chunk_dev;
    c->d.n_chunks     = (int) n_chunks;
    c->d.n_expert     = (int) n_expert;
    c->d.cap          = cap;
    c->d.max_ids      = max_ids;
    // the meta backend leaves one chunk of zeroed headroom past the pool (ggml_backend_meta_gathered_weight_get): a whole
    //   slot when the weight is split by rows
    c->d.cpu_slot     = n_chunks == 1 && chunk_full == expert_bytes ? cap : 0;
    c->warm.cache     = c;
    c->host_cpu       = (const char *) host_ptr;
    c->align = expert_cache_align({(size_t) (uintptr_t) dev_ptr, expert_bytes, chunk_full, chunk_dev, offset_dev, c->d.slot_bytes});
    c->covering = c->align < 4 && expert_cache_align({(size_t) (uintptr_t) dev_ptr, expert_bytes, chunk_full}) >= 8;

    const expert_cache_policy pol = expert_cache_policy_get();
    const size_t n_i32 = (size_t) n_expert + 2*cap + 1 + 3*max_ids + 1 + 7 + (pol.fr ? 2*(size_t) n_expert + cap + 2 : 0);
    ggml_cuda_set_device(device);
    if (cudaMalloc(&c->state, n_i32*sizeof(int32_t)) != cudaSuccess) {
        (void) cudaGetLastError();
        delete c;
        return nullptr;
    }
    int32_t * p = (int32_t *) c->state;
    c->d.slot_of     = p; p += n_expert;
    c->d.expert_of   = p; p += cap;
    c->d.last_used   = (uint32_t *) p; p += cap;
    c->d.clock       = (uint32_t *) p; p += 1;
    c->d.remap       = p; p += max_ids;
    c->d.miss_expert = p; p += max_ids;
    c->d.miss_slot   = p; p += max_ids;
    c->d.n_miss      = p; p += 1;
    c->d.stats       = (uint32_t *) p; p += 7;
    c->d.freq        = nullptr;
    c->d.tok_of      = nullptr;
    c->d.win_of      = nullptr;
    c->d.inv_hl      = 0.0f;
    c->d.inv_tau     = 0.0f;
    if (pol.fr) {
        c->d.freq    = (float *) p; p += n_expert;
        c->d.tok_of  = (uint32_t *) p; p += n_expert;
        c->d.win_of  = (uint32_t *) p;
        c->d.inv_hl  = 1.0f/pol.half_life;
        c->d.inv_tau = pol.tau > 0.0f ? 1.0f/pol.tau : 0.0f;
        CUDA_CHECK(cudaMemset(c->d.freq,   0, 2*(size_t) n_expert*sizeof(float)));
        CUDA_CHECK(cudaMemset(c->d.win_of, 0, ((size_t) cap + 2)*sizeof(uint32_t)));
    }
    CUDA_CHECK(cudaMemset(c->d.stats,     0,    7*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(c->d.slot_of,   0xff, (size_t) n_expert*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(c->d.expert_of, 0xff, (size_t) cap*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(c->d.last_used, 0,    (size_t) cap*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(c->d.clock,     0,    sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(c->d.n_miss,    0,    sizeof(int32_t)));
    // the meta backend may lay the host weight out anew once its caches exist (ggml_backend_meta_dm_apply): nothing
    //   queued before may still read it
    CUDA_CHECK(cudaDeviceSynchronize());
    {
        std::lock_guard<std::mutex> lock(g_caches_mutex);
        g_caches.push_back(c);
    }
    // outside any graph capture: the CPU threads and their mailbox, if asked for
    ggml_cuda_expert_cpu_init();
    return c;
}

void ggml_cuda_expert_cache_free(void * cache) {
    ggml_cuda_expert_cache * c = (ggml_cuda_expert_cache *) cache;
    if (c == nullptr) {
        return;
    }
    ggml_cuda_set_device(c->device);
    {
        std::lock_guard<std::mutex> lock(g_caches_mutex);
        g_caches.erase(std::remove(g_caches.begin(), g_caches.end(), c), g_caches.end());
    }
    // nothing may still copy from the host weight: the meta backend puts its rows back after this (ggml_backend_meta_dm_drop)
    cudaDeviceSynchronize();
    static const bool stats = getenv("GGML_META_EXPERT_CACHE_STATS") != nullptr;
    if (stats) {
        static std::once_flag cpu_once;
        std::call_once(cpu_once, ggml_cuda_expert_cpu_stats_print);
        uint32_t st[7] = {0, 0, 0, 0, 0, 0, 0};
        if (cudaMemcpy(st, c->d.stats, sizeof(st), cudaMemcpyDeviceToHost) == cudaSuccess && (st[0] > 0 || st[2] > 0)) {
            fprintf(stderr, "expert-cache-dev: dev %d cap %d n_expert %d slot_bytes %zu calls %u misses %u miss/call %.4f MiB/call %.3f"
                " warms %u warm misses %u cpu experts %u one-row calls %u one-row misses %u\n",
                c->device, c->d.cap, c->d.n_expert, c->d.slot_bytes, st[0], st[1], st[0] > 0 ? (double) st[1]/st[0] : 0.0,
                st[0] > 0 ? (double) st[1]/st[0]*c->d.slot_bytes/1048576.0 : 0.0, st[2], st[3], st[4], st[5], st[6]);
        }
        (void) cudaGetLastError();
    }
    // the pool is known once a warm tag was asked for
    static const bool check = getenv("GGML_META_EXPERT_CACHE_CHECK") != nullptr;
    if (check && c->warm.pool != nullptr) {
        unsigned int * bad = nullptr;
        unsigned int   res[2] = {0, 0};
        if (cudaMalloc(&bad, sizeof(res)) == cudaSuccess && cudaMemset(bad, 0, sizeof(res)) == cudaSuccess) {
            expert_cache_check<<<c->d.cap, 256>>>(c->d, c->warm.pool, bad);
            if (cudaMemcpy(res, bad, sizeof(res), cudaMemcpyDeviceToHost) == cudaSuccess) {
                fprintf(stderr, "expert-cache-check: dev %d slots %u bad %u\n", c->device, res[0], res[1]);
            }
        }
        cudaFree(bad);
        (void) cudaGetLastError();
    }
    cudaFree(c->state);
    delete c;
}

bool ggml_cuda_expert_cache_is(const ggml_tensor * pool) {
    return pool != nullptr && pool->extra != nullptr &&
        ((const ggml_cuda_expert_cache *) pool->extra)->magic == GGML_CUDA_EXPERT_CACHE_MAGIC;
}

void ggml_cuda_expert_cache_gather_group(ggml_backend_cuda_context & ctx, const ggml_tensor * const * pools, const int n,
        const ggml_tensor * ids, ggml_tensor * ids_remap, const ggml_cuda_expert_cpu_args * cpu) {
    GGML_ASSERT(n >= 1 && n <= GGML_CUDA_EXPERT_CACHE_GROUP_MAX);
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    const int n_per_row  = (int) ids->ne[0];
    const int n_rows     = (int) (ids->ne[1]*ids->ne[2]*ids->ne[3]);
    const int row_stride = (int) (ids->nb[1]/sizeof(int32_t));
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1);

    // enough threads in flight to keep the link busy, bounded so short slices do not launch empty blocks
    const int nthreads = 256;
    int       nbx      = 1;
    ggml_cuda_expert_cache_group g = {};
    g.n = n;
    for (int k = 0; k < n; k++) {
        const ggml_cuda_expert_cache * c = (const ggml_cuda_expert_cache *) pools[k]->extra;
        GGML_ASSERT(n_rows*n_per_row <= c->d.max_ids);
        GGML_ASSERT(pools[k]->nb[2] == c->d.slot_bytes);
        g.d[k]     = c->d;
        g.pool[k]  = (char *) pools[k]->data;
        g.align[k] = c->covering ? -c->align : c->align;
        nbx = std::max(nbx, (int) std::min<size_t>(64, (c->d.slot_bytes/c->align + nthreads - 1)/nthreads));
    }

    int cap_min = INT_MAX;
    int cap_max = 0;
    for (int k = 0; k < n; k++) {
        cap_min = std::min(cap_min, g.d[k].cap);
        cap_max = std::max(cap_max, g.d[k].cap);
    }
    const bool   fr    = g.d[0].freq != nullptr;
    const int    n_ids = n_rows*n_per_row;
    const bool   fast  = n_ids <= GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS && n_ids <= cap_min &&
                         cap_max <= (fr ? GGML_CUDA_EXPERT_CACHE_PLAN_FAST_CAP_FR : GGML_CUDA_EXPERT_CACHE_PLAN_FAST_CAP);
    const size_t smem  = fast ? (fr ? 3 : 2)*(size_t) cap_max*sizeof(uint32_t) : 0;
    ggml_cuda_expert_cpu_args cpu_args = {};
    if (cpu != nullptr) {
        cpu_args = *cpu;
    }
    expert_cache_plan<<<n, 512, smem, stream>>>(g, (const int32_t *) ids->data, n_rows, n_per_row, row_stride, fast, cpu_args);
    CUDA_CHECK(cudaGetLastError());

    // A decode step misses few experts, and most of its launches none: Qwen3.8-Flash ~28 of 36 a token. Blocks for every
    //   id made those launches 1920 blocks that only read n_miss, 4.7 us against ~2.2 for a small kernel; one block
    //   alone fills the x16 link while the SMs idle. GGML_CUDA_EXPERT_COPY_ROWS sets how many misses go at once,
    //   0 = one per id.
    static const int copy_rows = [] {
        const char * env = getenv("GGML_CUDA_EXPERT_COPY_ROWS");
        return env != nullptr ? std::max(0, atoi(env)) : 2;
    }();
    const int n_y = std::max(1, copy_rows > 0 ? std::min(n_ids, copy_rows) : n_ids);
    expert_cache_copy<<<dim3(nbx, n_y, n), nthreads, 0, stream>>>(g);
    CUDA_CHECK(cudaGetLastError());

    for (int k = 0; k < n; k++) {
        ggml_tensor * r = &ids_remap[k];
        *r = *ids;
        r->data  = g.d[k].remap;
        r->nb[1] = (size_t) n_per_row*sizeof(int32_t);
        r->nb[2] = r->nb[1]*ids->ne[1];
        r->nb[3] = r->nb[2]*ids->ne[2];
        r->view_src = nullptr;
        r->extra    = nullptr;
    }
}

void ggml_cuda_expert_cache_gather(ggml_backend_cuda_context & ctx, const ggml_tensor * pool, const ggml_tensor * ids, ggml_tensor * ids_remap) {
    ggml_cuda_expert_cache_gather_group(ctx, &pool, 1, ids, ids_remap);
}

bool ggml_cuda_expert_cache_read_slots(void * cache, int32_t * slot_of) {
    const ggml_cuda_expert_cache * c = (const ggml_cuda_expert_cache *) cache;
    if (c == nullptr || c->magic != GGML_CUDA_EXPERT_CACHE_MAGIC || slot_of == nullptr) {
        return false;
    }
    ggml_cuda_set_device(c->device);
    CUDA_CHECK(cudaMemcpy(slot_of, c->d.slot_of, (size_t) c->d.n_expert*sizeof(int32_t), cudaMemcpyDeviceToHost));
    return true;
}

bool ggml_cuda_expert_cache_fill(ggml_backend_cuda_context & ctx, void * cache, const void * pool_data, void * dst) {
    const ggml_cuda_expert_cache * c = (const ggml_cuda_expert_cache *) cache;
    if (c == nullptr || c->magic != GGML_CUDA_EXPERT_CACHE_MAGIC || c->device != ctx.device ||
            pool_data == nullptr || dst == nullptr || (uintptr_t) pool_data % c->align != 0 || (uintptr_t) dst % c->align != 0) {
        return false;
    }
    const ggml_cuda_expert_cache_dev & d = c->d;
    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();

    // the copy runs beside the prefill matmuls: qwen4exp prefill is best at 64 blocks (16: -6%, 256: -4%);
    //   GGML_META_EXPERT_CACHE_FILL_BLOCKS overrides
    static const int n_blocks = getenv("GGML_META_EXPERT_CACHE_FILL_BLOCKS") ? atoi(getenv("GGML_META_EXPERT_CACHE_FILL_BLOCKS")) : 64;
    const int  nthreads = 256;
    const dim3 grid(std::max(1, std::min(n_blocks, d.n_expert)));
    const char * pool = (const char *) pool_data;
    char       * out  = (char *) dst;
    switch (c->align) {
        case 16: expert_cache_fill<int4>   <<<grid, nthreads, 0, stream>>>(d, pool, out); break;
        case  8: expert_cache_fill<int2>   <<<grid, nthreads, 0, stream>>>(d, pool, out); break;
        case  4: expert_cache_fill<int32_t><<<grid, nthreads, 0, stream>>>(d, pool, out); break;
        case  2: expert_cache_fill<int16_t><<<grid, nthreads, 0, stream>>>(d, pool, out); break;
        default: expert_cache_fill<int8_t> <<<grid, nthreads, 0, stream>>>(d, pool, out); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void * ggml_cuda_expert_cache_warm_tag(void * cache, void * pool_data, int n_evict) {
    ggml_cuda_expert_cache * c = (ggml_cuda_expert_cache *) cache;
    if (c == nullptr || c->magic != GGML_CUDA_EXPERT_CACHE_MAGIC || pool_data == nullptr || n_evict < 0) {
        return nullptr;
    }
    c->warm.pool    = (char *) pool_data;
    c->warm.n_evict = std::min(n_evict, c->d.cap);
    return &c->warm;
}

bool ggml_cuda_expert_cache_is_warm(const ggml_tensor * staged) {
    return staged != nullptr && staged->extra != nullptr &&
        ((const expert_cache_warm_tag *) staged->extra)->magic == GGML_CUDA_EXPERT_CACHE_WARM_MAGIC;
}

void ggml_cuda_expert_cache_warm(ggml_backend_cuda_context & ctx, const ggml_tensor * staged, const ggml_tensor * ids) {
    const expert_cache_warm_tag *      tag = (const expert_cache_warm_tag *) staged->extra;
    const ggml_cuda_expert_cache *     c   = tag->cache;
    const ggml_cuda_expert_cache_dev & d   = c->d;
    if (c->device != ctx.device || staged->ne[2] != d.n_expert || staged->nb[2] != d.slot_bytes ||
            staged->ne[3] != 1 || ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[3] != 1) {
        return;
    }
    const size_t smem = ((size_t) d.n_expert + 4*(size_t) d.cap)*sizeof(int32_t);
    if (smem > 48*1024) {
        return;
    }
    const int n_per_row  = (int) ids->ne[0];
    const int n_rows     = (int) ids->ne[1];
    const int row_stride = (int) (ids->nb[1]/sizeof(int32_t));
    cudaStream_t stream = ctx.stream();
    expert_cache_warm_plan<<<1, 512, smem, stream>>>(d, (const int32_t *) ids->data, n_rows, n_per_row, row_stride, tag->n_evict);
    CUDA_CHECK(cudaGetLastError());

    // the misses come from the staged copy, one slot-sized run per expert, through the gather's copy kernel
    ggml_cuda_expert_cache_group g = {};
    g.n                 = 1;
    g.d[0]              = d;
    g.d[0].host         = (const char *) staged->data;
    g.d[0].expert_bytes = d.slot_bytes;
    g.d[0].chunk_full   = d.slot_bytes;
    g.d[0].chunk_dev    = d.slot_bytes;
    g.d[0].offset_dev   = 0;
    g.d[0].n_chunks     = 1;
    g.pool[0]           = tag->pool;
    g.align[0]          = expert_cache_align({(size_t) (uintptr_t) staged->data, (size_t) (uintptr_t) tag->pool, d.slot_bytes});
    const int nthreads = 256;
    const int nbx      = (int) std::min<size_t>(64, (d.slot_bytes/g.align[0] + nthreads - 1)/nthreads);
    expert_cache_copy<<<dim3(std::max(nbx, 1), d.cap, 1), nthreads, 0, stream>>>(g);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_expert_cache_cpu_mat(const ggml_tensor * pool, ggml_cuda_expert_cpu_mat & m) {
    if (!ggml_cuda_expert_cache_is(pool)) {
        return false;
    }
    const ggml_cuda_expert_cache * c = (const ggml_cuda_expert_cache *) pool->extra;
    const ggml_type t = pool->type;
    if (c->d.n_chunks != 1 || c->d.chunk_dev != c->d.slot_bytes) {
        return false;
    }
    m = {};
    m.type         = t;
    m.base         = c->host_cpu;
    m.expert_bytes = c->d.expert_bytes;
    if (c->d.chunk_full == c->d.expert_bytes) {
        // split by rows: the expert is plain rows
        const size_t row_bytes = ggml_row_size(t, pool->ne[0]);
        if (m.expert_bytes % row_bytes != 0) {
            return false;
        }
        m.n_rows             = (int64_t) (m.expert_bytes / row_bytes);
        m.k                  = pool->ne[0];
        m.n_pieces           = 1;
        m.piece_row_bytes[0] = row_bytes;
        m.piece_k[0]         = m.k;
        return true;
    }
    // split by K, device by device (ggml_backend_meta_dm): each device's slice is one piece of n_rows rows
    std::vector<std::pair<size_t, size_t>> pieces; // offset, bytes
    {
        std::lock_guard<std::mutex> lock(g_caches_mutex);
        for (const ggml_cuda_expert_cache * o : g_caches) {
            if (o->host_cpu == c->host_cpu) {
                if (o->d.expert_bytes != m.expert_bytes || o->d.n_chunks != 1 || o->d.chunk_full != o->d.chunk_dev) {
                    return false;
                }
                pieces.push_back({o->d.offset_dev, o->d.slot_bytes});
            }
        }
    }
    std::sort(pieces.begin(), pieces.end());
    m.n_rows = pool->ne[1];
    size_t off = 0;
    for (const auto & pc : pieces) {
        if (pc.first != off || m.n_pieces >= GGML_CUDA_MAX_DEVICES || pc.second % m.n_rows != 0) {
            return false;
        }
        const size_t  rb = pc.second / m.n_rows;
        const int64_t kp = (int64_t) (rb / ggml_type_size(t))*ggml_blck_size(t);
        if (ggml_row_size(t, kp) != rb) {
            return false;
        }
        m.piece_off[m.n_pieces]       = off;
        m.piece_row_bytes[m.n_pieces] = rb;
        m.piece_k0[m.n_pieces]        = m.k;
        m.piece_k[m.n_pieces]         = kp;
        m.k   += kp;
        off   += pc.second;
        m.n_pieces++;
    }
    return off == m.expert_bytes;
}

bool ggml_cuda_expert_cache_zero_slot(const ggml_tensor * pool) {
    return ggml_cuda_expert_cache_is(pool) && ((const ggml_cuda_expert_cache *) pool->extra)->d.cpu_slot ==
        ((const ggml_cuda_expert_cache *) pool->extra)->d.cap;
}
