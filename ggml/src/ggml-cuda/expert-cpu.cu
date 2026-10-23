#include "expert-cpu.cuh"
#include "expert-cache.cuh"
#include "spin.cuh"

#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

GGML_CUDA_SPIN_SETTER(expert_cpu)

// waits for the CPU result the plan asked for (state[1], 0 = none) and adds it to dst; n % 4 == 0, dst 16-byte aligned
static __global__ void expert_cpu_wait_add(const ggml_cuda_expert_cpu_mbox * mb, uint32_t * state, float * dst, const int n) {
    __shared__ uint32_t s_seq;
    if (threadIdx.x == 0) {
        const uint32_t seq = state[1];
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        if (seq != 0 && !ggml_cuda_spin_until([&]() { return mb->completed == seq; })) {
            ggml_cuda_spin_trap(GGML_CUDA_SPIN_SITE_EXPERT_CPU, -1, GGML_CUDA_SPIN_PEER_HOST);
        }
        s_seq = seq;
        if (seq != 0 && blockIdx.x == 0) {
            // the GPU work between the request and this wait, and the wait
            unsigned long long t1;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
            unsigned long long * st = (unsigned long long *) (state + 4);
            st[1] += t0 - st[0];
            st[2] += t1 - t0;
            st[3] += 1;
        }
    }
    __syncthreads();
    if (s_seq == 0) {
        return;
    }
    __threadfence_system();
    const float4 * src = (const float4 *) mb->out;
    float4 *       d4  = (float4 *) dst;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n/4; i += gridDim.x*blockDim.x) {
        const float4 v = __ldcv(src + i);
        float4       a = d4[i];
        a.x += v.x;
        a.y += v.y;
        a.z += v.z;
        a.w += v.w;
        d4[i] = a;
    }
}

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <immintrin.h>
#pragma comment(lib, "Synchronization.lib")

static int64_t expert_cpu_now_us() {
    static const double inv = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return 1e6 / (double) f.QuadPart; }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (int64_t) ((double) c.QuadPart * inv);
}

static float expert_cpu_env_f(const char * name, float def) {
    const char * v = getenv(name);
    return v != nullptr ? (float) atof(v) : def;
}

bool ggml_cuda_expert_cpu_enabled() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_EXPERT_CPU");
        return v != nullptr && atoi(v) != 0;
    }();
    return on;
}

int ggml_cuda_expert_cpu_device() {
    static const int dev = (int) expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_DEVICE", 0.0f);
    return dev;
}

// The two NUMA nodes with CPUs and the CPUs of each the pool runs on: the first CPU of a node is left out (CPU 0 takes
//   interrupts), then GGML_CUDA_EXPERT_CPU_THREADS of them (16 by default).
struct expert_cpu_topology {
    int              n_nodes = 0;
    int              node[2] = { 0, 0 };
    std::vector<int> cpus[2];
};

static const expert_cpu_topology & expert_cpu_topo() {
    static const expert_cpu_topology t = [] {
        expert_cpu_topology r;
        const int per_node = std::max(1, (int) expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_THREADS", 16.0f));
        ULONG highest = 0;
        GetNumaHighestNodeNumber(&highest);
        for (ULONG n = 0; n <= highest && r.n_nodes < 2; n++) {
            GROUP_AFFINITY ga = {};
            if (!GetNumaNodeProcessorMaskEx((USHORT) n, &ga) || ga.Mask == 0 || ga.Group != 0) {
                continue;
            }
            std::vector<int> c;
            for (int b = 0; b < 64; b++) {
                if (ga.Mask & ((KAFFINITY) 1 << b)) {
                    c.push_back(b);
                }
            }
            if (c.size() > 1) {
                c.erase(c.begin());
            }
            if ((int) c.size() > per_node) {
                c.resize(per_node);
            }
            r.node[r.n_nodes] = (int) n;
            r.cpus[r.n_nodes] = c;
            r.n_nodes++;
        }
        return r;
    }();
    return t;
}

static void expert_cpu_pin(int cpu) {
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);
}

// ---- NUMA placement of the host experts ----

bool ggml_cuda_expert_cpu_place(const ggml_tensor * t) {
    static const bool place = expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_PLACE", 1.0f) != 0.0f;
    const expert_cpu_topology & topo = expert_cpu_topo();
    if (!ggml_cuda_expert_cpu_enabled() || !place || topo.n_nodes < 2 || t->data == nullptr || t->ne[2] <= 1 ||
            !ggml_is_contiguous(t) || strstr(t->name, "_exps.weight") == nullptr) {
        return false;
    }
    // down is split by K: device-major it is one piece of every row per device (ggml_backend_meta_dm), even split assumed
    const int     n_pieces     = strstr(t->name, "ffn_down_exps") != nullptr ? ggml_cuda_info().device_count : 1;
    const size_t  expert_bytes = t->nb[2];
    const int64_t n_rows       = t->ne[1];
    if (expert_bytes % n_pieces != 0 || (expert_bytes / n_pieces) % n_rows != 0) {
        return false;
    }
    const size_t piece_bytes = expert_bytes / n_pieces;
    const size_t half        = (size_t) (n_rows/2) * (piece_bytes / n_rows); // rows [0, n_rows/2) of a piece on node 0
    const size_t page        = 4096;
    char *       base        = (char *) t->data;
    const int    n_exp       = (int) t->ne[2];

    // per node a few threads on its CPUs touch the pages of their half first
    const int per_node = std::min<int>(8, (int) std::min(topo.cpus[0].size(), topo.cpus[1].size()));
    std::vector<std::thread> threads;
    for (int g = 0; g < 2; g++) {
        for (int r = 0; r < per_node; r++) {
            threads.emplace_back([=]() {
                expert_cpu_pin(topo.cpus[g][r]);
                for (int e = r; e < n_exp; e += per_node) {
                    for (int p = 0; p < n_pieces; p++) {
                        char * a = base + (size_t) e*expert_bytes + (size_t) p*piece_bytes + (g == 0 ? 0 : half);
                        char * b = base + (size_t) e*expert_bytes + (size_t) p*piece_bytes + (g == 0 ? half : piece_bytes);
                        char * q = (char *) (((uintptr_t) a + page - 1) & ~(uintptr_t) (page - 1));
                        for (; q < b; q += page) {
                            *(volatile char *) q = 0;
                        }
                    }
                }
            });
        }
    }
    for (std::thread & th : threads) {
        th.join();
    }
    return true;
}

// A host buffer of 1 GiB or more (the host experts) is committed but not touched, so ggml_cuda_expert_cpu_place can put
//   the pages of its expert weights on their nodes before the loader writes them; the expert cache pins it later
//   (expert_cache_register_host).
static std::mutex                                g_host_mutex;
static std::vector<std::pair<void *, size_t>>    g_host_bufs;

void * ggml_cuda_expert_cpu_host_alloc(size_t size) {
    static const bool place = expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_PLACE", 1.0f) != 0.0f;
    if (!ggml_cuda_expert_cpu_enabled() || !place || size < (1ull << 30) || expert_cpu_topo().n_nodes < 2) {
        return nullptr;
    }
    void * p = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (p == nullptr) {
        GGML_LOG_WARN("%s: VirtualAlloc of %.2f GiB failed (%lu), pinned at once instead\n", __func__, size/1073741824.0,
            (unsigned long) GetLastError());
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_host_mutex);
    g_host_bufs.push_back({p, size});
    GGML_LOG_INFO("%s: %.2f GiB of host memory placed on the NUMA nodes by first touch, pinned by the expert cache\n", __func__,
        size/1073741824.0);
    return p;
}

bool ggml_cuda_expert_cpu_host_free(void * ptr) {
    size_t size = 0;
    {
        std::lock_guard<std::mutex> lock(g_host_mutex);
        auto it = std::find_if(g_host_bufs.begin(), g_host_bufs.end(), [&](const std::pair<void *, size_t> & b) { return b.first == ptr; });
        if (it == g_host_bufs.end()) {
            return false;
        }
        size = it->second;
        g_host_bufs.erase(it);
    }
    ggml_cuda_expert_cache_unregister_host(ptr, size);
    VirtualFree(ptr, 0, MEM_RELEASE);
    return true;
}

// share of sampled pages of the matrix rows [0, n_rows/2) of each piece on the first node and of the rest on the second
static void expert_cpu_census(const ggml_cuda_expert_cpu_mat & m, const char * what, int group) {
    const expert_cpu_topology & topo = expert_cpu_topo();
    const int n = 2048;
    std::vector<PSAPI_WORKING_SET_EX_INFORMATION> wi(n);
    std::vector<int> want(n);
    uint64_t rng = 0x9e3779b97f4a7c15ull ^ (uint64_t) group;
    const int64_t n_exp_probe = 64;
    for (int i = 0; i < n; i++) {
        rng = rng*6364136223846793005ull + 1442695040888963407ull;
        const int64_t e   = (int64_t) ((rng >> 33) % n_exp_probe);
        const int     p   = (int) ((rng >> 20) % m.n_pieces);
        const int64_t row = (int64_t) ((rng >> 40) % m.n_rows);
        wi[i].VirtualAddress = (PVOID) (m.base + e*m.expert_bytes + m.piece_off[p] + row*m.piece_row_bytes[p]);
        want[i] = row < m.n_rows/2 ? topo.node[0] : topo.node[1];
    }
    if (!QueryWorkingSetEx(GetCurrentProcess(), wi.data(), (DWORD) (n*sizeof(wi[0])))) {
        return;
    }
    int ok = 0, invalid = 0;
    for (int i = 0; i < n; i++) {
        if (!wi[i].VirtualAttributes.Valid) {
            invalid++;
        } else if ((int) wi[i].VirtualAttributes.Node == want[i]) {
            ok++;
        }
    }
    GGML_LOG_INFO("%s: group %d %s: %.1f%% of sampled pages on the node of the CPU half that reads them (%d not resident)\n",
        __func__, group, what, 100.0*ok/n, invalid);
}

// ---- groups ----

struct expert_cpu_group {
    const void *             key[3] = { nullptr, nullptr, nullptr };
    ggml_cuda_expert_cpu_mat mat[3]; // gate, up, down
    int                      glu_op    = 0;
    float                    glu_limit = 0.0f;
    ggml_type                xq_type   = GGML_TYPE_COUNT; // vec_dot type of gate and up
    ggml_type                hq_type   = GGML_TYPE_COUNT; // of down
};

#define EXPERT_CPU_MAX_GROUPS 1024

static expert_cpu_group  g_groups[EXPERT_CPU_MAX_GROUPS];
static std::atomic<int>  g_n_groups{0};
static std::mutex        g_groups_mutex;
static std::vector<std::pair<std::vector<const void *>, int>> g_group_keys; // key -> group or -1

// ---- the service: a pool of pinned spinning threads, thread 0 polls the mailbox ----

struct expert_cpu_service {
    ggml_cuda_expert_cpu_mbox * mb_host = nullptr;
    ggml_cuda_expert_cpu_mbox * mb_dev  = nullptr;
    uint32_t *                  state   = nullptr; // device memory of the publishing device

    int              T = 0;
    std::vector<int> cpus, grp, rank;
    int              n_grp[2] = { 0, 0 };
    int              n_grps   = 1;

    std::atomic<uint32_t> gen{0};
    std::atomic<int>      bar_cnt{0};
    std::atomic<uint32_t> bar_gen{0};
    std::atomic<int>      quit{0};
    std::atomic<int64_t>  active_until{0};
    std::atomic<uint32_t> kick{0};
    std::atomic<uint32_t> wake{0};
    std::atomic<int>      sleeping{0};

    // the job
    const expert_cpu_group * G = nullptr;
    int   k = 0;
    int   ids[GGML_CUDA_EXPERT_CPU_MAX_K];
    float w[GGML_CUDA_EXPERT_CPU_MAX_K];
    std::vector<uint8_t> xq;
    std::vector<float>   hidden; // k x n_ff
    std::vector<uint8_t> hq;     // k x row of hq_type
    size_t               hq_row = 0;

    // stats
    std::atomic<uint64_t> n_calls{0};
    std::atomic<uint64_t> n_experts{0};
    std::atomic<uint64_t> us_total{0};
    std::atomic<uint64_t> n_slow{0}; // calls over 2 ms

    std::vector<std::thread> threads;
};

static std::atomic<expert_cpu_service *> g_svc{nullptr};
static std::once_flag       g_svc_once;

static void expert_cpu_barrier(expert_cpu_service & S) {
    const uint32_t g = S.bar_gen.load(std::memory_order_acquire);
    if (S.bar_cnt.fetch_add(1, std::memory_order_acq_rel) == S.T - 1) {
        S.bar_cnt.store(0, std::memory_order_relaxed);
        S.bar_gen.store(g + 1, std::memory_order_release);
    } else {
        while (S.bar_gen.load(std::memory_order_acquire) == g) {
            _mm_pause();
        }
    }
}

// one output row of expert e: the dot of its pieces with y (quantized to the vec_dot type yt)
static inline float expert_cpu_row_dot(const ggml_cuda_expert_cpu_mat & m, const ggml_type_traits_cpu * tr, int e, int64_t row,
        const uint8_t * y, ggml_type yt) {
    const char * x = m.base + (size_t) e*m.expert_bytes;
    float sum = 0.0f;
    for (int p = 0; p < m.n_pieces; p++) {
        float s = 0.0f;
        tr->vec_dot((int) m.piece_k[p], &s, 0, x + m.piece_off[p] + (size_t) row*m.piece_row_bytes[p], 0,
            y + ggml_row_size(yt, m.piece_k0[p]), 0, 1);
        sum += s;
    }
    return sum;
}

static void expert_cpu_work(expert_cpu_service & S, int t) {
    const expert_cpu_group &        G  = *S.G;
    const ggml_cuda_expert_cpu_mat & mg = G.mat[0];
    const ggml_cuda_expert_cpu_mat & mu = G.mat[1];
    const ggml_cuda_expert_cpu_mat & md = G.mat[2];
    const ggml_type_traits_cpu * tg = ggml_get_type_traits_cpu(mg.type);
    const ggml_type_traits_cpu * tu = ggml_get_type_traits_cpu(mu.type);
    const ggml_type_traits_cpu * td = ggml_get_type_traits_cpu(md.type);
    const int g  = S.grp[t];
    const int ng = S.n_grp[g];
    const int r  = S.rank[t];
    const int k  = S.k;
    const int64_t n_ff = mg.n_rows;

    // gate and up: each socket the rows of its half of every expert
    const int64_t f0 = S.n_grps == 2 ? (g == 0 ? 0 : n_ff/2) : 0;
    const int64_t f1 = S.n_grps == 2 ? (g == 0 ? n_ff/2 : n_ff) : n_ff;
    const int64_t n1 = (int64_t) k*(f1 - f0);
    for (int64_t i = n1*r/ng; i < n1*(r + 1)/ng; i++) {
        const int     j   = (int) (i / (f1 - f0));
        const int64_t row = f0 + i % (f1 - f0);
        const float   gv  = expert_cpu_row_dot(mg, tg, S.ids[j], row, S.xq.data(), G.xq_type);
        const float   uv  = expert_cpu_row_dot(mu, tu, S.ids[j], row, S.xq.data(), G.xq_type);
        float a = gv;
        float b = uv;
        if (G.glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
            a = std::min(a, G.glu_limit);
            b = std::max(-G.glu_limit, std::min(b, G.glu_limit));
        }
        S.hidden[(size_t) j*n_ff + row] = a / (1.0f + expf(-a)) * b;
    }
    expert_cpu_barrier(S);

    // the hidden rows in the vec_dot type of down, in chunks of 256
    const int64_t n_chunk = n_ff % 256 == 0 ? n_ff/256 : 1;
    const int64_t chunk   = n_ff / n_chunk;
    const ggml_type_traits_cpu * th = ggml_get_type_traits_cpu(G.hq_type);
    for (int64_t c = t; c < (int64_t) k*n_chunk; c += S.T) {
        const int     j  = (int) (c / n_chunk);
        const int64_t c0 = (c % n_chunk)*chunk;
        th->from_float(S.hidden.data() + (size_t) j*n_ff + c0, S.hq.data() + (size_t) j*S.hq_row + ggml_row_size(G.hq_type, c0), chunk);
    }
    expert_cpu_barrier(S);

    // down: each socket the output rows of its half, the weighted sum over the experts
    const int64_t n_out = md.n_rows;
    const int64_t o0 = S.n_grps == 2 ? (g == 0 ? 0 : n_out/2) : 0;
    const int64_t o1 = S.n_grps == 2 ? (g == 0 ? n_out/2 : n_out) : n_out;
    float * out = S.mb_host->out;
    for (int64_t row = o0 + (o1 - o0)*r/ng; row < o0 + (o1 - o0)*(r + 1)/ng; row++) {
        float acc = 0.0f;
        for (int j = 0; j < k; j++) {
            acc += S.w[j]*expert_cpu_row_dot(md, td, S.ids[j], row, S.hq.data() + (size_t) j*S.hq_row, G.hq_type);
        }
        out[row] = acc;
    }
    expert_cpu_barrier(S);
}

static void expert_cpu_worker(expert_cpu_service * S, int t) {
    expert_cpu_pin(S->cpus[t]);
    static const int prio = (int) expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_PRIORITY", (float) THREAD_PRIORITY_HIGHEST);
    SetThreadPriority(GetCurrentThread(), prio);
    uint32_t seen = S->gen.load();
    for (;;) {
        const uint32_t g = S->gen.load(std::memory_order_acquire);
        if (g != seen) {
            seen = g;
            expert_cpu_work(*S, t);
            continue;
        }
        if (S->quit.load(std::memory_order_relaxed)) {
            return;
        }
        if (S->sleeping.load(std::memory_order_acquire)) {
            uint32_t e = S->wake.load();
            if (S->sleeping.load(std::memory_order_acquire)) {
                WaitOnAddress(&S->wake, &e, sizeof(e), 50);
            }
            continue;
        }
        _mm_pause();
    }
}

static void expert_cpu_serve(expert_cpu_service * S) {
    expert_cpu_pin(S->cpus[0]);
    static const int prio = (int) expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_PRIORITY", (float) THREAD_PRIORITY_HIGHEST);
    SetThreadPriority(GetCurrentThread(), prio);
    static const bool debug = getenv("GGML_CUDA_EXPERT_CPU_DEBUG") != nullptr;
    ggml_cuda_expert_cpu_mbox * mb = S->mb_host;
    uint32_t last = mb->request;
    for (;;) {
        const uint32_t req = mb->request;
        if (req != last) {
            last = req;
            std::atomic_thread_fence(std::memory_order_acquire);
            const int64_t t0 = expert_cpu_now_us();
            const int gi = mb->group;
            const int k  = std::min<int>(mb->k, GGML_CUDA_EXPERT_CPU_MAX_K);
            if (gi >= 0 && gi < g_n_groups.load(std::memory_order_acquire) && k > 0) {
                S->G = &g_groups[gi];
                S->k = k;
                for (int j = 0; j < k; j++) {
                    S->ids[j] = mb->ids[j];
                    S->w[j]   = mb->w[j];
                }
                static const bool zero = getenv("GGML_CUDA_EXPERT_CPU_ZERO") != nullptr; // debug: a zero result
                if (zero) {
                    memset(mb->out, 0, S->G->mat[2].n_rows*sizeof(float));
                } else {
                    ggml_get_type_traits_cpu(S->G->xq_type)->from_float(mb->x, S->xq.data(), S->G->mat[0].k);
                    S->gen.fetch_add(1, std::memory_order_acq_rel);
                    expert_cpu_work(*S, 0);
                }
            }
            if (debug && S->n_calls.load() < 12 && k > 0) {
                const int64_t n = S->G->mat[2].n_rows;
                double sx = 0, so = 0, mo = 0;
                for (int64_t i = 0; i < n; i++) {
                    sx += (double) mb->x[i]*mb->x[i];
                    so += (double) mb->out[i]*mb->out[i];
                    mo  = std::max(mo, (double) fabsf(mb->out[i]));
                }
                fprintf(stderr, "expert-cpu: req %u group %d k %d ids", req, gi, k);
                for (int j = 0; j < k; j++) {
                    fprintf(stderr, " %d:%.4f", S->ids[j], S->w[j]);
                }
                fprintf(stderr, " |x| %.4g |out| %.4g max %.4g\n", sqrt(sx), sqrt(so), mo);
            }
            _mm_sfence();
            std::atomic_thread_fence(std::memory_order_release);
            mb->completed = req;
            const int64_t us = expert_cpu_now_us() - t0;
            S->n_calls.fetch_add(1, std::memory_order_relaxed);
            S->n_experts.fetch_add(k, std::memory_order_relaxed);
            S->us_total.fetch_add(us, std::memory_order_relaxed);
            if (us > 2000) {
                S->n_slow.fetch_add(1, std::memory_order_relaxed);
            }
            if (debug && S->n_calls.load() % 4096 == 0) {
                fprintf(stderr, "expert-cpu: %llu calls, %.2f experts/call, %.1f us/call, %llu over 2 ms\n",
                    (unsigned long long) S->n_calls.load(), (double) S->n_experts.load()/S->n_calls.load(),
                    (double) S->us_total.load()/S->n_calls.load(), (unsigned long long) S->n_slow.load());
            }
            continue;
        }
        if (S->quit.load(std::memory_order_relaxed)) {
            return;
        }
        if (expert_cpu_now_us() > S->active_until.load(std::memory_order_relaxed)) {
            // nothing to do for a while: sleep until a kick or a request
            S->sleeping.store(1, std::memory_order_release);
            while (!S->quit.load() && mb->request == last && expert_cpu_now_us() > S->active_until.load()) {
                uint32_t kk = S->kick.load();
                WaitOnAddress(&S->kick, &kk, sizeof(kk), 50);
            }
            S->sleeping.store(0, std::memory_order_release);
            S->wake.fetch_add(1);
            WakeByAddressAll(&S->wake);
            continue;
        }
        _mm_pause();
    }
}

static void expert_cpu_stop() {
    expert_cpu_service * S = g_svc.load();
    if (S == nullptr) {
        return;
    }
    S->quit.store(1);
    S->kick.fetch_add(1);
    WakeByAddressAll(&S->kick);
    S->wake.fetch_add(1);
    WakeByAddressAll(&S->wake);
    for (std::thread & th : S->threads) {
        th.join();
    }
    S->threads.clear();
}

void ggml_cuda_expert_cpu_init() {
    if (!ggml_cuda_expert_cpu_enabled()) {
        return;
    }
    std::call_once(g_svc_once, [] {
        const expert_cpu_topology & topo = expert_cpu_topo();
        if (topo.n_nodes == 0) {
            return;
        }
        ggml_cpu_init();
        expert_cpu_service * S = new expert_cpu_service();

        int cur = 0;
        CUDA_CHECK(cudaGetDevice(&cur));
        ggml_cuda_set_device(ggml_cuda_expert_cpu_device());
        void * p = nullptr;
        CUDA_CHECK(cudaHostAlloc(&p, sizeof(ggml_cuda_expert_cpu_mbox), cudaHostAllocPortable | cudaHostAllocMapped));
        memset(p, 0, sizeof(ggml_cuda_expert_cpu_mbox));
        S->mb_host = (ggml_cuda_expert_cpu_mbox *) p;
        void * dp = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&dp, p, 0));
        S->mb_dev = (ggml_cuda_expert_cpu_mbox *) dp;
        CUDA_CHECK(cudaMalloc(&S->state, 12*sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(S->state, 0, 12*sizeof(uint32_t)));
        CUDA_CHECK(cudaDeviceSynchronize());
        ggml_cuda_set_device(cur);

        S->n_grps = std::min(topo.n_nodes, 2);
        for (int g = 0; g < S->n_grps; g++) {
            for (int c : topo.cpus[g]) {
                S->cpus.push_back(c);
                S->grp.push_back(g);
                S->rank.push_back(S->n_grp[g]++);
            }
        }
        S->T = (int) S->cpus.size();
        S->xq.resize(1 << 20);
        S->hidden.resize((size_t) GGML_CUDA_EXPERT_CPU_MAX_K*65536);
        S->hq.resize((size_t) GGML_CUDA_EXPERT_CPU_MAX_K*(1 << 17));
        S->hq_row = 1 << 17;
        S->active_until.store(expert_cpu_now_us() + 1000000);

        S->threads.emplace_back(expert_cpu_serve, S);
        for (int t = 1; t < S->T; t++) {
            S->threads.emplace_back(expert_cpu_worker, S, t);
        }
        g_svc = S;
        atexit(expert_cpu_stop);

        std::string cl;
        for (int g = 0; g < S->n_grps; g++) {
            cl += (g ? ", node " : "node ") + std::to_string(topo.node[g]) + ": " + std::to_string(topo.cpus[g].front()) + "-" +
                std::to_string(topo.cpus[g].back());
        }
        GGML_LOG_INFO("%s: CPU experts on %d threads (%s), results added on device %d\n", __func__, S->T, cl.c_str(),
            ggml_cuda_expert_cpu_device());
    });
}

void ggml_cuda_expert_cpu_kick() {
    expert_cpu_service * S = g_svc;
    if (S == nullptr) {
        return;
    }
    static const int64_t spin_us = (int64_t) (expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_SPIN_MS", 1000.0f)*1000.0f);
    S->active_until.store(expert_cpu_now_us() + spin_us, std::memory_order_relaxed);
    if (S->sleeping.load(std::memory_order_acquire)) {
        S->kick.fetch_add(1);
        WakeByAddressAll(&S->kick);
    }
}

static bool expert_cpu_mat_ok(const ggml_cuda_expert_cpu_mat & m, ggml_type yt) {
    const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(m.type);
    if (tr == nullptr || tr->vec_dot == nullptr || tr->vec_dot_type != yt) {
        return false;
    }
    const int64_t bs = ggml_blck_size(yt);
    for (int p = 0; p < m.n_pieces; p++) {
        if (m.piece_k[p] % bs != 0 || m.piece_k0[p] % bs != 0 || m.piece_k[p] % ggml_blck_size(m.type) != 0) {
            return false;
        }
    }
    return m.n_rows > 0 && m.k > 0;
}

int ggml_cuda_expert_cpu_group(const ggml_tensor * gate_pool, const ggml_tensor * up_pool, const ggml_tensor * down_pool,
        int glu_op, float glu_limit, int64_t n_embd) {
    if (!ggml_cuda_expert_cpu_enabled()) {
        return -1;
    }
    expert_cpu_group G;
    const ggml_tensor * pools[3] = { gate_pool, up_pool, down_pool };
    // gate and up give 0 for a CPU expert, so their GLU is 0 and down may read any slot
    bool ok = (glu_op == GGML_GLU_OP_SWIGLU || glu_op == GGML_GLU_OP_SWIGLU_CLAMP) &&
        ggml_cuda_expert_cache_zero_slot(gate_pool) && ggml_cuda_expert_cache_zero_slot(up_pool);
    for (int i = 0; i < 3 && ok; i++) {
        ok = ggml_cuda_expert_cache_cpu_mat(pools[i], G.mat[i]);
        G.key[i] = G.mat[i].base;
    }
    const std::vector<const void *> key = { G.key[0], G.key[1], G.key[2] };

    std::lock_guard<std::mutex> lock(g_groups_mutex);
    for (const auto & kv : g_group_keys) {
        if (kv.first == key) {
            return kv.second;
        }
    }
    if (ok) {
        const ggml_type_traits_cpu * tg = ggml_get_type_traits_cpu(G.mat[0].type);
        const ggml_type_traits_cpu * td = ggml_get_type_traits_cpu(G.mat[2].type);
        G.xq_type   = tg->vec_dot_type;
        G.hq_type   = td->vec_dot_type;
        G.glu_op    = glu_op;
        G.glu_limit = glu_limit;
        ok = expert_cpu_mat_ok(G.mat[0], G.xq_type) && expert_cpu_mat_ok(G.mat[1], G.xq_type) &&
             expert_cpu_mat_ok(G.mat[2], G.hq_type) && G.mat[0].k == n_embd && G.mat[1].k == n_embd &&
             G.mat[0].n_rows == G.mat[1].n_rows && G.mat[2].k == G.mat[0].n_rows && G.mat[2].n_rows == n_embd &&
             n_embd <= GGML_CUDA_EXPERT_CPU_MAX_EMBD && n_embd % 4 == 0 && G.mat[0].n_rows <= 65536 &&
             ggml_row_size(G.hq_type, G.mat[0].n_rows) <= (1 << 17) && ggml_row_size(G.xq_type, n_embd) <= (1 << 20) &&
             g_n_groups.load() < EXPERT_CPU_MAX_GROUPS;
    }
    int idx = -1;
    if (ok) {
        idx = g_n_groups.load();
        g_groups[idx] = G;
        if (getenv("GGML_CUDA_EXPERT_CPU_DEBUG") != nullptr) {
            for (int i = 0; i < 3; i++) {
                const ggml_cuda_expert_cpu_mat & m = G.mat[i];
                fprintf(stderr, "expert-cpu: group %d %s %s base %p expert_bytes %zu rows %lld k %lld pieces %d:", idx, pools[i]->name,
                    ggml_type_name(m.type), (const void *) m.base, m.expert_bytes, (long long) m.n_rows, (long long) m.k, m.n_pieces);
                for (int p = 0; p < m.n_pieces; p++) {
                    fprintf(stderr, " [off %zu rb %zu k0 %lld k %lld]", m.piece_off[p], m.piece_row_bytes[p], (long long) m.piece_k0[p],
                        (long long) m.piece_k[p]);
                }
                fprintf(stderr, "\n");
            }
        }
        g_n_groups.store(idx + 1, std::memory_order_release);
        static bool census_done = false;
        if (!census_done) {
            census_done = true;
            expert_cpu_census(G.mat[0], "gate", idx);
            expert_cpu_census(G.mat[2], "down", idx);
        }
    } else {
        static bool reported = false;
        if (!reported) {
            reported = true;
            GGML_LOG_WARN("%s: the CPU cannot take the experts of %s (types %s/%s/%s): they stay on the device\n", __func__,
                gate_pool->name, ggml_type_name(gate_pool->type), ggml_type_name(up_pool->type), ggml_type_name(down_pool->type));
        }
    }
    if (g_svc == nullptr) {
        idx = -1; // ggml_cuda_expert_cpu_init found no CPUs
    }
    g_group_keys.push_back({ key, idx });
    return idx;
}

void ggml_cuda_expert_cpu_args_get(int device, int group, const float * x, const float * w, int n_embd,
        ggml_cuda_expert_cpu_args & a) {
    a = {};
    const expert_cpu_service * S = g_svc.load();
    if (group < 0 || S == nullptr) {
        return;
    }
    static const float copy_us   = expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_COPY_US",   551.0f);
    static const float cpu_a_us  = expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_A_US",      1.0f);
    static const float cpu_b_us  = expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_B_US",      174.0f);
    static const float post_us   = expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_POST_US",   27.0f);
    static const float slack_rel = expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_SLACK",     0.10f);
    static const float slack_us  = expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_SLACK_US",  0.0f);
    static const int   min_copy  = (int) expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_MIN_COPY", 0.0f);
    const bool publish = device == ggml_cuda_expert_cpu_device();
    a.mbox      = publish ? S->mb_dev : nullptr;
    a.state     = publish ? S->state : nullptr;
    a.x         = x;
    a.w         = w;
    a.group     = group;
    a.n_embd    = n_embd;
    a.copy_us   = copy_us;
    a.cpu_a_us  = cpu_a_us;
    a.cpu_b_us  = cpu_b_us;
    a.post_us   = post_us;
    a.slack_rel = slack_rel;
    a.slack_us  = slack_us;
    a.min_copy  = min_copy;
    a.on        = 1;
}

void ggml_cuda_expert_cpu_merge(cudaStream_t stream, float * dst, int n_embd) {
    const expert_cpu_service * S = g_svc.load();
    GGML_ASSERT(S != nullptr && n_embd % 4 == 0 && (uintptr_t) dst % 16 == 0);
    const int nthreads = 128;
    const int nblocks  = std::max(1, std::min(8, (n_embd/4 + nthreads - 1)/nthreads));
    expert_cpu_wait_add<<<nblocks, nthreads, 0, stream>>>(S->mb_dev, S->state, dst, n_embd);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_expert_cpu_stats_print() {
    expert_cpu_service * S = g_svc;
    if (S == nullptr || S->n_calls.load() == 0) {
        return;
    }
    unsigned long long st[4] = { 0, 0, 0, 0 };
    int cur = 0;
    if (cudaGetDevice(&cur) == cudaSuccess) {
        ggml_cuda_set_device(ggml_cuda_expert_cpu_device());
        if (cudaMemcpy(st, S->state + 4, sizeof(st), cudaMemcpyDeviceToHost) == cudaSuccess && st[3] > 0) {
            fprintf(stderr, "expert-cpu: %llu waits, GPU work between request and wait %.1f us, wait %.1f us (means)\n", st[3],
                st[1]/1e3/st[3], st[2]/1e3/st[3]);
        }
        (void) cudaGetLastError();
        ggml_cuda_set_device(cur);
    }
    fprintf(stderr, "expert-cpu: %llu calls, %llu experts (%.3f/call), %.1f us/call, %llu calls over 2 ms\n",
        (unsigned long long) S->n_calls.load(), (unsigned long long) S->n_experts.load(),
        (double) S->n_experts.load()/S->n_calls.load(), (double) S->us_total.load()/S->n_calls.load(),
        (unsigned long long) S->n_slow.load());
}

#else // _WIN32

bool ggml_cuda_expert_cpu_enabled() {
    return false;
}

int ggml_cuda_expert_cpu_device() {
    return 0;
}

bool ggml_cuda_expert_cpu_place(const ggml_tensor * tensor) {
    GGML_UNUSED(tensor);
    return false;
}

void * ggml_cuda_expert_cpu_host_alloc(size_t size) {
    GGML_UNUSED(size);
    return nullptr;
}

bool ggml_cuda_expert_cpu_host_free(void * ptr) {
    GGML_UNUSED(ptr);
    return false;
}

int ggml_cuda_expert_cpu_group(const ggml_tensor * gate_pool, const ggml_tensor * up_pool, const ggml_tensor * down_pool,
        int glu_op, float glu_limit, int64_t n_embd) {
    GGML_UNUSED_VARS(gate_pool, up_pool, down_pool, glu_op, glu_limit, n_embd);
    return -1;
}

void ggml_cuda_expert_cpu_args_get(int device, int group, const float * x, const float * w, int n_embd,
        ggml_cuda_expert_cpu_args & args) {
    GGML_UNUSED_VARS(device, group, x, w, n_embd);
    args = {};
}

void ggml_cuda_expert_cpu_init() {
}

void ggml_cuda_expert_cpu_kick() {
}

void ggml_cuda_expert_cpu_merge(cudaStream_t stream, float * dst, int n_embd) {
    GGML_UNUSED_VARS(stream, dst, n_embd);
}

void ggml_cuda_expert_cpu_stats_print() {
}

#endif // _WIN32
