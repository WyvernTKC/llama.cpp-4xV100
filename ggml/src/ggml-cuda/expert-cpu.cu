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
#include <new>
#include <string>
#include <thread>
#include <vector>

GGML_CUDA_SPIN_SETTER(expert_cpu)

// waits for the CPU result the plan asked for (state[1], 0 = none) and adds it to dst; n % 4 == 0, dst 16-byte aligned.
//   parts = 2: the sockets' partial sums out and out1, each with its own completion. Stats in state (u64 from [4]): request
//   time, GPU work and wait sums, waits, the waits under 4 us (the GPU was slower) and their GPU work, a histogram of the
//   waits (< 4, 20, 50, 100, 200, more us) and, at state + 32, per split of the plan (CPU experts 1..EXPERT_CPU_BIN_K x copies
//   0..EXPERT_CPU_BIN_C - 1, the last of each the rest) the waits, the GPU work and the wait: these two with prof only
#define EXPERT_CPU_BIN_K 6
#define EXPERT_CPU_BIN_C 3
#define EXPERT_CPU_STATE_U32 (32 + 2*3*EXPERT_CPU_BIN_K*EXPERT_CPU_BIN_C)
static __global__ void expert_cpu_wait_add(const ggml_cuda_expert_cpu_mbox * mb, uint32_t * state, float * dst, const int n,
        const int parts, const int prof) {
    __shared__ uint32_t s_seq;
    if (threadIdx.x == 0) {
        const uint32_t seq = state[1];
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        // both flags in one round trip over the link
        if (seq != 0 && !ggml_cuda_spin_until([&]() {
                    const uint32_t c0 = mb->completed;
                    const uint32_t c1 = parts < 2 ? seq : mb->completed1;
                    return c0 == seq && c1 == seq; })) {
            ggml_cuda_spin_trap(GGML_CUDA_SPIN_SITE_EXPERT_CPU, -1, GGML_CUDA_SPIN_PEER_HOST);
        }
        s_seq = seq;
        if (seq != 0 && blockIdx.x == 0) {
            // the GPU work between the request and this wait, and the wait
            unsigned long long t1;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
            unsigned long long * st = (unsigned long long *) (state + 4);
            const unsigned long long w = t1 - t0;
            st[1] += t0 - st[0];
            st[2] += w;
            st[3] += 1;
            if (w < 4000) {
                // the GPU was the slower side (the first poll is a round trip over the link): how often, its work then
                st[4] += 1;
                st[5] += t0 - st[0];
            }
            if (prof) {
                st[6 + (w < 4000 ? 0 : w < 20000 ? 1 : w < 50000 ? 2 : w < 100000 ? 3 : w < 200000 ? 4 : 5)] += 1;
                // by the plan's split (state[2]: CPU experts, copies << 8)
                const uint32_t kc = state[2];
                const int b = min((int) (kc & 0xff), EXPERT_CPU_BIN_K) - 1 + EXPERT_CPU_BIN_K*min((int) (kc >> 8), EXPERT_CPU_BIN_C - 1);
                unsigned long long * bins = (unsigned long long *) (state + 32);
                bins[3*b + 0] += 1;
                bins[3*b + 1] += t0 - st[0];
                bins[3*b + 2] += w;
            }
        }
    }
    __syncthreads();
    if (s_seq == 0) {
        return;
    }
    __threadfence_system();
    const float4 * src  = (const float4 *) mb->out;
    const float4 * src1 = (const float4 *) mb->out1;
    float4 *       d4   = (float4 *) dst;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n/4; i += gridDim.x*blockDim.x) {
        float4 v = __ldcv(src + i);
        if (parts == 2) {
            const float4 v1 = __ldcv(src1 + i);
            v.x += v1.x;
            v.y += v1.y;
            v.z += v1.z;
            v.w += v1.w;
        }
        float4 a = d4[i];
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

// The sockets split the down projection by K (expert_cpu_work_k): each one reads the down pieces of its half of the
//   hidden, which it computed itself, so the sockets never wait for each other. GGML_CUDA_EXPERT_CPU_V1=1: by output
//   rows, with barriers across both sockets (expert_cpu_work_rows).
static bool expert_cpu_ksplit() {
    static const bool k = getenv("GGML_CUDA_EXPERT_CPU_V1") == nullptr && expert_cpu_topo().n_nodes >= 2;
    return k;
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
    const size_t page        = 4096;
    char *       base        = (char *) t->data;
    const int    n_exp       = (int) t->ne[2];
    // the K split: the first half of the pieces on node 0, whole (the devices of the first half sit on socket 0 too)
    const bool   by_piece    = expert_cpu_ksplit() && n_pieces >= 2 && n_pieces % 2 == 0;
    // else rows [0, n_rows/2) of a piece on node 0
    const size_t half        = (size_t) (n_rows/2) * (piece_bytes / n_rows);

    // per node a few threads on its CPUs touch the pages of their half first
    const int per_node = std::min<int>(8, (int) std::min(topo.cpus[0].size(), topo.cpus[1].size()));
    std::vector<std::thread> threads;
    for (int g = 0; g < 2; g++) {
        for (int r = 0; r < per_node; r++) {
            threads.emplace_back([=]() {
                expert_cpu_pin(topo.cpus[g][r]);
                for (int e = r; e < n_exp; e += per_node) {
                    for (int p = 0; p < n_pieces; p++) {
                        if (by_piece && (p < n_pieces/2) != (g == 0)) {
                            continue;
                        }
                        const size_t lo = by_piece ? 0 : (g == 0 ? 0 : half);
                        const size_t hi = by_piece ? piece_bytes : (g == 0 ? half : piece_bytes);
                        char * a = base + (size_t) e*expert_bytes + (size_t) p*piece_bytes + lo;
                        char * b = base + (size_t) e*expert_bytes + (size_t) p*piece_bytes + hi;
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
        const bool first = expert_cpu_ksplit() && m.n_pieces >= 2 && m.n_pieces % 2 == 0 ? p < m.n_pieces/2 : row < m.n_rows/2;
        want[i] = first ? topo.node[0] : topo.node[1];
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

// K split: the part of a down piece in the K range [f0, f0 + nh) of a socket
struct expert_cpu_dseg {
    size_t x_off;     // in an expert: the piece, then its first input in a row
    size_t row_bytes; // of the piece
    size_t y_off;     // in the socket's quantized hidden row
    int    n;         // inputs
};

struct expert_cpu_group {
    const void *             key[3] = { nullptr, nullptr, nullptr };
    ggml_cuda_expert_cpu_mat mat[3]; // gate, up, down
    int                      glu_op    = 0;
    float                    glu_limit = 0.0f;
    ggml_type                xq_type   = GGML_TYPE_COUNT; // vec_dot type of gate and up
    ggml_type                hq_type   = GGML_TYPE_COUNT; // of down
    // K split: the hidden rows [f0, f0 + nh) of each socket and the down pieces they meet
    int64_t                  f0[2]    = { 0, 0 };
    int64_t                  nh[2]    = { 0, 0 };
    int                      n_seg[2] = { 0, 0 };
    expert_cpu_dseg          seg[2][GGML_CUDA_MAX_DEVICES];
};

#define EXPERT_CPU_MAX_GROUPS 1024

static expert_cpu_group  g_groups[EXPERT_CPU_MAX_GROUPS];
static std::atomic<int>  g_n_groups{0};
static std::mutex        g_groups_mutex;
static std::vector<std::pair<std::vector<const void *>, int>> g_group_keys; // key -> group or -1

// x in the vec_dot type of gate and up, by each socket's leader before its threads start: quantize_row_q8_K is the scalar
//   reference on x86 (12.3 us at n_embd 4096); this is the same bytes in 2.4 us (checked against it over 20000 random
//   rows with ties and zero blocks, and once at start, see expert_cpu_q8_K_ok). The first value of the largest magnitude
//   sets the scale, as there.
static void expert_cpu_quantize_q8_K(const float * x, block_q8_K * y, const int64_t k) {
    const __m512  abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const __m512  magic    = _mm512_set1_ps(12582912.f);
    const __m512i mant     = _mm512_set1_epi32(0x007fffff);
    const __m512i off      = _mm512_set1_epi32(0x00400000);
    const __m512i q_max    = _mm512_set1_epi32(127);
    for (int64_t i = 0; i < k/QK_K; i++, x += QK_K) {
        __m512 am = _mm512_setzero_ps();
        for (int j = 0; j < QK_K; j += 16) {
            am = _mm512_max_ps(am, _mm512_and_ps(_mm512_loadu_ps(x + j), abs_mask));
        }
        const float amax = _mm512_reduce_max_ps(am);
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            continue;
        }
        const __m512 vmax = _mm512_set1_ps(amax);
        float max = 0.0f;
        for (int j = 0; j < QK_K; j += 16) {
            const __mmask16 m = _mm512_cmp_ps_mask(_mm512_and_ps(_mm512_loadu_ps(x + j), abs_mask), vmax, _CMP_EQ_OQ);
            if (m) {
                max = x[j + _tzcnt_u32(m)];
                break;
            }
        }
        const float  iscale = -127.f/max;
        const __m512 vs     = _mm512_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 16) {
            const __m512 v = _mm512_add_ps(_mm512_mul_ps(vs, _mm512_loadu_ps(x + j)), magic);
            __m512i      q = _mm512_sub_epi32(_mm512_and_si512(_mm512_castps_si512(v), mant), off);
            q = _mm512_min_epi32(q, q_max);
            _mm_storeu_si128((__m128i *) (y[i].qs + j), _mm512_cvtepi32_epi8(q));
            y[i].bsums[j/16] = (int16_t) _mm512_reduce_add_epi32(q);
        }
        y[i].d = 1/iscale;
    }
}

// AVX-512F in the CPU and the OS, and the same bytes as the CPU backend's quantize_row_q8_K on a random row
static bool expert_cpu_q8_K_ok() {
    static const bool ok = [] {
        int r[4];
        __cpuid(r, 0);
        if (r[0] < 7) {
            return false;
        }
        __cpuidex(r, 1, 0);
        const bool osxsave = (r[2] & (1 << 27)) != 0;
        __cpuidex(r, 7, 0);
        const bool avx512f = (r[1] & (1 << 16)) != 0;
        const bool bmi1    = (r[1] & (1 << 3)) != 0;
        if (!osxsave || !avx512f || !bmi1 || (_xgetbv(0) & 0xe6) != 0xe6) {
            return false;
        }
        const int64_t k = 4*QK_K;
        std::vector<float> x(k);
        uint64_t s = 0x9e3779b97f4a7c15ull;
        for (int64_t i = 0; i < k; i++) {
            s = s*6364136223846793005ull + 1442695040888963407ull;
            x[i] = ((float) (int64_t) (s >> 40) - 8388608.0f)/8388608.0f*(i < QK_K ? 1e-3f : 3.0f);
        }
        x[QK_K + 7] = -x[QK_K + 3]; // a tie in magnitude
        std::vector<block_q8_K> a(k/QK_K), b(k/QK_K);
        memset(a.data(), 0, a.size()*sizeof(block_q8_K));
        memset(b.data(), 0, b.size()*sizeof(block_q8_K));
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), a.data(), k);
        expert_cpu_quantize_q8_K(x.data(), b.data(), k);
        const bool same = memcmp(a.data(), b.data(), a.size()*sizeof(block_q8_K)) == 0;
        if (!same) {
            GGML_LOG_WARN("%s: the AVX-512 Q8_K quantizer differs from the CPU backend's, using that one\n", __func__);
        }
        return same;
    }();
    return ok;
}

static void expert_cpu_quantize_x(const expert_cpu_group & G, const float * x, uint8_t * xq) {
    static const bool fast = getenv("GGML_CUDA_EXPERT_CPU_REF_Q8K") == nullptr;
    if (fast && G.xq_type == GGML_TYPE_Q8_K && G.mat[0].k % QK_K == 0 && expert_cpu_q8_K_ok()) {
        expert_cpu_quantize_q8_K(x, (block_q8_K *) xq, G.mat[0].k);
    } else {
        ggml_get_type_traits_cpu(G.xq_type)->from_float(x, xq, G.mat[0].k);
    }
}

// ---- the service: a pool of pinned spinning threads ----

// GGML_CUDA_EXPERT_CPU_PROF=1: the time stamps (rdtsc) of every thread in every call, relative to the request, summed per
//   socket and printed with the stats: start, end of each phase, exit of each barrier; and per thread its time in the
//   gate/up and down phases.
#define EXPERT_CPU_PROF_EV 7

struct alignas(64) expert_cpu_thread_prof {
    uint64_t ev[EXPERT_CPU_PROF_EV];
    double   ph_gu   = 0.0; // us in gate/up, summed over calls
    double   ph_down = 0.0;
    uint64_t n       = 0;
};

// what the threads of one socket share (K split), in memory of its node
struct expert_cpu_socket {
    alignas(64) std::atomic<uint32_t> gen{0}; // bumped by the socket's leader for each call
    alignas(64) std::atomic<int>      bar_cnt{0};
    alignas(64) std::atomic<uint32_t> bar_gen{0};
    alignas(64) std::atomic<int>      done_cnt{0};
    alignas(64) std::atomic<int>      sleeping{0};
    std::atomic<uint32_t>             wake{0};

    // the job, written by the leader before gen
    alignas(64) const expert_cpu_group * G = nullptr;
    int      k   = 0;
    uint32_t req = 0;
    int      ids[GGML_CUDA_EXPERT_CPU_MAX_K];
    float    w[GGML_CUDA_EXPERT_CPU_MAX_K];
    uint64_t t_seen = 0; // when the leader saw the request
    uint64_t t_q    = 0; // when it had quantized x
    uint64_t t_done = 0; // when the last thread had written its rows

    uint8_t * xq     = nullptr;
    float   * hidden = nullptr; // k x nh
    uint8_t * hq     = nullptr; // k x hq_row

    // GGML_CUDA_EXPERT_CPU_PROF, by the last thread of each call (rows split: by thread 0)
    double   prof_mean[EXPERT_CPU_PROF_EV] = {};
    double   prof_max[EXPERT_CPU_PROF_EV]  = {};
    double   prof_q    = 0.0;
    double   prof_done = 0.0;
    uint64_t prof_n    = 0;
    // the time from the GPU's stamp of the request to the leader's sight of it, on two clocks that drift apart: per window of
    //   64 calls its excess over the window's least
    double   lat_win_sum = 0.0;
    double   lat_win_min = 0.0;
    uint64_t lat_win_n   = 0;
    double   lat_excess  = 0.0;
    uint64_t lat_n       = 0;
    uint64_t tsc_total = 0;
    uint64_t n_calls   = 0;
};

#define EXPERT_CPU_HQ_ROW   (1 << 17)
#define EXPERT_CPU_XQ_BYTES (1 << 20)
#define EXPERT_CPU_HIDDEN   65536

struct expert_cpu_service {
    ggml_cuda_expert_cpu_mbox * mb_host = nullptr;
    ggml_cuda_expert_cpu_mbox * mb_dev  = nullptr;
    uint32_t *                  state   = nullptr; // device memory of the publishing device

    int              T = 0;
    std::vector<int> cpus, grp, rank;
    int              n_grp[2] = { 0, 0 };
    int              n_grps   = 1;
    int              lead[2]  = { 0, 0 }; // the thread of each socket that polls the mailbox
    bool             ksplit   = false;
    bool             prof     = false;
    double           tsc_us   = 1.0;      // rdtsc ticks per us

    expert_cpu_socket *      sock[2] = { nullptr, nullptr };
    expert_cpu_thread_prof * tp      = nullptr; // [T]
    alignas(64) std::atomic<uint32_t> sock_done{0}; // K split: the sockets that finished; the second sums up the call

    // rows split: one barrier over all threads, thread 0 polls the mailbox
    alignas(64) std::atomic<uint32_t> gen{0};
    alignas(64) std::atomic<int>      bar_cnt{0};
    alignas(64) std::atomic<uint32_t> bar_gen{0};
    alignas(64) std::atomic<int>      sleeping{0};
    std::atomic<uint32_t>             wake{0};
    const expert_cpu_group * G = nullptr;
    int   k = 0;
    int   ids[GGML_CUDA_EXPERT_CPU_MAX_K];
    float w[GGML_CUDA_EXPERT_CPU_MAX_K];
    uint64_t t_seen = 0;
    std::vector<uint8_t> xq;
    std::vector<float>   hidden; // k x n_ff
    std::vector<uint8_t> hq;     // k x row of hq_type
    size_t               hq_row = 0;

    std::atomic<int>      quit{0};
    std::atomic<int64_t>  active_until{0};
    std::atomic<uint32_t> kick{0};

    // stats: per call from the request to the last result
    std::atomic<uint64_t> n_calls{0};
    std::atomic<uint64_t> n_experts{0};
    std::atomic<uint64_t> us_total{0};
    std::atomic<uint64_t> n_slow{0}; // calls over 2 ms
    std::atomic<uint64_t> k_n[EXPERT_CPU_BIN_K + 1]  = {}; // by CPU experts 1..EXPERT_CPU_BIN_K (the last: or more)
    std::atomic<uint64_t> k_us[EXPERT_CPU_BIN_K + 1] = {};

    std::vector<std::thread> threads;
};

static std::atomic<expert_cpu_service *> g_svc{nullptr};
static std::once_flag       g_svc_once;

#define EXPERT_CPU_MARK(S, t, e) do { if ((S).prof) { (S).tp[t].ev[e] = __rdtsc(); } } while (0)

// GGML_CUDA_EXPERT_CPU_PROF: the leader saw the request at tsc, see lat_win_sum
static inline void expert_cpu_lat_req(const expert_cpu_service & S, expert_cpu_socket & K, const uint64_t tsc) {
    const double raw = (double) tsc*1000.0/S.tsc_us - (double) S.mb_host->t_pub; // ns, plus an unknown offset
    K.lat_win_min  = K.lat_win_n == 0 ? raw : std::min(K.lat_win_min, raw);
    K.lat_win_sum += raw;
    if (++K.lat_win_n == 64) {
        K.lat_excess += K.lat_win_sum - (double) K.lat_win_n*K.lat_win_min;
        K.lat_n      += K.lat_win_n;
        K.lat_win_n   = 0;
        K.lat_win_sum = 0.0;
    }
}

static inline void expert_cpu_barrier_n(std::atomic<int> & cnt, std::atomic<uint32_t> & gen, const int n) {
    const uint32_t g = gen.load(std::memory_order_acquire);
    if (cnt.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
        cnt.store(0, std::memory_order_relaxed);
        gen.store(g + 1, std::memory_order_release);
    } else {
        while (gen.load(std::memory_order_acquire) == g) {
            _mm_pause();
        }
    }
}

static void expert_cpu_barrier(expert_cpu_service & S) {
    expert_cpu_barrier_n(S.bar_cnt, S.bar_gen, S.T);
}

static void expert_cpu_record(expert_cpu_service & S, const double us, const int k) {
    S.n_calls.fetch_add(1, std::memory_order_relaxed);
    S.n_experts.fetch_add(k, std::memory_order_relaxed);
    S.us_total.fetch_add((uint64_t) us, std::memory_order_relaxed);
    if (us > 2000.0) {
        S.n_slow.fetch_add(1, std::memory_order_relaxed);
    }
    const int b = std::min(std::max(k, 1), EXPERT_CPU_BIN_K);
    S.k_n[b].fetch_add(1, std::memory_order_relaxed);
    S.k_us[b].fetch_add((uint64_t) us, std::memory_order_relaxed);
}

// the stamps of the threads of socket g in this call relative to t0, into its sums
static void expert_cpu_prof_sum(expert_cpu_service & S, expert_cpu_socket & K, const int g, const uint64_t t0, const int n_ev) {
    double sum[EXPERT_CPU_PROF_EV] = {};
    double mx[EXPERT_CPU_PROF_EV]  = {};
    int n = 0;
    for (int t = 0; t < S.T; t++) {
        if (S.grp[t] != g) {
            continue;
        }
        n++;
        for (int e = 0; e < n_ev; e++) {
            const double v = (double) (int64_t) (S.tp[t].ev[e] - t0)/S.tsc_us;
            sum[e] += v;
            mx[e]   = std::max(mx[e], v);
        }
    }
    for (int e = 0; e < n_ev; e++) {
        K.prof_mean[e] += sum[e]/std::max(n, 1);
        K.prof_max[e]  += mx[e];
    }
    K.prof_n++;
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

static inline float expert_cpu_glu(const expert_cpu_group & G, float a, float b) {
    if (G.glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
        a = std::min(a, G.glu_limit);
        b = std::max(-G.glu_limit, std::min(b, G.glu_limit));
    }
    return a / (1.0f + expf(-a)) * b;
}

// The rows split (GGML_CUDA_EXPERT_CPU_V1=1): gate/up rows and down output rows each by halves per socket, so down needs
//   the hidden of both sockets and every phase ends in a barrier over all threads.
static void expert_cpu_work_rows(expert_cpu_service & S, int t) {
    EXPERT_CPU_MARK(S, t, 0);
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
        S.hidden[(size_t) j*n_ff + row] = expert_cpu_glu(G, gv, uv);
    }
    EXPERT_CPU_MARK(S, t, 1);
    expert_cpu_barrier(S);
    EXPERT_CPU_MARK(S, t, 2);

    // the hidden rows in the vec_dot type of down, in chunks of 256
    const int64_t n_chunk = n_ff % 256 == 0 ? n_ff/256 : 1;
    const int64_t chunk   = n_ff / n_chunk;
    const ggml_type_traits_cpu * th = ggml_get_type_traits_cpu(G.hq_type);
    for (int64_t c = t; c < (int64_t) k*n_chunk; c += S.T) {
        const int     j  = (int) (c / n_chunk);
        const int64_t c0 = (c % n_chunk)*chunk;
        th->from_float(S.hidden.data() + (size_t) j*n_ff + c0, S.hq.data() + (size_t) j*S.hq_row + ggml_row_size(G.hq_type, c0), chunk);
    }
    EXPERT_CPU_MARK(S, t, 3);
    expert_cpu_barrier(S);
    EXPERT_CPU_MARK(S, t, 4);

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
    EXPERT_CPU_MARK(S, t, 5);
    expert_cpu_barrier(S);
    EXPERT_CPU_MARK(S, t, 6);
}

// the last thread of socket g in a call (K split), after its completion: the stats
static void expert_cpu_call_done(expert_cpu_service & S, expert_cpu_socket & K, const int g) {
    K.t_done = __rdtsc();
    K.n_calls++;
    K.tsc_total += K.t_done - K.t_seen;
    if (S.prof) {
        expert_cpu_prof_sum(S, K, g, K.t_seen, 6);
        K.prof_q    += (double) (K.t_q - K.t_seen)/S.tsc_us;
        K.prof_done += (double) (K.t_done - K.t_seen)/S.tsc_us;
    }
    if (S.n_grps < 2) {
        expert_cpu_record(S, (double) (K.t_done - K.t_seen)/S.tsc_us, K.k);
        return;
    }
    if (S.sock_done.fetch_add(1, std::memory_order_acq_rel) % 2 == 1) {
        // the second socket of the call: from the first request seen to the last result
        const expert_cpu_socket & A = *S.sock[0];
        const expert_cpu_socket & B = *S.sock[1];
        const uint64_t t0 = std::min(A.t_seen, B.t_seen);
        const uint64_t t1 = std::max(A.t_done, B.t_done);
        expert_cpu_record(S, (double) (t1 - t0)/S.tsc_us, K.k);
    }
}

// The K split: socket g computes the hidden rows [f0, f0 + nh) of every expert (its half of gate and up, on its node),
//   quantizes them and multiplies them with the down pieces of the same K range, for every output row: a partial sum the
//   GPU adds to the other socket's. The barriers are within the socket, and each socket signals its own completion.
static void expert_cpu_work_k(expert_cpu_service & S, expert_cpu_socket & K, const int t) {
    EXPERT_CPU_MARK(S, t, 0);
    const expert_cpu_group &        G  = *K.G;
    const ggml_cuda_expert_cpu_mat & mg = G.mat[0];
    const ggml_cuda_expert_cpu_mat & mu = G.mat[1];
    const ggml_cuda_expert_cpu_mat & md = G.mat[2];
    const ggml_type_traits_cpu * tg = ggml_get_type_traits_cpu(mg.type);
    const ggml_type_traits_cpu * tu = ggml_get_type_traits_cpu(mu.type);
    const ggml_type_traits_cpu * td = ggml_get_type_traits_cpu(md.type);
    const ggml_type_traits_cpu * th = ggml_get_type_traits_cpu(G.hq_type);
    const int     g  = S.grp[t];
    const int     ng = S.n_grp[g];
    const int     r  = S.rank[t];
    const int     k  = K.k;
    const int64_t f0 = G.f0[g];
    const int64_t nh = G.nh[g];

    // gate and up: the socket's hidden rows of every expert
    const int64_t n1 = (int64_t) k*nh;
    for (int64_t i = n1*r/ng; i < n1*(r + 1)/ng; i++) {
        const int     j  = (int) (i / nh);
        const int64_t hr = i % nh;
        const float   gv = expert_cpu_row_dot(mg, tg, K.ids[j], f0 + hr, K.xq, G.xq_type);
        const float   uv = expert_cpu_row_dot(mu, tu, K.ids[j], f0 + hr, K.xq, G.xq_type);
        K.hidden[(size_t) j*nh + hr] = expert_cpu_glu(G, gv, uv);
    }
    EXPERT_CPU_MARK(S, t, 1);
    expert_cpu_barrier_n(K.bar_cnt, K.bar_gen, ng);
    EXPERT_CPU_MARK(S, t, 2);

    // The hidden rows in the vec_dot type of down. Every thread of the socket reads all of them for every output row, and
    //   lines that several cores wrote stay slow to read for all of them: chunks of 256 by every thread made the down
    //   phase 107 us instead of 88 (1.86 experts, 14 threads). So one thread writes each expert's row when the rows are
    //   whole cache lines (Q8_0: 1024 inputs are 17), else the first thread all of them.
    const size_t hq_row  = ggml_row_size(G.hq_type, nh);
    const bool   by_row  = hq_row % 64 == 0;
    for (int j = by_row ? r : (r == 0 ? 0 : k); j < k; j += by_row ? ng : 1) {
        th->from_float(K.hidden + (size_t) j*nh, K.hq + (size_t) j*hq_row, nh);
    }
    EXPERT_CPU_MARK(S, t, 3);
    expert_cpu_barrier_n(K.bar_cnt, K.bar_gen, ng);
    EXPERT_CPU_MARK(S, t, 4);

    // down over this socket's K range, every output row
    const int64_t n_out = md.n_rows;
    const int     n_seg = G.n_seg[g];
    const expert_cpu_dseg * seg = G.seg[g];
    float * out = g == 0 ? S.mb_host->out : S.mb_host->out1;
    for (int64_t row = n_out*r/ng; row < n_out*(r + 1)/ng; row++) {
        float acc = 0.0f;
        for (int j = 0; j < k; j++) {
            const char *    x = md.base + (size_t) K.ids[j]*md.expert_bytes;
            const uint8_t * y = K.hq + (size_t) j*hq_row;
            float s = 0.0f;
            for (int q = 0; q < n_seg; q++) {
                float v = 0.0f;
                td->vec_dot(seg[q].n, &v, 0, x + seg[q].x_off + (size_t) row*seg[q].row_bytes, 0, y + seg[q].y_off, 0, 1);
                s += v;
            }
            acc += K.w[j]*s;
        }
        out[row] = acc;
    }
    EXPERT_CPU_MARK(S, t, 5);
    if (S.prof) {
        expert_cpu_thread_prof & tp = S.tp[t];
        tp.ph_gu   += (double) (int64_t) (tp.ev[1] - tp.ev[0])/S.tsc_us;
        tp.ph_down += (double) (int64_t) (tp.ev[5] - tp.ev[4])/S.tsc_us;
        tp.n++;
    }
    if (K.done_cnt.fetch_add(1, std::memory_order_acq_rel) == ng - 1) {
        K.done_cnt.store(0, std::memory_order_relaxed);
        _mm_sfence();
        if (g == 0) {
            S.mb_host->completed = K.req;
        } else {
            S.mb_host->completed1 = K.req;
        }
        expert_cpu_call_done(S, K, g);
    }
}

// nothing to do for a while: the leader sleeps until a kick or a request, its workers on wake
static void expert_cpu_sleep(expert_cpu_service * S, std::atomic<int> & sleeping, std::atomic<uint32_t> & wake, const uint32_t last) {
    sleeping.store(1, std::memory_order_release);
    while (!S->quit.load() && S->mb_host->request == last && expert_cpu_now_us() > S->active_until.load()) {
        uint32_t kk = S->kick.load();
        WaitOnAddress(&S->kick, &kk, sizeof(kk), 50);
    }
    sleeping.store(0, std::memory_order_release);
    wake.fetch_add(1);
    WakeByAddressAll(&wake);
}

static void expert_cpu_thread_init(expert_cpu_service * S, int t) {
    expert_cpu_pin(S->cpus[t]);
    static const int prio = (int) expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_PRIORITY", (float) THREAD_PRIORITY_HIGHEST);
    SetThreadPriority(GetCurrentThread(), prio);
}

static void expert_cpu_worker(expert_cpu_service * S, int t) {
    expert_cpu_thread_init(S, t);
    std::atomic<uint32_t> & gen      = S->ksplit ? S->sock[S->grp[t]]->gen      : S->gen;
    std::atomic<int>      & sleeping = S->ksplit ? S->sock[S->grp[t]]->sleeping : S->sleeping;
    std::atomic<uint32_t> & wake     = S->ksplit ? S->sock[S->grp[t]]->wake     : S->wake;
    uint32_t seen = gen.load();
    for (;;) {
        const uint32_t g = gen.load(std::memory_order_acquire);
        if (g != seen) {
            seen = g;
            if (S->ksplit) {
                expert_cpu_work_k(*S, *S->sock[S->grp[t]], t);
            } else {
                expert_cpu_work_rows(*S, t);
            }
            continue;
        }
        if (S->quit.load(std::memory_order_relaxed)) {
            return;
        }
        if (sleeping.load(std::memory_order_acquire)) {
            uint32_t e = wake.load();
            if (sleeping.load(std::memory_order_acquire)) {
                WaitOnAddress(&wake, &e, sizeof(e), 50);
            }
            continue;
        }
        _mm_pause();
    }
}

static void expert_cpu_debug_print(const ggml_cuda_expert_cpu_mbox * mb, uint32_t req, int gi, int k, const int * ids, const float * w,
        int64_t n, bool two) {
    double sx = 0, so = 0, mo = 0;
    for (int64_t i = 0; i < n; i++) {
        const double o = (double) mb->out[i] + (two ? (double) mb->out1[i] : 0.0);
        sx += (double) mb->x[i]*mb->x[i];
        so += o*o;
        mo  = std::max(mo, fabs(o));
    }
    fprintf(stderr, "expert-cpu: req %u group %d k %d ids", req, gi, k);
    for (int j = 0; j < k; j++) {
        fprintf(stderr, " %d:%.4f", ids[j], w[j]);
    }
    fprintf(stderr, " |x| %.4g |out| %.4g max %.4g\n", sqrt(sx), sqrt(so), mo);
}

// K split: the leader of a socket polls the mailbox, takes the job and quantizes x for its socket, starts its workers
static void expert_cpu_lead_k(expert_cpu_service * S, int t) {
    expert_cpu_thread_init(S, t);
    static const bool debug = getenv("GGML_CUDA_EXPERT_CPU_DEBUG") != nullptr;
    static const bool zero  = getenv("GGML_CUDA_EXPERT_CPU_ZERO") != nullptr; // debug: a zero result
    const int g = S->grp[t];
    expert_cpu_socket &         K  = *S->sock[g];
    ggml_cuda_expert_cpu_mbox * mb = S->mb_host;
    volatile uint32_t *         done = g == 0 ? &mb->completed : &mb->completed1;
    uint32_t last = mb->request;
    for (;;) {
        const uint32_t req = mb->request;
        if (req != last) {
            last = req;
            std::atomic_thread_fence(std::memory_order_acquire);
            K.t_seen = __rdtsc();
            if (S->prof) {
                expert_cpu_lat_req(*S, K, K.t_seen);
            }
            const int gi = mb->group;
            const int k  = std::min<int>(mb->k, GGML_CUDA_EXPERT_CPU_MAX_K);
            const bool ok = gi >= 0 && gi < g_n_groups.load(std::memory_order_acquire) && k > 0;
            if (ok && !zero) {
                K.G   = &g_groups[gi];
                K.k   = k;
                K.req = req;
                for (int j = 0; j < k; j++) {
                    K.ids[j] = mb->ids[j];
                    K.w[j]   = mb->w[j];
                }
                // by the leader alone: x quantized by all threads (lines written by several cores, see expert_cpu_work_k)
                //   made gate/up 310 us instead of 155, and all threads quantizing a copy each took 15 us
                expert_cpu_quantize_x(*K.G, mb->x, K.xq);
                K.t_q = __rdtsc();
                K.gen.fetch_add(1, std::memory_order_acq_rel);
                expert_cpu_work_k(*S, K, t);
                if (debug && g == 0 && S->n_calls.load() < 12) {
                    while (mb->completed1 != req) {
                        _mm_pause();
                    }
                    expert_cpu_debug_print(mb, req, gi, k, K.ids, K.w, K.G->mat[2].n_rows, true);
                }
            } else {
                if (ok) {
                    memset(g == 0 ? mb->out : mb->out1, 0, g_groups[gi].mat[2].n_rows*sizeof(float));
                }
                _mm_sfence();
                *done = req;
                S->sock_done.fetch_add(1, std::memory_order_acq_rel);
            }
            continue;
        }
        if (S->quit.load(std::memory_order_relaxed)) {
            return;
        }
        if (expert_cpu_now_us() > S->active_until.load(std::memory_order_relaxed)) {
            expert_cpu_sleep(S, K.sleeping, K.wake, last);
            continue;
        }
        _mm_pause();
    }
}

// rows split: thread 0 polls the mailbox
static void expert_cpu_serve(expert_cpu_service * S) {
    expert_cpu_thread_init(S, 0);
    static const bool debug = getenv("GGML_CUDA_EXPERT_CPU_DEBUG") != nullptr;
    static const bool zero  = getenv("GGML_CUDA_EXPERT_CPU_ZERO") != nullptr; // debug: a zero result
    ggml_cuda_expert_cpu_mbox * mb = S->mb_host;
    uint32_t last = mb->request;
    for (;;) {
        const uint32_t req = mb->request;
        if (req != last) {
            last = req;
            std::atomic_thread_fence(std::memory_order_acquire);
            S->t_seen = __rdtsc();
            if (S->prof) {
                expert_cpu_lat_req(*S, *S->sock[0], S->t_seen);
            }
            const int gi = mb->group;
            const int k  = std::min<int>(mb->k, GGML_CUDA_EXPERT_CPU_MAX_K);
            if (gi >= 0 && gi < g_n_groups.load(std::memory_order_acquire) && k > 0) {
                S->G = &g_groups[gi];
                S->k = k;
                for (int j = 0; j < k; j++) {
                    S->ids[j] = mb->ids[j];
                    S->w[j]   = mb->w[j];
                }
                if (zero) {
                    memset(mb->out, 0, S->G->mat[2].n_rows*sizeof(float));
                } else {
                    expert_cpu_quantize_x(*S->G, mb->x, S->xq.data());
                    S->sock[0]->t_q = __rdtsc();
                    S->gen.fetch_add(1, std::memory_order_acq_rel);
                    expert_cpu_work_rows(*S, 0);
                }
            }
            if (debug && S->n_calls.load() < 12 && k > 0) {
                expert_cpu_debug_print(mb, req, gi, k, S->ids, S->w, S->G->mat[2].n_rows, false);
            }
            _mm_sfence();
            std::atomic_thread_fence(std::memory_order_release);
            mb->completed = req;
            const uint64_t t1 = __rdtsc();
            expert_cpu_record(*S, (double) (t1 - S->t_seen)/S->tsc_us, k);
            if (S->prof && k > 0 && !zero) {
                for (int g = 0; g < S->n_grps; g++) {
                    expert_cpu_socket & K = *S->sock[g];
                    expert_cpu_prof_sum(*S, K, g, S->t_seen, 6);
                    K.prof_q    += (double) (S->sock[0]->t_q - S->t_seen)/S->tsc_us;
                    K.prof_done += (double) (t1 - S->t_seen)/S->tsc_us;
                }
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
            expert_cpu_sleep(S, S->sleeping, S->wake, last);
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
    for (int g = 0; g < S->n_grps; g++) {
        S->sock[g]->wake.fetch_add(1);
        WakeByAddressAll(&S->sock[g]->wake);
    }
    for (std::thread & th : S->threads) {
        th.join();
    }
    S->threads.clear();
}

// rdtsc ticks per us
static double expert_cpu_tsc_rate() {
    LARGE_INTEGER f, c0, c1;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c0);
    const uint64_t r0 = __rdtsc();
    Sleep(20);
    QueryPerformanceCounter(&c1);
    const uint64_t r1 = __rdtsc();
    return (double) (r1 - r0) / ((double) (c1.QuadPart - c0.QuadPart)*1e6/(double) f.QuadPart);
}

// the shared state of a socket and its buffers, in memory of its node
static expert_cpu_socket * expert_cpu_socket_alloc(int node) {
    const size_t head  = (sizeof(expert_cpu_socket) + 4095) & ~(size_t) 4095;
    const size_t hid   = (size_t) GGML_CUDA_EXPERT_CPU_MAX_K*EXPERT_CPU_HIDDEN*sizeof(float);
    const size_t hq    = (size_t) GGML_CUDA_EXPERT_CPU_MAX_K*EXPERT_CPU_HQ_ROW;
    const size_t total = head + EXPERT_CPU_XQ_BYTES + hid + hq;
    void * mem = VirtualAllocExNuma(GetCurrentProcess(), nullptr, total, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, (DWORD) node);
    if (mem == nullptr) {
        mem = VirtualAlloc(nullptr, total, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    }
    GGML_ASSERT(mem != nullptr);
    expert_cpu_socket * K = new (mem) expert_cpu_socket();
    K->xq     = (uint8_t *) mem + head;
    K->hidden = (float *) ((uint8_t *) mem + head + EXPERT_CPU_XQ_BYTES);
    K->hq     = (uint8_t *) mem + head + EXPERT_CPU_XQ_BYTES + hid;
    return K;
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
        (void) expert_cpu_q8_K_ok(); // its check, here rather than in the first call
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
        CUDA_CHECK(cudaMalloc(&S->state, EXPERT_CPU_STATE_U32*sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(S->state, 0, EXPERT_CPU_STATE_U32*sizeof(uint32_t)));
        CUDA_CHECK(cudaDeviceSynchronize());
        S->prof   = getenv("GGML_CUDA_EXPERT_CPU_PROF") != nullptr;
        S->tsc_us = expert_cpu_tsc_rate();
        ggml_cuda_set_device(cur);

        S->n_grps = std::min(topo.n_nodes, 2);
        S->ksplit = expert_cpu_ksplit() && S->n_grps == 2;
        for (int g = 0; g < S->n_grps; g++) {
            S->lead[g] = (int) S->cpus.size();
            for (int c : topo.cpus[g]) {
                S->cpus.push_back(c);
                S->grp.push_back(g);
                S->rank.push_back(S->n_grp[g]++);
            }
            S->sock[g] = expert_cpu_socket_alloc(topo.node[g]);
        }
        S->T = (int) S->cpus.size();
        S->tp = new expert_cpu_thread_prof[S->T]();
        if (!S->ksplit) {
            S->xq.resize(EXPERT_CPU_XQ_BYTES);
            S->hidden.resize((size_t) GGML_CUDA_EXPERT_CPU_MAX_K*EXPERT_CPU_HIDDEN);
            S->hq.resize((size_t) GGML_CUDA_EXPERT_CPU_MAX_K*EXPERT_CPU_HQ_ROW);
            S->hq_row = EXPERT_CPU_HQ_ROW;
        }
        S->active_until.store(expert_cpu_now_us() + 1000000);

        for (int t = 0; t < S->T; t++) {
            if (S->ksplit && t == S->lead[S->grp[t]]) {
                S->threads.emplace_back(expert_cpu_lead_k, S, t);
            } else if (!S->ksplit && t == 0) {
                S->threads.emplace_back(expert_cpu_serve, S);
            } else {
                S->threads.emplace_back(expert_cpu_worker, S, t);
            }
        }
        g_svc = S;
        atexit(expert_cpu_stop);

        std::string cl;
        for (int g = 0; g < S->n_grps; g++) {
            cl += (g ? ", node " : "node ") + std::to_string(topo.node[g]) + ": " + std::to_string(topo.cpus[g].front()) + "-" +
                std::to_string(topo.cpus[g].back());
        }
        GGML_LOG_INFO("%s: CPU experts on %d threads (%s), %s, results added on device %d\n", __func__, S->T, cl.c_str(),
            S->ksplit ? "down split by K between the sockets" : "rows split", ggml_cuda_expert_cpu_device());
    });
}

void ggml_cuda_expert_cpu_kick() {
    expert_cpu_service * S = g_svc;
    if (S == nullptr) {
        return;
    }
    static const int64_t spin_us = (int64_t) (expert_cpu_env_f("GGML_CUDA_EXPERT_CPU_SPIN_MS", 1000.0f)*1000.0f);
    S->active_until.store(expert_cpu_now_us() + spin_us, std::memory_order_relaxed);
    bool asleep = S->sleeping.load(std::memory_order_acquire);
    for (int g = 0; g < S->n_grps; g++) {
        asleep = asleep || S->sock[g]->sleeping.load(std::memory_order_acquire);
    }
    if (asleep) {
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

// K split: the hidden range of each socket and the parts of the down pieces in it, on block boundaries of both types
static bool expert_cpu_group_ksplit(expert_cpu_group & G, const int n_grps) {
    const ggml_cuda_expert_cpu_mat & md = G.mat[2];
    const int64_t n_ff = G.mat[0].n_rows;
    const int64_t bq   = ggml_blck_size(G.hq_type);
    const int64_t bd   = ggml_blck_size(md.type);
    for (int g = 0; g < n_grps; g++) {
        G.f0[g] = n_ff*g/n_grps;
        G.nh[g] = n_ff*(g + 1)/n_grps - G.f0[g];
        if (G.f0[g] % bq != 0 || G.nh[g] % bq != 0 || ggml_row_size(G.hq_type, G.nh[g]) > EXPERT_CPU_HQ_ROW) {
            return false;
        }
        G.n_seg[g] = 0;
        for (int p = 0; p < md.n_pieces; p++) {
            const int64_t lo = std::max(G.f0[g], md.piece_k0[p]);
            const int64_t hi = std::min(G.f0[g] + G.nh[g], md.piece_k0[p] + md.piece_k[p]);
            if (lo >= hi) {
                continue;
            }
            if ((lo - md.piece_k0[p]) % bd != 0 || (hi - lo) % bd != 0 || (hi - lo) % bq != 0) {
                return false;
            }
            expert_cpu_dseg & s = G.seg[g][G.n_seg[g]++];
            s.x_off     = md.piece_off[p] + ggml_row_size(md.type, lo - md.piece_k0[p]);
            s.row_bytes = md.piece_row_bytes[p];
            s.y_off     = ggml_row_size(G.hq_type, lo - G.f0[g]);
            s.n         = (int) (hi - lo);
        }
    }
    return true;
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
    const expert_cpu_service * S = g_svc.load();
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
             n_embd <= GGML_CUDA_EXPERT_CPU_MAX_EMBD && n_embd % 4 == 0 && G.mat[0].n_rows <= EXPERT_CPU_HIDDEN &&
             ggml_row_size(G.hq_type, G.mat[0].n_rows) <= EXPERT_CPU_HQ_ROW &&
             ggml_row_size(G.xq_type, n_embd) <= EXPERT_CPU_XQ_BYTES && g_n_groups.load() < EXPERT_CPU_MAX_GROUPS;
        if (ok && S != nullptr && S->ksplit && !expert_cpu_group_ksplit(G, S->n_grps)) {
            ok = false;
            GGML_LOG_WARN("%s: the down experts of %s do not split by K between the sockets: they stay on the device "
                "(GGML_CUDA_EXPERT_CPU_V1=1 splits by rows)\n", __func__, down_pool->name);
        }
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
            for (int g = 0; S != nullptr && S->ksplit && g < S->n_grps; g++) {
                fprintf(stderr, "expert-cpu: group %d socket %d hidden [%lld, %lld) down segments:", idx, g, (long long) G.f0[g],
                    (long long) (G.f0[g] + G.nh[g]));
                for (int q = 0; q < G.n_seg[g]; q++) {
                    fprintf(stderr, " [x_off %zu rb %zu y_off %zu n %d]", G.seg[g][q].x_off, G.seg[g][q].row_bytes, G.seg[g][q].y_off,
                        G.seg[g][q].n);
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
    if (S == nullptr) {
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
    a.prof      = S->prof ? 1 : 0;
}

void ggml_cuda_expert_cpu_merge(cudaStream_t stream, float * dst, int n_embd) {
    const expert_cpu_service * S = g_svc.load();
    GGML_ASSERT(S != nullptr && n_embd % 4 == 0 && (uintptr_t) dst % 16 == 0);
    const int nthreads = 128;
    const int nblocks  = std::max(1, std::min(8, (n_embd/4 + nthreads - 1)/nthreads));
    expert_cpu_wait_add<<<nblocks, nthreads, 0, stream>>>(S->mb_dev, S->state, dst, n_embd, S->ksplit ? 2 : 1, S->prof ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_expert_cpu_stats_print() {
    expert_cpu_service * S = g_svc;
    if (S == nullptr || S->n_calls.load() == 0) {
        return;
    }
    unsigned long long st[12] = {};
    int cur = 0;
    if (cudaGetDevice(&cur) == cudaSuccess) {
        ggml_cuda_set_device(ggml_cuda_expert_cpu_device());
        if (cudaMemcpy(st, S->state + 4, sizeof(st), cudaMemcpyDeviceToHost) == cudaSuccess && st[3] > 0) {
            fprintf(stderr, "expert-cpu: %llu waits, GPU work between request and wait %.1f us, wait %.1f us (means); %.1f%% of "
                "them under 4 us (the GPU slower), GPU work there %.1f us, else %.1f us\n", st[3], st[1]/1e3/st[3], st[2]/1e3/st[3],
                100.0*st[4]/st[3], st[4] ? st[5]/1e3/st[4] : 0.0, st[3] > st[4] ? (st[1] - st[5])/1e3/(st[3] - st[4]) : 0.0);
        }
        unsigned long long bins[3*EXPERT_CPU_BIN_K*EXPERT_CPU_BIN_C] = {};
        if (S->prof && st[3] > 0 && cudaMemcpy(bins, S->state + 32, sizeof(bins), cudaMemcpyDeviceToHost) == cudaSuccess) {
            fprintf(stderr, "expert-cpu: waits < 4 / 20 / 50 / 100 / 200 / more us: %.1f / %.1f / %.1f / %.1f / %.1f / %.1f %%\n",
                100.0*st[6]/st[3], 100.0*st[7]/st[3], 100.0*st[8]/st[3], 100.0*st[9]/st[3], 100.0*st[10]/st[3], 100.0*st[11]/st[3]);
            for (int c = 0; c < EXPERT_CPU_BIN_C; c++) {
                std::string line;
                char buf[96];
                for (int k = 1; k <= EXPERT_CPU_BIN_K; k++) {
                    const unsigned long long * b = bins + 3*((k - 1) + EXPERT_CPU_BIN_K*c);
                    if (b[0] > 0) {
                        snprintf(buf, sizeof(buf), "  k%d%s %.1f%% %.0f+%.0f", k, k == EXPERT_CPU_BIN_K ? "+" : "",
                            100.0*b[0]/st[3], b[1]/1e3/b[0], b[2]/1e3/b[0]);
                        line += buf;
                    }
                }
                fprintf(stderr, "expert-cpu: %d%s copies: share, GPU work + wait us by CPU experts:%s\n", c,
                    c == EXPERT_CPU_BIN_C - 1 ? "+" : "", line.c_str());
            }
        }
        (void) cudaGetLastError();
        ggml_cuda_set_device(cur);
    }
    fprintf(stderr, "expert-cpu: %llu calls, %llu experts (%.3f/call), %.1f us/call, %llu calls over 2 ms\n",
        (unsigned long long) S->n_calls.load(), (unsigned long long) S->n_experts.load(),
        (double) S->n_experts.load()/S->n_calls.load(), (double) S->us_total.load()/S->n_calls.load(),
        (unsigned long long) S->n_slow.load());
    {
        std::string line;
        char buf[64];
        for (int k = 1; k <= EXPERT_CPU_BIN_K; k++) {
            if (S->k_n[k].load() > 0) {
                snprintf(buf, sizeof(buf), "  k%d%s %.1f%% %.0f", k, k == EXPERT_CPU_BIN_K ? "+" : "",
                    100.0*S->k_n[k].load()/S->n_calls.load(), (double) S->k_us[k].load()/S->k_n[k].load());
                line += buf;
            }
        }
        fprintf(stderr, "expert-cpu: calls and CPU us/call by CPU experts:%s\n", line.c_str());
    }
    for (int g = 0; S->ksplit && g < S->n_grps; g++) {
        const expert_cpu_socket & K = *S->sock[g];
        if (K.n_calls > 0) {
            fprintf(stderr, "expert-cpu: socket %d: %.1f us/call from its leader's request to its last row\n", g,
                (double) K.tsc_total/S->tsc_us/K.n_calls);
        }
    }
    for (int g = 0; S->prof && g < S->n_grps; g++) {
        const expert_cpu_socket & K = *S->sock[g];
        if (K.lat_n > 0) {
            fprintf(stderr, "expert-cpu: socket %d: the GPU's request seen %.2f us later than the quickest of its window of 64 "
                "(%llu calls)\n", g, K.lat_excess/1e3/K.lat_n, (unsigned long long) K.lat_n);
        }
    }
    static const char * ev_k[EXPERT_CPU_PROF_EV]    = { "start", "gate/up", "B1 out", "hidden q", "B2 out", "down", "" };
    static const char * ev_rows[EXPERT_CPU_PROF_EV] = { "start", "gate/up", "B1 out", "hidden q", "B2 out", "down", "B3 out" };
    for (int g = 0; S->prof && g < S->n_grps; g++) {
        const expert_cpu_socket & K = *S->sock[g];
        if (K.prof_n == 0) {
            continue;
        }
        const double n = (double) K.prof_n;
        std::string line;
        char buf[96];
        const int n_ev = 6;
        for (int e = 0; e < n_ev; e++) {
            snprintf(buf, sizeof(buf), ", %s %.1f/%.1f", (S->ksplit ? ev_k : ev_rows)[e], K.prof_mean[e]/n, K.prof_max[e]/n);
            line += buf;
        }
        fprintf(stderr, "expert-cpu: prof socket %d (%llu calls, us after the request, mean/max over threads): x quantized %.1f%s, done %.1f\n",
            g, (unsigned long long) K.prof_n, K.prof_q/n, line.c_str(), K.prof_done/n);
        std::string gu, dn;
        for (int t = 0; S->ksplit && t < S->T; t++) {
            if (S->grp[t] == g && S->tp[t].n > 0) {
                snprintf(buf, sizeof(buf), " %.1f", S->tp[t].ph_gu/S->tp[t].n);
                gu += buf;
                snprintf(buf, sizeof(buf), " %.1f", S->tp[t].ph_down/S->tp[t].n);
                dn += buf;
            }
        }
        if (!gu.empty()) {
            fprintf(stderr, "expert-cpu: prof socket %d per thread (CPU %d first), us in gate/up:%s; in down:%s\n", g,
                S->cpus[S->lead[g]], gu.c_str(), dn.c_str());
        }
    }
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
