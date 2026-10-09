#include "expert-cache.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <vector>

#define GGML_CUDA_EXPERT_CACHE_MAGIC 0x6578706572746373ull

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

    int32_t  * slot_of;     // [n_expert] pool slot of an expert, -1 when absent
    int32_t  * expert_of;   // [cap]
    uint32_t * last_used;   // [cap]
    uint32_t * clock;       // [1]
    int32_t  * remap;       // [max_ids]
    int32_t  * miss_expert; // [max_ids]
    int32_t  * miss_slot;   // [max_ids]
    int32_t  * n_miss;      // [1]
    uint32_t * stats;       // [2] calls, misses - GGML_META_EXPERT_CACHE_STATS
};

struct ggml_cuda_expert_cache {
    uint64_t                   magic = GGML_CUDA_EXPERT_CACHE_MAGIC;
    int                        device;
    int                        align;  // largest of 16/8/4/2/1 that divides every address and stride the copy uses
    bool                       covering; // align < 4, copied in the 8-byte words that cover each chunk
    ggml_cuda_expert_cache_dev d;
    void *                     state = nullptr;
};

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

// The scheduler and the LRU run in one block. Hits are marked in parallel, misses are then taken one by one
// so the victim of one is never the slot another still needs: the argmin over last_used cannot return a
// slot touched by this batch, its stamp is the newest. Same ids on every device give the same slots.
// fast (a few ids, decode): the same loop by warp 0 alone on a shared copy of last_used and expert_of, with the misses
//   so far in shared memory instead of slot_of: no block barrier and no global round trip per miss, same victims.
static __global__ void expert_cache_plan(const ggml_cuda_expert_cache_group g, const int32_t * ids,
        const int n_rows, const int n_per_row, const int row_stride, const bool fast) {
    const ggml_cuda_expert_cache_dev & d = g.d[blockIdx.x];
    __shared__ int      s_n_miss;
    __shared__ uint32_t s_min_val[32];
    __shared__ int      s_min_idx[32];
    __shared__ int      s_remap[GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS];
    __shared__ int      s_id[GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS];
    __shared__ int      s_miss_expert[GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS];
    __shared__ int      s_miss_slot[GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS];
    extern __shared__ uint32_t s_lru[]; // fast: last_used [cap], then expert_of [cap]

    const int      n_ids = n_rows*n_per_row;
    const uint32_t clock = *d.clock;

    uint32_t * s_last = s_lru;
    int32_t  * s_exp  = (int32_t *) (s_lru + d.cap);
    if (fast) {
        // it does not depend on the ids, so its round trip overlaps theirs and that of slot_of
        for (int s = threadIdx.x; s < d.cap; s += blockDim.x) {
            s_last[s] = d.last_used[s];
            s_exp[s]  = d.expert_of[s];
        }
    }

    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int id = ids[(i / n_per_row)*row_stride + (i % n_per_row)];
        const int sl = d.slot_of[id];
        d.remap[i] = sl;
        if (fast) {
            s_remap[i] = sl;
            s_id[i]    = id;
        }
        if (sl >= 0) {
            // several tokens of one ubatch may route to the same expert; max keeps the stamp deterministic
            atomicMax(&d.last_used[sl], clock + 1 + i);
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
            }
        }
        __syncthreads();
        if (threadIdx.x >= WARP_SIZE) {
            return;
        }
        const int lane   = threadIdx.x;
        int       n_miss = 0;
        for (int i = 0; i < n_ids; i++) {
            if (s_remap[i] >= 0) {
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
            if (lane == 0) {
                const int victim = best_i;
                const int old    = s_exp[victim];
                if (old >= 0) {
                    d.slot_of[old] = -1;
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
        }
        return;
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
        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x % 32;
        if (lane == 0) {
            s_min_val[warp] = best;
            s_min_idx[warp] = best_i;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int nw = blockDim.x / 32;
            for (int w = 1; w < nw; w++) {
                if (s_min_val[w] < best || (s_min_val[w] == best && s_min_idx[w] < best_i)) {
                    best   = s_min_val[w];
                    best_i = s_min_idx[w];
                }
            }
            const int victim = best_i;
            const int old    = d.expert_of[victim];
            if (old >= 0) {
                d.slot_of[old] = -1;
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

// blockIdx.z is the cache of the group, each copied with the widest unit its addresses allow (negative: covering)
static __global__ void expert_cache_copy(const ggml_cuda_expert_cache_group g) {
    const int z = blockIdx.z;
    const ggml_cuda_expert_cache_dev & d = g.d[z];
    const int m = blockIdx.y;
    if (m >= *d.n_miss) {
        return;
    }
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

// Registered host ranges, so several weights in one buffer register it once.
struct expert_cache_host_reg {
    const char * base;
    size_t       size;
    const char * dev_base;
};
static std::vector<expert_cache_host_reg> g_host_regs;
static std::mutex                         g_host_regs_mutex;

static const char * expert_cache_register_host(const void * buf, size_t size) {
    static const size_t page = 4096;
    const char * base = (const char *) ((uintptr_t) buf & ~(page - 1));
    size_t       len  = (((const char *) buf + size) - base + page - 1) & ~(page - 1);

    std::lock_guard<std::mutex> lock(g_host_regs_mutex);
    for (const expert_cache_host_reg & r : g_host_regs) {
        if (r.base == base && r.size == len) {
            return r.dev_base;
        }
    }

    // memory that is already page-locked (a cudaMallocHost buffer, or registered by someone else) cannot be
    // registered again, but already has a device address
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, buf) == cudaSuccess && attr.type == cudaMemoryTypeHost && attr.devicePointer != nullptr) {
        const char * dev_base = (const char *) attr.devicePointer - ((const char *) buf - base);
        g_host_regs.push_back({base, len, dev_base});
        return dev_base;
    }
    (void) cudaGetLastError();

    cudaError_t err = cudaHostRegister((void *) base, len, cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: could not register %.0f MiB of host memory at %p (buffer %p + %zu) for device access: %s\n",
            __func__, len/1024.0/1024.0, (const void *) base, buf, size, cudaGetErrorString(err));
        return nullptr;
    }
    void * dev_base = nullptr;
    err = cudaHostGetDevicePointer(&dev_base, (void *) base, 0);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        cudaHostUnregister((void *) base);
        return nullptr;
    }
    g_host_regs.push_back({base, len, (const char *) dev_base});
    return (const char *) dev_base;
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

    const char * dev_buf = expert_cache_register_host(host_buf, host_buf_size);
    if (dev_buf == nullptr) {
        return nullptr;
    }
    const char * dev_ptr = dev_buf + ((const char *) host_ptr - (const char *) ((uintptr_t) host_buf & ~(size_t) 4095));

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
    c->align = expert_cache_align({(size_t) (uintptr_t) dev_ptr, expert_bytes, chunk_full, chunk_dev, offset_dev, c->d.slot_bytes});
    c->covering = c->align < 4 && expert_cache_align({(size_t) (uintptr_t) dev_ptr, expert_bytes, chunk_full}) >= 8;

    const size_t n_i32 = (size_t) n_expert + 2*cap + 1 + 3*max_ids + 1 + 2;
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
    c->d.stats       = (uint32_t *) p;
    CUDA_CHECK(cudaMemset(c->d.stats,     0,    2*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(c->d.slot_of,   0xff, (size_t) n_expert*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(c->d.expert_of, 0xff, (size_t) cap*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(c->d.last_used, 0,    (size_t) cap*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(c->d.clock,     0,    sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(c->d.n_miss,    0,    sizeof(int32_t)));
    return c;
}

void ggml_cuda_expert_cache_free(void * cache) {
    ggml_cuda_expert_cache * c = (ggml_cuda_expert_cache *) cache;
    if (c == nullptr) {
        return;
    }
    ggml_cuda_set_device(c->device);
    static const bool stats = getenv("GGML_META_EXPERT_CACHE_STATS") != nullptr;
    if (stats) {
        uint32_t st[2] = {0, 0};
        cudaDeviceSynchronize();
        if (cudaMemcpy(st, c->d.stats, sizeof(st), cudaMemcpyDeviceToHost) == cudaSuccess && st[0] > 0) {
            fprintf(stderr, "expert-cache-dev: dev %d cap %d n_expert %d slot_bytes %zu calls %u misses %u miss/call %.4f MiB/call %.3f\n",
                c->device, c->d.cap, c->d.n_expert, c->d.slot_bytes, st[0], st[1], (double) st[1]/st[0],
                (double) st[1]/st[0]*c->d.slot_bytes/1048576.0);
        }
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
        const ggml_tensor * ids, ggml_tensor * ids_remap) {
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
    const int    n_ids = n_rows*n_per_row;
    const bool   fast  = n_ids <= GGML_CUDA_EXPERT_CACHE_PLAN_FAST_IDS && n_ids <= cap_min &&
                         cap_max <= GGML_CUDA_EXPERT_CACHE_PLAN_FAST_CAP;
    const size_t smem  = fast ? 2*(size_t) cap_max*sizeof(uint32_t) : 0;
    expert_cache_plan<<<n, 512, smem, stream>>>(g, (const int32_t *) ids->data, n_rows, n_per_row, row_stride, fast);
    CUDA_CHECK(cudaGetLastError());

    expert_cache_copy<<<dim3(nbx, n_rows*n_per_row, n), nthreads, 0, stream>>>(g);
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
