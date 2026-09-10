#include "expert-cache.cuh"

#include <algorithm>
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
};

struct ggml_cuda_expert_cache {
    uint64_t                   magic = GGML_CUDA_EXPERT_CACHE_MAGIC;
    int                        device;
    int                        align;  // largest of 16/8/4/2/1 that divides every address and stride the copy uses
    ggml_cuda_expert_cache_dev d;
    void *                     state = nullptr;
};

// The scheduler and the LRU run in one block. Hits are marked in parallel, misses are then taken one by one
// so the victim of one is never the slot another still needs: the argmin over last_used cannot return a
// slot touched by this batch, its stamp is the newest. Same ids on every device give the same slots.
static __global__ void expert_cache_plan(const ggml_cuda_expert_cache_dev d, const int32_t * ids,
        const int n_rows, const int n_per_row, const int row_stride) {
    __shared__ int      s_n_miss;
    __shared__ uint32_t s_min_val[32];
    __shared__ int      s_min_idx[32];

    const int      n_ids = n_rows*n_per_row;
    const uint32_t clock = *d.clock;

    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int id = ids[(i / n_per_row)*row_stride + (i % n_per_row)];
        const int sl = d.slot_of[id];
        d.remap[i] = sl;
        if (sl >= 0) {
            // several tokens of one ubatch may route to the same expert; max keeps the stamp deterministic
            atomicMax(&d.last_used[sl], clock + 1 + i);
        }
    }
    if (threadIdx.x == 0) {
        s_n_miss = 0;
    }
    __syncthreads();

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
    }
}

// One row of blocks per possible miss; rows past n_miss leave at once. Threads copy consecutive units of T,
// so a warp reads a contiguous span of host memory per instruction.
template <typename T>
static __global__ void expert_cache_copy(const ggml_cuda_expert_cache_dev d, char * pool) {
    const int m = blockIdx.y;
    if (m >= *d.n_miss) {
        return;
    }
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

    const size_t n_i32 = (size_t) n_expert + 2*cap + 1 + 3*max_ids + 1;
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
    c->d.n_miss      = p;
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
    cudaFree(c->state);
    delete c;
}

bool ggml_cuda_expert_cache_is(const ggml_tensor * pool) {
    return pool != nullptr && pool->extra != nullptr &&
        ((const ggml_cuda_expert_cache *) pool->extra)->magic == GGML_CUDA_EXPERT_CACHE_MAGIC;
}

void ggml_cuda_expert_cache_gather(ggml_backend_cuda_context & ctx, const ggml_tensor * pool, const ggml_tensor * ids, ggml_tensor * ids_remap) {
    const ggml_cuda_expert_cache * c = (const ggml_cuda_expert_cache *) pool->extra;
    const ggml_cuda_expert_cache_dev & d = c->d;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    const int n_per_row  = (int) ids->ne[0];
    const int n_rows     = (int) (ids->ne[1]*ids->ne[2]*ids->ne[3]);
    const int row_stride = (int) (ids->nb[1]/sizeof(int32_t));
    GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(n_rows*n_per_row <= d.max_ids);
    GGML_ASSERT(pool->nb[2] == d.slot_bytes);

    expert_cache_plan<<<1, 512, 0, stream>>>(d, (const int32_t *) ids->data, n_rows, n_per_row, row_stride);
    CUDA_CHECK(cudaGetLastError());

    // enough threads in flight to keep the link busy, bounded so short slices do not launch empty blocks
    const int  nthreads = 256;
    const int  nbx      = (int) std::min<size_t>(64, (d.slot_bytes/c->align + nthreads - 1)/nthreads);
    const dim3 grid(nbx, n_rows*n_per_row);
    char * pool_data = (char *) pool->data;
    switch (c->align) {
        case 16: expert_cache_copy<int4>   <<<grid, nthreads, 0, stream>>>(d, pool_data); break;
        case  8: expert_cache_copy<int2>   <<<grid, nthreads, 0, stream>>>(d, pool_data); break;
        case  4: expert_cache_copy<int32_t><<<grid, nthreads, 0, stream>>>(d, pool_data); break;
        case  2: expert_cache_copy<int16_t><<<grid, nthreads, 0, stream>>>(d, pool_data); break;
        default: expert_cache_copy<int8_t> <<<grid, nthreads, 0, stream>>>(d, pool_data); break;
    }
    CUDA_CHECK(cudaGetLastError());

    *ids_remap = *ids;
    ids_remap->data  = d.remap;
    ids_remap->nb[1] = (size_t) n_per_row*sizeof(int32_t);
    ids_remap->nb[2] = ids_remap->nb[1]*ids->ne[1];
    ids_remap->nb[3] = ids_remap->nb[2]*ids->ne[2];
    ids_remap->view_src = nullptr;
    ids_remap->extra    = nullptr;
}
