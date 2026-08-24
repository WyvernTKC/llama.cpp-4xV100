#include "allreduce-p2p.cuh"

#include <cstdlib>

// One block index is the unit of cross-device synchronisation: block b on every rank exchanges
// the same slice, so a per-block-index barrier is enough and no grid-wide sync is needed.
#define GGML_CUDA_P2P_AR_NBLOCKS  16
#define GGML_CUDA_P2P_AR_NTHREADS 256

// Two staging slots, alternating per call. A rank can run at most one call ahead of its peers
// (it cannot finish call k until every peer published its flag for k, and a peer only publishes
// that flag after it finished reading call k-1), so two slots are enough to keep a fast rank
// from overwriting payload a slow rank has not consumed yet.
#define GGML_CUDA_P2P_AR_NSLOTS 2

struct ggml_cuda_p2p_ar {
    int    devices[GGML_CUDA_MAX_DEVICES] = {};
    size_t n_devices                      = 0;

    // Peer-mapped. stage[j] is device j's staging area, laid out [slot][rank][stage_stride].
    // flag[j] is device j's flag array, laid out [slot][rank][nblocks].
    float4 *             stage[GGML_CUDA_MAX_DEVICES] = {};
    unsigned long long * flag [GGML_CUDA_MAX_DEVICES] = {};

    size_t             max_bytes    = 0; // per-rank payload cap, above which the caller uses NCCL
    int64_t            stage_stride = 0; // float4 elements per (slot, rank) region
    unsigned long long seq          = 0; // monotonic, identical across ranks for a given call
};

struct ggml_cuda_p2p_ar_args {
    float4 *             stage[GGML_CUDA_MAX_DEVICES];
    unsigned long long * flag [GGML_CUDA_MAX_DEVICES];

    const float4 *     src;
    float4 *           dst;
    int64_t            nvec;
    int64_t            stage_stride;
    int                n_devs;
    int                rank;
    int                slot;
    unsigned int       contrib_mask;
    unsigned long long seq;
};

static __global__ void ggml_cuda_p2p_ar_kernel(const ggml_cuda_p2p_ar_args a) {
    const int64_t nblocks = gridDim.x;
    const int64_t bid     = blockIdx.x;

    const int64_t chunk = (a.nvec + nblocks - 1) / nblocks;
    const int64_t begin = bid * chunk;
    const int64_t end   = min(begin + chunk, a.nvec);

    const int64_t slot_base = (int64_t) a.slot * a.n_devs * a.stage_stride;

    // 1. push this block's slice into every rank's staging area, our own included, so that the
    //    reduce below reads one uniform layout.
    if ((a.contrib_mask >> a.rank) & 1u) {
        for (int j = 0; j < a.n_devs; ++j) {
            float4 * dst = a.stage[j] + slot_base + (int64_t) a.rank * a.stage_stride;
            for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
                dst[i] = a.src[i];
            }
        }
    }

    // 2. publish. __syncthreads first, so the fence covers every thread's stores and not just
    //    those of thread 0.
    __syncthreads();
    __threadfence_system();

    if (threadIdx.x == 0) {
        for (int j = 0; j < a.n_devs; ++j) {
            volatile unsigned long long * f =
                a.flag[j] + (((int64_t) a.slot * a.n_devs + a.rank) * nblocks + bid);
            *f = a.seq; // naturally aligned 8 B volatile store, emitted as a single st.volatile
        }

        // 3. wait for every rank's slice to land on our own device. seq is monotonic, so a peer
        //    that already raced ahead to seq+1 still satisfies this.
        for (int j = 0; j < a.n_devs; ++j) {
            volatile const unsigned long long * f =
                a.flag[a.rank] + (((int64_t) a.slot * a.n_devs + j) * nblocks + bid);
            while (*f < a.seq) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(32);
#endif // __CUDA_ARCH__ >= 700
            }
        }
        __threadfence_system();
    }
    __syncthreads();

    // 4. reduce. __ldcg bypasses L1, so a peer's write cannot be shadowed by a stale line, while
    //    still loading 16 B at a time (plain volatile would split this into four 4 B loads).
    const float4 * stage = a.stage[a.rank] + slot_base;
    for (int64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int j = 0; j < a.n_devs; ++j) {
            if (((a.contrib_mask >> j) & 1u) == 0) {
                continue;
            }
            const float4 v = __ldcg(stage + (int64_t) j * a.stage_stride + i);
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        a.dst[i] = sum;
    }
}

void ggml_cuda_p2p_ar_free(ggml_cuda_p2p_ar * ar) {
    if (ar == nullptr) {
        return;
    }
    for (size_t i = 0; i < ar->n_devices; ++i) {
        if (cudaSetDevice(ar->devices[i]) != cudaSuccess) {
            continue;
        }
        cudaFree(ar->stage[i]);
        cudaFree(ar->flag [i]);
    }
    (void) cudaGetLastError();
    delete ar;
}

ggml_cuda_p2p_ar * ggml_cuda_p2p_ar_init(const int * devices, size_t n_devices) {
    if (n_devices < 2 || n_devices > GGML_CUDA_MAX_DEVICES) {
        return nullptr;
    }

    size_t max_bytes = 1024*1024;
    if (const char * env = getenv("GGML_CUDA_P2P_AR_MAX_BYTES")) {
        max_bytes = (size_t) atoll(env);
    }
    if (max_bytes == 0) {
        GGML_LOG_INFO("%s: one-shot P2P AllReduce disabled by GGML_CUDA_P2P_AR_MAX_BYTES=0\n", __func__);
        return nullptr;
    }
    max_bytes = GGML_PAD(max_bytes, sizeof(float4));

    // Every pair has to be reachable, a partial mesh would leave some ranks unable to push.
    for (size_t i = 0; i < n_devices; ++i) {
        for (size_t j = 0; j < n_devices; ++j) {
            if (i == j) {
                continue;
            }
            int can_access = 0;
            if (cudaDeviceCanAccessPeer(&can_access, devices[i], devices[j]) != cudaSuccess || !can_access) {
                (void) cudaGetLastError();
                GGML_LOG_INFO("%s: no peer access %d -> %d, keeping NCCL for AllReduce\n",
                    __func__, devices[i], devices[j]);
                return nullptr;
            }
        }
    }

    ggml_cuda_p2p_ar * ar = new ggml_cuda_p2p_ar;
    ar->n_devices    = n_devices;
    ar->max_bytes    = max_bytes;
    ar->stage_stride = (int64_t) (max_bytes / sizeof(float4));

    const size_t stage_bytes = (size_t) GGML_CUDA_P2P_AR_NSLOTS * n_devices * max_bytes;
    const size_t flag_bytes  = (size_t) GGML_CUDA_P2P_AR_NSLOTS * n_devices *
                               GGML_CUDA_P2P_AR_NBLOCKS * sizeof(unsigned long long);

    for (size_t i = 0; i < n_devices; ++i) {
        ar->devices[i] = devices[i];

        if (cudaSetDevice(devices[i]) != cudaSuccess) {
            (void) cudaGetLastError();
            ggml_cuda_p2p_ar_free(ar);
            return nullptr;
        }

        for (size_t j = 0; j < n_devices; ++j) {
            if (i == j) {
                continue;
            }
            const cudaError_t err = cudaDeviceEnablePeerAccess(devices[j], 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                (void) cudaGetLastError();
                GGML_LOG_INFO("%s: could not enable peer access %d -> %d, keeping NCCL for AllReduce\n",
                    __func__, devices[i], devices[j]);
                ggml_cuda_p2p_ar_free(ar);
                return nullptr;
            }
            (void) cudaGetLastError(); // clear a sticky AlreadyEnabled
        }

        if (cudaMalloc((void **) &ar->stage[i], stage_bytes) != cudaSuccess ||
            cudaMalloc((void **) &ar->flag [i], flag_bytes)  != cudaSuccess ||
            cudaMemset(ar->flag[i], 0, flag_bytes)           != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_INFO("%s: staging allocation failed, keeping NCCL for AllReduce\n", __func__);
            ggml_cuda_p2p_ar_free(ar);
            return nullptr;
        }
    }

    GGML_LOG_INFO("%s: one-shot P2P AllReduce enabled for %zu devices, up to %zu KiB per rank "
                  "(%zu MiB staging per device)\n",
        __func__, n_devices, max_bytes >> 10, (stage_bytes + flag_bytes) >> 20);

    return ar;
}

bool ggml_cuda_p2p_ar_allreduce(ggml_cuda_p2p_ar * ar, ggml_backend_t * backends, ggml_tensor ** tensors) {
    if (ar == nullptr) {
        return false;
    }

    const size_t  n_devices = ar->n_devices;
    const int64_t ne        = ggml_nelements(tensors[0]);

    // Empty reductions and the large, bandwidth-bound ones stay on the NCCL path.
    if (ne == 0 || tensors[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if ((size_t) ne * sizeof(float) > ar->max_bytes) {
        return false;
    }
    if (ne % 4 != 0) { // one float4 per lane, there is no scalar tail path
        return false;
    }

    unsigned int contrib_mask = 0;
    for (size_t i = 0; i < n_devices; ++i) {
        const ggml_tensor * t = tensors[i];
        if (t == nullptr || t->type != GGML_TYPE_F32 || ggml_nelements(t) != ne) {
            return false;
        }
        if (!ggml_is_contiguously_allocated(t) || ((uintptr_t) t->data % sizeof(float4)) != 0) {
            return false;
        }
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[i]->context;
        if (cuda_ctx->device != ar->devices[i]) {
            return false;
        }
        if (t->flags & GGML_TENSOR_FLAG_COMPUTE) {
            contrib_mask |= 1u << i;
        }
    }

    ar->seq++;

    ggml_cuda_p2p_ar_args args = {};
    for (size_t j = 0; j < n_devices; ++j) {
        args.stage[j] = ar->stage[j];
        args.flag [j] = ar->flag [j];
    }
    args.nvec         = ne / 4;
    args.stage_stride = ar->stage_stride;
    args.n_devs       = (int) n_devices;
    args.slot         = (int) (ar->seq % GGML_CUDA_P2P_AR_NSLOTS);
    args.contrib_mask = contrib_mask;
    args.seq          = ar->seq;

    for (size_t i = 0; i < n_devices; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backends[i]->context;

        args.rank = (int) i;
        args.src  = (const float4 *) tensors[i]->data;
        args.dst  = (float4 *)       tensors[i]->data;

        ggml_cuda_set_device(cuda_ctx->device);

        ggml_cuda_p2p_ar_kernel<<<GGML_CUDA_P2P_AR_NBLOCKS, GGML_CUDA_P2P_AR_NTHREADS, 0, cuda_ctx->stream()>>>(args);
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
