#pragma once

#include "common.cuh"

// Bounded spin waits. A kernel that polls a flag another device or another block of its launch sets (the P2P
//   all-reduce, the folds that poll its lines, the router waiter, the grid barriers of norm.cu) waits forever if that
//   writer never comes, and the host then hangs in its next synchronize: a peer that stopped or skipped a launch (a
//   rank that left out one LL kernel hung Ornith until killed), or one that faulted (2026-09 an illegal access left the
//   other three in ggml_cuda_p2p_ar_kernel). Each wait runs its polls through ggml_cuda_spin_until, which reads
//   %globaltimer once every GGML_CUDA_SPIN_POLLS polls; past the limit (GGML_CUDA_SPIN_TIMEOUT_S, 30 s by default, 0 =
//   no limit) the kernel leaves a record of the wait in pinned host memory and traps, so the host gets an error instead
//   of a hang, and ggml_cuda_error prints the records (ggml_cuda_spin_print): which wait, on which device, for which
//   rank.
// The kernel does not print: the call of a printf costs the kernels that hold it registers (rms_norm_pre_add_f32<1024,
//   5> 32 -> 40, half its resident blocks); with it and the timer checked inside the poll loop Ornith decoded 0.4-0.6%
//   slower, this form +0.4% (Q27 +0.2%), the outputs bitwise the same.
// Under a debugger or compute-sanitizer, where a peer can legitimately take longer, GGML_CUDA_SPIN_TIMEOUT_S=0.
// A file with a spin wait includes this header and defines its setter with GGML_CUDA_SPIN_SETTER, which
//   ggml_cuda_spin_init calls; without it the file's waits keep the default limit and leave no record.

#define GGML_CUDA_SPIN_POLLS             4096 // polls between two reads of the timer
#define GGML_CUDA_SPIN_TIMEOUT_S_DEFAULT 30
#define GGML_CUDA_SPIN_SHIFT             24   // the time unit is 2^24 ns (16.8 ms)
#define GGML_CUDA_SPIN_TIME_MASK         0xFFFFFu
#define GGML_CUDA_SPIN_LIMIT_MAX         0x7FFFFu // units, 2.4 h: the start is kept in 20 bits, which wrap after 4.9 h
#define GGML_CUDA_SPIN_LIMIT_DEFAULT     ((unsigned int) ((GGML_CUDA_SPIN_TIMEOUT_S_DEFAULT*1000000000ull) >> GGML_CUDA_SPIN_SHIFT))

#define GGML_CUDA_SPIN_PEER_BLOCKS -1 // the wait was for the other blocks of the same launch
#define GGML_CUDA_SPIN_PEER_DEVICE -2 // the wait was for another device, which one is not known
#define GGML_CUDA_SPIN_PEER_HOST   -3 // the wait was for the host (CPU threads)

// the waits, for the record; the names are in ggml_cuda_spin_print
enum ggml_cuda_spin_site : unsigned int {
    GGML_CUDA_SPIN_SITE_P2P_AR,         // ggml_cuda_p2p_ar_kernel
    GGML_CUDA_SPIN_SITE_P2P_AG,         // ggml_cuda_p2p_ag_kernel
    GGML_CUDA_SPIN_SITE_P2P_LL,         // ggml_cuda_p2p_ar_ll_kernel
    GGML_CUDA_SPIN_SITE_P2P_LL_N,       // ggml_cuda_p2p_ar_ll_kernel_n
    GGML_CUDA_SPIN_SITE_P2P_LL_RECV,    // ggml_cuda_p2p_ar_ll_kernel_n, receive half
    GGML_CUDA_SPIN_SITE_P2P_LL_MOE,     // ggml_cuda_p2p_ar_ll_moe_reduce_kernel
    GGML_CUDA_SPIN_SITE_FOLD_NORM,      // rms_norm_pre_add_ar_f32, the lines
    GGML_CUDA_SPIN_SITE_FOLD_NORM_Q8,   // rms_norm_pre_add_ar_f32, its q8_1 blocks
    GGML_CUDA_SPIN_SITE_FOLD_NORM_SPLIT,// rms_norm_pre_add_ar_split_f32, the lines
    GGML_CUDA_SPIN_SITE_AR_2,           // ggml_cuda_ar_kernel (allreduce.cu)
    GGML_CUDA_SPIN_SITE_ROUTER,         // mul_mat_vec_f_router, the waiting block
    GGML_CUDA_SPIN_SITE_HC_POST,        // hc_post_blocks_sync
    GGML_CUDA_SPIN_SITE_ROW_SPLIT,      // row_split_reduce
    GGML_CUDA_SPIN_SITE_NORM_Q8,        // rms_norm_pre_add_f32, its q8_1 blocks
    GGML_CUDA_SPIN_SITE_NORM_PAIR_Q8,   // rms_norm_add_rms_norm_f32, its q8_1 blocks
    GGML_CUDA_SPIN_SITE_EXPERT_CPU,     // expert_cpu_wait_add (expert-cpu.cu)
    GGML_CUDA_SPIN_SITE_FOLD_NORM_PAIR, // rms_norm_add_rms_norm_ar_split_f32, the lines
    GGML_CUDA_SPIN_SITE_COUNT,
};

// a wait that passed the limit, written by the kernel before it traps: pinned host memory outlives the context the trap
//   ends. One per device.
struct ggml_cuda_spin_record {
    unsigned int site;  // ggml_cuda_spin_site + 1, 0 = none
    int          rank;  // of this device in the all-reduce, -1 if not known
    int          peer;  // the rank whose data never came, or GGML_CUDA_SPIN_PEER_*
    unsigned int limit; // units of 2^24 ns
};

// what the waits of a file read, set per device by ggml_cuda_spin_init. Device code is not linked across files, so
//   every file that includes this has its own copy.
struct ggml_cuda_spin_config {
    unsigned int            limit;  // units of 2^24 ns, 0 = none
    ggml_cuda_spin_record * record; // this device's, nullptr = none
};

static __device__ ggml_cuda_spin_config ggml_cuda_spin_cfg = { GGML_CUDA_SPIN_LIMIT_DEFAULT, nullptr };

static_assert((1ull << 32) >> 20 == GGML_CUDA_SPIN_POLLS, "the poll count wraps in the 12 bits above the start");

// Calls poll(), which returns true once the wait is over, until it does, and returns true then; false once the wait has
//   passed the limit (or, once: after the first poll). The polls run in a loop of their own that only counts them, and
//   the timer is read outside it after every GGML_CUDA_SPIN_POLLS of them, so the loop the wait runs is that of the
//   unbounded wait plus an add and a compare: no branch of the timer is in it. One register: the polls since the last
//   read in bits 20..31, the time of the first read in bits 0..19 (0 = not read yet).
template <typename poll_t>
static __device__ __forceinline__ bool ggml_cuda_spin_until(const poll_t & poll, const bool once = false) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    unsigned int s = 0;
#pragma unroll 1
    while (true) {
#pragma unroll 1
        do {
            if (poll()) {
                return true;
            }
            if (once) {
                return false;
            }
            s += 1u << 20;
        } while (s >= 1u << 20);
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        const unsigned int now = (unsigned int) (t >> GGML_CUDA_SPIN_SHIFT) & GGML_CUDA_SPIN_TIME_MASK;
        if (s == 0) {
            s = now == 0 ? 1 : now;
            continue;
        }
        const unsigned int limit = ggml_cuda_spin_cfg.limit;
        if (limit != 0 && ((now - s) & GGML_CUDA_SPIN_TIME_MASK) > limit) {
            return false;
        }
    }
#else
    while (!poll()) {
        if (once) {
            return false;
        }
    }
    return true;
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// the end of a wait that passed the limit: the record, then the trap (no call, see above)
static __device__ __forceinline__ void ggml_cuda_spin_trap(const ggml_cuda_spin_site site, const int rank, const int peer) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    volatile ggml_cuda_spin_record * r = ggml_cuda_spin_cfg.record;
    if (r != nullptr) {
        r->rank  = rank;
        r->peer  = peer;
        r->limit = ggml_cuda_spin_cfg.limit;
        __threadfence_system();
        r->site  = (unsigned int) site + 1;
        __threadfence_system();
    }
    __trap();
#else
    GGML_UNUSED(site);
    GGML_UNUSED(rank);
    GGML_UNUSED(peer);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// the configuration of the waits of each file that includes this header, on the current device
void ggml_cuda_spin_set_allreduce    (const ggml_cuda_spin_config & cfg);
void ggml_cuda_spin_set_allreduce_p2p(const ggml_cuda_spin_config & cfg);
void ggml_cuda_spin_set_mmvf         (const ggml_cuda_spin_config & cfg);
void ggml_cuda_spin_set_norm         (const ggml_cuda_spin_config & cfg);
void ggml_cuda_spin_set_expert_cpu   (const ggml_cuda_spin_config & cfg);

// defines the setter above of the file it is in
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_SPIN_SETTER(name)                                                         \
    void ggml_cuda_spin_set_##name(const ggml_cuda_spin_config & cfg) {                     \
        CUDA_CHECK(cudaMemcpyToSymbol(ggml_cuda_spin_cfg, &cfg, sizeof(cfg)));              \
    }
#else
#define GGML_CUDA_SPIN_SETTER(name)                                                         \
    void ggml_cuda_spin_set_##name(const ggml_cuda_spin_config & cfg) {                     \
        GGML_UNUSED(cfg);                                                                   \
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// gives the waits of every file their record on a device and applies GGML_CUDA_SPIN_TIMEOUT_S, once per device
void ggml_cuda_spin_init(int device);

// prints the record of every wait that timed out, from ggml_cuda_error
void ggml_cuda_spin_print();
