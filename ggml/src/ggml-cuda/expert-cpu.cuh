#pragma once

#include "common.cuh"

// CPU/PCIe split of the expert-cache misses of a one-token MoE layer (GGML_CUDA_EXPERT_CPU=1, off by default, Windows).
//   The plan of the device expert cache (expert_cache_plan) sends the misses with the lowest decayed routing frequency
//   to the CPU instead of copying them: their ids read the zeroed slot past the pool (index cap), so the gathered GEMVs
//   give 0 for them, and they are not put into the pool. It copies as many of the other misses as fit under the time of
//   the CPU (a cost model, see ggml_cuda_expert_cpu_args). Every device makes the same choice from the same ids and
//   frequencies. The publishing device (0) writes x, the ids and their routing weights to a mailbox in mapped host
//   memory; a pool of pinned CPU threads computes those experts with the CPU vec_dot and writes their weighted sum; that
//   device adds it to the input of the all-reduce that ends the layer (ggml_cuda_expert_cpu_merge). The rows of every
//   host expert matrix are split between the two NUMA nodes (ggml_cuda_expert_cpu_place), each socket reads its half.

#define GGML_CUDA_EXPERT_CPU_MAX_K    16
#define GGML_CUDA_EXPERT_CPU_MAX_EMBD 16384

// in mapped pinned host memory, written by the GPU (request side) and the CPU (result side)
struct ggml_cuda_expert_cpu_mbox {
    alignas(64) volatile uint32_t request;   // sequence number of the last request
    alignas(64) volatile uint32_t completed; // sequence number of the last result
    alignas(64) int32_t group;               // ggml_cuda_expert_cpu_group index
    int32_t k;
    int32_t ids[GGML_CUDA_EXPERT_CPU_MAX_K];
    float   w[GGML_CUDA_EXPERT_CPU_MAX_K];
    alignas(64) float x[GGML_CUDA_EXPERT_CPU_MAX_EMBD];
    alignas(64) float out[GGML_CUDA_EXPERT_CPU_MAX_EMBD];
};

// What the plan of one gather group needs. on = 0: the plan copies every miss as before.
struct ggml_cuda_expert_cpu_args {
    ggml_cuda_expert_cpu_mbox * mbox;  // device address on the publishing device, nullptr elsewhere (decide only)
    uint32_t *                  state; // publishing device: [0] last sequence number, [1] the one to wait for (0 = none),
                                       //   u64 at [4]: time of the request, sums of GPU work before the wait and of the
                                       //   wait (ns), waits
    const float *               x;     // the experts' input, n_embd
    const float *               w;     // the routing weights of the row
    int   group;
    int   n_embd;
    // layer time of a split with c copies and k CPU experts: max(c*copy_us, k > 0 ? cpu_a_us + k*cpu_b_us : 0) +
    //   (k > 0 ? post_us : 0); the plan takes the fewest CPU experts within (1 + slack_rel)*best + slack_us, and copies at
    //   least min_copy misses when it can
    float copy_us;
    float cpu_a_us;
    float cpu_b_us;
    float post_us;
    float slack_rel;
    float slack_us;
    int   min_copy;
    int   on;
};

// One expert matrix in host memory as the CPU reads it: n_rows output rows over k inputs, the k range stored as n_pieces
//   blocks one after another inside each expert (device by device, ggml_backend_meta_dm), each with its own row size.
struct ggml_cuda_expert_cpu_mat {
    ggml_type    type;
    const char * base;          // host address of expert 0
    size_t       expert_bytes;
    int64_t      n_rows;
    int64_t      k;
    int          n_pieces;
    size_t       piece_off[GGML_CUDA_MAX_DEVICES];       // in an expert
    size_t       piece_row_bytes[GGML_CUDA_MAX_DEVICES];
    int64_t      piece_k0[GGML_CUDA_MAX_DEVICES];
    int64_t      piece_k[GGML_CUDA_MAX_DEVICES];
};

bool ggml_cuda_expert_cpu_enabled();

// The group of a layer's gate, up and down pools (ggml_cuda_expert_cache pools of one MUL_MAT_ID group), the same on
//   every device; -1 if the CPU cannot take it. glu_op is GGML_GLU_OP_SWIGLU or GGML_GLU_OP_SWIGLU_CLAMP.
int ggml_cuda_expert_cpu_group(const ggml_tensor * gate_pool, const ggml_tensor * up_pool, const ggml_tensor * down_pool,
    int glu_op, float glu_limit, int64_t n_embd);

// the plan arguments of a group on a device
void ggml_cuda_expert_cpu_args_get(int device, int group, const float * x, const float * w, int n_embd,
    ggml_cuda_expert_cpu_args & args);

// the device that publishes and adds the CPU result
int ggml_cuda_expert_cpu_device();

// starts the CPU threads and allocates the mailbox; called where no graph is being captured (expert cache creation)
void ggml_cuda_expert_cpu_init();

// GGML_META_EXPERT_CACHE_STATS: the calls and experts the CPU took
void ggml_cuda_expert_cpu_stats_print();

// keeps the CPU threads spinning for a while: a graph that may ask them is about to run
void ggml_cuda_expert_cpu_kick();

// on the publishing device, after the layer's work and before its all-reduce: waits for the CPU result if the plan asked
//   for one and adds it to dst (n_embd f32)
void ggml_cuda_expert_cpu_merge(cudaStream_t stream, float * dst, int n_embd);

// the host matrix behind a device pool of cached experts, see ggml_cuda_expert_cpu_mat (expert-cache.cu)
bool ggml_cuda_expert_cache_cpu_mat(const ggml_tensor * pool, ggml_cuda_expert_cpu_mat & m);

// the ids of a CPU expert read a zeroed slot of this pool (else a resident slot, whose input must then be zero)
bool ggml_cuda_expert_cache_zero_slot(const ggml_tensor * pool);

// pages of a host expert weight on the NUMA node of the CPU threads that read them, by first touch (ggml-cuda.cu
//   allocates such buffers untouched); false if the tensor is not one
bool ggml_cuda_expert_cpu_place(const ggml_tensor * tensor);

// a large host buffer committed but not touched (nullptr: not wanted, allocate it pinned), and its free (false: not one)
void * ggml_cuda_expert_cpu_host_alloc(size_t size);
bool   ggml_cuda_expert_cpu_host_free(void * ptr);
