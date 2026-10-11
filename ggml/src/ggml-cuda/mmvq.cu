#include "mmvq.cuh"
#include "allreduce-p2p.cuh"
#include "quantize.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10,
    MMVQ_PARAMETERS_VOLTA,  // GENERIC, except for the wide vec_dot of some types, see mmvq_wide
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_TURING
    return MMVQ_PARAMETERS_VOLTA;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_TURING) {
        return MMVQ_PARAMETERS_VOLTA;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// On Volta a thread reads twice the ints of a block of these types, as aligned words where the blocks are only 2-byte
//   aligned: fewer load instructions per byte, and the ints of a pair share their scales. With more tokens the wider
//   loads are slower for q8_0 (from 4 tokens) and q6_K (from 2), and q2_K gains nothing from 5.
static constexpr __host__ __device__ bool mmvq_wide(ggml_type type, mmvq_parameter_table_id table_id, int ncols_dst) {
    if (table_id != MMVQ_PARAMETERS_VOLTA) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_Q8_0: return ncols_dst <= 3;
        case GGML_TYPE_Q2_K: return ncols_dst <= 4;
        case GGML_TYPE_Q4_K: return ncols_dst <= 8;
        case GGML_TYPE_Q5_K: return ncols_dst <= 8;
        case GGML_TYPE_Q6_K: return ncols_dst == 1;
        default:             return false;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq_wide(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return VDR_Q8_0_Q8_1_MMVQ_WIDE;
        case GGML_TYPE_Q2_K: return VDR_Q2_K_Q8_1_MMVQ_WIDE;
        case GGML_TYPE_Q4_K: return VDR_Q4_K_Q8_1_MMVQ_WIDE;
        case GGML_TYPE_Q5_K: return VDR_Q5_K_Q8_1_MMVQ_WIDE;
        case GGML_TYPE_Q6_K: return VDR_Q6_K_Q8_1_MMVQ_WIDE;
        default:             return get_vdr_mmvq(type);
    }
}

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda_wide(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return vec_dot_q8_0_q8_1_wide;
        case GGML_TYPE_Q2_K: return vec_dot_q2_K_q8_1_wide;
        case GGML_TYPE_Q4_K: return vec_dot_q4_K_q8_1_wide;
        case GGML_TYPE_Q5_K: return vec_dot_q5_K_q8_1_wide;
        case GGML_TYPE_Q6_K: return vec_dot_q6_K_q8_1_wide;
        default:             return get_vec_dot_q_cuda(type);
    }
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

// The Volta kernels compile the shared expert channel out: its selects took 6-17 more registers in every fused launch
//   (q4_K gate 96 -> 113), and under -sm tensor the fusion never runs (no graph_optimize).
bool ggml_cuda_mmvq_shared_expert_supported(const int cc) {
    return get_device_table_id(cc) != MMVQ_PARAMETERS_VOLTA;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 6;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_VOLTA) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_VOLTA || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING ||
            table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// the blocks per SM of the conv epilogue: on Volta 9 keep it at the registers of the plain fused kernel where that has 56 or
//   fewer (q8_0: 64 without the bound, a block less per SM, which cost the whole convolution's time); the K-quants take
//   64+ already and would spill
static constexpr __host__ __device__ int mmvq_conv_min_blocks(const ggml_type type) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            return 9;
        default:
            return 1;
    }
#else
    GGML_UNUSED(type);
    return 1;
#endif
}

// the blocks per SM of the paired gate/up kernel: on Volta the q8_0 wide one takes 86 registers without a bound, which leaves
//   2 blocks of 256 threads per SM; 3 fit at 80
static constexpr __host__ __device__ int mmvq_paired_min_blocks(const ggml_type type, const bool wide) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
    return wide && type == GGML_TYPE_Q8_0 ? 3 : 1;
#else
    GGML_UNUSED(type);
    GGML_UNUSED(wide);
    return 1;
#endif
}

// paired: a fused gate on twice the warps, the first half summing x and the second half the gate over the same rows and
//   K slices as the plain kernel, so a thread keeps the sums of one matrix as there and each sum is the plain one
// conv: the convolution of build_conv_state in the epilogue of the rows of x (conv_args), one row per block
// ar: the rows go to the LL lines of an all-reduce folded into this launch (conv_args holds its ggml_cuda_ar_fold_args),
//   not to dst; see ggml_cuda_ar_fold_push
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, bool wide = false,
          bool paired = false, bool conv = false, bool ar = false>
__launch_bounds__((paired ? 2 : 1)*calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(),
    conv ? mmvq_conv_min_blocks(type) : paired ? mmvq_paired_min_blocks(type, wide) : 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride,
        const std::conditional_t<conv, ggml_cuda_mmvq_conv_args, std::conditional_t<ar, ggml_cuda_ar_fold_args, ggml_cuda_no_args>> conv_args) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int vdr = wide ? get_vdr_mmvq_wide(type) : get_vdr_mmvq(type);
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = wide ? get_vec_dot_q_cuda_wide(type) : get_vec_dot_q_cuda(type);

    static_assert(!paired || (has_fusion && ncols_dst == 1), "paired needs a fused gate and one column");
    const     int half = paired ? threadIdx.y / nwarps : 0; // 1: the warps of the gate
    const     int wy   = paired ? threadIdx.y % nwarps : threadIdx.y;
    const     int tid = warp_size*wy + threadIdx.x;

    // without ids, sib_x: the blocks from sib_block0 on compute the rows of the sibling sib_x, which reads the same y,
    //   and those from sib2_block0 on the rows of a second one, sib2_x (for any ncols_dst, a sibling's dst holding its
    //   columns one after the other); gdn_x, one column: the blocks from gdn_block0 on the rows of a gated delta net gate
    //   (with fusion.gate and its epilogue)
    bool gdn_rows  = false;
    if constexpr (ncols_dst == 1) {
        gdn_rows  = ids_ptr == nullptr && fusion.gdn_x  != nullptr && blockIdx.x >= fusion.gdn_block0;
    }
    const bool sib2_rows = ids_ptr == nullptr && fusion.sib2_x != nullptr && blockIdx.x >= fusion.sib2_block0 && !gdn_rows;
    const bool sib_rows  = ids_ptr == nullptr && fusion.sib_x  != nullptr && blockIdx.x >= fusion.sib_block0 && !gdn_rows &&
        !sib2_rows;
    if (sib_rows) {
        vx = fusion.sib_x;
    }
    if (sib2_rows) {
        vx = fusion.sib2_x;
    }
    if (gdn_rows) {
        vx = fusion.gdn_x;
    }
    const     int row0 = rows_per_cuda_block*(blockIdx.x - (gdn_rows ? fusion.gdn_block0 : sib2_rows ? fusion.sib2_block0 :
        sib_rows ? fusion.sib_block0 : 0));
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    // not on Volta, see ggml_cuda_mmvq_shared_expert_supported
    const bool shared_expert = has_fusion && get_device_table_id() != MMVQ_PARAMETERS_VOLTA && fusion.shared_up &&
        blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // with ids, a gate and sib_x: the channel past the used experts is the sibling pair sib_x/sib_gate, see
    //   ggml_cuda_mm_fusion_args_host
    bool sib = false;
    if constexpr (has_fusion && ncols_dst == 1) {
        sib = ids && fusion.sib_x != nullptr && channel_dst == gridDim.y - 1;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? (sib ? 0 : ids[channel_dst]) : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    // ar: the sequence number of the call, loaded while the dot product runs; a block's rows are in one LL block
    static_assert(!ar || (ncols_dst == 1 && !has_fusion && !paired && !conv), "the fold pushes the rows of one column");
    [[maybe_unused]] uint32_t ar_seq = 0;
    if constexpr (ar) {
        if (threadIdx.y == 0 && threadIdx.x < rows_per_cuda_block/2) {
            ar_seq = ggml_cuda_ar_seq(conv_args, row0/4);
        }
    }

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;
    const float * gdn_dt = nullptr;
    const float * gdn_a = nullptr;
    float * gdn_beta = nullptr;

    if constexpr (has_fusion) {
        // with gdn_x only its blocks have the gate and the gated delta net epilogue
        const bool gdn_here = fusion.gdn_x == nullptr || gdn_rows;
        gdn_dt        = gdn_here ? fusion.gdn_dt   : nullptr;
        gdn_a         = gdn_here ? fusion.gdn_a    : nullptr;
        gdn_beta      = gdn_here ? fusion.gdn_beta : nullptr;
        use_gate      = fusion.gate      != nullptr && gdn_here;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
        if (sib) {
            vx    = fusion.sib_x;
            vgate = fusion.sib_gate;
        }
        // gate_ids: the gate's expert sits at its own channel, so its rows are offset from x's
        if constexpr (ncols_dst == 1) {
            if (ids && fusion.gate_ids != nullptr && !sib) {
                vgate = (const char *) vgate + ((int64_t) fusion.gate_ids[channel_dst] - (int64_t) channel_x)*
                    stride_channel_x*ggml_cuda_type_traits<type>::bs;
            }
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    [[maybe_unused]] float x_sig = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < (ncols_dst == 1 && ids ? stride_channel_dst : stride_col_dst))) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (fusion.x_sigmoid) {
                x_sig = ggml_cuda_op_sigmoid_single(*fusion.x_sigmoid);
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // the state columns and weights of the channel of this block's row, loaded while the dot product runs
    static_assert(!conv || (has_fusion && ncols_dst == 1 && rows_per_cuda_block == 1 && !paired), "conv needs one row per block");
    [[maybe_unused]] const bool conv_row = conv && !sib_rows && !sib2_rows && !gdn_rows;
    [[maybe_unused]] float conv_win[4];
    [[maybe_unused]] float conv_wv[4];
    if constexpr (conv) {
        if (conv_row && threadIdx.x == 0 && threadIdx.y == 0) {
            const float * s = conv_args.states + int64_t(conv_args.ids[0])*conv_args.row_stride + int64_t(row0)*3;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                conv_win[k] = s[k];
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                conv_wv[k] = conv_args.w[int64_t(row0)*conv_args.w_stride + k];
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    [[maybe_unused]] const void * vsum = paired && half ? vgate : vx;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            constexpr int pf_dist = 2; // loop iterations, not blocks
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                if constexpr (paired) {
                    tmp[j][i] += vec_dot_q_cuda(
                        vsum, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    continue;
                }
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    // paired: the partial sums of all gate warps, the first of them taking the place of the plain kernel's own
    [[maybe_unused]] __shared__ float tmp_shared_gate[paired ? nwarps : (has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if constexpr (paired) {
        if (half || wy > 0) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                if (half) {
                    tmp_shared_gate[wy][0][i][threadIdx.x] = tmp[0][i];
                } else {
                    tmp_shared[wy-1][0][i][threadIdx.x] = tmp[0][i];
                }
            }
        }
    } else if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst = gdn_rows ? fusion.gdn_dst + row0 : sib2_rows ? fusion.sib2_dst + row0 : sib || sib_rows ? fusion.sib_dst + row0 :
        dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // bound for the rows of a block past the last one; with ids the channels are the used experts, a row stride apart
    const uint32_t nrows_dst = gdn_rows ? fusion.gdn_nrows : sib2_rows ? fusion.sib2_nrows : sib_rows ? fusion.sib_nrows :
        ncols_dst == 1 && ids ? stride_channel_dst : stride_col_dst;
    const uint32_t col_stride_dst = sib_rows || sib2_rows ? nrows_dst : stride_col_dst;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            if constexpr (paired) {
                tmp_gate[j][i] = tmp_shared_gate[0][j][i][threadIdx.x];
#pragma unroll
                for (int l = 1; l < nwarps; ++l) {
                    tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                }
            }
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion && !paired) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (!ar && threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    if (fusion.x_sigmoid) {
                        // rounded on its own, as the MUL before the ADD of the unfused ops
                        result = __fmul_rn(result, x_sig);
                    }
                    result += x_biases[j];
                    if (gdn_beta) {
                        // the gate and beta of a gated delta net layer, x and gate being its alpha and beta
                        //   projections; a single token, so beta is offset by the row alone
                        result = ggml_cuda_op_softplus_single(result + gdn_dt[row0 + i]) * gdn_a[row0 + i];
                        gdn_beta[row0 + i] = ggml_cuda_op_sigmoid_single(tmp_gate[j][i]);
                    } else if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*col_stride_dst + i] = result;
                if constexpr (conv) {
                    // as ssm_conv_state_f32 with one token; conv_dst may sit exactly on dst, so it is written last
                    if (conv_row) {
                        conv_win[3] = result;
                        if (conv_args.slot0 != nullptr) {
#pragma unroll
                            for (int k = 0; k < 3; ++k) {
                                conv_args.slot0[int64_t(row0)*3 + k] = conv_win[k];
                            }
                        }
                        if (conv_args.slot1 != nullptr) {
#pragma unroll
                            for (int k = 0; k < 3; ++k) {
                                conv_args.slot1[int64_t(row0)*3 + k] = conv_win[1 + k];
                            }
                        }
                        conv_args.dst[row0] = ggml_cuda_ssm_conv_silu<4>(conv_win, conv_wv, 0.0f);
                    }
                }
            }
        }
    }

    // ar: every lane holds the sums of the block's rows; lane p pushes rows 2p and 2p + 1 (the host only folds whole
    //   blocks of rows)
    if constexpr (ar && rows_per_cuda_block % 2 == 0) {
#pragma unroll
        for (int p = 0; p < rows_per_cuda_block/2; ++p) {
            if (threadIdx.x == p) {
                const int r = row0 + 2*p;
                ggml_cuda_ar_fold_push(conv_args, ar_seq, r/4, (r/2) % 2, tmp[0][2*p], tmp[0][2*p + 1]);
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate,
            gdn_dt, gdn_a, gdn_beta);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// One token, rows whose K takes at most MMVQ_SHORT_LANES lanes, e.g. the experts of a split MoE: in mul_mat_vec_q such a
// row leaves all but those lanes of a block idle and is bound by the block reduction. Here a warp sums several rows,
// each over its own lanes with the same per-lane K slices and the same xor tree, so a row sums as there (the idle lanes
// add zeros). Optional epilogue: dst = x_bias + sigmoid(x_sigmoid)*(x*y).
// quant_y (q8_0): y is f32 and each lane quantizes its own ints of it as quantize_q8_1 does. The scale and the values do
// not depend on the order of the lanes, and the q8_0 dot product reads no block sum, so the result is the same.
// lanes = WARP_SIZE: a warp per row for K up to MMVQ_SHORT_MAX_LANES lanes. Lane l takes in turn the K slices of lane l of
// each of mul_mat_vec_q's warps and adds them in that kernel's warp order (zeros past K, as there), so the partial sums,
// and with the same xor tree the row, are the same. There the rows of a K-quant expert slice of 2-3 super-blocks (a
// split DeepSeek down projection) keep one or two warps of a block busy at up to 229 registers.
#define MMVQ_SHORT_LANES 16
#define MMVQ_SHORT_NWARPS 4
#define MMVQ_SHORT_MAX_LANES (2*WARP_SIZE)

// ar (with ids, quant_y and the sibling): the MoE all-reduce folded in, its ggml_cuda_ar_fold_moe_args the one value of
//   ar_args (empty otherwise, so the other kernels keep their parameters)
template <typename T>
static __device__ __forceinline__ const T & mmvq_first_arg(const T & a) {
    return a;
}

// ar keeps the registers of the plain kernel (16 blocks per SM): Ornith's 9 x 128 blocks then stay in one wave
template <ggml_type type, int rows_per_thread, bool has_fusion, bool quant_y = false, int lanes = MMVQ_SHORT_LANES, bool ar = false,
          typename... ArArgs>
__launch_bounds__(MMVQ_SHORT_NWARPS*WARP_SIZE, ar ? 16 : 1)
static __global__ void mul_mat_vec_q_short(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr, const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const ArArgs... ar_args) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int rows_per_warp  = WARP_SIZE / lanes;
    constexpr int rows_per_slice = MMVQ_SHORT_NWARPS*rows_per_warp;

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
    static_assert(lanes == MMVQ_SHORT_LANES || lanes == WARP_SIZE, "a row takes 16 lanes or a warp");
    static_assert(lanes % (qi/vdr) == 0, "a block's ints must not straddle two rows");

    const uint32_t channel_dst = blockIdx.y;

    // quant_y with ids and sib_x: the channel past the used experts is the sibling sib_x with its own input sib_y
    const bool sib = quant_y && ids && fusion.sib_x != nullptr && channel_dst == gridDim.y - 1;
    if (sib) {
        vx  = fusion.sib_x;
        vy  = fusion.sib_y;
        dst = fusion.sib_dst;
    }

    const uint32_t channel_x   = ids ? (sib ? 0 : ids[channel_dst])                         : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = ids ? (sib ? 0 : fastmodulo(channel_dst, nchannels_y)) : channel_dst;

    const int lane = threadIdx.x % lanes;
    const int row0 = rows_per_thread*rows_per_slice*blockIdx.x + rows_per_warp*threadIdx.y + threadIdx.x / lanes;
    const int kbx  = lane / (qi/vdr);
    const int kqs  = vdr * (lane % (qi/vdr));

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y;
    if (!sib) {
        dst += channel_dst*stride_channel_dst;
    }

    [[maybe_unused]] float x_biases[rows_per_thread] = { 0.0f };
    [[maybe_unused]] float x_sig = 1.0f;
    if constexpr (has_fusion) {
        if (lane == 0) {
            const float * x_bias = (const float *) fusion.x_bias + channel_dst*stride_channel_dst;
#pragma unroll
            for (int i = 0; i < rows_per_thread; ++i) {
                const int row = row0 + i*rows_per_slice;
                if (x_bias && row < (int) nrows_x) {
                    x_biases[i] = x_bias[row];
                }
            }
            if (fusion.x_sigmoid) {
                x_sig = ggml_cuda_op_sigmoid_single(*fusion.x_sigmoid);
            }
        }
    }

    float tmp[rows_per_thread] = { 0.0f };
    if constexpr (quant_y) {
        static_assert(type == GGML_TYPE_Q8_0 && vdr == 2 && lanes == MMVQ_SHORT_LANES, "quant_y is for q8_0, 16 lanes");
        const bool active = kbx < (int) (ncols_x / qk);
        float4 yv[2] = { make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 0.0f, 0.0f, 0.0f) };
        if (active) {
            const float4 * yf = (const float4 *) ((const float *) vy + channel_y*stride_channel_y + kbx*QK8_1 + 4*kqs);
            yv[0] = yf[0];
            yv[1] = yf[1];
        }
        const float ys[8] = { yv[0].x, yv[0].y, yv[0].z, yv[0].w, yv[1].x, yv[1].y, yv[1].z, yv[1].w };
        float amax = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            amax = fmaxf(amax, fabsf(ys[k]));
        }
#pragma unroll
        for (int offset = 1; offset < qi/vdr; offset <<= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, qi/vdr));
        }
        const float d = amax / 127.0f;
        int u[vdr];
#pragma unroll
        for (int l = 0; l < vdr; ++l) {
            uint32_t packed = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int8_t q = amax == 0.0f ? 0 : roundf(ys[4*l + b] / d);
                packed |= (uint32_t) (uint8_t) q << (8*b);
            }
            u[l] = (int) packed;
        }
        const float d8_1 = __half2float(__float2half_rn(d));
        if (active) {
#pragma unroll
            for (int i = 0; i < rows_per_thread; ++i) {
                const int row = row0 + i*rows_per_slice;
                if (row < (int) nrows_x) {
                    const block_q8_0 * bq8_0 = (const block_q8_0 *) vx + channel_x*stride_channel_x + row*stride_row_x + kbx;
                    int v[vdr];
#pragma unroll
                    for (int l = 0; l < vdr; ++l) {
                        v[l] = get_int_b2(bq8_0->qs, kqs + l);
                    }
                    tmp[i] += vec_dot_q8_0_q8_1_impl<float, vdr>(v, u, __half2float(bq8_0->d), d8_1);
                }
            }
        }
    } else if constexpr (lanes == WARP_SIZE) {
        // the slices of mul_mat_vec_q's warps in its order, each started from zero as that warp's sum is
        constexpr int kbx_per_warp = WARP_SIZE/(qi/vdr);
#pragma unroll
        for (int w = 0; w < MMVQ_SHORT_NWARPS; ++w) {
            const int kbx_w = kbx + w*kbx_per_warp;
#pragma unroll
            for (int i = 0; i < rows_per_thread; ++i) {
                const int row = row0 + i*rows_per_slice;
                float tmp_w = 0.0f;
                if (row < (int) nrows_x && kbx_w < (int) (ncols_x / qk)) {
                    tmp_w += vec_dot_q_cuda(vx, &y[kbx_w*(qk/QK8_1)], channel_x*stride_channel_x + row*stride_row_x + kbx_w, kqs);
                }
                tmp[i] = w == 0 ? tmp_w : tmp[i] + tmp_w;
            }
        }
    } else if (kbx < (int) (ncols_x / qk)) {
#pragma unroll
        for (int i = 0; i < rows_per_thread; ++i) {
            const int row = row0 + i*rows_per_slice;
            if (row < (int) nrows_x) {
                tmp[i] += vec_dot_q_cuda(vx, &y[kbx*(qk/QK8_1)], channel_x*stride_channel_x + row*stride_row_x + kbx, kqs);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < rows_per_thread; ++i) {
        tmp[i] = warp_reduce_sum<lanes>(tmp[i]);
    }

    if (!ar && lane != 0) {
        return;
    }
#pragma unroll
    for (int i = 0; i < rows_per_thread; ++i) {
        const int row = row0 + i*rows_per_slice;
        if ((ar && lane != 0) || row >= (int) nrows_x) {
            break;
        }
        float result = tmp[i];
        if constexpr (has_fusion) {
            if (fusion.x_sigmoid) {
                result = __fmul_rn(result, x_sig);
            }
            result += x_biases[i];
        }
        dst[row] = result;
    }

    // ar: once every expert's block of these rows has written them, the last one sums them as moe_weighted_reduction_f32
    //   (and ggml_cuda_p2p_ar_ll_moe_reduce_kernel) does and pushes the sums; the host folds only whole groups of rows
    if constexpr (ar) {
        static_assert(sizeof...(ArArgs) == 1 && quant_y, "the fold needs its arguments and the sibling");
        const ggml_cuda_ar_fold_moe_args & m = mmvq_first_arg(ar_args...);
        constexpr int rows_per_block = rows_per_thread*rows_per_slice;
        static_assert(rows_per_block % 4 == 0, "whole float4s per block");
        const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;

        __shared__ bool last;
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            last = atomicAdd(m.tickets + blockIdx.x, 1u) == gridDim.y - 1;
            if (last) {
                m.tickets[blockIdx.x] = 0;
                __threadfence();
            }
        }
        __syncthreads();
        if (!last) {
            return;
        }

        // the products of all (row, expert) pairs at once, one per thread (up to 8 experts, which the host checks),
        //   then one thread per row adds them in expert order: few registers, so the blocks stay in one wave
        constexpr int ne_max = 8;
        __shared__ float prod[ne_max][rows_per_block];
        const int r0 = rows_per_block*blockIdx.x;
        for (int k = tid; k < m.n_used*rows_per_block; k += MMVQ_SHORT_NWARPS*WARP_SIZE) {
            const int e = k / rows_per_block;
            const int r = k % rows_per_block;
            prod[e][r] = __fmul_rn(__ldcg(dst_ptr + e*stride_channel_dst + r0 + r), m.weights[e]);
        }
        float xa = 0.0f;
        float sig = 0.0f;
        if (tid < rows_per_block) {
            xa  = __ldcg(fusion.sib_dst + r0 + tid);
            sig = ggml_cuda_op_sigmoid_single(*m.gate);
        }
        __syncthreads();
        float v = 0.0f;
        if (tid < rows_per_block) {
            v = prod[0][tid];
            for (int e = 1; e < m.n_used; ++e) {
                v = __fadd_rn(v, prod[e][tid]);
            }
            v = __fadd_rn(v, __fmul_rn(xa, sig));
        }
        // the warps of the rows, whole: the even row's thread pushes the pair
        if (tid < (rows_per_block + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE) {
            const float v1 = __shfl_down_sync(0xFFFFFFFF, v, 1);
            if (tid < rows_per_block && tid % 2 == 0) {
                const int r = r0 + tid;
                ggml_cuda_ar_fold_push(m.ar, ggml_cuda_ar_seq(m.ar, r/4), r/4, (r/2) % 2, v, v1);
            }
        }
    } else {
        ((void) ar_args, ...);
    }
}

// the lanes of a row in the one-token kernel for short rows (MMVQ_SHORT_LANES, or a warp for K up to MMVQ_SHORT_MAX_LANES
//   lanes), 0 when it does not take the matmul; the same rule for all of its callers
template <ggml_type type>
static int mmvq_short_lanes(const mmvq_parameter_table_id table_id, const int ncols_x, const int ncols_dst, const int nsamples_dst,
        const bool has_ids, const ggml_cuda_mm_fusion_args_device & fusion) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    if constexpr (qi/vdr > WARP_SIZE) {
        GGML_UNUSED_VARS(table_id, ncols_x, ncols_dst, nsamples_dst, has_ids, fusion);
        return 0;
    } else {
        const bool short_fusion = fusion.gate == nullptr && fusion.shared_up == nullptr && fusion.gate_bias == nullptr && fusion.x_scale == nullptr &&
            fusion.gate_scale == nullptr && fusion.gdn_beta == nullptr && (fusion.sib_x == nullptr || fusion.sib_y != nullptr) &&
            (fusion.x_bias == nullptr || !has_ids);
        // measured on Volta only
        if (table_id != MMVQ_PARAMETERS_VOLTA || ncols_dst != 1 || nsamples_dst != 1 || !short_fusion) {
            return 0;
        }
        const int k_lanes = (ncols_x/qk)*(qi/vdr);
        if (qi/vdr <= MMVQ_SHORT_LANES && k_lanes <= MMVQ_SHORT_LANES) {
            return MMVQ_SHORT_LANES;
        }
        // the warp-per-row kernel has no sibling input (quant_y); a cheap dot product (q8_0: 4 lanes a block) only gains
        //   while a row is one pass of the warp, a K-quant (16 or 32 lanes a super-block) for two
        const int max_lanes = qi/vdr >= MMVQ_SHORT_LANES ? MMVQ_SHORT_MAX_LANES : WARP_SIZE;
        return k_lanes <= max_lanes && fusion.sib_x == nullptr ? WARP_SIZE : 0;
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    // not on Volta, see ggml_cuda_mmvq_shared_expert_supported
    const bool shared_expert = has_fusion && get_device_table_id() != MMVQ_PARAMETERS_VOLTA && fusion.shared_up &&
        blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);
    if constexpr (has_fusion) {
        if (fusion.gate_ids != nullptr) {
            vgate = (const char *) vgate + ((int64_t) fusion.gate_ids[channel_dst + token_idx * ids_stride] - (int64_t) channel_x)*
                stride_channel_x*ggml_cuda_type_traits<type>::bs;
        }
    }

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        // picked with constant indices: indexed by threadIdx.x, tmp and tmp_gate would live in local memory
        float result = tmp[0];
#pragma unroll
        for (int i = 1; i < c_rows_per_block; ++i) {
            if ((int) threadIdx.x == i) {
                result = tmp[i];
            }
        }
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[0];
#pragma unroll
                for (int i = 1; i < c_rows_per_block; ++i) {
                    if ((int) threadIdx.x == i) {
                        gate_value = tmp_gate[i];
                    }
                }
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

// the types whose one-token launches can take an all-reduce fold (each one more kernel per layout)
static constexpr bool ggml_cuda_mmvq_ar_fold_type(const ggml_type type) {
    return type == GGML_TYPE_Q8_0;
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, bool wide = false, bool paired = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_launch & fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            dim3 grid = block_nums;
            if (ids && fusion.sib_x) {
                grid.y += 1; // the sibling pair, see mul_mat_vec_q
            }
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block_dims, nbytes_shared, stream);
            if constexpr (!small_k && !halve_iters && !paired) {
                if (fusion.conv.w != nullptr) {
                    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, false, false, wide, false, true>, launch_params,
                         vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                         channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                         sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fusion.conv);
                    return;
                }
            }
            GGML_ASSERT(fusion.conv.w == nullptr);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, wide, paired>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fusion.conv);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");
    if constexpr (paired) {
        GGML_ABORT("the paired layout needs a fused gate");
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);

    // an all-reduce folded into the launch: one column, whole blocks of an even number of rows, nothing else fused
    if constexpr (c_ncols_dst == 1 && small_k && ggml_cuda_mmvq_ar_fold_type(type)) {
        constexpr int rpb = calc_rows_per_block(c_ncols_dst, MMVQ_PARAMETERS_VOLTA, small_k,
            calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_VOLTA, small_k, halve_iters));
        if (fusion.ar_pushed != nullptr && ids == nullptr && fusion.sib_x == nullptr && fusion.sib2_x == nullptr &&
                fusion.gdn_x == nullptr && fusion.shared_up == nullptr && block_dims.y == (unsigned) rpb && rpb % 2 == 0 &&
                block_nums.x*rpb == stride_col_dst && block_nums.y == 1 && block_nums.z == 1 &&
                get_device_table_id(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) == MMVQ_PARAMETERS_VOLTA) {
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, wide, false, false, true>, launch_params,
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fusion.ar);
            *fusion.ar_pushed = true;
            return;
        }
    }

    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, wide>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, fusion.conv);
}

// The wide kernel is compiled only where mmvq_wide can pick it.
template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, bool paired = false, typename... Args>
static void mul_mat_vec_q_switch_wide(const bool wide, Args &&... args) {
    if constexpr (mmvq_wide(type, MMVQ_PARAMETERS_VOLTA, c_ncols_dst)) {
        if (wide) {
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, small_k, halve_iters, true, paired>(std::forward<Args>(args)...);
            return;
        }
    }
    GGML_UNUSED(wide);
    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, small_k, halve_iters, false, paired>(std::forward<Args>(args)...);
}

template <ggml_type type, int rows_per_thread, bool quant_y, int lanes>
static void mul_mat_vec_q_short_launch_rpt(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const int nchannels_dst, cudaStream_t stream,
        const ggml_cuda_mm_fusion_args_launch * fold = nullptr) {
    constexpr int rows_per_block = rows_per_thread*MMVQ_SHORT_NWARPS*(WARP_SIZE/lanes);
    // and the sibling, see mul_mat_vec_q_short
    const dim3 block_nums((nrows_x + rows_per_block - 1) / rows_per_block, nchannels_dst + (ids && fusion.sib_x ? 1 : 0), 1);
    const dim3 block_dims(WARP_SIZE, MMVQ_SHORT_NWARPS, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    // the MoE all-reduce folded in: whole groups of rows, the sibling (the shared expert) and nothing else fused
    if constexpr (quant_y && lanes == MMVQ_SHORT_LANES && rows_per_block % 4 == 0) {
        if (fold != nullptr && fold->ar_moe && fold->ar_pushed != nullptr && ids != nullptr && fusion.sib_x != nullptr &&
                fusion.x_bias == nullptr && fusion.x_sigmoid == nullptr && nrows_x % rows_per_block == 0 &&
                block_nums.x <= 4096 && stride_channel_dst >= nrows_x) {
            ggml_cuda_ar_fold_moe_args m;
            m.ar      = fold->ar;
            m.weights = fold->ar_weights;
            m.gate    = fold->ar_gate;
            m.tickets = fold->ar_tickets;
            m.n_used  = fold->ar_n_used;
            ggml_cuda_kernel_launch(mul_mat_vec_q_short<type, rows_per_thread, false, quant_y, lanes, true, ggml_cuda_ar_fold_moe_args>,
                launch_params, vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x, channel_ratio,
                stride_channel_x, stride_channel_y, stride_channel_dst, m);
            *fold->ar_pushed = true;
            return;
        }
    }
    if (fusion.x_bias != nullptr || fusion.x_sigmoid != nullptr) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_short<type, rows_per_thread, true, quant_y, lanes>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x, channel_ratio, stride_channel_x,
            stride_channel_y, stride_channel_dst);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_short<type, rows_per_thread, false, quant_y, lanes>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x, channel_ratio, stride_channel_x,
            stride_channel_y, stride_channel_dst);
    }
}

// as many rows per thread as keep 8 blocks per SM (Ornith experts, 8 x 2048 rows of K 128: 1/2/4/8 rows 6.3/6.0/6.3/7.8 us)
template <ggml_type type, bool quant_y, int lanes>
static void mul_mat_vec_q_short_launch_lanes(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const int nchannels_dst, const int nsm, cudaStream_t stream,
        const ggml_cuda_mm_fusion_args_launch * fold = nullptr) {
    constexpr int rows_per_slice = MMVQ_SHORT_NWARPS*(WARP_SIZE/lanes);
    const auto nblocks = [&](int rpt) {
        return int64_t((nrows_x + rpt*rows_per_slice - 1) / (rpt*rows_per_slice)) * nchannels_dst;
    };
    if (nblocks(4) >= 8*nsm) {
        mul_mat_vec_q_short_launch_rpt<type, 4, quant_y, lanes>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, nchannels_dst, stream, fold);
    } else if (nblocks(2) >= 8*nsm) {
        mul_mat_vec_q_short_launch_rpt<type, 2, quant_y, lanes>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, nchannels_dst, stream, fold);
    } else {
        mul_mat_vec_q_short_launch_rpt<type, 1, quant_y, lanes>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, nchannels_dst, stream, fold);
    }
}

// lanes from mmvq_short_lanes
template <ggml_type type, bool quant_y = false>
static void mul_mat_vec_q_short_launch(const int lanes,
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint3 nchannels_y, const uint32_t stride_row_x,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const int nchannels_dst, const int nsm, cudaStream_t stream,
        const ggml_cuda_mm_fusion_args_launch * fold = nullptr) {
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    if constexpr (!quant_y) {
        if (lanes == WARP_SIZE) {
            mul_mat_vec_q_short_launch_lanes<type, false, WARP_SIZE>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y,
                stride_row_x, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, nchannels_dst, nsm, stream);
            return;
        }
    }
    if constexpr (qi/vdr <= MMVQ_SHORT_LANES) {
        GGML_ASSERT(lanes == MMVQ_SHORT_LANES);
        mul_mat_vec_q_short_launch_lanes<type, quant_y, MMVQ_SHORT_LANES>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y,
            stride_row_x, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, nchannels_dst, nsm, stream, fold);
    } else {
        GGML_ABORT("no 16-lane short kernel for this type");
    }
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    const auto launch = [&](auto rows_per_block_tag) {
        constexpr int rows_per_block = decltype(rows_per_block_tag)::value;
        const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
        const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
        const dim3 block_dims(warp_size, ncols_dst);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        if (has_fusion) {
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride);
        } else {
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride);
        }
    };

    // 2 rows per block gives best perf based on tuning. A row that one pass of a warp covers (a split expert's K slice:
    //   Qwen3.8-Flash's down projection, 160 columns) leaves a block of 2 rows a vec_dot or two per lane, and the grid is
    //   bound by block launches; there 8 rows per block (each row sums as with 2). The kernel reads its rows unbounded,
    //   so only for whole blocks of rows.
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    static const bool no_short = getenv("GGML_CUDA_MMVQ_MOE_NO_SHORT") != nullptr;
    if (!no_short && (int64_t) (ncols_x/qk) <= vdr*warp_size/qi && nrows_x % 8 == 0) {
        launch(std::integral_constant<int, 8>{});
    } else {
        launch(std::integral_constant<int, 2>{});
    }
}

// On Volta the rows sharing a block reuse the activations and keep more loads in flight at any K, and
// each row still sums in the same order. It only loses once the grid drops below ~8 blocks per SM
// (measured, one token: q4_K 17408x5120 -17%, MoE mxfp4 512x4096 x6 -17%, q8_0 256x4096 +15%).
static bool mmvq_volta_multi_row(const int cc, const int nwarps, const int64_t nrows_x, const int64_t nchannels_dst,
        const int64_t nsamples_dst, const int min_blocks_per_sm = 8) {
    const bool is_nvidia_volta = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING;
    if (!is_nvidia_volta || nwarps <= 1) {
        return false;
    }
    const int64_t nblocks = int64_t((nrows_x + nwarps - 1) / nwarps) * nchannels_dst * nsamples_dst;
    return nblocks >= min_blocks_per_sm*ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
}

// Where several rows in a block are slower, by type and arch.
static bool mmvq_small_k_is_slow(const ggml_type type, const int cc, const int64_t ncols_dst) {
    constexpr std::array<ggml_type, 2> iq_slow_turing = {
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S,
    };
    constexpr std::array<ggml_type, 8> iq_slow_other = {
        GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
    };
    constexpr std::array<ggml_type, 3> slow_pascal = {
        GGML_TYPE_IQ3_S,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
    };

    const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
    const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

    if (is_nvidia_turing_plus) {
        return ncols_dst == 1 && std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end();
    }
    return (ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
        (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
        GGML_CUDA_CC_IS_RDNA(cc);
}

// whether the conv epilogue keeps the occupancy of the plain fused kernel, see mmvq_conv_min_blocks: on Volta for the types
//   whose plain kernel fits 56 registers (q4_K, 72 -> 80 registers, a block less per SM: tg -0.3% with the fold)
bool ggml_cuda_mmvq_conv_epilogue_pays(enum ggml_type type, int cc) {
    const bool is_nvidia_volta = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING;
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
            return is_nvidia_volta;
        default:
            return false;
    }
}

bool ggml_cuda_should_fuse_mmvq_gate(enum ggml_type type, int cc, int64_t nrows_x, int64_t nchannels_dst, int64_t nsamples_dst,
        bool mul_mat_id) {
    const int nwarps = calc_nwarps(type, 1, get_device_table_id(cc));
    if (!mmvq_volta_multi_row(cc, nwarps, nrows_x, nchannels_dst, nsamples_dst) || mmvq_small_k_is_slow(type, cc, 1)) {
        return true;
    }
    // Here the plain kernel keeps 4 rows per block. A dense GEMV then fuses in the paired layout, which keeps them (with
    //   one row per block it was slower than two launches: one token, q4_K 15360x3840 +11%). A MUL_MAT_ID fuses with one
    //   row per block, which wins on small grids, where the saved launches weigh the most (qwen35moe 512x2048 x8 experts:
    //   -1% ms/token end to end), and loses on big ones (q4_K 2048x4096 x6 +9%). Q3_K is slower so at any size.
    if (!mul_mat_id) {
        return true;
    }
    if (type == GGML_TYPE_Q3_K) {
        return false;
    }
    const int64_t nblocks = (nrows_x + nwarps - 1)/nwarps * nchannels_dst * nsamples_dst;
    return nblocks <= 16*ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_launch & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    // the wider threads only pay where the plain kernel takes 2 or more iterations of K: on shorter rows, such as the
    //   experts of a split MoE, they leave most of the block idle
    const auto use_wide = [&](int c_ncols_dst) {
        return mmvq_wide(type, table_id, c_ncols_dst) && blocks_per_row_x > calc_nwarps(type, c_ncols_dst, table_id) * blocks_per_iter_1warp;
    };

    // the rows of the launch: those of the siblings run in the same blocks
    const int64_t nrows_launch = nrows_x + (has_ids ? 0 :
        (fusion.sib_x ? fusion.sib_nrows : 0) + (fusion.sib2_x ? fusion.sib2_nrows : 0) + (fusion.gdn_x ? fusion.gdn_nrows : 0));

    // Whether the plain kernel puts several rows in a block.
    const auto multi_row = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        const int  bpi    = use_wide(c_ncols_dst) ? get_vdr_mmvq_wide(type) * warp_size / qi : blocks_per_iter_1warp;
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * bpi;

        // a row of one K iteration loads too little per block alone, which pays down to 4 blocks per SM (q8_0 2048x1024,
        //   Ornith's ssm_out and attn_output per device: 7.0 -> 5.8 us in situ)
        const int min_blocks_per_sm = blocks_per_row_x <= nwarps * bpi ? 4 : 8;
        if (c_ncols_dst == 1 && mmvq_volta_multi_row(cc, nwarps, nrows_launch, nchannels_dst, nsamples_dst, min_blocks_per_sm)) {
            use = true;
        }

        return use && !mmvq_small_k_is_slow(type, cc, ncols_dst);
    };

    // On Volta a fused gate never keeps several rows per block in one thread: it doubles the sums each row keeps live,
    //   several rows then cut the occupancy (q4_K: 91 registers at 4 rows, 64 at 2, 48 at 1; qwen35moe experts 512x2048
    //   x8, ms/token end to end: 1 row -1.1% vs 2 rows, -3.9% vs 4 rows). A dense one takes the paired layout instead,
    //   with the rows of the plain kernel; the experts of a MUL_MAT_ID keep one row per block.
    const bool is_nvidia_volta = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING;
    const bool volta_gate      = is_nvidia_volta && fusion.gate != nullptr;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        return multi_row(c_ncols_dst) && !volta_gate && fusion.conv.w == nullptr;
    };
    const bool use_paired = volta_gate && !has_ids && ncols_dst == 1 && multi_row(1) && fusion.gdn_x == nullptr;

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids || fusion.conv.w != nullptr) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    // the siblings' rows after those of x in a launch of 2 to 8 columns, see mul_mat_vec_q
    const auto sibling_rows = [&](std::pair<dim3, dim3> & dims, const int c_ncols_dst) {
        ggml_cuda_mm_fusion_args_launch fusion_rows = fusion;
        const int rpb = calc_rows_per_block(c_ncols_dst, table_id);
        if (!has_ids && fusion.sib_x != nullptr) {
            fusion_rows.sib_block0 = dims.first.x;
            dims.first.x += (fusion.sib_nrows + rpb - 1) / rpb;
        }
        if (!has_ids && fusion.sib2_x != nullptr) {
            fusion_rows.sib2_block0 = dims.first.x;
            dims.first.x += (fusion.sib2_nrows + rpb - 1) / rpb;
        }
        return fusion_rows;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag, auto paired_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                constexpr bool c_paired  = decltype(paired_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                        nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                if constexpr (c_paired) {
                    dims.second.y *= 2;
                }
                // the sibling's rows after those of x, see mul_mat_vec_q
                ggml_cuda_mm_fusion_args_launch fusion_rows = fusion;
                const int rpb = calc_rows_per_block(c_ncols_dst, table_id, c_small_k,
                    calc_nwarps(type, c_ncols_dst, table_id, c_small_k, c_halve_iters));
                if (!has_ids && fusion.sib_x != nullptr) {
                    fusion_rows.sib_block0 = dims.first.x;
                    dims.first.x += (fusion.sib_nrows + rpb - 1) / rpb;
                }
                if (!has_ids && fusion.sib2_x != nullptr) {
                    fusion_rows.sib2_block0 = dims.first.x;
                    dims.first.x += (fusion.sib2_nrows + rpb - 1) / rpb;
                }
                if (!has_ids && fusion.gdn_x != nullptr) {
                    fusion_rows.gdn_block0 = dims.first.x;
                    dims.first.x += (fusion.gdn_nrows + rpb - 1) / rpb;
                }
                mul_mat_vec_q_switch_wide<type, c_ncols_dst, c_small_k, c_halve_iters, c_paired>(use_wide(c_ncols_dst),
                    vx, vy, ids, fusion_rows, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (const int short_lanes = mmvq_short_lanes<type>(table_id, ncols_x, c_ncols_dst, nsamples_dst, has_ids, fusion)) {
                mul_mat_vec_q_short_launch<type>(short_lanes, vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y_fd, stride_row_x,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, nchannels_dst,
                    ggml_cuda_info().devices[device].nsm, stream);
                break;
            }

            if (use_paired) {
                launch(std::true_type{},  std::false_type{}, std::true_type{});
            } else if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{}, std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{},  std::false_type{});
            } else {
                launch(std::false_type{}, std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            const ggml_cuda_mm_fusion_args_launch fusion_rows = sibling_rows(dims, c_ncols_dst);
            mul_mat_vec_q_switch_wide<type, c_ncols_dst>(use_wide(c_ncols_dst), vx, vy, ids, fusion_rows, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            const ggml_cuda_mm_fusion_args_launch fusion_rows = sibling_rows(dims, c_ncols_dst);
            mul_mat_vec_q_switch_wide<type, c_ncols_dst>(use_wide(c_ncols_dst), vx, vy, ids, fusion_rows, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            const ggml_cuda_mm_fusion_args_launch fusion_rows = sibling_rows(dims, c_ncols_dst);
            mul_mat_vec_q_switch_wide<type, c_ncols_dst>(use_wide(c_ncols_dst), vx, vy, ids, fusion_rows, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            const ggml_cuda_mm_fusion_args_launch fusion_rows = sibling_rows(dims, c_ncols_dst);
            mul_mat_vec_q_switch_wide<type, c_ncols_dst>(use_wide(c_ncols_dst), vx, vy, ids, fusion_rows, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            const ggml_cuda_mm_fusion_args_launch fusion_rows = sibling_rows(dims, c_ncols_dst);
            mul_mat_vec_q_switch_wide<type, c_ncols_dst>(use_wide(c_ncols_dst), vx, vy, ids, fusion_rows, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            const ggml_cuda_mm_fusion_args_launch fusion_rows = sibling_rows(dims, c_ncols_dst);
            mul_mat_vec_q_switch_wide<type, c_ncols_dst>(use_wide(c_ncols_dst), vx, vy, ids, fusion_rows, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            const ggml_cuda_mm_fusion_args_launch fusion_rows = sibling_rows(dims, c_ncols_dst);
            mul_mat_vec_q_switch_wide<type, c_ncols_dst>(use_wide(c_ncols_dst), vx, vy, ids, fusion_rows, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_launch & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_mul_mat_vec_q_impl(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, const int src1_rot) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_launch fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        // past one column only sibling rows
        GGML_ASSERT(  ids || dst->ne[1] == 1 || (fusion->sib && !fusion->gate && !fusion->x_bias && !fusion->gate_bias &&
            !fusion->x_scale && !fusion->gate_scale && !fusion->x_sigmoid && !fusion->gdn_beta && !fusion->sib_gate &&
            !fusion->sib_y && !fusion->shared_up));
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            // the shared expert takes the channel past the used experts, which the sibling pair also uses
            GGML_ASSERT(!fusion->sib && !fusion->sib2 && !fusion->gdn_x && !fusion->gate_ids && !fusion->x_sigmoid);
            GGML_ASSERT(ggml_cuda_mmvq_shared_expert_supported(cc));
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, fusion->gdn_x ? fusion->gdn_x : src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
        if (fusion->gate_ids) {
            // a bias or scale would be read at x's channel
            GGML_ASSERT(ids && fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale &&
                !fusion->x_sigmoid && !fusion->gdn_beta);
            GGML_ASSERT(fusion->gate_ids->type == GGML_TYPE_I32 && ggml_are_same_shape(fusion->gate_ids, ids) &&
                fusion->gate_ids->nb[0] == ids->nb[0] && fusion->gate_ids->nb[1] == ids->nb[1]);
            fusion_local.gate_ids = (const int32_t *) fusion->gate_ids->data;
        }
        if (fusion->x_sigmoid) {
            GGML_ASSERT(!ids && fusion->x_bias && !fusion->gate && !fusion->x_scale);
            GGML_ASSERT(fusion->x_sigmoid->type == GGML_TYPE_F32 && ggml_nelements(fusion->x_sigmoid) == 1);
            fusion_local.x_sigmoid = (const float *) fusion->x_sigmoid->data;
        }
        if (fusion->sib_gate) {
            // one token with one input row for all experts, a GLU and nothing else
            GGML_ASSERT(ids && fusion->sib && fusion->sib_dst && fusion->gate && dst->ne[2] == 1 && src1->ne[1] == 1);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale && !fusion->x_sigmoid);
            for (const ggml_tensor * t : { fusion->sib, fusion->sib_gate }) {
                GGML_ASSERT(t->type == src0->type && t->ne[0] == ne00 && t->ne[1] == ne01 && t->nb[1] == nb01);
            }
            GGML_ASSERT(fusion->sib_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->sib_dst) && fusion->sib_dst->ne[0] == ne0);
            fusion_local.sib_x    = fusion->sib->data;
            fusion_local.sib_gate = fusion->sib_gate->data;
            fusion_local.sib_dst  = (float *) fusion->sib_dst->data;
        }
        if (fusion->sib && !ids) {
            // the sibling's rows after those of src0 and nothing else fused (but at one token a gate's rows after them);
            //   past one token the rows go in pairs, so they are whole pairs
            GGML_ASSERT(fusion->sib_dst && !fusion->sib_gate && !fusion->sib_y && !fusion->x_bias &&
                !fusion->x_scale && !fusion->x_sigmoid && ggml_nrows(src1) == src1->ne[1]);
            GGML_ASSERT((!fusion->gate && !fusion->gdn_beta) || (fusion->gdn_x && src1->ne[1] == 1));
            GGML_ASSERT(src1->ne[1] == 1 || (ne01 % 2 == 0 && fusion->sib->ne[1] % 2 == 0 && ggml_is_contiguous(dst)));
            GGML_ASSERT(fusion->sib->type == src0->type && fusion->sib->ne[0] == ne00 && fusion->sib->nb[1] == nb01 &&
                ggml_nrows(fusion->sib) == fusion->sib->ne[1]);
            GGML_ASSERT(fusion->sib_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->sib_dst) &&
                ggml_nelements(fusion->sib_dst) == fusion->sib->ne[1]*src1->ne[1]);
            fusion_local.sib_x     = fusion->sib->data;
            fusion_local.sib_dst   = (float *) fusion->sib_dst->data;
            fusion_local.sib_nrows = (uint32_t) fusion->sib->ne[1];
        }
        if (fusion->sib2) {
            GGML_ASSERT(fusion->sib && !ids && fusion->sib2_dst);
            GGML_ASSERT(fusion->sib2->type == src0->type && fusion->sib2->ne[0] == ne00 && fusion->sib2->nb[1] == nb01 &&
                ggml_nrows(fusion->sib2) == fusion->sib2->ne[1] && (src1->ne[1] == 1 || fusion->sib2->ne[1] % 2 == 0));
            GGML_ASSERT(fusion->sib2_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->sib2_dst) &&
                ggml_nelements(fusion->sib2_dst) == fusion->sib2->ne[1]*src1->ne[1]);
            fusion_local.sib2_x     = fusion->sib2->data;
            fusion_local.sib2_dst   = (float *) fusion->sib2_dst->data;
            fusion_local.sib2_nrows = (uint32_t) fusion->sib2->ne[1];
        }
        if (fusion->sib_y) {
            GGML_ASSERT(ggml_cuda_mul_mat_vec_q_short_sibling(src0, src1, ids, fusion->sib, fusion->sib_y) && fusion->sib_out);
            GGML_ASSERT(!fusion->gate && !fusion->x_bias && !fusion->x_scale && !fusion->x_sigmoid && !fusion->sib_gate);
            fusion_local.sib_x   = fusion->sib->data;
            fusion_local.sib_y   = (const float *) fusion->sib_y->data;
            fusion_local.sib_dst = fusion->sib_out;
        }
        if (fusion->gdn_beta) {
            GGML_ASSERT(!ids && fusion->gate && !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            // with gdn_x its rows go after those of src0 (and sib) into gdn_dst
            const int64_t nrows_gdn = fusion->gdn_x ? fusion->gdn_x->ne[1] : dst->ne[0];
            for (const ggml_tensor * t : { fusion->gdn_dt, fusion->gdn_a, (const ggml_tensor *) fusion->gdn_beta }) {
                GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == nrows_gdn);
            }
            if (fusion->gdn_x) {
                GGML_ASSERT(ggml_nrows(src1) == 1 && fusion->gdn_dst && ggml_is_contiguous(fusion->gdn_dst) &&
                    fusion->gdn_dst->type == GGML_TYPE_F32 && ggml_nelements(fusion->gdn_dst) == nrows_gdn);
                for (const ggml_tensor * t : { fusion->gdn_x, fusion->gate }) {
                    GGML_ASSERT(t->type == src0->type && t->ne[0] == ne00 && t->nb[1] == nb01 && t->ne[1] == nrows_gdn);
                }
                fusion_local.gdn_x     = fusion->gdn_x->data;
                fusion_local.gdn_dst   = (float *) fusion->gdn_dst->data;
                fusion_local.gdn_nrows = (uint32_t) nrows_gdn;
                if (fusion->conv_w != nullptr) {
                    GGML_ASSERT(fusion->conv_states && fusion->conv_ids && fusion->conv_dst);
                    fusion_local.conv.states     = fusion->conv_states;
                    fusion_local.conv.ids        = fusion->conv_ids;
                    fusion_local.conv.row_stride = fusion->conv_row_stride;
                    fusion_local.conv.w          = fusion->conv_w;
                    fusion_local.conv.w_stride   = fusion->conv_w_stride;
                    fusion_local.conv.dst        = fusion->conv_dst;
                    fusion_local.conv.slot0      = fusion->conv_slot[0];
                    fusion_local.conv.slot1      = fusion->conv_slot[1];
                }
            }
            GGML_ASSERT(fusion->conv_w == nullptr || fusion->gdn_x != nullptr);
            fusion_local.gdn_dt   = (const float *) fusion->gdn_dt->data;
            fusion_local.gdn_a    = (const float *) fusion->gdn_a->data;
            fusion_local.gdn_beta = (float *) fusion->gdn_beta->data;
        }
    }

    // the all-reduce of the output folded into the launch, if it can take it (see ggml_backend_cuda_context::ar_fold);
    //   with ar_weights that of the MoE sum of the experts this launch computes, which the caller checked is next
    if (ctx.ar_fold.next == dst && src1_rot == 0) {
        fusion_local.ar        = ctx.ar_fold.next_call.args;
        fusion_local.ar_pushed = &ctx.ar_fold.pushed;
    }
    if (fusion && fusion->ar_weights && ctx.ar_fold.next != nullptr && src1_rot == 0) {
        GGML_ASSERT(ids && fusion->sib_y && fusion->ar_gate && ggml_is_contiguous(fusion->ar_weights) &&
            ggml_nelements(fusion->ar_weights) == ne1);
        fusion_local.ar         = ctx.ar_fold.next_call.args;
        fusion_local.ar_pushed  = &ctx.ar_fold.pushed;
        fusion_local.ar_moe     = true;
        fusion_local.ar_weights = (const float *) fusion->ar_weights->data;
        fusion_local.ar_gate    = (const float *) fusion->ar_gate->data;
        fusion_local.ar_tickets = ctx.ar_tickets_get();
        fusion_local.ar_n_used  = (int) ne1;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    // a short q8_0 row quantizes its own part of src1, see mul_mat_vec_q_short
    {
        const int     cc            = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        const int64_t ncols_dst     = ids ? ne2  : ne1;
        const int64_t nchannels_y   = ids ? ne11 : ne12;
        const int64_t nchannels_dst = ids ? ne1  : ne2;
        if (src1_rot == 0 && src0->type == GGML_TYPE_Q8_0 &&
                mmvq_short_lanes<GGML_TYPE_Q8_0>(get_device_table_id(cc), ne00, ncols_dst, ne3, ids != nullptr, fusion_local) == MMVQ_SHORT_LANES &&
                (uintptr_t) src1_d % 16 == 0 && nb11 % 16 == 0 && nb12 % 16 == 0) {
            const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
            const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / ne02);
            mul_mat_vec_q_short_launch<GGML_TYPE_Q8_0, true>(MMVQ_SHORT_LANES, src0->data, src1_d, ids_d, fusion_local, dst_d, ne00, ne01,
                nchannels_y_fd, nb01/ts_src0, channel_ratio_fd, nb02/ts_src0, (ids ? nb11 : nb12)/ts_src1,
                (ids ? nb1 : nb2)/ts_dst, nchannels_dst, ggml_cuda_info().devices[ggml_cuda_get_device()].nsm, stream, &fusion_local);
            return;
        }
    }
    GGML_ASSERT(fusion_local.sib_y == nullptr);

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;

    ggml_cuda_pool_alloc<char> src1_q8_1_rot(ctx.pool());
    const char * src1_q8_1;

    if (src1_rot != 0) {
        // src1 is rotated on the way into q8_1, so the copy is not that of src1 and is not kept for reuse
        GGML_ASSERT(!ids);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_rot_cuda(src1_d, src1_q8_1_rot.alloc(nbytes_src1_q8_1), ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13,
            src1_rot, stream);
        src1_q8_1 = src1_q8_1_rot.get();
    } else {
        // quantize_row_q8_1_cuda ignores src0->type, so the result only depends on src1 and the padded
        // width: reuse it when a later matmul reads the same input, e.g. the q/k/v projections. A view of
        // the same rows of the same tensor reads the same bytes, e.g. the MUL_MAT_ID reshape of a norm
        // that the shared expert reads as it is.
        //   Several tokens: [ne0, 1, T] (MUL_MAT_ID) and [ne0, T] are the same rows in the same order, as is the copy.
        const ggml_tensor * src1_owner = src1->view_src ? src1->view_src : src1;
        auto & cached = ctx.q8_1_inputs[src1->data];
        const auto rows_in_order = [](const auto * ne, const auto * nb) {
            return int64_t(nb[2]) == int64_t(ne[1])*int64_t(nb[1]) && int64_t(nb[3]) == int64_t(ne[2])*int64_t(nb[2]);
        };
        const bool same_rows = cached.ne[0] == src1->ne[0] && int64_t(cached.nb[0]) == int64_t(src1->nb[0]) &&
            int64_t(cached.nb[1]) == int64_t(src1->nb[1]) &&
            cached.ne[1]*cached.ne[2]*cached.ne[3] == src1->ne[1]*src1->ne[2]*src1->ne[3] &&
            rows_in_order(cached.ne, cached.nb) && rows_in_order(src1->ne, src1->nb);
        const bool reuse = cached.buf && cached.src1 == src1_owner && cached.ne_padded == ne10_padded &&
            cached.stream_no == ctx.curr_stream_no &&
            ((std::equal(std::begin(cached.ne), std::end(cached.ne), src1->ne) &&
              std::equal(std::begin(cached.nb), std::end(cached.nb), src1->nb)) || same_rows);

        if (!reuse) {
            cached.buf = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), nbytes_src1_q8_1);
            cached.src1      = src1_owner;
            cached.ne_padded = ne10_padded;
            cached.stream_no = ctx.curr_stream_no;
            std::copy_n(src1->ne, GGML_MAX_DIMS, cached.ne);
            std::copy_n(src1->nb, GGML_MAX_DIMS, cached.nb);

            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            quantize_row_q8_1_cuda(src1_d, nullptr, cached.buf->get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        }

        src1_q8_1 = cached.buf->get();
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

bool ggml_cuda_mul_mat_vec_q_short_sibling(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * sib, const ggml_tensor * sib_y) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    ggml_cuda_mm_fusion_args_device fusion{};
    fusion.sib_x = sib->data;
    fusion.sib_y = (const float *) sib_y->data;
    return ids && src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        mmvq_short_lanes<GGML_TYPE_Q8_0>(get_device_table_id(cc), src0->ne[0], 1, 1, true, fusion) == MMVQ_SHORT_LANES &&
        (uintptr_t) src1->data % 16 == 0 && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0 &&
        sib->type == src0->type && sib->ne[0] == src0->ne[0] && sib->ne[1] == src0->ne[1] && sib->nb[1] == src0->nb[1] &&
        sib_y->type == GGML_TYPE_F32 && ggml_is_contiguous(sib_y) && ggml_nelements(sib_y) == src0->ne[0] &&
        (uintptr_t) sib_y->data % 16 == 0;
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    ggml_cuda_mul_mat_vec_q_impl(ctx, src0, src1, ids, dst, fusion, 0);
}

void ggml_cuda_mul_mat_vec_q_rot(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const int src1_rot) {
    GGML_ASSERT(src1_rot > 0);
    ggml_cuda_mul_mat_vec_q_impl(ctx, src0, src1, nullptr, dst, nullptr, src1_rot);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_launch fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
