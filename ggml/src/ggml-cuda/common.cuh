#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>

#if defined(GGML_USE_HIP)
#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#if defined(GGML_USE_MUSA)
#define GGML_COMMON_DECL_MUSA
#define GGML_COMMON_IMPL_MUSA
#endif
#endif
#include "ggml-common.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#elif defined(GGML_USE_MUSA)
#include "vendors/musa.h"
#else
#include "vendors/cuda.h"
#endif // defined(GGML_USE_HIP)

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32
#define CUDART_HMAX   11070 // CUDA 11.7, min. ver. for which __hmax and __hmax2 are known to work (may be higher than needed)
#define CUDART_HMASK  12000 // CUDA 12.0, min. ver. for half2 -> uint mask comparisons

#define GGML_CUDA_CC_PASCAL          600
#define GGML_CUDA_CC_DP4A            610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define GGML_CUDA_CC_VOLTA           700
#define GGML_CUDA_CC_TURING          750
#define GGML_CUDA_CC_AMPERE          800
#define GGML_CUDA_CC_AMPERE_GA10X    860
#define GGML_CUDA_CC_ORIN            870
#define GGML_CUDA_CC_ADA_LOVELACE    890
#define GGML_CUDA_CC_HOPPER          900
// While BW spans CC 1000, 1100 & 1200, we are integrating Tensor Core instructions available to 1200 family, see
// https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms
#define GGML_CUDA_CC_BLACKWELL       1200
#define GGML_CUDA_CC_DGX_SPARK       1210
#define GGML_CUDA_CC_RUBIN           1300
#define GGML_CUDA_CC_OFFSET_AMD      0x1000000
#define GGML_CUDA_CC_OFFSET_MTHREADS 0x0100000
#define GGML_CUDA_CC_IS_NVIDIA(cc)   (cc < GGML_CUDA_CC_OFFSET_MTHREADS)

// AMD
// GCN/CDNA, wave size is 64
#define GGML_CUDA_CC_GCN4       (GGML_CUDA_CC_OFFSET_AMD + 0x803)  // Tonga, Fiji, Polaris, minimum for fast fp16
#define GGML_CUDA_CC_VEGA       (GGML_CUDA_CC_OFFSET_AMD + 0x900)  // Vega56/64, minimum for fp16 dual issue
#define GGML_CUDA_CC_VEGA20     (GGML_CUDA_CC_OFFSET_AMD + 0x906)  // MI50/Radeon VII, minimum for dp4a
#define GGML_CUDA_CC_GFX909     (GGML_CUDA_CC_OFFSET_AMD + 0x909)  // GCN APU
#define GGML_CUDA_CC_GFX90C     (GGML_CUDA_CC_OFFSET_AMD + 0x90c)  // GCN APU
#define GGML_CUDA_CC_CDNA1      (GGML_CUDA_CC_OFFSET_AMD + 0x908)  // MI100, minimum for MFMA, acc registers
#define GGML_CUDA_CC_CDNA2      (GGML_CUDA_CC_OFFSET_AMD + 0x90a)  // MI210 (gfx90a), minimum acc register renaming
#define GGML_CUDA_CC_CDNA3      (GGML_CUDA_CC_OFFSET_AMD + 0x942)  // MI300
#define GGML_CUDA_CC_CDNA4      (GGML_CUDA_CC_OFFSET_AMD + 0x950)  // MI350X/MI355X

// RDNA removes MFMA, dp4a, xnack, acc registers, wave size is 32
#define GGML_CUDA_CC_RDNA1      (GGML_CUDA_CC_OFFSET_AMD + 0x1010) // RX 5000
#define GGML_CUDA_CC_RDNA2      (GGML_CUDA_CC_OFFSET_AMD + 0x1030) // RX 6000, minimum for dp4a
#define GGML_CUDA_CC_RDNA3      (GGML_CUDA_CC_OFFSET_AMD + 0x1100) // RX 7000, minimum for WMMA
#define GGML_CUDA_CC_RDNA3_5    (GGML_CUDA_CC_OFFSET_AMD + 0x1150) // AI 370, AI Max 395 laptops.
#define GGML_CUDA_CC_RDNA4      (GGML_CUDA_CC_OFFSET_AMD + 0x1200) // RX 9000

#define GGML_CUDA_CC_IS_AMD(cc)     (cc >= GGML_CUDA_CC_OFFSET_AMD)
#define GGML_CUDA_CC_IS_RDNA(cc)    (cc >= GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_RDNA1(cc)   (cc >= GGML_CUDA_CC_RDNA1 && cc < GGML_CUDA_CC_RDNA2)
#define GGML_CUDA_CC_IS_RDNA2(cc)   (cc >= GGML_CUDA_CC_RDNA2 && cc < GGML_CUDA_CC_RDNA3)
#define GGML_CUDA_CC_IS_RDNA3_0(cc) (cc >= GGML_CUDA_CC_RDNA3 && cc < GGML_CUDA_CC_RDNA3_5)
#define GGML_CUDA_CC_IS_RDNA3_5(cc) (cc >= GGML_CUDA_CC_RDNA3_5 && cc < GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_RDNA3(cc)   (GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc))
#define GGML_CUDA_CC_IS_RDNA4(cc)   (cc >= GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_GCN_APU(cc) ((cc) == GGML_CUDA_CC_GFX909 || (cc) == GGML_CUDA_CC_GFX90C)
#define GGML_CUDA_CC_IS_GCN(cc)     ((cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_CDNA1) || GGML_CUDA_CC_IS_GCN_APU(cc))
#define GGML_CUDA_CC_IS_CDNA(cc)    (!GGML_CUDA_CC_IS_GCN_APU(cc) && cc >= GGML_CUDA_CC_CDNA1 && cc < GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_CDNA1(cc)   (GGML_CUDA_CC_IS_CDNA(cc) && cc >= GGML_CUDA_CC_CDNA1 && cc < GGML_CUDA_CC_CDNA2)
#define GGML_CUDA_CC_IS_CDNA2(cc)   (GGML_CUDA_CC_IS_CDNA(cc) && cc >= GGML_CUDA_CC_CDNA2 && cc < GGML_CUDA_CC_CDNA3)
#define GGML_CUDA_CC_IS_CDNA3(cc)   (GGML_CUDA_CC_IS_CDNA(cc) && cc >= GGML_CUDA_CC_CDNA3 && cc < GGML_CUDA_CC_CDNA4)
#define GGML_CUDA_CC_IS_CDNA4(cc)   (GGML_CUDA_CC_IS_CDNA(cc) && cc >= GGML_CUDA_CC_CDNA4 && cc < GGML_CUDA_CC_RDNA1)

// Moore Threads
#define MUSART_HMASK 40300 // MUSA rc4.3, min. ver. for half2 -> uint mask comparisons

#define GGML_CUDA_CC_QY1 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x210) // MTT S80, MTT S3000
#define GGML_CUDA_CC_QY2 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x220) // MTT S4000
#define GGML_CUDA_CC_PH1 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x310) // MTT S5000

#define GGML_CUDA_CC_IS_MTHREADS(cc) (cc >= GGML_CUDA_CC_OFFSET_MTHREADS && cc < GGML_CUDA_CC_OFFSET_AMD)
#define GGML_CUDA_CC_IS_QY1(cc)      (cc >= GGML_CUDA_CC_QY1 && cc < GGML_CUDA_CC_QY2)
#define GGML_CUDA_CC_IS_QY2(cc)      (cc >= GGML_CUDA_CC_QY2 && cc < GGML_CUDA_CC_PH1)
#define GGML_CUDA_CC_IS_PH1(cc)      (cc >= GGML_CUDA_CC_PH1)

#if !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)
#    define GGML_CUDA_USE_CUB
#endif  // !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)

// PDL host-side support (cudaLaunchKernelEx) requires CUDART >= 11.8.
// However, this has been bugged in CTK < 12.3 for MSVC builds, see
// https://github.com/ggml-org/llama.cpp/pull/22522#discussion_r3302393293
// __CUDA_ARCH__  is undefined in host passes; GPU arch check happens in device-side code.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && \
    (CUDART_VERSION >= 12030 || (!(defined(_MSC_VER) && !defined(__clang__)) && CUDART_VERSION >= 11080))
#    define GGML_CUDA_USE_PDL
#endif  // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (CUDART_VERSION >= 12030 || (!(defined(_MSC_VER) && !defined(__clang__)) && CUDART_VERSION >= 11080))

static __device__ __forceinline__ void ggml_cuda_syncwarp() {
#ifndef GGML_USE_HIP
    __syncwarp();
#endif // GGML_USE_HIP
}

static __device__ __forceinline__ void ggml_cuda_pdl_sync() {
#if defined(GGML_CUDA_USE_PDL) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
    cudaGridDependencySynchronize();
#endif // defined(GGML_CUDA_USE_PDL) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
}

static __device__ __forceinline__ void ggml_cuda_pdl_lc() {
#if defined(GGML_CUDA_USE_PDL) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
    cudaTriggerProgrammaticLaunchCompletion();
#endif // defined(GGML_CUDA_USE_PDL) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
}

#ifdef __CUDA_ARCH_LIST__
constexpr bool ggml_cuda_has_arch_impl(int) {
    return false;
}

template<class ... Archs>
constexpr bool ggml_cuda_has_arch_impl(const int arch, const int first, Archs... rest) {
    return arch == first || ggml_cuda_has_arch_impl(arch, rest...);
}

constexpr bool ggml_cuda_has_arch(const int arch) {
    return ggml_cuda_has_arch_impl(arch, __CUDA_ARCH_LIST__);
}

constexpr int ggml_cuda_highest_compiled_arch_impl(const int /*arch*/, const int cur) {
    if (cur == 0) {
        return -1;
    }
    return cur;
}

template<class ... Archs>
constexpr int ggml_cuda_highest_compiled_arch_impl(const int arch, const int cur, const int first, Archs... rest) {
    if (first <= arch && first > cur) {
        return ggml_cuda_highest_compiled_arch_impl(arch, first, rest...);
    } else {
        return ggml_cuda_highest_compiled_arch_impl(arch, cur, rest...);
    }
}

constexpr int ggml_cuda_highest_compiled_arch(const int arch) {
    return ggml_cuda_highest_compiled_arch_impl(arch, 0, __CUDA_ARCH_LIST__);
}
#else
static int ggml_cuda_highest_compiled_arch(const int arch) {
    return arch;
}
#endif // __CUDA_ARCH_LIST__

// ---------------------------------------------------------------------------------------------------------

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#define GGML_CUDA_MAX_STREAMS 8

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

#define CUDA_CHECK_GEN(err, success, error_fn)                                      \
     do {                                                                           \
        auto err_ = (err);                                                          \
        if (err_ != (success)) {                                                    \
            ggml_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));    \
        }                                                                           \
    } while (0)

#define CUDA_CHECK(err) CUDA_CHECK_GEN(err, cudaSuccess, cudaGetErrorString)


#if CUDART_VERSION >= 12000 || defined(GGML_USE_MUSA)
    static const char * cublas_get_error_str(const cublasStatus_t err) {
        return cublasGetStatusString(err);
    }
#else
    static const char * cublas_get_error_str(const cublasStatus_t err) {
        switch (err) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
            default: return "unknown error";
        }
    }
#endif // CUDART_VERSION >= 12000

#define CUBLAS_CHECK(err) CUDA_CHECK_GEN(err, CUBLAS_STATUS_SUCCESS, cublas_get_error_str)

#ifdef GGML_USE_NCCL
#define NCCL_CHECK(err) CUDA_CHECK_GEN(err, ncclSuccess, ncclGetErrorString)
#endif // GGML_USE_NCCL

#if !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)
static const char * cu_get_error_str(CUresult err) {
    const char * err_str;
    cuGetErrorString(err, &err_str);
    return err_str;
}
#define CU_CHECK(err) CUDA_CHECK_GEN(err, CUDA_SUCCESS, cu_get_error_str)
#endif

#if !defined(GGML_USE_HIP)
#    define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)                                                       \
        do {                                                                                                   \
            static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = { false };                         \
            const int   id                                                = ggml_cuda_get_device();            \
            if (!shared_memory_limit_raised[id]) {                                                             \
                CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes)); \
                shared_memory_limit_raised[id] = true;                                                         \
            }                                                                                                  \
        } while (0)
#else
#    define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes) \
        do {                                             \
            GGML_UNUSED(nbytes);                         \
        } while (0)
#endif // !defined(GGML_USE_HIP)

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
#define GGML_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define GGML_CUDA_ASSUME(x)
#endif // CUDART_VERSION >= 11010

#if (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))
#define GGML_USE_VMM
#endif // (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define FP16_AVAILABLE
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL

#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif // defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610

#if defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)
#define AMD_MFMA_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)

#if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))
#define AMD_WMMA_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(RDNA4)

// The Volta instructions are in principle available on Turing or newer but they are effectively unusable:
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
#    define BLACKWELL_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define CP_ASYNC_AVAILABLE
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#if !defined(GGML_CUDA_NO_FA) && !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ < 220)
#define FLASH_ATTN_AVAILABLE
#endif // !defined(GGML_CUDA_NO_FA) && !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ < 220)

static bool fp16_available(const int cc) {
    return ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_PASCAL ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_PH1);
}

static bool fast_fp16_available(const int cc) {
    return GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_NVIDIA(cc) && fp16_available(cc) && ggml_cuda_highest_compiled_arch(cc) != 610) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && fp16_available(cc));
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fast_fp16_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL && cc != 610) || GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fp16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fast_bf16_hardware_available(const int cc) {
        return (GGML_CUDA_CC_IS_AMD(cc) && (cc >= GGML_CUDA_CC_RDNA3 || GGML_CUDA_CC_IS_CDNA(cc)))
            || (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE);
}

static bool bf16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE) ||
        GGML_CUDA_CC_IS_CDNA(cc) || cc >= GGML_CUDA_CC_RDNA3 ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_PH1);
}

static bool fp32_mma_hardware_available(const int cc) {
    return GGML_CUDA_CC_IS_CDNA(cc);
}

static bool amd_mfma_available(const int cc) {
#if !defined(GGML_HIP_NO_MMQ_MFMA)
    return GGML_CUDA_CC_IS_CDNA(cc);
#else
    return false;
#endif //!defined(GGML_HIP_NO_MMQ_MFMA)
}

static bool amd_wmma_available(const int cc) {
    return (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc));
}

static bool volta_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA;
}

static bool turing_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}

static bool ampere_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}

static bool cp_async_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}

static bool blackwell_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_BLACKWELL &&
           ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_RUBIN;
}

// Checks whether the tensor's base data pointer and higher-dimensional strides are byte-aligned to `alignment` bytes.
static bool ggml_cuda_is_aligned(const ggml_tensor * tensor, const size_t alignment) {
    GGML_ASSERT(tensor != nullptr);
    return (reinterpret_cast<uintptr_t>(tensor->data) % alignment) == 0 &&
           tensor->nb[1] % alignment == 0 &&
           tensor->nb[2] % alignment == 0 &&
           tensor->nb[3] % alignment == 0;
}

static bool ggml_cuda_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// ggml-alloc places the output of a fused chain as if the elided nodes ran on their own, so it can land partly over an
// input the chain still reads. Sitting exactly on a same-shape input is fine when each element is read before the same
// thread writes it; a partial overlap lets one block overwrite input rows another block has not read yet.
static bool ggml_cuda_same_or_disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    return a->data == b->data || !ggml_cuda_tensors_overlap(a, b);
}

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
    return 64;
#else
    return 32;
#endif // defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
}

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return 16;
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 16;
#else
    return 8;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}


[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {

#if defined(GGML_USE_HIP)
    printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %d.\n",
           file_name, line, function_name, arch);
    GGML_UNUSED(arch_list);
#else
    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. ggml-cuda.cu was compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
#endif // defined(GGML_USE_HIP)
    __trap();

    GGML_UNUSED(no_device_code); // suppress unused function warning
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ABORT("NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__

// The compiler is always able to unroll loops if they contain continue expressions.
// In such cases loop unrolling can still be achieved via recursion:
template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    return __reduce_add_sync(0xffffffff, x);
#else
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, offset, width);
        a.y += __shfl_xor_sync(0xffffffff, a.y, offset, width);
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#ifdef FP16_AVAILABLE
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, offset, width));
    }
    return a;

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_all(int x) {
    if (width == ggml_cuda_get_physical_warp_size()) {
        return __all_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) && x;
        }
        return x;
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_any(int x) {
    if (width == ggml_cuda_get_physical_warp_size()) {
        return __any_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) || x;
        }
        return x;
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

// The q8_1 copy of a row as quantize_q8_1 writes it: v is value lane of the block of 32 at yb, one per lane of a warp.
static __device__ __forceinline__ void quantize_q8_1_lane(const float v, block_q8_1 * yb, const int lane) {
    float amax = fabsf(v);
    float sum  = v;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
    yb->qs[lane] = q;
    if (lane == 0) {
        yb->ds = make_half2(d, sum);
    }
}

template<typename T, int width = WARP_SIZE>
static __device__ __forceinline__ T warp_prefix_inclusive_sum(T x) {
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const T t = __shfl_up_sync(0xffffffff, x, offset, width);
        if (lane_id >= offset) {
            x += t;
        }
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_prefix_inclusive_sum(float2 a) {
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const float t_x = __shfl_up_sync(0xffffffff, a.x, offset, width);
        const float t_y = __shfl_up_sync(0xffffffff, a.y, offset, width);
        if (lane_id >= offset) {
            a.x += t_x;
            a.y += t_y;
        }
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_prefix_inclusive_sum(half2 a) {
#ifdef FP16_AVAILABLE
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const half2 t = __shfl_up_sync(0xffffffff, a, offset, width);
        if (lane_id >= offset) {
            a = __hadd2(a, t);
        }
    }
    return a;

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

enum class block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T>
struct block_reduce_policy;

template <typename T, typename... Ts>
inline constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
inline constexpr bool ggml_cuda_dependent_false_v = false;

template <typename T> struct block_reduce_policy<block_reduce_method::SUM, T> {
    static __device__ T reduce(T val) {
        if constexpr(is_any<T, float, float2, half2, int>) {
            return warp_reduce_sum(val);
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    static __device__ T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return 0.0f;
        } else if constexpr (std::is_same_v<T, float2>) {
            return make_float2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, half2>) {
            return make_half2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, int>) {
            return 0;
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T> struct block_reduce_policy<block_reduce_method::MAX, T> {
    static __device__ T reduce(T val) {
        if constexpr (is_any<T, float, half2>) {
            return warp_reduce_max(val);
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    static __device__ T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return -INFINITY;
        } else if constexpr (std::is_same_v<T, half2>) {
            return make_half2(-INFINITY, -INFINITY);
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};

template <block_reduce_method reduce_method_t, const unsigned int block_size_template = 0, typename T>
static __device__ T block_reduce(T val, [[maybe_unused]] T * shared_vals) {
    // for multi-warp reductions, callers must not reuse shared_vals until all reads from this invocation have completed
    val                           = block_reduce_policy<reduce_method_t, T>::reduce(val);
    const unsigned int block_size = block_size_template == 0 ? blockDim.x : block_size_template;
    if (block_size > WARP_SIZE) {
        assert((block_size <= 1024) && (block_size % WARP_SIZE) == 0);
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        __syncthreads();
        val = block_reduce_policy<reduce_method_t, T>::sentinel();
        if (lane_id < (static_cast<int>(block_size) / WARP_SIZE)) {
            val = shared_vals[lane_id];
        }
        return block_reduce_policy<reduce_method_t, T>::reduce(val);
    }

    return val;
}

static __device__ __forceinline__ half ggml_cuda_hmax(const half a, const half b) {
#ifdef FP16_AVAILABLE

#if !defined(GGML_USE_HIP) && CUDART_VERSION < CUDART_HMAX
    return __float2half(fmaxf(__half2float(a), __half2float(b)));
#else
    return __hmax(a, b);
#endif // !defined(GGML_USE_HIP) && CUDART_VERSION < CUDART_HMAX

#else
   NO_DEVICE_CODE;
   GGML_UNUSED(b);
   return a;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ half2 ggml_cuda_hmax2(const half2 a, const half2 b) {
#if defined(GGML_USE_HIP)
    return half2(__hmax(a.x, b.x), __hmax(a.y, b.y));
#elif CUDART_VERSION >= CUDART_HMAX
    return __hmax2(a, b);
#else
    half2 ret;
    reinterpret_cast<half&>(ret.x) = __float2half(fmaxf( __low2float(a),  __low2float(b)));
    reinterpret_cast<half&>(ret.y) = __float2half(fmaxf(__high2float(a), __high2float(b)));
    return ret;
#endif
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
#pragma unroll
   for (int offset = width/2; offset > 0; offset >>= 1) {
       x = ggml_cuda_hmax2(x, __shfl_xor_sync(0xffffffff, x, offset, width));
   }
   return x;
}

#if (defined(CUDART_VERSION) && CUDART_VERSION < CUDART_HMASK) || defined(GGML_USE_HIP) || \
    (defined(MUSART_VERSION) && MUSART_VERSION < MUSART_HMASK)
static __device__ __forceinline__ uint32_t __hgt2_mask(const half2 a, const half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float( __low2half(a)) > float( __low2half(b)));
    const uint32_t mask_high = 0xFFFF0000 * (float(__high2half(a)) > float(__high2half(b)));
    return mask_low | mask_high;
}
#endif // (defined(CUDART_VERSION) && CUDART_VERSION < CUDART_HMASK) || defined(GGML_USE_HIP) || (defined(MUSART_VERSION) && MUSART_VERSION < MUSART_HMASK)

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIP)
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3) || defined(RDNA4)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(RDNA1) || defined(__gfx900__)
    int tmp1;
    int tmp2;
    asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
    return c;

#else // defined(GGML_USE_HIP)

#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)

#endif // defined(GGML_USE_HIP)
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float2 v, const float2 u) {
    acc += v.x*u.x;
    acc += v.y*u.y;
}

#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))
#define V_DOT2_F32_F16_AVAILABLE
#endif // defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const half2 v, const half2 u) {
#ifdef V_DOT2_F32_F16_AVAILABLE
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
#else
#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(v*u);
    acc += tmp.x + tmp.y;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    acc += tmpv.x * tmpu.x;
    acc += tmpv.y * tmpu.y;
#endif // FAST_FP16_AVAILABLE
#endif // V_DOT2_F32_F16_AVAILABLE
}

static __device__ __forceinline__ void ggml_cuda_mad(half2 & acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    acc += v*u;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    float2 tmpacc = __half22float2(acc);
    tmpacc.x += tmpv.x * tmpu.x;
    tmpacc.y += tmpv.y * tmpu.y;
    acc = make_half2(tmpacc.x, tmpacc.y);
#endif // FAST_FP16_AVAILABLE
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD.
// Important: do not use this function if dst and src both point at registers.
//     Due to the strict aliasing rule the compiler can do incorrect optimizations if src and dst have different types.
//     The function is intended for copies between registers and SRAM/VRAM to make the compiler emit the right instructions.
//     If dst and src point at different address spaces then they are guaranteed to not be aliased.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "You are misusing the alignment parameter for ggml_cuda_memcpy_1. "
        "The intent is for the parameter is only as a workaround if either one of the pointers is not properly aligned. "
        "If you use it to do more bytes per copy than ggml_cuda_max_cpy_bytes() the reads and writes may not be coalesced. "
        "Call ggml_cuda_memcpy_1 in a loop instead.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

// n ints from p, which is only 2-byte aligned (the quants of a q8_0 block): n aligned 32-bit loads and one 16-bit load
//   rather than 2n 16-bit loads, the ints are then shifted out of the words when p is not 4-byte aligned.
//   Every load stays within the n ints at p and the 2 bytes before them.
template <int n>
static __device__ __forceinline__ void ggml_cuda_load_ints_a2(int * __restrict__ dst, const void * __restrict__ p) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    ggml_cuda_memcpy_1<n*sizeof(int), 2>(dst, p);
#else
    const int   shift = 8*(uintptr_t(p) % 4); // 0 or 16
    const int * w     = (const int *) ((const char *) p - shift/8);

    int words[n + 1];
#pragma unroll
    for (int i = 0; i < n; ++i) {
        words[i] = w[i];
    }
    words[n] = shift ? *((const unsigned short *) p + 2*n - 1) : 0;
#pragma unroll
    for (int i = 0; i < n; ++i) {
        dst[i] = __funnelshift_r(words[i], words[i + 1], shift);
    }
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
#if CUDART_VERSION >= 12080
    const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
    return (float) e;
#else
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
#endif // CUDART_VERSION >= 12050
}

static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
#if defined(GGML_USE_HIP) && defined(CDNA3) && defined(FP8_AVAILABLE) && HIP_VERSION >= 60200000
    // ROCm does not support fp8 in software on devices with fp8 hardware,
    // but CDNA3 supports only e4m3_fnuz (no inf).
    const uint32_t bits = x * (x != 0x7F && x != 0xFF); // Convert NaN to 0.0f to match CPU implementation.
    const __hip_fp8_e4m3_fnuz xf = *reinterpret_cast<const __hip_fp8_e4m3_fnuz *>(&bits);
    return static_cast<float>(xf) / 2;
#else
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
    const uint32_t bits = x * (x != 0x7F && x != 0xFF); // Convert NaN to 0.0f to match CPU implementation.
    const __nv_fp8_e4m3 xf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&bits);
    return static_cast<float>(xf) / 2;
#else
    if (x == 0 || (x == 0x7F && x != 0xFF)) { // Convert NaN to 0.0f
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return static_cast<float>(raw / 2);
#endif // defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
#endif // defined(GGML_USE_HIP) && defined(CDNA3) && defined(FP8_AVAILABLE) && HIP_VERSION >= 60200000
}

static __device__ __forceinline__ uint8_t ggml_cuda_fp32_to_ue4m3(float x) {
#if defined(BLACKWELL_MMA_AVAILABLE) // This is used for NVFP4 subblock scale quantizations only
    if (!(x > 0.0f)) {
        return 0;
    }
    const __nv_fp8_e4m3 xf(x);
    return xf.__x;
#else
     GGML_UNUSED(x);
     NO_DEVICE_CODE; // Used only for NVFP4 Scales for Activations, only for Blackwell
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

__device__ __forceinline__ uint8_t ggml_cuda_float_to_fp4_e2m1(float x, float e) {
    const uint8_t sign_bit = (x < 0.0f) << 3;
    float         ax       = fabsf(x) * e;

    // Positive LUT
    static constexpr float pos_lut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

    int   best_i   = 0;
    float best_err = fabsf(ax - pos_lut[0]);

#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const float err = fabsf(ax - pos_lut[i]);
        if (err < best_err) {
            best_err = err;
            best_i   = i;
        }
    }

    return static_cast<uint8_t>(best_i | sign_bit);
}

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static const uint3 init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
    // fastdiv_values.z is unused and optimized away by the compiler.
    // Compute high 32 bits of n * mp
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    // add n, apply bit shift
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

// Calculate both division and modulo at once, returns <n/divisor, n%divisor>
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

typedef void (*dequantize_kernel_t)(const void * vx, const int64_t ib, const int iqs, float2 & v);

template<typename dst_t>
using dequantize_kq_t = void (*)(const void * vx, const int64_t ib, dst_t * y, const int tid);

static __device__ __forceinline__ float get_alibi_slope(
    const float max_bias, const uint32_t h, const uint32_t n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return powf(base, exph);
}

template <ggml_type type>
struct ggml_cuda_type_traits;

template<>
struct ggml_cuda_type_traits<GGML_TYPE_F16> {
    static constexpr int qk = 1;
    static constexpr int qr = 1;
    static constexpr int bs = sizeof(ggml_half);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q1_0> {
    static constexpr int qk = QK1_0;
    static constexpr int qr = QR1_0;
    static constexpr int qi = QI1_0;
    static constexpr int bs = sizeof(block_q1_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q2_0> {
    static constexpr int qk = QK2_0;
    static constexpr int qr = QR2_0;
    static constexpr int qi = QI2_0;
    static constexpr int bs = sizeof(block_q2_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;
    static constexpr int qr = QR4_0;
    static constexpr int qi = QI4_0;
    static constexpr int bs = sizeof(block_q4_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_1> {
    static constexpr int qk = QK4_1;
    static constexpr int qr = QR4_1;
    static constexpr int qi = QI4_1;
    static constexpr int bs = sizeof(block_q4_1);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_0> {
    static constexpr int qk = QK5_0;
    static constexpr int qr = QR5_0;
    static constexpr int qi = QI5_0;
    static constexpr int bs = sizeof(block_q5_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_1> {
    static constexpr int qk = QK5_1;
    static constexpr int qr = QR5_1;
    static constexpr int qi = QI5_1;
    static constexpr int bs = sizeof(block_q5_1);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;
    static constexpr int qr = QR8_0;
    static constexpr int qi = QI8_0;
    static constexpr int bs = sizeof(block_q8_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_MXFP4> {
    static constexpr int qk = QK_MXFP4;
    static constexpr int qr = QR_MXFP4;
    static constexpr int qi = QI_MXFP4;
    static constexpr int bs = sizeof(block_mxfp4);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_NVFP4> {
    static constexpr int qk = QK_NVFP4;
    static constexpr int qr = QR_NVFP4;
    static constexpr int qi = QI_NVFP4;
    static constexpr int bs = sizeof(block_nvfp4);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q2_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_K;
    static constexpr int qi = QI2_K;
    static constexpr int bs = sizeof(block_q2_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q3_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_K;
    static constexpr int qi = QI3_K;
    static constexpr int bs = sizeof(block_q3_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_K;
    static constexpr int qi = QI4_K;
    static constexpr int bs = sizeof(block_q4_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
    static constexpr int bs = sizeof(block_q5_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q6_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
    static constexpr int bs = sizeof(block_q6_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XXS;
    static constexpr int qi = QI2_XXS;
    static constexpr int bs = sizeof(block_iq2_xxs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XS;
    static constexpr int qi = QI2_XS;
    static constexpr int bs = sizeof(block_iq2_xs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_S;
    static constexpr int qi = QI2_S;
    static constexpr int bs = sizeof(block_iq2_s);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_XXS;
    static constexpr int qi = QI3_XXS;
    static constexpr int bs = sizeof(block_iq3_xxs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_S;
    static constexpr int qi = QI1_S;
    static constexpr int bs = sizeof(block_iq1_s);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_M> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_M;
    static constexpr int qi = QI1_M;
    static constexpr int bs = sizeof(block_iq1_m);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_NL> {
    static constexpr int qk = QK4_NL;
    static constexpr int qr = QR4_NL;
    static constexpr int qi = QI4_NL;
    static constexpr int bs = sizeof(block_iq4_nl);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
    static constexpr int bs = sizeof(block_iq4_xs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_S;
    static constexpr int qi = QI3_S;
    static constexpr int bs = sizeof(block_iq3_s);
};

//////////////////////

struct ggml_cuda_device_info {
    int device_count;           // number of (possibly virtual) devices exposed to the rest of ggml
    int physical_device_count;  // number of physical CUDA devices actually present

    struct cuda_device_info {
        int     cc;                             // compute capability
        int     nsm;                            // number of streaming multiprocessors
        size_t  smpb;                           // max. shared memory per block
        size_t  smpbo;                          // max. shared memory per block (with opt-in)
        bool    integrated;                     // Device is integrated as opposed to discrete
        bool    vmm;                            // virtual memory support
        size_t  vmm_granularity;                // granularity of virtual memory
        size_t  total_vram;
        int     warp_size;                      // Number of threads in a dispatch
        bool    supports_cooperative_launch;    // whether cooperative launch is supported
        int     physical_device;                // backing physical CUDA device for this (virtual) device
        int     physical_share_count;           // number of (virtual) devices sharing this device's physical GPU
        int     virtual_index;                  // index of this (virtual) device among those sharing its physical GPU
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};

    std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split = {};
};

const ggml_cuda_device_info & ggml_cuda_info();

void ggml_cuda_set_device(int device);
int ggml_cuda_get_device();

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// With lazy module loading (the CUDA 12 default) the driver loads a module, one per .cu file, on a device the first time
// one of its kernels runs there: 10-45 ms per module and device, one device at a time. A kernel that only big batches
// use then stalls the first long prompt. Every .cu file registers an empty kernel here so that the modules can be
// loaded ahead of time (ggml_cuda_preload_modules).
int ggml_cuda_register_module(const void * kernel);
static __global__ void ggml_cuda_module_anchor() {}
[[maybe_unused]] static const int ggml_cuda_module_id = ggml_cuda_register_module((const void *) ggml_cuda_module_anchor);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

struct ggml_cuda_pool {
    virtual ~ggml_cuda_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_cuda_pool_alloc() = default;

    explicit ggml_cuda_pool_alloc(ggml_cuda_pool & pool) : pool(&pool) {
    }

    ggml_cuda_pool_alloc(ggml_cuda_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_cuda_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_cuda_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc(ggml_cuda_pool_alloc &&) = delete;
    ggml_cuda_pool_alloc& operator=(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc& operator=(ggml_cuda_pool_alloc &&) = delete;
};


// backend interface

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]; // events for synchronizing multiple GPUs
};


#if (defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS)) || defined(GGML_MUSA_GRAPHS)
#define USE_CUDA_GRAPH
#endif

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph() {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    bool disable_due_to_gpu_arch = false;
    bool warmup_complete = false;
    uint64_t uid = 0;
    int64_t last_used_time = 0;
    struct node_properties {
        ggml_tensor node;
        void *   node_src_data_ptrs[GGML_MAX_SRC];
        void *   node_src_extra[GGML_MAX_SRC];
        int64_t  node_src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
        size_t   node_src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];
    };
    std::vector<node_properties> node_props;
    std::vector<const void *> session_keys; // a session graph: the keys of its subgraphs, see ggml_backend_cuda_session_plan
    int session_want  = 0;                  // and the mode the last plan asked for
    // A subgraph's properties are checked both alone and as part of a session, and each path captures its own
    //   instance. props_gen counts the changes of node_props: an instance is valid only for the generation it was
    //   captured at (a session's for those of its subgraphs), so properties that another path changed and that match
    //   again do not replay a capture of different tensors.
    uint64_t props_gen    = 0;
    uint64_t captured_gen = 0;
    std::vector<uint64_t> session_gens;
    int warmup_stable = 0;                  // and how many plans in a row found its properties unchanged

    bool is_enabled() const {
        static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);
        return !(disable_due_to_gpu_arch || disable_cuda_graphs_due_to_env);
    }
#endif
};

struct ggml_cuda_concurrent_event {
    std::vector<cudaEvent_t> join_events;
    cudaEvent_t              fork_event = nullptr;

    int                                          n_streams = 0;
    std::unordered_map<const ggml_tensor *, int> stream_mapping;

    // Original order of nodes in this concurrent region (before interleaving)
    // Used to restore grouping for fusion within streams
    std::vector<const ggml_tensor *> original_order;

    const ggml_tensor * join_node;

    ggml_cuda_concurrent_event() = default;

    ggml_cuda_concurrent_event(const ggml_cuda_concurrent_event &) = delete;
    ggml_cuda_concurrent_event & operator=(const ggml_cuda_concurrent_event &) = delete;

    explicit ggml_cuda_concurrent_event(int n_streams) : n_streams(n_streams) {
        join_events.resize(n_streams);

        for (size_t i = 0; i < join_events.size(); ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&join_events[i], cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
    }

    ggml_cuda_concurrent_event(ggml_cuda_concurrent_event && other) noexcept
    : join_events(std::move(other.join_events))
    , fork_event(other.fork_event)
    , n_streams(other.n_streams)
    , stream_mapping(std::move(other.stream_mapping))
    , original_order(std::move(other.original_order))
    , join_node(other.join_node) {
        other.fork_event = nullptr;
    }

    // 1. check if any branches write to overlapping memory ranges (except the join node)
    // 2. check whether all srcs are either within the branch or outside the nodes covered by ggml_cuda_concurrent_event
    // we assume all nodes have the same buffer
    bool is_valid() const {
        std::vector<std::vector<std::pair<int64_t, int64_t>>> write_ranges;
        write_ranges.resize(n_streams);

        // get join_node's memory range to exclude from overlap checking.
        // multiple nodes can use join_node's buffer; we synchronize on the join node.
        const ggml_tensor * join_t     = join_node->view_src ? join_node->view_src : join_node;
        const int64_t       join_start = (int64_t) join_t->data;
        const int64_t       join_end   = join_start + ggml_nbytes(join_t);

        for (const auto & [tensor, stream] : stream_mapping) {
            const ggml_tensor * t = tensor->view_src ? tensor->view_src : tensor;
            const int64_t       t_start = (int64_t) t->data;
            const int64_t       t_end   = t_start + ggml_nbytes(t);

            // skip tensors that overlap with join_node's buffer.
            if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
                continue;
            }

            // concurrent streams begin from 1
            write_ranges[stream - 1].emplace_back(t_start, t_end);
        }

        for (int i = 0; i < n_streams; ++i) {
            // sorts first by start then by end of write range
            std::sort(write_ranges[i].begin(), write_ranges[i].end());
        }

        bool writes_overlap = false;
        bool dependent_srcs = false;
        for (const auto & [tensor, stream] : stream_mapping) {
            const ggml_tensor * t = tensor->view_src ? tensor->view_src : tensor;
            const int64_t       t_start = (int64_t) t->data;
            const int64_t       t_end   = t_start + ggml_nbytes(t);

            // skip tensors that overlap with join_node's buffer
            if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
                continue;
            }

            // check if this buffer's write data overlaps with another stream's
            std::pair<int64_t, int64_t> data_range = std::make_pair(t_start, t_end);
            for (int i = 0; i < n_streams; ++i) {
                if (i == stream - 1) {
                    continue;
                }
                auto it = std::lower_bound(write_ranges[i].begin(), write_ranges[i].end(), data_range);

                if (it != write_ranges[i].end()) {
                    const std::pair<int64_t, int64_t> & other = *it;

                    // std::lower_bound returns the first element where other >= data_range (lexicographically).
                    // This guarantees other.first >= data_range.first.
                    // Therefore, overlap occurs iff other.first < data_range.second
                    // (i.e., the other range starts before this range ends).
                    if (other.first < data_range.second) {
                        GGML_LOG_DEBUG("Writes overlap for %s", tensor->name);
                        writes_overlap = true;
                        break;
                    }
                }
            }

            //check if all srcs are either in branch or don't have a branch
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (!tensor->src[i]) {
                    continue;
                }

                auto it = stream_mapping.find(tensor->src[i]);

                if (it == stream_mapping.end()) {
                    continue;
                }

                if (it->second != stream) {
                    dependent_srcs = true;
                    break;
                }
            }

            if (dependent_srcs || writes_overlap) {
                break;
            }
        }

        return !writes_overlap && !dependent_srcs;
    }

    ~ggml_cuda_concurrent_event() {
        if (fork_event != nullptr) {
            CUDA_CHECK(cudaEventDestroy(fork_event));
        }
        for (cudaEvent_t e : join_events) {
            if (e != nullptr) {
                CUDA_CHECK(cudaEventDestroy(e));
            }
        }
    }
};

struct ggml_cuda_stream_context {
    std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> concurrent_events;

    void reset() {
        concurrent_events.clear();
    }
};

// A DeepSeek-V4 compressor (llama_model_deepseek4::graph::build_*compressed_kv_from_state) folded into one launch at
//   its row sum, see ggml_cuda_dsv4_compress_fold_step. A half is the values or the scores: the rows of
//   CONCAT(state, cur) (with extra, the SCALE of its row 0, appended as one more row) read through idx.
struct ggml_cuda_dsv4_compress_half {
    const ggml_tensor * state = nullptr;
    const ggml_tensor * cur   = nullptr;
    const ggml_tensor * extra = nullptr;
    const ggml_tensor * idx   = nullptr;
};
struct ggml_cuda_dsv4_compress_fold {
    int              target = -1; // the SUM_ROWS node
    std::vector<int> skip;        // the chain's nodes before it
    ggml_cuda_dsv4_compress_half kv;
    ggml_cuda_dsv4_compress_half score;
    bool    overlap  = false;     // the first r of 2r entries from the first half of a row, the rest from its second
    int64_t hd       = 0;         // pooled values per block
    int64_t r        = 0;         // rows per block
    int64_t nb       = 0;         // blocks
    float   sm_scale = 1.0f;
};

// An LL all-reduce folded into the kernels around it (see ggml_cuda_ar_fold_push in allreduce-p2p.cuh): the matmul
//   that makes this rank's partial sum writes it as the LL lines of every rank, the kernel that reads the sum polls them
struct ggml_cuda_ar_fold_args {
    uint4 *        ll[4]     = {};      // the LL staging of every rank
    uint4 *        own       = nullptr; // that of this rank, ll[rank]
    unsigned int * ll_seq    = nullptr; // this rank's sequence number per LL block
    int64_t        ll_stride = 0;       // lines per (slot, rank) region
    int            rank      = 0;
    int            n_devs    = 0;
};

// the MoE all-reduce folded into the expert down projection of one token (see mul_mat_vec_q_short): the last block of a
//   group of rows to finish sums the experts' rows (dst) and the gated shared expert as moe_weighted_reduction does and
//   pushes the sums as the LL lines of ar
struct ggml_cuda_ar_fold_moe_args {
    ggml_cuda_ar_fold_args ar;
    const float *  weights = nullptr; // [n_used]
    const float *  gate    = nullptr; // the shared expert's gate (one value), its output being the sibling's dst
    unsigned int * tickets = nullptr; // a zeroed counter per group of rows, reset by its last block
    int            n_used  = 0;
};

struct ggml_backend_cuda_context {
    int device;
    std::string name;
    cudaEvent_t copy_event = nullptr;

    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {};
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {};
    void * cublas_workspaces[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {};
    size_t cublas_workspace_sizes[GGML_CUDA_MAX_DEVICES] = {};

    int curr_stream_no = 0;

#ifdef USE_CUDA_GRAPH
    // Map from first_node_ptr to cuda_graph - allows multiple graphs per context
    // when the computation is split across CPU/GPU (e.g., with --n-cpu-moe)
    std::unordered_map<const void *, std::unique_ptr<ggml_cuda_graph>> cuda_graphs;

    int64_t last_graph_eviction_sweep = 0;

    ggml_cuda_graph * cuda_graph(const void * first_node_ptr) {
        const int64_t time_now = ggml_time_us();

        // sweep every 5s, evicting cuda graphs unused for >=10 min: a server between requests a person makes would
        //   otherwise capture its decode graphs again for every request (~50 ms at 4 GPUs); the keys are the first
        //   nodes of graphs built over the same tensors each time, so the map does not grow with that
        if (time_now - last_graph_eviction_sweep >= 5'000'000) {
            last_graph_eviction_sweep = time_now;
            for (auto it = cuda_graphs.begin(); it != cuda_graphs.end(); ) {
                if (time_now - it->second->last_used_time >= 600'000'000) {
                    it = cuda_graphs.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto it = cuda_graphs.find(first_node_ptr);
        if (it == cuda_graphs.end()) {
            it = cuda_graphs.emplace(first_node_ptr, std::make_unique<ggml_cuda_graph>()).first;
        }
        it->second->last_used_time = time_now;
        return it->second.get();
    }

    // Check if any CUDA graph is enabled for this context (used by kernels that need to know
    // if graphs are in use without having access to the specific graph key)
    bool any_cuda_graph_enabled() const {
        for (const auto & [key, graph] : cuda_graphs) {
            if (graph && graph->is_enabled()) {
                return true;
            }
        }
        return false;
    }

    // Check if any CUDA graph has an instance for this context
    bool any_cuda_graph_has_instance() const {
        for (const auto & [key, graph] : cuda_graphs) {
            if (graph && graph->instance != nullptr) {
                return true;
            }
        }
        return false;
    }
#endif // USE_CUDA_GRAPH

    explicit ggml_backend_cuda_context(int device) :
        device(device),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ggml_cuda_stream_context concurrent_stream_context;

    // see ggml_backend_cuda_session_plan; the order is that of ggml-backend-meta's session modes
    enum session_mode_t { SESSION_OFF = -1, SESSION_EAGER = 0, SESSION_CAPTURE = 1, SESSION_REPLAY = 2 };
    session_mode_t session_mode = SESSION_OFF;
    const void *   session_key  = nullptr;

    // the MoE weighted sum (plus a gated shared expert) that ends a subgraph of a session, left to the all-reduce of its
    //   output right after it, which computes it as its input; see ggml_cuda_try_fuse_moe_down_shexp
    struct session_reduction_t {
        const ggml_tensor * experts = nullptr; // [n_embd, n_used], one token
        const ggml_tensor * weights = nullptr; // [n_used]
        const ggml_tensor * add_g   = nullptr; // the shared expert's gate (one value)
        ggml_tensor *       dst     = nullptr; // the sum
        std::unique_ptr<ggml_cuda_pool_alloc<float>> add_x; // the shared expert's down projection
    } session_red;

    // an all-reduce of a session folded into the kernels around it: next is the node it reduces, announced before its
    //   subgraph runs (ggml_backend_cuda_session_allreduce_next); a matmul writing it pushes its rows as the LL lines of
    //   every rank (pushed), the all-reduce then launches nothing and recv is left for the next subgraph, whose residual
    //   add and norm poll the lines, else a kernel that only receives (ggml_cuda_ar_fold_consume)
    struct ar_fold_call_t {
        void *                 comm_ctx = nullptr;
        ggml_tensor *          tensors[GGML_CUDA_MAX_DEVICES] = {};
        size_t                 rank = 0;
        ggml_cuda_ar_fold_args args;
    };
    struct ar_fold_t {
        ar_fold_call_t      next_call;
        const ggml_tensor * next   = nullptr;
        bool                pushed = false;
        ar_fold_call_t      recv_call;
        bool                recv   = false;
        bool                hc     = false; // the session has hyper-connection posts, see ggml_backend_cuda_session_begin
    } ar_fold;

    // the CONCAT of a build_conv_state whose chain the qkv projection already ran (ggml_cuda_gdn_conv_fold), and the
    //   nodes to skip from it, up to its last slot CPY
    const ggml_tensor * conv_folded_concat = nullptr;
    int                 conv_folded_skip   = 0;

    // the first node of a topk_moe that a router's launch already ran (ggml_cuda_topk_moe_fold), and its other nodes
    const ggml_tensor * topk_folded_node = nullptr;
    int                 topk_folded_skip = 0;

    ~ggml_backend_cuda_context();

    cudaStream_t stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
        }
        return streams[device][stream];
    }

    cudaStream_t stream() { return stream(device, curr_stream_no); }

    ggml_cuda_stream_context & stream_context() { return concurrent_stream_context; }

    cublasHandle_t cublas_handle() {
        if (cublas_handles[device][curr_stream_no] == nullptr) {
            ggml_cuda_set_device(device);
            CUBLAS_CHECK(cublasCreate(&cublas_handles[device][curr_stream_no]));
            CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device][curr_stream_no], CUBLAS_TF32_TENSOR_OP_MATH));
            CUBLAS_CHECK(cublasSetStream(cublas_handles[device][curr_stream_no], stream()));
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (CUBLAS_VER_MAJOR > 11 || (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR >= 2))
            if (cublas_workspace_sizes[device] == 0) {
                const int cc = ggml_cuda_info().devices[device].cc;
                cublas_workspace_sizes[device] = (cc >= GGML_CUDA_CC_HOPPER) ? 32 * 1024 * 1024 : 4 * 1024 * 1024;
            }
            CUDA_CHECK(cudaMalloc(&cublas_workspaces[device][curr_stream_no], cublas_workspace_sizes[device]));
            CUBLAS_CHECK(cublasSetWorkspace(cublas_handles[device][curr_stream_no], cublas_workspaces[device][curr_stream_no], cublas_workspace_sizes[device]));
#endif
        }
        return cublas_handles[device][curr_stream_no];
    }

    // two flags for a kernel whose blocks wait on its first block (see rms_norm_emit_q8_1), then the ticket of the
    //   topk_moe in a router's launch (ggml_cuda_topk_moe_fold), all zero between launches; then the epoch and the
    //   tagged slots of a row split over blocks (row_split_reduce in norm.cu)
    unsigned int * block_flags[GGML_CUDA_MAX_DEVICES] = { nullptr };

    // the tickets of ggml_cuda_ar_fold_moe_args, zero between launches
    unsigned int * ar_tickets[GGML_CUDA_MAX_DEVICES] = { nullptr };

    unsigned int * ar_tickets_get() {
        if (ar_tickets[device] == nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaMalloc(&ar_tickets[device], 4096*sizeof(unsigned int)));
            CUDA_CHECK(cudaMemset(ar_tickets[device], 0, 4096*sizeof(unsigned int)));
        }
        return ar_tickets[device];
    }

    unsigned int * block_flags_get() {
        if (block_flags[device] == nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaMalloc(&block_flags[device], 512*sizeof(unsigned int)));
            CUDA_CHECK(cudaMemset(block_flags[device], 0, 512*sizeof(unsigned int)));
        }
        return block_flags[device];
    }

    // pool
    std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS];

    static std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int device, int stream_no);

    ggml_cuda_pool & pool(int device) {
        if (pools[device][curr_stream_no] == nullptr) {
            pools[device][curr_stream_no] = new_pool_for_device(device, curr_stream_no);
        }
        return *pools[device][curr_stream_no];
    }

    ggml_cuda_pool & pool() {
        return pool(device);
    }

    // q8_1-quantized copies of matmul inputs, reused by later matmuls that read the same tensor.
    // One norm output typically feeds several projections, which would otherwise quantize it once
    // each. Entries live for one graph_compute and own their pool allocation until it ends.
    struct q8_1_input {
        const ggml_tensor * src1 = nullptr;
        int64_t ne[GGML_MAX_DIMS] = {0};
        int64_t nb[GGML_MAX_DIMS] = {0};
        int64_t ne_padded = 0;
        int stream_no = -1; // reuse only on the producing stream, nothing orders the other ones
        std::unique_ptr<ggml_cuda_pool_alloc<char>> buf;
    };

    std::unordered_map<const void *, q8_1_input> q8_1_inputs;

    // a node is about to overwrite this buffer, so any quantized copy of it is stale
    void q8_1_input_invalidate(const ggml_tensor * node) {
        if (q8_1_inputs.empty()) {
            return;
        }
        q8_1_inputs.erase(node->data);
        // a node that writes in place into part of a live tensor (a view at an offset) changes bytes of a copy that
        //   is keyed at another address; other nodes only reuse memory of tensors nothing reads again
        if (node->view_src == nullptr) {
            return;
        }
        const char * a0 = (const char *) node->data;
        const char * a1 = a0 + ggml_nbytes(node);
        for (auto it = q8_1_inputs.begin(); it != q8_1_inputs.end();) {
            const char * b0 = (const char *) it->first;
            int64_t nbytes = it->second.nb[0];
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                nbytes += (it->second.ne[i] - 1)*it->second.nb[i];
            }
            it = a0 < b0 + nbytes && b0 < a1 ? q8_1_inputs.erase(it) : std::next(it);
        }
    }

    void q8_1_input_reset() {
        q8_1_inputs.clear();
        q8_1_pending.clear();
        q8_1_emit    = nullptr;
        q8_1_emit_as = nullptr;
    }

    // The remapped ids of expert-cache pools gathered ahead, with the first MUL_MAT_ID of the same ids in the subgraph
    //   (see ggml_cuda_try_gather_expert_group); ggml_cuda_mul_mat_id takes them instead of gathering.
    struct expert_cache_gathered {
        const ggml_tensor * pool = nullptr;
        const ggml_tensor * ids  = nullptr;
        ggml_tensor         remap;
    };
    std::vector<expert_cache_gathered> expert_cache_ahead;

    // A one-token gate/up GLU deferred to the MUL_MAT_ID gate/up GLU of the same input that follows it (node target),
    //   which runs it as one more channel; see ggml_cuda_try_defer_glu. Launched on its own if that does not happen.
    struct deferred_glu {
        const ggml_tensor * up     = nullptr;
        const ggml_tensor * gate   = nullptr;
        ggml_tensor *       glu    = nullptr;
        int                 target = -1;
    } glu_deferred;

    ggml_cuda_dsv4_compress_fold dsv4_fold;

    // the sigmoid gate after a FLASH_ATTN_EXT, sigmoid(gate) * attn in the CONT, UNARY and MUL after it (qwen35): the
    //   combine of a split KV applies it, writes the MUL output and sets done (see ggml_cuda_try_fuse_fattn_gate)
    struct fattn_gate_args {
        const ggml_tensor * cont = nullptr;
        ggml_tensor *       mul  = nullptr;
        bool                done = false;
    } fattn_gate;

    // A kernel can write the q8_1 copy of its output for the MMVQ matmul that reads it next, which then
    //   skips its quantize launch. ggml_cuda_q8_1_request names the output and the tensor the matmul reads it as (the
    //   output or a view of all of it); the launcher takes the buffer with q8_1_emit_alloc. The copy waits in
    //   q8_1_pending until the evaluate loop has invalidated the copies of the nodes the kernel replaced.
    const ggml_tensor * q8_1_emit    = nullptr;
    const ggml_tensor * q8_1_emit_as = nullptr;
    std::vector<std::pair<const void *, q8_1_input>> q8_1_pending;

    // the buffer for the q8_1 copy of t if t is the named output, else nullptr; ne_padded is the padded row length, the
    //   rows follow each other at that stride. A kernel that writes only one row passes max_rows 1.
    void * q8_1_emit_alloc(const ggml_tensor * t, int64_t & ne_padded, const int64_t max_rows = 1) {
        if (q8_1_emit == nullptr || q8_1_emit != t) {
            return nullptr;
        }
        const ggml_tensor * y = q8_1_emit_as;
        q8_1_emit    = nullptr;
        q8_1_emit_as = nullptr;
        if (ggml_nrows(y) > max_rows) {
            return nullptr;
        }
        ne_padded = GGML_PAD(y->ne[0], MATRIX_ROW_PADDING);
        q8_1_input entry;
        entry.buf       = std::make_unique<ggml_cuda_pool_alloc<char>>(pool(), ggml_nrows(y)*ne_padded*sizeof(block_q8_1)/QK8_1);
        entry.src1      = y->view_src ? y->view_src : y;
        entry.ne_padded = ne_padded;
        entry.stream_no = curr_stream_no;
        std::copy_n(y->ne, GGML_MAX_DIMS, entry.ne);
        std::copy_n(y->nb, GGML_MAX_DIMS, entry.nb);
        void * ptr = entry.buf->get();
        q8_1_pending.emplace_back(y->data, std::move(entry));
        return ptr;
    }

    void q8_1_commit_pending() {
        for (auto & p : q8_1_pending) {
            q8_1_inputs[p.first] = std::move(p.second);
        }
        q8_1_pending.clear();
        q8_1_emit    = nullptr;
        q8_1_emit_as = nullptr;
    }

    // The f16 (or bf16) copy of an f32 matmul input that cuBLAS reads, kept for the next matmul of the same input: the
    // q/k/v projections of one norm convert it once instead of three times. One entry, replaced by the next conversion.
    // Its pool allocation outlives the matmul, so it is only kept where the pool frees in any order (not vmm).
    struct cublas_input {
        const ggml_tensor * owner = nullptr; // the tensor that owns the memory, the source of a view
        const void * data = nullptr;
        size_t nbytes = 0; // of the f32 input
        ggml_type type = GGML_TYPE_COUNT;
        int64_t ne[GGML_MAX_DIMS] = {0};
        size_t  nb[GGML_MAX_DIMS] = {0};
        int stream_no = -1;
        int64_t s11 = 0; // the strides of the copy, in elements
        int64_t s12 = 0;
        int64_t s13 = 0;
        bool cont_2 = false;
        std::unique_ptr<ggml_cuda_pool_alloc<char>> buf;
    };

    cublas_input cublas_src1;

    // The expert lists and the quantized activations of an MMQ MUL_MAT_ID, kept for the next one that reads the same
    // activations with the same ids: the up projection of an expert FFN after its gate. One entry, replaced by the next,
    // on a pool that frees in any order, as above.
    struct mmq_ids_input {
        const ggml_tensor * src1_owner = nullptr;
        const void * src1_data   = nullptr;
        size_t       src1_nbytes = 0;
        int64_t      src1_ne[GGML_MAX_DIMS] = {0};
        size_t       src1_nb[GGML_MAX_DIMS] = {0};
        const void * ids_data    = nullptr;
        size_t       ids_nbytes  = 0;
        int64_t      ids_ne[GGML_MAX_DIMS] = {0};
        size_t       ids_nb[GGML_MAX_DIMS] = {0};
        ggml_type    type      = GGML_TYPE_COUNT;
        int          prec      = -1;
        int64_t      n_experts = 0;
        size_t       nbytes    = 0; // of the quantized activations, with the tile padding of the type
        int          stream_no = -1;
        std::unique_ptr<ggml_cuda_pool_alloc<char>> ids_src1, ids_dst, expert_bounds, q8_1, scale;

        void reset() {
            ids_src1.reset();
            ids_dst.reset();
            expert_bounds.reset();
            q8_1.reset();
            scale.reset();
        }
    };

    mmq_ids_input mmq_ids;

    // The KV ranges that the vec and tile FA kernels read from a mask scan (see flash_attn_mask_to_KV_range), kept for
    // the next FA with the same mask and Q tiling: the layers of a model share one mask per cache, so it is scanned
    // once per graph instead of once per layer. A few entries, for models that interleave sliding-window and global
    // layers, replaced in turn; on a pool that frees in any order, as above.
    struct fattn_range_input {
        const void * data   = nullptr;
        size_t       nbytes = 0;
        int64_t      ne[GGML_MAX_DIMS] = {0};
        size_t       nb[GGML_MAX_DIMS] = {0};
        int64_t      n_kv      = 0;
        int          ncols1    = 0;
        int          ntiles_x  = 0;
        int64_t      n_seq     = 0;
        int          stream_no = -1;
        std::unique_ptr<ggml_cuda_pool_alloc<int>> buf;
    };

    static constexpr int FATTN_RANGE_ENTRIES = 4;
    fattn_range_input fattn_range[FATTN_RANGE_ENTRIES];
    int fattn_range_next = 0;

    // a node is about to write its memory
    void cublas_input_invalidate(const ggml_tensor * node) {
        const auto overlaps = [node](const void * data, const size_t nbytes) {
            const char * a = (const char *) node->data;
            const char * b = (const char *) data;
            return a < b + nbytes && b < a + ggml_nbytes(node);
        };
        if (cublas_src1.buf && overlaps(cublas_src1.data, cublas_src1.nbytes)) {
            cublas_src1.buf.reset();
        }
        if (mmq_ids.q8_1 && (overlaps(mmq_ids.src1_data, mmq_ids.src1_nbytes) || overlaps(mmq_ids.ids_data, mmq_ids.ids_nbytes))) {
            mmq_ids.reset();
        }
        for (fattn_range_input & e : fattn_range) {
            if (e.buf && overlaps(e.data, e.nbytes)) {
                e.buf.reset();
            }
        }
    }

    void cublas_input_reset() {
        cublas_src1.buf.reset();
        mmq_ids.reset();
        for (fattn_range_input & e : fattn_range) {
            e.buf.reset();
        }
    }
};

struct ggml_cuda_mm_fusion_args_host {
    const ggml_tensor * x_bias = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * gate_bias = nullptr;
    const ggml_tensor * x_scale = nullptr;
    const ggml_tensor * gate_scale = nullptr;
    ggml_glu_op glu_op;
    float glu_limit = 0.0f;
    // gated delta net gate and beta instead of a GLU: dst = softplus(x*y + gdn_dt) * gdn_a, gdn_beta = sigmoid(gate*y)
    // with gdn_x (mmvq, one token): these are the rows of gdn_x into gdn_dst, after the rows of src0 and sib
    const ggml_tensor * gdn_dt = nullptr;
    const ggml_tensor * gdn_a = nullptr;
    ggml_tensor * gdn_beta = nullptr;
    const ggml_tensor * gdn_x = nullptr;
    ggml_tensor * gdn_dst = nullptr;
    // mmvq with gdn_x, one token: the d_conv 4 convolution of build_conv_state in the epilogue of the rows of src0, each
    //   row one channel: conv_dst = silu(conv(window)), the window being the 3 columns of the cache row conv_states (at
    //   conv_ids[0]) and the row's value; the windows from column s on go to conv_slot[s], see ggml_cuda_gdn_conv_fold
    const float * conv_states = nullptr;
    const int32_t * conv_ids = nullptr;
    int64_t conv_row_stride = 0; // floats
    const float * conv_w = nullptr;
    int64_t conv_w_stride = 0; // floats
    float * conv_dst = nullptr;
    float * conv_slot[2] = { nullptr, nullptr };
    // mmvf, f32, one token: the topk_moe of the expert logits in dst, run by the block of the launch that ends last
    //   (ggml_cuda_topk_moe_fold); topk_ticket is a zeroed counter that block resets
    float * topk_weights = nullptr;
    int32_t * topk_ids = nullptr;
    unsigned int * topk_ticket = nullptr;
    int topk_n_expert = 0;
    int topk_n_used = 0;
    float topk_clamp = 0.0f;
    float topk_scale = 1.0f;
    bool topk_sigmoid = false;
    bool topk_sqrt_softplus = false;
    bool topk_with_norm = false;
    bool topk_delayed_softmax = false;
    // a sibling matrix of one row with the same input, computed as one more row: sib_dst = sib*y (f32, one token)
    // mmvq without ids and nothing else fused: a sibling matrix of the same type and K with the same input, its rows
    //   computed after those of src0 in the same launch: sib_dst = sib*y (one token)
    // mmvq with ids and a gate: a sibling pair of the same shape as one expert, computed as one more channel past the
    //   used experts: sib_dst = glu(sib_gate*y, sib*y) (one token)
    // mmvq with ids, short q8_0 rows: a sibling of the same shape as one expert with its own f32 input, computed as one
    //   more channel past the used experts: sib_out = sib*sib_y (one token)
    const ggml_tensor * sib = nullptr;
    const ggml_tensor * sib_gate = nullptr;
    ggml_tensor * sib_dst = nullptr;
    const ggml_tensor * sib_y = nullptr;
    float * sib_out = nullptr;
    // mmvq with ids, short q8_0 rows and sib_y: the all-reduce of the experts' weighted sum (weights [n_used]) plus the
    //   sibling gated by sigmoid(ar_gate) folded into the launch, see ggml_cuda_ar_fold_moe_args
    const ggml_tensor * ar_weights = nullptr;
    const ggml_tensor * ar_gate    = nullptr;
    // mmvq without ids, with sib: a second sibling of the same kind, its rows after those of sib (sib2_dst = sib2*y)
    const ggml_tensor * sib2 = nullptr;
    ggml_tensor * sib2_dst = nullptr;
    // mmvf: sib (and sib2) are sibling matrices of the same type and K whose rows follow those of src0, as for mmvq
    //   without ids; without it a sib of mmvf is the f32 single row above
    bool sib_rows = false;
    // mmvf with sib_rows and sib2: a third sibling, its rows after those of sib2 (in the gdn_x rows of the device args)
    const ggml_tensor * sib3 = nullptr;
    ggml_tensor * sib3_dst = nullptr;
    // one value per token gating the product before the bias: dst = x_bias + sigmoid(x_sigmoid)*(x*y), one token
    const ggml_tensor * x_sigmoid = nullptr;
    // mmvq with ids, a gate and nothing else: the gate's own ids, of the same layout (pools of cached experts that each
    //   hold the used experts in their own slots)
    const ggml_tensor * gate_ids = nullptr;
    const ggml_tensor * shared_up = nullptr;
    const ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_dst = nullptr;
};
struct ggml_cuda_mm_fusion_args_device {
    const void * x_bias = nullptr;
    const void * gate = nullptr;
    const void * gate_bias = nullptr;
    const void * x_scale = nullptr;
    const void * gate_scale = nullptr;
    ggml_glu_op glu_op;
    float glu_limit = 0.0f;
    const float * gdn_dt = nullptr;
    const float * gdn_a = nullptr;
    float * gdn_beta = nullptr;
    const void * sib_x = nullptr;
    const void * sib_gate = nullptr;
    float * sib_dst = nullptr;
    const float * sib_y = nullptr;
    uint32_t sib_block0 = 0; // mmvq without ids: the first block of the sibling's rows, and their count
    uint32_t sib_nrows  = 0;
    const void * sib2_x = nullptr; // and of the second sibling's rows
    float * sib2_dst = nullptr;
    uint32_t sib2_block0 = 0;
    uint32_t sib2_nrows  = 0;
    const void * gdn_x = nullptr; // and of the gated delta net gate's rows
    float * gdn_dst = nullptr;
    uint32_t gdn_block0 = 0;
    uint32_t gdn_nrows  = 0;
    const float * x_sigmoid = nullptr;
    const int32_t * gate_ids = nullptr;
    const void * shared_up = nullptr;
    const void * shared_gate = nullptr;
    float * shared_dst = nullptr;
    uint32_t shared_stride_col_dst = 0;
};

// The convolution in the epilogue of a one-token mmvq (ggml_cuda_gdn_conv_fold) and the topk_moe in the last block of a
//   router's mmvf launch (ggml_cuda_topk_moe_fold) are parameters of their own kernels only: every byte of
//   ggml_cuda_mm_fusion_args_device is one of every mmvq and mmvf launch (120 bytes more cost Goetia 1.3% of its tg).
struct ggml_cuda_mmvq_conv_args {
    const float *   states     = nullptr; // see ggml_cuda_mm_fusion_args_host
    const int32_t * ids        = nullptr;
    int64_t         row_stride = 0;
    const float *   w          = nullptr;
    int64_t         w_stride   = 0;
    float *         dst        = nullptr;
    float *         slot0      = nullptr;
    float *         slot1      = nullptr;
};
struct ggml_cuda_mmvf_topk_args {
    const float *  logits          = nullptr; // see ggml_cuda_mm_fusion_args_host
    float *        weights         = nullptr;
    int32_t *      ids             = nullptr;
    unsigned int * ticket          = nullptr;
    int            n_expert        = 0;
    int            n_used          = 0;
    float          clamp           = 0.0f;
    float          scale           = 1.0f;
    bool           sigmoid         = false;
    bool           sqrt_softplus   = false;
    bool           with_norm       = false;
    bool           delayed_softmax = false;
};
// in place of either in the other kernels
struct ggml_cuda_no_args {
    ggml_cuda_no_args() = default;
    template <typename T>
    ggml_cuda_no_args(const T &) {}
};
// the device arguments as the host code of mmvq and mmvf carries them to the launch
struct ggml_cuda_mm_fusion_args_launch : ggml_cuda_mm_fusion_args_device {
    ggml_cuda_mmvq_conv_args conv;
    ggml_cuda_mmvf_topk_args topk;
    // with ar_pushed: a launch that can push its rows as the LL lines of ar does so and sets *ar_pushed; with ar_moe the
    //   expert down projection of a MoE pushes the sum of its experts instead
    ggml_cuda_ar_fold_args   ar;
    bool *                   ar_pushed = nullptr;
    bool                     ar_moe    = false;
    const float *            ar_weights = nullptr;
    const float *            ar_gate    = nullptr;
    unsigned int *           ar_tickets = nullptr;
    int                      ar_n_used  = 0;
};

struct ggml_cuda_kernel_launch_params {
    dim3 block_nums;
    dim3 block_dims;
    size_t shmem;
    cudaStream_t stream;

    // size_t shmem
    ggml_cuda_kernel_launch_params(const dim3& block_nums_, const dim3& block_dims_, const size_t shmem_, const cudaStream_t stream_)
        : block_nums(block_nums_), block_dims(block_dims_), shmem(shmem_), stream(stream_) {}

    // Some call sites pass ints instead of the required size_t. This 2nd constructor casts int->size_t to avoid these -Wnarrowing warnings.
    ggml_cuda_kernel_launch_params(const dim3& block_nums_, const dim3& block_dims_, const int shmem_, const cudaStream_t stream_)
        : block_nums(block_nums_), block_dims(block_dims_), shmem((size_t)shmem_), stream(stream_) {}
};

#if defined(GGML_CUDA_USE_PDL)
struct ggml_cuda_pdl_config {
    cudaLaunchAttribute attr;
    cudaLaunchConfig_t  cfg;

    ggml_cuda_pdl_config(const ggml_cuda_kernel_launch_params & params) {
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;

        cfg = {};
        cfg.gridDim          = params.block_nums;
        cfg.blockDim         = params.block_dims;
        cfg.dynamicSmemBytes = params.shmem;
        cfg.stream           = params.stream;
        cfg.attrs            = &attr;
        cfg.numAttrs         = 1;
    }

    // Delete due to &attr
    ggml_cuda_pdl_config(const ggml_cuda_pdl_config&) = delete;
    ggml_cuda_pdl_config& operator=(const ggml_cuda_pdl_config&) = delete;
    ggml_cuda_pdl_config& operator=(ggml_cuda_pdl_config&&) = delete;

};

static bool ggml_cuda_kernel_can_use_pdl(const void * kernel) {
    const int device = ggml_cuda_get_device();

    struct cache_key {
        int          device;
        const void * kernel;

        bool operator==(const cache_key & other) const { return device == other.device && kernel == other.kernel; }
    };

    struct cache_key_hash {
        // MurmurHash3 mixing function for better hash distribution (vs. just std::hash which in some implementations simply returns the identity)
        static size_t hash_mix(size_t x) {
            std::uint64_t       y = x;
            const std::uint64_t m = 0xe9846af9b1a615d;

            y ^= y >> 32;
            y *= m;
            y ^= y >> 32;
            y *= m;
            y ^= y >> 28;

            return static_cast<size_t>(y);
        }

        size_t operator()(const cache_key & key) const {
            // Use a nonzero seed to avoid mapping all-zero keys to zero
            size_t h = 42;
            h        = hash_mix(h + key.device);
            h        = hash_mix(h + reinterpret_cast<size_t>(key.kernel));
            return h;
        }
    };

    static std::mutex                                          cache_mutex;
    static std::unordered_map<cache_key, bool, cache_key_hash> cache;

    const cache_key             key = { device, kernel };
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto                  it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    cudaFuncAttributes attr = {};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));

    // PDL device-side primitives are emitted only for PTX versions >= 90.
    // We have to guard on a loaded kernel's PTX version so a kernel forward-JIT'ed
    // from pre-Hopper PTX to a Hopper-or-newer GPU does not opt into PDL.
    const bool can_use_pdl = attr.ptxVersion >= 90;
    cache.emplace(key, can_use_pdl);
    return can_use_pdl;
}

#endif //defined(GGML_CUDA_USE_PDL)

// PDL and __restrict__ need to be mutually exclusive, see https://github.com/ggml-org/llama.cpp/pull/24030
# if (defined(GGML_CUDA_USE_PDL) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER)
# define GGML_CUDA_RESTRICT
# else
# define GGML_CUDA_RESTRICT __restrict__
# endif // defined(GGML_CUDA_USE_PDL) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER

template<typename Kernel, typename... Args>
static __inline__ void ggml_cuda_kernel_launch(Kernel kernel, const ggml_cuda_kernel_launch_params & launch_params, Args&&... args) {
#if defined(GGML_CUDA_USE_PDL)

    static const bool env_pdl_enabled = []() {
        const char * env = getenv("GGML_CUDA_PDL");
        return env == nullptr || std::atoi(env) != 0;
    }();

    if (env_pdl_enabled && ggml_cuda_kernel_can_use_pdl(reinterpret_cast<const void *>(kernel))) {
        auto pdl_cfg = ggml_cuda_pdl_config(launch_params);

        CUDA_CHECK(cudaLaunchKernelEx(&pdl_cfg.cfg, kernel, std::forward<Args>(args)... ));
        return;
    }
#endif //defined(GGML_CUDA_USE_PDL)

    kernel<<<launch_params.block_nums, launch_params.block_dims, launch_params.shmem, launch_params.stream>>>(std::forward<Args>(args)... );
    CUDA_CHECK(cudaGetLastError());
}
