#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif // GGML_USE_NCCL
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/allreduce-p2p.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/expert-cache.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/kq-mask.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/dsv4-compress.cuh"
#include "ggml-cuda/spin.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    // the log callback is async and GGML_ABORT kills the process before it flushes, so the message
    // is otherwise lost; write it synchronously first
    fprintf(stderr, "%s error: %s\n  current device: %d, in function %s at %s:%d\n  %s\n",
        GGML_CUDA_NAME, msg, id, func, file, line, stmt);
    fflush(stderr);
    // a kernel that trapped after a spin wait passed its limit left a record of it
    ggml_cuda_spin_print();

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static bool ggml_cuda_env_flag(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && std::atoi(env) != 0;
}

static int ggml_cuda_env_int(const char * name, int default_value) {
    const char * env = getenv(name);
    return env != nullptr ? std::atoi(env) : default_value;
}

static bool ggml_cuda_unified_memory_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
    return enabled;
}

static bool ggml_cuda_pinned_memory_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_NO_PINNED");
    return disabled;
}

static bool ggml_cuda_register_host_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_REGISTER_HOST");
    return enabled;
}

static bool ggml_cuda_fusion_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION");
    return disabled;
}

// peer access between every pair of devices at init: on by default, GGML_CUDA_P2P=0 turns it off
static bool ggml_cuda_p2p_enabled() {
    static const bool enabled = ggml_cuda_env_int("GGML_CUDA_P2P", 1) != 0;
    return enabled;
}

static int ggml_cuda_op_offload_min_batch() {
    static const int min_batch = ggml_cuda_env_int("GGML_OP_OFFLOAD_MIN_BATCH", 32);
    return min_batch;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (ggml_cuda_unified_memory_enabled()) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (ggml_cuda_p2p_enabled()) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// one kernel per module (common.cuh), registered during static initialization
static std::vector<const void *> & ggml_cuda_module_anchors() {
    static std::vector<const void *> anchors;
    return anchors;
}

int ggml_cuda_register_module(const void * kernel) {
    std::vector<const void *> & anchors = ggml_cuda_module_anchors();
    anchors.push_back(kernel);
    return (int) anchors.size();
}

// Load every module of the backend on a device in a background thread, started at the first buffer or backend of the
// device, i.e. while the model loads. Otherwise the first launch from a module that only big batches use waits for the
// module to load with the GPUs idle: 25-50 ms per module in the first long prompt on 4 GPUs. Loading all modules takes
// ~0.85 s on the first device (the fatbins are decompressed once) and ~0.2 s on each other one. The driver does one
// module load at a time and the weight upload waits behind them, so the devices go one after another with a pause after
// each module: without it the model load took ~1 s longer, started later (backend creation) it slowed the first requests.
// The kernel code is still copied to the device at its first use; the modules add a few MiB of data and can raise the
// local memory reserve to the largest stack of any kernel.
//
// Off by default, GGML_CUDA_PRELOAD_MODULES=1 turns it on: Qwen3.6 MoE decoded 0.2% slower with it on 4 V100s (13 of
// 13 rounds), Ornith not. The preload puts that reserve and the module data between the weights and the KV and compute
// buffers; a 40 MiB allocation at the same place without the preload cost part of it, CUDA_MODULE_LOADING=EAGER none.
//
// The preload ends when the backend starts to compute (ggml_cuda_preload_modules_close): a module load can wait for the
// device to be idle, and a kernel that spins until the other devices reach it (P2P all-reduce, NCCL) then never ends
// while the launch of its peer waits behind the load. A 4-device test model hung like that in its first all-reduce.
struct ggml_cuda_module_preloader {
    std::mutex               mutex;
    std::mutex               serial; // one device at a time; held across each module load
    std::vector<std::thread> threads;
    std::atomic<bool>        stop{false};
    bool                     started[GGML_CUDA_MAX_DEVICES] = {};

    ~ggml_cuda_module_preloader() {
        stop.store(true);
        for (std::thread & t : threads) {
            t.join();
        }
    }
};

// constructed at the first preload, after the CUDA runtime, so it is destroyed before it
static ggml_cuda_module_preloader & ggml_cuda_module_preloader_get() {
    static ggml_cuda_module_preloader p;
    return p;
}

static std::atomic<bool> g_cuda_preload_closed{false};  // the backend computed: no preload starts any more
static std::atomic<bool> g_cuda_preload_pending{false}; // a preload may run

static void ggml_cuda_preload_modules(int device) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    static const bool enabled = [] {
        const char * env  = getenv("GGML_CUDA_PRELOAD_MODULES");
        const char * mode = getenv("CUDA_MODULE_LOADING");
        // eager loading loaded everything when the context was created
        return env != nullptr && atoi(env) != 0 && (mode == nullptr || strcmp(mode, "EAGER") != 0);
    }();
    if (!enabled || g_cuda_preload_closed.load()) {
        return;
    }

    ggml_cuda_module_preloader & p = ggml_cuda_module_preloader_get();
    const int physical_device = ggml_cuda_get_physical_device(device);

    std::lock_guard<std::mutex> lock(p.mutex);
    if (p.stop.load() || p.started[physical_device]) {
        return;
    }
    // pending before closed is read again, and the other way round in ggml_cuda_preload_modules_close: one of them sees
    // the other's store
    g_cuda_preload_pending.store(true);
    if (g_cuda_preload_closed.load()) {
        return;
    }
    p.started[physical_device] = true;
    p.threads.emplace_back([physical_device] {
        ggml_cuda_module_preloader & p = ggml_cuda_module_preloader_get();
        const int64_t t_start_us = ggml_time_us();
        size_t n_loaded = 0;
        if (cudaSetDevice(physical_device) == cudaSuccess) {
            for (const void * kernel : ggml_cuda_module_anchors()) {
                std::lock_guard<std::mutex> serial(p.serial);
                if (p.stop.load()) {
                    return;
                }
                // loads the module of the kernel on this device
                cudaFuncAttributes attr;
                n_loaded += cudaFuncGetAttributes(&attr, kernel) == cudaSuccess;
                // the pause lets the weight upload in
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        (void) cudaGetLastError();
        GGML_LOG_DEBUG("ggml_cuda_preload_modules: loaded %zu of %zu modules on device %d in %.1f ms\n",
            n_loaded, ggml_cuda_module_anchors().size(), physical_device, (ggml_time_us() - t_start_us) / 1000.0);
    });
#else
    GGML_UNUSED(device);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

// Called before the backend computes: stops the preload and waits for the module load in flight.
static void ggml_cuda_preload_modules_close() {
    // not relaxed: a closed seen here must come with the pending of a preload the closing thread is still stopping
    if (g_cuda_preload_closed.load() && !g_cuda_preload_pending.load()) {
        return;
    }
    g_cuda_preload_closed.store(true);
    if (!g_cuda_preload_pending.load()) {
        return;
    }
    ggml_cuda_module_preloader & p = ggml_cuda_module_preloader_get();
    {
        std::lock_guard<std::mutex> lock(p.mutex);
        p.stop.store(true);
    }
    // the threads check stop with serial held, so none loads a module after this
    { std::lock_guard<std::mutex> serial(p.serial); }
    g_cuda_preload_pending.store(false);
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    ggml_backend_cuda_context & ctx;
    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    ggml_cuda_pool_leg(ggml_backend_cuda_context & ctx, int device) :
        ctx(ctx),
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    // captured CUDA graphs can still use freed buffers, so they must be captured again
    void invalidate_cuda_graphs() {
#ifdef USE_CUDA_GRAPH
        int n_captured = 0;
        for (const auto & [key, graph] : ctx.cuda_graphs) {
            n_captured += graph->instance != nullptr;
        }
        if (n_captured > 0) {
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: memory freed, %d captured CUDA graphs must be captured again\n", device, n_captured);
        }
        ctx.cuda_graphs.clear();
#endif // USE_CUDA_GRAPH
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            invalidate_cuda_graphs();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
        invalidate_cuda_graphs();
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = ggml_cuda_p2p_enabled();
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(ggml_backend_cuda_context & ctx,
                                                                               int                         device,
                                                                               [[maybe_unused]] int        stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(ctx, device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

// staging of ggml_backend_cuda_set_tensors_batch, per backend context
struct ggml_cuda_set_batch {
    void *         host     = nullptr; // pinned and mapped
    void *         host_dev = nullptr; // host as the device sees it
    void *         dev      = nullptr;
    size_t         size     = 0;
    unsigned int * done     = nullptr; // in host: the last batch the kernel finished
    unsigned int * done_dev = nullptr;
    unsigned int   seq      = 0;       // the last batch launched
};
static std::mutex ggml_cuda_set_batch_mutex;
static std::unordered_map<const ggml_backend_cuda_context *, ggml_cuda_set_batch> ggml_cuda_set_batches;

static void ggml_cuda_set_batch_free(const ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(ggml_cuda_set_batch_mutex);
    auto it = ggml_cuda_set_batches.find(ctx);
    if (it == ggml_cuda_set_batches.end()) {
        return;
    }
    ggml_cuda_set_device(ctx->device);
    // the kernel may still read the staging buffer
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFreeHost(it->second.host));
    CUDA_CHECK(cudaFree(it->second.dev));
    ggml_cuda_set_batches.erase(it);
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    ggml_cuda_set_batch_free(this);

    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
        if (block_flags[i] != nullptr) {
            CUDA_CHECK(cudaFree(block_flags[i]));
        }
        if (ar_tickets[i] != nullptr) {
            CUDA_CHECK(cudaFree(ar_tickets[i]));
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    ggml_cuda_preload_modules(buft_ctx->device);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

    // Latency path for small messages, tried ahead of try_allreduce. Null when peer access is
    // unavailable or the user disabled it, in which case try_allreduce handles every call.
    ggml_cuda_p2p_ar *          p2p_ar = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms; // not owned, see ggml_cuda_nccl_comms
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
        ggml_cuda_ar_pipeline_free(ar_pipeline);
        ggml_cuda_p2p_ar_free(p2p_ar);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
// Compress the payload to a 16-bit type for the reduction. Which one is faster is hardware dependent:
//   BF16 keeps the full FP32 exponent range, but before Ampere neither its conversion nor its addition
//   has native instructions, so both are emulated. FP16 is native as far back as Volta and carries three
//   more mantissa bits, but saturates at 65504 - a partial sum above that would turn into an infinity.
template <typename T>
static void ggml_backend_cuda_comm_allreduce_nccl_16bit(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors, const int64_t ne,
        to_t_cuda_t<T> to_16, to_fp32_cuda_t to_fp32, ncclDataType_t nccl_type) {
    const size_t n_backends = comm_ctx->backends.size();

    ggml_cuda_pool_alloc<T> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(T), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, nccl_type, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }
}

static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // Type used to reduce large tensors. GGML_CUDA_AR_DTYPE selects it: bf16 (default), f16 or f32.
    enum ar_dtype { AR_BF16, AR_F16, AR_F32 };
    static const ar_dtype ar_dt = [] {
        const char * env = getenv("GGML_CUDA_AR_DTYPE");
        if (env == nullptr)          return AR_BF16;
        if (strcmp(env, "f32") == 0) return AR_F32;
        if (strcmp(env, "f16") == 0) return AR_F16;
        return AR_BF16;
    }();

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if (ar_dt == AR_F32 ||
            (n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to 16 bit for the reduction:
    if (ar_dt == AR_F16) {
        ggml_backend_cuda_comm_allreduce_nccl_16bit<half>(comm_ctx, tensors, ne,
            ggml_get_to_fp16_cuda(GGML_TYPE_F32), ggml_get_to_fp32_cuda(GGML_TYPE_F16), ncclFloat16);
    } else {
        ggml_backend_cuda_comm_allreduce_nccl_16bit<nv_bfloat16>(comm_ctx, tensors, ne,
            ggml_get_to_bf16_cuda(GGML_TYPE_F32), ggml_get_to_fp32_cuda(GGML_TYPE_BF16), ncclBfloat16);
    }

    return true;
}

// AllGather via NCCL. srcs[i] holds device i's slice along dim 0, dsts[i] receives all slices concatenated.
// NCCL has no allgatherv, so uneven slices are handled with one broadcast per source inside a group.
static bool ggml_backend_cuda_comm_allgather_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** srcs, struct ggml_tensor ** dsts) {
    const size_t n_backends = comm_ctx->backends.size();

    const ggml_type type = dsts[0]->type;
    ncclDataType_t nccl_type;
    switch (type) {
        case GGML_TYPE_F32: nccl_type = ncclFloat; break;
        case GGML_TYPE_F16: nccl_type = ncclHalf;  break;
        case GGML_TYPE_I32: nccl_type = ncclInt32; break; // the top-k index rows of a token-split lightning indexer
        default:            return false;
    }

    const int64_t ne0    = dsts[0]->ne[0];
    const int64_t n_rows = ggml_nrows(dsts[0]);
    if (ne0 == 0 || n_rows == 0) {
        return true;
    }

    int64_t offset[GGML_CUDA_MAX_DEVICES + 1] = {0};
    bool    slices_equal = true;
    for (size_t i = 0; i < n_backends; ++i) {
        if (srcs[i] == nullptr || dsts[i] == nullptr) {
            return false;
        }
        if (srcs[i]->type != type || dsts[i]->type != type) {
            return false;
        }
        if (!ggml_is_contiguously_allocated(srcs[i]) || !ggml_is_contiguously_allocated(dsts[i])) {
            return false;
        }
        if (dsts[i]->ne[0] != ne0 || ggml_nrows(dsts[i]) != n_rows || ggml_nrows(srcs[i]) != n_rows) {
            return false;
        }
        offset[i + 1] = offset[i] + srcs[i]->ne[0];
        slices_equal &= srcs[i]->ne[0] == srcs[0]->ne[0];
    }
    if (offset[n_backends] != ne0) {
        return false;
    }

    // a single row is laid out exactly as an allgather wants it, no per-source offsets needed
    if (slices_equal && n_rows == 1) {
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllGather(srcs[i]->data, dsts[i]->data, srcs[i]->ne[0], nccl_type,
                comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t s = 0; s < n_backends; ++s) {
        if (srcs[s]->ne[0] == 0) {
            continue;
        }
        for (int64_t r = 0; r < n_rows; ++r) {
            for (size_t i = 0; i < n_backends; ++i) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                const void * sendbuf = (const char *) srcs[i]->data + r*srcs[i]->nb[1];
                void       * recvbuf =       (char *) dsts[i]->data + r*dsts[i]->nb[1] + offset[s]*dsts[i]->nb[0];
                NCCL_CHECK(ncclBroadcast(sendbuf, recvbuf, srcs[s]->ne[0], nccl_type, s,
                    comm_ctx->comms[i], cuda_ctx->stream()));
            }
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

#ifdef GGML_USE_NCCL
// One set of communicators per set of devices for the whole process, shared by its contexts and never destroyed:
//   ncclCommDestroy, and ncclCommAbort alike, could wait forever on a proxy thread (NCCL 2.32 on Windows) when a
//   context was freed before the process exited. NCCL orders the operations of a communicator across streams.
static const std::vector<ncclComm_t> * ggml_cuda_nccl_comms(const std::vector<int> & dev_ids, ncclResult_t & rc) {
    static std::mutex mtx;
    static std::map<std::vector<int>, std::vector<ncclComm_t>> cliques;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = cliques.find(dev_ids);
    if (it != cliques.end()) {
        return &it->second;
    }
    std::vector<ncclComm_t> comms(dev_ids.size());
    rc = ncclCommInitAll(comms.data(), (int) comms.size(), dev_ids.data());
    if (rc != ncclSuccess) {
        return nullptr;
    }
    return &cliques.emplace(dev_ids, std::move(comms)).first->second;
}
#endif // GGML_USE_NCCL

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    ncclResult_t rc = ncclSuccess;
    const std::vector<ncclComm_t> * comms = ggml_cuda_nccl_comms(ret->dev_ids, rc);
    if (comms != nullptr) {
        ret->comms = *comms;
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ggml_cuda_preload_modules_close();
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Default: NCCL where it is compiled in, on every platform (its init falls back to internal itself). The internal
        //   pipeline takes 2 devices only, so with more it ends on the meta-backend butterfly: on Windows at 4 GPUs that
        //   cost prefill 37-53% (Qwen3.8-27B F16, gemma-4-31B, Muse-Glimmer).
#if defined(GGML_USE_NCCL)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(GGML_USE_NCCL)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    // Independent of the mode above: small messages are latency-bound, where a ring loses badly
    // to a single round of direct peer writes. Declines fall through to the selected mode.
    ret->p2p_ar = ggml_cuda_p2p_ar_init(ret->dev_ids.data(), ret->dev_ids.size());

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    ggml_cuda_preload_modules_close();
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    // The one-shot P2P path only claims small, contiguous F32 messages; anything it declines
    // falls through to the mode chosen by the init chain.
    if (comm_ctx->p2p_ar != nullptr &&
            ggml_cuda_p2p_ar_allreduce(comm_ctx->p2p_ar, comm_ctx->backends.data(), tensors)) {
        return true;
    }

    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// Two-step form for the meta backend's launch threads: only the one-shot P2P path can be split by
// rank, so prepare declines whatever that path declines and the caller uses the one-step form.
static bool ggml_backend_cuda_comm_allreduce_prepare(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    ggml_cuda_preload_modules_close();
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->p2p_ar != nullptr &&
        ggml_cuda_p2p_ar_prepare(comm_ctx->p2p_ar, comm_ctx->backends.data(), tensors);
}

static void ggml_backend_cuda_comm_allreduce_launch(void * comm_ctx_v, struct ggml_tensor ** tensors, size_t rank) {
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    ggml_cuda_p2p_ar_launch(comm_ctx->p2p_ar, comm_ctx->backends.data(), tensors, rank);
}

static bool ggml_backend_cuda_comm_allgather_tensor(void * comm_ctx_v, struct ggml_tensor ** srcs, struct ggml_tensor ** dsts) {
    ggml_cuda_preload_modules_close();
    if (comm_ctx_v == nullptr) {
        return false;
    }
#ifdef GGML_USE_NCCL
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    if (comm_ctx->backends.size() > GGML_CUDA_MAX_DEVICES) {
        return false;
    }
    // the logits of a backend sampler, every decoded token: ~15 us through peer memory against ~45 us for NCCL's ring
    if (ggml_cuda_p2p_ar_allgather(comm_ctx->p2p_ar, comm_ctx->backends.data(), srcs, dsts)) {
        return true;
    }
    // no communicators with GGML_CUDA_ALLREDUCE=internal/none or a failed NCCL init: the meta backend gathers with copies
    if (comm_ctx->comms.empty()) {
        return false;
    }
    return ggml_backend_cuda_comm_allgather_nccl(comm_ctx, srcs, dsts);
#else
    GGML_UNUSED(srcs);
    GGML_UNUSED(dsts);
    return false;
#endif // GGML_USE_NCCL
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (!ggml_cuda_expert_cpu_host_free(buffer->context)) {
        CUDA_CHECK(cudaFreeHost(buffer->context));
    }
}

// a buffer from ggml_cuda_expert_cpu_host_alloc: its expert weights go on the NUMA nodes before anything writes them
static enum ggml_status ggml_backend_cuda_host_buffer_place_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src == nullptr) {
        ggml_cuda_expert_cpu_place(tensor);
    }
    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(buffer);
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (ggml_cuda_pinned_memory_disabled()) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // GGML_CUDA_EXPERT_CPU: the host experts are placed on the NUMA nodes first and pinned later, see expert-cpu.cuh
    void * ptr = ggml_cuda_expert_cpu_host_alloc(size);
    const bool place = ptr != nullptr;
    if (!place) {
        ptr = ggml_cuda_host_malloc(size);
    }

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;
    if (place) {
        buffer->iface.init_tensor = ggml_backend_cuda_host_buffer_place_tensor;
    }

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer        = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size        = */ NULL, // SIZE_MAX: on Windows 4 GiB parts run out of pinned memory long before one large allocation does
            /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

// whether the converted src1 of a cuBLAS matmul is kept in ctx.cublas_src1. Only the memory of graph tensors is followed
//   by the invalidation, not a pool copy that a caller passes in (the experts of MUL_MAT_ID)
static bool ggml_cuda_cublas_keep_src1(const ggml_backend_cuda_context & ctx, const ggml_tensor * src1) {
    const char * src1_base = src1->buffer ? (const char *) ggml_backend_buffer_get_base(src1->buffer) : nullptr;
    return !ggml_cuda_info().devices[ctx.device].vmm && src1_base != nullptr && (const char *) src1->data >= src1_base &&
        (const char *) src1->data + ggml_nbytes(src1) <= src1_base + ggml_backend_buffer_get_size(src1->buffer);
}

// the key of the converted src1 in ctx.cublas_src1, and the strides of the copy in elements
static void ggml_cuda_cublas_store_src1(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const ggml_type compute_type,
        const int64_t s11, const int64_t s12, const int64_t s13, const bool cont_2) {
    auto & cached = ctx.cublas_src1;
    cached.owner     = src1->view_src ? src1->view_src : src1;
    cached.data      = src1->data;
    cached.nbytes    = ggml_nbytes(src1);
    cached.type      = compute_type;
    cached.stream_no = ctx.curr_stream_no;
    std::copy_n(src1->ne, GGML_MAX_DIMS, cached.ne);
    std::copy_n(src1->nb, GGML_MAX_DIMS, cached.nb);
    cached.s11       = s11;
    cached.s12       = s12;
    cached.s13       = s13;
    cached.cont_2    = cont_2;
}

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    // F32 chunks can have padding between rows to preserve the original destination stride.
    GGML_ASSERT(ggml_is_contiguous(dst) ||
        (compute_type == GGML_TYPE_F32 && ggml_is_contiguous_1(dst)));

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ldc = nb1 / sizeof(float);
    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    // the matmul before may have converted the same input, e.g. q, k and v of one norm, or a GLU written it
    auto & cached = ctx.cublas_src1;
    const ggml_tensor * src1_owner = src1->view_src ? src1->view_src : src1;
    const bool keep_src1 = ggml_cuda_cublas_keep_src1(ctx, src1);

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else if (keep_src1 && cached.buf && cached.owner == src1_owner && cached.data == src1->data &&
            cached.type == compute_type && cached.stream_no == ctx.curr_stream_no &&
            std::equal(std::begin(cached.ne), std::end(cached.ne), src1->ne) &&
            std::equal(std::begin(cached.nb), std::end(cached.nb), src1->nb)) {
        src1_ptr       = (const cuda_t *) cached.buf->get();
        s11            = cached.s11;
        s12            = cached.s12;
        s13            = cached.s13;
        is_src1_cont_2 = cached.cont_2;
    } else {
        cuda_t * src1_conv = nullptr;
        if (keep_src1) {
            cached.buf.reset();
            cached.buf = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), ggml_nelements(src1)*sizeof(cuda_t));
            src1_conv  = (cuda_t *) cached.buf->get();
        } else {
            src1_conv  = src1_alloc.alloc(ggml_nelements(src1));
        }

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_conv, ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_conv, ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_conv;

        if (keep_src1) {
            ggml_cuda_cublas_store_src1(ctx, src1, compute_type, s11, s12, s13, is_src1_cont_2);
        }
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ldc));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ldc,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ldc, ne1*ldc, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ldc,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

// the type cuBLAS runs a MUL_MAT in, which src1 is converted to unless it has it
static ggml_type ggml_cuda_mul_mat_cublas_compute_type(const int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 0);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    // read once: a getenv takes a lock and scans the environment, and this runs for every cuBLAS matmul
    static const ggml_type env_type = [] {
        const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
        if (env_c == nullptr) {
            return GGML_TYPE_COUNT;
        }
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            return GGML_TYPE_F32;
        }
        if (env_cpp == "f16" || env_cpp == "fp16") {
            return GGML_TYPE_F16;
        }
        if (env_cpp == "bf16") {
            return GGML_TYPE_BF16;
        }
        if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
        return GGML_TYPE_COUNT;
    }();
    if (env_type != GGML_TYPE_COUNT) {
        compute_type = env_type;
    }
    return compute_type;
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;

    const ggml_type compute_type = ggml_cuda_mul_mat_cublas_compute_type(cc, src0, src1, dst);

    constexpr size_t max_src0_convert_size = 512ull * 1024 * 1024;
    const size_t src0_f32_size = ggml_nelements(src0) * sizeof(float);

    if (compute_type == GGML_TYPE_F32 &&
            (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
            src0_f32_size > max_src0_convert_size) {
        const size_t f32_row_size = src0_f32_size / src0->ne[1];
        const int64_t rows_per_chunk = std::max<int64_t>(1, (int64_t) (max_src0_convert_size / f32_row_size));

        if (rows_per_chunk < src0->ne[1]) {
            ggml_tensor src0_chunk = *src0;
            ggml_tensor dst_chunk = *dst;

            for (int64_t i01 = 0; i01 < src0->ne[1]; i01 += rows_per_chunk) {
                src0_chunk.ne[1] = std::min(rows_per_chunk, src0->ne[1] - i01);
                src0_chunk.data = (char *) src0->data + i01*src0->nb[1];

                dst_chunk.ne[0] = src0_chunk.ne[1];
                dst_chunk.data = (char *) dst->data + i01*dst->nb[0];

                ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, &src0_chunk, src1, &dst_chunk);
            }
            return;
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

// A quantized gate and up only fuse on mul_mat_vec_q, which reads the activations through the q8_1 copy that it makes
//   before its kernel writes: the GLU output may take their memory, as ggml-alloc does once the up projection is the
//   last reader (the gemma 4 MoE dense MLP)
static const ggml_tensor * ggml_cuda_glu_activations_may_alias(const ggml_tensor * ffn_up) {
    return ggml_is_quantized(ffn_up->src[0]->type) ? ffn_up->src[1] : nullptr;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    // a pool of cached experts is gathered inside ggml_cuda_mul_mat_id, the fused path would read it stale
    if (ggml_cuda_expert_cache_is(src0)) {
        return false;
    }

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

// cache_ok: the caller passes a pool of cached experts its gathered ids, see ggml_cuda_expert_cache_remap
static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor, const bool with_gate = false, const bool cache_ok = false) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // a pool of cached experts is gathered inside ggml_cuda_mul_mat_id, the fused path would read it stale
    if (ggml_cuda_expert_cache_is(src0) && !cache_ok) {
        return false;
    }

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    // a fused gate can lose to two launches, depending on the arch, type and grid
    if (with_gate) {
        const bool    mul_mat_id    = tensor->op == GGML_OP_MUL_MAT_ID;
        const int64_t nchannels_dst = mul_mat_id ? dst->ne[1] : dst->ne[2];
        if ((!mul_mat_id || dst->ne[2] == 1) &&
                !ggml_cuda_should_fuse_mmvq_gate(src0->type, cc, src0->ne[1], nchannels_dst, dst->ne[3], mul_mat_id)) {
            return false;
        }
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx) {
    if (routed_idx + 2 >= graph->n_nodes || shared_idx + 2 >= graph->n_nodes || shared_idx < routed_idx + 3) {
        return false;
    }
    const int nodes[] = { routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
                           GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int outputs[] = { routed_idx + 2, shared_idx + 2 };
    if (!ggml_can_fuse_subgraph_ext(graph, nodes, 6, ops, outputs, 2)) {
        return false;
    }

    const ggml_tensor * routed = graph->nodes[routed_idx + 2];
    const ggml_tensor * shared = graph->nodes[shared_idx + 2];
    const ggml_tensor * gate = routed->src[0];
    const ggml_tensor * up = routed->src[1];
    const ggml_tensor * shared_gate = shared->src[0];
    const ggml_tensor * shared_up = shared->src[1];
    const auto is_pair = [&](const ggml_tensor * a, const ggml_tensor * b, int idx) {
        return (a == graph->nodes[idx] && b == graph->nodes[idx + 1]) ||
               (b == graph->nodes[idx] && a == graph->nodes[idx + 1]);
    };
    // only batch-size independent checks here: graph_optimize must produce the same graph topology for every ubatch
    // size, otherwise ggml-alloc has to re-reserve (and the scheduler to synchronize) at runtime.
    // the MMVQ batch size check is done in ggml_cuda_try_fuse
    if (!is_pair(gate, up, routed_idx) || !is_pair(shared_gate, shared_up, shared_idx) ||
            !ggml_cuda_should_fuse_mul_mat(up, gate, routed) ||
            !ggml_cuda_should_fuse_mul_mat(shared_up, shared_gate, shared) ||
            !up->src[0]->buffer ||
            !ggml_is_quantized(up->src[0]->type)) {
        return false;
    }
    const ggml_tensor * input = up->src[1];
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * shared_weight = shared_up->src[0];
    if (input->op != GGML_OP_RESHAPE || input->src[0] != shared_up->src[1] ||
            input->ne[1] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(shared_up->src[1]) || !ggml_is_matrix(shared_up->src[1]) ||
            weight->type != shared_weight->type || weight->ne[0] != shared_weight->ne[0] ||
            weight->ne[1] != shared_weight->ne[1] || weight->nb[1] != shared_weight->nb[1] || weight->ne[3] != 1 ||
            !ggml_is_matrix(shared_weight) || !ggml_is_contiguous(shared_weight) ||
            !ggml_is_contiguous(shared_gate->src[0]) || !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
        return false;
    }
    if (shared_weight->op != GGML_OP_NONE || shared_gate->src[0]->op != GGML_OP_NONE ||
            ggml_get_glu_op(routed) != ggml_get_glu_op(shared) ||
            ggml_get_op_params_f32(routed, 3) != ggml_get_op_params_f32(shared, 3)) {
        return false;
    }
    return true;
}

enum ggml_cuda_mul_mat_path {
    GGML_CUDA_MUL_MAT_PATH_CUBLAS,
    GGML_CUDA_MUL_MAT_PATH_MMVF,
    GGML_CUDA_MUL_MAT_PATH_MMVF_TRANSPOSED,
    GGML_CUDA_MUL_MAT_PATH_MMF,
    GGML_CUDA_MUL_MAT_PATH_MMVQ,
    GGML_CUDA_MUL_MAT_PATH_MMQ,
};

// the kernel ggml_cuda_mul_mat runs a MUL_MAT without the Hadamard hint on
static ggml_cuda_mul_mat_path ggml_cuda_mul_mat_choose(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return GGML_CUDA_MUL_MAT_PATH_CUBLAS;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        return GGML_CUDA_MUL_MAT_PATH_MMVF;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        return GGML_CUDA_MUL_MAT_PATH_MMVF_TRANSPOSED;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        return GGML_CUDA_MUL_MAT_PATH_MMF;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        return GGML_CUDA_MUL_MAT_PATH_MMVQ;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        return GGML_CUDA_MUL_MAT_PATH_MMQ;
    }
    return GGML_CUDA_MUL_MAT_PATH_CUBLAS;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    switch (ggml_cuda_mul_mat_choose(ctx, src0, src1, dst)) {
        case GGML_CUDA_MUL_MAT_PATH_CUBLAS:
            ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
            break;
        case GGML_CUDA_MUL_MAT_PATH_MMVF:
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
            break;
        case GGML_CUDA_MUL_MAT_PATH_MMVF_TRANSPOSED: {
            ggml_tensor dst_vec = *dst;
            dst_vec.ne[0] = src1->ne[1];
            dst_vec.ne[1] = 1;
            dst_vec.nb[1] = dst_vec.nb[0]*src1->ne[1];
            dst_vec.nb[2] = dst_vec.nb[1];
            dst_vec.nb[3] = dst_vec.nb[1];
            ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        } break;
        case GGML_CUDA_MUL_MAT_PATH_MMF:
            ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
            break;
        case GGML_CUDA_MUL_MAT_PATH_MMVQ:
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
            break;
        case GGML_CUDA_MUL_MAT_PATH_MMQ:
            ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
            break;
    }
}

// The GLU of an FFN whose down projection follows on cuBLAS in f16 also writes the f16 copy of its output that the
//   matmul would convert, as the entry of ctx.cublas_src1 the matmul looks up
static bool ggml_cuda_glu_with_f16_copy(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    ggml_tensor * glu = cgraph->nodes[i];
    if (glu->op != GGML_OP_GLU || i + 1 >= cgraph->n_nodes || !ggml_cuda_op_glu_f16_copy_supported(glu)) {
        return false;
    }
    const ggml_tensor * mm = cgraph->nodes[i + 1];
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != glu || (mm->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            ggml_cuda_op_mul_mat_use_fwht(mm) || !ggml_cuda_cublas_keep_src1(ctx, glu)) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (ggml_cuda_mul_mat_choose(ctx, mm->src[0], glu, mm) != GGML_CUDA_MUL_MAT_PATH_CUBLAS ||
            ggml_cuda_mul_mat_cublas_compute_type(cc, mm->src[0], glu, mm) != GGML_TYPE_F16) {
        return false;
    }

    auto & cached = ctx.cublas_src1;
    cached.buf.reset();
    cached.buf = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), ggml_nelements(glu)*sizeof(half));
    ggml_cuda_op_glu_f16_copy(ctx, glu, (half *) cached.buf->get());
    // a contiguous input keeps its strides through the conversion
    ggml_cuda_cublas_store_src1(ctx, glu, GGML_TYPE_F16, glu->nb[1]/sizeof(float), glu->nb[2]/sizeof(float),
        glu->nb[3]/sizeof(float), ggml_is_contiguous_2(glu));
    return true;
}

// A float MUL_MAT_ID of ne2 <= MMVF_MAX_BATCH_SIZE tokens on mul_mat_vec_f. On Volta mul_mat_f launches a block per row
//   tile and expert, and at one token nearly all of them only scan the ids: F16 256 experts, 8 used, down 2048 x 128
//   27.6 -> 7.0 us, gate 128 x 2048 15.4 -> 9.6. From 2 tokens the down loses (31.7 vs 55.1 us), so F16 stays on
//   mul_mat_f where it takes the matrix. BF16 and F32 have no mul_mat_f there and would sort on the host instead (BF16
//   down at 1/8 tokens 123/711 -> 7.7/203 us). GGML_CUDA_NO_MMID_MMVF=1 restores that on Volta.
static bool ggml_cuda_mul_mat_id_use_mmvf(const ggml_tensor * src0, const int64_t ne2, const int cc) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        return true;
    }
    static const bool disabled = getenv("GGML_CUDA_NO_MMID_MMVF") != nullptr;
    const bool is_volta = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING;
    if (disabled || !is_volta || (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16)) {
        return false;
    }
    // the half2/float2 loads of mul_mat_vec_f
    const size_t ts = ggml_type_size(src0->type);
    if (src0->ne[0] % 2 != 0 || src0->nb[0] != ts || src0->nb[1] % (2*ts) != 0 || src0->nb[2] % (2*ts) != 0 ||
            src0->nb[3] % (2*ts) != 0) {
        return false;
    }
    return src0->type != GGML_TYPE_F16 || ne2 == 1 ||
        !ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, ne2, /*mul_mat_id=*/true);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (ggml_cuda_mul_mat_id_use_mmvf(src0, dst->ne[2], cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true, dst->src[2]->ne[0])) {
        return false;
    }

    return true;
}

// A pool of cached experts: bring in the routed ones and return the ids to read the pool with, unless an earlier
//   MUL_MAT_ID of the same ids has done that for it
static void ggml_cuda_expert_cache_remap(ggml_backend_cuda_context & ctx, const ggml_tensor * pool, const ggml_tensor * ids,
        ggml_tensor * ids_remap) {
    for (size_t k = 0; k < ctx.expert_cache_ahead.size(); k++) {
        if (ctx.expert_cache_ahead[k].pool == pool && ctx.expert_cache_ahead[k].ids == ids) {
            *ids_remap = ctx.expert_cache_ahead[k].remap;
            ctx.expert_cache_ahead.erase(ctx.expert_cache_ahead.begin() + k);
            return;
        }
    }
    ggml_cuda_expert_cache_gather(ctx, pool, ids, ids_remap);
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    // src0 is a pool of cached experts: read it through remapped ids
    ggml_tensor ids_remap;
    if (ggml_cuda_expert_cache_is(src0)) {
        ggml_cuda_expert_cache_remap(ctx, src0, ids, &ids_remap);
        ids = &ids_remap;
    } else if (ggml_cuda_expert_cache_is_warm(src0)) {
        // a whole staged copy of a pooled weight: its pool takes the experts this ubatch routed to most
        ggml_cuda_expert_cache_warm(ctx, src0, ids);
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else if (ggml_cuda_mul_mat_id_use_mmvf(src0, ne2, cc)) {
                ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                return;
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true, ids->ne[0])) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    // the gather indices for the sorted src1 rows, then the scatter indices back to dst
    std::vector<int32_t> ids_to_sorted_host(2*ne_get_rows);
    int32_t * ids_from_sorted_host = ids_to_sorted_host.data() + ne_get_rows;

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // every slot of a token that names the expert gets a row of its own, also when an expert repeats within the row.
    //   A counting sort: count the rows of each expert, then place the slots in (token, slot) order, so the rows of an
    //   expert keep that order. A scan of experts x tokens x slots took 24 ms for 4096 tokens of 8 of 256 experts.
    const auto expert_of = [&](const int64_t i12, const int64_t iex) {
        return *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
    };
    for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
        for (int64_t iex = 0; iex < n_expert_used; ++iex) {
            const int32_t expert_to_use = expert_of(i12, iex);
            GGML_ASSERT(expert_to_use >= 0 && expert_to_use < ne02);
            tokens_per_expert[expert_to_use]++;
        }
    }
    std::vector<int32_t> expert_next_row(ne02);
    for (int64_t i02 = 1; i02 < ne02; ++i02) {
        expert_next_row[i02] = expert_next_row[i02 - 1] + tokens_per_expert[i02 - 1];
    }
    for (int64_t i12 = 0; i12 < ne12; ++i12) {
        for (int64_t iex = 0; iex < n_expert_used; ++iex) {
            const int32_t row = expert_next_row[expert_of(i12, iex)]++;
            ids_from_sorted_host[i12*n_expert_used + iex] = row;
            ids_to_sorted_host[row] = i12*ne11 + iex % ne11;
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        memcpy(dst_slice.op_params, dst->op_params, sizeof(dst_slice.op_params));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_cuda_op_conv3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_MEAN:
            ggml_cuda_op_get_rows_mean(ctx, dst);
            break;
        case GGML_OP_KQ_MASK:
            ggml_cuda_op_kq_mask(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

struct ggml_cuda_set_batch_entry {
    char *   dst;
    uint32_t offset; // of the bytes in the staging buffer
    uint32_t size;
};

// block b copies the bytes of entry b from the staging buffer to their tensor; the last block to finish tells the host
//   through done that the staging buffer may be written again
static __global__ void k_set_tensors_batch(const char * __restrict__ staging, unsigned int * ticket,
        volatile unsigned int * done, const unsigned int seq) {
    const ggml_cuda_set_batch_entry e = ((const ggml_cuda_set_batch_entry *) staging)[blockIdx.x];
    const char * src = staging + e.offset;
    if (((uintptr_t) e.dst % 16) == 0 && e.size % 16 == 0) {
        for (uint32_t i = 16*threadIdx.x; i < e.size; i += 16*blockDim.x) {
            *(int4 *) (e.dst + i) = *(const int4 *) (src + i);
        }
    } else {
        for (uint32_t i = threadIdx.x; i < e.size; i += blockDim.x) {
            e.dst[i] = src[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (atomicAdd(ticket, 1) == gridDim.x - 1) {
            *ticket = 0;
            __threadfence_system();
            *done = seq;
        }
    }
}

// Small host inputs of one graph in one kernel launch instead of a copy each, which costs ~5 us of host time per copy
//   (more when the launch workers of all devices call into the driver at once), with the GPU idle between two decoded
//   tokens. The data is staged in pinned memory, so the caller may write it again at once, and the stream orders the
//   kernel before later work: no sync is needed after it. Up to 64 KiB the kernel reads the staging buffer in host
//   memory, above it a copy to the device goes first. false: nothing was copied (too many or too large).
//   GGML_CUDA_SET_TENSORS_BATCH=0 turns it off.
static bool ggml_backend_cuda_set_tensors_batch(ggml_backend_t backend, int n, ggml_tensor * const * tensors,
        const void * const * data, const size_t * sizes) {
#if defined(GGML_USE_MUSA)
    // no mapped pinned memory
    GGML_UNUSED_VARS(backend, n, tensors, data, sizes);
    return false;
#else
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_SET_TENSORS_BATCH");
        return env == nullptr || atoi(env) != 0;
    }();
    constexpr size_t max_bytes       = 1u << 20;
    constexpr size_t max_bytes_mapped = 64u << 10;
    if (!enabled || n < 2 || n > 1024) {
        return false;
    }
    const size_t header = GGML_PAD(n*sizeof(ggml_cuda_set_batch_entry), 16);
    size_t total = header;
    for (int i = 0; i < n; i++) {
        total += GGML_PAD(sizes[i], 16);
    }
    if (total > max_bytes) {
        return false;
    }

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_batch * b;
    {
        std::lock_guard<std::mutex> lock(ggml_cuda_set_batch_mutex);
        b = &ggml_cuda_set_batches[cuda_ctx];
    }
    ggml_cuda_set_device(cuda_ctx->device);
    cudaStream_t stream = cuda_ctx->stream();
    if (b->host != nullptr) {
        // the previous batch has left the staging buffer: done by the time the caller synchronized, else wait
        if (*(volatile unsigned int *) b->done != b->seq) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    if (b->size < total) {
        if (b->host) {
            CUDA_CHECK(cudaFreeHost(b->host));
            CUDA_CHECK(cudaFree(b->dev));
        }
        b->size = std::max(total, (size_t) 64*1024);
        // a word after the data for done, and the ticket of the blocks at the start of the device buffer
        CUDA_CHECK(cudaHostAlloc(&b->host, b->size + 16, cudaHostAllocPortable | cudaHostAllocMapped));
        CUDA_CHECK(cudaMalloc(&b->dev, b->size + 16));
        CUDA_CHECK(cudaMemsetAsync(b->dev, 0, 16, stream));
        b->done = (unsigned int *) ((char *) b->host + b->size);
        *(volatile unsigned int *) b->done = b->seq;
        if (cudaHostGetDevicePointer(&b->host_dev, b->host, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            b->host_dev = nullptr;
        }
        b->done_dev = b->host_dev ? (unsigned int *) ((char *) b->host_dev + b->size) : nullptr;
    }
    if (b->done_dev == nullptr) {
        return false; // the kernel cannot tell the host it is done
    }

    ggml_cuda_set_batch_entry * entries = (ggml_cuda_set_batch_entry *) b->host;
    size_t offset = header;
    for (int i = 0; i < n; i++) {
        ggml_tensor * t = tensors[i];
        ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
        GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");
        entries[i] = { (char *) t->data, (uint32_t) offset, (uint32_t) sizes[i] };
        memcpy((char *) b->host + offset, data[i], sizes[i]);
        offset += GGML_PAD(sizes[i], 16);
    }

    const char * staging = (const char *) b->host_dev;
    if (total > max_bytes_mapped) {
        CUDA_CHECK(cudaMemcpyAsync((char *) b->dev + 16, b->host, total, cudaMemcpyHostToDevice, stream));
        staging = (const char *) b->dev + 16;
    }
    b->seq++;
    k_set_tensors_batch<<<n, 128, 0, stream>>>(staging, (unsigned int *) b->dev, (volatile unsigned int *) b->done_dev, b->seq);
    CUDA_CHECK(cudaGetLastError());
    return true;
#endif // defined(GGML_USE_MUSA)
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_debug_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_GRAPH_DEBUG");
    return enabled;
}

static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    int  cc             = -1; // looked up at the first MUL_MAT_ID: cudaGetDevice is not free
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            if (cc < 0) {
                cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            }
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
                if (ggml_cuda_graph_debug_enabled()) {
                    GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
                }
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        }
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                prop.node_src_extra[j]     = cgraph->nodes[i]->src[j]->extra; // e.g. an expert cache warm tag
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    if (res) {
        graph->props_gen++;
    }
    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
        }

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

// multi: the rows go through rms_norm_mul_rope_f32, which also does the multi-section ropes (mrope, imrope)
static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows,
                                                const bool          multi = false) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code, and the fused norm kernel for the multi-section ropes
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
            !(multi && (mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE))) {
        return false;
    }

    return true;
}

// the norm (and its weight, when mul is set) that the fused rms_norm_mul_rope kernel can apply to its rows
static bool ggml_cuda_rms_norm_mul_fusable_rows(const ggml_tensor * rms_norm, const ggml_tensor * mul) {
    if (rms_norm->op != GGML_OP_RMS_NORM || (mul && mul->op != GGML_OP_MUL)) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32) {
        return false;
    }

    if (mul && (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32)) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (mul && rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (mul && !ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        (mul && (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])))) {
        return false;
    }

    // rms_norm_mul_rope_cuda puts the channels and samples of the rows on grid y and z
    if (rms_norm->ne[2] > UINT16_MAX || rms_norm->ne[3] > UINT16_MAX) {
        return false;
    }

    return rms_norm->ne[0] % 2 == 0;
}

// the rope that the fused rms_norm_mul_rope kernel can apply to its rows
static bool ggml_cuda_rope_fusable_rows(const ggml_tensor * rope) {
    if (rope->op != GGML_OP_ROPE || rope->type != GGML_TYPE_F32 || rope->src[0]->type != GGML_TYPE_F32) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes and multi-section rope, which rotates the neox pairs
    const int mode = ((const int32_t *) rope->op_params)[2];
    const bool is_mrope = (mode & GGML_ROPE_TYPE_MROPE) && mode != GGML_ROPE_TYPE_VISION;
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && !is_mrope) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset (DeepSeek V4 rotates the tail of each head) is handled for the norm pairing only
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0 && (mode != GGML_ROPE_TYPE_NORMAL || n_offs % 2 != 0 || n_offs + n_dims > rope->ne[0])) {
        return false;
    }

    if (rope->ne[2] > UINT16_MAX || rope->ne[3] > UINT16_MAX) {
        return false;
    }

    return true;
}

// mul is null for a norm without weight; reshape, when set, sits between the norm (or its weight) and the rope
static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope,
                                                    const ggml_tensor * reshape = nullptr) {
    if (!ggml_cuda_rms_norm_mul_fusable_rows(rms_norm, mul) || !ggml_cuda_rope_fusable_rows(rope)) {
        return false;
    }

    const ggml_tensor * normed = mul ? mul : rms_norm;
    if (reshape) {
        // a reshape of packed rows keeps every row intact, but a per-row weight would no longer line up
        const ggml_tensor * x = rms_norm->src[0];
        const ggml_tensor * w = mul ? (mul->src[0] == rms_norm ? mul->src[1] : mul->src[0]) : nullptr;
        if (reshape->op != GGML_OP_RESHAPE || reshape->src[0] != normed || rope->src[0] != reshape ||
            rope->ne[0] != rms_norm->ne[0] || x->nb[2] != x->nb[1]*x->ne[1] || x->nb[3] != x->nb[2]*x->ne[2] ||
            (w && ggml_nrows(w) != 1)) {
            return false;
        }
    } else if (rope->src[0] != normed) {
        return false;
    }

    return true;
}

// a norm without weight whose rows go straight into a cache row per token, the V of gemma 4: the rows of the norm
//   kernel, stored by it as set_rows does
static bool ggml_cuda_should_fuse_rms_norm_set_rows(const ggml_tensor * rms_norm, const ggml_tensor * view,
                                                    const ggml_tensor * set_rows) {
    if (!ggml_cuda_rms_norm_mul_fusable_rows(rms_norm, nullptr) || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }

    const ggml_tensor * idx = set_rows->src[1];
    return rms_norm->ne[3] == 1 && view->view_offs == 0 && ggml_is_contiguous(view) &&
           view->ne[0] == rms_norm->ne[0]*rms_norm->ne[1] && view->ne[1] == rms_norm->ne[2] && view->ne[2] == 1 && view->ne[3] == 1 &&
           ggml_cuda_rms_norm_mul_rope_fused_dst_supported(set_rows->type) && set_rows->ne[0] == view->ne[0] &&
           set_rows->ne[2] == 1 && set_rows->ne[3] == 1 && set_rows->nb[0] == ggml_type_size(set_rows->type) &&
           (!ggml_is_quantized(set_rows->type) || rms_norm->ne[0] % ggml_blck_size(set_rows->type) == 0) &&
           idx->type == GGML_TYPE_I64 && idx->ne[0] == view->ne[1] && ggml_nrows(idx) == 1;
}

// the qwen4exp QSA key pooling: get_rows_mean -> reshape -> rms_norm * weight -> reshape -> multi-section rope,
// one token per pooled row; the caller checked the edges
static bool ggml_cuda_should_fuse_get_rows_mean_norm_rope(const ggml_tensor * get_rows_mean, const ggml_tensor * rms_norm,
                                                          const ggml_tensor * mul, const ggml_tensor * rope) {
    const ggml_tensor * src0 = get_rows_mean->src[0];
    const ggml_tensor * w    = mul->src[1];

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_Q8_0) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) || get_rows_mean->src[1]->nb[0] != sizeof(int32_t)) {
        return false;
    }

    // one warp per row, 4 columns per lane
    if (get_rows_mean->ne[0] != 128 || !ggml_is_contiguous(get_rows_mean) || ggml_nrows(get_rows_mean) > INT32_MAX) {
        return false;
    }

    if (rms_norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(rms_norm, mul) || w->ne[0] != 128 || ggml_nelements(w) != 128 || !ggml_is_contiguous(w)) {
        return false;
    }

    const int mode = ((const int32_t *) rope->op_params)[2];
    if (!(mode & GGML_ROPE_TYPE_MROPE) || (mode & GGML_ROPE_TYPE_NEOX) || mode == GGML_ROPE_TYPE_VISION) {
        return false;
    }

    // a lane owns both channels of a pair
    const int n_dims = ((const int32_t *) rope->op_params)[1];
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if ((n_dims != 64 && n_dims != 128) || n_offs != 0 || rope->src[2] != nullptr) {
        return false;
    }

    const ggml_tensor * x = rope->src[0];
    return x->ne[0] == 128 && x->ne[1] == 1 && x->ne[2] == ggml_nrows(get_rows_mean) && x->ne[3] == 1 &&
        ggml_is_contiguous(rope) && rope->src[1]->type == GGML_TYPE_I32;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// match ssm_scan + the strided cpy that scatters its state snapshots into the cache, so the kernel writes them and skips the cpy
static int ggml_cuda_try_ssm_scan_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_ssm_scan_fused_cache & fused_state_cpy) {
    const ggml_tensor * ssm = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the scan output must not be a graph output
    if (ssm->op != GGML_OP_SSM_SCAN || ssm->type != GGML_TYPE_F32 || (ssm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const int64_t K = ggml_get_op_params_i32(ssm, 0); // snapshot slot count

    const ggml_tensor * s = ssm->src[0];
    const ggml_tensor * x = ssm->src[1];
    const ggml_tensor * A = ssm->src[3];

    const int64_t d_state = s->ne[0];
    const int64_t D       = d_state * s->ne[1] * x->ne[1]; // d_state * head_dim * n_head
    const int64_t n_tok   = x->ne[2];
    const int64_t n_seqs  = x->ne[3];

    // only the mamba-2 kernels (group scan and SSD) write to the cache; mamba-1 still uses the cpy
    if (A->nb[1] != sizeof(float) || (d_state != 96 && d_state != 128 && d_state != 256)) {
        return 0;
    }

    // the scan reads its input rows from the cache (picked by ids), so with more than one seq a seq can read a row that another seq writes in the same launch
    if (n_seqs != 1) {
        return 0;
    }

    const int64_t n_written = std::min<int64_t>(n_tok, K);
    const size_t  tail_off  = ggml_row_size(GGML_TYPE_F32, ggml_nelements(x));

    // snapshot cpy is the first real node after the scan (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the scan snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this scan's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != ssm || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D, the per-seq stride the kernel takes from src0->nb[3]
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback slot 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static int32_t ggml_cuda_graph_use_count(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

// the readers of the node at idx that read data, if the others are views of no elements after it and before end: build_rs
//   gathers the states of all sequences once and views the extra ones, no rows when the ubatch has every sequence
static int32_t ggml_cuda_graph_use_count_skip_empty_views(const ggml_cgraph * cgraph, const int idx, const int end) {
    const ggml_tensor * t = cgraph->nodes[idx];
    int32_t n = ggml_cuda_graph_use_count(cgraph, t);
    for (int j = idx + 1; n > 1 && j < end; j++) {
        const ggml_tensor * v = cgraph->nodes[j];
        if (v->op == GGML_OP_VIEW && v->src[0] == t && ggml_nelements(v) == 0) {
            n--;
        }
    }
    return n;
}

// the input of an l2 norm built as rms_norm + scale (build_gdn_l2_norm), if only t's reader uses the norm
static const ggml_tensor * ggml_cuda_gdn_l2_norm_input(
        const ggml_cgraph * cgraph, const ggml_tensor * t, float & eps, float & post_scale, float & post_bias) {
    if (t->op != GGML_OP_SCALE || t->src[0]->op != GGML_OP_RMS_NORM) {
        return nullptr;
    }
    const ggml_tensor * norm = t->src[0];
    const ggml_tensor * x    = norm->src[0];
    if (t->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(x) ||
            ggml_cuda_graph_use_count(cgraph, t) != 1 || ggml_cuda_graph_use_count(cgraph, norm) != 1 ||
            ((t->flags | norm->flags) & GGML_TENSOR_FLAG_OUTPUT)) {
        return nullptr;
    }
    memcpy(&eps,        norm->op_params,                  sizeof(float));
    memcpy(&post_scale, (const float *) t->op_params + 0, sizeof(float));
    memcpy(&post_bias,  (const float *) t->op_params + 1, sizeof(float));
    return x;
}

// A gated delta net at a few tokens whose q and k are their l2 norms: the kernel applies the norms and the two norm
//   launches are skipped. Only with the snapshot cpy fusion, which runs the op in ggml_cuda_try_fuse. At prefill the
//   norms stay apart, as in the kernel they would lengthen its serial loop over the tokens.
static bool ggml_cuda_gdn_l2_fusion(const ggml_cgraph * cgraph, const int gdn_idx, ggml_cuda_gated_delta_net_l2 & l2) {
    const ggml_tensor * gdn = cgraph->nodes[gdn_idx];
    if (gdn->op != GGML_OP_GATED_DELTA_NET) {
        return false;
    }
    const int64_t S_v       = gdn->src[2]->ne[0];
    const int     warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    if (warp_size != 32 || (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) || gdn->src[2]->ne[2] > 8) {
        return false;
    }
    float eps_k, post_scale_k, post_bias_k;
    const ggml_tensor * q = ggml_cuda_gdn_l2_norm_input(cgraph, gdn->src[0], l2.eps, l2.post_scale, l2.post_bias);
    const ggml_tensor * k = ggml_cuda_gdn_l2_norm_input(cgraph, gdn->src[1], eps_k, post_scale_k, post_bias_k);
    if (!q || !k || q->ne[0] != S_v || !ggml_are_same_shape(q, gdn->src[0]) || !ggml_are_same_shape(k, gdn->src[1]) ||
            !ggml_are_same_stride(q, k) || eps_k != l2.eps || post_scale_k != l2.post_scale || post_bias_k != l2.post_bias) {
        return false;
    }
    ggml_cuda_gated_delta_net_fused_cache cache;
    if (ggml_cuda_try_gdn_cache_fusion(cgraph, gdn_idx, cache) == 0) {
        return false;
    }
    // ggml-alloc frees the norm inputs after the norms, their last readers, so a node from there up to the gated delta
    //   net, the net's own output included, can sit over them. That holds when q and k are separate tensors (kimi-linear
    //   convolves them apart); as views of one convolution output (qwen35) they live on until v is read.
    const ggml_tensor * norms[4] = { gdn->src[0], gdn->src[0]->src[0], gdn->src[1], gdn->src[1]->src[0] };
    int first = gdn_idx;
    int found = 0;
    for (int j = gdn_idx - 1; j >= 0 && j >= gdn_idx - 256 && found < 2; j--) {
        if (cgraph->nodes[j] == norms[1] || cgraph->nodes[j] == norms[3]) {
            first = j;
            found++;
        }
    }
    if (found < 2) {
        return false;
    }
    for (int j = first; j <= gdn_idx; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_op_is_empty(n->op) || n == norms[0] || n == norms[1] || n == norms[2] || n == norms[3]) {
            continue;
        }
        if (ggml_cuda_tensors_overlap(n, q) || ggml_cuda_tensors_overlap(n, k)) {
            return false;
        }
    }
    l2.q = q;
    l2.k = k;
    return true;
}

// build_rs gathers the start state of a recurrent layer with GET_ROWS(states, ids) into a tensor that only the gated
//   delta net reads, and a cpy also stores it to the rollback slot when a batch is shorter than the slots. With one
//   sequence the kernel reads the cache row ids[0] itself and writes that slot, so the GET_ROWS and the cpy do not run.
//   With several, one sequence could read a row that another one writes in the same launch.
//   Only with the snapshot cpy fusion, which runs the net in ggml_cuda_try_fuse; skip_cache is its node count.
//   Returns the extra nodes after those that the net replaces (the cpy), or -1.
static int ggml_cuda_gdn_gather_fusion(
        ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int gdn_idx, const int skip_cache,
        ggml_cuda_gated_delta_net_gather & gather) {
    const ggml_tensor * gdn = cgraph->nodes[gdn_idx];
    const ggml_tensor * v   = gdn->src[2];
    const int64_t       D   = v->ne[0] * v->ne[0] * v->ne[1];
    if (gdn->op != GGML_OP_GATED_DELTA_NET || v->ne[3] != 1 || !cuda_ctx->stream_context().concurrent_events.empty()) {
        return -1;
    }

    // the net reads the GET_ROWS output through reshapes
    const ggml_tensor * s  = gdn->src[5];
    const ggml_tensor * gr = s;
    while (gr->op == GGML_OP_RESHAPE || (gr->op == GGML_OP_VIEW && gr->view_offs == 0)) {
        gr = gr->src[0];
        // the readers of the GET_ROWS are counted below, where its node is known
        if (gr->op != GGML_OP_GET_ROWS && ggml_cuda_graph_use_count(cgraph, gr) != 1) {
            return -1;
        }
    }
    if (gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || (gr->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            gr->ne[0] != D || ggml_nrows(gr) != 1 || !ggml_is_contiguous(gr) || ggml_nelements(s) != D) {
        return -1;
    }
    const ggml_tensor * states = gr->src[0];
    const ggml_tensor * ids    = gr->src[1];
    if (states->type != GGML_TYPE_F32 || states->ne[0] != D || states->nb[0] != sizeof(float) || states->nb[1] % sizeof(float) != 0 ||
            states->data == nullptr || ids->type != GGML_TYPE_I32 || ggml_nelements(ids) != 1 || ids->data == nullptr) {
        return -1;
    }

    // the kernel reads the cache and the ids when the net runs, so no node from the GET_ROWS to it may write either:
    //   past the GET_ROWS, their last reader in the graph, the ids may share memory with a later tensor
    int gr_idx = -1;
    for (int j = gdn_idx - 1; j >= 0 && j >= gdn_idx - 256; j--) {
        if (cgraph->nodes[j] == gr) {
            gr_idx = j;
            break;
        }
    }
    if (gr_idx < 0 || (gr != s && ggml_cuda_graph_use_count_skip_empty_views(cgraph, gr_idx, gdn_idx) != 1)) {
        return -1;
    }
    for (int j = gr_idx + 1; j < gdn_idx; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (!ggml_op_is_empty(n->op) && (ggml_cuda_tensors_overlap(n, states) || ggml_cuda_tensors_overlap(n, ids))) {
            return -1;
        }
    }

    // the cpy of the start state to its rollback slot, if any, is the first real node after the snapshot cpy. It reads s
    //   whole, directly or through a view of no offset, and has no rows when the net wrote every slot
    gather = {};
    int extra     = 0;
    int s_readers = 1; // the net
    for (int j = gdn_idx + skip_cache + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        const ggml_tensor * n_src = n->src[0];
        const bool pre_cpy = n->op == GGML_OP_CPY && !(n->flags & GGML_TENSOR_FLAG_OUTPUT) && (n_src == s ||
            (n_src->op == GGML_OP_VIEW && n_src->src[0] == s && n_src->view_offs == 0 && ggml_cuda_graph_use_count(cgraph, n_src) == 1));
        if (pre_cpy) {
            s_readers++;
            if (!ggml_is_empty(n)) {
                const ggml_tensor * pre = n->src[1];
                if (ggml_nelements(n_src) != D || pre->type != GGML_TYPE_F32 || ggml_nelements(pre) != D ||
                        !ggml_is_contiguous(pre) || pre->data == nullptr) {
                    return -1;
                }
                gather.pre = (float *) pre->data;
                extra      = j - (gdn_idx + skip_cache);
            }
            break;
        }
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        break;
    }
    if (ggml_cuda_graph_use_count(cgraph, s) != s_readers) {
        return -1;
    }

    gather.states     = (const float *) states->data;
    gather.ids        = (const int32_t *) ids->data;
    gather.row_stride = (int64_t) (states->nb[1] / sizeof(float));
    return extra;
}

// whether the GET_ROWS at gr_idx is the state gather of a gated delta net that reads the cache itself
static bool ggml_cuda_gdn_gather_is_folded(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int gr_idx) {
    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return false;
    }
    const ggml_tensor * gr = cgraph->nodes[gr_idx];
    for (int j = gr_idx + 1; j < cgraph->n_nodes && j <= gr_idx + 256; j++) {
        const ggml_tensor * gdn = cgraph->nodes[j];
        if (gdn->op != GGML_OP_GATED_DELTA_NET) {
            continue;
        }
        const ggml_tensor * s = gdn->src[5];
        while (s != gr && (s->op == GGML_OP_RESHAPE || (s->op == GGML_OP_VIEW && s->view_offs == 0))) {
            s = s->src[0];
        }
        if (s != gr) {
            continue;
        }
        ggml_cuda_gated_delta_net_fused_cache cache;
        ggml_cuda_gated_delta_net_gather      gather;
        const int skip_cache = ggml_cuda_try_gdn_cache_fusion(cgraph, j, cache);
        return skip_cache > 0 && ggml_cuda_gdn_gather_fusion(cuda_ctx, cgraph, j, skip_cache, gather) >= 0;
    }
    return false;
}

// build_conv_state of one sequence: GET_ROWS(states, ids) -> RESHAPE -> CONCAT with the transposed new tokens -> a CPY of
//   a window of the concat to each rollback slot -> ... -> SSM_CONV -> SILU. One kernel at the CONCAT reads the cache row
//   itself and writes the slots and the SILU output, so the GET_ROWS, the CONCAT, the CPYs and the SSM_CONV do not run;
//   past GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS (a prefill batch) the concat, ~0.16 ms a layer at 2048 tokens, is not made
//   either. The SILU output is written before the nodes between the CPYs and the SSM_CONV run, so none of them may read
//   or write its memory. The row may hold the states of several convolutions (bailingmoe3: q, k and v), each a view of
//   the GET_ROWS; it then still runs, and the SILU may read the SSM_CONV through a RESHAPE.
// The LFM2 short conv (kernel width 3) has no SILU: the output is a MUL of the convolution by a gate, and the new tokens
//   are the transpose of a MUL, which the kernel computes from its two inputs, so that MUL does not run either.
struct ggml_cuda_ssm_conv_state_fusion {
    const ggml_tensor * states   = nullptr;
    const ggml_tensor * ids      = nullptr;
    const ggml_tensor * x        = nullptr;
    const ggml_tensor * ssm_conv = nullptr;
    ggml_tensor       * silu     = nullptr; // the output: the SILU, or the MUL by the gate
    const ggml_tensor * gate     = nullptr;
    const ggml_tensor * x_a      = nullptr; // x is the transpose of x_a*x_b
    const ggml_tensor * x_b      = nullptr;
    int                 x_mul_idx = -1;
    int                 d_conv   = 4;
    size_t              state_offs = 0;     // bytes of the row before the state of this convolution
    bool                gr_shared  = false; // the GET_ROWS has other readers
    int                 gr_idx   = -1;
    int                 last_cpy = -1;
    int                 conv_idx = -1;
    int                 silu_idx = -1;
    bool                long_path = false; // more than GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS tokens
    ggml_cuda_ssm_conv_state_slots      slots      = {};
    ggml_cuda_ssm_conv_state_long_slots long_slots = {};
};

static bool ggml_cuda_ssm_conv_state_fusion_match(
        ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int concat_idx, ggml_cuda_ssm_conv_state_fusion & f) {
    const ggml_tensor * concat = cgraph->nodes[concat_idx];
    if (concat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(concat, 0) != 0 || concat->type != GGML_TYPE_F32 ||
            concat->ne[2] != 1 || concat->ne[3] != 1 || !cuda_ctx->stream_context().concurrent_events.empty()) {
        return false;
    }
    // the state columns: the GET_ROWS of one cache row through reshapes and views; one fewer than the kernel width
    static const bool no_short_conv = getenv("GGML_CUDA_NO_SHORT_CONV_FUSION") != nullptr;
    const ggml_tensor * st = concat->src[0];
    const int     d_conv = (int) st->ne[0] + 1;
    const int64_t nr     = concat->ne[1];
    const int64_t n_t    = concat->ne[0] - (d_conv - 1);
    if ((d_conv != 4 && (d_conv != 3 || no_short_conv)) || n_t < 1 || n_t > INT32_MAX) {
        return false;
    }
    f = {};
    f.d_conv    = d_conv;
    f.long_path = n_t > GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS;
    if (f.long_path && d_conv != 4) {
        return false;
    }
    if (st->ne[1] != nr || !ggml_is_contiguous(st) || ggml_cuda_graph_use_count(cgraph, st) != 1) {
        return false;
    }
    const ggml_tensor * gr = st;
    while (gr->op == GGML_OP_RESHAPE || gr->op == GGML_OP_VIEW) {
        gr = gr->src[0];
        // the readers of the GET_ROWS are counted below, where its node is known
        f.gr_shared = f.gr_shared || (gr->op != GGML_OP_GET_ROWS && ggml_cuda_graph_use_count(cgraph, gr) != 1);
    }
    if (gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || (gr->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_nrows(gr) != 1 || !ggml_is_contiguous(gr) || st->data == nullptr || gr->data == nullptr ||
            (const char *) st->data < (const char *) gr->data) {
        return false;
    }
    f.state_offs = (const char *) st->data - (const char *) gr->data;
    if (f.state_offs % sizeof(float) != 0 || f.state_offs + ggml_nbytes(st) > ggml_nbytes(gr)) {
        return false;
    }
    f.gr_shared = f.gr_shared || ggml_nbytes(st) != ggml_nbytes(gr);
    f.states = gr->src[0];
    f.ids    = gr->src[1];
    if (f.states->type != GGML_TYPE_F32 || f.states->ne[0] != gr->ne[0] || f.states->nb[0] != sizeof(float) ||
            f.states->nb[1] % sizeof(float) != 0 || f.states->data == nullptr ||
            f.ids->type != GGML_TYPE_I32 || ggml_nelements(f.ids) != 1 || f.ids->data == nullptr) {
        return false;
    }

    // the new tokens, read as x[t*nb0 + c*nb1]
    f.x = concat->src[1];
    if (f.x->type != GGML_TYPE_F32 || f.x->ne[0] != n_t || f.x->ne[1] != nr || f.x->ne[2] != 1 ||
            f.x->nb[0] % sizeof(float) != 0 || f.x->nb[1] % sizeof(float) != 0 || f.x->data == nullptr) {
        return false;
    }
    // x = (a*b)^T with nothing else reading the MUL: the kernel reads a and b instead
    if (d_conv == 3 && !f.long_path && f.x->op == GGML_OP_TRANSPOSE && f.x->src[0]->op == GGML_OP_MUL) {
        const ggml_tensor * m = f.x->src[0];
        const ggml_tensor * a = m->src[0];
        const ggml_tensor * b = m->src[1];
        const auto ok = [&](const ggml_tensor * t) {
            return t->type == GGML_TYPE_F32 && ggml_are_same_shape(t, m) && t->nb[0] == sizeof(float) &&
                t->nb[1] % sizeof(float) == 0 && t->data != nullptr;
        };
        if (ok(a) && ok(b) && a->nb[1] == b->nb[1] && !(m->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_cuda_graph_use_count(cgraph, m) == 1 &&
                ggml_cuda_graph_use_count(cgraph, f.x) == 1) {
            for (int j = concat_idx - 1; j >= 0 && j >= concat_idx - 64; j--) {
                if (cgraph->nodes[j] == m) {
                    f.x_mul_idx = j;
                    break;
                }
            }
            // a and b are read at the concat instead of at the MUL
            for (int j = f.x_mul_idx + 1; f.x_mul_idx >= 0 && j < concat_idx; j++) {
                const ggml_tensor * n = cgraph->nodes[j];
                if (!ggml_cuda_is_view_or_noop(n) && (ggml_cuda_tensors_overlap(n, a) || ggml_cuda_tensors_overlap(n, b))) {
                    f.x_mul_idx = -1;
                }
            }
            if (f.x_mul_idx >= 0) {
                f.x_a = a;
                f.x_b = b;
            }
        }
    }

    // the CPYs of the windows to the slots right after the concat, only views between
    int n_cpy = 0;
    for (int j = concat_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_CPY && n->src[0]->op == GGML_OP_VIEW && n->src[0]->src[0] == concat) {
            const ggml_tensor * win = n->src[0];
            const ggml_tensor * dst = n->src[1];
            const int64_t       i0  = (int64_t) (win->view_offs / sizeof(float));
            if (win->ne[0] != d_conv - 1 || win->ne[1] != nr || win->ne[2] != 1 || win->nb[1] != concat->nb[1] ||
                    win->view_offs % sizeof(float) != 0 || i0 > n_t ||
                    ggml_cuda_graph_use_count(cgraph, win) != 1 || ggml_cuda_graph_use_count(cgraph, n) != 0 ||
                    (n->flags & GGML_TENSOR_FLAG_OUTPUT) || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
                    ggml_nelements(dst) != (d_conv - 1)*nr || dst->data == nullptr || n->data != dst->data) {
                return false;
            }
            if (f.long_path) {
                ggml_cuda_ssm_conv_state_long_slots & ls = f.long_slots;
                for (int k = 0; k < ls.n; ++k) {
                    if (ls.start[k] == (int) i0) {
                        return false;
                    }
                }
                if (ls.n == GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS + 1) {
                    return false;
                }
                ls.start[ls.n] = (int) i0;
                ls.dst  [ls.n] = (float *) dst->data;
                ls.n++;
            } else {
                if (f.slots.dst[i0] != nullptr) {
                    return false;
                }
                f.slots.dst[i0] = (float *) dst->data;
            }
            f.last_cpy      = j;
            n_cpy++;
            continue;
        }
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        break;
    }
    if (n_cpy == 0) {
        return false;
    }

    // the convolution of the concat and its silu, and nothing else reads the concat
    for (int j = f.last_cpy + 1; j < cgraph->n_nodes - 1 && j <= f.last_cpy + 64; ++j) {
        if (cgraph->nodes[j]->op == GGML_OP_SSM_CONV && cgraph->nodes[j]->src[0] == concat) {
            f.conv_idx = j;
            break;
        }
    }
    if (f.conv_idx < 0) {
        return false;
    }
    f.ssm_conv = cgraph->nodes[f.conv_idx];
    f.silu_idx = f.conv_idx + 1;
    const ggml_tensor * silu_src = f.ssm_conv;
    if (cgraph->nodes[f.silu_idx]->op == GGML_OP_RESHAPE && cgraph->nodes[f.silu_idx]->src[0] == f.ssm_conv &&
            f.silu_idx + 1 < cgraph->n_nodes && ggml_cuda_graph_use_count(cgraph, cgraph->nodes[f.silu_idx]) == 1) {
        silu_src = cgraph->nodes[f.silu_idx++];
    }
    f.silu     = cgraph->nodes[f.silu_idx];
    // the LFM2 short conv: a MUL by the gate instead of the SILU
    if (d_conv == 3 && !f.long_path && silu_src == f.ssm_conv && f.silu->op == GGML_OP_MUL) {
        f.gate = f.silu->src[0] == f.ssm_conv ? f.silu->src[1] : f.silu->src[1] == f.ssm_conv ? f.silu->src[0] : nullptr;
        if (f.gate == nullptr || f.gate == f.ssm_conv || f.gate->type != GGML_TYPE_F32 || !ggml_are_same_shape(f.gate, f.ssm_conv) ||
                f.gate->nb[0] != sizeof(float) || f.gate->nb[1] % sizeof(float) != 0 || f.gate->data == nullptr) {
            return false;
        }
    }
    // the width 3 kernel is the LFM2 chain only
    if (d_conv == 3 && (f.gate == nullptr || f.x_a == nullptr)) {
        return false;
    }
    const bool silu_ok = f.gate != nullptr ||
        (f.silu->op == GGML_OP_UNARY && ggml_get_unary_op(f.silu) == GGML_UNARY_OP_SILU && f.silu->src[0] == silu_src);
    const ggml_tensor * w = f.ssm_conv->src[1];
    if (w->type != GGML_TYPE_F32 || w->ne[0] != d_conv || w->ne[1] != nr || w->nb[0] != sizeof(float) ||
            f.ssm_conv->ne[0] != nr || f.ssm_conv->ne[1] != n_t || f.ssm_conv->ne[2] != 1 ||
            ggml_cuda_graph_use_count(cgraph, f.ssm_conv) != 1 || (f.ssm_conv->flags & GGML_TENSOR_FLAG_OUTPUT) || !silu_ok ||
            f.silu->type != GGML_TYPE_F32 || !ggml_are_same_shape(f.silu, f.ssm_conv) || !ggml_is_contiguous(f.silu) ||
            ggml_cuda_graph_use_count(cgraph, concat) != n_cpy + 1) {
        return false;
    }

    // the kernel reads the cache and the ids at the concat, so no node from the GET_ROWS to it may write either (as for
    //   ggml_cuda_gdn_gather_fusion)
    for (int j = concat_idx - 1; j >= 0 && j >= concat_idx - 64; j--) {
        if (cgraph->nodes[j] == gr) {
            f.gr_idx = j;
            break;
        }
    }
    if (f.gr_idx < 0) {
        return false;
    }
    f.gr_shared = f.gr_shared || (gr != st && ggml_cuda_graph_use_count_skip_empty_views(cgraph, f.gr_idx, concat_idx) != 1);
    // a write into the cache within one row may only cover the states of other convolutions (their rollback slots)
    const size_t row_bytes = f.states->nb[1];
    const size_t read_end  = f.state_offs + ggml_nbytes(st);
    auto other_columns = [&](const ggml_tensor * n) {
        if (!ggml_is_contiguous(n) || (const char *) n->data < (const char *) f.states->data) {
            return false;
        }
        const size_t o = (size_t) ((const char *) n->data - (const char *) f.states->data) % row_bytes;
        return o + ggml_nbytes(n) <= row_bytes && (o + ggml_nbytes(n) <= f.state_offs || o >= read_end);
    };
    for (int j = f.gr_idx + 1; j < concat_idx; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (!ggml_cuda_is_view_or_noop(n) && ((ggml_cuda_tensors_overlap(n, f.states) && !other_columns(n)) ||
                                              ggml_cuda_tensors_overlap(n, f.ids))) {
            return false;
        }
    }

    // the SILU output is written at the concat: no node until the SSM_CONV may touch its memory, and it may sit on x only
    //   with the same layout, where each thread reads its channel before it writes it
    for (int j = f.last_cpy + 1; j < f.conv_idx; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        // the gate is read at the concat too
        if (ggml_cuda_tensors_overlap(n, f.silu) || (f.gate != nullptr && ggml_cuda_tensors_overlap(n, f.gate))) {
            return false;
        }
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            if (n->src[k] != nullptr && ggml_cuda_tensors_overlap(n->src[k], f.silu)) {
                return false;
            }
        }
    }
    const bool x_same_layout = f.x->nb[1] == sizeof(float) && f.x->nb[0] == f.silu->nb[1];
    if (f.x_a == nullptr ? ggml_cuda_tensors_overlap(f.silu, f.x) && (f.silu->data != f.x->data || !x_same_layout) :
            ggml_cuda_tensors_overlap(f.silu, f.x_a) || ggml_cuda_tensors_overlap(f.silu, f.x_b) ||
            ggml_cuda_tensors_overlap(f.x_a, f.states) || ggml_cuda_tensors_overlap(f.x_b, f.states)) {
        return false;
    }
    if (f.gate != nullptr && (ggml_cuda_tensors_overlap(f.silu, f.gate) || ggml_cuda_tensors_overlap(f.gate, f.states))) {
        return false;
    }
    if (ggml_cuda_tensors_overlap(f.silu, f.states) || ggml_cuda_tensors_overlap(f.silu, w) ||
            ggml_cuda_tensors_overlap(f.silu, f.ids) || ggml_cuda_tensors_overlap(f.x, f.states)) {
        return false;
    }
    return true;
}

// The chain of build_conv_state, one token, run in the epilogue of the qkv projection at node qkv_last (the last node of
//   that launch): each row of qkv is one channel, so its block has the channel's value and convolves it, as the kernel at
//   the CONCAT would. The SILU output and the slots are then written that many nodes earlier: no node run in between may
//   touch the SILU, the cache row or the ids (the GET_ROWS, the CONCAT and the CPYs do not run), and the SILU may not
//   sit on any input or output of the projection launch. Returns the CONCAT's index, or -1.
static int ggml_cuda_gdn_conv_fold(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int qkv_last,
                                   const ggml_tensor * qkv, const ggml_tensor * const * launch_tensors, const int n_launch_tensors,
                                   ggml_cuda_ssm_conv_state_fusion & f) {
    static const bool disabled = getenv("GGML_CUDA_NO_CONV_FOLD") != nullptr;
    // mul_mat_vec_f takes it for 2-byte float rows (ggml_cuda_mul_mat_vec_f_gdn_qkv), mul_mat_vec_q where it pays
    const int  cc   = ggml_cuda_info().devices[cuda_ctx->device].cc;
    const bool pays = ggml_is_quantized(qkv->src[0]->type) ? ggml_cuda_mmvq_conv_epilogue_pays(qkv->src[0]->type, cc) :
        qkv->src[0]->type == GGML_TYPE_F16 || qkv->src[0]->type == GGML_TYPE_BF16;
    if (disabled || !pays) {
        return -1;
    }
    int concat_idx = -1;
    for (int j = qkv_last + 1; j < cgraph->n_nodes && j <= qkv_last + 32; j++) {
        if (cgraph->nodes[j]->op == GGML_OP_CONCAT) {
            concat_idx = j;
            break;
        }
    }
    if (concat_idx < 0 || !ggml_cuda_ssm_conv_state_fusion_match(cuda_ctx, cgraph, concat_idx, f)) {
        return -1;
    }
    if (f.long_path || f.gr_shared || f.ssm_conv->ne[1] != 1 || f.gr_idx <= qkv_last ||
            f.x->data != qkv->data || f.x->nb[1] != sizeof(float) || f.x->ne[1] != qkv->ne[0] || ggml_nrows(qkv) != 1 ||
            f.ssm_conv->src[1]->ne[0] != 4) {
        return -1;
    }
    for (int s = 2; s <= GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS; s++) {
        if (f.slots.dst[s] != nullptr) {
            return -1;
        }
    }
    // the SILU may sit exactly on qkv when nothing but the views to the CONCAT read qkv: the block of a row writes the
    //   SILU over it last
    bool silu_on_qkv = f.silu->data == qkv->data && ggml_nbytes(f.silu) == ggml_nbytes(qkv) && ggml_is_contiguous(f.silu) &&
        ggml_is_contiguous(qkv) && f.silu->ne[0] == qkv->ne[0] && ggml_cuda_graph_use_count(cgraph, f.x) == 1;
    for (const ggml_tensor * t = f.x; silu_on_qkv && t != qkv; t = t->src[0]) {
        silu_on_qkv = t->view_src != nullptr && t->src[0] != nullptr && ggml_cuda_graph_use_count(cgraph, t->src[0]) == 1;
    }
    for (int a = 0; a < n_launch_tensors; a++) {
        if (ggml_cuda_tensors_overlap(f.silu, launch_tensors[a]) && !(launch_tensors[a] == qkv && silu_on_qkv)) {
            return -1;
        }
    }
    const auto folded = [&](const int j) {
        if (j == f.gr_idx || j == concat_idx) {
            return true;
        }
        const ggml_tensor * n = cgraph->nodes[j];
        return j <= f.last_cpy && n->op == GGML_OP_CPY && n->src[0]->op == GGML_OP_VIEW && n->src[0]->src[0] == cgraph->nodes[concat_idx];
    };
    for (int j = qkv_last + 1; j < f.conv_idx; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n) || folded(j)) {
            continue;
        }
        if (ggml_cuda_tensors_overlap(n, f.silu) || ggml_cuda_tensors_overlap(n, f.ids) ||
                (j <= f.last_cpy && ggml_cuda_tensors_overlap(n, f.states))) {
            return -1;
        }
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            if (n->src[k] != nullptr && (ggml_cuda_tensors_overlap(n->src[k], f.silu) ||
                    (j <= f.last_cpy && ggml_cuda_tensors_overlap(n->src[k], f.states)))) {
                return -1;
            }
        }
    }
    return concat_idx;
}

// whether node i, the GET_ROWS of a convolution state, the MUL of its new tokens or the SSM_CONV after it, belongs to a
//   fused build_conv_state; n_skip: the nodes after the SSM_CONV up to its SILU
static bool ggml_cuda_ssm_conv_state_is_folded(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i,
                                               int * n_skip = nullptr) {
    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return false;
    }
    const ggml_tensor * node = cgraph->nodes[i];
    ggml_cuda_ssm_conv_state_fusion f;
    if (node->op == GGML_OP_SSM_CONV) {
        for (int j = i - 1; j >= 0 && j >= i - 128; j--) {
            if (cgraph->nodes[j] == node->src[0]) {
                if (ggml_cuda_ssm_conv_state_fusion_match(cuda_ctx, cgraph, j, f) && f.conv_idx == i) {
                    if (n_skip) {
                        *n_skip = f.silu_idx - f.conv_idx;
                    }
                    return true;
                }
                return false;
            }
        }
        return false;
    }
    if (node->op == GGML_OP_GET_ROWS) {
        for (int j = i + 1; j < cgraph->n_nodes && j <= i + 64; j++) {
            const ggml_tensor * concat = cgraph->nodes[j];
            if (concat->op != GGML_OP_CONCAT) {
                continue;
            }
            const ggml_tensor * gr = concat->src[0];
            while (gr != node && (gr->op == GGML_OP_RESHAPE || gr->op == GGML_OP_VIEW)) {
                gr = gr->src[0];
            }
            if (gr == node) {
                if (!ggml_cuda_ssm_conv_state_fusion_match(cuda_ctx, cgraph, j, f) || f.gr_idx != i) {
                    return false;
                }
                break;
            }
        }
        if (f.gr_idx != i) {
            return false;
        }
        if (!f.gr_shared) {
            return true;
        }
        // a row of several convolution states (glm5-next's q, k and v): the GET_ROWS does not run either when each of
        //   its readers, through views and reshapes, is a CONCAT that runs fused (they read the cache row themselves).
        //   Every reader of the GET_ROWS and of the views on the way must be among the next 256 nodes.
        std::vector<std::pair<const ggml_tensor *, int>> parents = { { node, 0 } }; // and their readers found
        for (int j = i + 1; j < cgraph->n_nodes && j <= i + 256; j++) {
            const ggml_tensor * n = cgraph->nodes[j];
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (n->src[s] == nullptr) {
                    continue;
                }
                for (size_t p = 0; p < parents.size(); p++) {
                    if (n->src[s] != parents[p].first) {
                        continue;
                    }
                    parents[p].second++;
                    // a view of no elements (build_rs's extra states of a single sequence) reads nothing
                    if (ggml_nelements(n) == 0 && (n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW)) {
                        break;
                    }
                    if (n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW) {
                        parents.push_back({ n, 0 });
                        break;
                    }
                    ggml_cuda_ssm_conv_state_fusion g;
                    if (n->op != GGML_OP_CONCAT || !ggml_cuda_ssm_conv_state_fusion_match(cuda_ctx, cgraph, j, g) || g.gr_idx != i) {
                        return false;
                    }
                    break;
                }
            }
        }
        for (const auto & p : parents) {
            if (p.second != ggml_cuda_graph_use_count(cgraph, p.first)) {
                return false;
            }
        }
        return true;
    }
    // the TRANSPOSE that reads the MUL comes right after it
    if (node->op == GGML_OP_MUL && node->ne[1] <= GGML_CUDA_SSM_CONV_STATE_MAX_TOKENS && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_TRANSPOSE && cgraph->nodes[i + 1]->src[0] == node) {
        for (int j = i + 2; j < cgraph->n_nodes && j <= i + 64; j++) {
            const ggml_tensor * concat = cgraph->nodes[j];
            if (concat->op == GGML_OP_CONCAT && concat->src[1]->op == GGML_OP_TRANSPOSE && concat->src[1]->src[0] == node) {
                return ggml_cuda_ssm_conv_state_fusion_match(cuda_ctx, cgraph, j, f) && f.x_mul_idx == i;
            }
        }
    }
    return false;
}

struct ggml_cuda_rwkv7_output_fusion {
    const ggml_tensor * norm        = nullptr;
    const ggml_tensor * norm_weight = nullptr;
    const ggml_tensor * norm_bias   = nullptr;
    const ggml_tensor * k           = nullptr;
    const ggml_tensor * r           = nullptr;
    const ggml_tensor * r_k         = nullptr;
    const ggml_tensor * v           = nullptr;
    const ggml_tensor * gate        = nullptr;
    ggml_tensor *       dst         = nullptr;
};

static const ggml_tensor * ggml_cuda_other_src(const ggml_tensor * node, const ggml_tensor * src) {
    if (node->src[0] == src) {
        return node->src[1];
    }
    if (node->src[1] == src) {
        return node->src[0];
    }
    return nullptr;
}

static int ggml_cuda_try_rwkv7_output_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_rwkv7_output_fusion & fusion) {
    static const ggml_op ops[] = {
        GGML_OP_NORM,
        GGML_OP_RESHAPE,
        GGML_OP_MUL,
        GGML_OP_ADD,
        GGML_OP_MUL,
        GGML_OP_RESHAPE,
        GGML_OP_MUL,
        GGML_OP_SUM_ROWS,
        GGML_OP_MUL,
        GGML_OP_RESHAPE,
        GGML_OP_ADD,
    };

    constexpr int node_count = sizeof(ops) / sizeof(ops[0]);
    if (node_idx + node_count > cgraph->n_nodes) {
        return 0;
    }

    bool has_gate = false;
    if (node_idx + node_count < cgraph->n_nodes) {
        const ggml_tensor * maybe_gate = cgraph->nodes[node_idx + node_count];
        has_gate = maybe_gate->op == GGML_OP_MUL &&
                   (maybe_gate->src[0] == cgraph->nodes[node_idx + node_count - 1] ||
                    maybe_gate->src[1] == cgraph->nodes[node_idx + node_count - 1]);
    }

    std::array<ggml_op, node_count + 1> matched_ops;
    std::copy(std::begin(ops), std::end(ops), matched_ops.begin());
    if (has_gate) {
        matched_ops[node_count] = GGML_OP_MUL;
    }

    const int matched_count = node_count + (has_gate ? 1 : 0);
    const int out_idx       = node_idx + matched_count - 1;

    // op / compute-flag / output-flag / use-count checks, plus the view_src
    // containment check that keeps an elided reshape from aliasing a tensor
    // produced outside the fused range
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, matched_count, matched_ops.data(), &out_idx, 1)) {
        return 0;
    }

    const ggml_tensor * norm         = cgraph->nodes[node_idx + 0];
    const ggml_tensor * reshape_norm = cgraph->nodes[node_idx + 1];
    const ggml_tensor * mul_norm     = cgraph->nodes[node_idx + 2];
    const ggml_tensor * add_norm     = cgraph->nodes[node_idx + 3];
    const ggml_tensor * mul_kr       = cgraph->nodes[node_idx + 4];
    const ggml_tensor * reshape_r_k  = cgraph->nodes[node_idx + 5];
    const ggml_tensor * mul_krrk     = cgraph->nodes[node_idx + 6];
    const ggml_tensor * sum_rk       = cgraph->nodes[node_idx + 7];
    const ggml_tensor * mul_v        = cgraph->nodes[node_idx + 8];
    const ggml_tensor * reshape_v    = cgraph->nodes[node_idx + 9];
    const ggml_tensor * add_out      = cgraph->nodes[node_idx + 10];
    const ggml_tensor * gate_out     = has_gate ? cgraph->nodes[node_idx + 11] : nullptr;

    if (reshape_norm->src[0] != norm || add_norm->src[0] != mul_norm || reshape_r_k->src[0] == nullptr ||
        sum_rk->src[0] != mul_krrk || reshape_v->src[0] != mul_v) {
        return 0;
    }

    const ggml_tensor * norm_weight = ggml_cuda_other_src(mul_norm, reshape_norm);
    const ggml_tensor * norm_bias   = ggml_cuda_other_src(add_norm, mul_norm);
    const ggml_tensor * r_k_view    = ggml_cuda_other_src(mul_krrk, mul_kr);
    const ggml_tensor * v           = ggml_cuda_other_src(mul_v, sum_rk);
    if (norm_weight == nullptr || norm_bias == nullptr || r_k_view != reshape_r_k || v == nullptr ||
        ggml_cuda_other_src(add_out, add_norm) != reshape_v) {
        return 0;
    }

    const ggml_tensor * gate = nullptr;
    ggml_tensor *       dst  = const_cast<ggml_tensor *>(add_out);
    if (has_gate) {
        gate = ggml_cuda_other_src(gate_out, add_out);
        if (gate == nullptr) {
            return 0;
        }
        dst = const_cast<ggml_tensor *>(gate_out);
    }

    const ggml_tensor * x   = norm->src[0];
    const ggml_tensor * k   = mul_kr->src[0];
    const ggml_tensor * r   = mul_kr->src[1];
    const ggml_tensor * r_k = reshape_r_k->src[0];

    // Every value read by the fused kernel must already exist before this
    // range is executed. In particular, do not reinterpret an elided compute
    // node as a weight or activation input merely because the op sequence and
    // element counts happen to match.
    auto is_elided_node = [&](const ggml_tensor * tensor) {
        for (int i = 0; i < matched_count - 1; ++i) {
            if (tensor == cgraph->nodes[node_idx + i]) {
                return true;
            }
        }
        return false;
    };
    if (is_elided_node(x) || is_elided_node(norm_weight) || is_elided_node(norm_bias) || is_elided_node(k) ||
        is_elided_node(r) || is_elided_node(r_k) || is_elided_node(v) || (gate != nullptr && is_elided_node(gate))) {
        return 0;
    }

    if (x == nullptr || k == nullptr || r == nullptr || x->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 ||
        norm_weight->type != GGML_TYPE_F32 || norm_bias->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 ||
        r->type != GGML_TYPE_F32 || r_k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        (gate != nullptr && gate->type != GGML_TYPE_F32) || dst->type != GGML_TYPE_F32) {
        return 0;
    }

    const int64_t head_size = x->ne[0];
    const int64_t H         = x->ne[1];
    const int64_t C         = head_size * H;
    const int64_t n_tokens  = ggml_nelements(x) / C;

    auto has_shape = [](const ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        return tensor->ne[0] == ne0 && tensor->ne[1] == ne1 && tensor->ne[2] == ne2 && tensor->ne[3] == ne3;
    };

    if ((head_size != 64 && head_size != 128) || !ggml_are_same_shape(x, norm) || !ggml_are_same_shape(x, mul_kr) ||
        !ggml_are_same_shape(x, k) || !ggml_are_same_shape(x, r) || !ggml_are_same_shape(x, v) ||
        !has_shape(x, head_size, H, n_tokens, 1) || !has_shape(reshape_norm, C, n_tokens, 1, 1) ||
        !has_shape(mul_norm, C, n_tokens, 1, 1) || !has_shape(add_norm, C, n_tokens, 1, 1) ||
        !has_shape(norm_weight, C, 1, 1, 1) || !has_shape(norm_bias, C, 1, 1, 1) ||
        !has_shape(reshape_r_k, head_size, H, 1, 1) || !has_shape(r_k, C, 1, 1, 1) ||
        !has_shape(mul_krrk, head_size, H, n_tokens, 1) || !has_shape(sum_rk, 1, H, n_tokens, 1) ||
        !has_shape(mul_v, head_size, H, n_tokens, 1) || !has_shape(reshape_v, C, n_tokens, 1, 1) ||
        !has_shape(add_out, C, n_tokens, 1, 1) || !has_shape(dst, C, n_tokens, 1, 1) ||
        (gate != nullptr && !has_shape(gate, C, n_tokens, 1, 1))) {
        return 0;
    }

    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(norm_weight) || !ggml_is_contiguous(norm_bias) ||
        !ggml_is_contiguous(k) || !ggml_is_contiguous(r) || !ggml_is_contiguous(r_k) || !ggml_is_contiguous(v) ||
        (gate != nullptr && !ggml_is_contiguous(gate)) || !ggml_is_contiguous(dst)) {
        return 0;
    }

    fusion.norm        = norm;
    fusion.norm_weight = norm_weight;
    fusion.norm_bias   = norm_bias;
    fusion.k           = k;
    fusion.r           = r;
    fusion.r_k         = r_k;
    fusion.v           = v;
    fusion.gate        = gate;
    fusion.dst         = dst;
    return matched_count - 1;
}

// The group selection of grouped routing (DeepSeek-V3, see build_moe_ffn) between the bias ADD and the expert ARGSORT
static const ggml_op ggml_cuda_topk_moe_group_ops[] = {
    GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_ARGSORT,  GGML_OP_VIEW,     GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SUM_ROWS,
    GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW,     GGML_OP_GET_ROWS, GGML_OP_FILL,     GGML_OP_SET_ROWS, GGML_OP_RESHAPE,
};
static constexpr int ggml_cuda_topk_moe_n_group_ops = (int) (sizeof(ggml_cuda_topk_moe_group_ops) / sizeof(ggml_op));

// Matches the group selection at node k (after the bias ADD) and sets args.n_groups, n_group_used
static bool ggml_cuda_topk_moe_match_groups(ggml_tensor ** nodes, const int n_nodes, const int k, ggml_cuda_topk_moe_args & args) {
    static const bool disabled = getenv("GGML_CUDA_NO_TOPK_GROUPS") != nullptr;
    if (disabled || k < 1 || k + ggml_cuda_topk_moe_n_group_ops > n_nodes) {
        return false;
    }
    for (int j = 0; j < ggml_cuda_topk_moe_n_group_ops; j++) {
        if (nodes[k + j]->op != ggml_cuda_topk_moe_group_ops[j]) {
            return false;
        }
    }
    const ggml_tensor * add      = nodes[k - 1];
    const ggml_tensor * groups   = nodes[k];       // [G, n_groups, n_tokens]
    const ggml_tensor * groups4  = nodes[k + 1];   // [1, G, n_groups, n_tokens]
    const ggml_tensor * sort2    = nodes[k + 2];
    const ggml_tensor * top2     = nodes[k + 3];
    const ggml_tensor * top2_val = nodes[k + 4];
    const ggml_tensor * score    = nodes[k + 7];   // [n_groups, n_tokens]
    const ggml_tensor * sort_g   = nodes[k + 8];
    const ggml_tensor * top_g    = nodes[k + 9];   // [n_group_used, n_tokens]
    const ggml_tensor * kept     = nodes[k + 10];
    const ggml_tensor * fill     = nodes[k + 11];
    const ggml_tensor * masked   = nodes[k + 12];
    const ggml_tensor * masked2  = nodes[k + 13];  // [n_expert, n_tokens]

    const bool chain = groups->src[0] == add && groups4->src[0] == groups && sort2->src[0] == groups &&
        top2->src[0] == sort2 && top2_val->src[0] == groups4 && top2_val->src[1] == top2 &&
        nodes[k + 5]->src[0] == top2_val && nodes[k + 6]->src[0] == nodes[k + 5] && score->src[0] == nodes[k + 6] &&
        sort_g->src[0] == score && top_g->src[0] == sort_g && kept->src[0] == groups && kept->src[1] == top_g &&
        fill->src[0] == groups && masked->src[0] == kept && masked->src[1] == top_g && masked->src[2] == fill &&
        masked2->src[0] == masked;
    if (!chain || ggml_get_op_params_i32(sort2, 0) != GGML_SORT_ORDER_DESC ||
            ggml_get_op_params_i32(sort_g, 0) != GGML_SORT_ORDER_DESC || ggml_get_op_params_f32(fill, 0) != -INFINITY ||
            top2->ne[0] != 2 || top2->view_offs != 0 || top_g->view_offs != 0 || top_g->type != GGML_TYPE_I32 ||
            masked2->ne[0] != add->ne[0] || groups->ne[0] * groups->ne[1] != add->ne[0] ||
            !ggml_cuda_topk_moe_groups_supported(add->ne[0], groups->ne[1], top_g->ne[0])) {
        return false;
    }
    args.n_groups     = groups->ne[1];
    args.n_group_used = top_g->ne[0];
    return true;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.n_groups        = 1;
    args.n_group_used    = 1;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
            // ADD -> group selection -> RESHAPE
            if (node_idx < n_nodes && nodes[node_idx]->op == GGML_OP_RESHAPE) {
                if (!ggml_cuda_topk_moe_match_groups(nodes, n_nodes, node_idx, args)) {
                    return false;
                }
                node_idx += ggml_cuda_topk_moe_n_group_ops;
            }
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        // the kept groups must hold the used experts
        if (args.n_groups > 1 &&
                nodes[node_idx]->ne[0] * args.n_groups > (int64_t) args.n_group_used * nodes[node_idx - 1]->ne[0]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation; src_may_alias is an input the fused op
//   has finished reading before it writes
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false,
                                                 const ggml_tensor * src_may_alias = nullptr) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias || src == src_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes, more than ggml_can_fuse_subgraph() takes from k = 16 on, so the match passes its
// own node list to ggml_can_fuse_subgraph_ext(). k is bounded by the kernel's shared arrays.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 32;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    std::vector<int> idxs(node_count);
    for (int i = 0; i < node_count; ++i) {
        idxs[i] = node_idx + i;
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs.data(), node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}

// an elementwise kernel writes element i right after it read element i, so its output may share an input's memory
//   only element for element: same address and strides, or no overlap at all
static bool ggml_cuda_elementwise_out_ok(const ggml_tensor * dst, const ggml_tensor * src) {
    return !ggml_cuda_tensors_overlap(dst, src) || (dst->data == src->data && ggml_are_same_stride(dst, src));
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1, false,
                                                        ggml_cuda_glu_activations_may_alias(ffn_up));
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1, false,
                                                        ggml_cuda_glu_activations_may_alias(ffn_up));
        }
    }

    std::initializer_list<enum ggml_op> get_rows_mean_norm_rope_ops = { GGML_OP_GET_ROWS_MEAN, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ROPE };

    if (is_equal(get_rows_mean_norm_rope_ops, ops)) {
        if (ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 5 }) &&
            ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}, {5, 0, 4}}) &&
            ggml_cuda_should_fuse_get_rows_mean_norm_rope(cgraph->nodes[node_idx], cgraph->nodes[node_idx + 2],
                cgraph->nodes[node_idx + 3], cgraph->nodes[node_idx + 5])) {
            int out_nodes[] = { node_idx + 5 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows, /*multi =*/ true)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    // DeepSeek V4: the kv norm reshaped into one head before its rope, and the q heads normed without a weight
    std::initializer_list<enum ggml_op> rms_norm_mul_reshape_rope_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_rope_ops             = { GGML_OP_RMS_NORM, GGML_OP_ROPE };

    if (is_equal(rms_norm_mul_reshape_rope_ops, ops)) {
        if (ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3 }) &&
            ggml_check_edges(cgraph, node_idx, {{2, 0, 1}, {3, 0, 2}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(cgraph->nodes[node_idx], cgraph->nodes[node_idx + 1],
                cgraph->nodes[node_idx + 3], cgraph->nodes[node_idx + 2])) {
            int out_nodes[] = { node_idx + 3 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    if (is_equal(rms_norm_rope_ops, ops)) {
        if (ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 1 }) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(cgraph->nodes[node_idx], nullptr, cgraph->nodes[node_idx + 1])) {
            int out_nodes[] = { node_idx + 1 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rms_norm_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_set_rows_ops, ops)) {
        if (ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 }) &&
            ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}}) &&
            ggml_cuda_should_fuse_rms_norm_set_rows(cgraph->nodes[node_idx], cgraph->nodes[node_idx + 1], cgraph->nodes[node_idx + 2])) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE &&
            ops.begin()[2] == GGML_OP_MUL) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];
        const ggml_tensor * mul      = cgraph->nodes[node_idx+2];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        if (scale->type != GGML_TYPE_F32 || ggml_get_op_params_f32(scale, 1) != 0.0f) {
            return false;
        }
        if (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            return false;
        }
        // as for rms_norm + mul: the weight broadcasts over the norm, not the norm over it, and rows are contiguous
        if (scale == mul->src[1] && !ggml_are_same_shape(mul->src[0], scale)) {
            return false;
        }
        return ggml_is_contiguous_rows(mul->src[0]) && ggml_is_contiguous_rows(mul->src[1]);
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        if (!ggml_cuda_elementwise_out_ok(mul, unary->src[0]) || !ggml_cuda_elementwise_out_ok(mul, other)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0]) || !ggml_cuda_elementwise_out_ok(sqr, unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        // the kernel indexes the input as one flat array
        if (!ggml_is_contiguous(scale->src[0]) || !ggml_cuda_elementwise_out_ok(scale2, scale->src[0])) {
            return false;
        }

        return true;
    }

    return false;
}

typedef void (*ggml_cuda_mul_mat_vec_t)(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
    const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);

// The GEMV that ggml_cuda_mul_mat would run each of the alpha and beta projections of a gated delta net layer on,
//   if it can also run both at once: then the fused result is the one of each alone.
static ggml_cuda_mul_mat_vec_t ggml_cuda_gdn_gate_mul_mat_vec(const ggml_tensor * mm_a, const ggml_tensor * mm_b) {
    if (mm_a->ne[1] == 1) {
        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_a) && ggml_cuda_should_fuse_mul_mat_vec_f(mm_b)) {
            return ggml_cuda_mul_mat_vec_f;
        }
        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_a) && ggml_cuda_should_fuse_mul_mat_vec_q(mm_b)) {
            return ggml_cuda_mul_mat_vec_q;
        }
        return nullptr;
    }

    const ggml_tensor * w   = mm_a->src[0];
    const ggml_tensor * y   = mm_a->src[1];
    const int64_t      ntok = mm_a->ne[1];
    const int          cc   = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int     warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    if (y->type != GGML_TYPE_F32 || ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            ggml_get_op_params_i32(mm_a, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        return nullptr;
    }
    // Several tokens only on mul_mat_vec_f: mul_mat_vec_q with a gate unrolls twice the dot products per column, its
    //   K loop of a few iterations then waits on instruction fetch, and two plain launches win (q8_0 and q4_K with 8
    //   rows at ubatch 4: 1.4-1.8% faster per ubatch unfused).
    if (ggml_cuda_should_use_mmvf(w->type, cc, warp_size, w->ne, w->nb, ntok) && ggml_cuda_mmvf_fuses_tokens(w, y, mm_b->src[0])) {
        return ggml_cuda_mul_mat_vec_f;
    }
    return nullptr;
}

// try and fuse nodes and return the number of nodes to skip
// MUL_MAT(alpha) RESHAPE ADD(dt) SOFTPLUS MUL(a) RESHAPE MUL_MAT(beta) RESHAPE SIGMOID
static ggml_cuda_mul_mat_vec_t ggml_cuda_should_fuse_gdn_gate(const ggml_cgraph * cgraph, int i) {
    if (i + 8 >= cgraph->n_nodes) {
        return nullptr;
    }
    const ggml_tensor * const * n = cgraph->nodes + i;
    const ggml_tensor * mm_a = n[0];
    const ggml_tensor * mm_b = n[6];
    if (mm_a->op != GGML_OP_MUL_MAT || n[1]->op != GGML_OP_RESHAPE || n[2]->op != GGML_OP_ADD ||
            n[3]->op != GGML_OP_UNARY || ggml_get_unary_op(n[3]) != GGML_UNARY_OP_SOFTPLUS || n[4]->op != GGML_OP_MUL ||
            n[5]->op != GGML_OP_RESHAPE || mm_b->op != GGML_OP_MUL_MAT || n[7]->op != GGML_OP_RESHAPE ||
            n[8]->op != GGML_OP_UNARY || ggml_get_unary_op(n[8]) != GGML_UNARY_OP_SIGMOID) {
        return nullptr;
    }
    if (n[1]->src[0] != mm_a || (n[2]->src[0] != n[1] && n[2]->src[1] != n[1]) || n[3]->src[0] != n[2] ||
            (n[4]->src[0] != n[3] && n[4]->src[1] != n[3]) || n[5]->src[0] != n[4] ||
            mm_b->src[1] != mm_a->src[1] || n[7]->src[0] != mm_b || n[8]->src[0] != n[7]) {
        return nullptr;
    }
    // one matrix of tokens, the same weights layout, default precision so that both sum like the GEMV of each alone
    const ggml_tensor * wa = mm_a->src[0];
    const ggml_tensor * wb = mm_b->src[0];
    const int64_t nrows = mm_a->ne[0];
    if (mm_a->ne[2] != 1 || mm_a->ne[3] != 1 || !ggml_are_same_shape(mm_a, mm_b) || wa->type != wb->type ||
            !ggml_are_same_shape(wa, wb) || !ggml_are_same_stride(wa, wb) ||
            ggml_prec(mm_a->op_params[0]) != GGML_PREC_DEFAULT || ggml_prec(mm_b->op_params[0]) != GGML_PREC_DEFAULT) {
        return nullptr;
    }
    // dt and a one per row, the gate and beta laid out like the GEMV results
    const ggml_tensor * dt = n[2]->src[0] == n[1] ? n[2]->src[1] : n[2]->src[0];
    const ggml_tensor * a  = n[4]->src[0] == n[3] ? n[4]->src[1] : n[4]->src[0];
    for (const ggml_tensor * t : { dt, a, n[4], n[8] }) {
        const bool per_row = t == dt || t == a;
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t) != (per_row ? nrows : ggml_nelements(mm_a)) ||
                (per_row && t->ne[0] != nrows)) {
            return nullptr;
        }
    }
    if (!ggml_are_same_shape(n[2], n[1]) || !ggml_are_same_shape(n[4], n[1])) {
        return nullptr;
    }
    // every block reads all of x, and ggml-alloc may give beta the memory of x once the beta GEMV is allocated
    const ggml_tensor * x = mm_a->src[1];
    for (const ggml_tensor * out : { n[4], n[8] }) {
        for (const ggml_tensor * in : { x, dt, a }) {
            if (ggml_cuda_tensors_overlap(out, in)) {
                return nullptr;
            }
        }
    }
    ggml_cuda_mul_mat_vec_t mul_mat_vec = ggml_cuda_gdn_gate_mul_mat_vec(mm_a, mm_b);
    if (!mul_mat_vec) {
        return nullptr;
    }
    const int outputs[2] = { i + 5, i + 8 };
    const enum ggml_op ops[9] = { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL,
        GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY };
    return ggml_can_fuse_subgraph(cgraph, i, 9, ops, outputs, 2) ? mul_mat_vec : nullptr;
}

// The Walsh-Hadamard rotation of a quantized K/V cache (llama_mul_mat_hadamard: RESHAPE, MUL_MAT with the Hadamard
//   hint, RESHAPE) is one launch of its own four times per layer. It goes into the row kernel of what produces the
//   rotated rows, together with the SET_ROWS that stores them when that follows:
//     [RMS_NORM [MUL] [RESHAPE]] [ROPE] RESHAPE MUL_MAT(H) [RESHAPE VIEW SET_ROWS]
//     MUL_MAT(H) RESHAPE VIEW SET_ROWS (any other producer, e.g. the V projection)
//   and the rotation of the attention output into the q8_1 quantization of the MUL_MAT that reads it:
//     MUL_MAT(H) RESHAPE MUL_MAT, when that runs on mul_mat_vec_q.
// The transform is done as fwht_cuda does it and the rest by the same arithmetic, so the result is bitwise the same.
// Returns the number of nodes to skip, 0 when nothing was fused.
static int ggml_cuda_try_fuse_attn_rot(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    const auto node_at = [&](const int k) -> ggml_tensor * {
        return k < cgraph->n_nodes ? cgraph->nodes[k] : nullptr;
    };
    const auto is_rot = [](const ggml_tensor * t) {
        return t != nullptr && t->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD;
    };

    ggml_tensor * rms_norm     = nullptr;
    ggml_tensor * mul          = nullptr;
    ggml_tensor * reshape_norm = nullptr;
    ggml_tensor * rope         = nullptr;
    ggml_tensor * prev         = nullptr;

    int k = i;
    if (node_at(k)->op == GGML_OP_RMS_NORM) {
        rms_norm = prev = node_at(k++);
        ggml_tensor * t = node_at(k);
        if (t != nullptr && t->op == GGML_OP_MUL && (t->src[0] == prev || t->src[1] == prev)) {
            mul = prev = node_at(k++);
        }
        // the DeepSeek V4 kv norm, reshaped into one head before its rope
        t = node_at(k);
        ggml_tensor * t1 = node_at(k + 1);
        if (t != nullptr && t1 != nullptr && t->op == GGML_OP_RESHAPE && t->src[0] == prev && t1->op == GGML_OP_ROPE && t1->src[0] == t) {
            reshape_norm = prev = node_at(k++);
        }
    }
    if (node_at(k) != nullptr && node_at(k)->op == GGML_OP_ROPE && (prev == nullptr || node_at(k)->src[0] == prev)) {
        rope = prev = node_at(k++);
    }

    ggml_tensor * rot = nullptr;
    if (prev != nullptr) {
        ggml_tensor * reshape = node_at(k);
        ggml_tensor * mm      = node_at(k + 1);
        if (reshape == nullptr || reshape->op != GGML_OP_RESHAPE || reshape->src[0] != prev || !is_rot(mm) || mm->src[1] != reshape) {
            return 0;
        }
        rot = mm;
        k += 2;
    } else {
        if (!is_rot(node_at(k))) {
            return 0;
        }
        rot = node_at(k++);
    }

    // what ggml_cuda_op_fwht does: n from 64 to 512 values of contiguous f32
    const ggml_tensor * rot_in = rot->src[1];
    const int64_t n_rot = rot_in->ne[0];
    if (rot_in->type != GGML_TYPE_F32 || rot->type != GGML_TYPE_F32 || !ggml_is_contiguous(rot_in) || !ggml_is_contiguous(rot) ||
            !ggml_are_same_shape(rot, rot_in) || n_rot < 64 || n_rot > 512 || (n_rot & (n_rot - 1)) != 0) {
        return 0;
    }

    ggml_tensor * set_rows = nullptr;
    ggml_tensor * rows_out = nullptr;
    ggml_tensor * view     = nullptr;
    {
        ggml_tensor * reshape = node_at(k);
        ggml_tensor * v       = node_at(k + 1);
        ggml_tensor * sr      = node_at(k + 2);
        if (reshape != nullptr && v != nullptr && sr != nullptr && reshape->op == GGML_OP_RESHAPE && reshape->src[0] == rot &&
                v->op == GGML_OP_VIEW && v->src[0] == reshape && sr->op == GGML_OP_SET_ROWS && sr->src[0] == v) {
            rows_out = reshape;
            view     = v;
            set_rows = sr;
            k += 3;
        }
    }

    if (prev == nullptr && set_rows == nullptr) {
        // the attention output: MUL_MAT(H) RESHAPE... MUL_MAT
        ggml_op ops[8] = { GGML_OP_MUL_MAT };
        int n_ops = 1;
        const ggml_tensor * y = rot;
        while (n_ops < 7 && node_at(i + n_ops) != nullptr && node_at(i + n_ops)->op == GGML_OP_RESHAPE && node_at(i + n_ops)->src[0] == y) {
            y = node_at(i + n_ops);
            ops[n_ops++] = GGML_OP_RESHAPE;
        }
        ggml_tensor * mm = node_at(i + n_ops);
        if (mm == nullptr || mm->op != GGML_OP_MUL_MAT || mm->src[1] != y || is_rot(mm)) {
            return 0;
        }
        ops[n_ops++] = GGML_OP_MUL_MAT;

        const ggml_tensor * w = mm->src[0];
        if (!ggml_is_quantized(w->type) || n_rot > CUDA_QUANTIZE_BLOCK_SIZE || y->ne[0] % n_rot != 0 || !ggml_is_contiguous(y) ||
                ggml_cuda_mul_mat_choose(*cuda_ctx, w, y, mm) != GGML_CUDA_MUL_MAT_PATH_MMVQ) {
            return 0;
        }
        int out_nodes[] = { i + n_ops - 1 };
        if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
            return 0;
        }
        // the unrotated input with the shape of the MUL_MAT's
        ggml_tensor src1 = *y;
        src1.data = rot_in->data;
        ggml_cuda_mul_mat_vec_q_rot(*cuda_ctx, w, &src1, mm, (int) n_rot);
        return n_ops - 1;
    }

    // the rows the kernel works on: those of the rope, else of the norm, else of the rotated result
    const ggml_tensor * rows = rope ? rope : mul ? mul : rms_norm ? rms_norm : rows_out;
    if (rows == nullptr || rows->ne[0] % n_rot != 0 || rows->ne[0] > 8192 ||
            rows->ne[2] > UINT16_MAX || rows->ne[3] > UINT16_MAX) {
        return 0;
    }
    if (rope != nullptr && rms_norm != nullptr) {
        if (!ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope, reshape_norm)) {
            return 0;
        }
    } else if (rope != nullptr) {
        if (!ggml_cuda_rope_fusable_rows(rope) || !ggml_is_contiguous_rows(rope->src[0])) {
            return 0;
        }
    } else if (rms_norm != nullptr) {
        if (reshape_norm != nullptr || !ggml_cuda_rms_norm_mul_fusable_rows(rms_norm, mul)) {
            return 0;
        }
    }

    if (set_rows != nullptr) {
        // one cache row per token, the rows of all heads of a token side by side in it
        const ggml_tensor * idx = set_rows->src[1];
        if (!ggml_are_same_shape(rows_out, rows) || rows->ne[3] != 1 || !ggml_is_contiguous(view) ||
                view->ne[0] != rows->ne[0]*rows->ne[1] || view->ne[1] != rows->ne[2] || view->ne[2] != 1 || view->ne[3] != 1 ||
                !ggml_cuda_rms_norm_mul_rope_fused_dst_supported(set_rows->type) || set_rows->ne[0] != view->ne[0] ||
                set_rows->ne[2] != 1 || set_rows->ne[3] != 1 || set_rows->nb[0] != ggml_type_size(set_rows->type) ||
                (ggml_is_quantized(set_rows->type) && rows->ne[0] % ggml_blck_size(set_rows->type) != 0) ||
                idx->type != GGML_TYPE_I64 || idx->ne[0] != view->ne[1] || ggml_nrows(idx) != 1) {
            return 0;
        }
    }

    ggml_op ops[16];
    const int n_ops = k - i;
    GGML_ASSERT(n_ops <= 16);
    for (int j = 0; j < n_ops; ++j) {
        ops[j] = cgraph->nodes[i + j]->op;
    }
    int out_nodes[] = { k - 1 };
    if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1)) {
        return 0;
    }
    if (!ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
        // the rotated rows may take exactly the memory of the rows they are made from, as they often do: each block
        //   reads its whole row before it writes the same row of the result. Any other overlap is not allowed.
        const ggml_tensor * x   = rms_norm ? rms_norm->src[0] : rope ? rope->src[0] : rot_in;
        const ggml_tensor * dst = cgraph->nodes[k - 1];
        if (set_rows != nullptr || dst->data != x->data || !ggml_is_contiguous(x) || ggml_nbytes(x) != ggml_nbytes(dst)) {
            return 0;
        }
        const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            const char * a0 = (const char *) a->data;
            const char * b0 = (const char *) b->data;
            return a0 < b0 + ggml_backend_buft_get_alloc_size(b->buffer->buft, b) &&
                   b0 < a0 + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        };
        for (int j = i; j < k; ++j) {
            for (const ggml_tensor * src : cgraph->nodes[j]->src) {
                if (src == nullptr || src->op == GGML_OP_NONE || src->data == x->data) {
                    continue;
                }
                bool intermediate = false;
                for (int jj = i; jj < j; ++jj) {
                    intermediate = intermediate || cgraph->nodes[jj] == src;
                }
                if (!intermediate && overlap(dst, src)) {
                    return 0;
                }
            }
        }
    }

    ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, rms_norm, mul, rope, set_rows, rot, rows);
    return n_ops - 1;
}

// The nodes of a fused topk_moe starting at node i, as ggml_cuda_try_fuse runs it: false if they do not match
struct ggml_cuda_topk_moe_plan {
    int                      n_ops = 0;
    int                      out_nodes[2] = { -1, -1 }; // nodes which can't be elided: ids, weights
    const ggml_tensor *      logits  = nullptr;
    ggml_tensor *            weights = nullptr;
    ggml_tensor *            ids     = nullptr;
    const ggml_tensor *      bias    = nullptr;
    const ggml_tensor *      clamp   = nullptr;
    const ggml_tensor *      scale   = nullptr;
    ggml_cuda_topk_moe_args  args;
};

static bool ggml_cuda_topk_moe_plan_at(const ggml_cgraph * cgraph, const int i, ggml_cuda_topk_moe_plan & p) {
    const ggml_tensor * node = cgraph->nodes[i];
    p = {};
    if (!ggml_cuda_topk_moe_fusion(cgraph, i, p.args)) {
        return false;
    }
    const ggml_cuda_topk_moe_args & args = p.args;
    std::vector<ggml_op> ops;
    ops.reserve(13 + ggml_cuda_topk_moe_n_group_ops);  // max ops; avoids gcc -Wstringop-overflow false positive
    p.logits = node->src[0];

    if (!args.delayed_softmax) {
        if (args.sigmoid) {
            ops.insert(ops.end(), { GGML_OP_UNARY });
        } else if (args.sqrt_softplus) {
            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
        } else {
            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
        }
        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

        if (args.prob_bias) {
            p.bias = cgraph->nodes[i_probs + 2]->src[1];
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD });
            if (args.n_groups > 1) {
                ops.insert(ops.end(), std::begin(ggml_cuda_topk_moe_group_ops), std::end(ggml_cuda_topk_moe_group_ops));
            }
            ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
            p.out_nodes[0] = i + (int) ops.size() - 2;
        } else {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
            p.out_nodes[0] = i_probs + 3;
        }
        p.ids = cgraph->nodes[p.out_nodes[0]];

        if (args.norm) {
            ops.insert(ops.end(),
                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            p.clamp = cgraph->nodes[i + ops.size() - 3];
        }
        if (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            p.scale = cgraph->nodes[i + ops.size() - 1];
        }

        p.weights      = cgraph->nodes[i + ops.size() - 1];
        p.out_nodes[1] = i + ops.size() - 1;
        p.n_ops        = (int) ops.size();

        return ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), p.out_nodes, 2) &&
            ggml_cuda_should_use_topk_moe(node, p.logits, p.weights, p.ids);
    }
    if (!args.norm && !args.prob_bias) {
        //special case gpt-oss, no norm, no bias.
        ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
        p.weights                   = cgraph->nodes[i + 5];
        p.ids                       = cgraph->nodes[i + 1];
        const ggml_tensor * softmax = cgraph->nodes[i + 4];

        p.out_nodes[0] = i + 1;
        p.out_nodes[1] = i + 5;
        p.n_ops        = (int) ops.size();
        return ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), p.out_nodes, 2) &&
            ggml_cuda_should_use_topk_moe(softmax, p.logits, p.weights, p.ids);
    }
    return false;
}

// The outputs of the fused topk_moe may lie over the logits: ggml-alloc puts them there once nothing else reads the
// logits, as the alloc deps of graph_optimize are not made under the meta backend. Up to TOPK_MOE_ROWS_PER_BLOCK rows
// one block reads all logits before it writes; with more, the other blocks would read logits already overwritten, so
// they read a copy (a prefill batch otherwise took the unfused argsort path, ~0.1 ms per layer at 2048 tokens).
static bool ggml_cuda_topk_moe_launch(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i,
        const int n_ops, const int * out_nodes, const ggml_tensor * logits, ggml_tensor * weights, ggml_tensor * ids,
        const ggml_tensor * clamp, const ggml_tensor * scale, const ggml_tensor * bias, const ggml_cuda_topk_moe_args & args) {
    if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 2, /*is_topk_moe=*/true)) {
        ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
        return true;
    }
    if (!ggml_is_contiguous(logits) ||
            !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 2, /*is_topk_moe=*/true, logits)) {
        return false;
    }
    ggml_cuda_pool_alloc<char> logits_copy(cuda_ctx->pool(), ggml_nbytes(logits));
    CUDA_CHECK(cudaMemcpyAsync(logits_copy.get(), logits->data, ggml_nbytes(logits), cudaMemcpyDeviceToDevice, cuda_ctx->stream()));
    ggml_tensor logits_c = *logits;
    logits_c.data = logits_copy.get();
    ggml_cuda_op_topk_moe(*cuda_ctx, &logits_c, weights, ids, clamp, scale, bias, args);
    return true;
}

// The topk_moe that reads the router logits of the one-token f32 MUL_MAT at node i in the last block of the router's
//   mmvf launch (which ends at node last), one launch less per MoE layer: no bias or groups. Its outputs may lie over
//   the logits, which that block's warp reads first, but not over the launch's inputs or its sibling's output, and no
//   node in between may touch them. Returns whether it folded; ggml_cuda_try_fuse then skips the topk_moe.
// With a sibling on Volta, a block of the launch that waits for the logits runs it instead, see mul_mat_vec_f_router.
static bool ggml_cuda_topk_moe_fold(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i,
                                   const ggml_tensor * sib_dst, const int last, ggml_cuda_mm_fusion_args_host & fusion) {
    static const bool disabled = getenv("GGML_CUDA_NO_TOPK_FOLD") != nullptr;
    static const bool no_wait  = getenv("GGML_CUDA_NO_ROUTER_WAIT") != nullptr;
    const ggml_tensor * node = cgraph->nodes[i];
    // the gating op that reads the logits; qwen35moe has its shared expert's projections between
    int k = last + 1;
    for (; k < cgraph->n_nodes && k <= last + 16; k++) {
        const ggml_tensor * n = cgraph->nodes[k];
        if ((n->op == GGML_OP_SOFT_MAX || n->op == GGML_OP_UNARY || n->op == GGML_OP_ARGSORT) && n->src[0] == node) {
            break;
        }
    }
    ggml_cuda_topk_moe_plan p;
    if (disabled || k >= cgraph->n_nodes || k > last + 16 || !ggml_cuda_topk_moe_plan_at(cgraph, k, p)) {
        return false;
    }
    const int64_t n_expert = node->ne[0];
    if (p.logits != node || p.bias != nullptr || p.args.n_groups > 1 || ggml_nrows(node) != 1 || !ggml_is_contiguous(node) ||
            (n_expert != 128 && n_expert != 256 && n_expert != 512) || p.ids->nb[1] != n_expert*sizeof(int32_t) ||
            !ggml_is_contiguous(p.weights) || p.weights->ne[1] > n_expert || ggml_cuda_tensors_overlap(p.weights, p.ids)) {
        return false;
    }
    for (const ggml_tensor * t : { (const ggml_tensor *) p.weights, (const ggml_tensor *) p.ids }) {
        if (ggml_cuda_tensors_overlap(t, node->src[0]) || ggml_cuda_tensors_overlap(t, node->src[1]) ||
                (sib_dst != nullptr && ggml_cuda_tensors_overlap(t, sib_dst))) {
            return false;
        }
    }
    // the outputs are written before the nodes in between run: none of them may read or write them, or the logits
    const ggml_tensor * held[3] = { node, p.weights, p.ids };
    for (int j = last + 1; j < k; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        for (const ggml_tensor * t : held) {
            if (ggml_cuda_tensors_overlap(n, t)) {
                return false;
            }
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (n->src[s] != nullptr && ggml_cuda_tensors_overlap(n->src[s], t)) {
                    return false;
                }
            }
        }
    }
    if (!ggml_cuda_check_fusion_memory_ranges(cgraph, k, p.n_ops, p.out_nodes, 2, /*is_topk_moe=*/true) &&
            !ggml_cuda_check_fusion_memory_ranges(cgraph, k, p.n_ops, p.out_nodes, 2, /*is_topk_moe=*/true, node)) {
        return false;
    }
    fusion.topk_weights         = (float *) p.weights->data;
    fusion.topk_ids             = (int32_t *) p.ids->data;
    fusion.topk_ticket          = cuda_ctx->block_flags_get() + 2;
    fusion.topk_n_expert        = (int) n_expert;
    fusion.topk_n_used          = (int) p.weights->ne[1];
    fusion.topk_clamp           = p.clamp ? ggml_get_op_params_f32(p.clamp, 0) : -INFINITY;
    fusion.topk_scale           = p.scale ? ggml_get_op_params_f32(p.scale, 0) : 1.0f;
    fusion.topk_sigmoid         = p.args.sigmoid;
    fusion.topk_sqrt_softplus   = p.args.sqrt_softplus;
    fusion.topk_with_norm       = p.clamp != nullptr;
    fusion.topk_delayed_softmax = p.args.delayed_softmax;
    // on Volta a block of the launch waits for the tagged logits instead of a ticket (mul_mat_vec_f_router); measured on Volta only
    const int  cc    = ggml_cuda_info().devices[cuda_ctx->device].cc;
    const bool volta = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_TURING;
    if (!no_wait && volta && node->src[0]->ne[0] % 512 == 0 && sib_dst != nullptr) {
        fusion.topk_slots = (unsigned long long *) (cuda_ctx->block_flags_get() + GGML_CUDA_TOPK_SLOTS);
    }
    cuda_ctx->topk_folded_node = cgraph->nodes[k];
    cuda_ctx->topk_folded_skip = p.n_ops - 1;
    return true;
}

// Name t (the node at i_out) for its kernel to write the q8_1 copy of, see q8_1_emit_alloc, if the first MMVQ
//   matmul that reads t, or a view of all of it, comes before any node writes t again
// flat: the kernel writes the q8_1 copy in the order of t's elements, so the matmul may read t as rows of any length that
//   needs no q8_1 padding (qwen4exp's hc down projection reads the 4 normed streams of a token as one row, a gated delta
//   net's output projection the heads of a token). padded: it writes t's rows at the padded stride, zeros after them, so
//   the matmul may read them as they are when they need padding (a split shared expert's 160 columns at several tokens).
static void ggml_cuda_q8_1_request(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i_out,
        const bool flat = false, const bool padded = false) {
    const ggml_tensor * t = cgraph->nodes[i_out];
    cuda_ctx->q8_1_emit    = nullptr;
    cuda_ctx->q8_1_emit_as = nullptr;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || !cuda_ctx->stream_context().concurrent_events.empty()) {
        return;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    for (int j = i_out + 1; j < cgraph->n_nodes && j <= i_out + 64; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        const ggml_tensor * y = n->src[1];
        if ((n->op == GGML_OP_MUL_MAT || n->op == GGML_OP_MUL_MAT_ID) && y != nullptr &&
                (y == t || (y->view_src == t && y->view_offs == 0))) {
            const bool mmvq = n->op == GGML_OP_MUL_MAT ?
                ggml_cuda_mul_mat_choose(*cuda_ctx, n->src[0], y, n) == GGML_CUDA_MUL_MAT_PATH_MMVQ && !ggml_cuda_op_mul_mat_use_fwht(n) :
                ggml_is_quantized(n->src[0]->type) && n->ne[2] <= get_mmvq_mmid_max_batch(n->src[0]->type, cc);
            if (mmvq) {
                // all of t as one row, or its rows as the tokens of a batch when a row needs no q8_1 padding (an MMVQ
                //   batch reads its rows at the padded stride)
                const bool one_row = ggml_nelements(y) == y->ne[0] && y->ne[0] == ggml_nelements(t);
                const bool rows    = (y->ne[0] == t->ne[0] || flat) && ggml_nelements(y) == ggml_nelements(t) &&
                    (y->ne[0] % MATRIX_ROW_PADDING == 0 || (padded && y->ne[0] == t->ne[0]));
                if (ggml_is_contiguous(y) && (one_row || rows)) {
                    cuda_ctx->q8_1_emit    = t;
                    cuda_ctx->q8_1_emit_as = y;
                }
                return;
            }
            // another kernel reads t, e.g. the f32 router of a MoE before its experts
            continue;
        }
        if (n->data == t->data && !ggml_cuda_is_view_or_noop(n)) {
            return;
        }
    }
}

// The gate and up MUL_MAT_IDs of a GLU whose weights are pools of cached experts of the same layout: each pool holds the
//   used experts in its own slots, so the fused launch reads the gate through its own remapped ids (gate_ids)
static bool ggml_cuda_expert_cache_glu_pair(const ggml_tensor * up, const ggml_tensor * gate) {
    static const bool disabled = getenv("GGML_CUDA_NO_EXPERT_CACHE_GLU") != nullptr;
    return !disabled && up->op == GGML_OP_MUL_MAT_ID && gate->op == GGML_OP_MUL_MAT_ID &&
        up->src[1] == gate->src[1] && up->src[2] == gate->src[2] &&
        ggml_cuda_expert_cache_is(up->src[0]) && ggml_cuda_expert_cache_is(gate->src[0]) &&
        up->src[0]->type == gate->src[0]->type && ggml_are_same_shape(up->src[0], gate->src[0]) &&
        ggml_are_same_stride(up->src[0], gate->src[0]) && ggml_cuda_should_fuse_mul_mat_vec_q(up, true, true);
}

// A one-token gate/up GLU (MUL_MAT, MUL_MAT, GLU) with a MUL_MAT_ID gate/up GLU of the same input and matrices of the
//   same shape after it, e.g. the shared expert of Qwen3.5-MoE built before its routed experts: deferred to that launch,
//   which runs it as one more channel past the used experts (one launch less per layer). Nothing in between may read its
//   output or write its input or output. Returns the nodes to skip.
static int ggml_cuda_try_defer_glu(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_DEFER_GLU") != nullptr;
    if (disabled || cuda_ctx->glu_deferred.glu != nullptr ||
            !ggml_cuda_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        return 0;
    }
    const auto glu_pair = [&](int k, const ggml_tensor *& up, const ggml_tensor *& gate) {
        const ggml_tensor * glu = cgraph->nodes[k + 2];
        gate = glu->src[0];
        up   = glu->src[1];
        return ((gate == cgraph->nodes[k] && up == cgraph->nodes[k + 1]) || (gate == cgraph->nodes[k + 1] && up == cgraph->nodes[k])) &&
            gate->src[1] == up->src[1] && gate->src[2] == up->src[2];
    };
    const ggml_tensor * up;
    const ggml_tensor * gate;
    ggml_tensor * glu = cgraph->nodes[i + 2];
    if (!glu_pair(i, up, gate)) {
        return 0;
    }
    const ggml_tensor * x = up->src[1];
    // through mul_mat_vec_q, or for 2-byte float weights mul_mat_vec_f, which reads x itself in the launch that writes glu
    static const bool no_mmvf = getenv("GGML_CUDA_NO_DEFER_GLU_MMVF") != nullptr;
    const bool mmvf = !ggml_is_quantized(up->src[0]->type);
    if (x->type != GGML_TYPE_F32 || ggml_nrows(x) != 1 || glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) ||
            !(mmvf ? !no_mmvf && ggml_cuda_should_fuse_mul_mat_vec_f(up) && !ggml_cuda_tensors_overlap(glu, x) :
                     ggml_cuda_should_fuse_mul_mat_vec_q(up, true))) {
        return 0;
    }

    for (int j = i + 3; j + 2 < cgraph->n_nodes; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_MUL_MAT_ID && ggml_cuda_can_fuse(cgraph, j, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU }, {})) {
            const ggml_tensor * up2;
            const ggml_tensor * gate2;
            const ggml_tensor * glu2 = cgraph->nodes[j + 2];
            if (!glu_pair(j, up2, gate2)) {
                return 0;
            }
            const ggml_tensor * x2  = up2->src[1];
            const ggml_tensor * ids = up2->src[2];
            bool ok = x2->data == x->data && x2->type == GGML_TYPE_F32 && ggml_nelements(x2) == ggml_nelements(x) &&
                x2->ne[1] == 1 && glu2->ne[2] == 1 && glu2->ne[0] == glu->ne[0] &&
                memcmp(glu2->op_params, glu->op_params, sizeof(glu->op_params)) == 0 &&
                !ggml_cuda_tensors_overlap(glu, glu2) && !ggml_cuda_tensors_overlap(glu, ids) &&
                (mmvf ? ggml_cuda_should_fuse_mul_mat_vec_f(up2) &&
                        ggml_cuda_mul_mat_vec_f_moe_glu_sibling(up2->src[0], gate2->src[0], x2, ids, up->src[0], gate->src[0]) :
                        ggml_cuda_should_fuse_mul_mat_vec_q(up2, true) || ggml_cuda_expert_cache_glu_pair(up2, gate2));
            for (int k = 0; ok && k < 2; k++) {
                const ggml_tensor * w  = (k == 0 ? up : gate)->src[0];
                const ggml_tensor * w2 = (k == 0 ? up2 : gate2)->src[0];
                ok = w->type == w2->type && w->ne[0] == w2->ne[0] && w->ne[1] == w2->ne[1] && w->nb[1] == w2->nb[1];
            }
            if (!ok) {
                return 0;
            }
            cuda_ctx->glu_deferred = { up, gate, glu, j };
            return 2;
        }
        if (ggml_cuda_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (ggml_cuda_tensors_overlap(n, glu) || ggml_cuda_tensors_overlap(n, x)) {
            return 0;
        }
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            if (n->src[k] != nullptr && ggml_cuda_tensors_overlap(n->src[k], glu)) {
                return 0;
            }
        }
    }
    return 0;
}

// In a session, a MoE sum dst of one token that ends the subgraph (no compute node from k_next on) is the input of the
//   all-reduce right after it, which can compute it (ggml_backend_cuda_session_allreduce): the sum is not written, one
//   launch less per layer
static bool ggml_cuda_moe_sum_to_allreduce(const ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int k_next,
        const ggml_cuda_moe_weighted_reduction_match & match, const ggml_tensor * dst) {
    static const bool no_red_ar = getenv("GGML_CUDA_NO_REDUCE_AR") != nullptr;
    bool last = cuda_ctx->session_mode != ggml_backend_cuda_context::SESSION_OFF && !no_red_ar &&
        ggml_is_contiguous(match.weights) && ggml_nelements(match.weights) == match.experts->ne[1] &&
        ggml_is_contiguous(match.experts) && dst->ne[0] % 4 == 0;
    for (int k = k_next; last && k < cgraph->n_nodes; k++) {
        last = ggml_cuda_is_view_or_noop(cgraph->nodes[k]) || (cgraph->nodes[k]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
    }
    return last;
}

// At one token: the MUL_MAT_ID of the routed experts' down projection, their weighted sum, and a shared expert's down
//   projection added to the sum, gated by sigmoid(g) (Qwen3.5-MoE builds them in this order) or not (glm5-next, whose
//   shared expert's GLU runs before the routed experts): the shared expert runs in the experts' launch as one more
//   channel and the sum adds it, two launches instead of three. Returns the nodes to skip.
static int ggml_cuda_try_fuse_moe_down_shexp(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_SHEXP_MERGE") != nullptr;
    ggml_tensor * down = cgraph->nodes[i];
    if (disabled || down->type != GGML_TYPE_F32 || down->ne[2] != 1 || down->ne[3] != 1 || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_cuda_moe_weighted_reduction_match match;
    if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i + 1, match) || match.experts != down || match.expert_scale != nullptr) {
        return 0;
    }
    const int t = i + 1 + match.node_count;
    if (t + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * mm = cgraph->nodes[t];
    if (mm->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    // the gated tail MUL_MAT -> SIGMOID(g) -> MUL -> ADD, or MUL_MAT -> ADD
    const bool gated = t + 3 < cgraph->n_nodes && cgraph->nodes[t + 1]->op == GGML_OP_UNARY;
    const ggml_tensor * g   = nullptr;
    const ggml_tensor * sum = mm; // what the ADD adds to the weighted sum
    if (gated) {
        const ggml_tensor * sig = cgraph->nodes[t + 1];
        const ggml_tensor * mul = cgraph->nodes[t + 2];
        if (ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || mul->op != GGML_OP_MUL ||
                !((mul->src[0] == mm && mul->src[1] == sig) || (mul->src[0] == sig && mul->src[1] == mm))) {
            return 0;
        }
        g   = sig->src[0];
        sum = mul;
        if (g->type != GGML_TYPE_F32 || ggml_nelements(g) != 1) {
            return 0;
        }
    }
    const int     add_idx = t + (gated ? 3 : 1);
    ggml_tensor * add     = cgraph->nodes[add_idx];
    // the gated shared expert runs in the short q8_0 kernel, which quantizes its input itself; the plain mmvq kernel
    //   reads the ungated one's q8_1 row after those of the experts; 2-byte float experts run in mul_mat_vec_f
    const int  cc       = ggml_cuda_info().devices[cuda_ctx->device].cc;
    const bool use_mmvf = ggml_cuda_mul_mat_id_use_mmvf(down->src[0], 1, cc) &&
        ggml_cuda_mul_mat_vec_f_moe_sibling(down->src[0], down->src[1], down->src[2], mm->src[0], mm->src[1]);
    if (add->op != GGML_OP_ADD ||
            !((add->src[0] == match.dst && add->src[1] == sum) || (add->src[0] == sum && add->src[1] == match.dst)) ||
            add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) ||
            !ggml_are_same_shape(add, match.dst) || !ggml_are_same_shape(mm, match.dst) ||
            !(use_mmvf ||
              (gated ? ggml_cuda_mul_mat_vec_q_short_sibling(down->src[0], down->src[1], down->src[2], mm->src[0], mm->src[1]) :
                       ggml_cuda_mul_mat_vec_q_moe_sibling(down->src[0], down->src[1], down->src[2], mm->src[0], mm->src[1])))) {
        return 0;
    }

    // one fusion: nothing past it reads the sum or the shared expert's partial results
    std::vector<ggml_op> ops;
    for (int k = i; k <= add_idx; k++) {
        ops.push_back(cgraph->nodes[k]->op);
    }
    const int out = add_idx;
    if (ops.size() >= 32 || !ggml_can_fuse_subgraph(cgraph, i, (int) ops.size(), ops.data(), &out, 1)) {
        return 0;
    }

    // as for the weighted sum alone (see ggml_cuda_try_fuse), with the add as dst; every block reads the gate, and the
    //   shared expert's input is read in the launch that writes the experts
    if ((g != nullptr && ggml_cuda_tensors_overlap(add, g)) || ggml_cuda_tensors_overlap(down, mm->src[1])) {
        return 0;
    }
    if (ggml_cuda_tensors_overlap(add, down)) {
        const int64_t off = (const char *) add->data - (const char *) down->data;
        if (off < 0 || off % (int64_t) down->nb[1] != 0 || off / (int64_t) down->nb[1] >= down->ne[1]) {
            return 0;
        }
    }

    // a pool of cached experts is read through its remapped ids
    const ggml_tensor * ids = down->src[2];
    ggml_tensor ids_remap;
    if (ggml_cuda_expert_cache_is(down->src[0])) {
        ggml_cuda_expert_cache_remap(*cuda_ctx, down->src[0], ids, &ids_remap);
        ids = &ids_remap;
    }

    const bool last = ggml_cuda_moe_sum_to_allreduce(cuda_ctx, cgraph, add_idx + 1, match, add);

    auto shexp = std::make_unique<ggml_cuda_pool_alloc<float>>(cuda_ctx->pool(), ggml_nelements(mm));
    ggml_cuda_mm_fusion_args_host fusion_data{};
    fusion_data.sib     = mm->src[0];
    fusion_data.sib_y   = mm->src[1];
    fusion_data.sib_out = shexp->get();
    // the all-reduce of the sum folded into the launch (ggml_cuda_ar_fold_moe_args, gated only): the sum is not
    //   written anywhere
    static const bool no_moe_fold = getenv("GGML_CUDA_NO_AR_FOLD_MOE") != nullptr;
    const bool fold = !use_mmvf && last && gated && !no_moe_fold && cuda_ctx->ar_fold.next == add && ggml_are_same_shape(match.experts, down) &&
        match.experts->data == down->data && match.experts->ne[1] <= 8;
    if (fold) {
        fusion_data.ar_weights = match.weights;
        fusion_data.ar_gate    = g;
    }
    if (use_mmvf) {
        ggml_cuda_mul_mat_vec_f(*cuda_ctx, down->src[0], down->src[1], ids, down, &fusion_data);
    } else {
        ggml_cuda_mul_mat_vec_q(*cuda_ctx, down->src[0], down->src[1], ids, down, &fusion_data);
    }
    if (fold && cuda_ctx->ar_fold.pushed) {
        return add_idx - i;
    }

    if (last) {
        ggml_backend_cuda_context::session_reduction_t & red = cuda_ctx->session_red;
        GGML_ASSERT(red.dst == nullptr);
        red.experts = match.experts;
        red.weights = match.weights;
        red.add_g   = g;
        red.dst     = add;
        red.add_x   = std::move(shexp);
        return add_idx - i;
    }
    ggml_cuda_op_moe_weighted_reduction(*cuda_ctx, match.experts, nullptr, match.weights, add, false, shexp->get(), g);
    return add_idx - i;
}

// glm5-next's k-pool compression (build_kpool_select), from the GET_ROWS at i of the new pools' key | gate rows:
//   pooled = SUM_ROWS(SOFT_MAX(CONT(PERMUTE(gate + ape))) * CONT(PERMUTE(key))), SET_ROWS into the pooled slots of the
//   same cache. Eight launches in one (ggml_cuda_kpool_pool), with the values of the ops apart. Returns the nodes to skip.
static int ggml_cuda_try_fuse_kpool_pool(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_KPOOL_FUSION") != nullptr;
    const ggml_tensor * gr = cgraph->nodes[i];
    if (disabled || gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || !cuda_ctx->stream_context().concurrent_events.empty()) {
        return 0;
    }
    const ggml_tensor * kg      = gr->src[0];
    const ggml_tensor * members = gr->src[1];
    if ((kg->type != GGML_TYPE_F16 && kg->type != GGML_TYPE_F32) || kg->nb[0] != ggml_type_size(kg->type) || ggml_nrows(kg) != kg->ne[1] ||
            members->type != GGML_TYPE_I32 || !ggml_is_contiguous(members) || ggml_nrows(gr) != gr->ne[1] || gr->ne[0] % 2 != 0) {
        return 0;
    }
    const int64_t n_embd = gr->ne[0] / 2;

    // the SET_ROWS that ends the chain, and its sources back to gr
    int s = -1;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 32; j++) {
        if (cgraph->nodes[j]->op == GGML_OP_SET_ROWS) {
            s = j;
            break;
        }
    }
    if (s < 0) {
        return 0;
    }
    const ggml_tensor * set = cgraph->nodes[s];
    const ggml_tensor * rep = set->src[1];
    const ggml_tensor * dst = set->src[2];
    const auto through = [](const ggml_tensor * t, ggml_op op) { return t != nullptr && t->op == op ? t->src[0] : nullptr; };
    const ggml_tensor * sr  = through(set->src[0], GGML_OP_RESHAPE);
    const ggml_tensor * mul = sr != nullptr && sr->op == GGML_OP_SUM_ROWS ? sr->src[0] : nullptr;
    if (mul == nullptr || mul->op != GGML_OP_MUL || rep->type != GGML_TYPE_I64 || !ggml_is_contiguous(rep) ||
            dst->type != kg->type || dst->nb[1] != kg->nb[1] || dst->ne[0] != n_embd) {
        return 0;
    }
    const ggml_tensor * sm   = through(mul->src[0], GGML_OP_RESHAPE);
    const ggml_tensor * pkc  = mul->src[1];
    const ggml_tensor * lc   = sm != nullptr && sm->op == GGML_OP_SOFT_MAX ? through(sm->src[0], GGML_OP_RESHAPE) : nullptr;
    const ggml_tensor * addp = through(lc, GGML_OP_CONT);
    const ggml_tensor * add  = through(addp, GGML_OP_PERMUTE);
    const ggml_tensor * pkp  = through(pkc, GGML_OP_CONT);
    const ggml_tensor * pk   = through(pkp, GGML_OP_PERMUTE);
    if (add == nullptr || add->op != GGML_OP_ADD || pk == nullptr || pk->op != GGML_OP_VIEW || sm->src[1] != nullptr || sm->src[2] != nullptr) {
        return 0;
    }
    float sm_scale, sm_max_bias;
    memcpy(&sm_scale,    (const float *) sm->op_params + 0, sizeof(float));
    memcpy(&sm_max_bias, (const float *) sm->op_params + 1, sizeof(float));
    const ggml_tensor * pg  = add->src[0];
    const ggml_tensor * ape = add->src[1];
    const ggml_tensor * rows = pk->src[0];
    const int64_t kpool = add->ne[1];
    const int64_t n_new = add->ne[2];
    const size_t  ts    = sizeof(float);
    // gr reshaped to [2*n_embd, kpool, n_new], key the first half of a row and gate the second
    if (sm_scale != 1.0f || sm_max_bias != 0.0f || rows->op != GGML_OP_RESHAPE || rows->src[0] != gr ||
            pg->op != GGML_OP_VIEW || pg->src[0] != rows || pg->view_offs != n_embd*ts || pk->view_offs != 0 ||
            !ggml_are_same_shape(pg, pk) || !ggml_are_same_stride(pg, pk) || pg->ne[0] != n_embd || pg->nb[1] != 2*n_embd*ts ||
            ape->type != GGML_TYPE_F32 || !ggml_is_contiguous(ape) || ape->ne[0] != n_embd || ape->ne[1] != kpool || ggml_nrows(ape) != kpool ||
            kpool > WARP_SIZE || ggml_nelements(members) != kpool*n_new || ggml_nelements(rep) != n_new ||
            sr->ne[0] != 1 || ggml_nelements(sr) != n_embd*n_new || sm->ne[0] != kpool) {
        return 0;
    }
    // both permutes swap the first two axes
    for (const ggml_tensor * p : { addp, pkp }) {
        const int32_t * axes = (const int32_t *) p->op_params;
        if (axes[0] != 1 || axes[1] != 0 || axes[2] != 2 || axes[3] != 3) {
            return 0;
        }
    }
    // kg and dst are views of one cache, its key | gate and its pooled slot
    const ggml_tensor * cache = kg->view_src;
    if (cache == nullptr || dst->view_src != cache || dst->view_offs != kg->view_offs + 2*n_embd*ggml_type_size(kg->type)) {
        return 0;
    }
    // the compute nodes from gr to the SET_ROWS are this chain, and only the chain reads its intermediate results
    const ggml_tensor * chain[] = { gr, add, lc, sm, pkc, mul, sr, set };
    int n_compute = 0;
    for (int j = i; j <= s; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (std::find(std::begin(chain), std::end(chain), n) == std::end(chain)) {
            return 0;
        }
        n_compute++;
    }
    if (n_compute != 8) {
        return 0;
    }
    const ggml_tensor * inner[] = { gr, rows, add, addp, lc, sm, sm->src[0], mul->src[0], pkp, pkc, pk, pg, mul, sr, set->src[0] };
    for (const ggml_tensor * t : inner) {
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return 0;
        }
    }
    const ggml_tensor * once[] = { gr, add, lc, sm, pkc, mul, sr };
    for (const ggml_tensor * t : once) {
        if (ggml_cuda_graph_use_count(cgraph, t) != 1) {
            return 0;
        }
    }
    if (ggml_cuda_graph_use_count(cgraph, rows) != 2) {
        return 0;
    }

    ggml_cuda_kpool_pool(*cuda_ctx, kg, members, ape, rep, dst, (int) kpool);
    return s - i;
}

// the reduction left by ggml_cuda_try_fuse_moe_down_shexp on its own
static void ggml_cuda_session_reduction_flush(ggml_backend_cuda_context * cuda_ctx) {
    ggml_backend_cuda_context::session_reduction_t & red = cuda_ctx->session_red;
    if (red.dst == nullptr) {
        return;
    }
    ggml_cuda_op_moe_weighted_reduction(*cuda_ctx, red.experts, nullptr, red.weights, red.dst, false,
        red.add_x ? red.add_x->get() : nullptr, red.add_g);
    red = {};
}

// the deferred GLU on its own, as ggml_cuda_try_fuse would have run it
static void ggml_cuda_flush_deferred_glu(ggml_backend_cuda_context * cuda_ctx) {
    ggml_backend_cuda_context::deferred_glu & d = cuda_ctx->glu_deferred;
    if (d.glu == nullptr) {
        return;
    }
    ggml_cuda_mm_fusion_args_host fusion_data{};
    fusion_data.gate      = d.gate->src[0];
    fusion_data.glu_op    = ggml_get_glu_op(d.glu);
    fusion_data.glu_limit = ggml_get_op_params_f32(d.glu, 3);
    if (ggml_is_quantized(d.up->src[0]->type)) {
        ggml_cuda_mul_mat_vec_q(*cuda_ctx, d.up->src[0], d.up->src[1], nullptr, d.glu, &fusion_data);
    } else {
        ggml_cuda_mul_mat_vec_f(*cuda_ctx, d.up->src[0], d.up->src[1], nullptr, d.glu, &fusion_data);
    }
    d = {};
}

// GGML_CUDA_EXPERT_CPU: the CPU split of a one-token MoE layer whose gate, up and down pools are gathered as one group
//   (mm: their MUL_MAT_IDs, the first is node i), see expert-cpu.cuh. The CPU computes the GLU between them and the
//   routing weights the down outputs are multiplied by, so both must be what it knows; the publishing device adds the
//   CPU's weighted sum to the input of the all-reduce that ends the subgraph (ggml_cuda_expert_cpu_merge_pending).
static bool ggml_cuda_expert_cpu_try_args(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i,
        const ggml_tensor * const * mm, const int n, ggml_cuda_expert_cpu_args & args) {
    args = {};
    const ggml_tensor * ids = mm[0]->src[2];
    if (!ggml_cuda_expert_cpu_enabled() || n != 3 || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1) {
        return false;
    }
    // gate and up read the layer input, down the GLU of their outputs
    const ggml_tensor * x     = mm[0]->src[1];
    const ggml_tensor * gu[2] = { nullptr, nullptr };
    const ggml_tensor * down  = nullptr;
    int n_gu = 0;
    for (int k = 0; k < 3; k++) {
        if (mm[k]->src[1] == x && n_gu < 2) {
            gu[n_gu++] = mm[k];
        } else {
            down = mm[k];
        }
    }
    if (n_gu != 2 || down == nullptr) {
        return false;
    }
    const ggml_tensor * glu = down->src[1];
    if (glu->op != GGML_OP_GLU || glu->src[1] == nullptr || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }
    const int glu_op = (int) ggml_get_glu_op(glu);
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_SWIGLU_CLAMP) {
        return false;
    }
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (!((gate == gu[0] && up == gu[1]) || (gate == gu[1] && up == gu[0]))) {
        return false;
    }
    const float limit = glu_op == GGML_GLU_OP_SWIGLU_CLAMP ? ggml_get_op_params_f32(glu, 3) : 0.0f;

    // the routing weights: the MUL of the down outputs right after them, computed before node i
    int j_down = -1;
    for (int j = i; j < cgraph->n_nodes && j <= i + 64; j++) {
        if (cgraph->nodes[j] == down) {
            j_down = j;
            break;
        }
    }
    const ggml_tensor * w = nullptr;
    for (int j = j_down + 1; j_down >= 0 && j < cgraph->n_nodes && j <= j_down + 8; j++) {
        const ggml_tensor * m = cgraph->nodes[j];
        if (m->op == GGML_OP_MUL && m->src[0] == down) {
            w = m->src[1];
            break;
        }
    }
    if (w == nullptr || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || ggml_nelements(w) != ids->ne[0]) {
        return false;
    }
    for (int j = i; j < cgraph->n_nodes; j++) {
        if (cgraph->nodes[j] == w || (w->view_src != nullptr && cgraph->nodes[j] == w->view_src)) {
            return false;
        }
    }
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ggml_nrows(x) != 1 || (uintptr_t) x->data % 16 != 0) {
        return false;
    }

    const int group = ggml_cuda_expert_cpu_group(gate->src[0], up->src[0], down->src[0], glu_op, limit, x->ne[0]);
    if (group < 0) {
        return false;
    }
    ggml_cuda_expert_cpu_args_get(cuda_ctx->device, group, (const float *) x->data, (const float *) w->data, (int) x->ne[0], args);
    if (!args.on) {
        return false;
    }
    if (args.mbox != nullptr) {
        cuda_ctx->expert_cpu_merge  = true;
        cuda_ctx->expert_cpu_n_embd = (int) x->ne[0];
    }
    return true;
}

// The CPU's sum (see ggml_cuda_expert_cpu_try_args) added to the input of the all-reduce that ends the subgraph: its last
//   compute node. A MoE sum left for that all-reduce to compute is computed first.
static void ggml_cuda_expert_cpu_merge_pending(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph) {
    if (!cuda_ctx->expert_cpu_merge) {
        return;
    }
    cuda_ctx->expert_cpu_merge = false;
    ggml_tensor * dst = nullptr;
    for (int k = cgraph->n_nodes - 1; k >= 0 && dst == nullptr; k--) {
        ggml_tensor * node = cgraph->nodes[k];
        if (!ggml_cuda_is_view_or_noop(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            dst = node;
        }
    }
    GGML_ASSERT(dst != nullptr && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) &&
        dst->ne[0] == cuda_ctx->expert_cpu_n_embd && ggml_nrows(dst) == 1);
    static bool reported = false;
    if (!reported) {
        reported = true;
        GGML_LOG_INFO("%s: the CPU experts are added to %s (%s)\n", __func__, dst->name, ggml_op_desc(dst));
    }
    if (cuda_ctx->session_red.dst != nullptr && cuda_ctx->session_red.dst->data == dst->data) {
        ggml_cuda_session_reduction_flush(cuda_ctx);
    }
    GGML_ASSERT(!(cuda_ctx->ar_fold.pushed && cuda_ctx->ar_fold.next == dst) && "an all-reduce fold already sent the lines");
    static const bool no_merge = getenv("GGML_CUDA_EXPERT_CPU_NO_MERGE") != nullptr; // debug: drop the CPU experts
    if (!no_merge) {
        ggml_cuda_expert_cpu_merge(cuda_ctx->stream(), (float *) dst->data, cuda_ctx->expert_cpu_n_embd);
    }
}

// The expert-cache pools of the MUL_MAT_IDs after node i that read the same ids (the up and down experts of the layer
//   whose gate is i) are gathered with i's in one plan and one copy launch, two launches instead of six. Nothing between
//   may write the ids; a pool's own MUL_MAT_ID is the only reader of its slots, so moving its copy ahead is safe.
static void ggml_cuda_try_gather_expert_group(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_EXPERT_GROUP_GATHER") != nullptr;
    const ggml_tensor * node = cgraph->nodes[i];
    if (disabled || node->op != GGML_OP_MUL_MAT_ID || !ggml_cuda_expert_cache_is(node->src[0])) {
        return;
    }
    const ggml_tensor * ids = node->src[2];
    for (const auto & a : cuda_ctx->expert_cache_ahead) {
        if (a.pool == node->src[0] && a.ids == ids) {
            return; // gathered with an earlier one
        }
    }
    const ggml_tensor * pools[3] = { node->src[0], nullptr, nullptr };
    const ggml_tensor * mm[3]    = { node, nullptr, nullptr };
    int n = 1;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 64 && n < 3; j++) {
        const ggml_tensor * m = cgraph->nodes[j];
        if (m->op == GGML_OP_MUL_MAT_ID && m->src[2] == ids && ggml_cuda_expert_cache_is(m->src[0]) &&
                m->src[0] != pools[0] && m->src[0] != pools[1]) {
            mm[n]      = m;
            pools[n++] = m->src[0];
            continue;
        }
        if (!ggml_cuda_is_view_or_noop(m) && ggml_cuda_tensors_overlap(m, ids)) {
            break;
        }
    }
    if (n < 2) {
        return;
    }
    ggml_cuda_expert_cpu_args cpu;
    const bool use_cpu = ggml_cuda_expert_cpu_try_args(cuda_ctx, cgraph, i, mm, n, cpu);
    ggml_tensor remap[3];
    ggml_cuda_expert_cache_gather_group(*cuda_ctx, pools, n, ids, remap, use_cpu ? &cpu : nullptr);
    for (int k = 0; k < n; k++) {
        cuda_ctx->expert_cache_ahead.push_back({ pools[k], ids, remap[k] });
    }
}

// The scores of the qwen4exp QSA indexer as its unfused branch builds them (llama_model_qwen4exp::graph, every batch
//   size): RELU of the heads' block scores, their sum over the heads (CONT of head 0, ADD of each next), ADD of the
//   block bias, CONT of the transpose, GET_ROWS of each cell's block, CONT of the transpose back, CPY of the f16 KQ mask
//   to f32 and ADD of it. Eleven launches per layer at four heads, here two (see ggml_cuda_qsa_cell_scores). Returns
//   the nodes to skip.
static int ggml_cuda_try_fuse_qsa_cell_scores(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_QSA_SCORES") != nullptr;
    const ggml_tensor * relu = cgraph->nodes[i];
    if (disabled || relu->op != GGML_OP_UNARY || ggml_get_unary_op(relu) != GGML_UNARY_OP_RELU ||
            relu->type != GGML_TYPE_F32 || relu->src[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(relu) ||
            relu->src[0]->nb[0] != sizeof(float) || !ggml_are_same_shape(relu, relu->src[0])) {
        return 0;
    }
    const ggml_tensor * x = relu->src[0];
    const int64_t n_blocks = relu->ne[0];
    const int64_t n_h      = relu->ne[1];
    const int64_t n_tps    = relu->ne[2];
    const int64_t n_stream = relu->ne[3];

    // the compute nodes after i, past views; any other node in between ends the match
    int j = i;
    auto next = [&]() -> ggml_tensor * {
        for (j++; j < cgraph->n_nodes && j < i + 31; j++) {
            if (!ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
                return cgraph->nodes[j];
            }
        }
        return nullptr;
    };
    // head h of the rectified scores, [n_blocks, n_tps, n_stream]
    auto is_head = [&](const ggml_tensor * v, const int64_t h) {
        return v->op == GGML_OP_VIEW && v->src[0] == relu && v->view_offs == (size_t) h*relu->nb[1] &&
            v->ne[0] == n_blocks && v->ne[1] == n_tps && v->ne[2] == n_stream && v->ne[3] == 1 &&
            v->nb[1] == relu->nb[2] && v->nb[2] == relu->nb[3];
    };
    // t is CONT(PERMUTE(a, 1, 0, 2, 3))
    auto is_transpose = [&](const ggml_tensor * t, const ggml_tensor * a) {
        const ggml_tensor * p = t->src[0];
        return t->op == GGML_OP_CONT && p->op == GGML_OP_PERMUTE && p->src[0] == a &&
            p->ne[0] == a->ne[1] && p->ne[1] == a->ne[0] && p->ne[2] == a->ne[2] && p->ne[3] == a->ne[3] &&
            p->nb[0] == a->nb[1] && p->nb[1] == a->nb[0] && p->nb[2] == a->nb[2] && p->nb[3] == a->nb[3];
    };

    ggml_tensor * n = next();
    if (n == nullptr || n->op != GGML_OP_CONT || !is_head(n->src[0], 0)) {
        return 0;
    }
    const ggml_tensor * sum = n;
    for (int64_t h = 1; h < n_h; h++) {
        n = next();
        if (n == nullptr || n->op != GGML_OP_ADD || n->src[0] != sum || !is_head(n->src[1], h)) {
            return 0;
        }
        sum = n;
    }
    n = next();
    if (n == nullptr || n->op != GGML_OP_ADD || n->src[0] != sum) {
        return 0;
    }
    const ggml_tensor * bias = n->src[1];
    if (bias->type != GGML_TYPE_F32 || !ggml_are_same_shape(bias, n) || bias->nb[0] != sizeof(float)) {
        return 0;
    }
    sum = n;
    n = next();
    if (n == nullptr || !is_transpose(n, sum)) {
        return 0;
    }
    const ggml_tensor * tr = n;
    n = next();
    if (n == nullptr || n->op != GGML_OP_GET_ROWS || n->src[0] != tr) {
        return 0;
    }
    const ggml_tensor * cell_blk = n->src[1];
    if (cell_blk->type != GGML_TYPE_I32 || cell_blk->ne[1] != n_stream || cell_blk->ne[2] != 1 ||
            cell_blk->nb[0] != sizeof(int32_t)) {
        return 0;
    }
    const int64_t n_kv = cell_blk->ne[0];
    const ggml_tensor * rows = n;
    n = next();
    if (n == nullptr || !is_transpose(n, rows)) {
        return 0;
    }
    const ggml_tensor * expanded = n;
    n = next();
    if (n == nullptr || n->op != GGML_OP_CPY) {
        return 0;
    }
    const ggml_tensor * cast   = n;
    const int           i_cast = j;
    ggml_tensor * dst = next();
    if (dst == nullptr || dst->op != GGML_OP_ADD || dst->src[0] != expanded || dst->type != GGML_TYPE_F32 ||
            dst->ne[0] != n_kv || dst->ne[1] != n_tps || dst->ne[2] != n_stream || dst->nb[0] != sizeof(float)) {
        return 0;
    }
    // the mask is the f32 copy made here of an f16 mask [n_kv, n_tps, 1, n_stream], read through reshapes
    const ggml_tensor * m = dst->src[1];
    while (m != cast && (m->op == GGML_OP_RESHAPE || (m->op == GGML_OP_VIEW && m->view_offs == 0))) {
        m = m->src[0];
    }
    const ggml_tensor * mask = cast->src[0];
    if (m != cast || cast->type != GGML_TYPE_F32 || !ggml_are_same_shape(cast, mask) || mask->type != GGML_TYPE_F16 ||
            mask->nb[0] != ggml_type_size(mask->type) || mask->ne[0] != n_kv || mask->ne[1] != n_tps || mask->ne[2] != 1 ||
            mask->ne[3] != n_stream) {
        return 0;
    }

    const int count = j - i + 1;
    if (count >= 32) {
        return 0;
    }
    ggml_op ops[32];
    for (int k = 0; k < count; k++) {
        ops[k] = cgraph->nodes[i + k]->op;
    }
    // the cast is its own src[1] (ggml_cast), which its use count may include: checked here, not by the subgraph
    const int outputs[2] = { i_cast, j };
    if (!ggml_can_fuse_subgraph(cgraph, i, count, ops, outputs, 2)) {
        return 0;
    }
    int cast_uses = 0;
    for (int k = i_cast + 1; k <= j; k++) {
        for (int l = 0; l < GGML_MAX_SRC; l++) {
            cast_uses += cgraph->nodes[k]->src[l] == cast;
        }
    }
    const int cast_count = ggml_node_get_use_count(cgraph, i_cast);
    if ((cast->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            (cast_count != cast_uses && !(cast->src[1] == cast && cast_count == cast_uses + 1))) {
        return 0;
    }
    // the inputs other than the scores are graph inputs; the scores may lie under dst, they are read in a first pass
    for (const ggml_tensor * t : { bias, cell_blk, mask }) {
        if (ggml_cuda_tensors_overlap(dst, t)) {
            return 0;
        }
    }

    ggml_cuda_qsa_cell_scores(*cuda_ctx, x, bias, cell_blk, mask, mask->nb[3], dst);
    return count - 1;
}

// The attention mask of the qwen4exp QSA top-k (build_attn_qsa): FILL of a KQ-mask-shaped tensor, SET_ROWS of the picked
//   cells through a view of it as rows of one, ADD of the KQ mask through a view back: one launch instead of three (see
//   ggml_cuda_qsa_topk_mask). Returns the nodes to skip.
static int ggml_cuda_try_fuse_qsa_topk_mask(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_QSA_MASK") != nullptr;
    const ggml_tensor * fill = cgraph->nodes[i];
    if (disabled || fill->op != GGML_OP_FILL || (fill->type != GGML_TYPE_F16 && fill->type != GGML_TYPE_F32) ||
            !ggml_is_contiguous(fill) || fill->ne[2] != 1) {
        return 0;
    }
    int j = i;
    auto next = [&]() -> ggml_tensor * {
        for (j++; j < cgraph->n_nodes && j < i + 8; j++) {
            if (!ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
                return cgraph->nodes[j];
            }
        }
        return nullptr;
    };
    const ggml_tensor * sr = next();
    if (sr == nullptr || sr->op != GGML_OP_SET_ROWS) {
        return 0;
    }
    // the fill as rows of one: [1, n_kv, n_tps, n_stream]
    const ggml_tensor * rows = sr->src[2];
    if (rows->op != GGML_OP_VIEW || rows->src[0] != fill || rows->view_offs != 0 || rows->ne[0] != 1 ||
            rows->ne[1] != fill->ne[0] || rows->ne[2] != fill->ne[1] || rows->ne[3] != fill->ne[3] ||
            rows->nb[1] != fill->nb[0] || rows->nb[2] != fill->nb[1] || rows->nb[3] != fill->nb[3]) {
        return 0;
    }
    const ggml_tensor * vals = sr->src[0];
    const ggml_tensor * idx  = sr->src[1];
    if ((vals->type != GGML_TYPE_F32 && vals->type != GGML_TYPE_F16) || idx->type != GGML_TYPE_I32 ||
            idx->nb[0] != sizeof(int32_t) || idx->ne[1] != fill->ne[1] || idx->ne[2] != fill->ne[3] || idx->ne[3] != 1) {
        return 0;
    }
    ggml_tensor * dst = next();
    if (dst == nullptr || dst->op != GGML_OP_ADD || dst->type != fill->type || !ggml_are_same_shape(dst, fill) ||
            !ggml_is_contiguous(dst)) {
        return 0;
    }
    // the rows viewed back as the fill: [n_kv, n_tps, 1, n_stream]
    const ggml_tensor * back = dst->src[0];
    const ggml_tensor * kq   = dst->src[1];
    if (back->op != GGML_OP_VIEW || back->src[0] != sr || back->view_offs != 0 || !ggml_are_same_shape(back, fill) ||
            back->nb[1] != fill->nb[1] || back->nb[3] != fill->nb[3] || kq->type != fill->type ||
            !ggml_are_same_shape(kq, fill) || kq->nb[0] != ggml_type_size(kq->type)) {
        return 0;
    }

    const int count = j - i + 1;
    ggml_op ops[8];
    for (int k = 0; k < count; k++) {
        ops[k] = cgraph->nodes[i + k]->op;
    }
    if (!ggml_can_fuse_subgraph(cgraph, i, count, ops, &j, 1)) {
        return 0;
    }
    // a block reads its cells' kq before it writes them, so dst may sit exactly on kq; idx and vals are copied first
    //   when dst covers them
    if (!(dst->data == kq->data ? ggml_are_same_stride(dst, kq) : !ggml_cuda_tensors_overlap(dst, kq)) ||
            (ggml_cuda_tensors_overlap(dst, idx) && !ggml_is_contiguous(idx)) ||
            (ggml_cuda_tensors_overlap(dst, vals) && !ggml_is_contiguous(vals))) {
        return 0;
    }

    float value;
    memcpy(&value, fill->op_params, sizeof(float));
    ggml_cuda_qsa_topk_mask(*cuda_ctx, value, vals, idx, kq, dst);
    return count - 1;
}

// The first node after `from` (up to `to`) that reads t, or -1.
static int ggml_cuda_find_user(const ggml_cgraph * cgraph, const ggml_tensor * t, const int from, const int to) {
    for (int j = from + 1; j < cgraph->n_nodes && j <= to; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            if (n->src[k] == t) {
                return j;
            }
        }
    }
    return -1;
}

// One half of a DeepSeek-V4 compressor from its CONCAT(state, cur) at i_c1 to the CONT of the window transpose, see
//   llama_model_deepseek4::graph::build_compressed_kv_from_state (rows -> GET_ROWS -> RESHAPE -> CONT(PERMUTE)) and
//   build_overlap_compressed_kv_from_state (an extra row first: VIEW of row 0 -> SCALE -> RESHAPE -> CONCAT; then the
//   first half of the first n_read rows and the second half of the next n_read, each VIEW -> CONT -> RESHAPE, CONCAT,
//   CONT(PERMUTE)). Appends the chain's nodes to idxs.
struct ggml_cuda_dsv4_half_match {
    ggml_cuda_dsv4_compress_half h;
    const ggml_tensor * t = nullptr;
    int     i_c1 = -1; // the nodes that read the inputs
    int     i_g  = -1;
    bool    overlap = false;
    int64_t hd = 0;
    int64_t r  = 0;
    int64_t nb = 0;
};

static bool ggml_cuda_dsv4_parse_half(const ggml_cgraph * cgraph, const int i_c1, const int to, ggml_cuda_dsv4_half_match & m,
        std::vector<int> & idxs) {
    const ggml_tensor * c1 = cgraph->nodes[i_c1];
    if (c1->op != GGML_OP_CONCAT || ggml_get_op_params_i32(c1, 0) != 1 || c1->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * state = c1->src[0];
    const ggml_tensor * cur   = c1->src[1];
    const int64_t W = c1->ne[0];
    for (const ggml_tensor * t : { state, cur, c1 }) {
        if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || t->ne[0] != W || t->ne[2] != 1 || t->ne[3] != 1) {
            return false;
        }
    }
    m = {};
    m.h.state = state;
    m.h.cur   = cur;
    m.i_c1    = i_c1;
    idxs.push_back(i_c1);

    const ggml_tensor * source = c1;
    int j = ggml_cuda_find_user(cgraph, c1, i_c1, to);
    if (j < 0) {
        return false;
    }
    if (cgraph->nodes[j]->op == GGML_OP_VIEW) {
        // the extra row: SCALE of row 0, appended
        const ggml_tensor * v0 = cgraph->nodes[j];
        if (v0->view_offs != 0 || ggml_nelements(v0) != W) {
            return false;
        }
        idxs.push_back(j);
        const int js = ggml_cuda_find_user(cgraph, v0, j, to);
        if (js < 0 || cgraph->nodes[js]->op != GGML_OP_SCALE || cgraph->nodes[js]->type != GGML_TYPE_F32) {
            return false;
        }
        const ggml_tensor * z = cgraph->nodes[js];
        idxs.push_back(js);
        const int jr = ggml_cuda_find_user(cgraph, z, js, to);
        if (jr < 0 || cgraph->nodes[jr]->op != GGML_OP_RESHAPE) {
            return false;
        }
        const ggml_tensor * zr = cgraph->nodes[jr];
        idxs.push_back(jr);
        const int jc = ggml_cuda_find_user(cgraph, zr, jr, to);
        if (jc < 0) {
            return false;
        }
        const ggml_tensor * c2 = cgraph->nodes[jc];
        if (c2->op != GGML_OP_CONCAT || ggml_get_op_params_i32(c2, 0) != 1 || c2->src[0] != c1 || c2->src[1] != zr ||
                c2->ne[1] != c1->ne[1] + 1) {
            return false;
        }
        idxs.push_back(jc);
        m.h.extra = z;
        m.overlap = true;
        source = c2;
        j = jc;
    } else {
        j = i_c1; // the GET_ROWS is the first user
    }
    const int jg = ggml_cuda_find_user(cgraph, source, j, to);
    if (jg < 0) {
        return false;
    }
    const ggml_tensor * g = cgraph->nodes[jg];
    if (g->op != GGML_OP_GET_ROWS || g->src[0] != source || g->type != GGML_TYPE_F32 || g->src[1]->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(g->src[1]) || ggml_nelements(g->src[1]) != g->ne[1] || g->ne[2] != 1) {
        return false;
    }
    m.h.idx = g->src[1];
    m.i_g   = jg;
    idxs.push_back(jg);

    // the window [n, hd, nb] before its transpose
    const ggml_tensor * window = nullptr;
    int jw = jg;
    if (m.overlap) {
        if (W % 2 != 0 || g->ne[1] % 2 != 0) {
            return false;
        }
        m.hd = W/2;
        const int64_t n_read = g->ne[1]/2;
        const ggml_tensor * half_r[2] = { nullptr, nullptr };
        int jv = jg;
        for (int k = 0; k < 2; k++) {
            jv = ggml_cuda_find_user(cgraph, g, jv, to);
            if (jv < 0) {
                return false;
            }
            const ggml_tensor * v = cgraph->nodes[jv];
            const size_t offs = k == 0 ? 0 : (size_t) n_read*g->nb[1] + m.hd*sizeof(float);
            if (v->op != GGML_OP_VIEW || v->view_offs != offs || v->ne[0] != m.hd || v->ne[1] != n_read ||
                    v->nb[1] != g->nb[1]) {
                return false;
            }
            idxs.push_back(jv);
            const int jcont = ggml_cuda_find_user(cgraph, v, jv, to);
            if (jcont < 0 || cgraph->nodes[jcont]->op != GGML_OP_CONT) {
                return false;
            }
            idxs.push_back(jcont);
            const int jre = ggml_cuda_find_user(cgraph, cgraph->nodes[jcont], jcont, to);
            if (jre < 0 || cgraph->nodes[jre]->op != GGML_OP_RESHAPE || cgraph->nodes[jre]->ne[0] != m.hd) {
                return false;
            }
            idxs.push_back(jre);
            half_r[k] = cgraph->nodes[jre];
            jw = std::max(jw, jre);
        }
        const int jc3 = ggml_cuda_find_user(cgraph, half_r[0], jw, to);
        if (jc3 < 0) {
            return false;
        }
        const ggml_tensor * c3 = cgraph->nodes[jc3];
        if (c3->op != GGML_OP_CONCAT || ggml_get_op_params_i32(c3, 0) != 1 || c3->src[0] != half_r[0] ||
                c3->src[1] != half_r[1] || !ggml_are_same_shape(half_r[0], half_r[1])) {
            return false;
        }
        m.r  = half_r[0]->ne[1];
        m.nb = half_r[0]->ne[2];
        if (m.r*m.nb != n_read) {
            return false;
        }
        idxs.push_back(jc3);
        window = c3;
        jw = jc3;
    } else {
        const int jre = ggml_cuda_find_user(cgraph, g, jg, to);
        if (jre < 0 || cgraph->nodes[jre]->op != GGML_OP_RESHAPE || cgraph->nodes[jre]->ne[0] != W) {
            return false;
        }
        window = cgraph->nodes[jre];
        m.hd = W;
        m.r  = window->ne[1];
        m.nb = window->ne[2];
        if (m.r*m.nb != g->ne[1]) {
            return false;
        }
        idxs.push_back(jre);
        jw = jre;
    }
    const int jp = ggml_cuda_find_user(cgraph, window, jw, to);
    if (jp < 0) {
        return false;
    }
    const ggml_tensor * perm = cgraph->nodes[jp];
    if (perm->op != GGML_OP_PERMUTE || perm->ne[0] != window->ne[1] || perm->ne[1] != window->ne[0] ||
            perm->nb[0] != window->nb[1] || perm->nb[1] != window->nb[0]) {
        return false;
    }
    idxs.push_back(jp);
    const int jt = ggml_cuda_find_user(cgraph, perm, jp, to);
    if (jt < 0 || cgraph->nodes[jt]->op != GGML_OP_CONT) {
        return false;
    }
    m.t = cgraph->nodes[jt];
    idxs.push_back(jt);
    return true;
}

// The DeepSeek-V4 compressor: the values' half (from the CONCAT at i), the scores' half, SOFT_MAX of the scores, MUL of
//   the values by them, SUM_ROWS. 19 launches for an overlap compressor (8 without), here one at the SUM_ROWS: the
//   chain's nodes before it are skipped and the nodes in between that it does not own (the scores' matmul and position
//   bias) run where they are.
static bool ggml_cuda_dsv4_compress_match(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i,
        ggml_cuda_dsv4_compress_fold & f) {
    if (!cuda_ctx->stream_context().concurrent_events.empty()) {
        return false;
    }
    const int to = std::min(cgraph->n_nodes - 1, i + 96);
    std::vector<int> idxs;
    ggml_cuda_dsv4_half_match kv;
    if (!ggml_cuda_dsv4_parse_half(cgraph, i, to, kv, idxs)) {
        return false;
    }
    const int jm = ggml_cuda_find_user(cgraph, kv.t, idxs.back(), to);
    if (jm < 0) {
        return false;
    }
    const ggml_tensor * mul = cgraph->nodes[jm];
    if (mul->op != GGML_OP_MUL || mul->src[0] != kv.t || mul->src[1]->op != GGML_OP_SOFT_MAX ||
            !ggml_are_same_shape(mul, kv.t)) {
        return false;
    }
    const ggml_tensor * sm = mul->src[1];
    float sm_scale;
    float max_bias;
    memcpy(&sm_scale, (const float *) sm->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) sm->op_params + 1, sizeof(float));
    if (sm->src[1] != nullptr || sm->src[2] != nullptr || max_bias != 0.0f || !ggml_are_same_shape(sm, kv.t)) {
        return false;
    }
    // the scores' half: from the CONCAT its window transpose comes from
    int i_sm = -1;
    int i_c1 = -1;
    for (int j = i + 1; j < jm; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n == sm) {
            i_sm = j;
        }
        if (i_c1 < 0 && n->op == GGML_OP_CONCAT && std::find(idxs.begin(), idxs.end(), j) == idxs.end()) {
            i_c1 = j;
        }
    }
    if (i_sm < 0 || i_c1 < 0) {
        return false;
    }
    ggml_cuda_dsv4_half_match sc;
    if (!ggml_cuda_dsv4_parse_half(cgraph, i_c1, to, sc, idxs) || sc.t != sm->src[0] || sc.overlap != kv.overlap ||
            sc.hd != kv.hd || sc.r != kv.r || sc.nb != kv.nb) {
        return false;
    }
    const int js = ggml_cuda_find_user(cgraph, mul, jm, to);
    if (js < 0) {
        return false;
    }
    const ggml_tensor * sr = cgraph->nodes[js];
    if (sr->op != GGML_OP_SUM_ROWS || sr->src[0] != mul || sr->type != GGML_TYPE_F32 || !ggml_is_contiguous(sr) ||
            ggml_nelements(sr) != kv.hd*kv.nb) {
        return false;
    }
    const int64_t n = kv.overlap ? 2*kv.r : kv.r;
    const int     nsm = ggml_cuda_info().devices[cuda_ctx->device].nsm;
    // SUM_ROWS reduces a row of more than one warp in another order when its rows do not fill the GPU twice
    if (n > GGML_CUDA_DSV4_COMPRESS_MAX || kv.nb > 65535 || (n > WARP_SIZE && kv.hd*kv.nb/nsm < 2)) {
        return false;
    }
    idxs.push_back(i_sm);
    idxs.push_back(jm);
    idxs.push_back(js);
    std::sort(idxs.begin(), idxs.end());
    if (std::adjacent_find(idxs.begin(), idxs.end()) != idxs.end()) {
        return false;
    }
    std::vector<ggml_op> ops(idxs.size());
    for (size_t k = 0; k < idxs.size(); k++) {
        ops[k] = cgraph->nodes[idxs[k]]->op;
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs.data(), (int) idxs.size(), ops.data(), &js, 1)) {
        return false;
    }
    // the inputs are read at the SUM_ROWS: ggml-alloc keeps each alive up to the chain node that reads it, nothing after
    //   that may write it (its memory may be reused once the chain is done with it), nor may the output cover it
    const ggml_tensor * inputs[6]  = { kv.h.state, kv.h.cur, kv.h.idx, sc.h.state, sc.h.cur, sc.h.idx };
    const int           readers[6] = { kv.i_c1,    kv.i_c1,  kv.i_g,   sc.i_c1,    sc.i_c1,  sc.i_g   };
    for (int k = 0; k < 6; k++) {
        for (int j = readers[k] + 1; j < js; j++) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (std::binary_search(idxs.begin(), idxs.end(), j) || ggml_cuda_is_view_or_noop(n)) {
                continue;
            }
            if (ggml_cuda_tensors_overlap(n, inputs[k])) {
                return false;
            }
        }
        if (ggml_cuda_tensors_overlap(sr, inputs[k])) {
            return false;
        }
    }

    f.target   = js;
    f.skip.assign(idxs.begin(), idxs.end() - 1);
    f.kv       = kv.h;
    f.score    = sc.h;
    f.overlap  = kv.overlap;
    f.hd       = kv.hd;
    f.r        = kv.r;
    f.nb       = kv.nb;
    f.sm_scale = sm_scale;
    return true;
}

// Node i of a folded compressor: true when it is handled here (skipped, or the compressor launched at its SUM_ROWS).
static bool ggml_cuda_dsv4_compress_fold_step(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_DSV4_COMPRESS") != nullptr ||
        (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION")));
    if (disabled) {
        return false;
    }
    ggml_cuda_dsv4_compress_fold & f = cuda_ctx->dsv4_fold;
    if (f.target < 0) {
        if (cgraph->nodes[i]->op != GGML_OP_CONCAT || !ggml_cuda_dsv4_compress_match(cuda_ctx, cgraph, i, f)) {
            f = {};
            return false;
        }
        return true; // i is the chain's first node
    }
    // a fusion of the nodes in between must not have run past it
    GGML_ASSERT(i <= f.target);
    if (i == f.target) {
        ggml_cuda_dsv4_compress(*cuda_ctx, f, cgraph->nodes[i]);
        f = {};
        return true;
    }
    return std::binary_search(f.skip.begin(), f.skip.end(), i);
}

// A FLASH_ATTN_EXT whose output only the CONT, SIGMOID and MUL after it read, sigmoid(gate) * attn (the attention output
//   gate of qwen35, see ggml_cuda_should_fuse_cont_unary_mul): with a split KV the combine applies the gate and writes the
//   MUL output, so the attention output is not written; else the gate kernel runs after the attention.
static int ggml_cuda_try_fuse_fattn_gate(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_FATTN_GATE") != nullptr;
    ggml_tensor * fa = cgraph->nodes[i];
    if (disabled || fa->op != GGML_OP_FLASH_ATTN_EXT || fa->type != GGML_TYPE_F32 || (fa->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            !ggml_is_contiguous(fa) || ggml_cuda_graph_use_count(cgraph, fa) != 1) {
        return 0;
    }
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        j++;
    }
    if (j + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * cont  = cgraph->nodes[j];
    ggml_tensor * unary = cgraph->nodes[j + 1];
    ggml_tensor * mul   = cgraph->nodes[j + 2];
    if (!ggml_cuda_should_fuse_cont_unary_mul(cont, unary, mul) || ggml_get_unary_op(unary) != GGML_UNARY_OP_SIGMOID) {
        return 0;
    }
    const int outputs[1] = { j + 2 };
    const enum ggml_op ops[3] = { GGML_OP_CONT, GGML_OP_UNARY, GGML_OP_MUL };
    const ggml_tensor * attn = mul->src[0] == unary ? mul->src[1] : mul->src[0];
    // the gate rows are those of the attention output: one per head and token
    if (!ggml_can_fuse_subgraph(cgraph, j, 3, ops, outputs, 1) || cont->src[0]->ne[0] != fa->ne[0] ||
            (attn->op != GGML_OP_RESHAPE && attn->op != GGML_OP_VIEW) || attn->view_src != fa || attn->view_offs != 0 ||
            (attn->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_cuda_graph_use_count(cgraph, attn) != 1) {
        return 0;
    }

    static const bool no_gate_q8_1 = getenv("GGML_CUDA_NO_GATE_Q8_1") != nullptr;
    if (!no_gate_q8_1) {
        ggml_cuda_q8_1_request(cuda_ctx, cgraph, j + 2);
    }
    cuda_ctx->fattn_gate = { cont, mul, false };
    ggml_cuda_flash_attn_ext(*cuda_ctx, fa);
    const bool done = cuda_ctx->fattn_gate.done;
    cuda_ctx->fattn_gate = {};
    if (!done) {
        ggml_cuda_op_cont_unary_mul(*cuda_ctx, cont, unary, mul);
    }
    return j + 2 - i;
}

// DeepSeek V4 before a block: the MUL -> [VIEW] -> ADD -> SIGMOID -> SCALE chain at i of the stream weights that the
//   DSV4_HC_PRE after it mixes the streams with, as ggml_cuda_op_dsv4_hc_pre_coef fuses them: the number of nodes from i to
//   that DSV4_HC_PRE (the ADD at n - 3, the SCALE at n - 1), or 0
static int ggml_cuda_match_hc_pre_coef(const ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_MUL) {
        return 0;
    }
    for (const bool with_view : { true, false }) {
        const int n_ops = with_view ? 5 : 4;
        const int i_pre = i + n_ops;
        const bool ops_match = with_view ?
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_SCALE,
                GGML_OP_DSV4_HC_PRE }, { i_pre }) :
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_SCALE,
                GGML_OP_DSV4_HC_PRE }, { i_pre });
        if (!ops_match) {
            continue;
        }
        const ggml_tensor * add   = cgraph->nodes[i + n_ops - 3];
        const ggml_tensor * sig   = cgraph->nodes[i + n_ops - 2];
        const ggml_tensor * scale = cgraph->nodes[i + n_ops - 1];
        const ggml_tensor * pre   = cgraph->nodes[i_pre];
        const ggml_tensor * mixes = node->src[0];
        const ggml_tensor * m     = node->src[1];
        const ggml_tensor * a     = add->src[1];
        const ggml_tensor * x     = pre->src[0];
        const int64_t hc = x->ne[1];
        const bool ok = add->src[0] == node && sig->src[0] == add && scale->src[0] == sig &&
            (!with_view || a == cgraph->nodes[i + 1]) && ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID &&
            pre->src[1] == scale && ggml_get_op_params_i32(pre, 1) == 0 &&
            mixes->type == GGML_TYPE_F32 && m->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 &&
            mixes->ne[0] == hc && mixes->nb[0] == sizeof(float) && ggml_nrows(mixes) == mixes->ne[1] &&
            ggml_is_contiguous(m) && ggml_nrows(m) == 1 && hc % m->ne[0] == 0 &&
            ggml_is_contiguous(a) && ggml_nrows(a) == 1 && hc % a->ne[0] == 0 &&
            ggml_are_same_shape(node, mixes) && ggml_are_same_shape(add, node) && ggml_are_same_shape(scale, node) &&
            hc <= 4 && x->type == GGML_TYPE_F32 && x->ne[2] <= 8 && scale->ne[1] == x->ne[2] && pre->type == GGML_TYPE_F32 &&
            !ggml_cuda_tensors_overlap(pre, x) && !ggml_cuda_tensors_overlap(pre, mixes) &&
            !ggml_cuda_tensors_overlap(pre, m) && !ggml_cuda_tensors_overlap(pre, a);
        if (ok) {
            return n_ops;
        }
    }
    return 0;
}

// A hyper-connection combine (with comb) and the norm + mixes matvec of the next block that reads its streams as one row
//   (GLM-5.3, DeepSeek-V4), one token: one launch that also writes the combine's dst. fold: the nodes that make its post
//   weights and comb, computed in the launch too (see ggml_cuda_should_fold_dsv4_hc_post). The nodes after i the launch
//   covers, or 0.
static int ggml_cuda_try_fuse_hc_post(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i,
        const ggml_tensor * const * fold) {
    ggml_tensor * node = cgraph->nodes[i];
    if (node->op == GGML_OP_DSV4_HC_POST && i + 3 < cgraph->n_nodes) {
        ggml_tensor * flat = cgraph->nodes[i + 1];
        ggml_tensor * norm = cgraph->nodes[i + 2];
        ggml_tensor * mm   = cgraph->nodes[i + 3];
        if (ggml_cuda_is_view_or_noop(flat) && flat->src[0] == node && norm->src[0] == flat &&
                ggml_node_get_use_count(cgraph, i + 2) == 1 && (norm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                (norm->flags & GGML_TENSOR_FLAG_COMPUTE) && (mm->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                ggml_cuda_should_fuse_dsv4_hc_post_rms_norm_mul_mat(node, norm, mm)) {
            // and the stream weights, mix, norm and weight of the next block's input that read the mixes, in the last
            //   block of the launch: the views of the mixes and of the weights between
            int i_mul = i + 4;
            while (i_mul < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[i_mul])) {
                i_mul++;
            }
            static const bool no_hc_pre_coef = getenv("GGML_CUDA_NO_HC_PRE_COEF") != nullptr;
            const int n_coef = i_mul < cgraph->n_nodes && !no_hc_pre_coef ? ggml_cuda_match_hc_pre_coef(cgraph, i_mul) : 0;
            const int i_pre  = i_mul + n_coef;
            if (n_coef > 0 && i_pre + 2 < cgraph->n_nodes) {
                ggml_tensor * mul_w = cgraph->nodes[i_mul];
                ggml_tensor * add_w = cgraph->nodes[i_pre - 3];
                ggml_tensor * scl_w = cgraph->nodes[i_pre - 1];
                ggml_tensor * pre   = cgraph->nodes[i_pre];
                ggml_tensor * norm2 = cgraph->nodes[i_pre + 1];
                ggml_tensor * mul2  = cgraph->nodes[i_pre + 2];
                const int  pre_uses = ggml_node_get_use_count(cgraph, i_pre);
                const bool pre_out  = pre_uses > 1 || (pre->flags & GGML_TENSOR_FLAG_OUTPUT);
                if ((norm2->flags & GGML_TENSOR_FLAG_COMPUTE) && (mul2->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                        (norm2->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && ggml_node_get_use_count(cgraph, i_pre + 1) == 1 &&
                        pre_uses >= 1 && ggml_cuda_should_fuse_dsv4_hc_post_pre(node, mm, mul_w, add_w, scl_w, pre, norm2, mul2, pre_out)) {
                    ggml_cuda_q8_1_request(cuda_ctx, cgraph, i_pre + 2);
                    ggml_cuda_op_dsv4_hc_post_pre(*cuda_ctx, node, norm, mm, mul_w, add_w, scl_w, pre, pre_out, norm2, mul2, fold);
                    return i_pre + 2 - i;
                }
            }
            ggml_cuda_op_dsv4_hc_post_rms_norm_mul_mat(*cuda_ctx, node, norm, mm, fold);
            return 3;
        }
    }
    return 0;
}

// The MUL -> ADD -> SIGMOID -> SCALE at i of the post weights of the DSV4_HC_POST after it (the add's bias view between
//   the mul and the add, or not), as ggml_cuda_op_mul_add_sigmoid_scale fuses them: the number of nodes, or 0
static int ggml_cuda_match_mul_add_sigmoid_scale(const ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_MUL) {
        return 0;
    }
    for (const bool with_view : { true, false }) {
        const int n_ops = with_view ? 5 : 4;
        const bool ops_match = with_view ?
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_SCALE }, { i + 4 }) :
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_SCALE }, { i + 3 });
        if (!ops_match) {
            continue;
        }
        const ggml_tensor * add   = cgraph->nodes[i + n_ops - 3];
        const ggml_tensor * sig   = cgraph->nodes[i + n_ops - 2];
        const ggml_tensor * scale = cgraph->nodes[i + n_ops - 1];
        const ggml_tensor * x = node->src[0];
        const ggml_tensor * m = node->src[1];
        const ggml_tensor * a = add->src[1];
        const bool ok = add->src[0] == node && sig->src[0] == add && scale->src[0] == sig &&
            (!with_view || a == cgraph->nodes[i + 1]) &&
            ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID &&
            x->type == GGML_TYPE_F32 && m->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 &&
            node->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && sig->type == GGML_TYPE_F32 &&
            scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) &&
            ggml_are_same_shape(x, node) && ggml_are_same_shape(node, add) && ggml_are_same_shape(add, scale) &&
            x->nb[0] == sizeof(float) && ggml_nrows(x) == x->ne[1] &&
            ggml_is_contiguous(m) && ggml_nrows(m) == 1 && x->ne[0] % m->ne[0] == 0 &&
            ggml_is_contiguous(a) && ggml_nrows(a) == 1 && x->ne[0] % a->ne[0] == 0;
        if (ok) {
            return n_ops;
        }
    }
    return 0;
}

// The post weights (MUL..SCALE at i) and the DSV4_HC_COMB of a combine that ggml_cuda_try_fuse_hc_post takes, computed in
//   its launch: they are read by nothing else. The nodes after i the launch covers, or 0.
static int ggml_cuda_try_fuse_hc_post_fold(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const int i) {
    const int n_ms = ggml_cuda_match_mul_add_sigmoid_scale(cgraph, i);
    if (n_ms == 0) {
        return 0;
    }
    int j = i + n_ms;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        j++;
    }
    int k = j + 1;
    while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
        k++;
    }
    if (k >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * scale_p = cgraph->nodes[i + n_ms - 1];
    ggml_tensor * comb    = cgraph->nodes[j];
    ggml_tensor * post    = cgraph->nodes[k];
    if (comb->op != GGML_OP_DSV4_HC_COMB || post->op != GGML_OP_DSV4_HC_POST ||
            (comb->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || (post->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            ggml_node_get_use_count(cgraph, i + n_ms - 1) != 1 || ggml_node_get_use_count(cgraph, j) != 1 ||
            (scale_p->flags & GGML_TENSOR_FLAG_OUTPUT) || (comb->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * fold[4] = { cgraph->nodes[i], cgraph->nodes[i + n_ms - 3], scale_p, comb };
    if (!ggml_cuda_should_fold_dsv4_hc_post(post, fold[0], fold[1], fold[2], fold[3])) {
        return 0;
    }
    const int n = ggml_cuda_try_fuse_hc_post(cuda_ctx, cgraph, k, fold);
    return n > 0 ? k - i + n : 0;
}


static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_cuda_try_gather_expert_group(cuda_ctx, cgraph, i);

    if (ggml_cuda_fusion_disabled()) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    // a build_conv_state that the qkv projection ran (ggml_cuda_gdn_conv_fold), or a topk_moe that the router's launch
    //   ran (ggml_cuda_topk_moe_fold)
    if (node == cuda_ctx->conv_folded_concat) {
        cuda_ctx->conv_folded_concat = nullptr;
        return cuda_ctx->conv_folded_skip;
    }
    if (node == cuda_ctx->topk_folded_node) {
        cuda_ctx->topk_folded_node = nullptr;
        return cuda_ctx->topk_folded_skip;
    }

    if (node->op == GGML_OP_UNARY) {
        const int n_skip = ggml_cuda_try_fuse_qsa_cell_scores(cuda_ctx, cgraph, i);
        if (n_skip > 0) {
            return n_skip;
        }
    }

    if (node->op == GGML_OP_FILL) {
        const int n_skip = ggml_cuda_try_fuse_qsa_topk_mask(cuda_ctx, cgraph, i);
        if (n_skip > 0) {
            return n_skip;
        }
    }

    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
        const int n_skip = ggml_cuda_try_fuse_fattn_gate(cuda_ctx, cgraph, i);
        if (n_skip > 0) {
            return n_skip;
        }
    }

    // gathered rows added to a matrix (DeepSeek-V4's compressor position bias) or set into a state (its persisted rows)
    static const bool no_get_rows_fusion = getenv("GGML_CUDA_NO_GET_ROWS_FUSION") != nullptr;
    if (node->op == GGML_OP_GET_ROWS && !no_get_rows_fusion && (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
            ggml_node_get_use_count(cgraph, i) == 1) {
        if (ggml_can_fuse(cgraph, i, { GGML_OP_GET_ROWS, GGML_OP_ADD }) &&
                ggml_cuda_should_fuse_get_rows_add(node, cgraph->nodes[i + 1])) {
            ggml_cuda_op_get_rows_add(*cuda_ctx, node, cgraph->nodes[i + 1]);
            return 1;
        }
        int j = i + 1;
        while (j < cgraph->n_nodes && j <= i + 4 && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
            j++;
        }
        if (j < cgraph->n_nodes && j <= i + 4 && cgraph->nodes[j]->op == GGML_OP_SET_ROWS &&
                ggml_cuda_should_fuse_get_rows_set_rows(node, cgraph->nodes[j])) {
            // a second pair with the same indices right after it (the kv and score rows a DeepSeek-V4 compressor
            //   persists) goes into the same launch
            int i1 = j + 1;
            while (i1 < cgraph->n_nodes && i1 <= j + 4 && ggml_cuda_is_view_or_noop(cgraph->nodes[i1])) {
                i1++;
            }
            if (i1 < cgraph->n_nodes && i1 <= j + 4 && cgraph->nodes[i1]->op == GGML_OP_GET_ROWS &&
                    (cgraph->nodes[i1]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && ggml_node_get_use_count(cgraph, i1) == 1) {
                int j1 = i1 + 1;
                while (j1 < cgraph->n_nodes && j1 <= i1 + 4 && ggml_cuda_is_view_or_noop(cgraph->nodes[j1])) {
                    j1++;
                }
                if (j1 < cgraph->n_nodes && j1 <= i1 + 4 && cgraph->nodes[j1]->op == GGML_OP_SET_ROWS &&
                        ggml_cuda_should_fuse_get_rows_set_rows(cgraph->nodes[i1], cgraph->nodes[j1]) &&
                        ggml_cuda_can_pair_get_rows_set_rows(node, cgraph->nodes[j], cgraph->nodes[i1], cgraph->nodes[j1])) {
                    ggml_cuda_op_get_rows_set_rows(*cuda_ctx, node, cgraph->nodes[j], cgraph->nodes[i1], cgraph->nodes[j1]);
                    return j1 - i;
                }
            }
            ggml_cuda_op_get_rows_set_rows(*cuda_ctx, node, cgraph->nodes[j]);
            return j - i;
        }
    }

    if (node->op == GGML_OP_RMS_NORM || node->op == GGML_OP_ROPE || node->op == GGML_OP_MUL_MAT) {
        const int n_skip = ggml_cuda_try_fuse_attn_rot(cuda_ctx, cgraph, i);
        if (n_skip != 0) {
            return n_skip;
        }
    }

    if (node->op == GGML_OP_MUL_MAT) {
        const int n_skip = ggml_cuda_try_defer_glu(cuda_ctx, cgraph, i);
        if (n_skip != 0) {
            return n_skip;
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID) {
        const int n_skip = ggml_cuda_try_fuse_moe_down_shexp(cuda_ctx, cgraph, i);
        if (n_skip != 0) {
            return n_skip;
        }
    }

    if (node->op == GGML_OP_GET_ROWS) {
        const int n_skip = ggml_cuda_try_fuse_kpool_pool(cuda_ctx, cgraph, i);
        if (n_skip != 0) {
            return n_skip;
        }
    }

    // A run of scalar f32 copies of one shape and stride, with only views between them, in one launch: a
    //   recurrent layer keeps its conv window once per rollback slot, 6 copies of 30 KiB per layer at 6 tokens
    //   (288 launches of ~2.8 us per MTP verify on Qwen3.8-27B). The copies must not touch each other's data.
    if (node->op == GGML_OP_CPY && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_cpy_is_scalar_f32(node->src[0], node->src[1])) {
        const ggml_tensor * s0 = node->src[0];
        const ggml_tensor * s1 = node->src[1];
        const auto extent = [](const ggml_tensor * t) {
            size_t e = ggml_element_size(t);
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                e += (t->ne[d] - 1)*t->nb[d];
            }
            return e;
        };
        const auto overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
            const char * a0 = (const char *) a->data;
            const char * b0 = (const char *) b->data;
            return a0 < b0 + extent(b) && b0 < a0 + extent(a);
        };
        const ggml_tensor * srcs[GGML_CUDA_CPY_BATCH_MAX] = { s0 };
        ggml_tensor       * dsts[GGML_CUDA_CPY_BATCH_MAX] = { node->src[1] };
        int n    = 1;
        int last = i;
        for (int j = i + 1; j < cgraph->n_nodes && n < GGML_CUDA_CPY_BATCH_MAX; j++) {
            ggml_tensor * nj = cgraph->nodes[j];
            if (ggml_cuda_is_view_or_noop(nj) || (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                continue;
            }
            if (nj->op != GGML_OP_CPY) {
                break;
            }
            const ggml_tensor * t0 = nj->src[0];
            ggml_tensor       * t1 = nj->src[1];
            bool ok = t0->type == s0->type && t1->type == s1->type;
            for (int d = 0; ok && d < GGML_MAX_DIMS; d++) {
                ok = t0->ne[d] == s0->ne[d] && t0->nb[d] == s0->nb[d] && t1->ne[d] == s1->ne[d] && t1->nb[d] == s1->nb[d];
            }
            // no copy may read or write what another one of the run writes
            for (int k = 0; ok && k < n; k++) {
                ok = !overlap(t1, dsts[k]) && !overlap(t1, srcs[k]) && !overlap(t0, dsts[k]);
            }
            if (!ok) {
                break;
            }
            srcs[n] = t0;
            dsts[n] = t1;
            n++;
            last = j;
        }
        if (n > 1) {
            ggml_cuda_cpy_scalar_f32_batch(*cuda_ctx, srcs, dsts, n);
            return last - i;
        }
    }

    // build_lora_mm_id's per-expert scale (gemma4): the scale vector REPEATed over the tokens and GET_ROWS by the expert
    //   ids right before the weighted reduction, which reads the table through the ids instead (two launches less)
    if (node->op == GGML_OP_REPEAT && i + 2 < cgraph->n_nodes) {
        int i_gr = i + 1; // views may come between (the ids)
        while (i_gr + 1 < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[i_gr])) {
            i_gr++;
        }
        const ggml_tensor * table = node->src[0];
        const ggml_tensor * gr    = cgraph->nodes[i_gr];
        ggml_cuda_moe_weighted_reduction_match match;
        if (i_gr + 1 < cgraph->n_nodes &&
                gr->op == GGML_OP_GET_ROWS && gr->src[0] == node && table->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(table) && table->ne[0] == 1 && ggml_nelements(table) == table->ne[1] &&
                node->ne[0] == 1 && node->ne[1] == table->ne[1] && node->ne[3] == 1 &&
                ggml_cuda_graph_use_count(cgraph, node) == 1 && ggml_cuda_graph_use_count(cgraph, gr) == 1 &&
                !(node->flags & GGML_TENSOR_FLAG_OUTPUT) && !(gr->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                ggml_cuda_match_moe_weighted_reduction(cgraph, i_gr + 1, match) && match.expert_scale == gr) {
            const ggml_tensor * ids      = gr->src[1];
            const int64_t       n_tokens = match.experts->ne[2] * match.experts->ne[3];
            const bool over_w   = ggml_cuda_tensors_overlap(match.dst, match.weights);
            const bool over_ids = ggml_cuda_tensors_overlap(match.dst, ids);
            bool       safe     = ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t) &&
                ids->ne[0] == match.experts->ne[1] && ggml_nrows(ids) == n_tokens && (!over_ids || n_tokens == 1);
            if (safe && ggml_cuda_tensors_overlap(match.dst, match.experts)) {
                const int64_t off = (const char *) match.dst->data - (const char *) match.experts->data;
                safe = n_tokens == 1 && off >= 0 && off % (int64_t) match.experts->nb[1] == 0 &&
                    off / (int64_t) match.experts->nb[1] < match.experts->ne[1];
            }
            if (safe) {
                ggml_cuda_op_moe_weighted_reduction(*cuda_ctx, match.experts, table, match.weights, match.dst,
                    over_w && n_tokens > 1, nullptr, nullptr, ids);
                return i_gr - i + match.node_count;
            }
        }
    }

    // the shared expert after the routed gate/up/GLU (graph_optimize moves it there, so never under -sm tensor), run as
    //   one more channel of the routed launch
    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_mmvq_shared_expert_supported(ggml_cuda_info().devices[cuda_ctx->device].cc) &&
            ggml_cuda_match_shared_expert(cgraph, i, i + 3) &&
            ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1], true)) {
        const int outputs[] = { i + 2, i + 5 };
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, outputs, 2)) {
            ggml_tensor * routed = cgraph->nodes[i + 2];
            ggml_tensor * shared = cgraph->nodes[i + 5];
            const ggml_tensor * up = routed->src[1];
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate = routed->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(routed);
            fusion.glu_limit = ggml_get_op_params_f32(routed, 3);
            fusion.shared_up = shared->src[1]->src[0];
            fusion.shared_gate = shared->src[0]->src[0];
            fusion.shared_dst = shared;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], routed, &fusion);
            return 5;
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            // The kernel reads only the experts, the expert scales and the router weights. Under -sm tensor the meta
            //   backend has no graph_optimize, so the alloc deps of ggml_backend_cuda_graph_optimize are not made and
            //   ggml-alloc can put dst over the weights, which the elided MUL read last: a block reads its token's
            //   weights before it writes, and with several tokens the kernel reads a copy. Over the experts, dst can
            //   only sit on the rows of a lone token, which the threads read before they write the same columns.
            const int64_t n_tokens  = match.experts->ne[2] * match.experts->ne[3];
            const bool    over_w    = ggml_cuda_tensors_overlap(match.dst, match.weights) ||
                (match.expert_scale != nullptr && ggml_cuda_tensors_overlap(match.dst, match.expert_scale));
            bool          safe      = true;
            if (ggml_cuda_tensors_overlap(match.dst, match.experts)) {
                const int64_t off = (const char *) match.dst->data - (const char *) match.experts->data;
                safe = n_tokens == 1 && off >= 0 && off % (int64_t) match.experts->nb[1] == 0 &&
                    off / (int64_t) match.experts->nb[1] < match.experts->ne[1];
            }
            // a sum that ends the session subgraph goes to the all-reduce after it, as in ggml_cuda_try_fuse_moe_down_shexp
            //   without the shared expert; all of its blocks read the weights, so dst may not sit on them
            if (safe && n_tokens == 1 && match.expert_scale == nullptr && !over_w && cuda_ctx->session_red.dst == nullptr &&
                    ggml_cuda_moe_sum_to_allreduce(cuda_ctx, cgraph, i + match.node_count, match, match.dst)) {
                ggml_backend_cuda_context::session_reduction_t & red = cuda_ctx->session_red;
                red.experts = match.experts;
                red.weights = match.weights;
                red.dst     = match.dst;
                return match.node_count - 1;
            }
            if (safe) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst, over_w && n_tokens > 1);
                return match.node_count - 1;
            }
        }

        // DeepSeek V4 before a block: the stream weights (the chain below) computed inside the DSV4_HC_PRE that mixes the
        //   streams with them, see ggml_cuda_op_dsv4_hc_pre_coef. Every column recomputes them, so only small batches
        //   (a prefill ubatch was 0.3% slower).
        static const bool no_hc_pre_coef = getenv("GGML_CUDA_NO_HC_PRE_COEF") != nullptr;
        const int n_coef = no_hc_pre_coef ? 0 : ggml_cuda_match_hc_pre_coef(cgraph, i);
        if (n_coef > 0) {
            const int i_pre = i + n_coef;
            ggml_tensor * add   = cgraph->nodes[i_pre - 3];
            ggml_tensor * scale = cgraph->nodes[i_pre - 1];
            ggml_tensor * pre   = cgraph->nodes[i_pre];
            ggml_cuda_q8_1_request(cuda_ctx, cgraph, i_pre);
            ggml_cuda_op_dsv4_hc_pre_coef(*cuda_ctx, node, add, scale, pre);
            return i_pre - i;
        }

        // the post weights and comb of a combine computed in its fused launch
        {
            const int n = ggml_cuda_try_fuse_hc_post_fold(cuda_ctx, cgraph, i);
            if (n > 0) {
                return n;
            }
        }

        // DeepSeek V4 hyper-connection coefficients: 172 of these four-kernel chains per token. The add's
        //   bias is a view of a weight, and that view sits between the mul and the add in node order.
        for (const bool with_view : { true, false }) {
            const int n_ops = with_view ? 5 : 4;
            const bool ops_match = with_view ?
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_SCALE }, { i + 4 }) :
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_SCALE }, { i + 3 });
            if (!ops_match) {
                continue;
            }
            ggml_tensor * add   = cgraph->nodes[i + n_ops - 3];
            ggml_tensor * sig   = cgraph->nodes[i + n_ops - 2];
            ggml_tensor * scale = cgraph->nodes[i + n_ops - 1];
            const ggml_tensor * x = node->src[0];
            const ggml_tensor * m = node->src[1];
            const ggml_tensor * a = add->src[1];
            const bool ok = add->src[0] == node && sig->src[0] == add && scale->src[0] == sig &&
                (!with_view || a == cgraph->nodes[i + 1]) &&
                ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID &&
                x->type == GGML_TYPE_F32 && m->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 &&
                node->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && sig->type == GGML_TYPE_F32 &&
                scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale) &&
                ggml_are_same_shape(x, node) && ggml_are_same_shape(node, add) && ggml_are_same_shape(add, scale) &&
                x->nb[0] == sizeof(float) && ggml_nrows(x) == x->ne[1] &&
                ggml_is_contiguous(m) && ggml_nrows(m) == 1 && x->ne[0] % m->ne[0] == 0 &&
                ggml_is_contiguous(a) && ggml_nrows(a) == 1 && x->ne[0] % a->ne[0] == 0;
            const int out_nodes[] = { i + n_ops - 1 };
            if (ok && ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                ggml_cuda_op_mul_add_sigmoid_scale(*cuda_ctx, node, add, scale);
                return n_ops - 1;
            }
        }
    }

    if (node->op == GGML_OP_NORM) {
        ggml_cuda_rwkv7_output_fusion fusion;
        const int nodes_to_skip = ggml_cuda_try_rwkv7_output_fusion(cgraph, i, fusion);
        const int out_nodes[]   = { i + nodes_to_skip };
        if (nodes_to_skip > 0 &&
            ggml_cuda_check_fusion_memory_ranges(cgraph, i, nodes_to_skip + 1, out_nodes, 1)) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused RWKV7 output preparation for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_rwkv7_output_fused(
                *cuda_ctx,
                fusion.norm,
                fusion.norm_weight,
                fusion.norm_bias,
                fusion.k,
                fusion.r,
                fusion.r_k,
                fusion.v,
                fusion.gate,
                fusion.dst);
            return nodes_to_skip;
        }

        // a layer norm with weight and bias (the glm5-next indexer key): three launches in one
        static const bool no_norm_mul_add = getenv("GGML_CUDA_NO_NORM_MUL_ADD") != nullptr;
        if (!no_norm_mul_add && i + 2 < cgraph->n_nodes &&
                ggml_cuda_should_fuse_norm_mul_add(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
            const enum ggml_op ops[3] = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
            const int outputs[1] = { i + 2 };
            if (ggml_can_fuse_subgraph(cgraph, i, 3, ops, outputs, 1)) {
                ggml_cuda_op_norm_mul_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
                return 2;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_gated_delta_net_l2 l2;
            const bool fuse_l2 = ggml_cuda_gdn_l2_fusion(cgraph, i, l2);
            // the GET_ROWS of the start state was not run, see ggml_cuda_gdn_gather_is_folded
            ggml_cuda_gated_delta_net_gather gather;
            const int n_gather = ggml_cuda_gdn_gather_fusion(cuda_ctx, cgraph, i, nodes_to_skip, gather);
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy, fuse_l2 ? &l2 : nullptr,
                                                     n_gather >= 0 ? &gather : nullptr);
            nodes_to_skip += n_gather >= 0 ? n_gather : 0;
            return nodes_to_skip;
        }
    }

    // ssm_scan -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_SSM_SCAN) {
        ggml_cuda_ssm_scan_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_ssm_scan_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused ssm_scan snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_ssm_scan_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_plan p;
        if (ggml_cuda_topk_moe_plan_at(cgraph, i, p) &&
                ggml_cuda_topk_moe_launch(cuda_ctx, cgraph, i, p.n_ops, p.out_nodes, p.logits, p.weights, p.ids, p.clamp, p.scale,
                    p.bias, p.args)) {
            return p.n_ops - 1;
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n, true)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n, true)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n, true)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                // the GLU deferred to this launch, see ggml_cuda_try_defer_glu
                ggml_backend_cuda_context::deferred_glu & d = cuda_ctx->glu_deferred;
                if (op == GGML_OP_MUL_MAT_ID && d.target == i) {
                    fusion_data.sib      = d.up->src[0];
                    fusion_data.sib_gate = d.gate->src[0];
                    fusion_data.sib_dst  = d.glu;
                    d = {};
                }

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            const bool cache_pair = op == GGML_OP_MUL_MAT_ID && ggml_cuda_expert_cache_glu_pair(up, gate);
            if (cache_pair || ggml_cuda_should_fuse_mul_mat_vec_q(up, true)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                // the GLU deferred to this launch, see ggml_cuda_try_defer_glu
                ggml_backend_cuda_context::deferred_glu & d = cuda_ctx->glu_deferred;
                if (op == GGML_OP_MUL_MAT_ID && d.target == i) {
                    fusion_data.sib      = d.up->src[0];
                    fusion_data.sib_gate = d.gate->src[0];
                    fusion_data.sib_dst  = d.glu;
                    d = {};
                }

                // pools of cached experts, each read through its own remapped ids
                ggml_tensor remap[2];
                if (cache_pair) {
                    ggml_cuda_expert_cache_remap(*cuda_ctx, src0, ids, &remap[0]);
                    ggml_cuda_expert_cache_remap(*cuda_ctx, gate->src[0], ids, &remap[1]);
                    fusion_data.gate_ids = &remap[1];
                    ids = &remap[0];
                }

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_f_glu(up, gate)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_f_glu(*cuda_ctx, src0, src1, &fusion_data, glu);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat, then times the sigmoid of one value, plus another tensor, at one token: Qwen3.5-MoE adds its gated shared
    //   expert to the routed experts like this. Each thread reads the row of the added tensor it then writes.
    if (node->op == GGML_OP_MUL_MAT) {
        constexpr ggml_op ops[] = { GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD };
        const int out_nodes[] = { i + 3 };
        if (ggml_can_fuse_subgraph(cgraph, i, 4, ops, out_nodes, 1) &&
                ggml_get_unary_op(cgraph->nodes[i + 1]) == GGML_UNARY_OP_SIGMOID) {
            const ggml_tensor * sig = cgraph->nodes[i + 1];
            const ggml_tensor * mul = cgraph->nodes[i + 2];
            ggml_tensor *       add = cgraph->nodes[i + 3];
            const ggml_tensor * g   = sig->src[0];
            const ggml_tensor * acc = add->src[0] == mul ? add->src[1] : add->src[0];
            if (((mul->src[0] == node && mul->src[1] == sig) || (mul->src[0] == sig && mul->src[1] == node)) &&
                    (add->src[0] == mul || add->src[1] == mul) &&
                    g->type == GGML_TYPE_F32 && ggml_nelements(g) == 1 &&
                    acc->type == GGML_TYPE_F32 && ggml_is_contiguous(acc) && ggml_are_same_shape(acc, node) &&
                    ggml_is_contiguous(add) && ggml_cuda_should_fuse_mul_mat_vec_q(node) && !ggml_cuda_op_mul_mat_use_fwht(node) &&
                    !ggml_cuda_tensors_overlap(add, g) && ggml_cuda_same_or_disjoint(add, acc)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.x_bias    = acc;
                fusion_data.x_sigmoid = g;
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, node->src[0], node->src[1], nullptr, add, &fusion_data);
                return 3;
            }
        }
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // A MUL_MAT of one f32 row right after another f32 MUL_MAT of the same input and width runs as one more row
    //   of the first one's mmvf launch, at 1-4 tokens: Qwen3.5-MoE expands its shared expert gate right after its
    //   router (one 1-block launch per layer, ~3 us at one token, 5 us at the 4 of a verify step). With no node between
    //   the two, the second one's dst holds nothing live when it is written early. GGML_CUDA_NO_SIBLING_ROW=1 runs them
    //   apart.
    if (node->op == GGML_OP_MUL_MAT && i + 1 < cgraph->n_nodes && cuda_ctx->stream_context().concurrent_events.empty()) {
        static const bool no_sibling_row = getenv("GGML_CUDA_NO_SIBLING_ROW") != nullptr;
        ggml_tensor * sib = cgraph->nodes[i + 1];
        if (!no_sibling_row && sib->op == GGML_OP_MUL_MAT && sib->src[1] == node->src[1] &&
                node->src[0]->type == GGML_TYPE_F32 && sib->src[0]->type == GGML_TYPE_F32 &&
                sib->src[0]->ne[0] == node->src[0]->ne[0] && ggml_nrows(sib->src[0]) == 1 &&
                ggml_is_contiguous(sib->src[0]) && node->ne[1] <= 4 && ggml_nrows(node) == node->ne[1] &&
                ggml_is_contiguous(node) && ggml_is_contiguous(sib) && ggml_nelements(sib) == node->ne[1] &&
                !ggml_cuda_op_mul_mat_use_fwht(node) && !ggml_cuda_op_mul_mat_use_fwht(sib) &&
                (node->ne[1] == 1 ? ggml_cuda_should_fuse_mul_mat_vec_f(node) :
                    !ggml_cuda_expert_cache_is(node->src[0]) &&
                    ggml_cuda_mul_mat_choose(*cuda_ctx, node->src[0], node->src[1], node) == GGML_CUDA_MUL_MAT_PATH_MMVF) &&
                ggml_cuda_mul_mat_choose(*cuda_ctx, sib->src[0], sib->src[1], sib) == GGML_CUDA_MUL_MAT_PATH_MMVF &&
                !ggml_cuda_tensors_overlap(sib, node) && !ggml_cuda_tensors_overlap(sib, node->src[1])) {
            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.sib     = sib->src[0];
            fusion_data.sib_dst = sib;
            // and the expert selection of the router logits after them
            ggml_cuda_topk_moe_fold(cuda_ctx, cgraph, i, sib, i + 1, fusion_data);
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, node->src[0], node->src[1], nullptr, node, &fusion_data);
            return 1;
        }
    }

    // A one-token MUL_MAT and the next one or two MUL_MATs of the same input, type and K with only views between them
    //   (the qkv and z projections of a Qwen3.5 gated delta net layer, the Q, K and V projections of an attention layer):
    //   one mmvq or mmvf launch, the siblings' rows after the first's. Nothing runs between them, so a sibling's dst
    //   holds nothing live when it is written early.
    const auto sibling_kernel = [&](const ggml_tensor * t) -> int { // 1: mmvq, 2: mmvf, 0: neither
        if (ggml_cuda_should_fuse_mul_mat_vec_q(t)) {
            return ggml_cuda_op_mul_mat_use_fwht(t) ? 0 : 1;
        }
        if (ggml_cuda_should_fuse_mul_mat_vec_f(t) && cuda_ctx->stream_context().concurrent_events.empty() &&
                ggml_cuda_mul_mat_choose(*cuda_ctx, t->src[0], t->src[1], t) == GGML_CUDA_MUL_MAT_PATH_MMVF) {
            return 2;
        }
        return 0;
    };
    // 3: mmvf of 2-6 tokens with the 16-byte loads and rows in blocks of 4 (the verify batch of a draft)
    const auto sibling_kernel_mt = [&](const ggml_tensor * t) -> int {
        const ggml_tensor * w = t->src[0];
        const ggml_tensor * y = t->src[1];
        const bool ok = (w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16) && ggml_nrows(w) == w->ne[1] &&
            w->ne[1] % 4 == 0 && w->ne[0] % 8 == 0 && w->nb[1] % 16 == 0 && (uintptr_t) w->data % 16 == 0 &&
            y->ne[1] >= 2 && y->ne[1] <= 6 && ggml_nrows(y) == y->ne[1] && y->nb[1] % 16 == 0 &&
            (uintptr_t) y->data % 16 == 0 && !ggml_cuda_expert_cache_is(w) &&
            cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_mul_mat_choose(*cuda_ctx, w, y, t) == GGML_CUDA_MUL_MAT_PATH_MMVF;
        return ok ? 3 : 0;
    };
    // 4: mmvq of 2-4 tokens (the verify batch of a draft: qwen4exp's hyper-connection down projection and its 4-row inject,
    //   the qkv and gate projections of a gated delta net layer), rows in pairs
    const auto sibling_kernel_mmvq_mt = [&](const ggml_tensor * t) -> int {
        const ggml_tensor * w = t->src[0];
        const ggml_tensor * y = t->src[1];
        const bool bad_padding_clear = ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_nbytes(w) != ggml_backend_buffer_get_alloc_size(w->buffer, w) && w->view_src;
        const bool ok = ggml_is_quantized(w->type) && !bad_padding_clear && ggml_nrows(w) == w->ne[1] && w->ne[1] % 2 == 0 &&
            y->ne[1] >= 2 && y->ne[1] <= 4 && ggml_nrows(y) == y->ne[1] && !ggml_cuda_expert_cache_is(w) &&
            !ggml_cuda_op_mul_mat_use_fwht(t) && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_mul_mat_choose(*cuda_ctx, w, y, t) == GGML_CUDA_MUL_MAT_PATH_MMVQ;
        return ok ? 4 : 0;
    };
    static const bool no_sibling_rows_mt   = getenv("GGML_CUDA_NO_SIBLING_ROWS_MT") != nullptr;
    static const bool no_sibling_rows_mmvq = getenv("GGML_CUDA_NO_SIBLING_ROWS_MMVQ_MT") != nullptr;
    const auto sibling_kernel_any_mt = [&](const ggml_tensor * t) -> int {
        if (no_sibling_rows_mt) {
            return 0;
        }
        const int k = sibling_kernel_mt(t);
        return k != 0 || no_sibling_rows_mmvq ? k : sibling_kernel_mmvq_mt(t);
    };
    const int node_sibling_kernel = node->op == GGML_OP_MUL_MAT && node->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(node) ?
        (ggml_nrows(node->src[1]) == 1 ? sibling_kernel(node) : sibling_kernel_any_mt(node)) : 0;
    if (node_sibling_kernel != 0) {
        static const bool no_sibling_rows  = getenv("GGML_CUDA_NO_SIBLING_ROWS") != nullptr;
        static const bool no_sibling_rows3 = getenv("GGML_CUDA_NO_SIBLING_ROWS3") != nullptr;
        static const bool no_sibling_rows4 = getenv("GGML_CUDA_NO_SIBLING_ROWS4") != nullptr;
        const auto next_node = [&](int n) {
            n++;
            while (n < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[n])) {
                n++;
            }
            return n;
        };
        const auto reads = [](const ggml_tensor * n, const ggml_tensor * t) {
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (n->src[s] != nullptr && ggml_cuda_tensors_overlap(n->src[s], t)) {
                    return true;
                }
            }
            return false;
        };
        // mmvf takes a third sibling
        const int max_sibs = no_sibling_rows3 ? 1 : (node_sibling_kernel == 2 || node_sibling_kernel == 3) && !no_sibling_rows4 ? 3 : 2;
        ggml_tensor * sibs[3] = { nullptr, nullptr, nullptr };
        int n_sibs = 0;
        int last   = i;
        while (!no_sibling_rows && n_sibs < max_sibs) {
            const int j = next_node(last);
            const int k = next_node(j);
            ggml_tensor * sib = j < cgraph->n_nodes ? cgraph->nodes[j] : nullptr;
            // the node after it must not read it: it may fuse with it (not past one token for mmvq, which fuses no gate
            //   or epilogue there: a shared expert's gate and up before their GLU)
            if (sib == nullptr || sib->op != GGML_OP_MUL_MAT || !(sib->flags & GGML_TENSOR_FLAG_COMPUTE) ||
                    sib->src[1]->data != node->src[1]->data || !ggml_are_same_shape(sib->src[1], node->src[1]) ||
                    !ggml_are_same_stride(sib->src[1], node->src[1]) || sib->src[0]->type != node->src[0]->type ||
                    sib->src[0]->ne[0] != node->src[0]->ne[0] || sib->src[0]->nb[1] != node->src[0]->nb[1] ||
                    ggml_nrows(sib->src[0]) != sib->src[0]->ne[1] || !ggml_is_contiguous(sib) ||
                    (node_sibling_kernel >= 3 ? sibling_kernel_any_mt(sib) : sibling_kernel(sib)) != node_sibling_kernel ||
                    ggml_cuda_tensors_overlap(sib, node) || ggml_cuda_tensors_overlap(sib, node->src[1]) ||
                    (n_sibs > 0 && ggml_cuda_tensors_overlap(sib, sibs[0])) ||
                    (n_sibs > 1 && ggml_cuda_tensors_overlap(sib, sibs[1])) ||
                    (node_sibling_kernel != 4 && k < cgraph->n_nodes && reads(cgraph->nodes[k], sib))) {
                break;
            }
            sibs[n_sibs++] = sib;
            last = j;
        }
        if (n_sibs > 0) {
            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.sib     = sibs[0]->src[0];
            fusion_data.sib_dst = sibs[0];
            if (n_sibs > 1) {
                fusion_data.sib2     = sibs[1]->src[0];
                fusion_data.sib2_dst = sibs[1];
            }
            if (n_sibs > 2) {
                fusion_data.sib3     = sibs[2]->src[0];
                fusion_data.sib3_dst = sibs[2];
            }
            if (node_sibling_kernel == 1 || node_sibling_kernel == 4) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, node->src[0], node->src[1], nullptr, node, &fusion_data);
            } else {
                fusion_data.sib_rows = true;
                ggml_cuda_mul_mat_vec_f(*cuda_ctx, node->src[0], node->src[1], nullptr, node, &fusion_data);
            }
            return last - i;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_GET_ROWS_MEAN, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_get_rows_mean_norm_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 2], cgraph->nodes[i + 3], cgraph->nodes[i + 5]);
        return 5;
    }

    // a hyper-connection combine (with comb) and the norm + mixes matvec of the next block, see ggml_cuda_try_fuse_hc_post
    if (node->op == GGML_OP_DSV4_HC_POST) {
        const int n = ggml_cuda_try_fuse_hc_post(cuda_ctx, cgraph, i, nullptr);
        if (n > 0) {
            return n;
        }
    }

    // DeepSeek-V4's hyper-connection mixes: a weightless norm of one long row and a few f32 rows of w over it
    //   (not ggml_can_fuse: it wants the nodes of the same shape)
    if (node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes && ggml_node_get_use_count(cgraph, i) == 1 &&
            (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && (cgraph->nodes[i + 1]->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            ggml_cuda_should_fuse_rms_norm_mul_mat(node, cgraph->nodes[i + 1])) {
        ggml_cuda_op_rms_norm_mul_mat(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    // the norms (and ropes) of the Q, K and V of one token, two or three of the fusions below with only views between
    //   them, or after the first the plain SET_ROWS of a V into its cache: one launch
    if (node->op == GGML_OP_RMS_NORM) {
        static const bool no_norm_rope_multi = getenv("GGML_CUDA_NO_NORM_ROPE_MULTI") != nullptr;
        // the fusion at j as { rms_norm, mul, rope, set_rows } and its last node, else -1
        const auto match = [&](const int j, ggml_tensor ** t) -> int {
            ggml_tensor ** n = cgraph->nodes;
            if (ggml_cuda_can_fuse(cgraph, j, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
                t[0] = n[j]; t[1] = n[j + 1]; t[2] = n[j + 2]; t[3] = n[j + 4];
                return j + 4;
            }
            if (ggml_cuda_can_fuse(cgraph, j, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
                t[0] = n[j]; t[1] = n[j + 1]; t[2] = n[j + 2]; t[3] = nullptr;
                return j + 2;
            }
            if (ggml_cuda_can_fuse(cgraph, j, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ROPE }, {})) {
                t[0] = n[j]; t[1] = n[j + 1]; t[2] = n[j + 3]; t[3] = nullptr;
                return j + 3;
            }
            if (ggml_cuda_can_fuse(cgraph, j, { GGML_OP_RMS_NORM, GGML_OP_ROPE }, {})) {
                t[0] = n[j]; t[1] = nullptr; t[2] = n[j + 1]; t[3] = nullptr;
                return j + 1;
            }
            if (ggml_cuda_can_fuse(cgraph, j, { GGML_OP_RMS_NORM, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
                t[0] = n[j]; t[1] = nullptr; t[2] = nullptr; t[3] = n[j + 2];
                return j + 2;
            }
            return -1;
        };
        ggml_tensor * segs[3][4];
        int n_segs = 0;
        int last   = -1;
        for (int j = i; !no_norm_rope_multi && n_segs < 3 && j < cgraph->n_nodes; ) {
            ggml_tensor * n = cgraph->nodes[j];
            int end = -1;
            if (n->op == GGML_OP_RMS_NORM) {
                end = match(j, segs[n_segs]);
            } else if (n_segs > 0 && n->op == GGML_OP_SET_ROWS && (n->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                    n->src[0]->type == GGML_TYPE_F32 && ggml_nrows(n->src[0]) == 1) {
                segs[n_segs][0] = nullptr;
                segs[n_segs][1] = nullptr;
                segs[n_segs][2] = nullptr;
                segs[n_segs][3] = n;
                end = j;
            }
            if (end < 0) {
                break;
            }
            last = end;
            n_segs++;
            j = end + 1;
            while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
                j++;
            }
        }
        // all run at once: no output may sit on the input or the output of another
        const auto out = [&](const int s) {
            return segs[s][3] ? segs[s][3] : segs[s][2] ? segs[s][2] : segs[s][1] ? segs[s][1] : segs[s][0];
        };
        const auto in = [&](const int s) {
            return segs[s][0] ? segs[s][0]->src[0] : segs[s][3]->src[0];
        };
        bool apart = true;
        for (int s = 0; s < n_segs; ++s) {
            for (int t = 0; t < n_segs; ++t) {
                if (s != t && (ggml_cuda_tensors_overlap(out(s), in(t)) || ggml_cuda_tensors_overlap(out(s), out(t)))) {
                    apart = false;
                }
            }
        }
        if (n_segs >= 2 && apart && ggml_cuda_op_rms_norm_mul_rope_multi(*cuda_ctx, segs, n_segs)) {
            return last - i;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 3], nullptr);
        return 3;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, nullptr, cgraph->nodes[i + 1], nullptr);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, nullptr, nullptr, cgraph->nodes[i + 2]);
        return 2;
    }

    // gemma4's MoE block ends in the post-norms of the dense MLP and of the experts, summed, before the sandwich norms
    //   below: the sum is made in the same launch
    if (node->op == GGML_OP_RMS_NORM && i + 9 < cgraph->n_nodes && cgraph->nodes[i + 2]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[i + 4]->op == GGML_OP_ADD && cgraph->nodes[i + 5]->op == GGML_OP_RMS_NORM) {
        const int  j         = i + 5;
        const bool has_scale = cgraph->nodes[j + 3]->op == GGML_OP_MUL;
        const int  n_ops     = 5 + (has_scale ? 6 : 5);
        if (i + n_ops <= cgraph->n_nodes &&
                ggml_cuda_should_fuse_rms_norm_add_rms_norm(cgraph->nodes[j], cgraph->nodes[j + 1], cgraph->nodes[j + 2],
                    has_scale ? cgraph->nodes[j + 3] : nullptr, cgraph->nodes[i + n_ops - 2], cgraph->nodes[i + n_ops - 1]) &&
                ggml_cuda_should_fuse_rms_norm_pair_add(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3],
                    cgraph->nodes[i + 4], cgraph->nodes[j], cgraph->nodes[i + n_ops - 3], cgraph->nodes[i + n_ops - 1])) {
            const int outputs[2] = { i + n_ops - 3, i + n_ops - 1 };
            const enum ggml_op ops_scale[11] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD,
                GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_MUL, GGML_OP_RMS_NORM, GGML_OP_MUL };
            const enum ggml_op ops[10]       = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD,
                GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL };
            if (ggml_can_fuse_subgraph(cgraph, i, n_ops, has_scale ? ops_scale : ops, outputs, 2)) {
                ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + n_ops - 1);
                ggml_cuda_op_rms_norm_add_rms_norm(*cuda_ctx, cgraph->nodes[j], cgraph->nodes[j + 1], cgraph->nodes[j + 2],
                    has_scale ? cgraph->nodes[j + 3] : nullptr, cgraph->nodes[i + n_ops - 2], cgraph->nodes[i + n_ops - 1],
                    node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3]);
                return n_ops - 1;
            }
        }
    }

    // the attention post-norm of a gemma4 MoE layer, its residual add and the router input, rms_norm(y) * s * w: the
    //   router norm in the launch of the post-norm, one launch less per layer and token
    if (node->op == GGML_OP_RMS_NORM && i + 5 < cgraph->n_nodes && cgraph->nodes[i + 3]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[i + 4]->op == GGML_OP_SCALE &&
            ggml_cuda_should_fuse_rms_norm_add_rms_norm(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr,
                cgraph->nodes[i + 3], cgraph->nodes[i + 5], cgraph->nodes[i + 4])) {
        const int outputs[2] = { i + 2, i + 5 };
        const enum ggml_op ops[6] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_SCALE, GGML_OP_MUL };
        if (ggml_can_fuse_subgraph(cgraph, i, 6, ops, outputs, 2)) {
            ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + 5);
            ggml_cuda_op_rms_norm_add_rms_norm(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr,
                cgraph->nodes[i + 3], cgraph->nodes[i + 5], nullptr, nullptr, nullptr, nullptr, cgraph->nodes[i + 4]);
            return 5;
        }
    }

    // the sandwich norms of gemma: a block's post-norm, the residual add (and the layer output scale) and the next block's
    //   pre-norm of the same row, two or three launches per block
    if (node->op == GGML_OP_RMS_NORM && i + 4 < cgraph->n_nodes) {
        const bool has_scale = cgraph->nodes[i + 3]->op == GGML_OP_MUL;
        const int  n_ops     = has_scale ? 6 : 5;
        if (i + n_ops <= cgraph->n_nodes &&
                ggml_cuda_should_fuse_rms_norm_add_rms_norm(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2],
                    has_scale ? cgraph->nodes[i + 3] : nullptr, cgraph->nodes[i + n_ops - 2], cgraph->nodes[i + n_ops - 1])) {
            const int outputs[2] = { i + n_ops - 3, i + n_ops - 1 };
            const enum ggml_op ops_scale[6] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_MUL, GGML_OP_RMS_NORM, GGML_OP_MUL };
            const enum ggml_op ops[5]       = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL };
            if (ggml_can_fuse_subgraph(cgraph, i, n_ops, has_scale ? ops_scale : ops, outputs, 2)) {
                ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + n_ops - 1);
                ggml_cuda_op_rms_norm_add_rms_norm(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2],
                    has_scale ? cgraph->nodes[i + 3] : nullptr, cgraph->nodes[i + n_ops - 2], cgraph->nodes[i + n_ops - 1]);
                return n_ops - 1;
            }
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    // The alpha and beta projections of a gated delta net layer read the same input, have a few dozen rows and are
    //   followed by elementwise ops of their own: two GEMV and two elementwise launches per layer, in one.
    //   The GEMV sums x and gate exactly as it sums x alone.
    ggml_cuda_mul_mat_vec_t gdn_gate_mul_mat_vec = node->op == GGML_OP_MUL_MAT ? ggml_cuda_should_fuse_gdn_gate(cgraph, i) : nullptr;

    // ... and the qkv and z projections right after them, of the same input (qwen35 builds them in this order): one mmvq
    //   launch, the qkv rows, z's after them, then alpha's with beta and the gate epilogue. Each row sums as in its own
    //   launch. GGML_CUDA_NO_SIBLING_ROWS=1 runs them apart. The same on mmvf for 2-byte float rows
    //   (ggml_cuda_mul_mat_vec_f_gdn_qkv, GGML_CUDA_NO_GDN_QKV_MMVF=1 runs them apart).
    static const bool no_sibling_rows = getenv("GGML_CUDA_NO_SIBLING_ROWS") != nullptr;
    static const bool no_gdn_qkv_mmvf = getenv("GGML_CUDA_NO_GDN_QKV_MMVF") != nullptr;
    const bool gdn_qkv_mmvf = gdn_gate_mul_mat_vec == ggml_cuda_mul_mat_vec_f && !no_gdn_qkv_mmvf;
    if ((gdn_gate_mul_mat_vec == ggml_cuda_mul_mat_vec_q || gdn_qkv_mmvf) && !no_sibling_rows) {
        int j = i + 9;
        while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
            j++;
        }
        int k = j + 1;
        while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
            k++;
        }
        int l = k + 1;
        while (l < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[l])) {
            l++;
        }
        ggml_tensor * qkv = j < cgraph->n_nodes ? cgraph->nodes[j] : nullptr;
        ggml_tensor * z   = k < cgraph->n_nodes ? cgraph->nodes[k] : nullptr;
        ggml_tensor * mul_a = cgraph->nodes[i + 4];
        ggml_tensor * beta  = cgraph->nodes[i + 8];
        const ggml_tensor * x = node->src[1];
        const auto reads = [](const ggml_tensor * n, const ggml_tensor * t) {
            for (int q = 0; q < GGML_MAX_SRC; q++) {
                if (n->src[q] != nullptr && ggml_cuda_tensors_overlap(n->src[q], t)) {
                    return true;
                }
            }
            return false;
        };
        const auto same_rows = [&](const ggml_tensor * w) {
            return w->type == node->src[0]->type && w->ne[0] == node->src[0]->ne[0] && w->nb[1] == node->src[0]->nb[1] &&
                ggml_nrows(w) == w->ne[1];
        };
        // mmvf: where each would run on its own, and summed in the precision of alpha and beta
        const auto mmvf_alone = [&](const ggml_tensor * t) {
            return ggml_cuda_should_fuse_mul_mat_vec_f(t) && cuda_ctx->stream_context().concurrent_events.empty() &&
                ggml_cuda_mul_mat_choose(*cuda_ctx, t->src[0], t->src[1], t) == GGML_CUDA_MUL_MAT_PATH_MMVF &&
                ggml_prec(t->op_params[0]) == GGML_PREC_DEFAULT;
        };
        bool ok = qkv != nullptr && z != nullptr && qkv->op == GGML_OP_MUL_MAT && z->op == GGML_OP_MUL_MAT &&
            (qkv->flags & GGML_TENSOR_FLAG_COMPUTE) && (z->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            qkv->src[1]->data == x->data && z->src[1]->data == x->data &&
            ggml_are_same_shape(qkv->src[1], x) && ggml_are_same_stride(qkv->src[1], x) &&
            ggml_are_same_shape(z->src[1], x) && ggml_are_same_stride(z->src[1], x) && ggml_nrows(x) == 1 &&
            same_rows(qkv->src[0]) && same_rows(z->src[0]) && same_rows(cgraph->nodes[i + 6]->src[0]) &&
            ggml_is_contiguous(qkv) && ggml_is_contiguous(z) &&
            (gdn_qkv_mmvf ? mmvf_alone(qkv) && mmvf_alone(z) &&
                ggml_cuda_mul_mat_vec_f_gdn_qkv(qkv->src[0], x, z->src[0], node->src[0], cgraph->nodes[i + 6]->src[0]) :
                ggml_cuda_should_fuse_mul_mat_vec_q(qkv) && ggml_cuda_should_fuse_mul_mat_vec_q(z)) &&
            !ggml_cuda_op_mul_mat_use_fwht(qkv) && !ggml_cuda_op_mul_mat_use_fwht(z) &&
            (l >= cgraph->n_nodes || !reads(cgraph->nodes[l], z));
        const ggml_tensor * outs[4] = { qkv, z, mul_a, beta };
        for (int a = 0; ok && a < 4; a++) {
            ok = !ggml_cuda_tensors_overlap(outs[a], x);
            for (int b = a + 1; ok && b < 4; b++) {
                ok = !ggml_cuda_tensors_overlap(outs[a], outs[b]);
            }
        }
        if (ok) {
            ggml_tensor * add = cgraph->nodes[i + 2];
            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.sib      = z->src[0];
            fusion_data.sib_dst  = z;
            fusion_data.gdn_x    = node->src[0];
            fusion_data.gdn_dst  = mul_a;
            fusion_data.gate     = cgraph->nodes[i + 6]->src[0];
            fusion_data.gdn_dt   = add->src[0] == cgraph->nodes[i + 1] ? add->src[1] : add->src[0];
            fusion_data.gdn_a    = mul_a->src[0] == cgraph->nodes[i + 3] ? mul_a->src[1] : mul_a->src[0];
            fusion_data.gdn_beta = beta;
            // and the convolution of qkv in the epilogue of its rows
            ggml_cuda_ssm_conv_state_fusion cf;
            const ggml_tensor * launch_tensors[5] = { x, qkv, z, mul_a, beta };
            const int concat_idx = ggml_cuda_gdn_conv_fold(cuda_ctx, cgraph, k, qkv, launch_tensors, 5, cf);
            if (concat_idx >= 0) {
                fusion_data.conv_states     = (const float *) ((const char *) cf.states->data + cf.state_offs);
                fusion_data.conv_ids        = (const int32_t *) cf.ids->data;
                fusion_data.conv_row_stride = (int64_t) (cf.states->nb[1] / sizeof(float));
                fusion_data.conv_w          = (const float *) cf.ssm_conv->src[1]->data;
                fusion_data.conv_w_stride   = (int64_t) (cf.ssm_conv->src[1]->nb[1] / sizeof(float));
                fusion_data.conv_dst        = (float *) cf.silu->data;
                fusion_data.conv_slot[0]    = cf.slots.dst[0];
                fusion_data.conv_slot[1]    = cf.slots.dst[1];
                cuda_ctx->q8_1_input_invalidate(cf.silu);
                cuda_ctx->cublas_input_invalidate(cf.silu);
                cuda_ctx->conv_folded_concat = cgraph->nodes[concat_idx];
                cuda_ctx->conv_folded_skip   = cf.last_cpy - concat_idx;
            }
            if (gdn_qkv_mmvf) {
                ggml_cuda_mul_mat_vec_f(*cuda_ctx, qkv->src[0], qkv->src[1], nullptr, qkv, &fusion_data);
            } else {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, qkv->src[0], qkv->src[1], nullptr, qkv, &fusion_data);
            }
            return k - i;
        }
    }

    if (gdn_gate_mul_mat_vec) {
        ggml_tensor * mul_a = cgraph->nodes[i + 4];
        ggml_tensor * add   = cgraph->nodes[i + 2];
        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.gate     = cgraph->nodes[i + 6]->src[0];
        fusion_data.gdn_dt   = add->src[0] == cgraph->nodes[i + 1] ? add->src[1] : add->src[0];
        fusion_data.gdn_a    = mul_a->src[0] == cgraph->nodes[i + 3] ? mul_a->src[1] : mul_a->src[0];
        fusion_data.gdn_beta = cgraph->nodes[i + 8];
        // the shape and precision of alpha, the memory of the gate
        ggml_tensor gate = *node;
        gate.data = mul_a->data;
        gdn_gate_mul_mat_vec(*cuda_ctx, node->src[0], node->src[1], nullptr, &gate, &fusion_data);
        return 8;
    }

    // the gate of a gated delta net layer, softplus(alpha + dt) * a
    if (node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
            ggml_cuda_should_fuse_add_softplus_mul(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
        const int outputs[1] = { i + 2 };
        const enum ggml_op ops[3] = { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL };
        if (ggml_can_fuse_subgraph(cgraph, i, 3, ops, outputs, 1)) {
            ggml_cuda_op_add_softplus_mul(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            return 2;
        }
    }

    // the residual add ahead of a norm: two launches per layer and token, and the sum stays an output
    if (node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
            ggml_cuda_should_fuse_rms_norm_pre_add(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
        const int outputs[2] = { i, i + 2 };
        const enum ggml_op ops[3] = { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse_subgraph(cgraph, i, 3, ops, outputs, 2)) {
            ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + 2);
            ggml_cuda_op_rms_norm_pre_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            return 2;
        }
    }

    // the same after the sigmoid-gated shared expert of a MoE layer (qwen35moe, qwen3next): its SIGMOID, MUL and the ADD
    //   onto the routed experts would be three more launches per layer, the two ADDs ahead of the norm fit neither fusion
    if (node->op == GGML_OP_UNARY && i + 5 < cgraph->n_nodes &&
            ggml_cuda_should_fuse_rms_norm_pre_add_gated(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3],
                                                         cgraph->nodes[i + 4], cgraph->nodes[i + 5])) {
        const int outputs[2] = { i + 3, i + 5 };
        const enum ggml_op ops[6] = { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse_subgraph(cgraph, i, 6, ops, outputs, 2)) {
            ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + 5);
            ggml_cuda_op_rms_norm_pre_add_gated(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3],
                                                cgraph->nodes[i + 4], cgraph->nodes[i + 5]);
            return 5;
        }
    }

    // the gated output norm of a gated delta net layer, rms_norm(x) * w * op(z): the unary reads z, not the norm
    if (node->op == GGML_OP_RMS_NORM && i + 3 < cgraph->n_nodes &&
            ggml_cuda_should_fuse_rms_norm_gated(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3])) {
        const int outputs[1] = { i + 3 };
        const enum ggml_op ops[4] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_UNARY, GGML_OP_MUL };
        if (ggml_can_fuse_subgraph(cgraph, i, 4, ops, outputs, 1)) {
            ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + 3, /*flat =*/ true);
            ggml_cuda_op_rms_norm_gated(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3]);
            return 3;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + 1);
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    // the router input of gemma4, rms_norm(x) * s * w: the values of the three ops apart
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 2], cgraph->nodes[i + 1]);
        return 2;
    }

    // l2 norm built as rms_norm + a scalar scale (build_gdn_l2_norm)
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        // not run when the gated delta net that reads it applies the norm itself
        for (int j = i + 2; j < cgraph->n_nodes && j < i + 256; j++) {
            const ggml_tensor * gdn = cgraph->nodes[j];
            if (gdn->op == GGML_OP_GATED_DELTA_NET && (gdn->src[0] == cgraph->nodes[i + 1] || gdn->src[1] == cgraph->nodes[i + 1])) {
                ggml_cuda_gated_delta_net_l2 l2;
                if (ggml_cuda_gdn_l2_fusion(cgraph, j, l2)) {
                    return 1;
                }
                break;
            }
        }
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    // build_conv_state: the CONCAT runs the whole chain, the SSM_CONV and its SILU were written there
    if (node->op == GGML_OP_CONCAT) {
        ggml_cuda_ssm_conv_state_fusion f;
        if (ggml_cuda_ssm_conv_state_fusion_match(cuda_ctx, cgraph, i, f)) {
            cuda_ctx->q8_1_input_invalidate(f.silu);
            cuda_ctx->cublas_input_invalidate(f.silu);
            ggml_tensor states = *f.states; // the row from the state of this convolution
            states.data = (char *) f.states->data + f.state_offs;
            if (f.long_path) {
                ggml_cuda_op_ssm_conv_state_long(*cuda_ctx, &states, f.ids, f.x, f.ssm_conv, f.silu, f.long_slots);
            } else {
                ggml_cuda_op_ssm_conv_state(*cuda_ctx, &states, f.ids, f.x, f.ssm_conv, f.silu, f.slots, f.gate, f.x_a, f.x_b);
            }
            return f.last_cpy - i;
        }
    }
    if (node->op == GGML_OP_SSM_CONV) {
        int n_skip = 0;
        if (ggml_cuda_ssm_conv_state_is_folded(cuda_ctx, cgraph, i, &n_skip)) {
            return n_skip;
        }
    }

    // the fused kernel writes the SILU output while it reads the convolution input, so they may not share memory: with a
    //   node between the convolution and the SILU, ggml-alloc can give the SILU the memory of the freed concat
    const auto ssm_conv_silu_disjoint = [&](const ggml_tensor * silu) {
        return !ggml_cuda_tensors_overlap(silu, node->src[0]) && !ggml_cuda_tensors_overlap(silu, node->src[1]);
    };

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU }) &&
            ssm_conv_silu_disjoint(cgraph->nodes[i + 2])) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU }) &&
            ssm_conv_silu_disjoint(cgraph->nodes[i + 1])) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    // the same with a RESHAPE between (bailingmoe3): the SILU output is written in the shape of the convolution
    if (node->op == GGML_OP_SSM_CONV && i + 2 < cgraph->n_nodes) {
        ggml_tensor *      reshape    = cgraph->nodes[i + 1];
        ggml_tensor *      silu       = cgraph->nodes[i + 2];
        const ggml_op      ops[3]     = { GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_UNARY };
        const int          outputs[1] = { i + 2 };
        if (reshape->op == GGML_OP_RESHAPE && reshape->src[0] == node && silu->op == GGML_OP_UNARY &&
                ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU && silu->src[0] == reshape && node->type == GGML_TYPE_F32 &&
                silu->type == GGML_TYPE_F32 && ggml_is_contiguous(node) && ggml_is_contiguous(silu) &&
                ssm_conv_silu_disjoint(silu) && ggml_can_fuse_subgraph(cgraph, i, 3, ops, outputs, 1)) {
            ggml_tensor silu_conv = *silu;
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                silu_conv.ne[d] = node->ne[d];
                silu_conv.nb[d] = node->nb[d];
            }
            ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, &silu_conv);
            return 2;
        }
    }

    // the attention output gate of qwen35: the gate is the second half of each head of the q projection, made contiguous
    //   only for its sigmoid and the mul onto the attention output
    if (node->op == GGML_OP_CONT && i + 2 < cgraph->n_nodes &&
            ggml_cuda_should_fuse_cont_unary_mul(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
        const int outputs[1] = { i + 2 };
        const enum ggml_op ops[3] = { GGML_OP_CONT, GGML_OP_UNARY, GGML_OP_MUL };
        if (ggml_can_fuse_subgraph(cgraph, i, 3, ops, outputs, 1)) {
            // one token: the q8_1 copy of the output for the output projection
            static const bool no_gate_q8_1 = getenv("GGML_CUDA_NO_GATE_Q8_1") != nullptr;
            if (!no_gate_q8_1) {
                ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + 2);
            }
            ggml_cuda_op_cont_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            return 2;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    // the scatter weights of a hyper-connection combine (qwen4exp): 2*sigmoid(inject/hc) computed in the HC_POST that
    //   reads them, three launches less per block
    if (node->op == GGML_OP_SCALE && i + 3 < cgraph->n_nodes) {
        ggml_tensor * sig  = cgraph->nodes[i + 1];
        ggml_tensor * sc2  = cgraph->nodes[i + 2];
        ggml_tensor * post = cgraph->nodes[i + 3];
        const ggml_tensor * raw = node->src[0];
        if (sig->op == GGML_OP_UNARY && ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID && sig->src[0] == node &&
                sc2->op == GGML_OP_SCALE && sc2->src[0] == sig && post->op == GGML_OP_DSV4_HC_POST && post->src[2] == sc2 &&
                ggml_node_has_n_uses(cgraph, i, 1) && ggml_node_has_n_uses(cgraph, i + 1, 1) && ggml_node_has_n_uses(cgraph, i + 2, 1) &&
                (sig->flags & GGML_TENSOR_FLAG_COMPUTE) && (sc2->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                (post->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                raw->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 && sig->type == GGML_TYPE_F32 && sc2->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(raw) && ggml_is_contiguous(sc2) && ggml_are_same_shape(raw, sc2) &&
                !ggml_cuda_tensors_overlap(post, raw)) {
            // with the grouped norm of the next block that reads it
            static const bool no_sync = getenv("GGML_CUDA_NO_HC_POST_SYNC") != nullptr;
            const int fuse = i + 5 < cgraph->n_nodes && ggml_node_has_n_uses(cgraph, i + 4, 1) &&
                (cgraph->nodes[i + 4]->flags & GGML_TENSOR_FLAG_COMPUTE) && (cgraph->nodes[i + 5]->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                !(cgraph->nodes[i + 4]->flags & GGML_TENSOR_FLAG_OUTPUT) ?
                ggml_cuda_should_fuse_dsv4_hc_post_rms_norm(post, cgraph->nodes[i + 4], cgraph->nodes[i + 5], raw) : 0;
            if (fuse == 1 || (fuse == 2 && !no_sync)) {
                ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + 5, /*flat =*/ true);
                ggml_cuda_op_dsv4_hc_post_rms_norm(*cuda_ctx, post, node, sc2, cgraph->nodes[i + 4], cgraph->nodes[i + 5], fuse == 2);
                return 5;
            }
            ggml_cuda_op_dsv4_hc_post_gated(*cuda_ctx, post, node, sc2);
            return 3;
        }
    }

    // the gated shared expert of qwen35moe / qwen4exp added to the routed experts, for more than one token (one token
    //   folds it into the weighted reduction, see ggml_cuda_try_fuse_moe_down_shexp)
    if (node->op == GGML_OP_UNARY && i + 2 < cgraph->n_nodes &&
            ggml_cuda_should_fuse_sigmoid_mul_add(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
        const ggml_op ops[3]     = { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD };
        const int     outputs[1] = { i + 2 };
        if (ggml_can_fuse_subgraph(cgraph, i, 3, ops, outputs, 1)) {
            ggml_cuda_op_sigmoid_mul_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            return 2;
        }
    }

    // the forget gate of bailingmoe3: scale * sigmoid(gate * a), a per head; the decay gate of a KDA layer (GLM-5.3):
    //   lb * sigmoid(-(g * A)), a SCALE before the sigmoid
    if (ggml_can_fuse(cgraph, i, { GGML_OP_MUL, GGML_OP_UNARY, GGML_OP_SCALE }) &&
            ggml_cuda_should_fuse_mul_sigmoid_scale(node, nullptr, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
        ggml_cuda_op_mul_sigmoid_scale(*cuda_ctx, node, nullptr, cgraph->nodes[i + 2]);
        return 2;
    }
    if (ggml_can_fuse(cgraph, i, { GGML_OP_MUL, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }) &&
            ggml_cuda_should_fuse_mul_sigmoid_scale(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3])) {
        ggml_cuda_op_mul_sigmoid_scale(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 3]);
        return 3;
    }

    // qwen4exp, 1 to 4 tokens: SCALE -> SILU of a hyper-connection mix, its up projection and the gated DSV4_HC_PRE that
    //   reads it (past the reshape views) in one launch, two launches less per mix
    if (node->op == GGML_OP_SCALE && i + 4 < cgraph->n_nodes) {
        static const bool disabled = getenv("GGML_CUDA_NO_HC_UP_PRE") != nullptr;
        int j = i + 3;
        while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
            j++;
        }
        bool ok = !disabled && j < cgraph->n_nodes && (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_COMPUTE);
        for (int k = i; ok && k < i + 3; k++) {
            ok = ggml_node_has_n_uses(cgraph, k, 1) && (cgraph->nodes[k]->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                !(cgraph->nodes[k]->flags & GGML_TENSOR_FLAG_OUTPUT);
        }
        if (ok && ggml_cuda_should_fuse_dsv4_hc_up_pre(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[j])) {
            ggml_cuda_q8_1_request(cuda_ctx, cgraph, j);
            ggml_cuda_op_dsv4_hc_up_pre(*cuda_ctx, node, cgraph->nodes[i + 2], cgraph->nodes[j]);
            return j - i;
        }
    }

    if (ggml_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY }) &&
            ggml_cuda_should_fuse_scale_unary(node, cgraph->nodes[i + 1])) {
        ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + 1);
        ggml_cuda_op_scale_unary(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    return 0;
}

// a GET_ROWS of the only row of src with one id: the id is 0, so the output is src
static bool ggml_cuda_get_rows_is_identity(const ggml_cgraph * cgraph, const ggml_tensor * g) {
    return g->op == GGML_OP_GET_ROWS && g->type == GGML_TYPE_F32 && g->src[0]->type == GGML_TYPE_F32 &&
        ggml_nrows(g->src[0]) == 1 && ggml_nelements(g->src[1]) == 1 && ggml_are_same_shape(g, g->src[0]) &&
        !(g->flags & GGML_TENSOR_FLAG_OUTPUT) && ggml_cuda_graph_use_count(cgraph, g) == 1;
}

// The all-reduce whose lines the matmul before it pushed (ggml_backend_cuda_context::ar_fold), at node i, the first
//   compute node of the next subgraph: if that is the residual add of the reduced tensor with the norm after it, they
//   run as one launch that polls the lines (3 nodes done); otherwise the receive half of the LL kernel runs first (0).
//   The last layer of a model that keeps only the output rows (inp_out_ids) gathers both inputs of the add first: at one
//   row with one output that is the identity, so the gathers do not run either (5 nodes done).
//   cgraph nullptr: the receive half.
static int ggml_cuda_ar_fold_consume(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_backend_cuda_context::ar_fold_t & f = cuda_ctx->ar_fold;
    if (!f.recv) {
        return 0;
    }
    f.recv = false;
    ggml_backend_cuda_context::ar_fold_call_t & c = f.recv_call;
    ggml_tensor * t = c.tensors[c.rank];
    if (cgraph != nullptr && i + 2 < cgraph->n_nodes) {
        ggml_tensor * add  = cgraph->nodes[i];
        ggml_tensor * norm = cgraph->nodes[i + 1];
        ggml_tensor * mul  = cgraph->nodes[i + 2];
        const int outputs[2] = { i, i + 2 };
        const enum ggml_op ops[3] = { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (add->op == GGML_OP_ADD && norm->op == GGML_OP_RMS_NORM && mul->op == GGML_OP_MUL &&
                ggml_cuda_rms_norm_pre_add_ar_supported(add, norm, mul, t) && ggml_can_fuse_subgraph(cgraph, i, 3, ops, outputs, 2)) {
            ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + 2);
            ggml_cuda_op_rms_norm_pre_add_ar(*cuda_ctx, add, norm, mul, c.args, t);
            return 3;
        }
    }
    static const bool no_gathered = getenv("GGML_CUDA_NO_AR_FOLD_GATHER") != nullptr;
    if (cgraph != nullptr && !no_gathered && i + 4 < cgraph->n_nodes) {
        ggml_tensor * add  = cgraph->nodes[i + 2];
        ggml_tensor * norm = cgraph->nodes[i + 3];
        ggml_tensor * mul  = cgraph->nodes[i + 4];
        const int outputs[2] = { i + 2, i + 4 };
        const enum ggml_op ops[5] = { GGML_OP_GET_ROWS, GGML_OP_GET_ROWS, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL };
        const bool gathered = ggml_cuda_get_rows_is_identity(cgraph, cgraph->nodes[i]) &&
            ggml_cuda_get_rows_is_identity(cgraph, cgraph->nodes[i + 1]) && add->op == GGML_OP_ADD &&
            (add->src[0] == cgraph->nodes[i] || add->src[0] == cgraph->nodes[i + 1]) &&
            (add->src[1] == cgraph->nodes[i] || add->src[1] == cgraph->nodes[i + 1]) && add->src[0] != add->src[1] &&
            norm->op == GGML_OP_RMS_NORM && mul->op == GGML_OP_MUL && ggml_can_fuse_subgraph(cgraph, i, 5, ops, outputs, 2);
        if (gathered) {
            // the same nodes reading the sources of the gathers, so that the checks see the tensors the kernel reads
            ggml_tensor add_s  = *add;
            ggml_tensor norm_s = *norm;
            ggml_tensor mul_s  = *mul;
            add_s.src[0]  = add->src[0]->src[0];
            add_s.src[1]  = add->src[1]->src[0];
            norm_s.src[0] = &add_s;
            mul_s.src[mul->src[0] == norm ? 0 : 1] = &norm_s;
            if (ggml_cuda_rms_norm_pre_add_ar_supported(&add_s, &norm_s, &mul_s, t)) {
                ggml_cuda_q8_1_request(cuda_ctx, cgraph, i + 4);
                if (cuda_ctx->q8_1_emit == mul) {
                    cuda_ctx->q8_1_emit = &mul_s;
                }
                ggml_cuda_op_rms_norm_pre_add_ar(*cuda_ctx, &add_s, &norm_s, &mul_s, c.args, t);
                return 5;
            }
        }
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(c.comm_ctx);
    ggml_cuda_p2p_ar_launch_ll_recv(comm_ctx->p2p_ar, comm_ctx->backends.data(), c.tensors, c.rank);
    return 0;
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (cuda_ctx->glu_deferred.glu != nullptr && i > cuda_ctx->glu_deferred.target) {
                    ggml_cuda_flush_deferred_glu(cuda_ctx);
                }

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                cuda_ctx->q8_1_input_invalidate(node);
                cuda_ctx->cublas_input_invalidate(node);

                // an all-reduce folded into the matmul before it, see ggml_cuda_ar_fold_consume
                if (cuda_ctx->ar_fold.recv) {
                    const int n_done = ggml_cuda_ar_fold_consume(cuda_ctx, cgraph, i);
                    if (n_done > 0) {
                        for (int j = i; j < i + n_done; j++) {
                            cuda_ctx->q8_1_input_invalidate(cgraph->nodes[j]);
                            cuda_ctx->cublas_input_invalidate(cgraph->nodes[j]);
                        }
                        cuda_ctx->q8_1_commit_pending();
                        i += n_done - 1;
                        continue;
                    }
                }

                // a DeepSeek-V4 compressor launched at its row sum
                if (ggml_cuda_dsv4_compress_fold_step(cuda_ctx, cgraph, i)) {
                    continue;
                }

                // the state gather of a gated delta net or a convolution that reads the cache itself
                if (node->op == GGML_OP_GET_ROWS && (ggml_cuda_gdn_gather_is_folded(cuda_ctx, cgraph, i) ||
                                                     ggml_cuda_ssm_conv_state_is_folded(cuda_ctx, cgraph, i))) {
                    continue;
                }
                if (node->op == GGML_OP_MUL && ggml_cuda_ssm_conv_state_is_folded(cuda_ctx, cgraph, i)) {
                    continue;
                }

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {
                    for (int j = i; j <= i + nodes_to_skip; j++) {
                        cuda_ctx->q8_1_input_invalidate(cgraph->nodes[j]);
                        cuda_ctx->cublas_input_invalidate(cgraph->nodes[j]);
                    }
                    // a q8_1 copy the fused kernel wrote of its output, see ggml_cuda_q8_1_request
                    cuda_ctx->q8_1_commit_pending();
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                // one token: the combine of a split KV writes the q8_1 copy of the output, see launch_fattn; so does
                //   the mix of the hyper-connection streams that a block reads without a norm (qwen4exp). Several
                //   tokens: an f32 SWIGLU (the shared expert of a verify step, its gate not fused there), rows padded
                static const bool fattn_no_q8_1 = getenv("GGML_CUDA_NO_FATTN_Q8_1") != nullptr;
                const bool glu_q8_1  = node->op == GGML_OP_GLU && ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU &&
                    node->type == GGML_TYPE_F32 && ggml_nrows(node) > 1 && ggml_nrows(node) <= MMVQ_MAX_BATCH_SIZE;
                const bool emit_q8_1 = (node->op == GGML_OP_FLASH_ATTN_EXT && node->src[0]->ne[1] == 1 && !fattn_no_q8_1) ||
                    (node->op == GGML_OP_DSV4_HC_PRE && ggml_nrows(node) == 1) || glu_q8_1;
                if (emit_q8_1) {
                    ggml_cuda_q8_1_request(cuda_ctx, cgraph, i, /*flat =*/ false, /*padded =*/ glu_q8_1);
                }

                bool ok = ggml_cuda_glu_with_f16_copy(*cuda_ctx, cgraph, i) || ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (emit_q8_1) {
                    cuda_ctx->q8_1_commit_pending();
                }

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
            ggml_cuda_flush_deferred_glu(cuda_ctx);
            ggml_cuda_expert_cpu_merge_pending(cuda_ctx, cgraph);
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph->captured_gen = graph->props_gen;
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch && ggml_cuda_graph_debug_enabled()) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}

// Capture a graph after two calls in a row that found nothing changed, not one: a capture costs ~2x an eager run,
//   and a short prompt graph that comes back once (the same prompt length on the next chat turn) is never replayed.
//   A one-token graph runs eagerly once more per rebuild. GGML_CUDA_GRAPH_WARMUP sets it.
static int ggml_cuda_graph_n_warmup() {
    static const int n = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_WARMUP");
        return env != nullptr ? std::max(1, atoi(env)) : 2;
    }();
    return n;
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_preload_modules_close();

    ggml_cuda_set_device(cuda_ctx->device);
    cuda_ctx->flags_alloc();

    // every tensor is written again this pass, so no quantized or converted copy from the last one survives
    cuda_ctx->q8_1_input_reset();
    cuda_ctx->cublas_input_reset();
    cuda_ctx->expert_cache_ahead.clear();
    cuda_ctx->expert_cpu_merge = false;
    cuda_ctx->dsv4_fold = {};
    if (cuda_ctx->device == ggml_cuda_expert_cpu_device()) {
        ggml_cuda_expert_cpu_kick();
    }

    // in a session the subgraph is one part of the session's graph, see ggml_backend_cuda_session_plan
    if (cuda_ctx->session_mode != ggml_backend_cuda_context::SESSION_OFF) {
        ggml_cuda_session_reduction_flush(cuda_ctx);
        ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, false, false, nullptr);
        // a subgraph without compute nodes did not take the all-reduce before it
        ggml_cuda_ar_fold_consume(cuda_ctx, nullptr, 0);
        return GGML_STATUS_SUCCESS;
    }
    GGML_ASSERT(!cuda_ctx->ar_fold.recv && cuda_ctx->ar_fold.next == nullptr);

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
                // Warmup: ggml_cuda_graph_n_warmup calls in a row with no property change
                if (properties_changed) {
                    graph->warmup_stable = 0;
                } else if (++graph->warmup_stable >= ggml_cuda_graph_n_warmup()) {
                    graph->warmup_complete = true;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    }
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed || graph->captured_gen != graph->props_gen) {
                    // Properties changed, here or in a session since the capture - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    graph->warmup_stable   = 0;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                    }
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}

// A session: subgraphs a device runs in a row for a ggml_backend_meta graph and the all-reduces between them as one CUDA
//   graph. Apart, each subgraph is a graph launch and each all-reduce an eager kernel, and on Volta every such boundary
//   leaves the GPU idle for ~2-2.5 us, twice per all-reduce. plan says from all subgraphs at once whether the session's
//   graph can be replayed, must be captured again or the subgraphs run eagerly (warmup, a changed subgraph); the caller
//   takes the slowest mode of all devices. To capture or run eagerly, begin, graph_compute of each subgraph,
//   session_allreduce after it and end; launch then starts the captured graph.
static const void * ggml_cuda_session_key(ggml_cgraph * cgraph) {
    // next to the key of the first subgraph, which keeps its own entry for its properties
    return (const char *) ggml_cuda_graph_get_key(cgraph) + 1;
}

static int ggml_backend_cuda_session_plan(ggml_backend_t backend, ggml_cgraph ** cgraphs, int n_cgraphs) {
    ggml_cuda_preload_modules_close();
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    GGML_ASSERT(cuda_ctx->session_mode == ggml_backend_cuda_context::SESSION_OFF && n_cgraphs > 0);
    ggml_cuda_set_device(cuda_ctx->device);

    ggml_backend_cuda_context::session_mode_t mode = ggml_backend_cuda_context::SESSION_EAGER;
#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_session_key(cgraphs[0]);
    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        bool graph_compatible = true;
        for (int i = 0; i < n_cgraphs; i++) {
            graph_compatible = graph_compatible && ggml_cuda_graph_check_compability(cgraphs[i]);
        }
        if (graph_compatible) {
            bool properties_changed = (int) graph->session_keys.size() != n_cgraphs;
            // a subgraph whose properties changed since the capture, here or computed alone, even if they match again
            bool stale = (int) graph->session_gens.size() != n_cgraphs;
            graph->session_keys.resize(n_cgraphs);
            for (int i = 0; i < n_cgraphs; i++) {
                const void * key_i = ggml_cuda_graph_get_key(cgraphs[i]);
                properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraphs[i]) || graph->session_keys[i] != key_i ||
                    properties_changed;
                graph->session_keys[i] = key_i;
                stale = stale || graph->session_gens[i] != cuda_ctx->cuda_graph(key_i)->props_gen;
            }

            if (!graph->warmup_complete) {
                if (properties_changed) {
                    graph->warmup_stable = 0;
                } else if (++graph->warmup_stable >= ggml_cuda_graph_n_warmup()) {
                    graph->warmup_complete = true;
                    mode = ggml_backend_cuda_context::SESSION_CAPTURE;
                }
            } else if (properties_changed || stale) {
                graph->warmup_complete = false;
                graph->warmup_stable   = 0;
            } else {
                mode = graph->instance == nullptr ? ggml_backend_cuda_context::SESSION_CAPTURE : ggml_backend_cuda_context::SESSION_REPLAY;
            }
        }
    }
    graph->session_want = mode;
#else
    GGML_UNUSED(cgraphs);
#endif // USE_CUDA_GRAPH
    return mode;
}

static void ggml_backend_cuda_session_begin(ggml_backend_t backend, ggml_cgraph ** cgraphs, int n_cgraphs, int mode) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    GGML_ASSERT(cuda_ctx->session_mode == ggml_backend_cuda_context::SESSION_OFF && n_cgraphs > 0);
    GGML_ASSERT(mode == ggml_backend_cuda_context::SESSION_EAGER || mode == ggml_backend_cuda_context::SESSION_CAPTURE);
    ggml_cuda_set_device(cuda_ctx->device);
    cuda_ctx->flags_alloc();
    if (cuda_ctx->device == ggml_cuda_expert_cpu_device()) {
        ggml_cuda_expert_cpu_kick();
    }

    const void * graph_key = ggml_cuda_session_key(cgraphs[0]);
#ifdef USE_CUDA_GRAPH
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (mode == ggml_backend_cuda_context::SESSION_EAGER && graph->session_want == ggml_backend_cuda_context::SESSION_CAPTURE) {
        graph->warmup_complete = false; // another device runs eagerly this time; capture on the next stable plan
    }
    if (mode == ggml_backend_cuda_context::SESSION_CAPTURE) {
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }
        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }
#else
    GGML_ASSERT(mode == ggml_backend_cuda_context::SESSION_EAGER);
#endif // USE_CUDA_GRAPH

    cuda_ctx->session_key  = graph_key;
    cuda_ctx->session_mode = (ggml_backend_cuda_context::session_mode_t) mode;

    // the sums of the all-reduces of a model with hyper-connections go to DSV4_HC_POST, see
    //   ggml_backend_cuda_session_allreduce_next
    bool hc = false;
    for (int k = 0; k < n_cgraphs && !hc; ++k) {
        for (int n = 0; n < cgraphs[k]->n_nodes && !hc; ++n) {
            hc = cgraphs[k]->nodes[n]->op == GGML_OP_DSV4_HC_POST;
        }
    }
    cuda_ctx->ar_fold.hc = hc;
}

// whether the all-reduce of these tensors (one per device) can be part of a session graph
static bool ggml_backend_cuda_session_allreduce_supported(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx != nullptr && ggml_cuda_p2p_ar_ll_capturable(comm_ctx->p2p_ar, comm_ctx->backends.data(), tensors);
}

// The all-reduce of tensors (one per device) after the subgraph about to run: the matmul that writes tensors[rank] may
//   push its rows as the LL lines of every rank, see ggml_backend_cuda_context::ar_fold. GGML_CUDA_NO_AR_FOLD=1 turns
//   it off. Not in a model with hyper-connections (ar_fold.hc): DSV4_HC_POST cannot poll the lines, so the receive kernel
//   would run, and with the pushing matmul that is two kernels of their own per layer, which costs DeepSeek-V4-Flash
//   more than the LL kernel they replace.
static void ggml_backend_cuda_session_allreduce_next(void * comm_ctx_v, struct ggml_tensor ** tensors, size_t rank) {
    static const bool no_fold = getenv("GGML_CUDA_NO_AR_FOLD") != nullptr;
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[rank]->context;
    GGML_ASSERT(cuda_ctx->session_mode != ggml_backend_cuda_context::SESSION_OFF);
    ggml_backend_cuda_context::ar_fold_t & f = cuda_ctx->ar_fold;
    f.next   = nullptr;
    f.pushed = false;
    // the norm that polls the lines keeps the row in registers, up to 8 columns per thread of its 1024-thread layout (see
    //   rms_norm_pre_add_ar_split_f32)
    const int64_t ncols = ggml_nelements(tensors[rank]);
    if (no_fold || f.hc || ncols > 8*1024 || comm_ctx->p2p_ar == nullptr ||
            !ggml_cuda_p2p_ar_fold_args(comm_ctx->p2p_ar, comm_ctx->backends.data(), tensors, rank, f.next_call.args)) {
        return;
    }
    f.next_call.comm_ctx = comm_ctx_v;
    std::copy_n(tensors, comm_ctx->backends.size(), f.next_call.tensors);
    f.next_call.rank = rank;
    f.next = tensors[rank];
}

static void ggml_backend_cuda_session_allreduce(void * comm_ctx_v, struct ggml_tensor ** tensors, size_t rank) {
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[rank]->context;
    GGML_ASSERT(cuda_ctx->session_mode != ggml_backend_cuda_context::SESSION_OFF);
    // the matmul pushed the lines: the next subgraph receives them
    ggml_backend_cuda_context::ar_fold_t & f = cuda_ctx->ar_fold;
    const bool pushed = f.pushed && f.next == tensors[rank];
    f.next   = nullptr;
    f.pushed = false;
    if (pushed) {
        GGML_ASSERT(!f.recv);
        ggml_cuda_session_reduction_flush(cuda_ctx);
        f.recv_call = f.next_call;
        f.recv      = true;
        return;
    }
    // the MoE sum that ended the subgraph, if it is this all-reduce's input, see ggml_cuda_try_fuse_moe_down_shexp
    ggml_backend_cuda_context::session_reduction_t & red = cuda_ctx->session_red;
    const ggml_tensor * t = tensors[rank];
    if (red.dst != nullptr && t->data == red.dst->data && ggml_nelements(t) == ggml_nelements(red.dst) &&
            ggml_nelements(t) == red.experts->ne[0]) {
        ggml_cuda_p2p_ar_launch_ll_moe_reduce(comm_ctx->p2p_ar, comm_ctx->backends.data(), tensors, rank,
            (const float *) red.experts->data, (const float *) red.weights->data, (int) red.experts->ne[1],
            red.add_x ? red.add_x->get() : nullptr, red.add_g ? (const float *) red.add_g->data : nullptr);
        red = {};
        return;
    }
    ggml_cuda_session_reduction_flush(cuda_ctx);
    ggml_cuda_p2p_ar_launch_ll(comm_ctx->p2p_ar, comm_ctx->backends.data(), tensors, rank);
}

static void ggml_backend_cuda_session_end(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    const ggml_backend_cuda_context::session_mode_t mode = cuda_ctx->session_mode;
    GGML_ASSERT(mode != ggml_backend_cuda_context::SESSION_OFF);
    // a sum that ended the last subgraph has no all-reduce after it
    ggml_cuda_set_device(cuda_ctx->device);
    ggml_cuda_session_reduction_flush(cuda_ctx);
    // the subgraph that reads an all-reduce after the last one is in the next session
    ggml_cuda_ar_fold_consume(cuda_ctx, nullptr, 0);
    cuda_ctx->ar_fold.next   = nullptr;
    cuda_ctx->ar_fold.pushed = false;
    cuda_ctx->session_mode = ggml_backend_cuda_context::SESSION_OFF;

#ifdef USE_CUDA_GRAPH
    if (mode == ggml_backend_cuda_context::SESSION_CAPTURE) {
        ggml_cuda_set_device(cuda_ctx->device);
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(cuda_ctx->session_key);
        if (graph->graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph->graph));
            graph->graph = nullptr;
        }
        CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        }
        graph->session_gens.resize(graph->session_keys.size());
        for (size_t i = 0; i < graph->session_keys.size(); i++) {
            graph->session_gens[i] = cuda_ctx->cuda_graph(graph->session_keys[i])->props_gen;
        }
        if (graph->instance == nullptr) {
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        } else {
            ggml_cuda_graph_update_executable(cuda_ctx, cuda_ctx->session_key);
        }
    }
#else
    GGML_UNUSED(mode);
#endif // USE_CUDA_GRAPH
}

// starts the graph of a session that was captured, now or before
static void ggml_backend_cuda_session_launch(ggml_backend_t backend, ggml_cgraph ** cgraphs, int n_cgraphs) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    GGML_ASSERT(cuda_ctx->session_mode == ggml_backend_cuda_context::SESSION_OFF && n_cgraphs > 0);
#ifdef USE_CUDA_GRAPH
    ggml_cuda_set_device(cuda_ctx->device);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(ggml_cuda_session_key(cgraphs[0]));
    GGML_ASSERT(graph->instance != nullptr);
    if (cuda_ctx->device == ggml_cuda_expert_cpu_device()) {
        ggml_cuda_expert_cpu_kick();
    }
    CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
#else
    GGML_UNUSED(cgraphs);
    GGML_ABORT("fatal error");
#endif // USE_CUDA_GRAPH
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    const bool disable_fusion = ggml_cuda_fusion_disabled();

    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    if (!disable_fusion) {
        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        ggml_cuda_set_device(cuda_ctx->device);
        const bool shared_expert = ggml_cuda_mmvq_shared_expert_supported(ggml_cuda_info().devices[cuda_ctx->device].cc);
        for (int i = 0; shared_expert && i + 5 < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            for (int j = i + 3; j + 2 < cgraph->n_nodes; ++j) {
                if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j + 1]->op == GGML_OP_MUL_MAT_ID) {
                    break;
                }
                if (cgraph->nodes[j]->op != GGML_OP_MUL_MAT || !ggml_cuda_match_shared_expert(cgraph, i, j)) {
                    continue;
                }
                // Group both outputs before allocation so the shared result cannot alias intervening nodes.
                std::rotate(cgraph->nodes + i + 3, cgraph->nodes + j, cgraph->nodes + j + 3);
                ggml_tensor * up = cgraph->nodes[i + 2]->src[1];
                params->add_alloc_dep(params->user_data, up->src[1], cgraph->nodes[i + 5]);
                params->add_alloc_dep(params->user_data, up->src[2], cgraph->nodes[i + 5]);
                i += 5;
                break;
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
                if (match.expert_scale != nullptr) {
                    params->add_alloc_dep(
                        params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
                }
                i += match.node_count - 1;
            }

            if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                    cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
                std::vector<ggml_op>    ops;
                ops.reserve(13 + ggml_cuda_topk_moe_n_group_ops);  // max ops; avoids gcc -Wstringop-overflow false positive

                const ggml_tensor * node = cgraph->nodes[i];

                if (can_fuse) {
                    const ggml_tensor * logits  = node->src[0];
                    ggml_tensor *       weights = nullptr;
                    ggml_tensor *       ids     = nullptr;

                    if (!args.delayed_softmax) {
                        int out_nodes[2];  // nodes which can't be elided

                        if (args.sigmoid) {
                            ops.insert(ops.end(), { GGML_OP_UNARY });
                        } else if (args.sqrt_softplus) {
                            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                        } else {
                            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                        }
                        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                        if (args.prob_bias) {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD });
                            if (args.n_groups > 1) {
                                ops.insert(ops.end(), std::begin(ggml_cuda_topk_moe_group_ops),
                                           std::end(ggml_cuda_topk_moe_group_ops));
                            }
                            ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i + (int) ops.size() - 2;
                        } else {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 3;
                        }
                        ids = cgraph->nodes[out_nodes[0]];

                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }
                        if (args.scale) {
                            ops.insert(ops.end(), { GGML_OP_SCALE });
                        }

                        weights      = cgraph->nodes[i + ops.size() - 1];
                        out_nodes[1] = i + ops.size() - 1;

                        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                            // until the last node of the fused group, not the node after it (which may not exist)
                            add_alloc_deps(i, i + ops.size() - 1);
                            i += ops.size() - 1;
                        }
                    }
                }
            }
        }
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (!ggml_cuda_register_host_enabled()) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (!ggml_cuda_register_host_enabled()) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool is_uma = prop.integrated > 0 || ggml_cuda_unified_memory_enabled();

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    const bool host_buffer = !ggml_cuda_pinned_memory_disabled();
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                        return false;
                    }
                    return op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    if (op->src[0]->type == GGML_TYPE_BF16 &&
                            (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_CLAMP)) {
                        return false;
                    }
                    if (op->src[0]->type == GGML_TYPE_F16 && ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI) {
                        return false;
                    }
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_cuda_op_mul_mat_use_fwht(op)) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                // rows copied unchanged into a dst of the source type
                if (op->type == op->src[0]->type) {
                    return true;
                }
                if (op->type != GGML_TYPE_F32) {
                    return false;
                }
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            }
        case GGML_OP_GET_ROWS_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            }
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            }
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            }
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            return true;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            }
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            }
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            }
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type;
        case GGML_OP_CLAMP:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_BF16) {
                return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_BF16 &&
                    (op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_F32);
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const int64_t d_state = op->src[0]->ne[0];
                return (d_state == 96 || d_state == 128 || d_state == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // the kernel bounds-checks a d_inner that the block size does not divide
            return true;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
            // src0 gives the kernel shape only, its data is never read
            return ggml_is_contiguous(op->src[1]);
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_3D:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguously_allocated(op->src[0]);
        case GGML_OP_TOP_K:
            return op->src[0]->ne[0] <= INT_MAX;
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            {
                // bitonic path: the padded row must fit in shared memory
                int64_t ncols_pad = 1;
                while (ncols_pad < op->src[0]->ne[0]) {
                    ncols_pad *= 2;
                }
                return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
            }
#else
            return op->src[0]->ne[0] <= INT_MAX;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return true;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS_MEAN:
            return ggml_cuda_get_rows_mean_supported(op);
        case GGML_OP_KQ_MASK:
            return ggml_cuda_kq_mask_supported(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

// Expert cache in device memory for a MoE weight left in host memory, see expert-cache.cuh.
static void * ggml_backend_cuda_expert_cache_create(ggml_backend_dev_t dev, const void * host_buf, size_t host_buf_size,
        const void * host_ptr, int64_t n_expert, int cap, size_t expert_bytes, size_t chunk_full, size_t chunk_dev,
        size_t offset_dev, int64_t n_chunks, int max_ids) {
    if (dev == nullptr || ggml_backend_dev_backend_reg(dev) != ggml_backend_cuda_reg()) {
        return nullptr;
    }
    const int device = ((ggml_backend_cuda_device_context *) dev->context)->device;
    return ggml_cuda_expert_cache_create(device, host_buf, host_buf_size, host_ptr, n_expert, cap, expert_bytes,
        chunk_full, chunk_dev, offset_dev, n_chunks, max_ids);
}

static void ggml_backend_cuda_expert_cache_free(void * cache) {
    ggml_cuda_expert_cache_free(cache);
}

static bool ggml_backend_cuda_expert_cache_fill(ggml_backend_t backend, void * cache, const void * pool_data, void * dst) {
    if (backend == nullptr || !ggml_backend_is_cuda(backend)) {
        return false;
    }
    return ggml_cuda_expert_cache_fill(*(ggml_backend_cuda_context *) backend->context, cache, pool_data, dst);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_comm_allgather_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allgather_tensor;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_prepare") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_prepare;
    }
    if (strcmp(name, "ggml_backend_expert_cache_create") == 0) {
        return (void *)ggml_backend_cuda_expert_cache_create;
    }
    if (strcmp(name, "ggml_backend_expert_cache_free") == 0) {
        return (void *)ggml_backend_cuda_expert_cache_free;
    }
    if (strcmp(name, "ggml_backend_expert_cache_fill") == 0) {
        return (void *)ggml_backend_cuda_expert_cache_fill;
    }
    if (strcmp(name, "ggml_backend_expert_cache_read_slots") == 0) {
        return (void *)ggml_cuda_expert_cache_read_slots;
    }
    if (strcmp(name, "ggml_backend_expert_cache_warm_tag") == 0) {
        return (void *)ggml_cuda_expert_cache_warm_tag;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_launch") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_launch;
    }
    if (strcmp(name, "ggml_backend_session_plan") == 0) {
        return (void *)ggml_backend_cuda_session_plan;
    }
    if (strcmp(name, "ggml_backend_set_tensors_batch") == 0) {
        return (void *)ggml_backend_cuda_set_tensors_batch;
    }
    if (strcmp(name, "ggml_backend_session_begin") == 0) {
        return (void *)ggml_backend_cuda_session_begin;
    }
    if (strcmp(name, "ggml_backend_session_launch") == 0) {
        return (void *)ggml_backend_cuda_session_launch;
    }
    if (strcmp(name, "ggml_backend_session_allreduce_supported") == 0) {
        return (void *)ggml_backend_cuda_session_allreduce_supported;
    }
    if (strcmp(name, "ggml_backend_session_allreduce") == 0) {
        return (void *)ggml_backend_cuda_session_allreduce;
    }
    if (strcmp(name, "ggml_backend_session_allreduce_next") == 0) {
        return (void *)ggml_backend_cuda_session_allreduce_next;
    }
    if (strcmp(name, "ggml_backend_session_end") == 0) {
        return (void *)ggml_backend_cuda_session_end;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = ggml_cuda_op_offload_min_batch();

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

// the records of the spin waits that timed out, one per device (see spin.cuh), in pinned host memory; nullptr if it
//   could not be allocated, then the waits only trap
static ggml_cuda_spin_record * ggml_cuda_spin_records = nullptr;

// GGML_CUDA_SPIN_TIMEOUT_S: the limit of the kernels' spin waits in seconds, 0 = none (see spin.cuh)
void ggml_cuda_spin_init(int device) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    std::call_once(once[device], [device]() {
        static const char * env = getenv("GGML_CUDA_SPIN_TIMEOUT_S");
        ggml_cuda_set_device(device);
        static std::once_flag once_alloc;
        std::call_once(once_alloc, []() {
            void * p = nullptr;
            if (cudaHostAlloc(&p, GGML_CUDA_MAX_DEVICES*sizeof(ggml_cuda_spin_record), cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) {
                (void) cudaGetLastError();
                return;
            }
            memset(p, 0, GGML_CUDA_MAX_DEVICES*sizeof(ggml_cuda_spin_record));
            ggml_cuda_spin_records = (ggml_cuda_spin_record *) p;
        });
        ggml_cuda_spin_config cfg = { GGML_CUDA_SPIN_LIMIT_DEFAULT, nullptr };
        if (env != nullptr) {
            const double s = atof(env);
            cfg.limit = 0;
            if (s > 0.0) {
                const double units = s*1e9 / (double) (1ull << GGML_CUDA_SPIN_SHIFT);
                cfg.limit = units >= GGML_CUDA_SPIN_LIMIT_MAX ? GGML_CUDA_SPIN_LIMIT_MAX : std::max(1u, (unsigned int) units);
            }
            if (cfg.limit == 0) {
                GGML_LOG_INFO("%s: device %d: no limit on spin waits (GGML_CUDA_SPIN_TIMEOUT_S=%s)\n", __func__, device, env);
            } else {
                GGML_LOG_INFO("%s: device %d: spin waits abort after %.1f s (GGML_CUDA_SPIN_TIMEOUT_S=%s)\n", __func__, device,
                    cfg.limit * (double) (1ull << GGML_CUDA_SPIN_SHIFT) * 1e-9, env);
            }
        }
        if (ggml_cuda_spin_records != nullptr) {
            void * dev = nullptr;
            if (cudaHostGetDevicePointer(&dev, ggml_cuda_spin_records + device, 0) == cudaSuccess) {
                cfg.record = (ggml_cuda_spin_record *) dev;
            } else {
                (void) cudaGetLastError();
            }
        }
        ggml_cuda_spin_set_allreduce(cfg);
        ggml_cuda_spin_set_allreduce_p2p(cfg);
        ggml_cuda_spin_set_mmvf(cfg);
        ggml_cuda_spin_set_norm(cfg);
        ggml_cuda_spin_set_expert_cpu(cfg);
    });
}

void ggml_cuda_spin_print() {
    static const char * names[GGML_CUDA_SPIN_SITE_COUNT] = {
        "ggml_cuda_p2p_ar_kernel", "ggml_cuda_p2p_ag_kernel", "ggml_cuda_p2p_ar_ll_kernel", "ggml_cuda_p2p_ar_ll_kernel_n",
        "ggml_cuda_p2p_ar_ll_kernel_n (receive)", "ggml_cuda_p2p_ar_ll_moe_reduce_kernel", "rms_norm_pre_add_ar_f32",
        "rms_norm_pre_add_ar_f32 (q8_1 blocks)", "rms_norm_pre_add_ar_split_f32", "ggml_cuda_ar_kernel", "mul_mat_vec_f_router",
        "hc_post_blocks_sync", "row_split_reduce", "rms_norm_pre_add_f32 (q8_1 blocks)", "rms_norm_add_rms_norm_f32 (q8_1 blocks)",
        "expert_cpu_wait_add",
    };
    if (ggml_cuda_spin_records == nullptr) {
        return;
    }
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
        const volatile ggml_cuda_spin_record & r = ggml_cuda_spin_records[d];
        const unsigned int site = r.site;
        if (site == 0 || site > GGML_CUDA_SPIN_SITE_COUNT) {
            continue;
        }
        const char * name = names[site - 1];
        const double s    = r.limit * (double) (1ull << GGML_CUDA_SPIN_SHIFT) * 1e-9;
        if (r.peer >= 0) {
            fprintf(stderr, "%s%d: %s (rank %d) waited %.1f s for rank %d, which faulted or stopped "
                "(GGML_CUDA_SPIN_TIMEOUT_S sets the limit, 0 = none)\n", GGML_CUDA_NAME, d, name, r.rank, s, r.peer);
        } else if (r.peer == GGML_CUDA_SPIN_PEER_HOST) {
            fprintf(stderr, "%s%d: %s waited %.1f s for the CPU threads (GGML_CUDA_SPIN_TIMEOUT_S sets the limit, 0 = none)\n",
                GGML_CUDA_NAME, d, name, s);
        } else if (r.peer == GGML_CUDA_SPIN_PEER_DEVICE) {
            fprintf(stderr, "%s%d: %s waited %.1f s for another device, which faulted or stopped "
                "(GGML_CUDA_SPIN_TIMEOUT_S sets the limit, 0 = none)\n", GGML_CUDA_NAME, d, name, s);
        } else {
            fprintf(stderr, "%s%d: %s waited %.1f s for the other blocks of its launch "
                "(GGML_CUDA_SPIN_TIMEOUT_S sets the limit, 0 = none)\n", GGML_CUDA_NAME, d, name, s);
        }
    }
    fflush(stderr);
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    ggml_cuda_spin_init(device);
    ggml_cuda_preload_modules(device);

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
