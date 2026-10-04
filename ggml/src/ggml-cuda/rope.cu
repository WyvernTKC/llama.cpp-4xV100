#include "convert.cuh"
#include "cpy-utils.cuh"
#include "fwht.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            ggml_cuda_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    ggml_cuda_pdl_lc();
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;
    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs/2 + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

// the angle of rotated channel pair iw/2 of token i2, from the position of the section the pair belongs to
static __device__ __forceinline__ float rope_multi_theta_base(
        const int32_t * pos, const int i2, const int ne02, const int iw, const float theta_scale,
        const mrope_sections sections, const bool is_imrope) {
    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (iw / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    }
    return theta_base;
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_multi(const T *            x,
                                  T *                  dst,
                                  const int            ne00,
                                  const int            ne01,
                                  const int            ne02,
                                  const int            s01,
                                  const int            s02,
                                  const int            s03,
                                  const int            s1,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
                                  const int            n_offs,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
                                  const float          attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const mrope_sections sections,
                                  const bool           is_imrope,
                                  const bool           inplace) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0/2 + 0] = x[ix + i0/2 + 0];
        dst[idst + i0/2 + 1] = x[ix + i0/2 + 1];

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = rope_multi_theta_base(pos, i2, ne02, iw, theta_scale, sections, is_imrope);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_offs/2 + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T *            x,
                                   T *                  dst,
                                   const int            ne00,
                                   const int            ne01,
                                   const int            ne02,
                                   const int            s01,
                                   const int            s02,
                                   const int            s03,
                                   const int            s1,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
                                   const float          attn_factor,
                                   const rope_corr_dims corr_dims,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[i2] * powf(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = pos[i2 + ne02] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, typename T, typename D>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};

    if (freq_factors == nullptr) {
        ggml_cuda_kernel_launch(rope_neox<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        ggml_cuda_kernel_launch(rope_neox<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T>
static void rope_multi_cuda(const T *            x,
                            T *                  dst,
                            const int            ne00,
                            const int            ne01,
                            const int            ne02,
                            const int            s01,
                            const int            s02,
                            const int            s03,
                            const int            s1,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
                            const int            n_offs,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
                            const float          freq_base,
                            const float          ext_factor,
                            const float          attn_factor,
                            const rope_corr_dims corr_dims,
                            const float *        freq_factors,
                            const mrope_sections sections,
                            const bool           is_imrope,
                            const bool           inplace,
                            cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, false, T>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, inplace);
    } else {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, true, T>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, inplace);
    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T *            x,
                             T *                  dst,
                             const int            ne00,
                             const int            ne01,
                             const int            ne02,
                             const int            s01,
                             const int            s02,
                             const int            s03,
                             const int            s1,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
                             const float          freq_base,
                             const float          ext_factor,
                             const float          attn_factor,
                             const rope_corr_dims corr_dims,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               dst,
                            const ggml_tensor *         set_rows = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    GGML_ASSERT(src0->type == dst->type || (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    const size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    const size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    mrope_sections sections;

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = dst_d == src0->data;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_multi_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_multi_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * set_rows) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}

// fused [RMS_NORM [+ MUL]] [+ ROPE] [+ RESHAPE + MUL_MAT (Hadamard)] [+ RESHAPE + VIEW + SET_ROWS]
// one block per row: block_reduce gives the norm scale, then each thread applies mul and rope to the elements it owns.
// The Walsh-Hadamard rotation of quantized K/V caches, and a SET_ROWS into a quantized cache, need the whole row, which
//   then goes through shared memory; the rotation is fwht_block_shared and the quantization that of set_rows.
template <typename D>
static constexpr __host__ __device__ bool rope_fused_dst_is_block() {
    return !std::is_same_v<D, float> && !std::is_same_v<D, half>;
}

template <typename block_t>
static __device__ __forceinline__ void rope_fused_store_quantized(const float * s, block_t * y, const int ncols) {
    static_assert(QK8_0 == 32 && QK4_0 == 32 && QK4_1 == 32 && QK5_0 == 32 && QK5_1 == 32 && QK4_NL == 32, "bad block size");
    const int nblocks = ncols/32;

    if constexpr (std::is_same_v<block_t, block_q8_0> || std::is_same_v<block_t, block_q4_0>) {
        // all threads of a group run the same iterations
        for (int ib = threadIdx.x/QUANTIZE_GROUP_SIZE; ib < nblocks; ib += blockDim.x/QUANTIZE_GROUP_SIZE) {
            if constexpr (std::is_same_v<block_t, block_q8_0>) {
                quantize_f32_q8_0_group(s + 32*ib, y + ib, threadIdx.x % QUANTIZE_GROUP_SIZE);
            } else {
                quantize_f32_q4_0_group(s + 32*ib, y + ib, threadIdx.x % QUANTIZE_GROUP_SIZE);
            }
        }
    } else {
        for (int ib = threadIdx.x; ib < nblocks; ib += blockDim.x) {
            if constexpr (std::is_same_v<block_t, block_q4_1>) {
                quantize_f32_q4_1_block(s + 32*ib, y + ib);
            } else if constexpr (std::is_same_v<block_t, block_q5_0>) {
                quantize_f32_q5_0_block(s + 32*ib, y + ib);
            } else if constexpr (std::is_same_v<block_t, block_q5_1>) {
                quantize_f32_q5_1_block(s + 32*ib, y + ib);
            } else {
                static_assert(std::is_same_v<block_t, block_iq4_nl>, "bad block type");
                quantize_f32_iq4_nl_block(s + 32*ib, y + ib);
            }
        }
    }
}

// one row of rms_norm_mul_rope_f32 (nchannels: the channels of the grid), also of rms_norm_mul_rope_multi
template <int block_size, bool has_ff, typename D>
static __device__ __forceinline__ void rms_norm_mul_rope_row(
        const float * x, D * dst, const int ncols,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const bool has_norm, const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const mrope_sections sections, const bool is_mrope, const bool is_imrope,
        const int n_offs, const int n_rot, const float rot_scale,
        const int row, const int channel, const int sample, const int nchannels) {
    const int tid     = threadIdx.x;

    x += sample*s03 + channel*s02 + row*s01;

    // a plain rms_norm has no weight, and scaling by it alone rounds the same as the norm kernel
    const bool has_mul = mul != nullptr;
    if (has_mul) {
        const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
        mul += mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;
    }

    extern __shared__ float s_sum[];
    float * s_row = s_sum + 32; // the whole row, with a rotation or a quantized destination

    const bool via_shared = rope_fused_dst_is_block<D>() || n_rot != 0;

    float scale = 1.0f;

    ggml_cuda_pdl_sync();
    if (has_norm) {
        float tmp = 0.0f;
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = x[col];
            tmp += xi * xi;
        }

        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

        scale = rsqrtf(tmp/ncols + eps);
    }

    int64_t idst = sample*s3 + channel*s2 + row*s1;
    if (set_rows_stride != 0) {
        idst = row*s1 + row_indices[channel]*set_rows_stride;
    }
    dst += idst;

    for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
        int ix0;
        int ix1;
        if (is_neox && i0 < n_dims) {
            ix0 = i0/2;
            ix1 = i0/2 + n_dims/2;
        } else {
            ix0 = i0 + 0;
            ix1 = i0 + 1;
        }

        float x0 = x[ix0];
        float x1 = x[ix1];
        if (has_norm) {
            x0 = has_mul ? scale * x0 * mul[fastmodulo(ix0, mul_ncols_packed)] : scale * x0;
            x1 = has_mul ? scale * x1 * mul[fastmodulo(ix1, mul_ncols_packed)] : scale * x1;
        }

        float y0;
        float y1;

        // n_offs (only with the norm pairing) rotates the n_dims channels that follow it, as rope_norm does
        if (i0 < n_offs || i0 >= n_offs + n_dims) {
            y0 = x0;
            y1 = x1;
        } else {
            const int iw = i0 - n_offs;

            // multi-section rope takes each pair's position from its section, as rope_multi does
            const float theta_base  = is_mrope ? rope_multi_theta_base(pos, channel, nchannels, iw, theta_scale, sections, is_imrope)
                                               : pos[channel]*powf(theta_scale, iw/2.0f);
            const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

            // spelled out so that the rounding matches what rope_norm and rope_neox compile to (the x1 product rounded,
            //   then one fma over x0); left to the compiler the contraction of the second sum can differ between kernels
            y0 = __fmaf_rn(cos_theta, x0, -__fmul_rn(sin_theta, x1));
            y1 = __fmaf_rn(sin_theta, x0,  __fmul_rn(cos_theta, x1));
        }

        if (via_shared) {
            s_row[ix0] = y0;
            s_row[ix1] = y1;
        } else if constexpr (!rope_fused_dst_is_block<D>()) {
            dst[ix0] = ggml_cuda_cast<D>(y0);
            dst[ix1] = ggml_cuda_cast<D>(y1);
        }
    }

    if (!via_shared) {
        return;
    }

    if (n_rot != 0) {
        fwht_block_shared(s_row, ncols, n_rot, rot_scale);
    } else {
        __syncthreads();
    }

    if constexpr (rope_fused_dst_is_block<D>()) {
        rope_fused_store_quantized(s_row, dst, ncols);
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            dst[col] = ggml_cuda_cast<D>(s_row[col]);
        }
    }
}

template <int block_size, bool has_ff, typename D>
static __global__ void rms_norm_mul_rope_f32(
        const float * x, D * dst, const int ncols,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const bool has_norm, const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const mrope_sections sections, const bool is_mrope, const bool is_imrope,
        const int n_offs, const int n_rot, const float rot_scale) {
    ggml_cuda_pdl_lc();
    rms_norm_mul_rope_row<block_size, has_ff, D>(x, dst, ncols, s01, s02, s03, s1, s2, s3, has_norm, eps, mul,
        mul_s01, mul_s02, mul_s03, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
        n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale, freq_factors, row_indices,
        set_rows_stride, is_neox, sections, is_mrope, is_imrope, n_offs, n_rot, rot_scale,
        blockIdx.x, blockIdx.y, blockIdx.z, gridDim.y);
}

// The arguments of rms_norm_mul_rope_f32 for the rows of one token of one fusion (f32 or f16 out, no rotation), a
//   segment of rms_norm_mul_rope_multi; nrows 0: not one
struct rms_norm_rope_seg {
    const float *   x;
    void *          dst;
    int             ncols;
    int64_t         s01, s02, s03;
    int64_t         s1, s2, s3;
    bool            has_norm;
    float           eps;
    const float *   mul;
    int64_t         mul_s01, mul_s02, mul_s03;
    uint3           mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed;
    int             n_dims;
    const int32_t * pos;
    float           freq_scale, ext_factor, attn_factor;
    rope_corr_dims  corr_dims;
    float           theta_scale;
    const float *   freq_factors;
    const int64_t * row_indices;
    int             set_rows_stride;
    bool            is_neox;
    mrope_sections  sections;
    bool            is_mrope, is_imrope;
    int             n_offs;
    int             nrows;
    ggml_type       dst_type;
};

template <typename D>
static __device__ __forceinline__ void rms_norm_mul_rope_seg_row(const rms_norm_rope_seg & a, const int row) {
    if (a.freq_factors != nullptr) {
        rms_norm_mul_rope_row<256, true, D>(a.x, (D *) a.dst, a.ncols, a.s01, a.s02, a.s03, a.s1, a.s2, a.s3, a.has_norm,
            a.eps, a.mul, a.mul_s01, a.mul_s02, a.mul_s03, a.mul_ncols_packed, a.mul_nrows_packed, a.mul_nchannels_packed,
            a.mul_nsamples_packed, a.n_dims, a.pos, a.freq_scale, a.ext_factor, a.attn_factor, a.corr_dims, a.theta_scale,
            a.freq_factors, a.row_indices, a.set_rows_stride, a.is_neox, a.sections, a.is_mrope, a.is_imrope, a.n_offs,
            0, 0.0f, row, 0, 0, 1);
    } else {
        rms_norm_mul_rope_row<256, false, D>(a.x, (D *) a.dst, a.ncols, a.s01, a.s02, a.s03, a.s1, a.s2, a.s3, a.has_norm,
            a.eps, a.mul, a.mul_s01, a.mul_s02, a.mul_s03, a.mul_ncols_packed, a.mul_nrows_packed, a.mul_nchannels_packed,
            a.mul_nsamples_packed, a.n_dims, a.pos, a.freq_scale, a.ext_factor, a.attn_factor, a.corr_dims, a.theta_scale,
            a.freq_factors, a.row_indices, a.set_rows_stride, a.is_neox, a.sections, a.is_mrope, a.is_imrope, a.n_offs,
            0, 0.0f, row, 0, 0, 1);
    }
}

// The rows of two or three fusions of one token in one launch of 256 threads per block: the blocks of a0, then those
//   of a1 and a2. Each block runs the row as rms_norm_mul_rope_f32 does.
template <typename D0, typename D1, typename D2>
static __global__ void rms_norm_mul_rope_multi(const rms_norm_rope_seg a0, const rms_norm_rope_seg a1, const rms_norm_rope_seg a2) {
    ggml_cuda_pdl_lc();
    int row = blockIdx.x;
    if (row < a0.nrows) {
        rms_norm_mul_rope_seg_row<D0>(a0, row);
        return;
    }
    row -= a0.nrows;
    if (row < a1.nrows) {
        rms_norm_mul_rope_seg_row<D1>(a1, row);
        return;
    }
    rms_norm_mul_rope_seg_row<D2>(a2, row - a1.nrows);
}

template <typename D>
static void rms_norm_mul_rope_cuda(
        const float * x, D * dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const bool has_norm, const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint32_t mul_ncols, const uint32_t mul_nrows,
        const uint32_t mul_nchannels, const uint32_t mul_nsamples,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const mrope_sections sections, const bool is_mrope, const bool is_imrope, const int n_offs,
        const int n_rot, cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);
    GGML_ASSERT(has_norm || mul == nullptr);
    GGML_ASSERT(n_rot == 0 || (n_rot >= 2 && (n_rot & (n_rot - 1)) == 0 && ncols % n_rot == 0));
    GGML_ASSERT(!rope_fused_dst_is_block<D>() || (set_rows_stride != 0 && ncols % 32 == 0));

    const dim3 blocks_num(nrows, nchannels, nsamples);

    const float theta_scale = n_dims > 0 ? powf(freq_base, -2.0f/n_dims) : 0.0f;
    const float rot_scale   = n_rot  > 0 ? ggml_cuda_fwht_scale(n_rot)   : 0.0f;

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    const bool   via_shared = rope_fused_dst_is_block<D>() || n_rot != 0;
    const size_t nbytes_shared = (32 + (via_shared ? ncols : 0))*sizeof(float);

    auto launch = [&](auto kernel, const int block_size) {
        const dim3 block_dims(block_size, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, nbytes_shared, stream};
        ggml_cuda_kernel_launch(kernel, launch_params,
            x, dst, ncols, s01, s02, s03, s1, s2, s3, has_norm, eps, mul, mul_s01, mul_s02, mul_s03,
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
            n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
            freq_factors, row_indices, set_rows_stride, is_neox, sections, is_mrope, is_imrope, n_offs, n_rot, rot_scale);
    };

    if (ncols < 1024) {
        if (freq_factors == nullptr) {
            launch(rms_norm_mul_rope_f32<256, false, D>, 256);
        } else {
            launch(rms_norm_mul_rope_f32<256, true, D>, 256);
        }
    } else {
        if (freq_factors == nullptr) {
            launch(rms_norm_mul_rope_f32<1024, false, D>, 1024);
        } else {
            launch(rms_norm_mul_rope_f32<1024, true, D>, 1024);
        }
    }
}

// seg: fill in the arguments of a segment of rms_norm_mul_rope_multi instead of launching
static void rms_norm_mul_rope_fused_impl(ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows,
        ggml_tensor * rot, const ggml_tensor * rows, rms_norm_rope_seg * seg) {
    GGML_ASSERT(rms_norm != nullptr || mul == nullptr);
    // a segment may also be a SET_ROWS alone of its rows
    GGML_ASSERT(rms_norm != nullptr || rope != nullptr || rot != nullptr || (seg != nullptr && set_rows != nullptr && rows != nullptr));

    const ggml_tensor * x = rms_norm ? rms_norm->src[0] : rope ? rope->src[0] : rot ? rot->src[1] : rows;
    // mul is null for a plain rms_norm straight into the rope
    const ggml_tensor * mul_src = mul == nullptr ? nullptr : mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    // the rows the kernel works on: those of the rope, else of the norm, else the caller's
    if (rope != nullptr) {
        rows = rope;
    } else if (rows == nullptr) {
        rows = mul ? mul : rms_norm;
    }
    GGML_ASSERT(rows != nullptr);

    float eps = 0.0f;
    if (rms_norm != nullptr) {
        memcpy(&eps, rms_norm->op_params, sizeof(float));
        GGML_ASSERT(eps >= 0.0f);
    }

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_src == nullptr || mul_src->type == GGML_TYPE_F32);
    GGML_ASSERT(rows->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(x) == ggml_nelements(rows));

    // the rotation acts on consecutive rows of n_rot values of the contiguous result, which never cross a row here
    const int n_rot = rot != nullptr ? rot->src[1]->ne[0] : 0;
    GGML_ASSERT(rot == nullptr || (rot->type == GGML_TYPE_F32 && ggml_is_contiguous(rot) && rows->ne[0] % n_rot == 0));

    void *          dst_d           = rot ? rot->data : rows->data;
    ggml_type       dst_type        = GGML_TYPE_F32;
    const int64_t * row_indices     = nullptr;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        dst_d           = set_rows->data;
        dst_type        = set_rows->type;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }

    int   n_dims      = 0;
    int   mode        = 0;
    int   n_ctx_orig  = 0;
    float freq_base   = 0.0f;
    float freq_scale  = 1.0f;
    float ext_factor  = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast   = 0.0f;
    float beta_slow   = 0.0f;
    int   n_offs      = 0;

    mrope_sections sections = {};
    const int32_t * pos          = nullptr;
    const float *   freq_factors = nullptr;
    rope_corr_dims  corr_dims    = {};

    if (rope != nullptr) {
        GGML_ASSERT(rope->type == GGML_TYPE_F32);

        n_dims     = ((const int32_t *) rope->op_params)[1];
        mode       = ((const int32_t *) rope->op_params)[2];
        n_ctx_orig = ((const int32_t *) rope->op_params)[4];

        memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
        memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
        memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
        memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
        memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
        memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));

        memcpy(&sections.v, (const int32_t *) rope->op_params + 11, sizeof(int)*4);

        pos          = (const int32_t *) rope->src[1]->data;
        freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;

        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

        n_offs = ((const int32_t *) rope->op_params)[15];
    }

    // multi-section rope rotates the same channel pairs as neox
    const bool is_mrope  = (mode & GGML_ROPE_TYPE_MROPE) && mode != GGML_ROPE_TYPE_VISION;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_neox   = (mode & GGML_ROPE_TYPE_NEOX) || is_mrope;

    // the grid follows the rows, which may be a reshape of the normed or rotated rows; that is only a relabeling of
    //   packed rows, so x is then addressed as rows of its own row stride, or as packed rows when it is contiguous
    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    int64_t s01 = x->nb[1] / ts0;
    int64_t s02 = x->nb[2] / ts0;
    int64_t s03 = x->nb[3] / ts0;
    if (!ggml_are_same_shape(x, rows)) {
        GGML_ASSERT(mul_src == nullptr || ggml_nrows(mul_src) == 1);
        if (x->ne[0] != rows->ne[0]) {
            GGML_ASSERT(ggml_is_contiguous(x));
            s01 = rows->ne[0];
        } else {
            GGML_ASSERT(x->nb[2] == x->nb[1]*x->ne[1] && x->nb[3] == x->nb[2]*x->ne[2]);
        }
        s02 = s01*rows->ne[1];
        s03 = s02*rows->ne[2];
    }

    const size_t ts_mul = ggml_type_size(GGML_TYPE_F32);
    GGML_ASSERT(mul_src == nullptr || mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src ? mul_src->nb[1] / ts_mul : 0;
    const int64_t mul_s02 = mul_src ? mul_src->nb[2] / ts_mul : 0;
    const int64_t mul_s03 = mul_src ? mul_src->nb[3] / ts_mul : 0;
    const float * mul_d   = mul_src ? (const float *) mul_src->data : nullptr;
    const int64_t mul_ne0 = mul_src ? mul_src->ne[0] : 1;
    const int64_t mul_ne1 = mul_src ? mul_src->ne[1] : 1;
    const int64_t mul_ne2 = mul_src ? mul_src->ne[2] : 1;
    const int64_t mul_ne3 = mul_src ? mul_src->ne[3] : 1;

    // destination strides in its elements: the rows' own, or packed rows for the rotation and a quantized cache row
    const size_t ts_dst = ggml_type_size(rows->type);
    int64_t s1 = rows->nb[1] / ts_dst;
    int64_t s2 = rows->nb[2] / ts_dst;
    int64_t s3 = rows->nb[3] / ts_dst;
    if (rot != nullptr || ggml_is_quantized(dst_type)) {
        GGML_ASSERT(ggml_is_contiguous(rows) || set_rows != nullptr);
        s1 = rows->ne[0] / ggml_blck_size(dst_type);
        s2 = s1*rows->ne[1];
        s3 = s2*rows->ne[2];
    }

    if (seg != nullptr) {
        *seg = {};
        seg->dst_type = dst_type;
        if (rows->ne[2] != 1 || rows->ne[3] != 1 || rot != nullptr || rows->ne[0] >= 1024 || rows->ne[0] % 2 != 0 ||
                (dst_type != GGML_TYPE_F32 && dst_type != GGML_TYPE_F16)) {
            return;
        }
        seg->x               = (const float *) x->data;
        seg->dst             = dst_d;
        seg->ncols           = rows->ne[0];
        seg->s01             = s01;
        seg->s02             = s02;
        seg->s03             = s03;
        seg->s1              = s1;
        seg->s2              = s2;
        seg->s3              = s3;
        seg->has_norm        = rms_norm != nullptr;
        seg->eps             = eps;
        seg->mul             = mul_d;
        seg->mul_s01         = mul_s01;
        seg->mul_s02         = mul_s02;
        seg->mul_s03         = mul_s03;
        seg->mul_ncols_packed     = init_fastdiv_values(mul_ne0);
        seg->mul_nrows_packed     = init_fastdiv_values(mul_ne1);
        seg->mul_nchannels_packed = init_fastdiv_values(mul_ne2);
        seg->mul_nsamples_packed  = init_fastdiv_values(mul_ne3);
        seg->n_dims          = n_dims;
        seg->pos             = pos;
        seg->freq_scale      = freq_scale;
        seg->ext_factor      = ext_factor;
        seg->attn_factor     = attn_factor;
        seg->corr_dims       = corr_dims;
        seg->theta_scale     = n_dims > 0 ? powf(freq_base, -2.0f/n_dims) : 0.0f; // as rms_norm_mul_rope_cuda
        seg->freq_factors    = freq_factors;
        seg->row_indices     = row_indices;
        seg->set_rows_stride = set_rows_stride;
        seg->is_neox         = is_neox;
        seg->sections        = sections;
        seg->is_mrope        = is_mrope;
        seg->is_imrope       = is_imrope;
        seg->n_offs          = n_offs;
        seg->nrows           = rows->ne[1];
        return;
    }

    cudaStream_t stream = ctx.stream();

    auto run = [&](auto * dst_typed) {
        rms_norm_mul_rope_cuda(
            (const float *) x->data, dst_typed,
            rows->ne[0], rows->ne[1], rows->ne[2], rows->ne[3], s01, s02, s03, s1, s2, s3, rms_norm != nullptr, eps,
            mul_d, mul_s01, mul_s02, mul_s03,
            mul_ne0, mul_ne1, mul_ne2, mul_ne3,
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, sections, is_mrope, is_imrope, n_offs, n_rot, stream);
    };

    switch (dst_type) {
        case GGML_TYPE_F32:    run((float        *) dst_d); break;
        case GGML_TYPE_F16:    run((half         *) dst_d); break;
        case GGML_TYPE_Q8_0:   run((block_q8_0   *) dst_d); break;
        case GGML_TYPE_Q4_0:   run((block_q4_0   *) dst_d); break;
        case GGML_TYPE_Q4_1:   run((block_q4_1   *) dst_d); break;
        case GGML_TYPE_Q5_0:   run((block_q5_0   *) dst_d); break;
        case GGML_TYPE_Q5_1:   run((block_q5_1   *) dst_d); break;
        case GGML_TYPE_IQ4_NL: run((block_iq4_nl *) dst_d); break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows,
        ggml_tensor * rot, const ggml_tensor * rows) {
    rms_norm_mul_rope_fused_impl(ctx, rms_norm, mul, rope, set_rows, rot, rows, nullptr);
}

bool ggml_cuda_op_rms_norm_mul_rope_multi(ggml_backend_cuda_context & ctx, ggml_tensor * nodes[][4], const int n) {
    GGML_ASSERT(n == 2 || n == 3);
    rms_norm_rope_seg segs[3] = {};
    for (int s = 0; s < n; ++s) {
        // a SET_ROWS alone stores its rows as they are
        const bool store = nodes[s][0] == nullptr && nodes[s][2] == nullptr;
        rms_norm_mul_rope_fused_impl(ctx, nodes[s][0], nodes[s][1], nodes[s][2], nodes[s][3], nullptr,
            store ? nodes[s][3]->src[0] : nullptr, &segs[s]);
        if (segs[s].nrows == 0) {
            return false;
        }
    }
    // the first one into f32 (a Q), the others into f32 or f16 (a K or V cache)
    if (segs[0].dst_type != GGML_TYPE_F32) {
        return false;
    }
    const bool h1 = segs[1].dst_type == GGML_TYPE_F16;
    const bool h2 = n == 3 ? segs[2].dst_type == GGML_TYPE_F16 : h1;

    const dim3 blocks_num(segs[0].nrows + segs[1].nrows + segs[2].nrows, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, dim3(256, 1, 1), 32*sizeof(float), ctx.stream());
    if (h1 && h2) {
        ggml_cuda_kernel_launch(rms_norm_mul_rope_multi<float, half, half>, launch_params, segs[0], segs[1], segs[2]);
    } else if (h1) {
        ggml_cuda_kernel_launch(rms_norm_mul_rope_multi<float, half, float>, launch_params, segs[0], segs[1], segs[2]);
    } else if (h2) {
        ggml_cuda_kernel_launch(rms_norm_mul_rope_multi<float, float, half>, launch_params, segs[0], segs[1], segs[2]);
    } else {
        ggml_cuda_kernel_launch(rms_norm_mul_rope_multi<float, float, float>, launch_params, segs[0], segs[1], segs[2]);
    }
    return true;
}

bool ggml_cuda_rms_norm_mul_rope_fused_dst_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

// fused GET_ROWS_MEAN + RESHAPE + RMS_NORM + MUL + RESHAPE + ROPE (multi-section): the qwen4exp QSA key pooling
// One warp per pooled row of 128, lane l owns the columns l + 32*k. Every step repeats the arithmetic of the
// unfused kernels in their order, so the result is bitwise the same: the member sum and scale of k_get_rows_mean,
// the two-level sum of rms_norm_f32<256>, where each column is the only element of one thread, and rope_multi.
template <typename T>
static __device__ __forceinline__ float get_rows_mean_norm_rope_load(const char * row, const int col) {
    if constexpr (std::is_same_v<T, block_q8_0>) {
        const block_q8_0 * b = (const block_q8_0 *) row + col/QK8_0;
        const float d = b->d;
        float v = b->qs[col % QK8_0];
        v *= d;
        return v;
    } else {
        return ggml_cuda_cast<float>(((const T *) row)[col]);
    }
}

template <typename T, int n_group_fixed, int n_dims_64>
static __global__ void get_rows_mean_norm_rope_f32(
        const char * src0, const int32_t * src1, float * dst,
        const int nrows, const int ncols, const int n_group, const float group_scale,
        const uint3 ne1_fdv, const uint3 ne2_fdv,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12,
        const float eps, const float * mul, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const mrope_sections sections, const bool is_imrope) {
    constexpr int nk     = 128/WARP_SIZE;
    constexpr int n_dims = 64*n_dims_64;
    constexpr int k_half = n_dims/2/WARP_SIZE; // the slot that holds the second channel of a pair

    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (row >= nrows) {
        return;
    }

    const uint2 r1  = fast_div_modulo((uint32_t) row, ne1_fdv); // z,   i1
    const uint2 r2  = fast_div_modulo(r1.x, ne2_fdv);           // i12, i11
    const int   i1  = r1.y;
    const int   i11 = r2.y;
    const int   i12 = r2.x;

    const int ng = n_group_fixed > 0 ? n_group_fixed : n_group;

    ggml_cuda_pdl_sync();

    float x[nk] = {0.0f};
#pragma unroll
    for (int g = 0; g < ng; ++g) {
        const int i01 = src1[(i1*ng + g)*s10 + i11*s11 + i12*s12];
        const char * src_row = src0 + i01*nb01 + i11*nb02 + i12*nb03;
#pragma unroll
        for (int k = 0; k < nk; ++k) {
            const float v = get_rows_mean_norm_rope_load<T>(src_row, lane + k*WARP_SIZE);
            x[k] = g == 0 ? v : __fadd_rn(x[k], v);
        }
    }

    // warp k of the unfused block holds the columns of slot k, and lane k of its second pass the sum of that warp
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < nk; ++k) {
        x[k] = group_scale*x[k];
        const float p = warp_reduce_sum(x[k]*x[k]);
        sum = lane == k ? p : sum;
    }
    sum = warp_reduce_sum(sum);

    const float mean  = sum / ncols;
    const float scale = rsqrtf(mean + eps);

#pragma unroll
    for (int k = 0; k < nk; ++k) {
        x[k] = scale * x[k] * mul[lane + k*WARP_SIZE];
    }

    float * dst_row = dst + (int64_t) row*128;
#pragma unroll
    for (int k = 0; k < nk; ++k) {
        const int col = lane + k*WARP_SIZE;
        if (k >= 2*k_half) {
            dst_row[col] = x[k];
            continue;
        }
        if (k >= k_half) {
            continue;
        }

        const int   iw          = 2*col;
        const float theta_base  = rope_multi_theta_base(pos, row, nrows, iw, theta_scale, sections, is_imrope);
        const float freq_factor = 1.0f;

        float cos_theta;
        float sin_theta;
        rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

        const float x0 = x[k];
        const float x1 = x[k + k_half];

        dst_row[col]            = x0*cos_theta - x1*sin_theta;
        dst_row[col + n_dims/2] = x0*sin_theta + x1*cos_theta;
    }
}

template <typename T>
static auto get_rows_mean_norm_rope_kernel(const int n_group, const int n_dims) {
    if (n_group == 4) {
        return n_dims == 64 ? get_rows_mean_norm_rope_f32<T, 4, 1> : get_rows_mean_norm_rope_f32<T, 4, 2>;
    }
    return n_dims == 64 ? get_rows_mean_norm_rope_f32<T, 0, 1> : get_rows_mean_norm_rope_f32<T, 0, 2>;
}

void ggml_cuda_op_get_rows_mean_norm_rope_fused(ggml_backend_cuda_context & ctx,
        ggml_tensor * get_rows_mean, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope) {
    const ggml_tensor * src0 = get_rows_mean->src[0];
    const ggml_tensor * src1 = get_rows_mean->src[1];
    const ggml_tensor * w    = mul->src[1];

    GGML_ASSERT(get_rows_mean->ne[0] == 128 && w->type == GGML_TYPE_F32 && rope->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nrows(get_rows_mean) <= INT32_MAX);

    const int n_group = ggml_get_op_params_i32(get_rows_mean, 0);

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const int n_dims     = ((const int32_t *) rope->op_params)[1];
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    mrope_sections sections;

    memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (const int32_t *) rope->op_params + 11, sizeof(int)*4);

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    decltype(&get_rows_mean_norm_rope_f32<float, 4, 1>) kernel = nullptr;
    switch (src0->type) {
        case GGML_TYPE_F32:  kernel = get_rows_mean_norm_rope_kernel<float>     (n_group, n_dims); break;
        case GGML_TYPE_F16:  kernel = get_rows_mean_norm_rope_kernel<half>      (n_group, n_dims); break;
        case GGML_TYPE_Q8_0: kernel = get_rows_mean_norm_rope_kernel<block_q8_0>(n_group, n_dims); break;
        default:             GGML_ABORT("fatal error");
    }

    const int  nrows          = (int) ggml_nrows(get_rows_mean);
    const int  rows_per_block = 256/WARP_SIZE;
    const dim3 block_nums((nrows + rows_per_block - 1)/rows_per_block, 1, 1);
    const dim3 block_dims(rows_per_block*WARP_SIZE, 1, 1);

    kernel<<<block_nums, block_dims, 0, ctx.stream()>>>(
        (const char *) src0->data, (const int32_t *) src1->data, (float *) rope->data,
        nrows, (int) get_rows_mean->ne[0], n_group, 1.0f/n_group,
        init_fastdiv_values(get_rows_mean->ne[1]), init_fastdiv_values(get_rows_mean->ne[2]),
        src0->nb[1], src0->nb[2], src0->nb[3],
        src1->nb[0]/sizeof(int32_t), src1->nb[1]/sizeof(int32_t), src1->nb[2]/sizeof(int32_t),
        eps, (const float *) w->data, (const int32_t *) rope->src[1]->data,
        freq_scale, ext_factor, attn_factor, corr_dims, theta_scale, sections, mode == GGML_ROPE_TYPE_IMROPE);
}
