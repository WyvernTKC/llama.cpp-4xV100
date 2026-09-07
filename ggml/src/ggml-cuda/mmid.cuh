#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

// Lists the (expert, tile column) pairs that hold work, so mul_mat_q can launch blocks only for
// those instead of for every expert x ncols_max/J entry. n_tiles_max must bound the pair count.
void ggml_cuda_launch_mmq_ids_tile_map(
        const int32_t * expert_bounds, int2 * tile_map,
        int n_experts, int J, int n_tiles_max, int jt_empty, cudaStream_t stream);
