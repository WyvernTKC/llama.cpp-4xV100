#include "common.cuh"

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst);

// The block scores of the qwen4exp QSA indexer (its unfused branch) expanded to the cells and masked, in two launches:
//   dst[c, t, s] = (((relu(x[b,0,t,s]) + relu(x[b,1,t,s])) + ...) + bias[b,t,s]) + mask[c,t,s]
//   with b = cell_blk[c, s], each add rounded as its own op. x: [n_blocks, n_h, n_tps, n_stream], bias: [n_blocks,
//   n_tps, n_stream], cell_blk: [n_kv, n_stream], mask: rows of n_kv per token (f16 or f32), mask_nb_stream bytes apart
//   per stream; dst: [n_kv, n_tps, n_stream].
void ggml_cuda_qsa_cell_scores(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * bias,
    const ggml_tensor * cell_blk, const ggml_tensor * mask, size_t mask_nb_stream, ggml_tensor * dst);

// The QSA attention mask of the top-k cells as FILL -> SET_ROWS -> ADD compute it, in one launch: rows of the fill value
//   with the cells idx[k, t, s] set to vals[0, k, t, s] (in dst's type), plus kq. dst, kq: [n_kv, n_tps, 1, n_stream];
//   idx (i32): [n_top_k, n_tps, n_stream]; vals: [1, n_top_k, n_tps, n_stream]. dst may sit exactly on kq; idx or vals
//   under dst are copied first (they must be contiguous then).
void ggml_cuda_qsa_topk_mask(ggml_backend_cuda_context & ctx, float fill, const ggml_tensor * vals, const ggml_tensor * idx,
    const ggml_tensor * kq, ggml_tensor * dst);
