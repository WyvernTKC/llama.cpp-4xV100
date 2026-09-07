# MoE expert offload: `-ncmoe`, `GGML_META_PARTIAL_COPY`, `GGML_META_STAGE_SLOTS`, `GGML_META_MOE_OFFLOAD_MAX_KIB`

These three knobs control where a MoE model's expert weights live and how they get
streamed to the GPU(s) when they don't fit in VRAM. They matter most for large MoE
models (DeepSeek-V3/V4, Mixtral, Qwen-MoE, etc.) that are bigger than available VRAM.

## `-ncmoe` / `--n-cpu-moe N`

```
-ncmoe N
--n-cpu-moe N
env: LLAMA_ARG_N_CPU_MOE
```

Keeps the MoE expert weights of the **first N transformer layers** on the CPU
(pinned host memory, see below) instead of the GPU. The attention/norm/router
tensors of those layers, and everything in the other layers, still go to the GPU
as usual. `-cmoe` / `--cpu-moe` is shorthand for "all layers" (`-ncmoe 999` or similar).

There's also a draft-model variant for speculative decoding:
`--spec-draft-n-cpu-moe` / `-ncmoed` / `--n-cpu-moe-draft`.

**Why this exists:** MoE models route each token through only a handful of
experts (e.g. 6 of 256), so the expert weights are the "cold" part of the model
per forward pass. Leaving them on the CPU and copying only what's used means you
can run a model much larger than VRAM at a much smaller decode-time penalty than
naively offloading dense weights would cost.

**Usage notes learned from testing (V100 boxes, DeepSeek-V4-Flash and similar):**

- **Always pass `--load-mode none` (`-lm none`) alongside `-ncmoe`.** The default
  `auto` load mode resolves to `mmap`, which forces the CPU-resident expert
  buffer to be *pageable* memory instead of the pinned host buffer the CUDA
  backend would otherwise pick. Measured cost of leaving this on default: **2.8x
  slower prefill, 1.15x slower decode**. llama.cpp already prints a warning
  about this ("consider using --load-mode none for better performance") — heed it.
- **`-ncmoe N` takes the *first* N layers**, which is not necessarily a balanced
  split of memory across devices under `-sm tensor`/`-sm row`. Taking the first N
  layers off GPU0 disproportionately can starve or overload one GPU relative to
  the others (seen OOM at N=16 on a config that had headroom on average).
  Prefer spreading host residency across the layer range with `-ot` overrides
  if you hit this.
- **Some models don't need it at all.** If the model uses MLA (multi-head latent
  attention, e.g. DeepSeek-V3/V4) the KV cache is tiny regardless of context
  length (thousands of MiB, not tens of GB), so it's very possible the *expert
  weights* are the only thing keeping the model from fitting fully in VRAM.
  Check the per-device `model buffer size` / `KV buffer size` / `compute buffer
  size` lines in the load log before assuming you need to offload anything —
  don't assume long context forces expert offload without checking.
- **Under `-sm tensor`, offloaded experts are now streamed to the GPU during
  prefill too** (see `GGML_META_PARTIAL_COPY` context below) — this wasn't
  always true. Older builds without this fix ran offloaded-expert prefill
  entirely on CPU under `-sm tensor`, at zero benefit from having multiple
  GPU links available. If you're on an older checkout and `-sm tensor` prefill
  isn't faster than `-sm layer` with the same `-ncmoe`, that's why.

## `GGML_META_PARTIAL_COPY` (env var)

```
GGML_META_PARTIAL_COPY=1
GGML_META_PARTIAL_COPY_MAX_BATCH=32   # default 32
```

Only relevant under `-sm tensor` (the "meta" / tensor-parallel backend), and only
when some expert weights are CPU-resident (`-ncmoe`/`-cmoe`).

Normally, when the meta backend needs a host-resident `MUL_MAT_ID` (expert
matmul) tensor, it copies the **entire layer's expert weights** to the GPUs
before computing (a "whole-layer" copy). Single-GPU backends instead copy only
the experts actually selected by the token routing for that batch ("used-experts-
only" copy), which is far cheaper when few experts are active — e.g. a decode
batch that only touches 6 of 256 experts moves ~1/40th the bytes.

Setting `GGML_META_PARTIAL_COPY=1` opts the meta backend into that same
used-experts-only path instead of always doing a whole-layer copy.

**Why it's off by default:** it's a net win only while few experts are live per
batch. As batch size grows, the fraction of experts touched climbs toward 100%
(a batch of B tokens touches roughly `n_expert * (1 - (1 - k/n_expert)^B)` of
them for top-k routing), at which point you're paying the fixed overhead of a
partial copy — a stream drain plus a per-layer routing-ids readback, which
under tensor-parallel is a synchronization barrier across all GPUs — for no
bandwidth savings. `GGML_META_PARTIAL_COPY_MAX_BATCH` (default 32) is the batch
size ceiling above which the code falls back to the whole-layer copy
regardless of the flag; raise or lower it if your batch sizes/expert counts
differ meaningfully from the default assumption.

**When to try it:** decode-heavy or small-batch workloads (interactive serving,
batch size close to 1) with a MoE model that has a low active-expert fraction
(large `n_expert`, small `n_expert_used`) and a meaningful share of experts on
CPU. Benchmark before/after — the win depends heavily on your batch size
distribution and expert count; large-batch prefill workloads should generally
leave it off.

## `GGML_META_STAGE_SLOTS` (env var)

```
GGML_META_STAGE_SLOTS=N   # default 4, set 0-1 to disable staging
```

Also `-sm tensor`-only. Controls how many host-to-device staging slots the meta
backend uses when streaming CPU-resident expert weights to the GPUs — i.e. how
many expert-tensor-sized copies can be in flight / queued ahead at once, each
costing one expert tensor's worth of extra VRAM per slot.

**Why it exists:** with one slot, the host queues a copy, waits for it to land,
computes, then queues the next — copy and compute don't overlap. More slots let
the host get further ahead, keeping the GPU fed continuously instead of
stalling on each transfer.

**Tuning:** measured on 4x V100 with DeepSeek-V4-Flash: 4 slots reached 94% of
the compute-bound ceiling, 12 slots reached 99%. Below 2 slots, staging ahead
doesn't help and the code disables staging (falls back to synchronous
per-op copy). There are diminishing returns past ~4-12 depending on your
transfer/compute ratio, and each additional slot costs VRAM (one expert
tensor's size, times however many devices stage concurrently) — don't raise it
past the point of measurable benefit, and watch for OOM if you're already close
to the VRAM ceiling from `-ncmoe` residency choices.

## `GGML_META_MOE_OFFLOAD_MAX_KIB` (env var)

```
GGML_META_MOE_OFFLOAD_MAX_KIB=N   # default 5120, set 0 to never offload
```

`-sm tensor`-only. The meta backend normally offloads an op only above the CUDA
backend's batch threshold (~32 tokens), but a `mul_mat_id` whose experts live in
host memory also offloads below it while the transfer stays cheap. The budget is
KiB moved per token, per expert tensor, per device:

```
staged  = partial copy ? min(n_tokens*n_expert_used, n_expert) : n_expert
KiB/tok = nbytes(experts)/n_expert * staged / (n_devices * n_tokens * 1024)
```

The `staged` term is the part that matters. Without `GGML_META_PARTIAL_COPY` the
whole expert tensor moves every ubatch, so decode drags the entire tensor over
PCIe per token and the CPU path always wins. With it, only the routed experts
move and the transfer is small enough to pay off on most models.

Measured on 4x V100, `-sm tensor` tg32, against the old expert-count gate:

| model | experts | MiB/tok/tensor | old gate | budget | old + partial | budget + partial |
| --- | --- | --- | --- | --- | --- | --- |
| nemotron_h_moe 33B | 128 / 6 | 7.58 | 1.37 | **34.97** | 17.15 | **33.24** |
| qwen35moe 35B Q8_0 | 256 / 8 | 2.13 | 0.66 | **14.95** | - | - |
| qwen35moe 35B Q4_K | 256 / 8 | 1.64 | 0.99 | **24.98** | 15.49 | 15.59 |

Prefill is unaffected in every case (both gates offload at prompt batch sizes),
and perplexity is unchanged to 4 decimals.

**What the budget does not model** is the other side of the comparison: how long
the CPU would take to do the same matmul. That is why a Q2_K DeepSeek-V4 wants the
offload at 3.94 MiB/token while a Q4_K qwen35moe does not at 1.64 - the DeepSeek
CPU matmul is far slower. The budget is set at 5 MiB so every model measured here
lands on the right side, but it is a one-sided test. Where the two disagree, the
expert cache (`GGML_META_EXPERT_CACHE`) beats both and overrides the budget.

`GGML_META_MOE_OFFLOAD_MIN_EXPERTS` still works and replaces the budget with the
old `n_expert >= N` test when set, so existing command lines keep their behaviour.

## Quick recipe

For a MoE model too large for VRAM under `-sm tensor`:

```bash
GGML_META_STAGE_SLOTS=8 ./llama-server \
  -m model.gguf \
  -sm tensor \
  -ncmoe <N>       \
  -lm none         \
  -ub 1024
# add GGML_META_PARTIAL_COPY=1 if your workload is decode/low-batch heavy,
# and GGML_META_EXPERT_CACHE=<n> on top of it for the best decode
```

Use [scripts/estimate-ncmoe.py](../scripts/estimate-ncmoe.py) to get a starting
value for `<N>` from your GGUF file and GPU VRAM. It sizes the KV cache from the model's own per-layer head counts and your `-c`, and prints the env vars above with it.
