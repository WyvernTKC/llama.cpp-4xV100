# llama.cpp on 4× V100 — tensor split and MoE expert offload

Notes for the `sm-tensor-4xv100` branch of this fork. Everything here targets one machine:
**4× Tesla V100-SXM2-32GB** (sm_70, 128 GiB VRAM total), Windows, CUDA 12.8.

The branch adds a *tensor-parallel* split mode (`-sm tensor`) on top of upstream llama.cpp, plus the
machinery to run **MoE models that are larger than VRAM** by keeping some expert layers in system RAM
and streaming them per token.

If you only read one thing: for a 156 GiB model on 128 GiB of VRAM, decode went from **15.7 to
39.7 tokens/s** over the past week. Roughly half of that was configuration and half was code.

---

## Table of contents

- [Quick start](#quick-start)
- [How the split works](#how-the-split-works)
- [How a dense model runs](#how-a-dense-model-runs)
- [How an MoE model runs](#how-an-moe-model-runs)
- [Where the memory goes](#where-the-memory-goes)
- [Benchmarks](#benchmarks)
- [Flags that matter](#flags-that-matter)
- [Estimating `-ncmoe`](#estimating--ncmoe)
- [What changed this week](#what-changed-this-week)
- [Measuring things on this box](#measuring-things-on-this-box)
- [Known gaps](#known-gaps)

---

## Quick start

**Qwen3.8-Flash-Next (`qwen4exp`, 156 GiB) — best decode, 16 k context:**

```bat
set CUDA_VISIBLE_DEVICES=0,1,2,3
set GGML_CUDA_ALLREDUCE=nccl
set GGML_CUDA_P2P=1
set GGML_META_PARTIAL_COPY=1
set GGML_META_STAGE_SLOTS=4

llama-server -m Qwen3.8-Flash-Next-Uncensored-Q6_K-00001-of-00005.gguf ^
  -sm tensor -ncmoe 2 --ctx-size 16384 ^
  --load-mode none -lzm off ^
  -ub 4096 -b 4096 -np 1 --cache-ram -1 ^
  --flash-attn on --fit off -t 18
```

That gives about **39.7 t/s**. Note there is deliberately **no `-ts`** — see
[Flags that matter](#flags-that-matter).

**Same model at full 256 k context:** use `--ctx-size 262144 -ncmoe 12` and expect ~21.8 t/s. The KV
cache and the indexer cache eat about 23 GB, which is roughly ten layers' worth of residency, so
`-ncmoe` cannot go below 12 there.

**A model that fits entirely in VRAM:** drop `-ncmoe` and the two `GGML_META_*` variables. The rest of
the recipe still applies.

---

## How the split works

Upstream llama.cpp's `-sm layer` gives each GPU a contiguous *range of layers*. Only one GPU is busy at
a time, and a single layer must fit on a single card.

`-sm tensor` splits every weight matrix by **column** instead, so all four GPUs work on the same layer
simultaneously and then combine their partial results:

```
                 one linear layer:  weight [n_in, n_out]
                                    activations [n_in, n_tokens]
                                          │
        ┌─────────────────┬───────────────┼───────────────┬─────────────────┐
        ▼                 ▼               ▼               ▼                 
   ┌─────────┐       ┌─────────┐     ┌─────────┐     ┌─────────┐
   │  GPU 0  │       │  GPU 1  │     │  GPU 2  │     │  GPU 3  │
   │ cols    │       │ cols    │     │ cols    │     │ cols    │
   │ 0..¼    │       │ ¼..½    │     │ ½..¾    │     │ ¾..1    │
   └────┬────┘       └────┬────┘     └────┬────┘     └────┬────┘
        │                 │               │               │
        └─────────────────┴───────┬───────┴───────────────┘
                                  ▼
                    all-reduce  (NCCL, or a one-shot P2P
                    kernel for small messages)
                                  ▼
                    full activations, replicated
```

The "meta backend" (`ggml/src/ggml-backend-meta.cpp`) implements this. It presents itself to the
scheduler as one backend, and underneath it holds a real CUDA backend per device, a per-device *mirror*
of every tensor, and a table describing how each tensor is split — by column, by row, by expert, or
**mirrored** (replicated on every device) when a reduction would otherwise cross a split boundary.

Attention is a partial exception. Heads cannot be cut in half, so the split has to land on head
boundaries. On `qwen4exp` — 24 query heads, 2 KV heads, GQA 12 — the granularity works out such that
attention lands on **two of the four GPUs**, with the other two holding zero-width slices for those
tensors. That is correct, just not free; the remainder rotates per layer so the load evens out across
the model.

---

## How a dense model runs

Nothing leaves the GPUs. Every layer is column-split four ways, with an all-reduce after the
attention output and after the FFN down-projection:

```
  token ─▶ ┌──────────────────── repeated per layer ─────────────────────┐
           │                                                             │
           │  norm ─▶ Q/K/V (split by head) ─▶ attention ─▶ out-proj     │
           │                                        │                    │
           │                                   all-reduce                │
           │                                        ▼                    │
           │  norm ─▶ gate/up (split by column) ─▶ act ─▶ down           │
           │                                        │                    │
           │                                   all-reduce                │
           └─────────────────────────────────────────────────────────────┘
                                    ▼
                   lm_head (mirrored) ─▶ logits ─▶ sampler
```

`lm_head` is mirrored rather than split, so the logits come out replicated and every sampler primitive
(argmax, top-k, softmax — all reductions along the vocabulary axis) works unchanged. The cost is the
matrix being duplicated on each card and the GEMM being done four times. The intended replacement is an
all-gather at the logits boundary; the mirror is a deliberate stopgap.

---

## How an MoE model runs

An MoE layer replaces one FFN with `n_expert` of them plus a router. Only `n_expert_used` run per
token — for `qwen4exp`, 10 of 512. The experts are the overwhelming majority of the weights, which is
what makes offload viable: **most of the model is idle on any given token.**

### Everything resident

```
  ┌──────────────────────────── GPU 0..3 ───────────────────────────┐
  │  router (mirrored) ─▶ top-10 expert ids                         │
  │                             │                                   │
  │  each GPU holds ¼ of the columns of ALL 512 experts             │
  │                             ▼                                   │
  │  mul_mat_id: gathers the 10 routed experts, ¼ width each       │
  │                             │                                   │
  │                        all-reduce                               │
  └─────────────────────────────────────────────────────────────────┘
```

### With `-ncmoe N` — experts in system RAM

The first `N` MoE layers keep their expert weights in **pinned host memory**. Per token, only the
routed experts are copied to the GPUs:

```
        SYSTEM RAM (pinned)                              GPUs 0..3
  ┌────────────────────────────┐
  │ layers 0..N-1              │
  │   ffn_gate_exps  (512 exp) │   per token, 10 of 512 experts,
  │   ffn_up_exps    (512 exp) │   sliced ¼ per device
  │   ffn_down_exps  (512 exp) │ ───────────────────────────────▶ ┌──────────────┐
  └────────────────────────────┘        ~46 MB / token / layer    │ staging slot │
                                                                  └──────┬───────┘
  ┌────────────────────────────┐                                         │
  │ per_layer_token_embd       │   only the rows this token needs        ▼
  │   (50.66 GiB for qwen4exp) │ ──────────────────────────────▶  mul_mat_id
  └────────────────────────────┘                                         │
                                                                    all-reduce
        VRAM
  ┌────────────────────────────┐
  │ layers N..end : experts resident, never copied                      │
  │ all attention, norms, router, lm_head, KV cache                     │
  └─────────────────────────────────────────────────────────────────────┘
```

Two things make this fast enough to be worth doing:

**Only the routed experts move.** `GGML_META_PARTIAL_COPY=1` reads the router's expert ids back to the
host, works out which experts this ubatch actually needs, and copies only those — about 10 of 512 at
batch 1. It self-limits to batches ≤ 32, because by batch 4096 essentially every expert is live and
there is nothing left to save.

**Copies overlap compute.** Each device has a second CUDA stream and a small pool of *staging slots*.
While the compute stream works on layer L, the copy stream is filling a slot for layer L+1:

```
  copy stream    │ ██ slot0←L ██ │ ██ slot1←L+1 ██ │ ██ slot2←L+2 ██ │
                 │               ╲                 ╲
                 │      ready event                 ready event
                 ▼                ╲                 ╲
  compute stream │               │ ██ kernel reads slot0 ██ │ ██ reads slot1 ██ │
                                                    ╱
                                          release event → slot may be refilled
```

`GGML_META_STAGE_SLOTS` sets the pool size. Slots rotate, so a slot that held layer L's `gate` tensor
will later hold some other layer's `down` tensor — which turned out to matter a great deal; see
[the over-read bug](#the-over-read-bug).

### Who actually does the expert multiply

Worth being explicit, because "offload" can mean two different things and they perform very
differently:

```
  (a) CPU computes the experts        -ot  ...=CPU   (or no op-offload)

      activations ──▶ CPU ──▶ result          weights never move
                       │                      CPU reads ~50 MB/layer/token
                       │                      from DRAM at ~19 GB/s
      GPUs ░░░░░░░░ idle 89% ░░░░░░░░         ~2.2-2.6 ms per layer per token
                                              and it is serialized against GPU work


  (b) GPU computes them, weights stream in    <- what this branch does

      system RAM ──PCIe──▶ staging slot ──▶ GPU kernel
                                              ~46 MB/layer/token spread over 4 links
      GPUs ████████ busy ███████                ~2.0 ms per layer per token
                                              and it overlaps compute
```

Mode (a) was the starting point and it is genuinely CPU-bound: measured at ~19 GB/s against a
127 GB/s NUMA-local DRAM ceiling, with the GPUs 89 % idle and the CPU work fully exposed on the
critical path. Mode (b) is what `10812ed48` enabled — the matmul runs on the GPU and only the weights
travel — and it is what every number in this file refers to.

The CPU still does real work in mode (b): it runs the sampler, the scheduler, all the `cudaMemcpyAsync`
and graph launches, and — with `GGML_META_PARTIAL_COPY` — reads the router's expert ids back per layer
to decide what to copy. At `-ncmoe 2` the host is the busier side of the ledger, which is what the
decomposition cache addressed.

### Where the time actually goes

At `-ncmoe 12` the GPUs are **idle 84–93 % of the time**. Decode is bound by PCIe transfer, not
compute:

```
  per token, -ncmoe 12, 608 MB total moved
  dev0  ████████████████░░░░░░░░  149 MB @ 9.0 GB/s   16.6 ms
  dev1  ██████████████████░░░░░░  174 MB @ 9.8 GB/s   17.8 ms
  dev2  █████████░░░░░░░░░░░░░░░   91 MB @ 5.3 GB/s   17.6 ms   ← x8 link
  dev3  ████████████████████░░░░  192 MB @ 9.7 GB/s   19.8 ms   ← critical path
        kernels:  5–11 ms          copy engines: 17–20 ms
```

GPU 2 sits on an x8 link and runs at roughly half the bandwidth of the others. That is why `-ts` skew
looks attractive — and why it still loses, since it costs more in residency than it gains in balance.

---

## Where the memory goes

For `qwen4exp` (156 GiB of weights) on 128 GiB of VRAM, at `-c 16384`:

```
  per GPU, 32 768 MiB total
  ┌────────────────────────────────────────────────────────┬──────┬────┐
  │ model weights (¼ of the resident layers)   ~28 GB      │ KV   │ '  │
  └────────────────────────────────────────────────────────┴──────┴────┘
                                                    compute buf ┘   └ staging + spare

  SYSTEM RAM
  ┌──────────────────────────────────────────────────────────────────┐
  │ per_layer_token_embd  50.7 GiB   │ offloaded experts  ~4.5 GiB   │
  └──────────────────────────────────────────────────────────────────┘
     (pinned; needs -lzm off or it gets mmap-read on demand)
```

The per-layer embedding table is over a third of the file and never belongs in VRAM. With
`--load-mode none -lzm off` it lands in pinned host memory and only the rows a token needs are
gathered. **Without `-lzm off` it gets mmap-read on demand instead, which is 12.7× slower prefill off
a network share.**

Context is the real trade against residency:

| `-c` | VRAM/device | lowest `-ncmoe` that fits | decode |
|---|---|---|---|
| 16 384 | 25.9 GB | 2 | **30.0 t/s** |
| 262 144 | 31.8 GB | 12 | 18.5 t/s |

Full context costs ~23 GB — KV plus the mirrored indexer cache — which is about ten layers of
residency, or ~38 % of decode speed.

---

## Benchmarks

All on the 4× V100 box, from `llama-server` timings unless noted.

### `-ncmoe` is the dominant lever

`qwen4exp`, `-sm tensor`, even split, 4 slots, `-c 16384`:

| `-ncmoe` | decode | ms/token | VRAM/device |
|---|---|---|---|
| 12 | 18.63 | 53.7 | 25.9 / 25.9 / 26.4 / 25.4 GB |
| 8 | 21.72 | 46.0 | 27.6 / 28.5 / 28.5 / 27.5 GB |
| 4 | 26.90 | 37.2 | 30.2 / 30.2 / 30.6 / 29.6 GB |
| **2** | **30.02** | 33.3 | 30.9 / 31.3 / 31.3 / 30.8 GB |

Dead linear at **~2.0 ms/token per offloaded layer** — which is exactly the transfer budget of
46 MB/token/layer. Steps measured 1.93 / 2.20 / 1.95.

### Graph decomposition cache

The scheduler hands the meta backend the same handful of split graphs round-robin, and the meta backend
was re-deriving its per-device decomposition **every single call**. Caching it:

| `-ncmoe 2` | rebuild rate | enqueue (launch) | sync | decode |
|---|---|---|---|---|
| before | 1.37 ms, **100 %** | 2.43 ms (0.78) | 1.88 ms | 29.24 |
| after | 0.00 ms, **0 %** | 1.04 ms (0.77) | 1.91 ms | **41.44** |

Sustained over a 9 900-token run: **29.5 → 39.7 t/s**, and the output is byte-identical.

### Every MoE model on the box

`-ncmoe 4` (12 for deepseek4), `-c 4096`, two seeds each. **All ten comparisons byte-for-byte
identical to the previous commit.**

| model | arch | experts | before | after | gain |
|---|---|---|---|---|---|
| `bench-dsv4-q2k` | deepseek4 | 256/6 | 15.73 | **18.62** | +18 % |
| `test-gemma4moe` | gemma4 | 128/8 | 53.36 | **67.61** | +27 % |
| `test-ornith` | qwen35moe | 256/8 | 52.51 | **67.60** | +29 % |
| `test-qwen36-moe` | qwen35moe | 256/8 | 56.19 | **72.86** | +30 % |
| `test-qwen3next` | qwen3next | 512/10 | 44.07 | **59.55** | +35 % |

### Volta flash-attention retune

Head size 256 was spilling ~3 kiB/thread. `Q_in_reg=false` removes the spill, and 2 blocks × 128
threads beats 1 × 256 by ~17 % at identical warp count:

| | before | after |
|---|---|---|
| isolated kernel | 20 078 µs | 14 770 µs (**−26 %**) |
| 1 GPU, pp4096@d8192 | 551.2 | 592.5 (**+7.5 %**) |
| 4 GPU `-sm tensor` | 1 634.0 | 1 707.0 (**+4.5 %**) |

### Things that were measured and rejected

Recorded so nobody rebuilds them:

| idea | measurement | verdict |
|---|---|---|
| Prefetch the expert-id readback a layer ahead | 18.7 readbacks/token, but the sync costs **0.03 ms/token — 0.0 % of wall**, median 1.3 µs | worthless; ids are already resident |
| Skew `-ts` further toward the fast links | GPU 2 already gets 15.0 % against a 15.4 % bandwidth-fair share | already optimal |
| Skew `-ts` at all | balances transfer (38.2 → 19.8 ms) but device 1 binds sooner, so `-ncmoe` cannot drop as far; OOMs at `-c 262144` where an even split fits | **drop it** |
| Key staging slots to their owning tensor | correct, but 7.3 → 0.5 t/s — one slot serves all 18 layers of a kind, so the owner changes 18× per slot per token | dead end |
| Parallel per-device launch | launch time 3.2× better, but **+1.6 %** end to end — 93 % of the saving reappeared as `sync` | kept, but small |

---

## Flags that matter

| flag | why |
|---|---|
| `-sm tensor` | the split mode this branch adds; requires `--flash-attn on` |
| `-ncmoe N` | keep the first `N` MoE layers' experts in host RAM. **The single biggest lever.** Push it as low as VRAM allows |
| `-lzm off` | **mandatory.** Otherwise the 50.66 GiB embedding table is mmap-read on demand — 12.7× slower prefill off a network share |
| `--load-mode none` | mmap costs ~2.8× prefill even on local storage |
| `--fit off` | you are choosing `-ncmoe` yourself; auto-fit will fight you |
| `-ub 4096 -b 4096` | keeps prefill above the partial-copy batch gate, so prefill uses the cheaper full-layer staged copy |
| `-np 1` | at `-ncmoe 2` there is only ~1.1 GB/device spare; more slots will not fit |
| `-t 18` | 20 and 40 threads measure the same; 28 was 14 % worse |
| **no `-ts`** | skewing for link balance costs more residency than it gains. See the table above |

### Environment variables

| variable | default | what it does |
|---|---|---|
| `GGML_META_PARTIAL_COPY` | off | copy only the experts this ubatch routes to. Big decode win with `-ncmoe` |
| `GGML_META_PARTIAL_COPY_MAX_BATCH` | 32 | above this batch size the partial copy stops paying |
| `GGML_META_STAGE_SLOTS` | 4 | staging slots per device for offloaded weights |
| `GGML_META_DEC_CACHE` | 256 | graph decomposition cache entries. Raise if the eviction warning appears |
| `GGML_META_STAGE_TRACE` | 0 | log N staging events — slot handouts and split boundaries |
| `GGML_META_SERIAL_LAUNCH` | off | restore serial per-device launch, for A/B testing |
| `GGML_META_STATS` | 0 | print a host-timing breakdown every N graph computes |
| `GGML_CUDA_ALLREDUCE` | — | set to `nccl` at 4 GPUs; the internal all-reduce is 2-device only |
| `GGML_CUDA_P2P` | — | set to `1` |

`GGML_META_STATS` is the most useful of these when tuning. It reports, per call:
`enqueue (launch, comm, rebuild) / sync`. Bear in mind **`sync` is the host waiting on the GPU**, not
host work — mistaking it for work leads to chasing the wrong bottleneck.

See `docs/moe-offload.md` for a per-variable reference. Its `GGML_META_STAGE_SLOTS` section predates
the over-read fix, so read its slot-count advice alongside this file.

---

## Estimating `-ncmoe`

`scripts/estimate-ncmoe.py` reads the real per-layer expert-tensor sizes out of a GGUF — accounting for
quantization, `n_expert` and hidden dims properly rather than guessing — and prints a ready-to-run
command line.

```bash
python scripts/estimate-ncmoe.py model.gguf --vram-gib 32 --gpus 4 --ctx 32768

python scripts/estimate-ncmoe.py model.gguf --vram-gib 32 --gpus 4 \
    --binary llama-server --sm tensor --stage-slots 4 --partial-copy
```

Other options: `--kv-gib`, `--compute-buffer-gib`, `--headroom-gib`, `--ubatch`, `--host`, `--port`.

Two caveats worth knowing:

- Its `--stage-slots` default of **2** predates the over-read fix, when any slot count that was not a
  multiple of three destroyed the output. It is safe now, but pass `--stage-slots 4`.
- It does not model multi-GPU split imbalance, and this week showed that imbalance is decisive: the
  *binding* device sets how low `-ncmoe` can go, so a skewed `-ts` reduces your headroom.

Treat the result as a starting point and confirm against the `model buffer size` / `KV buffer size` /
`compute buffer size` lines llama.cpp prints at load.

---

## What changed this week

Commits on `sm-tensor-4xv100`, 2026-08-25 → 09-01.

### MoE expert offload

This is the bulk of the work. At the start of the week there was no usable offload path under
`-sm tensor`.

| commit | change |
|---|---|
| `b7cd84c18` | async copy fan-out across the 4 GPUs, one copy per device in flight |
| `6b79f057c` | expose the pinned host buffer type to `-ot` and `llama-bench` |
| `22c71e33a` | don't take `PARTIAL` tensors in `cpy_tensor_async` |
| `638e04063` … `98c849efb` | stage offloaded expert weights on a **second stream** so copies overlap compute; slot sizing matched to the buffer type; sync before resizing slots |
| `0941d440c` | opt-in **used-experts-only** copy (`GGML_META_PARTIAL_COPY`) |
| `8d7f74b3d` | fix an illegal memory access in it — a `buffer_clear` was wiping the whole scheduler allocator buffer rather than one tensor |
| `d8e36a868` | gate the partial copy on batch size; it only pays at decode |
| `10812ed48` | offload MoE matmuls regardless of batch size (**+39 … 47 % prefill**) |
| `5ca938ea1` | **correctness:** pad each staged range past the MMQ over-read |

#### The over-read bug

Worth understanding before touching this path, because the failure was silent and total.

The partial copy writes only the experts an ubatch routes to. But MMQ's `mul_mat_id` reads a little
**past** the rows it is handed. It multiplies those bytes by zero-padded activations, so finite garbage
is harmless — but a stale F16 block scale that decodes to `inf` is not, and `inf × 0 = NaN`.

Slots rotate, and one slot serves every layer's tensor of a given kind. So whether the bytes after a
range decoded finitely came down to **whether the previous occupant had the same quantization**. With
`GGML_META_STAGE_SLOTS` a multiple of three (the three expert tensors per layer), slot 0 always held a
`gate`, slot 1 an `up`, slot 2 a `down`, and the leftovers were at least the right type. Any other slot
count — **including the default of 4** — mixed Q6_K `gate` rows with Q8_0 `down` rows, produced `inf`
scales, and destroyed the output completely.

| slots | before the fix | after |
|---|---|---|
| 2 | every response is `/////…` | 18.9 pp / 7.3 tg |
| **4 (default)** | every response is `/////…` | 18.7 / 7.3 |
| 3, 6, 12 | correct | 18.8 / 6.8 |

The fix extends each staged range by enough whole chunks to cover the over-read with the tensor's own
data. It is free: the only tensor that needs it is split along axis 0, so a chunk is one 680-byte row.

**If you ever see generation collapse into one repeated character, check *which* character.** Token 0
in a Qwen tokenizer is `!`, and NaN logits make `argmax` return index 0 forever, because every
comparison against NaN is false. So repeated `!` means non-finite logits; any *other* repeated token
means finite-but-wrong weights.

### Host-side graph overhead

Once transfers were balanced, decode became limited by the single host thread issuing work.

| commit | change | effect |
|---|---|---|
| `558c2a039` | issue per-device subgraph launches in parallel | launch 3.2× faster, +1.6 % overall |
| `14bf760a2` | only restamp graph and split uids **when they change** | none alone — prerequisite |
| `7ce979783` | cache the graph decomposition per split graph | rebuild 100 % → 0 %, **+35 %** decode |
| `df2545f73` | size that cache for the whole split working set | fixes a silent 100 %-miss at `-ncmoe 12` |

The decomposition was rebuilt on every call for two independent reasons, and either one alone hides the
other. The scheduler restamped every graph and split uid from a global counter on every eval, so
nothing downstream could cache anything; and the meta backend compared against a **single** remembered
uid while being handed ~7 to ~37 different splits per token in rotation.

Two traps in the fix are worth recording:

**Making a decomposition reusable also makes its contents look static.** The rebuild used to mint a
fresh uid per per-device subgraph, which *incidentally* forced ggml-cuda to revalidate node properties
every eval. Staged expert weights have their data pointer rebound to a different slot each eval, so
once the uids were stable, ggml-cuda replayed a captured CUDA graph against **stale weight pointers**.
The give-away was launch time *dropping* 0.78 → 0.52 ms — work being skipped that must not be. A
throughput-only reading called it a 42 t/s win.

**A cache pool smaller than the working set is worse than no pool.** There are ~3 splits per offloaded
layer, so ~7 at `-ncmoe 2` but ~37 at `-ncmoe 12`. The scheduler hands them round-robin, which is the
worst case for LRU — every lookup misses:

| pool | rebuilds | decode at `-ncmoe 12` |
|---|---|---|
| 1 | 100 % | 18.74 |
| 16 (first default) | 100 % | 18.49 |
| 64 | 0 % | 21.30 |
| **256 (default now)** | **0 %** | **21.84** |

Entries are created on demand and sized to their own graph, so a high ceiling costs nothing unreached
(+80 MiB/device at `-ncmoe 12`). A warning now fires on first eviction, because a pool one entry short
has no symptom other than the rebuild rate.

### Volta kernels

| commit | change |
|---|---|
| `028c1a96b` | retune the FA MMA config for head size 256 — removes a ~3 kiB/thread register spill |
| `ba3088a16` | split DV accumulation for head size 512 |
| `eca0bd8b9` | fix a divergent `__syncthreads` in the FA MMA combine (undefined behaviour; `compute-sanitizer --tool synccheck` aborted the launch) |
| `6f8982646` | print the CUDA error before aborting |

8 warps/SM is a hard ceiling for `DV=256` on Volta — `VKQ_C` alone is 128 registers per thread — so
every legal config lands there and tuning is purely about tile shape.

### Correctness under `-sm tensor`

| commit | change |
|---|---|
| `c441eeee5` | don't read a split state from host-resident nodes (was an access violation) |
| `7b121bc7f` | keep gemma4 expert weights resident under `SPLIT_MODE_TENSOR` |
| `5057819e0` | let `CONCAT` expand the alloc size — upstream's new assert aborted on quantized CONCAT |

### Tests and other

`24d173dfb` vocab-scale `ARGMAX` perf cases · `51204c319` deepseek4-scale `FLASH_ATTN_EXT` cases ·
`15793de79` MLA-shaped `FLASH_ATTN_EXT` cases · `dca89e4a5` experimental draft-length bucketing for
`--spec-draft-adaptive`, off by default (variable draft length breaks CUDA graph reuse under
`-sm tensor`, costing ~2.5×).

Upstream merges: `78d65dc2d` master · `403e98723` b10679 · `d4052c0a8` b10705.

---

## Measuring things on this box

Each of these cost real time to learn.

**Check for orphaned servers before trusting any number.** `llama-cli` spawns `llama-server` as a child
process in this fork, and interrupting it can orphan one. An orphan holding 22 GiB per device silently
halved every figure — prefill 19 → 9 t/s, decode 7.3 → 0.5 — before eventually failing to allocate.
Confirm `nvidia-smi` reads ~10 MiB per device first.

**`nsys` on `llama-cli` captures nothing**, for the same reason: the CUDA work is all in the child.
Profile `llama-server` and drive it over HTTP, using `--delay` to skip the model load.

**Never measure speed in a run with `-lv 5`.** Its per-node logging throttles generation hard. It is
required for any `GGML_LOG_INFO` or `GGML_SCHED_DEBUG` output, so traces and timings need separate runs.

**Byte-identical output is the correctness bar** when only the code changed. Across different `-c`
values it is not available — KV padding changes flash-attention tiling and therefore rounding, so the
text diverges by design. Fall back to "no degeneration" there.

**`cmake --build` can exit 0 with compile errors.** Check the exe timestamp and grep the log.

**Let benchmarks warm up.** `--no-warmup` prevents CUDA graphs from being established, which made
decode look host-bound when it was not — a 2.7× difference in host enqueue time.

---

## Known gaps

- The `mirror_gen` guard in the decomposition cache has **never been observed firing**. It protects
  against a graph rebuilt into a reset `ggml_context` coming back identical — uid stable, per-device
  mirrors replaced. Reasoned about, not provoked.
- `test-dflash.gguf` fails to load on this branch *and* on the previous commit, so the `dflash`
  architecture is untested rather than passing.
- `-sm layer` at `-ncmoe 18 -c 262144` runs out of memory on device 2. It doesn't split weights by
  column, so it needs a considerably higher `-ncmoe`.
- `lm_head` is mirrored rather than all-gathered, which duplicates the matrix on every card and repeats
  the GEMM four times. The all-gather is the intended fix.
- The T4 in the machine cannot share a process with the V100s — `cudaSetDevice` on it fails whenever a
  V100 is visible.
