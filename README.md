# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

# llama.cpp on 4× V100 — tensor split and MoE expert offload

Notes for the `sm-tensor-4xv100` branch of this fork. Everything here targets one machine:
**4× Tesla V100-SXM2-32GB** (sm_70, 128 GiB VRAM total), Windows, CUDA 12.8.

The branch adds a *tensor-parallel* split mode (`-sm tensor`) on top of upstream llama.cpp, plus the
machinery to run **MoE models that are larger than VRAM** by keeping some expert layers in system RAM
and streaming them per token.

If you only read one thing: for a 156 GiB model on 128 GiB of VRAM, decode went from **15.7 to
42.2 tokens/s** over the past week. Roughly half of that was configuration and half was code.

> **Power note:** the four V100s are capped at **150 W each, against a 300 W default TDP** — halved
> deliberately to save power. Every number in this file is measured at half power, so treat them as a
> floor rather than as representative V100 performance.

---

## The machine

| | |
|---|---|
| GPUs | 4× Tesla V100-SXM2-32GB — sm_70, 32 768 MiB each, **128 GiB total** |
| GPU power | **150 W limit each (default and max are 300 W)** — deliberately lowered |
| PCIe | Gen 3. GPU 0/1/3 at **x16**, **GPU 2 at x8** — it runs at roughly half the bandwidth of the others, which shapes a lot of what follows |
| CPU | 2× Intel Xeon Gold 6230 @ 2.10 GHz — 20 cores per socket, **40 cores total**, no HT |
| System RAM | **382.6 GiB** |
| OS | Windows Server 2022 Standard (10.0.20348) |
| CUDA | 12.8, driver 573.96 |
| Also present | 1× Tesla T4 (60 W limit) — **cannot share a process with the V100s**; `cudaSetDevice` on it fails whenever a V100 is visible, so `--mmproj-device CUDA4` can never work |

Two consequences worth knowing before reading any benchmark here:

- **GPU 2's x8 link** measures 5.3 GB/s against 8.6–9.8 GB/s for the others. When weights stream from
  host RAM per token, it sets the critical path.
- **`-t 18`, not 40.** Two sockets means NUMA, and thread count is not monotonic: 20 and 40 threads
  measure the same, while 28 was 14 % *worse*.

---

## Table of contents

- [The machine](#the-machine)
- [Quick start](#quick-start)
- [How the split works](#how-the-split-works)
- [How a dense model runs](#how-a-dense-model-runs)
- [How an MoE model runs](#how-an-moe-model-runs)
- [Where the memory goes](#where-the-memory-goes)
- [Benchmarks](#benchmarks)
- [Flags that matter](#flags-that-matter)
- [Estimating `-ncmoe`](#estimating--ncmoe)
- [Split-tensor enablement work](#split-tensor-enablement-work)
- [What changed this week](#what-changed-this-week)
- [Upstream PRs evaluated](#upstream-prs-evaluated)
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

That gives about **42.2 t/s**. Note there is deliberately **no `-ts`** — see
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

### `-sm layer` vs `-sm tensor`, models fully on GPU

This is the core benefit of the split mode, with no expert offload involved — every model here fits
entirely in the 128 GiB of VRAM. `llama-bench -p 512 -n 128 -r 2 -fa on -lm none -lzm off -ub 2048`,
build `d8464c440` (upstream b10731), **at the 150 W power cap**. Re-measured after the merge: every
cell is within noise of the pre-merge run, so the merge's +6.3 % showed up in the offload path, not in
fully-resident models.

| model | arch | | size | pp `layer` | pp `tensor` | | tg `layer` | tg `tensor` | |
|---|---|---|---|---|---|---|---|---|---|
| glm4 9B Q8_0 | glm4 | dense | 9.3 GiB | 1187.9 | **2924.3** | +146 % | 67.6 | **123.0** | +82 % |
| qwen35 27B Q8_K_P | qwen35 | dense | 29.3 | 640.4 | **1717.6** | +168 % | 22.2 | **52.4** | +137 % |
| gemma4 31B Q8_0 | gemma4 | dense | 30.4 | 679.8 | **1621.2 ±322** | *noisy* | 20.6 | **46.7** | +127 % |
| muse-glimmer 30B F16 | muse-glimmer | dense | 51.9 | 1048.4 | **2395.6** | +128 % | 15.1 | **40.4** | +168 % |
| llama 70B Q8_0 | llama | dense | 69.8 | 302.5 | **950.5** | +214 % | 9.8 | **28.6** | +192 % |
| qwen35moe 35B-A3B Q8_0 | qwen35moe | MoE 256×8 | 34.4 | 1602.6 | **3143.7** | +96 % | 93.6 | **113.7** | +21 % |
| qwen3next 80B-A3B Q4_K_M | qwen3next | MoE 512×10 | 45.9 | 889.8 | **1647.0** | +85 % | 76.6 | **86.8** | +13 % |
| deepseek4 284B Q2_K | deepseek4 | MoE 256×6 | 90.9 | 188.8 | **616.1** | +226 % | 27.3 | **37.7** | +38 % |
| nemotron_h_moe 31B-A3.5B Q8_0 | nemotron_h_moe | MoE 128×6 | 32.6 | 1513.9 | *deny-listed* | — | 115.7 | *deny-listed* | — |

**Prefill gains everywhere — +85 % to +226 %.** Prompt processing is compute-bound, and four GPUs
working the same layer bring four times the FLOPs. The largest gain is on the largest model, where
`-sm layer` leaves three cards idle for most of each token.

**Decode splits sharply by model type.** Dense models gain **+80 % to +193 %**; MoE models only
**+12 % to +39 %**. Dense decode is bandwidth-bound on the *whole* weight set, so splitting it four
ways multiplies the bandwidth available. An MoE model at batch 1 only reads its active experts — 3 B
of 80 B for qwen3next — so each device was already reading little, and the all-reduce overhead is a
larger share of a smaller total. If you are choosing hardware for MoE decode specifically, that is the
number to look at.

`nemotron_h_moe` is on the deny-list in `llm_arch_supports_sm_tensor()`, with the other Mamba-2-style
hybrids: their in-projection fuses `z, x, B, C` and `dt` into one tensor, and `B`/`C` are shared by a
group of heads, so splitting by head would split them too. That needs a per-segment split plus enough
groups to distribute, which `ggml_ssm_scan` does not accept. `gemma4` works but is excluded from
*expert offload* specifically (`llm_arch_supports_sm_tensor_expert_offload`).

### Expert offload sweep — `-ncmoe 0 / 2 / 8 / 99`

`-sm tensor`, partial copy on, 4 slots, `llama-bench -p 512 -n 128 -r 2` (so a tiny context — KV is
negligible here, which is why `-ncmoe 0` fits at all). Two architectures with quite different expert
geometry:

| `-ncmoe` | deepseek4 256×6 Q2_K, 90.9 GiB | | qwen4exp 512×10 Q6_K, 156 GiB | |
|---|---|---|---|---|
| | pp512 | tg128 | pp512 | tg128 |
| **0** | 612.9 | **37.9** | 1153.2 | **50.8** |
| 2 | 539.2 | 32.6 | 874.3 | 42.5 |
| 8 | 395.1 | 23.1 | 471.9 | 27.9 |
| **99** (all layers) | 132.4 | **7.93** | 109.6 | **7.86** |

**The cost per offloaded layer is ~2.1 ms/token, and it barely depends on the architecture.** Converting
each step to ms/token per layer:

| step | deepseek4 | qwen4exp |
|---|---|---|
| 0 → 2 | 2.16 | 1.91 |
| 2 → 8 | 2.10 | 2.05 |
| 8 → 99 | 2.37 | 2.29 |

That holds across 256 experts top-6 at 2.06 bpw and 512 experts top-10 at Q6_K, and the slope only
creeps up at the extreme, where there is less resident compute left to hide the transfers behind. The
practical reading: **decide how many layers you can keep resident, and the decode cost follows
mechanically.**

**Both models converge to ~7.9 t/s with everything streaming** (7.93 and 7.86) — that is the floor set
by PCIe, not by the model.

**Prefill degrades far harder in relative terms:** qwen4exp `pp512` falls 10.5× from `-ncmoe 0` to 99,
deepseek4 4.6×. Prompt processing wants every expert at once, so there is nothing for the partial copy
to skip — it is gated off above batch 32 for exactly that reason.

**`-ncmoe 0` is reachable for a 156 GiB model on 128 GiB of VRAM**, which looks impossible until you
notice the 50.66 GiB per-layer embedding table lives in host RAM regardless — so only ~105 GiB needs
to fit. It only works at small context: an earlier sweep at `-c 16384` bottomed out at `-ncmoe 2`, and
at `-c 262144` the floor is 12. **The floor is a function of context, not of the model.**

### Offload sweep on the server — `-c 16384`, 1000-token generation

`llama-server`, `-sm tensor`, partial copy on, 4 slots, `-ub 4096 -b 4096`, one 1000-token request:

| `-ncmoe` | deepseek4 256×6 Q2_K | | | qwen4exp 512×10 Q6_K | | |
|---|---|---|---|---|---|---|
| | tg t/s | ms/tok | VRAM/GPU | tg t/s | ms/tok | VRAM/GPU |
| **2** | 31.28 | 31.97 | 27.5 GB | **41.85** | 23.89 | 31.3 GB |
| **8** | 22.51 | 44.43 | 24.9 GB | 27.42 | 36.47 | 28.5 GB |
| **16** | 16.39 | 61.01 | 21.5 GB | 17.98 | 55.62 | 23.8 GB |
| **99** (all) | 7.90 | 126.65 | 8.3 GB | 7.83 | 127.70 | 6.5 GB |

Per-layer cost from the ms/tok deltas — deepseek4 **2.08 / 2.07 / 2.43**, qwen4exp
**2.10 / 2.39 / 2.25** — so **~2.1 ms/token per offloaded layer** holds here too. That is now three
independent confirmations across two architectures, two harnesses and two context sizes.

Both models converge on **~7.9 t/s** with everything streaming. That is the PCIe floor of this box, not
a property of either model.

**Prefill is not measurable from this run.** The prompts are 40 and 89 tokens, so `prompt eval` is one
forward pass amortised over a handful of tokens — 47.99 down to 7.38 t/s for deepseek4, 67.00 down to
8.06 for qwen4exp. Those numbers are the per-pass cost of offloading (74–223 ms per offloaded layer,
30–100× the decode cost, which is why the partial copy is gated off above batch 32), **not** prefill
throughput. For that see the `pp512` figures in the all-GPU table above, or the 2000-token sweep
below, which was run specifically to close this gap.

VRAM is the real constraint on the low end: qwen4exp at `-ncmoe 2` sits at 31.3 GB of 32.0, which is why
2 is the floor at this context even though `-ncmoe 0` fits at a tiny one.

### Offload sweep with a real prompt — 2000-token prompt, 1000-token generation

The sweep above cannot report prefill: its prompts are 40 and 89 tokens, so `prompt eval` is one
forward pass amortised over a handful of tokens. This one sends an **exact 2000-token prompt** as raw
token ids, so PP is a genuine throughput figure, and it adds **TTFT** measured client-side — wall clock
from issuing the HTTP request to the first streamed content chunk.

Note this is the **Q4_K** build of deepseek4 (153 GiB), not the Q2_K used above. At Q4_K the model no
longer fits at any low `-ncmoe`, so its sweep starts at its floor.

`llama-server` build 10784 (`11f3772f1`), `-sm tensor -ngl 999`, `--load-mode none`,
`GGML_META_PARTIAL_COPY=1`, `GGML_META_STAGE_SLOTS=4`, `-ub 4096 -b 4096`, `--flash-attn on`,
`--fit off`, `-np 1`, `-c 16384`, even tensor split, greedy sampling, `cache_prompt: false`.

**deepseek4 Q4_K** — 43 MoE layers, 3.38 GiB of experts per layer:

| `-ncmoe` | PP ms/tok | PP tok/s | TG ms/tok | TG tok/s | TTFT | VRAM/GPU | RAM offloaded |
|---|---|---|---|---|---|---|---|
| **16** | 2.43 | 410.7 | 88.19 | 11.33 | 4.90 s | 30.9 GiB | 54.1 GiB |
| **20** | 2.44 | 410.2 | 102.41 | 9.76 | 4.90 s | 27.5 GiB | 67.6 GiB |
| **26** | 2.77 | 360.9 | 124.93 | 8.00 | 5.57 s | 22.5 GiB | 87.9 GiB |
| **99** (all 43) | 3.73 | 268.1 | 187.95 | 5.32 | 7.47 s | 8.2 GiB | 145.3 GiB |

**qwen4exp Q6_K** — 48 MoE layers, 2.11 GiB of experts per layer, plus 50.7 GiB of per-layer token
embeddings that are always host-resident:

| `-ncmoe` | PP ms/tok | PP tok/s | TG ms/tok | TG tok/s | TTFT | VRAM/GPU | RAM offloaded |
|---|---|---|---|---|---|---|---|
| **2** | 1.01 | 993.0 | 23.99 | 41.64 | 2.03 s | 30.6 GiB | 54.9 GiB |
| **8** | 1.29 | 775.3 | 36.58 | 27.31 | 2.59 s | 27.9 GiB | 67.5 GiB |
| **16** | 1.65 | 607.7 | 55.81 | 17.90 | 3.30 s | 23.2 GiB | 84.4 GiB |
| **99** (all 48) | 3.05 | 328.3 | 127.02 | 7.86 | 6.11 s | 6.5 GiB | 151.9 GiB |

**Decode cost per offloaded layer is set by bytes, not by architecture.** deepseek4 costs
**3.56 ms/token per layer** (88.19 → 187.95 across 27 layers), qwen4exp **2.24** (23.99 → 127.02 across
46). The ratio of those costs is 1.59; the ratio of their expert bytes per layer, 3.38 / 2.11 GiB, is
1.60. Decode under offload is purely transfer-bound — only the bytes crossing PCIe matter. This is a
sharper statement of the ~2.1 ms/layer figure above, which was measured on two models that happened to
have similar per-layer expert sizes.

**Prefill degrades far more gently than decode.** Over its full range deepseek4 loses 2.1× on decode
but only 1.5× on prefill; qwen4exp 5.3× against 3.0×. Offloading spends decode first, which is the
right trade for prompt-heavy work.

**deepseek4's first four layers past the floor are nearly free on prefill.** `-ncmoe` 16 → 20 moves
prefill 410.7 → 410.2 tok/s, inside noise, while decode drops 14 %. If you need 3.4 GiB/device back and
your workload is prompt-heavy, 20 costs almost nothing.

**The floor for deepseek4 Q4_K is `-ncmoe 16`**, and it is a hard one. `-ncmoe 8` aborts after 19 s on
a `cudaMalloc` failure; `-ncmoe 2` is arithmetically impossible — 146.6 GiB of resident weights against
127.4 GiB usable. At 16 the devices sit at 31.6 of 32.8 GiB.

Two caveats when reading the last column. For qwen4exp it is dominated by the always-host per-layer
embeddings, so at `-ncmoe 2` only 4.2 GiB of the 54.9 is actually experts — the two models' RAM columns
are not measuring the same thing. And TTFT is end-to-end client wall clock including HTTP, so it tracks
`PP total + one decode step` closely rather than exactly.

These runs use an **even** tensor split. `server-qwen38-flash.bat` ships `-ts 1.15,1.15,0.7,1.15`, which
OOMs at `-ncmoe` 2 and 8: it drives devices 1 and 3 to 27.3 GiB while device 2 idles at 16.5. That is
the same conclusion as the skew experiment in [Things that were measured and
rejected](#things-that-were-measured-and-rejected) — the binding device is what sets the floor.

### `-ncmoe` at `-c 16384` — the dominant lever



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

## Split-tensor enablement work

`-sm tensor` is the fork's reason for existing, and most of the work is not the split arithmetic
itself — it is making every architecture, collective and sampler survive being cut four ways. The
branch is **34 commits and ~2 850 inserted lines** ahead of upstream across 25 files, mostly in
`ggml/src/ggml-backend-meta.cpp` and `src/llama-model.cpp`.

### Making architectures work under the split

Each tensor needs a rule for *how* it splits, and getting one wrong is usually silent. The split-state
table in `src/llama-model.cpp` now handles:

| case | rule | why |
|---|---|---|
| standard attention | Q and KV split on the head axis, `attn_output` on the row axis | heads cannot be cut in half, so the granularity is the head size |
| **MLA** (`cb693151b`) | the latent KV, `q_a`, `kv_a_mqa` and their norms **mirrored**; only `q_b`, `k_b`, `v_b` and `attn_output` split | after absorption MLA is effectively MQA — every head shares one latent, so the latent cannot be split |
| **MQA / 1 KV head** (`1c707cf1b`) | mirror K, V and the KV cache; split only the Q heads | one KV head cannot be divided across four devices; mirroring costs N× KV cache but is the only correct option |
| MoE experts | split on the expert-inner axis, with the granularity taken from `ffn_down`'s quantization block so `gate`/`up`/`down` agree | mismatched boundaries would have each device's down-projection consume a slice it did not produce |
| gemma4 experts (`7b121bc7f`) | kept resident under `SPLIT_MODE_TENSOR` | — |
| `lm_head` | **mirrored** | logits then come out replicated and every sampler reduction (argmax, top-k, softmax) works unchanged |
| PLE / indexer caches | mirrored | the conv runs on every device, so each needs the whole history |
| host-resident nodes (`c441eeee5`) | no split state queried at all | previously an access violation |

### Collectives

| commit | change |
|---|---|
| `cb693151b` | a **one-shot P2P all-reduce** for small messages, alongside NCCL |
| `6406802e4` | make the all-reduce kernel block counts env-tunable (`GGML_CUDA_P2P_AR_NBLOCKS`) |

At 4 GPUs `GGML_CUDA_ALLREDUCE=nccl` is required — the internal all-reduce is 2-device only, and the
Windows default otherwise falls back to a slower meta butterfly. The P2P path handles small messages,
capped by `GGML_CUDA_P2P_AR_MAX_BYTES`.

### Attention kernels on Volta

`-sm tensor` shifts attention onto shapes upstream does not tune for on sm_70 — see the retune table
in [Benchmarks](#benchmarks). `028c1a96b` removed a ~3 kiB/thread register spill at head size 256,
`ba3088a16` split DV accumulation at head size 512, and `eca0bd8b9` fixed a divergent `__syncthreads`
in the MMA combine that `compute-sanitizer --tool synccheck` flagged as undefined behaviour.

### Test coverage added

`24d173dfb` vocab-scale `ARGMAX` cases · `51204c319` deepseek4-scale `FLASH_ATTN_EXT` cases ·
`15793de79` MLA-shaped `FLASH_ATTN_EXT` cases · `37d44070a` drive the MLA fixture off a single
`is_mla` predicate.

Worth knowing what this coverage does *not* prove: the synthetic architecture sweep is **F16-only**, so
a green sweep says nothing about MMQ, quantized padding or expert geometry. Both of this week's
corruption bugs were in quantized MoE paths that no test reaches.

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

Upstream merges: `78d65dc2d` master · `403e98723` b10679 · `d4052c0a8` b10705 · `d8464c440` **b10731**.

### b10731 merge (2026-09-01)

26 upstream commits, 8 touching files we own. Two conflicts, both resolved as unions:

- **`fattn-mma-f16.cuh`**, one hunk inside the `#else // Volta` branch. Upstream's XOR swizzle
  (#25635) routes `load_ldmatrix` through `ggml_cuda_fattn_smem_swizzle::`; ours is the
  `VKQ_C[(i_VKQ_0 - dvp*DV_acc)/i0_stride]` index from `ba3088a16`. Orthogonal, so upstream's call
  form plus our index. Safe on sm_70 because the wrapper's `if constexpr (swz)` **else-branch is
  literally the original call**. Taking upstream's side wholesale would have silently dropped the DV
  split and regressed deepseek4-scale FA.
- **`test-backend-ops.cpp`**, one hunk in the FLASH_ATTN_EXT perf list. Took upstream's wider loop
  (kv ≤ 65536, hs ≤ 576, nr ≤ 8, nb ∈ {1, 4096}) and kept all five of our explicit cases: upstream's
  `hs` list has no 512, and its 576 cases use `nh=8 / nr≤8` where ours are `nh=1 / nr=16` — a
  different GQA ratio, so not the subsumption it first looks like.

The meta backend, `llama-model.cpp` and `llama-arch.cpp` all auto-merged.

**Verified:** `test-backend-ops test -o FLASH_ATTN_EXT` passes; the `-ncmoe 2` hammer is
byte-identical to the pre-merge run with the decomposition cache still at 0.0 % rebuilds; and FA perf
at `hsk=256 kv=20000 nb=512 f16` is 14 902 µs against a 14 770 µs baseline — no regression, so the
swizzle plumbing costs nothing on sm_70 as expected (it needs cp.async/ldmatrix, neither of which
exists there).

**The merge is also a speedup: tg 39.7 → 42.2 t/s (+6.3 %).** Most likely `41ef91f7c`, which lifts the
1-token restriction on the MoE glu and topk-router fusion, and/or the qwen4exp indexer slice change
in `09412af38`. Not attributed further. Two upstream commits in this range land directly on our
production arch — `0eadefebd` qwen4exp recurrent state rollback and `09412af38` indexer slices — and
`6d1479c14` fixes the `ggml_backend_buft_get_alloc_size()` guard that our MMQ over-read padding reads,
which is why the hammer was the load-bearing check rather than a nicety.

---

## Upstream PRs evaluated

Branches in this repo named `pr-*` are **fetched upstream pull requests by other authors**, kept for
review and testing. They are not part of this fork's work and are not merged into
`sm-tensor-4xv100`. Two are worth recording because we measured them on this hardware.

### `pr-27016` — F16 activation scaling (kungfudaibi)

**A real bug on our exact hardware, but inert for every model we run.** `ggml_cuda_mul_mat_cublas_impl`
casts F32 `src1` to F16 with a plain convert, so values above the F16 max (65 504) become `inf` and
poison the GEMM. Reproduced precisely with `test-backend-ops test -o MUL_MAT -b CUDA0`:

| case | without the PR | with it |
|---|---|---|
| q8_0, n=**63**, b_max 1e5 | OK | OK |
| q8_0, n=**64**, b_max 1e5 | **FAIL** — `NaN at index 0` | OK |

The 63/64 boundary is where cuBLAS takes over from MMQ. But perplexity is **bit-identical** with and
without it for every model here — the scale resolves to 1.0 because our activations never approach
32 768. Exposure needs a model whose F32 activations exceed the F16 range at batch ≥ 64.

**Verdict: tracked, not carried.** It adds two kernels per F16 cuBLAS matmul for no benefit to us.

A larger effect turned up alongside it, unrelated to the PR: **gemma4 perplexity moves 3.4 % on
`GGML_CUDA_CUBLAS_COMPUTE_TYPE` alone** (1.7549 forced `f32` vs 1.8139 default under `-sm layer`). That
is plain F16 rounding, not overflow. If you ever chase a gemma4 numerics gap, sweep the compute type
before suspecting the split.

### `pr-26812` — ARGMAX split over multiple blocks (wjinxu)

**Already merged upstream and present in this branch** — it arrived via one of our upstream merges, as
`10939eedd` / `4bd613406` / `4cb4de7af` / `a1a7bc405`. Easy to miss, because the `origin/master` ref in
this repo is 236 commits stale, so diffing against it makes the change look absent.

It replaces one-block-per-row with a tiled kernel plus a combine pass, which is exactly the right fix
for a single long row. Measured here by reverting `argmax.cu` to the pre-PR version (`58062860a`) and
rebuilding `test-backend-ops`:

| shape | before | after | |
|---|---|---|---|
| `[32, 10]` | 2.04 µs | 2.05 µs | launch floor, unchanged |
| `[1024, 10]` | 2.39 µs | 2.40 µs | unchanged |
| `[32000, 512]` | 96.84 µs / 630 GB/s | 84.75 µs / **720 GB/s** | **1.14×** |
| `[129023, 3]` | 26.93 µs / 54 GB/s | 6.66 µs / **217 GB/s** | **4.04×** |
| `[151936, 8]` | 31.44 µs / 144 GB/s | 7.87 µs / **575 GB/s** | **3.99×** |
| **`[151936, 1]`** — vocab-wide, batch 1 | 31.38 µs / 18 GB/s | 6.99 µs / **81 GB/s** | **4.49×** |

**Verdict: it helps, substantially, on exactly the shape backend sampling uses** — 4.0–4.5× on
vocab-scale rows, and it does not regress the small shapes. `test-backend-ops test -o ARGMAX` passes.
Nothing to do; we already have it.

The 81 GB/s that remains on `[151936, 1]` is *not* headroom worth chasing. That row is 594 kB, and
6.99 µs is close to the launch floor visible in the `[32, 10]` row — there is simply not enough data in
one row to saturate the bus. The many-row cases reaching 575–720 GB/s show the kernel itself is fine.

Our contribution here was coverage, not the kernel: upstream's ARGMAX perf list stopped at 32 000
columns, so the single-row vocab-wide shape was never measured. `24d173dfb` added it.

In absolute terms this is still tiny for us — 6.99 µs once per token against a ~25 ms token is 0.03 %,
and it was 31 µs before, or 0.13 %. Worth knowing it is fixed rather than pending.

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
- **Fixed since:** `qwen4exp` at `-ncmoe 99` used to abort on a contiguity assert in
  `ggml_backend_meta_get_tensor_async`. The tensor was an **empty** ids tensor (`ne=[10,0,1,1]`,
  `nbytes=0`) and the read was for zero bytes; an empty tensor has no usable split state and a
  zero-element view reports non-contiguous, so the checks rejected it on the way to doing nothing.
  The abort was hiding an **unbounded out-of-bounds read** in `ggml_backend_sched_compute_splits`,
  whose used-experts scan assumed at least one bit was set in the bitset. Both fixed in `56cbbb32e`.
  The second one is not specific to `-ncmoe 99` or to the meta backend — any graph routing zero
  tokens to an MoE layer reaches it, so it is worth reporting upstream.
- `test-dflash.gguf` fails to load on this branch *and* on the previous commit, so the `dflash`
  architecture is untested rather than passing.
- `-sm layer` at `-ncmoe 18 -c 262144` runs out of memory on device 2. It doesn't split weights by
  column, so it needs a considerably higher `-ncmoe`.
- `lm_head` is mirrored rather than all-gathered, which duplicates the matrix on every card and repeats
  the GEMM four times. The all-gather is the intended fix.
- The T4 in the machine cannot share a process with the V100s — `cudaSetDevice` on it fails whenever a
  V100 is visible.
## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation
## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
