#!/usr/bin/env python3
"""Estimate a starting value for -ncmoe / --n-cpu-moe from a GGUF file and your VRAM.

Reads real per-layer expert-tensor sizes out of the GGUF (so it accounts for
quantization, n_expert, and hidden dims correctly instead of guessing), then
figures out how many of the largest-to-fit layers' worth of experts have to stay
on the CPU for the rest of the model + KV cache + compute buffer to fit in the
VRAM you give it.

The KV cache size is computed from the model's own per-layer head counts and
your --ctx, mirroring what llama_kv_cache allocates: n_embd_k_gqa/n_embd_v_gqa
per attention layer, K-only for MLA models, and per-sequence recurrent state for
the mamba layers of a hybrid. --kv-gib overrides it if you already have the exact
figure from a load log.

This is a starting point, not a substitute for reading the actual `model buffer
size` / `KV buffer size` / `compute buffer size` lines llama.cpp prints at load
time -- always confirm against those and adjust. In particular this script does
not attempt to model multi-GPU tensor-split imbalance (see docs/moe-offload.md
on why -ncmoe's "first N layers" isn't necessarily a balanced split).

Also prints a ready-to-run llama-server/llama-cli command line with the
suggested -ncmoe value plugged in, plus --load-mode none, (for -sm tensor)
GGML_META_STAGE_SLOTS / GGML_META_PARTIAL_COPY /
GGML_META_MOE_OFFLOAD_MIN_EXPERTS, and (for multi-GPU) GGML_CUDA_ALLREDUCE /
GGML_CUDA_P2P / GGML_CUDA_P2P_AR_MAX_BYTES, per the recipe in
docs/moe-offload.md.

GGML_META_MOE_OFFLOAD_MIN_EXPERTS is pinned to 0 for models with >= 64 experts:
the in-tree default of 64 makes the meta backend stream offloaded experts at
batch 1 as well, which measured 9-24x slower decode on nemotron_h_moe and
granitehybrid while leaving prefill unchanged.

Usage:
    python scripts/estimate-ncmoe.py model.gguf --vram-gib 32
    python scripts/estimate-ncmoe.py model.gguf --vram-gib 32 --gpus 4 --ctx 32768 \
        --compute-buffer-gib 2.5 --headroom-gib 1.0
    python scripts/estimate-ncmoe.py model.gguf --vram-gib 32 --gpus 4 \
        --binary llama-server --sm tensor --stage-slots 8 --partial-copy
    python scripts/estimate-ncmoe.py model.gguf --vram-gib 32 --gpus 4 \
        --partial-copy --allreduce internal --p2p --p2p-ar-max-bytes 1048576
"""

import argparse
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))

from gguf.gguf_reader import GGUFReader  # noqa: E402

EXPERT_TENSOR_SUFFIXES = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")

SPLIT_NAME_RE = re.compile(r"^(?P<prefix>.*)-(?P<part>\d+)-of-(?P<total>\d+)(?P<ext>\.gguf)$", re.IGNORECASE)


def get_field_int(reader, suffix):
    for key, field in reader.fields.items():
        if key.endswith(suffix) and field.parts:
            return int(field.parts[field.data[0]][0])
    return None


def get_field_int_list(reader, suffix, n):
    """Per-layer ints for a key that llama.cpp allows to be a scalar or a per-layer array."""
    for key, field in reader.fields.items():
        if key.endswith(suffix) and field.parts:
            vals = [int(field.parts[i][0]) for i in field.data]
            if len(vals) == 1:
                return [vals[0]] * n
            if len(vals) >= n:
                return vals[:n]
            return vals + [vals[-1]] * (n - len(vals))
    return [0] * n


def get_field_bool_list(reader, suffix, n):
    """Per-layer bools for a key stored as an array (e.g. attention.sliding_window_pattern)."""
    for key, field in reader.fields.items():
        if key.endswith(suffix) and field.parts:
            vals = [bool(field.parts[i][0]) for i in field.data]
            if len(vals) >= n:
                return vals[:n]
            if vals:
                return vals + [vals[-1]] * (n - len(vals))
    return None


def load_shards(path):
    """Return (metadata_reader, list_of_tensors) across all shards of a split GGUF.

    A split GGUF's tensors are spread across N files (`*-00001-of-000NN.gguf`, ...);
    only the first shard carries the full key-value metadata, so callers should read
    metadata (block_count, expert_count, ...) from the returned reader, but iterate
    tensors from the returned list, not from `metadata_reader.tensors` alone.
    """
    reader = GGUFReader(str(path))
    split_count = get_field_int(reader, "split.count")

    if not split_count or split_count <= 1:
        return reader, list(reader.tensors)

    match = SPLIT_NAME_RE.match(path.name)
    if not match:
        print(f"warning: split.count={split_count} but filename doesn't match the "
              f"'*-NNNNN-of-NNNNN.gguf' pattern -- reading only this shard, tensor totals will be short",
              file=sys.stderr)
        return reader, list(reader.tensors)

    prefix, part_width, total, ext = match["prefix"], len(match["part"]), int(match["total"]), match["ext"]
    if total != split_count:
        print(f"warning: split.count={split_count} but filename says {total} shards -- using the filename",
              file=sys.stderr)

    all_tensors = []
    missing = []
    this_part = int(match["part"])
    for part in range(1, total + 1):
        shard_path = path.with_name(f"{prefix}-{part:0{part_width}d}-of-{total:0{part_width}d}{ext}")
        if part == this_part:
            all_tensors.extend(reader.tensors)
            continue
        if not shard_path.exists():
            missing.append(shard_path.name)
            continue
        all_tensors.extend(GGUFReader(str(shard_path)).tensors)

    if missing:
        print(f"warning: {len(missing)}/{total} shard file(s) not found next to {path.name} "
              f"({', '.join(missing)}) -- expert/model sizes below are undercounted", file=sys.stderr)

    return reader, all_tensors


def human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} PiB"


def layer_index(name: str):
    # names look like "blk.<N>.ffn_gate_exps.weight"
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "blk":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


# f16 is the default kv cache type; the quantized ones are close enough to their
# bytes-per-weight for a VRAM estimate.
KV_TYPE_BYTES = {"f16": 2.0, "bf16": 2.0, "f32": 4.0, "q8_0": 34.0/32, "q5_1": 24.0/32, "q4_0": 18.0/32}


# These use their own cache class (llama_kv_cache_dsv4) with fixed-size windows plus a separate
# compressed CSA cache, so the size does not follow from the generic head counts and does not scale
# with -c at all. Measured: 768- and 512-cell caches, unchanged between ctx 2048 and 16384.
ARCH_OPAQUE_KV = {"deepseek4", "dflash"}


def estimate_kv_bytes(reader, n_layers, n_ctx, n_seq, type_k, type_v, n_ubatch):
    """Per-layer KV cache size, mirroring llama_kv_cache's allocation.

    llama-kv-cache.cpp allocates n_embd_k_gqa(il) x kv_size for K and n_embd_v_gqa(il) x kv_size
    for V per layer, with has_v = !is_mla, and hparams.n_embd_k_gqa = n_embd_head_k * n_head_kv.
    Layers with no kv heads are recurrent instead: they hold n_embd_r + n_embd_s per *sequence*,
    not per token.

    Returns (bytes, n_attn_layers, n_recr_layers, notes), or (None, None, None, notes) for an
    architecture whose cache cannot be derived from the generic metadata. Validated against the
    llama_kv_cache / llama_memory_recurrent lines of real load logs -- exact on nemotron_h_moe
    (hybrid + MTP), qwen35moe and qwen3next (linear-attention stride), gemma4 (ISWA) and the
    120B nemotron_h_moe.
    """
    arch_field = reader.get_field("general.architecture")
    arch = arch_field.contents() if arch_field is not None else None
    if arch in ARCH_OPAQUE_KV:
        return None, None, None, [
            f"{arch} uses its own KV cache (fixed windows + a compressed CSA cache) that does not "
            "follow from the generic head counts, and does not scale with -c. Pass --kv-gib with "
            "the figure from a load log; treating it as 0 here."]

    n_head    = get_field_int(reader, "attention.head_count") or 0
    n_embd    = get_field_int(reader, "embedding_length") or 0
    head_k    = get_field_int(reader, "attention.key_length")
    head_v    = get_field_int(reader, "attention.value_length")
    if head_k is None:
        head_k = (n_embd // n_head) if n_head else 0
    if head_v is None:
        head_v = head_k

    kv_lora   = get_field_int(reader, "attention.kv_lora_rank")
    n_rot     = get_field_int(reader, "rope.dimension_count") or 0
    is_mla    = kv_lora is not None and kv_lora > 0

    # Interleaved sliding-window attention (Gemma-style): the SWA layers get their own, much
    # shorter cache, and may also use different head dims. Measured on test-gemma4moe: 5 full
    # layers at 4096 cells plus 25 SWA layers at swa+n_ubatch = 1536 cells, not 30 x 4096.
    swa_window = get_field_int(reader, "attention.sliding_window") or 0
    swa_pattern = get_field_bool_list(reader, "attention.sliding_window_pattern", n_layers)
    head_k_swa = get_field_int(reader, "attention.key_length_swa") or head_k
    head_v_swa = get_field_int(reader, "attention.value_length_swa") or head_v
    n_ctx_swa = min(n_ctx, swa_window + n_ubatch) if swa_window else n_ctx

    # Linear-attention hybrids (Qwen 3 Next / 3.5) mark the recurrent layers with a stride instead
    # of n_head_kv == 0: hparams::set_recr_pattern makes layer il recurrent when
    # il % interval < interval - 1. Measured on test-ornith: interval 4 over 40 layers = 10 attn.
    attn_interval = get_field_int(reader, "full_attention_interval") or 0

    # A hybrid marks its mamba layers with n_head_kv == 0, but so are its FFN-only layers: the
    # is_recr rule these models use is (n_head_kv == 0 && n_ff == 0), so both arrays are needed or
    # the recurrent state gets counted twice over (measured: 46 layers guessed vs 23 allocated).
    head_kv_per_layer = get_field_int_list(reader, "attention.head_count_kv", n_layers)
    n_ff_per_layer    = get_field_int_list(reader, "feed_forward_length", n_layers)

    # Trailing NextN/MTP blocks are not part of the trunk: llama.cpp reports their weights as
    # "unused tensor blk.N.*" and the kv cache filters them out.
    n_mtp = get_field_int(reader, "nextn_predict_layers") or 0
    n_trunk = max(0, n_layers - n_mtp)

    bytes_k = KV_TYPE_BYTES.get(type_k, 2.0)
    bytes_v = KV_TYPE_BYTES.get(type_v, 2.0)

    # recurrent state, for the hybrid layers that have no kv heads
    d_conv  = get_field_int(reader, "ssm.conv_kernel") or 0
    d_inner = get_field_int(reader, "ssm.inner_size") or 0
    d_state = get_field_int(reader, "ssm.state_size") or 0
    n_group = get_field_int(reader, "ssm.group_count") or 0
    n_embd_r = (d_conv - 1) * (d_inner + 2 * n_group * d_state) if d_conv > 0 else 0
    n_embd_s = d_state * d_inner

    total = 0.0
    n_attn = n_recr = n_swa = 0
    for il in range(n_trunk):
        n_head_kv = head_kv_per_layer[il]
        if n_head_kv == 0:
            if n_ff_per_layer[il] == 0:
                # recurrent layer: state is per sequence and f32, not per token
                total += (n_embd_r + n_embd_s) * n_seq * 4.0
                n_recr += 1
            # else: an FFN-only layer, which holds no cache of either kind
            continue
        if attn_interval and (il % attn_interval) < (attn_interval - 1):
            # linear attention layer: recurrent state, no per-token KV
            total += (n_embd_r + n_embd_s) * n_seq * 4.0
            n_recr += 1
            continue
        n_attn += 1
        if is_mla:
            # only the latent is cached, and V is absorbed (has_v = !is_mla)
            total += (kv_lora + n_rot) * n_ctx * bytes_k
        elif swa_pattern is not None and swa_pattern[il]:
            n_swa += 1
            total += head_k_swa * n_head_kv * n_ctx_swa * bytes_k
            total += head_v_swa * n_head_kv * n_ctx_swa * bytes_v
        else:
            total += head_k * n_head_kv * n_ctx * bytes_k
            total += head_v * n_head_kv * n_ctx * bytes_v

    notes = []
    if n_mtp:
        notes.append(f"skipping {n_mtp} trailing NextN/MTP block(s), which hold no cache")
    if is_mla:
        notes.append(f"MLA: caching a {kv_lora}+{n_rot} latent per token, no separate V")
    if n_recr:
        notes.append(f"{n_recr} recurrent layer(s) hold per-sequence state, not per-token KV")
    if swa_window:
        if swa_pattern is not None:
            notes.append(f"{n_swa} of {n_attn} attn layer(s) are sliding-window, sized at "
                         f"{n_ctx_swa} cells (window {swa_window} + ubatch {n_ubatch})")
        else:
            notes.append(f"model has sliding_window={swa_window} but no per-layer pattern in the "
                         "metadata; assuming full ctx everywhere, so the real figure is LOWER")
    if attn_interval:
        notes.append(f"full_attention_interval={attn_interval}: the other layers are linear "
                     "attention and hold per-sequence state")
    return total, n_attn, n_recr, notes


def print_command(args, ncmoe, offloading_experts, cmoe_all=False, n_expert=None):
    """Print a ready-to-run command line, per the recipe in docs/moe-offload.md."""
    split_mode = args.split_mode or ("tensor" if args.gpus > 1 else None)

    env = {}
    if split_mode == "tensor" and offloading_experts:
        env["GGML_META_STAGE_SLOTS"] = str(args.stage_slots if args.stage_slots is not None else 4)
        if args.partial_copy:
            env["GGML_META_PARTIAL_COPY"] = "1"

        # ggml_backend_meta_moe_offload_always() streams the offloaded expert matmul at *every*
        # batch size once n_expert >= GGML_META_MOE_OFFLOAD_MIN_EXPERTS (default 64). That gate is
        # a proxy for "bytes per token are small", and it mispredicts badly for models with many
        # large experts: measured on 4x V100, decode went 14.16 -> 1.58 t/s on granitehybrid
        # (72 experts) and 36.5 -> 1.3 t/s on nemotron_h_moe (128 experts). Setting it to 0 recovers
        # decode fully and costs nothing on prefill, because at prefill batch sizes the plain CUDA
        # offload threshold (~32 tokens) already streams them anyway.
        if args.moe_offload_min_experts is not None:
            env["GGML_META_MOE_OFFLOAD_MIN_EXPERTS"] = str(args.moe_offload_min_experts)
        elif n_expert is not None and n_expert >= 64:
            env["GGML_META_MOE_OFFLOAD_MIN_EXPERTS"] = "0"
            print(f"(note: n_expert={n_expert} >= 64 would make the meta backend stream offloaded "
                  "experts at batch 1 too, which costs ~9-24x decode; pinning "
                  "GGML_META_MOE_OFFLOAD_MIN_EXPERTS=0 below. Prefill is unaffected.)")
    elif args.stage_slots is not None or args.partial_copy:
        print("(note: --stage-slots / --partial-copy only apply under -sm tensor with offloaded "
              "experts; omitted from the command below)")

    # How the devices talk to each other, so these apply to any multi-GPU run, with or without
    # offloaded experts.
    if args.gpus > 1:
        if args.allreduce is not None:
            env["GGML_CUDA_ALLREDUCE"] = args.allreduce
        if args.p2p:
            env["GGML_CUDA_P2P"] = "1"
        if args.p2p_ar_max_bytes is not None:
            env["GGML_CUDA_P2P_AR_MAX_BYTES"] = str(args.p2p_ar_max_bytes)
    elif args.allreduce is not None or args.p2p or args.p2p_ar_max_bytes is not None:
        print("(note: --allreduce / --p2p / --p2p-ar-max-bytes need more than one GPU; omitted "
              "from the command below)")

    argv = [args.binary, "-m", str(args.gguf_path)]
    if split_mode:
        argv += ["-sm", split_mode]
        if split_mode == "tensor":
            # common/fit.cpp: common_params_fit_impl() throws immediately for
            # LLAMA_SPLIT_MODE_TENSOR, so --fit (on by default) always fails with a
            # "not implemented for SPLIT_MODE_TENSOR" warning and never adjusts
            # anything -- turn it off to skip the pointless attempt, and size
            # -ngl/-c yourself (see docs/multi-gpu.md's "CUDA OOM" row).
            argv += ["--fit", "off"]
    if cmoe_all:
        argv += ["-cmoe", "--load-mode", "none"]
    elif ncmoe is not None:
        argv += ["-ncmoe", str(ncmoe)]
        argv += ["--load-mode", "none"]
    if args.ctx is not None:
        argv += ["-c", str(args.ctx)]
    argv += ["-ub", str(args.ubatch)]
    if args.binary == "llama-server":
        if args.host is not None:
            argv += ["--host", args.host]
        if args.port is not None:
            argv += ["--port", str(args.port)]

    env_prefix = "".join(f"{k}={v} " for k, v in env.items())
    print("Command:")
    print(f"  {env_prefix}{shlex.join(argv)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf_path", type=Path)
    ap.add_argument("--vram-gib", type=float, required=True,
                     help="usable VRAM per GPU, in GiB (before your own headroom margin)")
    ap.add_argument("--gpus", type=int, default=4,
                     help="number of GPUs the model+KV+compute buffer is split across (default 1)")
    ap.add_argument("--kv-gib", type=float, default=None,
                     help="override the computed KV cache size, in GiB (e.g. the exact figure from "
                          "a prior load log). By default the KV size is computed from the model's "
                          "own head counts and --ctx, so you should not normally need this.")
    ap.add_argument("--cache-type-k", choices=tuple(KV_TYPE_BYTES), default="f16",
                     help="-ctk value, used to size the K cache (default f16)")
    ap.add_argument("--cache-type-v", choices=tuple(KV_TYPE_BYTES), default="f16",
                     help="-ctv value, used to size the V cache (default f16)")
    ap.add_argument("--parallel", type=int, default=1,
                     help="-np/--parallel value: number of sequences. Scales the recurrent state of "
                          "hybrid models, and is what llama-server multiplies its KV cache by "
                          "(default 1)")
    ap.add_argument("--compute-buffer-gib", type=float, default=2.0,
                     help="expected compute buffer size in GiB, total across GPUs (default 2.0; "
                          "read the actual value from a load log once you have one)")
    ap.add_argument("--headroom-gib", type=float, default=1.0,
                     help="extra safety margin to leave free per GPU, in GiB (default 1.0)")
    ap.add_argument("--binary", choices=("llama-server", "llama-cli"), default="llama-server",
                     help="which binary to print a command line for (default llama-server)")
    ap.add_argument("--sm", "--split-mode", dest="split_mode",
                     choices=("none", "layer", "row", "tensor"), default="tensor",
                     help="-sm value to include in the printed command (default: 'tensor' if "
                          "--gpus > 1, otherwise omitted)")
    ap.add_argument("--ctx", type=int, default=262144, help="-c/--ctx-size value to include in the printed command")
    ap.add_argument("--ubatch", type=int, default=4096,
                     help="-ub/--ubatch-size value to include in the printed command (default 4096, "
                          "see docs/moe-offload.md -- larger ubatch generally helps prefill throughput "
                          "with CPU-resident experts)")
    ap.add_argument("--stage-slots", type=int, default=4,
                     help="GGML_META_STAGE_SLOTS to include (only meaningful with -sm tensor and "
                          "offloaded experts; default 4, omitted otherwise)")
    ap.add_argument("--moe-offload-min-experts", type=int, default=None,
                     help="GGML_META_MOE_OFFLOAD_MIN_EXPERTS to include (only meaningful with "
                          "-sm tensor and offloaded experts). Omit to let the script decide: it "
                          "pins 0 when the model has >= 64 experts, which is where the in-tree "
                          "default of 64 starts costing an order of magnitude of decode speed. "
                          "Pass 64 to keep the in-tree behaviour")
    ap.add_argument("--partial-copy", action="store_true",
                     help="include GGML_META_PARTIAL_COPY=1 in the printed command "
                          "(only worth it for decode-heavy/low-batch workloads, see docs/moe-offload.md)")
    ap.add_argument("--allreduce", choices=("nccl", "internal", "none"), default=None,
                     help="GGML_CUDA_ALLREDUCE to include (needs more than one GPU). Omit to leave "
                          "the platform default: nccl on Linux, internal elsewhere")
    ap.add_argument("--p2p", action="store_true",
                     help="include GGML_CUDA_P2P=1 (needs more than one GPU) to enable peer access; "
                          "NCCL builds already enable it implicitly")
    ap.add_argument("--p2p-ar-max-bytes", type=int, default=None,
                     help="GGML_CUDA_P2P_AR_MAX_BYTES to include (needs more than one GPU): size "
                          "ceiling for the one-shot P2P allreduce, which beats a ring on small "
                          "latency-bound messages. In-tree default is 1048576; 0 disables it")
    ap.add_argument("--host", default=None, help="--host value to include for llama-server")
    ap.add_argument("--port", type=int, default=None, help="--port value to include for llama-server")
    args = ap.parse_args()

    reader, tensors = load_shards(args.gguf_path)

    expert_bytes_by_layer = {}
    non_expert_bytes = 0
    n_expert = get_field_int(reader, "expert_count")
    n_expert_used = get_field_int(reader, "expert_used_count")
    block_count = get_field_int(reader, "block_count")

    for tensor in tensors:
        nbytes = tensor.n_bytes
        idx = layer_index(tensor.name)
        if idx is not None and any(s in tensor.name for s in EXPERT_TENSOR_SUFFIXES):
            expert_bytes_by_layer[idx] = expert_bytes_by_layer.get(idx, 0) + nbytes
        else:
            non_expert_bytes += nbytes

    if not expert_bytes_by_layer:
        print("No blk.<N>.ffn_{gate,up,down}_exps tensors found -- this doesn't look like a MoE GGUF.",
              file=sys.stderr)
        sys.exit(1)

    n_layers = block_count or (max(expert_bytes_by_layer) + 1)
    total_expert_bytes = sum(expert_bytes_by_layer.values())
    total_model_bytes = total_expert_bytes + non_expert_bytes

    print(f"Model: {args.gguf_path.name}")
    print(f"  layers (block_count):      {n_layers}")
    if n_expert is not None:
        print(f"  n_expert / n_expert_used:  {n_expert} / {n_expert_used}")
    print(f"  non-expert weights:        {human(non_expert_bytes)}")
    print(f"  expert weights (total):    {human(total_expert_bytes)}")
    print(f"  model total:               {human(total_model_bytes)}")
    print()

    vram_total = args.vram_gib * args.gpus * (1024 ** 3)

    if args.kv_gib is not None:
        kv_bytes = args.kv_gib * (1024 ** 3)
        kv_note = ["overridden by --kv-gib"]
        kv_attn = kv_recr = None
    else:
        kv_bytes, kv_attn, kv_recr, kv_note = estimate_kv_bytes(
            reader, n_layers, args.ctx, args.parallel, args.cache_type_k, args.cache_type_v,
            args.ubatch)
        if kv_bytes is None:
            kv_bytes = 0.0
    compute_bytes = args.compute_buffer_gib * (1024 ** 3)
    headroom_bytes = args.headroom_gib * args.gpus * (1024 ** 3)

    budget_for_weights = vram_total - kv_bytes - compute_bytes - headroom_bytes

    print(f"VRAM budget: {args.gpus} x {args.vram_gib:.2f} GiB = {human(vram_total)}")
    if kv_attn is not None:
        print(f"  - KV cache:              {human(kv_bytes)}  "
              f"(ctx {args.ctx}, {kv_attn} attn layer(s), {args.cache_type_k}/{args.cache_type_v})")
    else:
        print(f"  - KV cache:              {human(kv_bytes)}")
    for note in kv_note:
        print(f"      note: {note}")
    print(f"  - compute buffer:        {human(compute_bytes)}")
    print(f"  - headroom:              {human(headroom_bytes)}")
    print(f"  = budget for weights:    {human(budget_for_weights)}")
    print()

    if budget_for_weights <= 0:
        print("VRAM budget is already exhausted by KV + compute buffer + headroom alone -- "
              "every expert layer (and possibly non-expert weights too) needs to be offloaded. "
              "Consider -cmoe / --cpu-moe, a smaller context, or fewer GPUs' worth of KV replication.")
        print()
        print_command(args, ncmoe=None, offloading_experts=True, cmoe_all=True, n_expert=n_expert)
        sys.exit(0)

    if budget_for_weights >= total_model_bytes:
        print("The full model fits in the given VRAM budget -- you likely don't need -ncmoe at all. "
              "(Still confirm against the actual load log: this ignores allocator fragmentation "
              "and multi-GPU tensor-split imbalance.)")
        print()
        print_command(args, ncmoe=None, offloading_experts=False, n_expert=n_expert)
        sys.exit(0)

    remaining = budget_for_weights - non_expert_bytes
    if remaining <= 0:
        print("Non-expert weights alone exceed the weight budget -- -ncmoe cannot make this model fit "
              "as-is. You need more VRAM, more GPUs, or a smaller/more-quantized model.")
        sys.exit(1)

    # Offload the largest-index layers' experts first (matches -ncmoe's "first N stay on CPU"
    # semantics only in that we report a COUNT; -ncmoe takes the count from the front, this
    # loop is just finding how many layers' worth of experts must be evicted in total).
    sorted_layers = sorted(expert_bytes_by_layer.items())
    kept_expert_bytes = 0
    n_offload = 0
    for idx, nbytes in reversed(sorted_layers):
        if kept_expert_bytes + nbytes > remaining:
            n_offload += 1
        else:
            kept_expert_bytes += nbytes
    # n_offload counts how many layers didn't fit; -ncmoe wants that many layers offloaded.
    n_offload = min(n_offload, n_layers)

    on_gpu_expert_bytes = kept_expert_bytes

    print(f"Suggested starting point:  -ncmoe {n_offload}   (of {n_layers} layers)")
    print(f"  experts kept on GPU:     {human(on_gpu_expert_bytes)}")
    print(f"  experts offloaded to CPU:{human(total_expert_bytes - on_gpu_expert_bytes)}")
    print()
    print("Remember: -ncmoe strips layers from the front, which is not guaranteed to match this")
    print("script's largest-fit selection or to balance evenly across multiple GPUs under -sm")
    print("tensor/-sm row. Re-check against the real load log and adjust.")
    print()
    print_command(args, ncmoe=n_offload, offloading_experts=True, n_expert=n_expert)


if __name__ == "__main__":
    main()
