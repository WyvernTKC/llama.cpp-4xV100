#!/usr/bin/env python3
"""Estimate a starting value for -ncmoe / --n-cpu-moe from a GGUF file and your VRAM.

Reads real per-layer expert-tensor sizes out of the GGUF (so it accounts for
quantization, n_expert, and hidden dims correctly instead of guessing), then
figures out how many of the largest-to-fit layers' worth of experts have to stay
on the CPU for the rest of the model + KV cache + compute buffer to fit in the
VRAM you give it.

This is a starting point, not a substitute for reading the actual `model buffer
size` / `KV buffer size` / `compute buffer size` lines llama.cpp prints at load
time -- always confirm against those and adjust. In particular this script does
not attempt to model multi-GPU tensor-split imbalance (see docs/moe-offload.md
on why -ncmoe's "first N layers" isn't necessarily a balanced split).

Also prints a ready-to-run llama-server/llama-cli command line with the
suggested -ncmoe value plugged in, plus --load-mode none and (for -sm tensor)
GGML_META_STAGE_SLOTS / GGML_META_PARTIAL_COPY, per the recipe in
docs/moe-offload.md.

Usage:
    python scripts/estimate-ncmoe.py model.gguf --vram-gib 32
    python scripts/estimate-ncmoe.py model.gguf --vram-gib 32 --gpus 4 --ctx 32768 \
        --compute-buffer-gib 2.5 --headroom-gib 1.0
    python scripts/estimate-ncmoe.py model.gguf --vram-gib 32 --gpus 4 \
        --binary llama-server --sm tensor --stage-slots 8 --partial-copy
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


def print_command(args, ncmoe, offloading_experts, cmoe_all=False):
    """Print a ready-to-run command line, per the recipe in docs/moe-offload.md."""
    split_mode = args.split_mode or ("tensor" if args.gpus > 1 else None)

    env = {}
    if split_mode == "tensor" and offloading_experts:
        env["GGML_META_STAGE_SLOTS"] = str(args.stage_slots if args.stage_slots is not None else 4)
        if args.partial_copy:
            env["GGML_META_PARTIAL_COPY"] = "1"
    elif args.stage_slots is not None or args.partial_copy:
        print("(note: --stage-slots / --partial-copy only apply under -sm tensor with offloaded "
              "experts; omitted from the command below)")

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
    ap.add_argument("--kv-gib", type=float, default=2.0,
                     help="expected total KV cache size in GiB, if known "
                          "(e.g. from a prior load log). Omit to skip accounting for it "
                          "-- do this for MLA/GQA-with-small-KV models where it barely matters, "
                          "and always double check against the load log for MHA models.")
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
                     help="-ub/--ubatch-size value to include in the printed command (default 1024, "
                          "see docs/moe-offload.md -- larger ubatch generally helps prefill throughput "
                          "with CPU-resident experts)")
    ap.add_argument("--stage-slots", type=int, default=2,
                     help="GGML_META_STAGE_SLOTS to include (only meaningful with -sm tensor; "
                          "default 4 if -sm tensor is selected and experts are offloaded, omitted otherwise)")
    ap.add_argument("--partial-copy", action="store_true",
                     help="include GGML_META_PARTIAL_COPY=1 in the printed command "
                          "(only worth it for decode-heavy/low-batch workloads, see docs/moe-offload.md)")
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
    kv_bytes = (args.kv_gib * (1024 ** 3)) if args.kv_gib is not None else 0.0
    compute_bytes = args.compute_buffer_gib * (1024 ** 3)
    headroom_bytes = args.headroom_gib * args.gpus * (1024 ** 3)

    budget_for_weights = vram_total - kv_bytes - compute_bytes - headroom_bytes

    print(f"VRAM budget: {args.gpus} x {args.vram_gib:.2f} GiB = {human(vram_total)}")
    if args.kv_gib is not None:
        print(f"  - KV cache:              {human(kv_bytes)}")
    print(f"  - compute buffer:        {human(compute_bytes)}")
    print(f"  - headroom:              {human(headroom_bytes)}")
    print(f"  = budget for weights:    {human(budget_for_weights)}")
    print()

    if budget_for_weights <= 0:
        print("VRAM budget is already exhausted by KV + compute buffer + headroom alone -- "
              "every expert layer (and possibly non-expert weights too) needs to be offloaded. "
              "Consider -cmoe / --cpu-moe, a smaller context, or fewer GPUs' worth of KV replication.")
        print()
        print_command(args, ncmoe=None, offloading_experts=True, cmoe_all=True)
        sys.exit(0)

    if budget_for_weights >= total_model_bytes:
        print("The full model fits in the given VRAM budget -- you likely don't need -ncmoe at all. "
              "(Still confirm against the actual load log: this ignores allocator fragmentation "
              "and multi-GPU tensor-split imbalance.)")
        print()
        print_command(args, ncmoe=None, offloading_experts=False)
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
    print_command(args, ncmoe=n_offload, offloading_experts=True)


if __name__ == "__main__":
    main()
