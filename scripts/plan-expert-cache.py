#!/usr/bin/env python3
"""Plan the device expert cache of an offloaded MoE model from its measured routing.

A host layer keeps a pool of `cap` experts per GPU (GGML_META_EXPERT_CACHE, least recently used), a resident layer
keeps all of them. Routing reuses experts so much that a pool of a few hundred catches nearly everything, so for a
fixed VRAM budget a bigger cap on more layers usually copies far fewer bytes than whole resident layers, and the
layers worth keeping resident are the ones whose routing caches worst - often the first ones, which -ncmoe offloads.

  trace  runs llama-server with every MoE expert in host memory and the expert cache off, sends a fixed set of chat
         requests (plus an optional long document) and records which experts each layer routes to (GGML_MOE_IDS_TRACE).
         Pass your launcher's arguments after `--`, without its expert placement (-ncmoe / expert -ot): the script
         puts every expert on the host itself.
  plan   replays that routing through the pool's exact policy at every cap, layer by layer, and picks the resident
         layers and the cap that copy the fewest expert bytes per token within the VRAM your current config uses.
         Fits on half of the requests and scores on the other half, to show the choice transfers. Prints the
         GGML_META_EXPERT_CACHE and -ot lines for the launcher.

The byte counts are a replay; measure decode speed on the server before switching (A/B, fresh process per arm).
On Qwen3.8-Flash-Next and DeepSeek-V4.1 the recommended configs were +50% and +32% decode at the same VRAM.

Usage:
    python scripts/plan-expert-cache.py trace --server build-v100/bin/Release/llama-server.exe \\
        --model Z:/model-00001-of-00006.gguf --out trace-dir -- -sm tensor -ngl 99 -ts 1/1/1/1 --fit off \\
        -lm none -lzm off -fa on -c 131072 -b 2048 -ub 2048
    python scripts/plan-expert-cache.py plan --model Z:/model-00001-of-00006.gguf --trace-dir trace-dir \\
        --current-ncmoe 24 --current-cap 128
    python scripts/plan-expert-cache.py plan --model ... --trace-dir ... --current-host 1-39 --current-cap 56
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path

_spec = importlib.util.spec_from_file_location("estimate_ncmoe", Path(__file__).resolve().parent / "estimate-ncmoe.py")
_est = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_est)

PROMPTS = [
    ("explain", "Explain how a CPU branch predictor works, including two-level adaptive predictors and why mispredictions are expensive on deep pipelines."),
    ("code_py", "Write a Python module that implements an LRU cache with a TTL per entry, thread safety, and a decorator for memoizing functions. Include docstrings and a short usage example."),
    ("debug_cpp", "This C++ code sometimes crashes. Find the bug and explain the fix:\n\n```cpp\n#include <vector>\n#include <thread>\nstd::vector<int> v;\nvoid producer(){ for(int i=0;i<100000;i++) v.push_back(i); }\nint main(){ std::thread a(producer), b(producer); a.join(); b.join(); return v.size(); }\n```"),
    ("story", "Write a short story (about 500 words) about a lighthouse keeper who discovers that the light has been signalling to something far out at sea."),
    ("math", "A train leaves city A at 60 km/h and another leaves city B, 300 km away, at 90 km/h toward A, 30 minutes later. Where and when do they meet? Then generalize the solution to arbitrary speeds and delays."),
    ("summary", "Summarize the key ideas of the following passage and list three open questions it raises:\n\nMixture-of-experts language models route each token to a small subset of feed-forward experts chosen by a learned router. Because only a few experts are active per token, the compute per token is a fraction of a dense model with the same parameter count. However, the full set of experts must still be stored, which makes memory capacity rather than arithmetic the binding constraint on consumer hardware. Systems that offload experts to host memory must move expert weights across the PCIe link or compute them on the CPU, and the routing pattern determines how much can be cached on the accelerator."),
    ("json_tool", "Design a JSON schema for a calendar event API that supports recurring events with exceptions, time zones, and attendees with RSVP status. Then show two example payloads."),
    ("rust", "Implement a lock-free single-producer single-consumer ring buffer in Rust with a fixed capacity that is a power of two. Explain the memory ordering you chose for each atomic."),
]

EXPERT_PLACEMENT_FLAGS = {"-ncmoe", "--n-cpu-moe", "-cmoe", "--cpu-moe"}


# ---------------------------------------------------------------------------------------------------------------- trace

def http_json(url, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def cmd_trace(args):
    server, model = Path(args.server).resolve(), Path(args.model).resolve()
    for p in (server, model):
        if not p.exists():
            sys.exit(f"not found: {p}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace, log_path = out / "trace.txt", out / "server.log"
    if trace.exists():
        trace.unlink()

    server_args = list(args.server_args)
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]
    for i, a in enumerate(server_args):
        if a in EXPERT_PLACEMENT_FLAGS:
            sys.exit(f"remove {a} from the server arguments: the trace puts every expert on the host itself")

    env = dict(os.environ)
    for k in ("GGML_META_EXPERT_CACHE", "GGML_META_EXPERT_CACHE_DEVICE"):
        env.pop(k, None)  # the ids trace rides on the uncached copy path
    env["GGML_META_PARTIAL_COPY"] = "1"
    env["GGML_META_MOE_OFFLOAD_MAX_KIB"] = "100000000"  # stream every expert: a CPU-computed expert is never traced
    env["GGML_MOE_IDS_TRACE"] = str(trace)

    # the first -ot that matches a tensor wins, so every expert goes to the host whatever else is overridden
    cmd = [str(server), "-m", str(model), "-ot", r"blk\.[0-9]+\.ffn_.*_exps=CPU"] + server_args + \
          ["-np", "1", "--host", "127.0.0.1", "--port", str(args.port)]
    print("launching:", " ".join(cmd), flush=True)
    base = f"http://127.0.0.1:{args.port}"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                sys.exit(f"llama-server exited during load (code {proc.returncode}), see {log_path}")
            try:
                if http_json(base + "/health", timeout=5).get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(5)
        print(f"server ready after {time.time() - t0:.0f} s", flush=True)
        model_id = http_json(base + "/v1/models")["data"][0]["id"]

        results = []

        def run(name, msgs, max_tokens):
            l0 = count_lines(trace)
            t = time.time()
            j = http_json(base + "/v1/chat/completions", {"model": model_id, "messages": msgs, "max_tokens": max_tokens})
            l1 = count_lines(trace)
            tim = j.get("timings", {})
            results.append({"name": name, "lines": [l0, l1], "timings": tim})
            print(f"{name}: {time.time() - t:.0f} s, prompt {tim.get('prompt_n')}, generated {tim.get('predicted_n')} "
                  f"at {tim.get('predicted_per_second', 0):.2f} t/s, trace lines {l0} -> {l1}", flush=True)
            if l1 == l0:
                sys.exit("nothing was traced: the experts did not stream through the copy path (see the log)")
            return j["choices"][0]["message"].get("content") or ""

        for name, text in PROMPTS:
            run(name, [{"role": "user", "content": text}], args.max_tokens)

        if args.long_doc and args.long_chars > 0 and Path(args.long_doc).exists():
            doc = Path(args.long_doc).read_text(encoding="utf-8")[:args.long_chars]
            msgs = [{"role": "user", "content": doc + "\n\nSummarize the main topics of the text above, one paragraph each."}]
            try:
                answer = run("long1", msgs, 600)
                msgs += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": "Now list every person named in the text above, one line each."}]
                run("long2", msgs, 600)
            except urllib.error.HTTPError as e:
                print(f"long-context request skipped ({e}); lower --long-chars or raise -c", flush=True)

        (out / "requests.json").write_text(json.dumps(results, indent=1))
        print(f"wrote {trace} and {out / 'requests.json'}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()


# ----------------------------------------------------------------------------------------------------------------- plan

WARMUP = 64  # ubatches per request and layer left out of the cost: the replay starts each request with an empty pool


def parse_trace(path, requests, layers):
    """{(request, layer): [ids of one ubatch, ...]} in trace order"""
    lay_re = re.compile(r"blk\.(\d+)\.")
    marks = [(i, r["lines"][0], r["lines"][1]) for i, r in enumerate(requests)]
    ub, last_eval = {}, {}
    pr = 0
    with open(path) as f:
        for ln, line in enumerate(f):
            p = line.partition("|")[0].split()
            m = lay_re.search(p[1]) if len(p) >= 5 else None
            if m is None or int(m.group(1)) not in layers:
                continue
            while pr < len(marks) and ln >= marks[pr][2]:
                pr += 1
            if pr >= len(marks) or ln < marks[pr][1]:
                continue
            key = (pr, int(m.group(1)))
            ev = int(p[0])
            lst = ub.setdefault(key, [])
            if last_eval.get(key) != ev:
                last_eval[key] = ev
                lst.append([])
            lst[-1].extend(int(x) for x in p[4:])
    return ub


def simulate(ubatches, cap, n_expert):
    """experts copied at one cap, as ggml_backend_sched_expert_cache_plan_make does it: hits are touched first,
    misses take the least recently used slots, and a ubatch with more distinct experts than slots copies them all"""
    if cap >= n_expert:
        return 0
    lru, cost = OrderedDict(), 0
    for t, ids in enumerate(ubatches):
        distinct = list(dict.fromkeys(ids))
        for e in ids:
            if e in lru:
                lru.move_to_end(e)
        if len(distinct) > cap:
            c = len(distinct)
        else:
            c = 0
            for e in distinct:
                if e not in lru:
                    c += 1
                    if len(lru) >= cap:
                        lru.popitem(last=False)
                    lru[e] = True
        if t >= WARMUP:
            cost += c
    return cost


def _job(a):
    key, ubatches, grid, n_expert = a
    return key, [simulate(ubatches, c, n_expert) for c in grid]


def parse_layer_list(s):
    out = set()
    for part in re.split(r"[,\s]+", s.strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


class Plan:
    def __init__(self, layers, n_expert, bytes_of, grid, curves, tokens):
        self.layers, self.n_expert, self.bytes_of, self.grid = layers, n_expert, bytes_of, grid
        self.curves, self.tokens = curves, tokens  # curves[il][j]: experts copied at grid[j], summed over the requests

    def cost(self, il, cap):
        return 0 if cap >= self.n_expert else self.curves[il][self.grid.index(cap)]

    def mib_per_token(self, caps):
        return sum(self.cost(il, caps[il]) * self.bytes_of[il] for il in self.layers) / 2**20 / max(self.tokens, 1)

    def vram(self, caps):
        return sum(min(caps[il], self.n_expert) * self.bytes_of[il] for il in self.layers)

    def pick(self, budget):
        """best resident set + one cap for the rest within budget bytes: deployable with -ot and the cache env"""
        best = None
        for cap in self.grid:
            if cap >= self.n_expert:
                continue
            caps = {il: cap for il in self.layers}
            used = self.vram(caps)
            if used > budget:
                continue
            # a resident layer saves its copies at this cap and costs (n_expert - cap) more slots
            order = sorted(self.layers, key=lambda il: -self.cost(il, cap) / (self.n_expert - cap))
            for il in order:
                extra = (self.n_expert - cap) * self.bytes_of[il]
                if used + extra <= budget and self.cost(il, cap) > 0:
                    caps[il] = self.n_expert
                    used += extra
            v = self.mib_per_token(caps)
            if best is None or v < best[0]:
                best = (v, cap, caps)
        return best

    def per_layer(self, budget):
        """any cap per layer (multiple-choice knapsack); needs a per-layer cap in the pool, so information only"""
        unit = max(1.0, budget / 4000)
        B = int(budget // unit)
        opts = self.grid + ([self.n_expert] if self.n_expert not in self.grid else [])
        INF = float("inf")
        best = [0.0] + [INF] * B
        choice = []
        for il in self.layers:
            nb, ch = [INF] * (B + 1), [-1] * (B + 1)
            for b in range(B + 1):
                if best[b] == INF:
                    continue
                for j, c in enumerate(opts):
                    w = b + int(-(-c * self.bytes_of[il] // unit))
                    if w > B:
                        continue
                    v = best[b] + self.cost(il, c) * self.bytes_of[il]
                    if v < nb[w]:
                        nb[w], ch[w] = v, j
            best = nb
            choice.append(ch)
        b = min(range(B + 1), key=lambda x: best[x])
        caps = {}
        for k in range(len(self.layers) - 1, -1, -1):
            il = self.layers[k]
            c = opts[choice[k][b]]
            caps[il] = c
            b -= int(-(-c * self.bytes_of[il] // unit))
        return caps


def cmd_plan(args):
    reader, tensors = _est.load_shards(Path(args.model))
    n_used = _est.get_field_int(reader, "expert_used_count")
    per_layer = {}
    for t in tensors:
        il = _est.layer_index(t.name)
        if il is not None and "_exps" in t.name.split(".")[2]:
            e = per_layer.setdefault(il, [0, int(t.shape[-1])])
            e[0] += int(t.n_bytes)
    if not per_layer:
        sys.exit("no blk.<N>.ffn_*_exps tensors in the GGUF: not a MoE model")
    layers = sorted(per_layer)
    n_expert = per_layer[layers[0]][1]
    bytes_of = {il: per_layer[il][0] / per_layer[il][1] for il in layers}

    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    trace = Path(args.trace) if args.trace else trace_dir / "trace.txt"
    requests = json.loads((Path(args.requests) if args.requests else trace_dir / "requests.json").read_text())
    tokens = [int(r.get("timings", {}).get("predicted_n") or 0) for r in requests]

    ub = parse_trace(trace, requests, set(layers))
    if not ub:
        sys.exit("the trace has no rows for this model's MoE layers")
    traced = sorted({il for _, il in ub})
    if len(traced) < len(layers):
        print(f"warning: no routing traced for layers {sorted(set(layers) - set(traced))}; left out of the plan "
              f"and counted as they are in the current config")
    untraced = [il for il in layers if il not in traced]
    layers = traced
    max_distinct = max(len(set(ids)) for lst in ub.values() for ids in lst)

    # current config and its VRAM budget
    if args.current_ncmoe is not None:
        host = {il for il in layers if il < args.current_ncmoe}
    elif args.current_host is not None:
        host = parse_layer_list(args.current_host) & set(layers)
    elif args.current_resident is not None:
        host = set(layers) - parse_layer_list(args.current_resident)
    else:
        host = None
    if host is not None and args.current_cap is None:
        sys.exit("--current-cap is required with --current-ncmoe / --current-host / --current-resident")

    # a ubatch with more distinct experts than the cap copies them all, as the pool does, so small caps stay valid
    step = 8 if n_expert <= 256 else 16
    lo = max(step, -(-(n_used or 1) // step) * step)
    grid = sorted(set(range(lo, n_expert, step)) | ({args.current_cap} if args.current_cap else set()) | {n_expert})
    cur_caps = {il: (args.current_cap if il in host else n_expert) for il in layers} if host is not None else None
    if untraced and host is not None and any(il in host for il in untraced):
        sys.exit(f"layers {untraced} are on the host in the current config but have no routing in the trace")

    print(f"model: {len(layers)} MoE layers, {n_expert} experts ({n_used} used), "
          f"{sum(bytes_of.values()) / len(layers) / 2**20:.2f} MiB per expert on average; "
          f"trace: {len(requests)} requests, {sum(tokens)} generated tokens, at most {max_distinct} experts per ubatch")
    print(f"replaying {len(grid)} caps x {len(layers)} layers x {len(requests)} requests ...", flush=True)

    tasks = [((pr, il), ub.get((pr, il), []), grid, n_expert) for pr in range(len(requests)) for il in layers]
    with Pool(args.workers) as pool:
        res = dict(pool.map(_job, tasks, chunksize=4))

    def plan_for(req_ids):
        curves = {il: [0] * len(grid) for il in layers}
        for pr in req_ids:
            for il in layers:
                for j, x in enumerate(res[(pr, il)]):
                    curves[il][j] += x
        return Plan(layers, n_expert, bytes_of, grid, curves, sum(tokens[i] for i in req_ids))

    full = plan_for(range(len(requests)))
    if cur_caps is not None:
        budget = full.vram(cur_caps)
    elif args.budget_gib is not None:
        budget = args.budget_gib * 2**30 * args.gpus
    else:
        sys.exit("give the current config (--current-ncmoe/--current-host/--current-resident with --current-cap) or --budget-gib")

    def gib_dev(b):
        return b / args.gpus / 2**30

    print(f"expert VRAM budget: {gib_dev(budget):.2f} GiB per GPU over {args.gpus} GPUs\n")

    v, cap, caps = full.pick(budget)
    resident = sorted(il for il in layers if caps[il] >= n_expert)
    host_new = sorted(il for il in layers if caps[il] < n_expert)
    pl = full.per_layer(budget)

    print("expert copies per generated token (all requests):")
    if cur_caps is not None:
        print(f"  current  ({len(layers) - len(host)} resident, cap {args.current_cap}):  {full.mib_per_token(cur_caps):8.1f} MiB")
    print(f"  plan     ({len(resident)} resident, cap {cap}):  {v:8.1f} MiB   {gib_dev(full.vram(caps)):.2f} GiB per GPU")
    print(f"  per-layer caps (needs code, for reference): {full.mib_per_token(pl):8.1f} MiB")

    if len(requests) >= 4:
        halves = [list(range(0, len(requests), 2)), list(range(1, len(requests), 2))]
        print("\nheld-out (chosen on one half of the requests, scored on the other):")
        for fit, ev in ((0, 1), (1, 0)):
            pf, pe = plan_for(halves[fit]), plan_for(halves[ev])
            _, cf, capsf = pf.pick(budget)
            line = f"  fit {fit} -> score {ev}: plan {pe.mib_per_token(capsf):8.1f} MiB"
            if cur_caps is not None:
                c0 = pe.mib_per_token(cur_caps)
                line += f" vs current {c0:8.1f} ({100 * (pe.mib_per_token(capsf) / c0 - 1):+.0f}%)"
            print(line + f", resident {sorted(il for il in layers if capsf[il] >= n_expert)}, cap {cf}")

    print("\nper layer: copies per token at caps", ", ".join(str(c) for c in grid[::max(1, len(grid) // 6)][:6]),
          "| current -> plan")
    shown = grid[::max(1, len(grid) // 6)][:6]
    for il in layers:
        cur = "" if cur_caps is None else ("res" if cur_caps[il] >= n_expert else str(cur_caps[il]))
        new = "res" if caps[il] >= n_expert else str(caps[il])
        print(f"  blk.{il:<3d} " + " ".join(f"{full.cost(il, c) / max(full.tokens, 1):6.2f}" for c in shown) + f"   {cur:>4s} -> {new}")

    ot = r"blk\.(" + "|".join(str(il) for il in host_new) + r")\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU"
    print("\nlauncher config (replaces -ncmoe / --cpu-moe and any -ot that places experts):")
    print(f"  set GGML_META_EXPERT_CACHE={cap}")
    print("  set GGML_META_EXPERT_CACHE_DEVICE=1")
    print("  set GGML_META_PARTIAL_COPY=1")
    print(f'  -ot "{ot}"')
    print(f"  resident layers: {resident}")
    print("\nThis is a byte count from a replay: A/B decode and prefill on the server (fresh process per arm) and check the")
    print("load log for pool allocation warnings before switching; more host layers can shift the compute buffer size.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trace", help="record routing with every expert on the host")
    t.add_argument("--server", required=True, help="llama-server executable")
    t.add_argument("--model", required=True, help="GGUF (first shard of a split model)")
    t.add_argument("--out", required=True, help="directory for trace.txt, requests.json and server.log")
    t.add_argument("--port", type=int, default=11499)
    t.add_argument("--max-tokens", type=int, default=800, help="generated tokens per chat request")
    t.add_argument("--long-doc", default=str(Path(__file__).resolve().parent.parent / "wiki.test.raw"),
                   help="text for the two long-context requests (default: wiki.test.raw in the repo root)")
    t.add_argument("--long-chars", type=int, default=330000, help="characters of --long-doc to send, 0 = skip (~78k tokens)")
    t.add_argument("server_args", nargs=argparse.REMAINDER, help="-- then the launcher's llama-server arguments")

    p = sub.add_parser("plan", help="choose resident layers and cap from a trace")
    p.add_argument("--model", required=True, help="GGUF (first shard of a split model), for the expert sizes")
    p.add_argument("--trace-dir", help="directory written by `trace`")
    p.add_argument("--trace", help="trace file, instead of --trace-dir")
    p.add_argument("--requests", help="requests.json, instead of --trace-dir")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--current-ncmoe", type=int, help="current -ncmoe N (blk 0..N-1 on the host)")
    g.add_argument("--current-host", help="current host layers, e.g. '1-39' or '1,2,4-14'")
    g.add_argument("--current-resident", help="current resident layers; all others are on the host")
    p.add_argument("--current-cap", type=int, help="current GGML_META_EXPERT_CACHE")
    p.add_argument("--budget-gib", type=float, help="expert VRAM per GPU to plan for, instead of the current config")
    p.add_argument("--gpus", type=int, default=4, help="GPUs the experts are split over (for the per-GPU figures)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))

    args = ap.parse_args()
    if args.cmd == "trace":
        cmd_trace(args)
    else:
        if not args.trace_dir and not (args.trace and args.requests):
            ap.error("plan needs --trace-dir or both --trace and --requests")
        cmd_plan(args)


if __name__ == "__main__":
    main()
