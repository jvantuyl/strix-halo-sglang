#!/usr/bin/env python3
"""Throughput and memory benchmark for a running strix-halo-sglang server.

Three scenario families, each measured cold (``/flush_cache`` first) and
each with the GPU and host memory sampled while it runs:

* ``single``: one request per prompt length (TTFT, prefill tok/s, decode tok/s)
* ``concurrent``: N streams x prompt length (aggregate tok/s, per-stream
  decode, worst TTFT)
* ``mixed``: N short streams decoding while M long prompts prefill together
  (the worst case for VRAM)

Memory comes from the driver (``/sys/class/drm/card*/device/mem_info_*``:
VRAM used/free and GTT used, no root needed), ``/proc/meminfo`` (host
available and shmem) and, with ``--container``, the container's cgroup
(memory.current, anon, shmem). Each row reports the peak used / minimum
free inside its own window, so the prefill transient and the allocator's
high-water mark are visible per scenario. Weights parked in pinned host
memory (patch 27) are userptr mappings: they show up as container shmem,
not as GTT.

MTP on/off is a server launch option, not a request option, so run the
suite once per launch; each row carries the server's speculative settings,
request cap and context length from ``/get_server_info`` so JSONL rows
from different launches can be compared.

Examples (server up on :30001):

    tools/bench_qwen38.py --model qwen38-flash-next-derisked --container sglang-qwen38-derisked \\
        --single 128,2048,8192,32000 --concurrent 4,8,16,20 --concurrent-lengths 128 \\
        --mixed 18:26000:2 --passes 2 --out bench-mtp.jsonl --label "MTP cap 20"

    tools/bench_qwen38.py --single 128 --concurrent 20 --concurrent-lengths 128,2048   # quick

Prompt lengths are targets; the actual token count is reported (the
synthetic text runs ~124 tokens per paragraph plus a question).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

PARA = (
    "Paragraph {i}. The Strix Halo APU couples a 16-core Zen 5 CPU with a 40-CU RDNA 3.5 GPU "
    "sharing a unified 128 GB LPDDR5X pool at 256 GB/s; on this box 96 GB is carved out as VRAM. "
    "Qwen3.8-Flash-Next is a 48-layer hybrid of gated delta-net and sparse attention over a "
    "512-expert MoE, with a 51-billion-parameter n-gram embedding table that we keep on disk. "
)
TOKENS_PER_PARAGRAPH = 124
SHORT_PROMPT = "Write a 200-word story about a robot learning to paint."


def make_prompt(target_tokens, tag=""):
    n = max(1, round(target_tokens / TOKENS_PER_PARAGRAPH))
    head = f"({tag})\n" if tag else ""
    return head + "\n".join(PARA.format(i=i) for i in range(n)) + "\n\nSummarize the above in one paragraph."


# ----------------------------------------------------------------------------
# server


class Server:
    def __init__(self, base, model):
        self.base = base
        self.model = model

    def _post(self, path, body=None, timeout=120):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=timeout).read()

    def info(self):
        d = json.loads(urllib.request.urlopen(self.base + "/get_server_info", timeout=30).read())
        keys = [
            "served_model_name", "speculative_algorithm", "speculative_num_steps",
            "speculative_num_draft_tokens", "max_running_requests", "cuda_graph_max_bs",
            "context_length", "kv_cache_dtype", "max_total_tokens", "max_mamba_cache_size",
            "version",
        ]
        out = {k: d.get(k) for k in keys}
        if self.model is None:
            self.model = out["served_model_name"]
        return out

    def flush(self):
        self._post("/flush_cache")

    def stream(self, text, max_tokens):
        """One streaming chat completion; returns prompt/completion tokens, TTFT, rates."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        first = None
        usage = None
        with urllib.request.urlopen(req, timeout=36000) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    usage = d["usage"]
                if first is None:
                    for c in d.get("choices", []):
                        dl = c.get("delta", {})
                        if dl.get("content") or dl.get("reasoning_content"):
                            first = time.time()
        t1 = time.time()
        if usage is None or first is None:
            raise RuntimeError("stream ended without usage or without a token")
        pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
        ttft = first - t0
        decode = (ct - 1) / (t1 - first) if ct > 1 and t1 > first else float("nan")
        return dict(prompt=pt, completion=ct, ttft=ttft, prefill_tps=pt / ttft, decode_tps=decode, wall=t1 - t0)


# ----------------------------------------------------------------------------
# memory sampler


def _read_int(path):
    with open(path) as f:
        return int(f.read().split()[0])


def find_card():
    for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
        if os.path.exists(os.path.join(dev, "mem_info_vram_total")):
            return dev
    return None


def cgroup_dir(container):
    if not container:
        return None
    try:
        cid = subprocess.run(
            ["docker", "inspect", "-f", "{{.Id}}", container], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for pat in (f"/sys/fs/cgroup/system.slice/docker-{cid}.scope", f"/sys/fs/cgroup/docker/{cid}"):
        if os.path.isdir(pat):
            return pat
    return None


class MemorySampler(threading.Thread):
    """Samples driver VRAM/GTT, host memory and (optionally) a container cgroup."""

    def __init__(self, container=None, period=1.0):
        super().__init__(daemon=True)
        self.card = find_card()
        self.cg = cgroup_dir(container)
        self.period = period
        self.samples = []
        self._stop = threading.Event()
        self.container = container

    def sample(self):
        s = {"t": time.time()}
        if self.card:
            total = _read_int(f"{self.card}/mem_info_vram_total") // 2**20
            used = _read_int(f"{self.card}/mem_info_vram_used") // 2**20
            s.update(vram_used=used, vram_free=total - used, vram_total=total,
                     gtt_used=_read_int(f"{self.card}/mem_info_gtt_used") // 2**20)
        with open("/proc/meminfo") as f:
            mi = {k.rstrip(":"): int(v.split()[0]) // 1024 for k, v, *_ in (l.split() for l in f)}
        s.update(host_avail=mi.get("MemAvailable"), host_shmem=mi.get("Shmem"),
                 host_swap_used=(mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)))
        if self.cg:
            try:
                s["ctr_current"] = _read_int(f"{self.cg}/memory.current") // 2**20
                with open(f"{self.cg}/memory.stat") as f:
                    st = {k: int(v) for k, v in (l.split() for l in f)}
                s["ctr_anon"] = st.get("anon", 0) // 2**20
                s["ctr_shmem"] = st.get("shmem", 0) // 2**20
            except OSError:
                pass
        return s

    def run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self.sample())
            except OSError:
                pass
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()

    def window(self, t0, t1):
        """Extremes over [t0, t1] (plus one sample either side so short windows are covered)."""
        idx = [i for i, s in enumerate(self.samples) if t0 <= s["t"] <= t1]
        if idx:
            lo, hi = max(0, idx[0] - 1), min(len(self.samples) - 1, idx[-1] + 1)
        else:
            lo = hi = len(self.samples) - 1
        w = self.samples[lo:hi + 1] if self.samples else []
        if not w:
            return {}

        def ext(key, fn):
            vals = [s[key] for s in w if key in s and s[key] is not None]
            return fn(vals) if vals else None

        out = dict(
            vram_used_peak=ext("vram_used", max), vram_free_min=ext("vram_free", min),
            gtt_used_peak=ext("gtt_used", max), host_avail_min=ext("host_avail", min),
            host_swap_used_peak=ext("host_swap_used", max),
        )
        if self.cg:
            out.update(ctr_current_peak=ext("ctr_current", max), ctr_anon_peak=ext("ctr_anon", max),
                       ctr_shmem_peak=ext("ctr_shmem", max))
        return out


# ----------------------------------------------------------------------------
# scenarios


def fmt_mem(m):
    if not m:
        return "(no memory samples)"
    parts = []
    if m.get("vram_used_peak") is not None:
        parts.append(f"vram peak {m['vram_used_peak']}M / min free {m['vram_free_min']}M, gtt {m['gtt_used_peak']}M")
    if m.get("host_avail_min") is not None:
        parts.append(f"host avail min {m['host_avail_min']}M")
    if m.get("ctr_current_peak") is not None:
        parts.append(f"ctr {m['ctr_current_peak']}M (anon {m['ctr_anon_peak']}M, shmem {m['ctr_shmem_peak']}M)")
    return "; ".join(parts)


def run_single(srv, mem, lengths, gen):
    print(f"{'prompt':>7} {'TTFT s':>8} {'prefill tps':>12} {'decode tps':>11} {'gen':>5}  memory", flush=True)
    rows = []
    for tgt in lengths:
        srv.flush()
        t0 = time.time()
        r = srv.stream(make_prompt(tgt), gen)
        t1 = time.time()
        m = mem.window(t0, t1)
        print(f"{r['prompt']:>7} {r['ttft']:>8.2f} {r['prefill_tps']:>12.0f} {r['decode_tps']:>11.2f} {r['completion']:>5}  {fmt_mem(m)}", flush=True)
        rows.append(dict(scenario="single", target_tokens=tgt, **r, memory=m))
    return rows


def run_concurrent(srv, mem, counts, lengths, gen):
    rows = []
    for tgt in lengths:
        for n in counts:
            srv.flush()
            res = [None] * n
            errs = []

            def go(i):
                try:
                    text = SHORT_PROMPT if tgt <= 128 else make_prompt(tgt, tag=f"request {i}")
                    if tgt <= 128:
                        text = f"Request {i}: {text}"
                    res[i] = srv.stream(text, gen)
                except Exception as e:  # noqa: BLE001 - report and keep going
                    errs.append(repr(e))

            ths = [threading.Thread(target=go, args=(i,)) for i in range(n)]
            t0 = time.time()
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            t1 = time.time()
            ok = [r for r in res if r]
            tot = sum(r["completion"] for r in ok)
            agg = tot / (t1 - t0)
            per = sum(r["decode_tps"] for r in ok) / len(ok) if ok else float("nan")
            ttft_max = max((r["ttft"] for r in ok), default=float("nan"))
            m = mem.window(t0, t1)
            print(f"{n:>3} x {ok[0]['prompt'] if ok else tgt:>6} tok: {tot} tokens in {t1 - t0:.1f}s -> {agg:.1f} tok/s aggregate, "
                  f"per-stream {per:.2f}, TTFT max {ttft_max:.2f}s{', errors ' + str(len(errs)) if errs else ''}  {fmt_mem(m)}", flush=True)
            rows.append(dict(scenario="concurrent", streams=n, target_tokens=tgt,
                             prompt=ok[0]["prompt"] if ok else None, completion_total=tot, wall=t1 - t0,
                             aggregate_tps=agg, per_stream_decode_tps=per, ttft_max=ttft_max,
                             errors=errs, memory=m))
    return rows


def run_mixed(srv, mem, spec, gen, short_tokens):
    n_short, long_tokens, n_long = spec
    srv.flush()
    res = {}

    def go(k, text, mt):
        try:
            res[k] = srv.stream(text, mt)
        except Exception as e:  # noqa: BLE001
            res[k] = dict(error=repr(e))

    ths = [threading.Thread(target=go, args=(f"short{i}", f"Request {i}: {SHORT_PROMPT}" if short_tokens <= 128
                                             else make_prompt(short_tokens, tag=f"request {i}"), gen))
           for i in range(n_short)]
    ths += [threading.Thread(target=go, args=(f"long{j}", make_prompt(long_tokens, tag=f"variant {j}"), 64))
            for j in range(n_long)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    t1 = time.time()
    m = mem.window(t0, t1)
    errs = [k for k, r in res.items() if "error" in r]
    longs = [res[f"long{j}"] for j in range(n_long) if "error" not in res[f"long{j}"]]
    shorts = [res[f"short{i}"] for i in range(n_short) if "error" not in res[f"short{i}"]]
    print(f"mixed {n_short} short + {n_long} x {longs[0]['prompt'] if longs else long_tokens} tok: wall {t1 - t0:.1f}s, "
          f"errors {errs}  {fmt_mem(m)}", flush=True)
    for j, r in enumerate(longs):
        print(f"    long{j}: TTFT {r['ttft']:.1f}s prefill {r['prefill_tps']:.0f} tok/s decode {r['decode_tps']:.1f}", flush=True)
    if shorts:
        print(f"    shorts: per-stream decode {sum(r['decode_tps'] for r in shorts) / len(shorts):.2f} tok/s, "
              f"TTFT max {max(r['ttft'] for r in shorts):.2f}s", flush=True)
    return [dict(scenario="mixed", short_streams=n_short, long_count=n_long, long_target_tokens=long_tokens,
                 long_prompt=longs[0]["prompt"] if longs else None, wall=t1 - t0, errors=errs,
                 longs=longs, short_decode_tps=[r["decode_tps"] for r in shorts],
                 short_ttft_max=max((r["ttft"] for r in shorts), default=None), memory=m)]


# ----------------------------------------------------------------------------


def parse_ints(s):
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.environ.get("BASE", "http://127.0.0.1:30001"))
    ap.add_argument("--model", default=os.environ.get("MODEL"), help="served model name (default: from the server)")
    ap.add_argument("--container", default=None, help="docker container name for cgroup memory sampling")
    ap.add_argument("--label", default="", help="free text stored with every row (e.g. 'MTP cap 20')")
    ap.add_argument("--single", default="128,2048,8192", help="prompt lengths for single requests ('' to skip)")
    ap.add_argument("--concurrent", default="4,8", help="stream counts ('' to skip)")
    ap.add_argument("--concurrent-lengths", default="128", help="prompt lengths for the concurrent streams")
    ap.add_argument("--mixed", default="", help="N_SHORT:LONG_TOKENS:N_LONG, e.g. 18:26000:2 ('' to skip)")
    ap.add_argument("--gen", type=int, default=128, help="tokens generated per single request")
    ap.add_argument("--concurrent-gen", type=int, default=200, help="tokens generated per concurrent stream")
    ap.add_argument("--passes", type=int, default=1, help="repeat everything (the first pass warms Triton)")
    ap.add_argument("--out", default=None, help="append one JSON line per scenario row")
    args = ap.parse_args()

    srv = Server(args.base, args.model)
    info = srv.info()
    mem = MemorySampler(args.container)
    mem.start()
    time.sleep(1.5)
    idle0 = mem.sample()
    print(f"server: {info['served_model_name']}, spec={info['speculative_algorithm']} "
          f"(steps {info['speculative_num_steps']}, draft {info['speculative_num_draft_tokens']}), "
          f"cap {info['max_running_requests']}, context {info['context_length']}, kv {info['kv_cache_dtype']}")
    print(f"idle: vram used {idle0.get('vram_used')}M free {idle0.get('vram_free')}M, gtt {idle0.get('gtt_used')}M, "
          f"host avail {idle0.get('host_avail')}M shmem {idle0.get('host_shmem')}M"
          + (f", ctr {idle0.get('ctr_current')}M (anon {idle0.get('ctr_anon')}M, shmem {idle0.get('ctr_shmem')}M)" if mem.cg else
             ("" if not args.container else "  [container cgroup not found]")), flush=True)

    rows = []
    for p in range(1, args.passes + 1):
        if args.passes > 1:
            print(f"-- pass {p}", flush=True)
        if parse_ints(args.single):
            rows += [dict(r, **{"pass": p}) for r in run_single(srv, mem, parse_ints(args.single), args.gen)]
        if parse_ints(args.concurrent):
            rows += [dict(r, **{"pass": p}) for r in run_concurrent(
                srv, mem, parse_ints(args.concurrent), parse_ints(args.concurrent_lengths), args.concurrent_gen)]
        if args.mixed:
            spec = tuple(int(x) for x in args.mixed.split(":"))
            if len(spec) != 3:
                sys.exit("--mixed wants N_SHORT:LONG_TOKENS:N_LONG")
            short_len = parse_ints(args.concurrent_lengths)[0] if parse_ints(args.concurrent_lengths) else 128
            rows += [dict(r, **{"pass": p}) for r in run_mixed(srv, mem, spec, args.concurrent_gen, short_len)]

    time.sleep(1.5)
    idle1 = mem.sample()
    mem.stop()
    print(f"end: vram used {idle1.get('vram_used')}M free {idle1.get('vram_free')}M (driver view keeps the allocator's "
          f"high-water mark until /flush_cache), gtt {idle1.get('gtt_used')}M, host avail {idle1.get('host_avail')}M", flush=True)

    if args.out:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(dict(time=stamp, label=args.label, server=info, idle_before=idle0, idle_after=idle1, **r)) + "\n")
        print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
