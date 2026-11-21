"""Per-epoch serving statistics from the qwen38 SGLang journal.

Splits the journal into epochs (one per boot, keyed on the server's ready
line) and reports, per epoch:

* prefill: percentiles of the per-chunk `input throughput` from
  "Prefill batch" lines, plus total chunked-prefill tokens
* generation: percentiles of `gen throughput` from "Decode batch" lines
* accept len / accept rate: percentiles from the same lines

P50/P90/P99 rather than min/max/avg: the tails (first chunk after idle,
retraction events) are structurally different events, not noise.

Usage:
    python3 bench/journal_stats.py [--since '2026-11-19'] [--until ...]
        [--unit qwen38-sglang.service] [--json]

Reads only; needs journalctl access to the user unit's logs.
"""

import argparse
import json
import re
import subprocess
import sys

READY = "The server is fired up and ready to roll!"
PREFILL_RE = re.compile(
    r"Prefill batch.*?#new-token: (\d+).*?input throughput \(token/s\): ([0-9.]+)"
)
DECODE_RE = re.compile(
    r"Decode batch.*?"
    r"accept len: ([0-9.]+), accept rate: ([0-9.]+).*?"
    r"gen throughput \(token/s\): ([0-9.]+)"
)
TS_RE = re.compile(r"^\w+ (\d+) (\d\d:\d\d:\d\d)")


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def summarize(vals, total=None):
    if not vals:
        return "n=0"
    s = sorted(vals)
    mean = sum(s) / len(s)
    out = (f"n={len(vals)} p50={pct(s, 50):.4g} p90={pct(s, 90):.4g} "
           f"p99={pct(s, 99):.4g} mean={mean:.4g}")
    if total is not None:
        out += f" total={total:.4g}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="-48 hours")
    ap.add_argument("--until", default=None)
    ap.add_argument("--unit", default="qwen38-sglang.service")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    cmd = ["journalctl", "--user", "-a", "--no-pager",
           f"-u{args.unit}", f"--since={args.since}"]
    if args.until:
        cmd.append(f"--until={args.until}")
    out = subprocess.run(cmd, capture_output=True, text=True).stdout

    epochs = []  # list of dicts: ts, prefill_tp, prefill_tokens, gen_tp, acc_len, acc_rate
    current = {"ts": "(pre-first-boot)", "data": []}
    epochs.append(current)

    for line in out.splitlines():
        if READY in line:
            m = TS_RE.match(line)
            current = {"ts": f"{m.group(1)} {m.group(2)}" if m else "?",
                       "data": []}
            epochs.append(current)
            continue
        m = PREFILL_RE.search(line)
        if m:
            toks, tp = int(m.group(1)), float(m.group(2))
            current["data"].append(("prefill", tp, toks))
            continue
        m = DECODE_RE.search(line)
        if m:
            current["data"].append(("decode", float(m.group(1)),
                                    float(m.group(2)), float(m.group(3))))

    if args.json:
        rows = []
        for e in epochs:
            d = e["data"]
            pre = [x[1] for x in d if x[0] == "prefill"]
            pre_toks = sum(x[2] for x in d if x[0] == "prefill")
            dec = [x for x in d if x[0] == "decode"]
            rows.append({
                "epoch": e["ts"],
                "prefill_tp": pre,
                "prefill_tokens": pre_toks,
                "gen_tp": [x[3] for x in dec],
                "accept_len": [x[1] for x in dec],
                "accept_rate": [x[2] for x in dec],
            })
        print(json.dumps(rows, indent=1))
        return

    print(f"{'epoch start':>14}  {'prefill chunk tok/s (n, p50, p90, p99, tokens)':<58}  "
          f"{'gen tok/s p50/p90/p99':<24}  {'accept len p50/p90/p99':<24}  {'accept rate p50/p90/p99'}")
    for e in epochs:
        d = e["data"]
        pre = [x[1] for x in d if x[0] == "prefill"]
        pre_toks = sum(x[2] for x in d if x[0] == "prefill")
        dec = [x for x in d if x[0] == "decode"]
        gen = sorted(x[3] for x in dec)
        alen = sorted(x[1] for x in dec)
        arate = sorted(x[2] for x in dec)

        def trip(v):
            return "n=0" if not v else f"{pct(v, 50):.4g}/{pct(v, 90):.4g}/{pct(v, 99):.4g}"

        pre_s = "n=0" if not pre else (
            f"n={len(pre)} p50={pct(pre, 50):.0f} p90={pct(pre, 90):.0f} "
            f"p99={pct(pre, 99):.0f} tok={pre_toks}")
        print(f"{e['ts']:>14}  {pre_s:<58}  {trip(gen):<24}  {trip(alen):<24}  {trip(arate)}")


if __name__ == "__main__":
    main()
