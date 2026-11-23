#!/usr/bin/env python3
"""check_status_color.py — verify patch 51 in the image, no GPU, no import.

Patch 51 color-codes five scheduler status-line fields in the metrics
reporter (the file sits behind a GPU-probing import chain, so this check
reads the patched source directly). It also re-checks the design invariant
with a rendered sample: the ANSI wraps must flank whole label+value
segments so regex field extraction (bench/journal_stats.py, grep -o
pipelines) keeps matching.

Run: python3 tests/check_status_color.py   (inside the image)
"""

import re
import sys

PATH = ("/sgl-workspace/sglang/python/sglang/srt/managers/"
        "scheduler_components/metrics_reporter.py")

# (wrapped form in the source, color, label) for the six edit sites.
SITES = [
    ('f"{_CLR_RUN}#running-req: {prefill_stats.num_running_reqs.total}{_CLR_END}, "',
     "\x1b[35m", "#running-req"),
    ('f"{_CLR_QUEUE}#queue-req: {len(self.scheduler.waiting_queue)}{_CLR_END}, "',
     "\x1b[31m", "#queue-req"),
    ('f"{_CLR_PEND}#pending-token: {prefill_stats.num_pending_tokens}{_CLR_END}, "',
     "\x1b[33m", "#pending-token"),
    ('f"{_CLR_PRE}input throughput (token/s): {self.last_input_throughput:.2f}{_CLR_END}"',
     "\x1b[32m", "input throughput (token/s)"),
    ('f"Decode batch{iter_msg}, {_CLR_RUN}#running-req: {num_running_reqs}{_CLR_END}, {token_usage_msg}"',
     "\x1b[35m", "#running-req"),
    ('f"{_CLR_GEN}gen throughput (token/s): {self.last_gen_throughput:.2f}{_CLR_END}, "',
     "\x1b[92m", "gen throughput (token/s)"),
]

# journal_stats.py's two regexes, plus the runbook grep-style extractions.
PARSER_PATTERNS = [
    r"Prefill batch.*?#new-token: (\d+).*?input throughput \(token/s\): ([0-9.]+)",
    r"gen throughput \(token/s\): ([0-9.]+)",
    r"#running-req: (\d+)",
    r"#queue-req: (\d+)",
    r"#pending-token: (\d+)",
    r"input throughput \(token/s\): ([0-9.]+)",
]

SAMPLE_PREFILL = (
    "Prefill batch [8], #new-seq: 1, #new-token: 1024, #cached-token: 70144, "
    "\x1b[35m#running-req: 3\x1b[0m, \x1b[31m#queue-req: 0\x1b[0m, "
    "\x1b[33m#pending-token: 1342\x1b[0m, backend: False, "
    "\x1b[32minput throughput (token/s): 500.26\x1b[0m"
)
SAMPLE_DECODE = (
    "Decode batch [7], \x1b[35m#running-req: 4\x1b[0m, token usage: 0.41, "
    "\x1b[92mgen throughput (token/s): 2.63\x1b[0m, accept len: 2.91"
)


def main():
    failures = []
    try:
        src = open(PATH).read()
    except OSError as e:
        print(f"FAIL: cannot read {PATH}: {e}")
        sys.exit(1)

    # 1. All six edit sites wrapped, with the right color constant.
    for wrapped, color, label in SITES:
        if wrapped not in src:
            failures.append(f"edit site missing: {label} ({color} variant)")
    # 2. Color constants defined once, with the opt-out branch.
    if src.count('_CLR_GEN = "\\033[92m"') != 1:
        failures.append("light-green constant missing or duplicated")
    if 'environ.get("SGLANG_QWEN4_STATUS_COLOR", "1") == "0"' not in src:
        failures.append("SGLANG_QWEN4_STATUS_COLOR=0 opt-out missing")
    # 3. The un-colored originals are gone (all six were unique).
    for plain in (
        'f"#running-req: {prefill_stats.num_running_reqs.total}, "',
        'f"#queue-req: {len(self.scheduler.waiting_queue)}, "',
        'f"#pending-token: {prefill_stats.num_pending_tokens}, "',
        'f"input throughput (token/s): {self.last_input_throughput:.2f}"',
        'f"Decode batch{iter_msg}, #running-req: {num_running_reqs}, {token_usage_msg}"',
        'f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "',
    ):
        if plain in src:
            failures.append(f"un-wrapped original still present: {plain!r}")
    # 4. Design invariant: parser regexes match through the color wraps.
    expected = {
        (SAMPLE_PREFILL, PARSER_PATTERNS[0]): ("1024", "500.26"),
        (SAMPLE_DECODE, PARSER_PATTERNS[1]): ("2.63",),
        (SAMPLE_PREFILL, PARSER_PATTERNS[2]): ("3",),
        (SAMPLE_PREFILL, PARSER_PATTERNS[3]): ("0",),
        (SAMPLE_PREFILL, PARSER_PATTERNS[4]): ("1342",),
        (SAMPLE_PREFILL, PARSER_PATTERNS[5]): ("500.26",),
    }
    for (line, pat), want in expected.items():
        m = re.search(pat, line)
        if not m or m.groups() != want:
            failures.append(f"parser regex fails on colored line: {pat!r} "
                            f"(got {m.groups() if m else None}, want {want})")
    # 5. Patch marker for provenance.
    if "gfx1151 patch 51" not in src:
        failures.append("patch 51 marker comment missing")

    if failures:
        print("FAIL:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("check_status_color: 5/5 OK (six sites wrapped, opt-out present, "
          "originals replaced, parser regexes match through the wraps, "
          "marker present). Live check: a decode status line in the journal "
          "shows gen tps in light green with `journalctl -a`.")


if __name__ == "__main__":
    main()
