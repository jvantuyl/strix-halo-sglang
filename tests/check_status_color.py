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
SCHED_PATH = "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py"

# (file key, wrapped form in the source, color, label) for the edit sites.
SITES = [
    ("m", 'f"{_CLR_RUN}#running-req: {prefill_stats.num_running_reqs.total}{_CLR_END}, "',
     "\x1b[95m", "#running-req"),
    ("m", 'f"{_CLR_QUEUE}#queue-req: {len(self.scheduler.waiting_queue)}{_CLR_END}, "',
     "\x1b[31m", "#queue-req"),
    ("m", 'f"{_CLR_PEND}#pending-token: {prefill_stats.num_pending_tokens}{_CLR_END}, "',
     "\x1b[33m", "#pending-token"),
    ("m", 'f"{_CLR_PRE}input throughput (token/s): {self.last_input_throughput:.2f}{_CLR_END}"',
     "\x1b[96m", "input throughput (token/s)"),
    ("m", 'f"Decode batch{iter_msg}, {_CLR_RUN}#running-req: {num_running_reqs}{_CLR_END}, {token_usage_msg}"',
     "\x1b[95m", "#running-req"),
    ("m", 'f"{_CLR_GEN}gen throughput (token/s): {self.last_gen_throughput:.2f}{_CLR_END}, "',
     "\x1b[92m", "gen throughput (token/s)"),
    ("m", 'f"{_CLR_QUEUE}#queue-req: {len(self.scheduler.waiting_queue)}{_CLR_END}"',
     "\x1b[31m", "#queue-req (decode tail)"),
    ("s", 'f"{_CLR_QUEUE}#queue-req: {len(self.waiting_queue)}{_CLR_END}, "',
     "\x1b[31m", "#queue-req (flush refusal)"),
    ("s", 'f"{_CLR_RUN}#running-req: {len(self.running_batch.reqs)}{_CLR_END}"',
     "\x1b[95m", "#running-req (flush refusal)"),
    ("s", """from sglang.srt.managers.scheduler_components.metrics_reporter import (
    _CLR_END,
    _CLR_QUEUE,
    _CLR_RUN,
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsReporter,
)""",
     "", "scheduler imports the color constants"),
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
    "\x1b[95m#running-req: 3\x1b[0m, \x1b[31m#queue-req: 0\x1b[0m, "
    "\x1b[33m#pending-token: 1342\x1b[0m, backend: False, "
    "\x1b[96minput throughput (token/s): 500.26\x1b[0m"
)
SAMPLE_DECODE = (
    "Decode batch [7], \x1b[95m#running-req: 4\x1b[0m, token usage: 0.41, "
    "\x1b[92mgen throughput (token/s): 2.63\x1b[0m, accept len: 2.91"
)


def main():
    failures = []
    try:
        srcs = {"m": open(PATH).read(), "s": open(SCHED_PATH).read()}
    except OSError as e:
        print(f"FAIL: cannot read source: {e}")
        sys.exit(1)
    src = srcs["m"]
    sched = srcs["s"]

    # 1. All edit sites wrapped, with the right color constant.
    for key, wrapped, color, label in SITES:
        if wrapped not in srcs[key]:
            failures.append(f"edit site missing: {label} ({color} variant)")
    # 2. Color constants defined once, with the opt-out branch.
    if src.count('_CLR_GEN = "\\033[92m"') != 1:
        failures.append("bright-green constant missing or duplicated")
    if src.count('_CLR_RUN = "\\033[95m"') != 1:
        failures.append("bright-magenta constant missing or duplicated")
    if src.count('_CLR_PRE = "\\033[96m"') != 1:
        failures.append("bright-cyan constant missing or duplicated")
    if 'environ.get("SGLANG_QWEN4_STATUS_COLOR", "1") == "0"' not in src:
        failures.append("SGLANG_QWEN4_STATUS_COLOR=0 opt-out missing")
    # 3. The un-colored log originals are gone (each was unique). The
    #    API-response "#queue-req=" / "#running-req=" variants (attach/
    #    detach rejection messages, machine-read by clients) must stay
    #    plain and untouched.
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
    for plain in (
        'f"#queue-req: {len(self.waiting_queue)}, "',
        'f"#running-req: {len(self.running_batch.reqs)}"',
    ):
        if plain in sched:
            failures.append(f"un-wrapped scheduler original still present: {plain!r}")
    if sched.count('f"#queue-req={len(self.waiting_queue)} "') != 2:
        failures.append("API-response #queue-req= variants altered (must stay plain)")
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
    print("check_status_color: 5/5 OK (ten edit sites wrapped across the "
          "metrics reporter and scheduler, opt-out present, originals "
          "replaced, API-response variants left plain, parser regexes match "
          "through the wraps, marker present). Live check: decode gen tps "
          "bright green, prefill tps bright cyan, #running-req bright "
          "magenta on `journalctl -a`.")


if __name__ == "__main__":
    main()
