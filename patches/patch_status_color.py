#!/usr/bin/env python3
"""gfx1151 patch 51: color-code scheduler status-line fields.

The scheduler's per-batch status lines ("Prefill batch, ...", "Decode
batch, ...") are the primary wading surface in the journal, but every field
is plain text, so scanning a day of logs for "is generation healthy / is
the queue backing up" is slow. This wraps five fields in ANSI color so the
eye can triage a live `journalctl --user -a -u qwen38-sglang.service -f`:

    gen throughput (token/s):  light green  (92)  decode speed
    input throughput (token/s): dark green   (32)  prefill speed
    #pending-token:            yellow       (33)  waiting work
    #running-req:              magenta      (35)  in-flight
    #queue-req:                red          (31)  queued at the cap

Design constraint: whole label+value segments are wrapped (`ESC ... label:
value ... ESC[0m`), never the number alone, so the inner text stays
contiguous and byte-identical. Everything that greps a field still matches:
bench/journal_stats.py's `gen throughput \\(token/s\\): ([0-9.]+)` and the
runbook's `grep -o` field-extraction pipelines all keep working — ANSI
bytes only flank the matched region. journald stores the ESC bytes and
`journalctl -a` passes them through to the terminal (the status bars'
escape codes already ride the same path).

SGLANG_QWEN4_STATUS_COLOR=0 turns every wrapper into the empty string
(plain output, e.g. for piping the journal into a file diff).

Six edit sites in sglang/srt/managers/scheduler_components/metrics_reporter.py
(five prefill-segment f-strings plus the decode line's combined f-string);
color constants are module-level, read once at import.

Run from the Dockerfile with the target tree as argv[1].
"""

import argparse
import os
import sys

TARGET = "python/sglang/srt/managers/scheduler_components/metrics_reporter.py"

# Module-level constants, inserted after the LOG_FORWARD_ITERS definition.
ANCHOR = "LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()"

CONSTANTS = """

# gfx1151 patch 51: color-code scheduler status-line fields for journal
# wading (journalctl -a passes the ANSI escapes through to the terminal).
# Whole label+value segments are wrapped so the inner text stays contiguous
# and byte-identical; regex field extraction (bench/journal_stats.py,
# grep -o pipelines) keeps matching. SGLANG_QWEN4_STATUS_COLOR=0 disables.
import os as _os51

if _os51.environ.get("SGLANG_QWEN4_STATUS_COLOR", "1") == "0":
    _CLR_GEN = _CLR_PRE = _CLR_PEND = _CLR_RUN = _CLR_QUEUE = _CLR_END = ""
else:
    _CLR_GEN = "\\033[92m"  # light green - gen throughput
    _CLR_PRE = "\\033[32m"  # dark green  - input (prefill) throughput
    _CLR_PEND = "\\033[33m"  # yellow     - #pending-token
    _CLR_RUN = "\\033[35m"  # magenta     - #running-req
    _CLR_QUEUE = "\\033[31m"  # red        - #queue-req
    _CLR_END = "\\033[0m"
"""

# (old, new) — each old string is asserted unique in the file. Literals are
# single-quoted so the f-string braces and inner double quotes pass through.
EDITS = [
    # Prefill line: #running-req -> magenta
    (
        '            f"#running-req: {prefill_stats.num_running_reqs.total}, "',
        '            f"{_CLR_RUN}#running-req: {prefill_stats.num_running_reqs.total}{_CLR_END}, "',
    ),
    # Prefill line: #queue-req -> red
    (
        '            f"#queue-req: {len(self.scheduler.waiting_queue)}, "',
        '            f"{_CLR_QUEUE}#queue-req: {len(self.scheduler.waiting_queue)}{_CLR_END}, "',
    ),
    # Prefill line: #pending-token -> yellow
    (
        '            f"#pending-token: {prefill_stats.num_pending_tokens}, "',
        '            f"{_CLR_PEND}#pending-token: {prefill_stats.num_pending_tokens}{_CLR_END}, "',
    ),
    # Prefill line: input throughput -> dark green
    (
        '        msg += f"input throughput (token/s): {self.last_input_throughput:.2f}"',
        '        msg += f"{_CLR_PRE}input throughput (token/s): {self.last_input_throughput:.2f}{_CLR_END}"',
    ),
    # Decode line: #running-req -> magenta (inside the combined f-string)
    (
        '        msg = f"Decode batch{iter_msg}, #running-req: {num_running_reqs}, {token_usage_msg}"',
        '        msg = f"Decode batch{iter_msg}, {_CLR_RUN}#running-req: {num_running_reqs}{_CLR_END}, {token_usage_msg}"',
    ),
    # Decode line: gen throughput -> light green
    (
        '            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "',
        '            f"{_CLR_GEN}gen throughput (token/s): {self.last_gen_throughput:.2f}{_CLR_END}, "',
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/sgl-workspace/sglang")
    args = ap.parse_args()
    path = os.path.join(args.target, TARGET)
    with open(path) as f:
        text = f.read()
    if "gfx1151 patch 51" in text:
        sys.exit("patch 51 already applied")
    assert text.count(ANCHOR) == 1, "LOG_FORWARD_ITERS anchor not found"
    text = text.replace(ANCHOR, ANCHOR + CONSTANTS, 1)
    for i, (old, new) in enumerate(EDITS, 1):
        assert text.count(old) == 1, f"edit {i} anchor not found: {old!r}"
        text = text.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(text)
    print("patch 51 applied: status-line fields color-coded "
          "(gen light green, prefill dark green, pending yellow, "
          "running magenta, queue red; SGLANG_QWEN4_STATUS_COLOR=0 off)")


if __name__ == "__main__":
    main()
