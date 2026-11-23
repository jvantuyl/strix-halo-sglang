#!/usr/bin/env python3
"""gfx1151 patch 51: color-code scheduler status-line fields.

The scheduler's per-batch status lines ("Prefill batch, ...", "Decode
batch, ...") are the primary wading surface in the journal, but every field
is plain text, so scanning a day of logs for "is generation healthy / is
the queue backing up" is slow. This wraps five fields in ANSI color so the
eye can triage a live `journalctl --user -a -u qwen38-sglang.service -f`:

    gen throughput (token/s):  bright green   (92)  decode speed
    input throughput (token/s): bright cyan    (96)  prefill speed
    #pending-token:            yellow         (33)  waiting work
    #running-req:              bright magenta (95)  in-flight
    #queue-req:                red            (31)  queued at the cap

(2026-11-23 tuning after live-journal feedback: dark green 32 was too
dark on the user's theme and plain magenta 35 read as unhighlighted, so
prefill moved to bright cyan 96 and running-req to bright magenta 95.
Coverage extended to every emitter of the fields: the decode tail's
#queue-req in metrics_reporter.py and the flush-cache refusal's
#queue-req/#running-req in managers/scheduler.py.)

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

Edit sites: five f-string segments in the prefill message, the decode
line's combined f-string, and the decode tail's #queue-req, all in
scheduler_components/metrics_reporter.py; plus the flush-cache refusal's
#queue-req/#running-req in managers/scheduler.py (constants imported from
the metrics reporter — scheduler already imports that module at module
top, and metrics_reporter imports scheduler only under TYPE_CHECKING, so
no new cycle). Color constants are module-level, read once at import.

Run from the Dockerfile with the target tree as argv[1].
"""

import argparse
import os
import sys

TARGET = "python/sglang/srt/managers/scheduler_components/metrics_reporter.py"
SCHEDULER = "python/sglang/srt/managers/scheduler.py"

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
    _CLR_GEN = "\\033[92m"  # bright green  - gen throughput
    _CLR_PRE = "\\033[96m"  # bright cyan   - input (prefill) throughput
    _CLR_PEND = "\\033[33m"  # yellow       - #pending-token
    _CLR_RUN = "\\033[95m"  # bright magenta - #running-req
    _CLR_QUEUE = "\\033[31m"  # red         - #queue-req
    _CLR_END = "\\033[0m"
"""

# (path, old, new) — each old string is asserted unique in its file.
# Literals are single-quoted so f-string braces and double quotes pass.
EDITS = [
    # Prefill line: #running-req -> bright magenta
    (
        TARGET,
        '            f"#running-req: {prefill_stats.num_running_reqs.total}, "',
        '            f"{_CLR_RUN}#running-req: {prefill_stats.num_running_reqs.total}{_CLR_END}, "',
    ),
    # Prefill line: #queue-req -> red
    (
        TARGET,
        '            f"#queue-req: {len(self.scheduler.waiting_queue)}, "',
        '            f"{_CLR_QUEUE}#queue-req: {len(self.scheduler.waiting_queue)}{_CLR_END}, "',
    ),
    # Prefill line: #pending-token -> yellow
    (
        TARGET,
        '            f"#pending-token: {prefill_stats.num_pending_tokens}, "',
        '            f"{_CLR_PEND}#pending-token: {prefill_stats.num_pending_tokens}{_CLR_END}, "',
    ),
    # Prefill line: input throughput -> bright cyan
    (
        TARGET,
        '        msg += f"input throughput (token/s): {self.last_input_throughput:.2f}"',
        '        msg += f"{_CLR_PRE}input throughput (token/s): {self.last_input_throughput:.2f}{_CLR_END}"',
    ),
    # Decode line: #running-req -> bright magenta (inside combined f-string)
    (
        TARGET,
        '        msg = f"Decode batch{iter_msg}, #running-req: {num_running_reqs}, {token_usage_msg}"',
        '        msg = f"Decode batch{iter_msg}, {_CLR_RUN}#running-req: {num_running_reqs}{_CLR_END}, {token_usage_msg}"',
    ),
    # Decode line: gen throughput -> bright green
    (
        TARGET,
        '            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "',
        '            f"{_CLR_GEN}gen throughput (token/s): {self.last_gen_throughput:.2f}{_CLR_END}, "',
    ),
    # Decode tail: #queue-req -> red (no-trailing-comma variant)
    (
        TARGET,
        '            f"#queue-req: {len(self.scheduler.waiting_queue)}"',
        '            f"{_CLR_QUEUE}#queue-req: {len(self.scheduler.waiting_queue)}{_CLR_END}"',
    ),
    # scheduler.py: import the constants (no new import cycle: scheduler
    # already imports metrics_reporter at module top; metrics_reporter
    # imports scheduler only under TYPE_CHECKING).
    (
        SCHEDULER,
        """from sglang.srt.managers.scheduler_components.metrics_reporter import (
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsReporter,
)
""",
        """from sglang.srt.managers.scheduler_components.metrics_reporter import (
    _CLR_END,
    _CLR_QUEUE,
    _CLR_RUN,
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsReporter,
)
""",
    ),
    # scheduler.py flush-cache refusal: #queue-req -> red
    (
        SCHEDULER,
        '                f"#queue-req: {len(self.waiting_queue)}, "',
        '                f"{_CLR_QUEUE}#queue-req: {len(self.waiting_queue)}{_CLR_END}, "',
    ),
    # scheduler.py flush-cache refusal: #running-req -> bright magenta
    (
        SCHEDULER,
        '                f"#running-req: {len(self.running_batch.reqs)}"',
        '                f"{_CLR_RUN}#running-req: {len(self.running_batch.reqs)}{_CLR_END}"',
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/sgl-workspace/sglang")
    args = ap.parse_args()
    texts = {}
    reporter = os.path.join(args.target, TARGET)
    with open(reporter) as f:
        texts[reporter] = f.read()
    if "gfx1151 patch 51" in texts[reporter]:
        sys.exit("patch 51 already applied")
    assert texts[reporter].count(ANCHOR) == 1, "LOG_FORWARD_ITERS anchor not found"
    texts[reporter] = texts[reporter].replace(ANCHOR, ANCHOR + CONSTANTS, 1)
    for i, (rel, old, new) in enumerate(EDITS, 1):
        path = os.path.join(args.target, rel)
        if path not in texts:
            with open(path) as f:
                texts[path] = f.read()
            if "gfx1151 patch 51" in texts[path]:
                sys.exit(f"patch 51 already applied ({rel})")
        assert texts[path].count(old) == 1, f"edit {i} anchor not found in {rel}: {old!r}"
        texts[path] = texts[path].replace(old, new, 1)
    for path, text in texts.items():
        with open(path, "w") as f:
            f.write(text)
    print("patch 51 applied: status-line fields color-coded on every "
          "emitter (gen bright green 92, prefill bright cyan 96, pending "
          "yellow 33, running bright magenta 95, queue red 31; "
          "SGLANG_QWEN4_STATUS_COLOR=0 off)")


if __name__ == "__main__":
    main()
