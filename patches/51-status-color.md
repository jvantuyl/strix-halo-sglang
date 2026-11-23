# Patch 51 — color-code scheduler status-line fields

**File:** `python/sglang/srt/managers/scheduler_components/metrics_reporter.py` · **Test:** `tests/check_status_color.py` (5/5 in-image)

## Symptom

The per-batch status lines are the primary surface for wading through the
journal, but every field is plain text, so a live `journalctl --user -a -u
qwen38-sglang.service -f` gives the eye nothing to triage with: spotting
"decode is slow" or "the queue is backing up at the cap" means reading every
number.

## The patch

Wrap five fields in ANSI color (module-level constants, read once at import;
`SGLANG_QWEN4_STATUS_COLOR=0` turns every wrapper into the empty string):

| Field | Color |
|---|---|
| `gen throughput (token/s):` (decode) | bright green (92) |
| `input throughput (token/s):` (prefill) | bright cyan (96) |
| `#pending-token:` | yellow (33) |
| `#running-req:` (prefill + decode + flush refusal) | bright magenta (95) |
| `#queue-req:` (prefill + decode tail + flush refusal) | red (31) |

2026-11-23 tuning after live-journal feedback: dark green 32 was too dark
on the user's theme and plain magenta 35 read as unhighlighted, so prefill
moved to bright cyan 96 and `#running-req` to bright magenta 95. The same
pass extended coverage to every log emitter of the fields: the decode
tail's `#queue-req` in the metrics reporter, and the flush-cache refusal's
`#queue-req`/`#running-req` in `managers/scheduler.py` (which imports the
constants from the metrics reporter — no new import cycle: scheduler
already imports it at module top, and metrics_reporter imports scheduler
only under TYPE_CHECKING). The `#queue-req=` / `#running-req=` spellings
in the attach/detach rejection **API messages are deliberately left
plain** — they are machine-read response payloads, not log lines.

Ten edit sites: seven in the metrics reporter (five prefill-segment
f-strings, the decode line's combined f-string, the decode tail), three in
scheduler.py (import block, two flush-refusal f-strings).

## Design constraint: parsers keep working

Whole label+value segments are wrapped (`ESC[92m gen throughput (token/s):
2.63 ESC[0m`), never the number alone, so the inner text stays contiguous
and byte-identical. Anything that greps a field still matches:

- `bench/journal_stats.py`'s two regexes (`input throughput \(token/s\):
  ([0-9.]+)` in the prefill pattern, `gen throughput \(token/s\):
  ([0-9.]+)`) — verified against rendered samples in the test.
- The runbook's `grep -o` field-extraction pipelines.
- Plain `grep` matching (ANSI bytes only flank the matched region).

journald stores the ESC bytes and `journalctl -a` passes them through to
the terminal — the same path the existing status-bar escape codes already
ride, so no new journal flag is needed.

## Trade-offs accepted

- Piping journal output into a file gives it embedded escape bytes; for a
  file-diff use, set `SGLANG_QWEN4_STATUS_COLOR=0` (or run output through
  `sed 's/\x1b\[[0-9;]*m//g'`).
- Color choice is fixed, not per-field configurable — five env knobs for
  five shades would be noise.
- The static test can't prove the terminal rendering; the live check is
  reading a decode line in the journal after restart.
