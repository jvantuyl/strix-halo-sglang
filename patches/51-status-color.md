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
| `gen throughput (token/s):` (decode) | light green (92) |
| `input throughput (token/s):` (prefill) | dark green (32) |
| `#pending-token:` | yellow (33) |
| `#running-req:` (prefill + decode) | magenta (35) |
| `#queue-req:` | red (31) |

Six edit sites: five f-string segments in the prefill message, one combined
f-string in the decode message (`#running-req` lives inside it, split with
the color constants spliced in around the label+value).

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
