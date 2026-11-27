#!/usr/bin/env python3
"""Patch 58: label MIXED results and report the shape of that forward."""

import sys
from pathlib import Path


def replace_once(source, old, new):
    count = source.count(old)
    assert count == 1, f"expected one metrics reporter anchor, got {count}: {old[:100]}"
    return source.replace(old, new, 1)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
    path = root / "python/sglang/srt/managers/scheduler_components/metrics_reporter.py"
    source = replace_once(
        path.read_text(),
        '''        msg = (
            f"Prefill batch{iter_msg}, "''',
        '''        mixed_fields = {}
        if batch is not None and batch.forward_mode is not None and batch.forward_mode.is_mixed():
            # mix_with_running appends the decode rows. Read the host-side
            # result snapshot, not the scheduler's current running batch.
            decode_rows = len(batch.decoding_reqs)
            prefill_rows = len(batch.reqs) - decode_rows
            total_positions = batch.extend_num_tokens
            decode_positions = sum(batch.extend_lens[prefill_rows:])
            prefill_positions = total_positions - decode_positions
            mixed_fields = dict(
                batch_mode="mixed", prefill_rows=prefill_rows, decode_rows=decode_rows,
                prefill_positions=prefill_positions, decode_positions=decode_positions,
                total_positions=total_positions,
            )
            heading = (
                f"Mixed batch{iter_msg}, prefill rows/positions: {prefill_rows}/{prefill_positions}, "
                f"decode rows/positions: {decode_rows}/{decode_positions}, total positions: {total_positions}, "
            )
        else:
            heading = f"Prefill batch{iter_msg}, "
        msg = (
            f"{heading}"''',
    )
    source = replace_once(
        source,
        '''        self._emit_batch_event("prefill_batch", batch, {
            "new_seq":''',
        '''        self._emit_batch_event("prefill_batch", batch, {
            **mixed_fields,
            "new_seq":''',
    )
    path.write_text(source)
    print("patch 58: label MIXED forwards and report prefill/decode rows and positions")


if __name__ == "__main__":
    main()
