"""gfx1151 patch 44: make the PLE row prefetcher actually warm the cache.

`PleFilePrefetcher` (patch 13-era) hints the page cache about the pages a
prefill-sized gather is about to read via `posix_fadvise(WILLNEED)`, one
call per page, on a background thread. On THIS kernel that syscall is a
no-op — measured twice (2026-11-21): the presharded dump files during the
patch-42 work, and this table directly (2 GiB range, residency 30.1% ->
30.3% across the WILLNEED). The prefetch therefore does nothing here and
every prefill chunk pays for its cold rows synchronously inside the gather:

* measured 2026-11-21, 62k-token cold prefill: the scheduler read ~19 MiB
  per chunk from the SSD for the whole run (~1.2 GB total), constant from
  first chunk to last — the working set does not stay cached under this
  box's memory pressure, so the cost is per-chunk and permanent.
* profile over the same run: ~5% of the scheduler thread inside
  `_gather_ple_embedding_on_cpu` (the fault path), and prefill chunk
  throughput 674 tok/s vs 730-735 in the same-content repeated-probe runs
  where the rows had been faulted in by earlier runs.

Fix: replace the fadvise loop with real reads. The sorted page list is
merged into contiguous spans and each span is `preadv`-ed into a scratch
buffer (data discarded, page cache populated). Volume is unchanged —
~19 MiB per chunk — but it moves to the background thread and runs at NVMe
sequential speed (~5 ms) instead of as dozens of synchronous 2 MiB folio
faults on the scheduler's critical path.
"""

import argparse
import os

OLD_ADVISE = '''    def _advise(self, pages: list[int]) -> None:
        for p in pages:
            try:
                os.posix_fadvise(
                    self._fd, p << _PAGE_SHIFT, 1 << _PAGE_SHIFT, os.POSIX_FADV_WILLNEED
                )
            except OSError:
                return'''

NEW_ADVISE = '''    _WARM_CHUNK_BYTES = 4 << 20

    def _advise(self, pages: list[int]) -> None:
        # gfx1151 patch 44: posix_fadvise(WILLNEED) is a no-op on this kernel
        # (measured: 2 GiB of this table, residency 30.1% -> 30.3%). Warm the
        # cache with real reads instead: merge the sorted page list into
        # contiguous spans and pread each span into a scratch buffer. The
        # bytes are discarded; only the page-cache population matters.
        buf = bytearray(self._WARM_CHUNK_BYTES)
        span_start = None
        span_end = None
        try:
            for p in pages:
                if span_start is None:
                    span_start = span_end = p
                elif p == span_end:
                    span_end += 1
                else:
                    self._read_span(span_start, span_end, buf)
                    span_start = span_end = p
            if span_start is not None:
                self._read_span(span_start, span_end, buf)
        except OSError:
            return

    def _read_span(self, start_page: int, end_page: int, buf: bytearray) -> None:
        """Read [start_page, end_page) into cache; the data is discarded."""
        off = start_page << _PAGE_SHIFT
        remaining = (end_page - start_page) << _PAGE_SHIFT
        while remaining > 0:
            n = min(remaining, self._WARM_CHUNK_BYTES)
            view = memoryview(buf)[:n]
            got = os.preadv(self._fd, [view], off)
            if got <= 0:
                return
            off += got
            remaining -= got'''

OLD_DOC = '''    """Hint the page cache about the rows a prefill-sized gather is about to read.

    With the table on storage, a cold prefill chunk faults tens of thousands of
    4 KiB pages one at a time from inside the gather kernel. Advising them
    first (``posix_fadvise(WILLNEED)`` per distinct page, on one background
    thread) lets the block layer serve them concurrently. Measured on a GB10 /
    NVMe: cold prefill 650-750 tok/s -> 1,000-2,100 tok/s (warm: ~2,200-2,600).
    Decode-sized gathers are skipped; nothing runs during CUDA-graph capture.
    """'''

NEW_DOC = '''    """Warm the page cache for the rows a prefill-sized gather is about to read.

    With the table on storage, a cold prefill chunk faults thousands of pages
    one at a time from inside the gather kernel. gfx1151 patch 44: on this
    kernel ``posix_fadvise(WILLNEED)`` is a no-op (measured 2026-11-21, twice),
    so the hint is delivered as real reads: the sorted page list is merged
    into contiguous spans and each span is pread into a scratch buffer on the
    background thread, populating the page cache. ~19 MiB per prefill chunk
    moves off the scheduler's critical path this way. Decode-sized gathers are
    skipped; nothing runs during CUDA-graph capture.
    """'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=".")
    args = parser.parse_args()
    path = os.path.abspath(os.path.expanduser(args.path))

    p = f"{path}/python/sglang/srt/models/qwen4_exp_ple_table.py"
    text = open(p).read()

    assert text.count(OLD_ADVISE) == 1, "PleFilePrefetcher._advise anchor not found"
    text = text.replace(OLD_ADVISE, NEW_ADVISE)
    assert text.count(OLD_DOC) == 1, "PleFilePrefetcher docstring anchor not found"
    text = text.replace(OLD_DOC, NEW_DOC)

    open(p, "w").write(text)
    print("patch_ple_warm_prefetch: ok")


if __name__ == "__main__":
    main()
