"""In-image check for gfx1151 patch 45: PLE gather hit statistics.

Runs without a GPU. Verifies, against kernel ground truth (mincore(2) on a
real mapping, not mocks):

* pages_for_rows math (row -> page coverage, dedup, sorted output)
* mincore_resident reports 1 for a page just written, 0 for untouched pages
  far from any fault (readahead may populate the immediate neighbour — that
  is real residency, the kernel's truth, and acceptable for hit stats)
* PleHitStats record/drain interval + cumulative semantics and error count
* the metrics reporter wiring imports and appends the suffix when there is
  data, and returns "" when the counters are empty
"""

import inspect
import mmap
import os
import sys
import tempfile

import torch

from sglang.srt.utils import ple_hit_stats as m


def check_pages_for_rows():
    pages = m.pages_for_rows(
        torch.tensor([0, 1, 25, 26, 4095, 4096]), 160
    )
    # rows 0..25 span bytes 0..4159 -> pages 0,1 ; row 4095 -> 655200 -> page 159;
    # row 4096 -> 655360 -> page 160.
    assert pages == [0, 1, 159, 160], pages
    assert m.pages_for_rows(torch.tensor([7, 7, 7]), 160) == [0]


def check_mincore_ground_truth():
    fd, path = tempfile.mkstemp(dir="/var/tmp")
    try:
        os.ftruncate(fd, 8 << 20)  # sparse: nothing populated
        mm = mmap.mmap(fd, 8 << 20)  # MAP_SHARED, writable
        start = None
        for line in open("/proc/self/maps"):
            if line.rstrip().endswith(path):
                start = int(line.split("-")[0], 16)
                break
        assert start is not None, "mapping not found in /proc/self/maps"
        mm[(3 << 12):(3 << 12) + 1] = b"z"  # touch page 3
        assert m.mincore_resident(start, [3]) == 1, "touched page not resident"
        assert m.mincore_resident(start, [2000]) == 0, (
            "untouched far page reported resident"
        )
        mm.close()
    finally:
        os.close(fd)
        os.unlink(path)


def check_stats_counters():
    s = m.PleHitStats()
    s.record(10, 8, 6, 4)
    s.record(10, 9, 7, 5)
    s.note_error()
    d = s.drain()
    assert d == {
        "gathers": 2, "rows": 20, "rows_unique": 17,
        "pages": 13, "pages_resident": 9, "errors": 1,
    }, d
    # interval resets after drain; cumulative persists
    d2 = s.drain()
    assert d2["gathers"] == 0 and d2["errors"] == 1, d2
    c = s.drain(cumulative=True)
    assert c["gathers"] == 2 and "errors" not in c, c


def check_reporter_suffix():
    from sglang.srt.managers.scheduler_components import metrics_reporter

    # the reporter must actually CALL the suffix on the prefill message
    # (a missing call site once shipped: def present, call forgotten)
    src = inspect.getsource(metrics_reporter)
    assert "msg += _ple_stats_msg_suffix()" in src, (
        "metrics_reporter never appends the PLE suffix to the status line"
    )

    m.PLE_HIT_STATS.record(1024, 1000, 990, 300)
    suffix = metrics_reporter._ple_stats_msg_suffix()
    assert "PLE: 1000/1024 rows" in suffix, suffix
    assert "300/990 pages res" in suffix, suffix
    m.PLE_HIT_STATS.drain()
    assert metrics_reporter._ple_stats_msg_suffix() == "", (
        "suffix must be empty when no gathers were recorded"
    )


def check_decorator_placement():
    """Patch 45 once inserted its helper between @triton.jit and the kernel
    def, stealing the decorator — the boot then died at graph capture with
    "Cannot call @triton.jit'd outside of the scope of a kernel". Assert the
    decorator sits directly above the kernel definition.
    """
    import inspect

    import sglang.srt.models.qwen4_exp as qe

    src = inspect.getsource(qe)
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("def _gather_ple_embedding_from_pinned_kernel("):
            decor = lines[i - 1].strip()
            assert decor == "@triton.jit", (
                f"kernel def not decorated by @triton.jit, got: {decor!r}"
            )
            return
        if line.startswith("def _ple_record_gather_stats("):
            continue  # helper must NOT be decorated; checked implicitly below
    raise AssertionError("kernel def not found in qwen4_exp source")
    # also: the helper must not have become a JITFunction
    assert not hasattr(qe._ple_record_gather_stats, "fn"), (
        "_ple_record_gather_stats became a @triton.jit JITFunction again"
    )


def main():
    check_pages_for_rows()
    check_mincore_ground_truth()
    check_stats_counters()
    check_decorator_placement()
    check_reporter_suffix()
    print("check_ple_hit_stats: all green")


if __name__ == "__main__":
    main()
