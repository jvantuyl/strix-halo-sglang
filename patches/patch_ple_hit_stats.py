"""gfx1151 patch 45: report PLE row/page-cache hit stats on the status line.

The PLE table (47.7 GiB fp8) is read through a file-backed mmap; whether a
gather is cheap depends on how many of its rows' pages are already resident
in the page cache. Until now that was invisible: we measured it out-of-band
with sudo mincore over /proc/<pid>/smaps ranges. This patch makes the server
report it itself, on the Prefill batch status line, from kernel ground truth:

  PLE: <unique>/<rows> rows, <resident>/<pages> pages res, <errors> err

* rows       — row reads this prefill batch asked of the table (with the
               160 B/row n-gram window, one row per token plus window)
* rows/rows_unique — duplication inside the batch (sorted-table locality)
* pages      — distinct 4 KiB pages those rows touch (rows share pages)
* resident   — of those pages, how many mincore() reports PAGE-CACHE
               RESIDENT right now. This is the hit prediction: a resident
               page is a minor fault, a non-resident page is disk I/O.
* err        — stats-recording errors seen since boot (recording disables
               itself after the first one; diagnostics must never take the
               server down).

Counters are per-report-interval (drained by the metrics reporter) plus
cumulative-since-boot exposed via drain(cumulative=True).

Cost: one ids D2H copy + one torch.unique + <= ~1100 one-page mincore
syscalls per prefill chunk (pages_for_rows is sorted-unique; random rows do
not merge). Prefill chunks are ~1 s; the measured gather fault path already
costs ~5% of the chunk. Decode CUDA-graph replay never reaches this code
(gather() is not re-entered during replay), so decode latency is untouched.

Ground-truth rules: residency comes from mincore(2) on the mapping itself —
no models, no estimates. The interval counters are drained by the same
metrics reporter that prints the other server numbers.
"""

import argparse
import os
import sys

STATS_MODULE = '''"""gfx1151 patch 45: PLE gather hit statistics (kernel ground truth).

Counters are recorded on every non-captured PLE gather of prefill size and
drained by the scheduler's metrics reporter onto the Prefill batch status
line. Residency is measured with mincore(2) against the table mapping —
what the kernel reports, nothing derived.
"""

import ctypes
import threading

_PAGE_SHIFT = 12
_PAGE_SIZE = 1 << _PAGE_SHIFT

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


class PleHitStats:
    """Interval + cumulative counters for PLE gathers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._interval = self._zero()
        self._cumulative = self._zero()
        self.errors = 0
        self.enabled = True

    @staticmethod
    def _zero():
        return {
            "gathers": 0,
            "rows": 0,
            "rows_unique": 0,
            "pages": 0,
            "pages_resident": 0,
        }

    def record(
        self,
        rows: int,
        rows_unique: int,
        pages: int,
        pages_resident: int,
    ) -> None:
        with self._lock:
            self._interval["gathers"] += 1
            self._interval["rows"] += rows
            self._interval["rows_unique"] += rows_unique
            self._interval["pages"] += pages
            self._interval["pages_resident"] += pages_resident
            self._cumulative["gathers"] += 1
            self._cumulative["rows"] += rows
            self._cumulative["rows_unique"] += rows_unique
            self._cumulative["pages"] += pages
            self._cumulative["pages_resident"] += pages_resident

    def drain(self, cumulative: bool = False):
        """Return counters and reset the interval window."""
        with self._lock:
            if cumulative:
                return dict(self._cumulative)
            out = dict(self._interval)
            self._interval = self._zero()
            out["errors"] = self.errors
            return out

    def note_error(self) -> None:
        with self._lock:
            self.errors += 1


PLE_HIT_STATS = PleHitStats()


def pages_for_rows(row_ids, row_bytes: int):
    """Distinct 4 KiB pages touched by ``row_ids`` (0-based row indexes).

    ``row_ids``: 1-D integer CPU tensor (or anything torch.unique takes).
    Returns a sorted list of page indexes.
    """
    import torch

    starts = row_ids.to(torch.int64) * row_bytes
    ends = starts + (row_bytes - 1)
    return (
        torch.cat([starts >> _PAGE_SHIFT, ends >> _PAGE_SHIFT])
        .unique(sorted=True)
        .tolist()
    )


def mincore_resident(base_addr: int, pages) -> int:
    """Count of ``pages`` (page indexes above ``base_addr``) resident now.

    One mincore(2) syscall per page: the queried range is a single page, so
    the 1-byte vector fits in a stack buffer. Raises on the first kernel
    error; the caller is expected to disable stats on failure.
    """
    count = 0
    vec = (ctypes.c_ubyte * 1)()
    for p in pages:
        addr = base_addr + (p << _PAGE_SHIFT)
        if _libc.mincore(
            ctypes.c_void_p(addr), ctypes.c_size_t(_PAGE_SIZE), vec
        ) != 0:
            raise OSError(ctypes.get_errno(), "mincore failed")
        count += 1 if vec[0] & 1 else 0
    return count
'''

QWEN4_EXP_IMPORT_ANCHOR = """    make_ple_file_prefetcher,
    make_ple_file_rss_trimmer,
)"""

QWEN4_EXP_IMPORT_NEW = QWEN4_EXP_IMPORT_ANCHOR + """
from sglang.srt.utils.ple_hit_stats import (
    PLE_HIT_STATS,
    mincore_resident,
    pages_for_rows as _ple_pages_for_rows,
)"""

GATHER_ANCHOR = """        if flat_ids.numel():
            if self._file_prefetcher is not None:
                self._file_prefetcher.enqueue(
                    flat_ids,
                    vocab_start=self.shard_indices.org_vocab_start_index,
                    vocab_end=self.shard_indices.org_vocab_end_index,
                )"""

GATHER_NEW = GATHER_ANCHOR + """
            _ple_record_gather_stats(self, flat_ids)"""

HELPER_ANCHOR = """def _gather_ple_embedding_from_pinned_kernel("""

HELPER_NEW = '''def _ple_record_gather_stats(embedding, flat_ids) -> None:
    """gfx1151 patch 45: count rows/pages and mincore their residency.

    Prefill-sized gathers only (CUDA-graph replay never re-enters here; tiny
    eager gathers are skipped to keep the syscall cost proportional). Any
    error disables recording for the process — diagnostics must never be
    able to take the server down.
    """
    stats = PLE_HIT_STATS
    if not stats.enabled or flat_ids.numel() < 512:
        return
    if flat_ids.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    try:
        vocab_start = embedding.shard_indices.org_vocab_start_index
        vocab_end = embedding.shard_indices.org_vocab_end_index
        ids = flat_ids.detach().cpu()
        ids = ids[(ids >= vocab_start) & (ids < vocab_end)] - vocab_start
        rows = int(ids.numel())
        if rows == 0:
            return
        uniq = ids.unique()
        row_bytes = embedding.weight.element_size() * embedding.weight.shape[-1]
        pages = _ple_pages_for_rows(uniq, row_bytes)
        resident = mincore_resident(embedding.weight.data_ptr(), pages)
        stats.record(rows, int(uniq.numel()), len(pages), resident)
    except Exception:
        stats.enabled = False
        stats.note_error()


def _gather_ple_embedding_from_pinned_kernel('''

REPORTER_MODULE_ANCHOR = "LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()"

REPORTER_PREFILL_ANCHOR = '''        msg += f"{self._graph_backend_label}: {can_run_cuda_graph}, "
        msg += f"input throughput (token/s): {self.last_input_throughput:.2f}"'''

REPORTER_PREFILL_NEW = '''        msg += f"{self._graph_backend_label}: {can_run_cuda_graph}, "
        msg += f"input throughput (token/s): {self.last_input_throughput:.2f}"
        msg += _ple_stats_msg_suffix()'''


def _build_reporter_suffix_block():
    return '''_PLE_STATS_MSG: str = ""


def _ple_stats_msg_suffix() -> str:
    """gfx1151 patch 45: drain the PLE gather counters onto the status line."""
    try:
        from sglang.srt.utils.ple_hit_stats import PLE_HIT_STATS

        s = PLE_HIT_STATS.drain()
        if not s["gathers"]:
            return ""
        return (
            f", PLE: {s['rows_unique']}/{s['rows']} rows, "
            f"{s['pages_resident']}/{s['pages']} pages res, "
            f"{s['errors']} err"
        )
    except Exception:
        return ""
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/sgl-workspace/sglang",
                    help="sglang source tree root (contains python/)")
    args = ap.parse_args()

    root = os.path.join(args.target, "python", "sglang", "srt")
    if not os.path.isdir(root):
        sys.exit(f"not an sglang tree: {args.target}")

    # 1. New module: sglang/srt/utils/ple_hit_stats.py
    stats_path = os.path.join(root, "utils", "ple_hit_stats.py")
    assert not os.path.exists(stats_path), (
        f"ple_hit_stats.py already exists at {stats_path}; refusing to clobber"
    )
    with open(stats_path, "w") as f:
        f.write(STATS_MODULE)

    # 2. qwen4_exp.py: import + gather hook + helper
    qwen = os.path.join(root, "models", "qwen4_exp.py")
    with open(qwen) as f:
        text = f.read()
    assert QWEN4_EXP_IMPORT_ANCHOR in text, "qwen4_exp.py import anchor moved"
    assert GATHER_ANCHOR in text, "qwen4_exp.py gather anchor moved"
    assert HELPER_ANCHOR in text, "qwen4_exp.py helper anchor moved"
    assert "ple_hit_stats" not in text, "patch 45 already applied?"
    assert text.count(QWEN4_EXP_IMPORT_ANCHOR) == 1, (
        "qwen4_exp.py import tail anchor not unique"
    )
    text = text.replace(QWEN4_EXP_IMPORT_ANCHOR, QWEN4_EXP_IMPORT_NEW, 1)
    text = text.replace(GATHER_ANCHOR, GATHER_NEW, 1)
    text = text.replace(HELPER_ANCHOR, HELPER_NEW, 1)
    with open(qwen, "w") as f:
        f.write(text)

    # 3. metrics_reporter.py: suffix helper + prefill msg append
    rep = os.path.join(root, "managers", "scheduler_components", "metrics_reporter.py")
    with open(rep) as f:
        rtext = f.read()
    assert REPORTER_MODULE_ANCHOR in rtext, "metrics_reporter module anchor moved"
    assert REPORTER_PREFILL_ANCHOR in rtext, "metrics_reporter anchor moved"
    assert "ple_stats_msg_suffix" not in rtext, "patch 45 already applied?"
    rtext = rtext.replace(
        REPORTER_MODULE_ANCHOR,
        REPORTER_MODULE_ANCHOR + "\n\n" + _build_reporter_suffix_block().rstrip(),
        1,
    )
    with open(rep, "w") as f:
        f.write(rtext)

    print("patch 45 applied: PLE hit stats on the Prefill status line")


if __name__ == "__main__":
    main()
