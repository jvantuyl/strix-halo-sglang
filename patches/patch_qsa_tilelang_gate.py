#!/usr/bin/env python3
"""gfx1151 patch 48: env gate for the TileLang QSA MQA kernels.

The ROCm 10 base's tilelang cannot compile the QSA indexer decode GEMM for
our geometry at graph capture ("N must be divisible by 16, but got 8" from
GemmWarpPolicyNode::computeWarpPartition) and takes the scheduler down. The
Triton MQA decode/prefill kernels (gfx1151 patches 24/26) are the validated
stand-ins. This gate defaults to TileLang (status quo on the 7.13 base);
SGLANG_QSA_TILELANG=0 selects the Triton path on stacks where TileLang is
broken. A failed TileLang compile at import time already falls back via
HAS_TILELANG; this gate covers the worse case: import succeeds, capture dies.
"""

import argparse
import os
import sys

TARGET = "python/sglang/srt/layers/attention/qsa/mqa.py"

HELPER_ANCHOR = '''def qsa_mqa_triton_allowed() -> bool:'''

HELPER_NEW = '''def qsa_tilelang_allowed() -> bool:
    """gfx1151 patch 48: SGLANG_QSA_TILELANG=0 stands down the TileLang MQA
    kernels for the validated Triton path (patches 24/26). Default on — the
    7.13 base's TileLang is the tuned path; the ROCm 10 base's tilelang
    cannot compile the decode GEMM for our geometry and dies at capture."""
    return os.environ.get("SGLANG_QSA_TILELANG", "1") != "0"


def qsa_mqa_triton_allowed() -> bool:'''

PREFILL_OLD = '''    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)'''
PREFILL_NEW = '''    if q.is_cuda and HAS_TILELANG and qsa_tilelang_allowed():
        return tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)'''

DECODE_OLD = '''    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, max_model_len, score_scale
        )'''
DECODE_NEW = '''    if q.is_cuda and HAS_TILELANG and qsa_tilelang_allowed():
        return tilelang_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, max_model_len, score_scale
        )'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/sgl-workspace/sglang")
    args = ap.parse_args()
    path = os.path.join(args.target, TARGET)
    with open(path) as f:
        text = f.read()
    if 'gfx1151 patch 48' in text:
        sys.exit("patch 48 already applied")
    assert text.count(HELPER_ANCHOR) == 1, "triton_allowed anchor not found"
    assert text.count(PREFILL_OLD) == 1, "prefill dispatch anchor not found"
    assert text.count(DECODE_OLD) == 1, "decode dispatch anchor not found"
    text = text.replace(HELPER_ANCHOR, HELPER_NEW, 1)
    text = text.replace(PREFILL_OLD, PREFILL_NEW, 1)
    text = text.replace(DECODE_OLD, DECODE_NEW, 1)
    with open(path, "w") as f:
        f.write(text)
    print("patch 48 applied: SGLANG_QSA_TILELANG gate")


if __name__ == "__main__":
    main()
