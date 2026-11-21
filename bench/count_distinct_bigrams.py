"""Measure the PLE working set: distinct token bigrams AND their frequency
distribution over real corpus text.

The PLE table indexes 2-token contexts (ngram_size=3) across 16 hash heads;
its realistic working set is bounded by the distinct consecutive token pairs
the workload produces. This counts them over the compiled preference-pair
corpus using the model's own tokenizer AND reports the rank-frequency
coverage curve (how many distinct bigrams cover 50/80/90/95/99% of all
bigram occurrences) — the number that sizes a read-through LRU under a
power-law assumption, not just a total.

Memory: accumulates every bigram occurrence as one int64 (25M occurrences ≈
200 MiB) plus unigrams; no incremental dedup, so occurrence counts survive.
"""

import json
import sys

import numpy as np
from transformers import AutoTokenizer

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "/models/corpus/compiled_corpus.jsonl"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "/models/qwen38"

tok = AutoTokenizer.from_pretrained(MODEL)
uni = []
bi = []
total = 0
batch_texts = []


def flush(texts):
    global total
    if not texts:
        return
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    for ids in enc:
        if len(ids) < 2:
            continue
        a = np.asarray(ids, dtype=np.int64)
        total += len(a)
        uni.append(a)
        bi.append(a[:-1] * (1 << 24) + a[1:])


def coverage_curve(counts):
    """counts sorted descending -> distinct items needed to cover X% of all."""
    total_occ = counts.sum()
    cum = np.cumsum(counts)
    out = {}
    for pct in (50, 80, 90, 95, 99):
        k = int(np.searchsorted(cum, total_occ * pct / 100.0) + 1)
        out[pct] = k
    return out


def report(name, arr):
    vals, counts = np.unique(arr, return_counts=True)
    order = np.argsort(counts)[::-1]
    sorted_counts = counts[order]
    cov = coverage_curve(sorted_counts)
    top1m = int(sorted_counts[:1_000_000].sum())
    top10m = int(sorted_counts[:10_000_000].sum())
    print(f"== {name}")
    print(f"distinct: {vals.size:,}")
    print(f"occurrences: {int(counts.sum()):,}")
    for pct, k in cov.items():
        print(f"  top {k:,} items cover {pct}% of occurrences "
              f"({k * 160 / 2**20:.1f} MiB at 160 B/row)")
    print(f"  top 1M items: {top1m / counts.sum() * 100:.1f}% of occurrences")
    if vals.size > 10_000_000:
        print(f"  top 10M items: {top10m / counts.sum() * 100:.1f}% of occurrences")
    # head histogram: counts bucketed by rank decade
    for lo, hi in ((1, 100), (100, 1_000), (1_000, 10_000), (10_000, 100_000),
                   (100_000, 1_000_000), (1_000_000, 10_000_000),
                   (10_000_000, 10**18)):
        if lo >= vals.size:
            break
        seg = sorted_counts[lo:hi]
        if seg.size:
            print(f"  ranks {lo}-{min(hi, vals.size):,}: "
                  f"{int(seg.sum()) / counts.sum() * 100:.1f}% of occurrences "
                  f"(mean count {seg.mean():.2f})")


with open(CORPUS) as f:
    for line in f:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("chosen") or ""
        p = d.get("prompt") or ""
        if t:
            batch_texts.append(t)
        if p:
            batch_texts.append(p)
        if len(batch_texts) >= 512:
            flush(batch_texts)
            batch_texts = []
            print(f"  ... {total/1e6:.1f}M tokens", flush=True)
flush(batch_texts)

all_uni = np.concatenate(uni)
all_bi = np.concatenate(bi)
del uni, bi
report("unigrams", all_uni)
del all_uni
report("bigrams (the PLE rows)", all_bi)
print(f"RESULT total_tokens={total}")
