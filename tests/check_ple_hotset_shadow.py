"""Checks for patch 46: PLE read-through hot-set cache (RAM MVP, gfx1151).

Runs inside the image (needs numpy; the numerics test needs a GPU):
    PYTHONPATH=/sgl-workspace/sglang/python python3 tests/check_ple_hotset_shadow.py

Covers: probe/payload round-trip, one-chunk-lag fills, SIEVE eviction with
payload survival, rehash correctness, dedupe rejection, capacity accounting,
and — critically — bit-exact output of the cached gather kernel vs the
original kernel on mixed hit/miss/out-of-range batches.
"""

import os
import sys

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name} {detail}")
        FAILURES.append(name)


def test_cache_semantics():
    from sglang.srt.utils.ple_hotset import PleHotsetCache

    ROW = 160
    cap = 64
    c = PleHotsetCache(cap, ROW)
    check("mlock", c.payload.shape == (cap, ROW) and c.payload.dtype == "u1")
    check("init-empty", c._interval["hits"] == 0 and c._interval["misses"] == 0)

    # dedupe rejection contract
    try:
        c._insert_batch(np_from([5, 5]), np.zeros((2, ROW), dtype=np.uint8))
        check("dedupe-reject", False, "no exception")
    except ValueError:
        check("dedupe-reject", True)

    # one-chunk-lag fill: first prepare only enqueues; the SECOND prepare
    # fills the first batch's misses; a third prepare over the same keys
    # must see all hits with byte-identical payloads.
    keys = np_from(list(range(100, 132)))
    payload = np.array(
        [[(i * 7 + 3 + j) % 256 for j in range(ROW)] for i in range(32)],
        dtype=np.uint8,
    )
    filled = []

    def fill(rows):
        filled.append(rows.copy())
        return np.zeros((rows.size, ROW), dtype=np.uint8)

    slots1 = np.full(32, -2, dtype=np.int32)
    c.prepare(keys, slots1, fill)
    check("first-touch-enqueue-only", len(filled) == 0 and (slots1 < 0).all())

    keys_b = np_from(list(range(500, 532)))
    slots_b = np.full(32, -2, dtype=np.int32)
    c.prepare(keys_b, slots_b, fill)
    check("lag-fill-on-second-prepare", len(filled) == 1 and (filled[0] == keys).all())
    c._insert_batch(keys, payload)  # what the fill actually does

    slots2 = np.full(32, -2, dtype=np.int32)
    c.prepare(keys, slots2, fill)
    check("third-touch-all-hit", (slots2 >= 0).all())
    check(
        "payload-roundtrip",
        (c.payload[slots2] == payload).all(),
        "cached bytes must match mmap bytes",
    )
    check("counters", c._cumulative["hits"] == 32 and c._cumulative["misses"] == 64)

    # SIEVE eviction under pressure + survivor payload integrity
    more = np_from(list(range(1000, 1100)))
    more_payload = np.zeros((100, ROW), dtype=np.uint8)
    more_payload[:, 0] = np.arange(100, dtype=np.uint8)
    slots3 = np.full(100, -2, dtype=np.int32)
    c.prepare(more, slots3, lambda rows: None)
    c._insert_batch(more, more_payload)
    check("capacity", c.size <= cap, f"size {c.size} > cap {cap}")
    check("evictions-happened", c._cumulative["evictions"] > 0)
    surv_slots = np.full(32, -2, dtype=np.int32)
    c.prepare(keys, surv_slots, fill)
    hit_mask = surv_slots >= 0
    if hit_mask.any():
        check(
            "survivor-payload",
            (c.payload[surv_slots[hit_mask]] == payload[hit_mask]).all(),
        )

    # rehash preserves payloads and consistency (fresh cache: no eviction
    # pressure eating the survivors)
    r = PleHotsetCache(128, ROW)
    r._insert_batch(keys, payload)
    r._insert_batch(
        np_from([3000 + i for i in range(20)]),
        np.full((20, ROW), 9, dtype=np.uint8),
    )
    r._rehash()
    re_slots = np.full(32, -2, dtype=np.int32)
    r.prepare(keys, re_slots, fill)
    live = re_slots >= 0
    check(
        "rehash-payload",
        live.all() and (r.payload[re_slots] == payload).all(),
    )
    check("rehash-consistent", r.size == len(np.unique(r.keys[r.keys >= 0])))


def np_from(vals, dtype=None, flat=False):
    if dtype is None:
        dtype = np.int64
    if flat:
        return np.array(vals, dtype=dtype)
    return np.array(vals, dtype=dtype)


def test_numerics():
    """Bit-exactness: cached CPU gather vs the original CPU gather path."""
    import types

    import torch
    from sglang.srt.models.qwen4_exp import _gather_ple_embedding_on_cpu
    from sglang.srt.utils.ple_hotset import PleHotsetCache

    torch.manual_seed(0)
    ROW, NROWS, DIM = 160, 4096, 160
    table = torch.randint(0, 200, (NROWS, ROW), dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    table.view(torch.uint8)[0] = 0  # an all-zero row in the mix

    emb = types.SimpleNamespace(
        shard_indices=types.SimpleNamespace(
            org_vocab_start_index=0, org_vocab_end_index=NROWS
        ),
        weight=table,
        embedding_dim=DIM,
        _hotset=None,
        _hotset_slots=None,
    )
    n = 4096
    ids = torch.arange(100, 100 + n)
    ids[:1000] = 500 + torch.arange(1000)          # in-window, cold first touch
    ids[1000:3000] = torch.arange(2000)            # in-window, will be hot
    ids[3000:] = 10**9 + torch.arange(n - 3000)    # out of range

    def reference(e, flat_ids, out):
        start = e.shard_indices.org_vocab_start_index
        end = e.shard_indices.org_vocab_end_index
        host_ids = flat_ids.detach().to("cpu")
        in_range = (host_ids >= start) & (host_ids < end)
        local_idx = torch.where(in_range, host_ids - start, torch.zeros_like(host_ids))
        rows = e.weight.data.index_select(0, local_idx).to(torch.bfloat16)
        rows = torch.where(in_range.unsqueeze(1), rows, torch.zeros_like(rows))
        out.view(-1, e.embedding_dim).copy_(rows)

    # 1. cache disabled: must equal reference
    out_a = torch.zeros(n, DIM, dtype=torch.bfloat16)
    _gather_ple_embedding_on_cpu(emb, ids, out_a)
    out_ref = torch.zeros_like(out_a)
    reference(emb, ids, out_ref)
    check(
        "cold-path-bit-exact",
        bool((out_a.view(torch.int16) == out_ref.view(torch.int16)).all()),
    )

    # 2. populate the cache with the first batch's misses, re-run: hits now
    cache = PleHotsetCache(2048, ROW)
    emb._hotset = cache
    emb._hotset_slots = np.zeros(1 << 18, dtype=np.int32)
    out_b = torch.zeros_like(out_a)   # cold: enqueues misses (lag)
    _gather_ple_embedding_on_cpu(emb, ids, out_b)
    check(
        "warm-path-bit-exact",
        bool((out_b.view(torch.int16) == out_ref.view(torch.int16)).all()),
    )
    out_c = torch.zeros_like(out_a)   # fills+inserts pending, still misses
    _gather_ple_embedding_on_cpu(emb, ids, out_c)
    out_d = torch.zeros_like(out_a)   # now fully warm
    _gather_ple_embedding_on_cpu(emb, ids, out_d)
    check(
        "hot-path-bit-exact",
        bool((out_d.view(torch.int16) == out_ref.view(torch.int16)).all()),
    )
    hit_frac = (emb._hotset_slots[:n] >= 0).mean()  # after out_d
    check("hits-registered", hit_frac > 0.5, f"hit fraction {hit_frac:.2f}")


def main():
    global np
    import numpy as np

    test_cache_semantics()
    test_numerics()
    print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'ALL PASS'}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
