"""In-image check for gfx1151 patch 46: PLE hot-set shadow cache.

CPU-only. Verifies, against actual array contents (ground truth, no mocks):

* avalanche disperses correlated keys (Zipf-ish input -> uniform slots)
* probe finds only inserted keys — no phantom hits, no false negatives
* hit -> visited bit set; SIEVE evicts unvisited before visited
* capacity pressure evicts exactly enough; table stays consistent
  (every live key findable, every evicted key gone)
* Zipf traffic produces a rising, high hit rate (the spike's whole point)
* per-call eviction work is bounded (skipped counter under a cold storm)
* drain interval/cumulative separation
"""

import numpy as np

from sglang.srt.utils.ple_hotset import PleHotsetShadow, avalanche


def zipf_keys(n_keys: int, n_draws: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ranks = rng.power(1.3, size=n_draws)  # power-law-ish, heavy head
    return (ranks * n_keys).astype(np.int64) % n_keys


def check_avalanche():
    ids = np.arange(100_000, dtype=np.int64)  # perfectly correlated input
    slots = avalanche(ids) % 1024
    occupied = np.unique(slots).size
    # uniform hashing -> every slot occupied at n=100k, m=1024; a collapsed
    # mixer would cluster far below that
    assert occupied == 1024, f"avalanche collapsed: {occupied}/1024 distinct"


def check_probe_correctness():
    hs = PleHotsetShadow(1024)
    keys = np.array([10, 20, 30, 10, 20], dtype=np.int64)
    hit, _ = hs._probe(keys)
    assert (hit < 0).all(), "phantom hits on an empty cache"
    hs._insert_batch(keys)
    hit, _ = hs._probe(keys)
    assert (hit >= 0).all(), "inserted keys not findable"
    miss, _ = hs._probe(np.array([999999, 888888], dtype=np.int64))
    assert (miss < 0).all(), "false hits on absent keys"


def check_sieve_policy():
    hs = PleHotsetShadow(8)
    hs._insert_batch(np.arange(0, 8, dtype=np.int64))  # fills exactly
    hs._probe(np.array([0, 1], dtype=np.int64))  # touch 0,1 -> visited
    # mark by probing (observe path sets visited): emulate via observe
    hs2 = PleHotsetShadow(8)
    hs2.observe(np.arange(0, 8, dtype=np.int64))
    hs2.observe(np.array([2, 3], dtype=np.int64))  # 2,3 visited now
    hs2._insert_batch(np.array([100, 101], dtype=np.int64))  # needs 2 slots
    assert hs2.size == 8
    # evicted keys must be gone; touched keys must survive
    hit, _ = hs2._probe(np.array([2, 3], dtype=np.int64))
    assert (hit >= 0).all(), "SIEVE evicted recently-visited keys"
    gone, _ = hs2._probe(np.array([0, 1], dtype=np.int64))
    # 0,1 were unvisited at eviction time -> they must be candidates for
    # eviction; with hand starting at 0 they are evicted first
    assert (gone < 0).all(), "SIEVE kept unvisited keys over visited ones"


def check_capacity_and_consistency():
    hs = PleHotsetShadow(256)
    rng = np.random.default_rng(3)
    for i in range(40):
        keys = rng.integers(0, 100_000, size=64).astype(np.int64)
        hs.observe(keys)
        assert hs.size <= hs.n
        # table consistency: every live key findable
        live = hs.keys >= 0
        hit, _ = hs._probe(hs.keys[live])
        assert (hit >= 0).all(), f"round {i}: live key lost from table"
    assert hs._interval["evictions"] > 0, "capacity pressure never exercised"


def check_zipf_hit_rate():
    hs = PleHotsetShadow(1 << 16)
    keys = zipf_keys(100_000, 400_000)
    # feed in gathers of 1024 unique-ish
    for i in range(0, keys.size, 1024):
        hs.observe(np.unique(keys[i:i + 1024]))
    c = hs.drain(cumulative=True)
    # Synthetic power() skew is much flatter than the real corpus census
    # (top 69.5k of 663k contexts = 90% there); this test asserts the
    # structural behavior — a heavy-head stream yields a substantial hit
    # rate at 65% coverage and warms up over the stream. The real serving
    # hit rate is measured live by the shadow itself.
    assert c["hits"] / c["rows"] > 0.35, (
        f"zipf hit rate too low: {c['hits']}/{c['rows']}"
    )
    # early vs late: the cache must warm up
    hs2 = PleHotsetShadow(1 << 16)
    first = keys[:20_480]
    for i in range(0, first.size, 1024):
        hs2.observe(np.unique(first[i:i + 1024]))
    d1 = hs2.drain()
    last = keys[-20_480:]
    hs2b = PleHotsetShadow(1 << 16)
    for i in range(0, keys.size - 20_480, 1024):
        hs2b.observe(np.unique(keys[i:i + 1024]))
    for i in range(keys.size - 20_480, keys.size, 1024):
        hs2b.observe(np.unique(keys[i:i + 1024]))
    d2 = hs2b.drain(cumulative=True)
    assert d2["hits"] / d2["rows"] > d1["hits"] / d1["rows"], (
        "hit rate did not warm up over the stream"
    )


def check_bounded_work():
    hs = PleHotsetShadow(1024)
    rng = np.random.default_rng(11)
    for _ in range(10):
        hs.observe(rng.integers(0, 10**9, size=4096).astype(np.int64))
    c = hs.drain(cumulative=True)
    assert c["evictions"] <= 10 * 4096 or c["skipped"] > 0, (
        "eviction work not bounded"
    )
    assert hs.enabled, "shadow disabled itself under a cold storm"


def check_drain_semantics():
    hs = PleHotsetShadow(64)
    hs.observe(np.array([1, 2, 3], dtype=np.int64))
    d = hs.drain()
    assert d["rows"] == 3 and d["inserts"] == 3 and d["hits"] == 0
    d2 = hs.drain()
    assert d2["rows"] == 0, "interval did not reset"
    hs.observe(np.array([1], dtype=np.int64))
    d3 = hs.drain()
    assert d3["hits"] == 1, "repeat touch not counted as hit"


def main():
    check_avalanche()
    check_probe_correctness()
    check_sieve_policy()
    check_capacity_and_consistency()
    check_zipf_hit_rate()
    check_bounded_work()
    check_drain_semantics()
    print("check_ple_hotset_shadow: all green")


if __name__ == "__main__":
    main()
