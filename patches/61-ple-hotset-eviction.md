# 61: Bound PLE hotset replacement and count actual fills

This fixes the host-RAM embedding row cache, not KV prefix caching. It changes
no serving flags, cache capacity, snapshot format or model weights.

At capacity, a burst of 10,000 distinct misses previously evicted 10,000
old rows even though the admission budget kept only 4,096 new ones. The
journal called all 10,000 misses fills. Even below budget, the SIEVE hand
could spare a visited row and evict its neighbor, only for the FIFO tail to
overwrite the spared row. One reported eviction then cost two cache hits.

The [patch](patch_ple_hotset_eviction.py) reserves reachable hash buckets
before evicting, consumes free slots first, then writes into exactly the slots
chosen by the hand. A fill can displace at most 4,096 residents. An in-place
refill of a lazy snapshot row needs no eviction. Hash-table rebuilds keep the
old index if the new one cannot represent every live row within the probe
bound. They do not move or discard payloads. Replacement also marks a full
cache eligible for its next scheduled keys-only snapshot even if its size
doesn't change.

**Reading the journal:** `fills` counts rows actually installed or refilled.
`ev` counts rows actually displaced. `hotset skipped: N` appears only when
the fill budget or hash probe limit prevented admission. The current gather
still gets all missing bytes from the backing table. Only the deferred copy
into the cache skips those rows. Counters are per report interval, so a
report containing multiple fills can still exceed 4,096 evictions.

**Regression coverage:** [the tests](../tests/check_ple_hotset_eviction.py)
exercise both original failures, partially full caches, wraparound, repeated
replacement, collisions at the real probe bound, lazy refills, snapshots,
payload bytes, and journal counters. They run against the freshly replayed
tree in the existing CPU CI job:

```sh
TMPDIR=/var/tmp python3 tests/check_patches.py --test-ple-hotset
```

Requires NumPy, not torch or a GPU. In the image,
`python3 tests/check_ple_hotset_eviction.py -v` can also run independently.

**Verified on CPU:** All seven new tests failed on the unchanged patch 57
image and passed through patch 61. Clean replay plus regressions passed
122/122 checks. The existing bit-exact CPU gather, lazy restore, page
residency and lock-on-fault checks also passed.

With 2,097,152 live 160-byte rows in a disposable CPU process, a burst of
11,668 misses produced 4,096 actual fills and evictions, 7,572 skipped,
and no loss of occupancy. Bursts of 32,768 and 262,144 misses also copied
and replaced only 4,096 rows; all admitted bytes matched the backing fixture.
These runs do not establish GPU performance or production hit rates.
