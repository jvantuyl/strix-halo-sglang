# 58: Distinguish mixed forwards in the journal

Before this patch, both pure prefill and mixed forwards printed `Prefill batch`.
`#running-req` counts running requests at admission, not the decode rows that
actually took part. Only a result whose forward mode is `MIXED` now prints:

```text
Mixed batch [42], prefill rows/positions: 1/1856, decode rows/positions: 3/3, total positions: 1859, #new-seq: 1, #new-token: 1856, ...
```

One row is one request in this forward; a chunk continuation counts as a
prefill row. Positions are model inputs for this forward, not prefix-cache
hits, generated tokens, accepted drafts, or shares of GPU time. The reporter
uses the result batch's decode membership and host-side extend lengths, which
remain available on the overlap copy. It does not inspect device tensors.

Pure prefill and decode keep their existing text. `#new-token`, `#cached-token`
and `input throughput` keep their meanings on mixed lines: the rate uses new
plus replayed prefill tokens over the time since the last prefill-bearing
report, not the duration of this forward. `bench/journal_stats.py` includes
both labels in the same historical series.

The optional event stream still emits `prefill_batch` and step kind `P`.
Only a mixed `prefill_batch` adds `batch_mode: "mixed"`, `prefill_rows`,
`decode_rows`, `prefill_positions`, `decode_positions`, and `total_positions`.
There is no scheduling, timing, or trace configuration change.
