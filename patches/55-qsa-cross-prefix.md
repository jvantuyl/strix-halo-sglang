# 55 — Complete QSA groups across cached extend prefixes

Backports the cross-prefix algorithm from SGLang
[#39575](https://github.com/sgl-project/sglang/pull/39575) (closed, unmerged;
head [`cf4b0c1b`](https://github.com/sgl-project/sglang/commit/cf4b0c1bd656cd3d8c97d6d3944fd06fe95796fe))
to pin `70b5b03e`. [#38180](https://github.com/sgl-project/sglang/pull/38180)
reports the ratio-4 write-plan assert on mixed-chunk prefill.

Page-granular radix hits are ratio-aligned, but an eager `ForwardMode.MIXED`
batch appends running decode rows with a one-token extend and arbitrary prefix
lengths. An ordinary `EXTEND` can also retain a partial private chunk. If a
group completes across this boundary its prefix keys are in the request's
pending ring, not in the packed rows of this forward.

- Gate the cross-prefix plan on host prefix and extend lengths for real eager
  `EXTEND` or `MIXED`, not the upstream PR's gfx950-only exact-chunk-fill gate.
  An unaligned row that completes no group only updates its pending tail.
  Missing mirrors and other non-paged modes retain the alignment assert;
  ordinary aligned, decode/speculative-paged and graph paths do not build the
  cross-prefix plan.
- Clamp the main gather to the owning request's current rows, then correct only
  the first compacted group per request using its ring members and the
  group-start RoPE. Empty and non-crossing rows write only reserved slot 0 in
  this correction. Use the normal fused or fallback compression path for both.
- Snapshot each layer's pending keys before projection can overwrite the ring.
  The RoPE sidecar is shared across layers, so snapshot it once on the compute
  stream before the first layer writes, and reuse it for the rest of the
  forward. The new pending tail and its RoPE remain untouched by recompression.

`tests/check_qsa_cross_prefix.py` exercises the patched backend and indexer
methods with CPU tensors. It checks the fallback and emulates the fused
store to verify its input layout and ordering, not the GPU kernel itself.
