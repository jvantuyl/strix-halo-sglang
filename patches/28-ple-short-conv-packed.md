# Patch 28: PLE short conv over the packed prefill batch

**Files:** `python/sglang/srt/models/qwen4_exp.py`
**Script:** [`patch_ple_short_conv_packed.py`](patch_ple_short_conv_packed.py)

## Symptom

The first run of [`tools/bench_qwen38.py`](../tools/bench_qwen38.py)
(DERISKED, MTP, cap 20, patches ≤ 27) took the box to **117 MiB of free
VRAM** in its mixed scenario, 18 short streams decoding while two
26k-token prompts prefill, and stayed there until the next `/flush_cache`.
The same scenario measured by hand earlier that hour had left 3.3 GiB. The
difference was the short prompts: the suite's 71-token story prompt versus
a 300-token synthetic paragraph. Nothing else. A single request of either
kind moved nothing; 20 story streams alone moved nothing; story shorts plus
a long prefill did it every time, in either order of the long prompts and
with one or two of them.

## Cause

`Qwen4ExpPLEGroupedNorm._short_conv` runs a causal depthwise conv1d
(kernel 4, dilation 3, 10,240 channels) over each PLE layer's input, with 9
carried state columns per request. For prefill it padded the packed batch
into `[requests, row_width, channels]` with `row_width` the longest
request in the batch, concatenated the state (second copy) and took the
conv output at the same size (third). That is O(requests × longest), not
O(tokens), and chunked prefill produces exactly the bad case: the scheduler
fills an 8,192-token chunk with all the short prompts it has plus as much
of the long one as fits. With 17 × 71-token prompts that is a 6,024-token
slice of the long prompt in an 18-row batch: 18 × 6,024 × 10,240 × 2 B =
2.1 GiB per copy, 6.3 GiB live, for 7,231 real tokens. The 300-token
prompts filled 5,580 of the chunk and left the long prompt a 2,612-token
row, 2.7 GiB, which happened to fit. At the 20-request cap with a full
8,192-token row the padded layout needs 9.6 GiB, more than the server has
free, so the same mix with slightly shorter prompts would have OOMed.

The `/start_profile {"activities": ["MEM"]}` snapshot stopped within a
second of the jump showed the three tensors at 2,115–2,118 MiB each on the
forward stream (`qwen4_exp.py:1102/1106/1107 _short_conv`). It was slow as
well as large: 714 ms per layer at that shape, which is where the 11 s
gaps between prefill batches in that run came from.

## Fix

`_packed_short_conv`: one `[channels, tokens + requests × 9]` row with
each request's state columns spliced in front of its tokens, one conv1d
over it, outputs read back at the token columns. The next-state and
track-slot gathers read the same row at each request's boundary, the
window the padded layout gathered. Padding rows (tokens past a request's
length, which the extend path can carry) go to a scratch column and get a
finite unused output. The target-verify path keeps the padded layout, its
row width is the draft length (4), and its intermediate-state unfold;
decode has its own fast path and is untouched. The conv is the same
depthwise op on the same values, so the output is bit-identical to the
padded layout's.

## Verification

`tools/test_qwen38_rocm.py` item 17: bit-identical conv output, next-state
and track gathers against the padded layout on five length mixes (1 to
6,024 tokens, 1 to 18 requests) in fp32 and bf16, padding rows leaving
valid outputs untouched, peak allocation at the real mixed shape 431 MiB
(padded: 6.2 GiB). Standalone timing at that shape: 11.9 ms vs 714 ms.

End to end (DERISKED, MTP, cap 20): greedy repeats identical with the same
accept histograms as patches 26–27; vision exact. The mixed scenario that
hit 117 MiB free now bottoms at 5.8 GiB (18 story streams + two 26k
prompts), and 18 streams + two 43k prompts at 5.8 GiB as well (3.3 before).
The rest of the numbers are in the runbook's MTP table.
