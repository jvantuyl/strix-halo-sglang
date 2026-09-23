# Patch 22: greedy first draft under rejection sampling

**Files:** `python/sglang/srt/speculative/eagle_worker_v2.py`
**Script:** [`patch_spec_draft_greedy.py`](patch_spec_draft_greedy.py)

## Symptom

With patches 18, 20 and 21 the DERISKED checkpoint decodes bit-identically
across cold runs without MTP. With `--speculative-algorithm NEXTN` (3 steps,
4 draft tokens) the same greedy request was not repeatable: the same tokens
for ~80 positions but different top-1 logprobs from position 6 or 7 on
(differences in the 4th decimal, a handful of recurring values), a token
flip after ~80 tokens in some runs, and `spec_verify_ct` alternating 40 / 41
/ 42 with a different accept histogram each run. `--disable-overlap-schedule`
and `--disable-cuda-graph` changed nothing.

A module-by-module checksum trace of the scheduler's target and draft
forwards (GPU-side sums, no host copies) showed target prefill and the draft
extend identical across runs, and the first difference at the draft
model's `embed_tokens` input of the first draft decode step: the first
draft token itself differed (`Also` vs `In`) while the draft-extend logits
that produced it were identical. Selection, not arithmetic.

## Cause

On HIP upstream turns `speculative_use_rejection_sampling` on by default for
EAGLE/NEXTN (log line: "ROCm needs rejection sampling for EAGLE spec-decode
to sample at all"; its greedy verify kernel is CUDA-only). Under rejection
sampling each per-step draft goes through `sample_draft_proposal`, which
draws a Gumbel-max sample and then, for rows with `top_k <= 1` (what
`temperature=0` is rewritten to), replaces it with the argmax; its docstring
explains why.

The draft pass that runs right after the target's prefill,
`_draft_extend_for_prefill`, does not go through it: it calls `fast_sample`
directly on the renormalised draft probabilities, so the first draft token of
every request is a random draw even at temperature 0. The target still
commits its own argmax, so the output is "greedy" in the sense that no
sampled token is ever emitted, but whether that first draft is accepted or
rejected changes how the following verify steps are batched and, through the
batch shape, the rounding of every later logit. It also gives away accept
length on the first verify (a sharp-but-not-one-hot draft distribution
proposes a non-argmax token often; the parity test's `fast_sample` misses the
argmax in 30/30 draws on such rows).

## Fix

In the rejection-sampling branch of `_draft_extend_for_prefill`, call
`sample_draft_proposal(logits, batch.sampling_info.temperatures,
batch.sampling_info.top_ks)`, exactly as the per-step draft
(`draft_forward`) and the decode-side draft extend already do, and use the
`probs` it returns. The non-rejection-sampling branch (`fast_topk` over the
renormalised probabilities) is unchanged. Not gfx1151-specific: any ROCm
build, and any CUDA build that opts into rejection sampling, hits the same
path.

## Verification

`tools/test_qwen38_rocm.py` item 11: on a sharp distribution with a
runner-up 1.5 nats below the top, `sample_draft_proposal` returns the argmax
for `top_k <= 1` rows, its probability, and is bit-identical 30/30, while
`fast_sample` on the same rows misses the argmax 30/30; the
rejection-sampling branch of `_draft_extend_for_prefill` calls
`sample_draft_proposal` and no longer calls `fast_sample`.

End to end (DERISKED, MTP 3 steps / 4 draft tokens, every run cold): a
14k-token prompt went from different to 3/3 bit-identical at 64 tokens with
this patch alone. Short prompts still differed (first logprob difference
moved from position 6 to 7 or 88); that is patch 23. With both, see
[23-qsa-mtp-tail.md](23-qsa-mtp-tail.md).
