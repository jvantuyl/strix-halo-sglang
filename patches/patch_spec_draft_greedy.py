#!/usr/bin/env python3
"""gfx1151 patch 22: greedy requests get a greedy first draft under rejection
sampling.

On HIP upstream turns `speculative_use_rejection_sampling` on by default for
EAGLE/NEXTN (its greedy verify path is CUDA-only). Under rejection sampling
every per-step draft goes through `sample_draft_proposal`, which draws a
Gumbel-max sample and then, for rows with `top_k <= 1` (what `temperature=0`
becomes), replaces it with the argmax. The draft pass that runs right after
the target's prefill (`_draft_extend_for_prefill`) does not: it calls
`fast_sample` directly, so the first draft token of every request is a random
sample even at temperature 0. The target still commits its own argmax, but a
randomly accepted or rejected first draft changes how the following verify
steps are batched, and with it the rounding of every later logit: greedy
output under MTP differed from run to run (logprobs from the first few
positions, a token flip after ~80) while the same image without MTP was
bit-identical. It also throws away accept length on the first verify.

The fix routes that call through `sample_draft_proposal` with the batch's
temperatures and `top_ks`, exactly as the per-step path does; the non
rejection-sampling branch is untouched. Anchors on upstream text.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/speculative/eagle_worker_v2.py"
text = open(p).read()

old = """        # Assemble the next-iter draft spec_info from the extend output.
        use_rejection_sampling = get_spec().speculative_use_rejection_sampling
        probs = renorm_draft_probs(
            logits_output.next_token_logits,
            batch.sampling_info,
            use_rejection_sampling,
        )
        if use_rejection_sampling:
            topk_p, topk_index = fast_sample(probs, num_samples=1)
        else:
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
"""
assert text.count(old) == 1, "eagle_worker_v2.py: _draft_extend_for_prefill sampling anchor not found"
new = """        # Assemble the next-iter draft spec_info from the extend output.
        use_rejection_sampling = get_spec().speculative_use_rejection_sampling
        if use_rejection_sampling:
            # gfx1151 patch 22: same proposal as the per-step drafts, so a
            # greedy row (top_k <= 1) gets its argmax instead of a random
            # Gumbel-max draw for the first draft token.
            probs, topk_p, topk_index = sample_draft_proposal(
                logits_output.next_token_logits,
                batch.sampling_info.temperatures,
                batch.sampling_info.top_ks,
            )
        else:
            probs = renorm_draft_probs(
                logits_output.next_token_logits,
                batch.sampling_info,
                use_rejection_sampling,
            )
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
"""
text = text.replace(old, new, 1)

assert "sample_draft_proposal," in text, "eagle_worker_v2.py: sample_draft_proposal import not found"

open(p, "w").write(text)
print("patched", p)
print("patch 22 (spec draft greedy) applied")
