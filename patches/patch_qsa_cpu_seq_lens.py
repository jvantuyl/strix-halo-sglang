#!/usr/bin/env python3
"""gfx1151 patch 37: QSA declares needs_cpu_seq_lens so spec forwards stop
reading the token-slot bound back from the device every forward.

`decide_needs_cpu_seq_lens()` (managers/overlap_utils.py) is an OR over the
attention backends of the spec-v2 chain and gates whether the overlap
FutureMap publishes seq_lens_cpu/sum to the next round's batches. The QSA
backend declared False (mirroring upstream's DSA backend, which can afford
the opt-out because its metadata path is fully device-side), so with MTP on
the OR was False and the speculative ForwardBatches arrived with
seq_lens_cpu=None. `_speculative_max_row_length()` — the host-side over-bound
for the token-slot gather width, used for target-verify and draft-extend —
then took its documented fallback: `int(sequence_lengths.max())` on a device
tensor, i.e. a stream sync + D2H per forward, four times per MTP round.

The eagle worker already moves `new_seq_lens` to the host once per round
("one D2H per round (TODO: async or upper-bound estimate)"), so opting the
backend back in does not add a transfer; it routes the existing host copy
into the backend's sync-free branch (`seq_lens_cpu.max() + draft_window`).

The plain graphed decode path still never reads seq_lens_cpu; the flag now
costs nothing there either (the value is published regardless once the OR is
True, and the draft backend keeps its False — only one backend needs to opt
in for the OR to go True).
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
text = open(p).read()

old = """    # Every seq_lens_cpu read here has a device fallback (one readback);
    # the graphed decode path never reads it, so opting out is safe.
    needs_cpu_seq_lens: bool = False

    def __init__(self, runner=None) -> None:"""
new = """    # The graphed decode path never reads seq_lens_cpu, but the speculative
    # metadata path does (_speculative_max_row_length: the target-verify and
    # draft-extend token-slot gather bound). With the opt-out, every such
    # forward fell back to a device readback (int(sequence_lengths.max()):
    # sync + D2H, four times per MTP round); opting in reuses the per-round
    # seq_lens_cpu the eagle worker already materializes instead.
    needs_cpu_seq_lens: bool = True

    def __init__(self, runner=None) -> None:"""
assert text.count(old) == 1, "QSA needs_cpu_seq_lens anchor not found (once expected)"
open(p, "w").write(text.replace(old, new))

print("patch_qsa_cpu_seq_lens: ok")
