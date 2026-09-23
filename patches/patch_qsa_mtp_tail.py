#!/usr/bin/env python3
"""gfx1151 patch 23: MTP draft decode attends to the tokens drafted since the
capture (and stops reading stale scratch).

Under MTP the draft model's decode steps do not run the QSA indexer; they
reuse the block selection captured at the draft-extend pass
(`QSAMTPSharedSparseIndices`) and append a "tail" of the positions drafted
since then. `lookup` writes that tail into the last `tail_width` columns of
the row, after the captured selection's `-1` padding, so a short prompt's row
looks like `[0..L-1, -1 ... -1, L, -1, -1, -1]`.

The KV gather that feeds the attention kernel (`_compact_kv` in
qsa/sparse_attn.py, used by both the varlen fallback and the paged path)
packs each row's valid entries as a *prefix*: `valid_count` is a count of
valid entries, and column `c` is packed only if `c < valid_count`. Its
docstring says so. With the tail past the padding, column L (a `-1`) is
skipped and column 2051 (position L) is never reached, so packed entry
`valid_count - 1` is never written: on the varlen path it holds whatever the
scratch had from an earlier request (the paged path zero-fills it). The
draft therefore never sees the newest token, and on this box its output
depended on the previous request: greedy MTP decode differed between cold
runs from the first draft step (logprobs from position ~7, token flips
later) while the same image without MTP was bit-identical.

The fix places the tail immediately after the captured row's valid entries
(`(captured >= 0).sum()` per row), which keeps the row a valid prefix as the
gather expects. Captured positions are all below `captured_len`, so the tail
never collides with them. Pure tensor ops, graph-capturable.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
text = open(p).read()

old = """        slot = self.layer_slots[int(layer_id)]
        rows = req_pool_indices.to(torch.long)
        out = self.indices[slot, rows]
        base = self.captured_len[slot, rows].to(torch.int64)
        tail = base.unsqueeze(1) + self._tail_offsets.unsqueeze(0)
        valid = tail <= current_positions.to(torch.int64).unsqueeze(1)
        out[:, out.shape[1] - self.tail_width :] = torch.where(valid, tail, -1).to(
            out.dtype
        )
        return out
"""
assert text.count(old) == 1, "qwen_sparse_attn_backend.py: QSAMTPSharedSparseIndices.lookup anchor not found"
new = """        slot = self.layer_slots[int(layer_id)]
        rows = req_pool_indices.to(torch.long)
        out = self.indices[slot, rows]
        base = self.captured_len[slot, rows].to(torch.int64)
        tail = base.unsqueeze(1) + self._tail_offsets.unsqueeze(0)
        valid = tail <= current_positions.to(torch.int64).unsqueeze(1)
        tail_values = torch.where(valid, tail, -1).to(out.dtype)
        # gfx1151 patch 23: the KV gather packs a row's valid entries as a
        # prefix (valid_count is a count, not a mask), so the tail has to
        # follow the captured selection's valid entries, not its -1 padding;
        # otherwise the drafted positions are dropped and the packed slot
        # they should fill is left unwritten.
        width = out.shape[1]
        captured_valid = (out[:, : width - self.tail_width] >= 0).sum(
            dim=1, keepdim=True
        )
        tail_cols = (captured_valid + self._tail_offsets.unsqueeze(0)).clamp(
            max=width - 1
        )
        out[:, width - self.tail_width :] = -1
        out.scatter_(1, tail_cols, tail_values)
        return out
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)
print("patch 23 (qsa mtp tail) applied")
