#!/usr/bin/env python3
"""gfx1151 patch 28: PLE short conv over the packed prefill batch.

`Qwen4ExpPLEGroupedNorm._short_conv` runs a causal depthwise conv1d
(kernel 4, dilation 3, 10,240 channels) over each PLE layer's input with 9
carried state columns per request. For prefill it padded the packed batch
into `[requests, row_width, channels]` with `row_width` the longest
request in the batch, then `cat` with the state and the conv output made
two more copies of that size. Memory is O(requests × longest), not
O(tokens): a chunked-prefill batch of 17 short prompts (71 tokens each)
plus a 6,024-token chunk of a long prompt is 7,231 real tokens but
18 × 6,024 × 10,240 × 2 B = 2.1 GiB per copy, 6.3 GiB for the three,
and at the 20-request cap with an 8,192-token chunk it would be 9.6 GiB,
more than the server has free. Found by the benchmark suite's mixed
scenario, which took the box to 117 MiB of free VRAM (the same mix with
300-token short prompts left 3.5 GiB, because they filled more of the
chunk and left the long prompt a shorter row); pinned down with a
`/start_profile MEM` snapshot stopped at the jump.

Fix: `_packed_short_conv` splices each request's state columns in front
of its tokens in one `[channels, tokens + requests × 9]` row, runs the
conv1d once, and reads the outputs back at the token columns; the
next-state and track-slot gathers read the same row at the request's
boundary, as the padded layout's gather did. The target-verify path keeps
the padded layout (its row width is the draft length, 4) and its
intermediate-state unfold. Decode is untouched (own fast path).
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
p = f"{path}/python/sglang/srt/models/qwen4_exp.py"
text = open(p).read()

# 1. the packed conv helper, before the PLE module class
anchor = "def _use_attn_tp_ngram() -> bool:\n"
assert text.count(anchor) == 1, "qwen4_exp.py: _use_attn_tp_ngram anchor not found"
helper = '''def _packed_short_conv(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    dilation: int,
    state_len: int,
    lengths: torch.Tensor,
    req_indices: torch.Tensor,
    token_offsets: torch.Tensor,
    valid_tokens: torch.Tensor,
):
    """Causal depthwise conv over a packed batch, O(tokens) memory.

    Each request's ``state_len`` carried columns are spliced in front of its
    tokens in one ``[channels, tokens + requests * state_len]`` row and the
    conv runs once over it, instead of padding every request to the longest
    one (``[requests, row_width, channels]``, three copies of it).

    ``x`` is ``[tokens, channels]``, ``state`` ``[requests, channels,
    state_len]``. Returns the conv output ``[tokens, channels]`` and
    ``gather_at(offsets)``, which reads the ``state_len`` columns ending
    just before position ``offsets[r]`` of request ``r``'s
    ``[state | tokens]`` row, the same window the padded layout gathered.
    Tokens with ``valid_tokens`` false (padding rows) are routed to a
    scratch column and get a finite, unused output.
    """
    channels = x.shape[1]
    requests = lengths.shape[0]
    segment = lengths + state_len
    starts = torch.cumsum(segment, dim=0) - segment
    total = x.shape[0] + requests * state_len
    packed = x.new_zeros((channels, total + 1))
    state_cols = torch.arange(state_len, device=x.device, dtype=torch.long)
    packed[:, (starts.unsqueeze(1) + state_cols).reshape(-1)] = state.permute(
        1, 0, 2
    ).reshape(channels, requests * state_len)
    token_cols = starts.index_select(0, req_indices) + state_len + token_offsets
    token_cols = torch.where(valid_tokens, token_cols, torch.full_like(token_cols, total))
    packed[:, token_cols] = x.t()
    out = F.conv1d(
        packed.unsqueeze(0), weight, bias=None, dilation=dilation, groups=channels
    ).squeeze(0)
    conv_output = out.index_select(1, token_cols - state_len).t()

    def gather_at(offsets: torch.Tensor) -> torch.Tensor:
        cols = (starts + offsets).unsqueeze(1) + state_cols
        return (
            packed.index_select(1, cols.reshape(-1))
            .reshape(channels, requests, state_len)
            .permute(1, 0, 2)
        )

    return conv_output, gather_at


'''
text = text.replace(anchor, helper + anchor, 1)

# 2. prefill takes the packed path; verify keeps the padded one
old_head = """        state = conv_state.index_select(0, batch.state_indices).to(dtype=x.dtype)
        padded_seq = x.new_zeros(
            (batch.lengths.shape[0], batch.row_width, self.conv_channels)
        )
"""
assert text.count(old_head) == 1, "qwen4_exp.py: padded short conv anchor not found"
new_head = """        state = conv_state.index_select(0, batch.state_indices).to(dtype=x.dtype)
        if not batch.mode.is_target_verify():
            # gfx1151 patch 28: the padded layout below is O(requests x
            # longest request) per copy; a chunked prefill mixing short
            # prompts with a long one made that gigabytes per PLE layer.
            conv_output, gather_at = _packed_short_conv(
                x,
                state,
                self.conv1d.weight.to(dtype=x.dtype),
                self.short_conv_dilation,
                self.short_conv_state_len,
                batch.lengths,
                batch.req_indices,
                batch.token_offsets,
                batch.valid_tokens,
            )
            conv_state[batch.state_indices] = gather_at(batch.lengths).to(
                dtype=conv_state.dtype
            )
            # Same boundary mamba uses, into the slot the radix tree reads.
            track = _ple_track_targets(forward_batch, batch)
            if track is not None:
                track_indices, track_offsets = track
                conv_state[track_indices] = gather_at(track_offsets).to(
                    dtype=conv_state.dtype
                )
            return F.silu(conv_output)

        padded_seq = x.new_zeros(
            (batch.lengths.shape[0], batch.row_width, self.conv_channels)
        )
"""
text = text.replace(old_head, new_head, 1)

# 3. the padded path's non-verify gather branch is now unreachable
old_tail = """                intermediate_cache[: batch.lengths.shape[0], : batch.row_width].copy_(
                    intermediate_state.to(dtype=intermediate_cache.dtype)
                )
        else:
            state_cols = torch.arange(
                self.short_conv_state_len, device=x.device, dtype=torch.long
            )

            def _gather_at(offsets: torch.Tensor) -> torch.Tensor:
                return conv_input.gather(
                    2,
                    (offsets.unsqueeze(1) + state_cols.unsqueeze(0))
                    .unsqueeze(1)
                    .expand(-1, self.conv_channels, -1),
                )

            next_state = _gather_at(batch.lengths)
            conv_state[batch.state_indices] = next_state.to(dtype=conv_state.dtype)

            # Same boundary mamba uses, into the slot the radix tree reads.
            track = _ple_track_targets(forward_batch, batch)
            if track is not None:
                track_indices, track_offsets = track
                conv_state[track_indices] = _gather_at(track_offsets).to(
                    dtype=conv_state.dtype
                )

        return F.silu(conv_output[batch.req_indices, batch.token_offsets])
"""
assert text.count(old_tail) == 1, "qwen4_exp.py: padded gather branch anchor not found"
new_tail = """                intermediate_cache[: batch.lengths.shape[0], : batch.row_width].copy_(
                    intermediate_state.to(dtype=intermediate_cache.dtype)
                )

        return F.silu(conv_output[batch.req_indices, batch.token_offsets])
"""
text = text.replace(old_tail, new_tail, 1)

open(p, "w").write(text)
print("patched", p)
print("patch 28 (packed PLE short conv for prefill) applied")
