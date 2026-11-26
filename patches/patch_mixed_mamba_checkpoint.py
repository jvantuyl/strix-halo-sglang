#!/usr/bin/env python3
"""Patch 56: preserve non-lazy Mamba prefill checkpoints in MIXED batches."""

import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    assert count == 1, f"expected one schedule_batch.py anchor, got {count}: {old[:100]}"
    return source.replace(old, new, 1)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
    path = root / "python/sglang/srt/managers/schedule_batch.py"
    source = path.read_text()

    source = replace_once(
        source,
        '''def set_mamba_track_indices_from_reqs(
    batch, track_positions: Optional[List[int]] = None
):
    """Build mamba_track_indices from req objects (authoritative source).

    track_positions: optional per-req ping-pong position override (the lazy
    spec track plan, see mamba_lazy_spec_prepare).
    """
    req_to_token_pool = batch.req_to_token_pool
    all_buffers = req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[
        batch.req_pool_indices
    ]  # (bs, ping_pong_size), int64, on device
    if track_positions is None:
        # Guard: mamba_next_track_idx may be None for requests that haven't
        # gone through _alloc_ping_pong_buffer yet (e.g., spec v2 verify path).
        # Default to 0 (first ping-pong slot) to avoid TypeError.
        track_positions = [
            (
                req.kv.mamba_next_track_idx
                if req.kv.mamba_next_track_idx is not None
                else 0
            )
            for req in batch.reqs
        ]
    batch.mamba_track_buffer_indices = list(track_positions)
    idx = (
        torch.tensor(
            track_positions,
            dtype=torch.int64,
            pin_memory=True,
        )
        .unsqueeze(1)
        .to(device=all_buffers.device, non_blocking=True)
    )
    batch.mamba_track_indices = (
        torch.gather(all_buffers, 1, idx).squeeze(1).to(torch.int64)
    )''',
        '''def mamba_track_indices_from_reqs(
    *,
    req_to_token_pool: HybridReqToTokenPool,
    req_pool_indices: torch.Tensor,
    reqs: List[Req],
    track_positions: Optional[List[int]] = None,
) -> Tuple[List[int], torch.Tensor]:
    """Look up the per-request ping-pong positions and slot ids without mutation.

    track_positions: optional per-req ping-pong position override (the lazy
    spec track plan, see mamba_lazy_spec_prepare).
    """
    all_buffers = req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[
        req_pool_indices
    ]  # (bs, ping_pong_size), int64, on device
    if track_positions is None:
        track_positions = [
            (
                req.kv.mamba_next_track_idx
                if req.kv.mamba_next_track_idx is not None
                else 0
            )
            for req in reqs
        ]
    freed_rows = [
        i for i, req in enumerate(reqs) if req.kv.mamba_next_track_idx is None
    ]
    idx = (
        torch.tensor(
            track_positions,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(all_buffers.device),
        )
        .unsqueeze(1)
        .to(device=all_buffers.device, non_blocking=True)
    )
    track_indices = torch.gather(all_buffers, 1, idx).squeeze(1).to(torch.int64)
    if freed_rows:
        # Overlap can leave an in-flight verify row whose old slot has already
        # been reused. A negative destination makes the scatter skip this row.
        track_indices[freed_rows] = -1
    return list(track_positions), track_indices


def set_mamba_track_indices_from_reqs(
    batch, track_positions: Optional[List[int]] = None
):
    """Set the forward's Mamba tracking positions and slot ids."""
    batch.mamba_track_buffer_indices, batch.mamba_track_indices = (
        mamba_track_indices_from_reqs(
            req_to_token_pool=batch.req_to_token_pool,
            req_pool_indices=batch.req_pool_indices,
            reqs=batch.reqs,
            track_positions=track_positions,
        )
    )''',
    )
    source = replace_once(
        source,
        """        out_cache_loc = torch.cat([self.out_cache_loc, running_out_cache_loc])

        self.merge_batch(running_batch)
        self.out_cache_loc = out_cache_loc
        if merged_seq_lens_cpu is not None:""",
        """        out_cache_loc = torch.cat([self.out_cache_loc, running_out_cache_loc])
        prefill_track = self._mamba_track_for_mixed(running_batch, running_bs)

        self.merge_batch(running_batch)
        self.out_cache_loc = out_cache_loc
        if prefill_track is not None:
            (
                self.mamba_track_indices,
                self.mamba_track_mask,
                self.mamba_track_seqlens,
                self.mamba_prefill_track_mask_cpu,
                self.mamba_track_seqlens_cpu,
            ) = prefill_track
        if merged_seq_lens_cpu is not None:""",
    )
    source = replace_once(
        source,
        '''    def convert_decode_to_extend(self):
        """View every decode request as a 1-token extend''',
        '''    def _mamba_track_for_mixed(
        self, running_batch: ScheduleBatch, running_bs: int
    ) -> Optional[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[List[bool]],
            Optional[List[int]],
        ]
    ]:
        if self.mamba_track_mask is None:
            return None
        mamba = get_exec().mamba
        if not mamba.enable_mamba_extra_buffer or mamba.enable_mamba_extra_buffer_lazy:
            return None

        running_track_indices = running_batch.mamba_track_indices
        if running_track_indices is None:
            # Spec decode can defer building track indices until its forward.
            _, running_track_indices = mamba_track_indices_from_reqs(
                req_to_token_pool=running_batch.req_to_token_pool,
                req_pool_indices=running_batch.req_pool_indices,
                reqs=running_batch.reqs,
            )

        # The extend rows have already claimed their checkpoint slots and
        # stamped the depth at which they will donate them. A MIXED forward
        # must still write those slots. Decode tails never checkpoint here:
        # retain their valid slot ids for translation, but mask off writes.
        return (
            torch.cat([self.mamba_track_indices, running_track_indices]),
            torch.cat(
                [self.mamba_track_mask, self.mamba_track_mask.new_zeros(running_bs)]
            ),
            torch.cat(
                [
                    self.mamba_track_seqlens,
                    self.mamba_track_seqlens.new_full((running_bs,), -1),
                ]
            ),
            (
                self.mamba_prefill_track_mask_cpu + [False] * running_bs
                if self.mamba_prefill_track_mask_cpu is not None
                else None
            ),
            (
                self.mamba_track_seqlens_cpu + [-1] * running_bs
                if self.mamba_track_seqlens_cpu is not None
                else None
            ),
        )

    def convert_decode_to_extend(self):
        """View every decode request as a 1-token extend''',
    )
    path.write_text(source)
    print("patch 56: preserved non-lazy Mamba checkpoints in MIXED batches")


if __name__ == "__main__":
    main()
