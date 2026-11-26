#!/usr/bin/env python3
"""Backport SGLang #39575 to the pinned QSA implementation, including MIXED."""

import sys
from pathlib import Path


ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
ATTN = ROOT / "python/sglang/srt/layers/attention"


def patch(relpath, replacements):
    path = ATTN / relpath
    source = path.read_text()
    for old, new in replacements:
        assert source.count(old) == 1, f"{relpath}: anchor not found once: {old!r}"
        source = source.replace(old, new)
    path.write_text(source)


patch(
    "qsa/metadata.py",
    [
        (
            "from typing import Optional, Tuple",
            "from dataclasses import dataclass\nfrom typing import Optional, Tuple",
        ),
        (
            "class QSAIndexerMetadata(msgspec.Struct, frozen=True):",
            """@dataclass
class QSACrossPrefixPlan:
    # One entry per compacted group, and the first entry/member count per row.
    prefix_members: torch.Tensor
    first_entries: torch.Tensor
    first_prefix_members: torch.Tensor
    ring_locs: torch.Tensor
    # The pool's RoPE sidecar is shared across layers. Save its old values on
    # the compute stream before the first layer writes this forward's tail.
    rope_snapshot: Optional[torch.Tensor] = None


class QSAIndexerMetadata(msgspec.Struct, frozen=True):""",
        ),
        (
            """    # The first member's token row in this forward's packed tensors is extend only,
    # where group-aligned chunks keep every member in-chunk; None on paged forwards.""",
            """    # Extend-only packed first-member row. With a cross-prefix plan the
    # group can start before its request's current packed rows.""",
        ),
        (
            "    compress_member_rows: Optional[torch.Tensor] = None\n",
            "    compress_member_rows: Optional[torch.Tensor] = None\n"
            "    cross_prefix_plan: Optional[QSACrossPrefixPlan] = None\n",
        ),
    ],
)

patch(
    "qwen_sparse_attn_backend.py",
    [
        ("    QSAIndexerMetadata,\n", "    QSACrossPrefixPlan,\n    QSAIndexerMetadata,\n"),
        (
            """        row_token_starts=None,
        prefix_lens=None,
    ):""",
            """        row_token_starts=None,
        prefix_lens=None,
        cross_prefix=False,
    ):""",
        ),
        (
            """            # member_rows index this forward's packed token rows;
            # the chunk is group-aligned, so a group starts at block * ratio - prefix.""",
            """            # A group starts at block * ratio - prefix relative to this row's
            # extend, which is negative for a group crossing a cached tail.""",
        ),
        (
            "        return write_locs, group_end_positions, rows, member_rows\n",
            """        cross_entries = None
        if cross_prefix:
            prefix_members = torch.where(
                valid,
                (prefix_lens[rows] - blocks * compress_ratio).clamp(
                    min=0, max=compress_ratio - 1
                ),
                torch.zeros_like(blocks),
            )
            # A row's first entry is the only one that can cross the prefix.
            # Empty rows can borrow an entry for indexing but must never write it.
            first_entries = starts.clamp_max(capacity - 1)
            first_prefix_members = torch.where(
                counts > 0, prefix_members[first_entries], 0
            )
            cross_entries = prefix_members, first_entries, first_prefix_members
        return write_locs, group_end_positions, rows, member_rows, cross_entries
""",
        ),
        (
            """        # Prefix sharing is page-granular and the page is a ratio
        # multiple, so a matched prefix always covers whole groups. A
        # misaligned prefix would leave a shared group half-written.
        torch._assert_async((prefix_lens % ratio == 0).all())""",
            """        # Page-aligned radix hits only cover whole groups, but a private
        # chunk tail (or MIXED's single-token decode rows) can start mid-group.
        # Graph and speculative-paged modes use their existing ring path.
        prefix_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        has_residue = cross_prefix = False
        if (
            forward_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
            and getattr(forward_batch, "_original_forward_mode", None) is None
            and prefix_cpu is not None
            and extend_cpu is not None
            and len(prefix_cpu) >= lengths.numel()
            and len(extend_cpu) >= lengths.numel()
        ):
            for prefix, extend in zip(
                prefix_cpu[: lengths.numel()], extend_cpu[: lengths.numel()]
            ):
                residue = int(prefix) % ratio
                if residue:
                    has_residue = True
                    if residue + int(extend) >= ratio:
                        cross_prefix = True
                        break
        if not cross_prefix:
            aligned = prefix_lens % ratio == 0
            if has_residue:
                # Nothing to compress yet, but keep the incomplete ring tail.
                aligned = aligned | (prefix_lens // ratio == end_blocks)
            torch._assert_async(
                aligned.all(), "QSA extend write plan requires a ratio-aligned prefix"
            )""",
        ),
        (
            """            row_token_starts=row_token_starts,
            prefix_lens=prefix_lens,
        )""",
            """            row_token_starts=row_token_starts,
            prefix_lens=prefix_lens,
            cross_prefix=cross_prefix,
        )""",
        ),
        (
            """        extend_rope_matrix = None
        write_locs, group_positions, group_sequence_ids, group_member_rows = (""",
            """        extend_rope_matrix = None
        cross_prefix_plan = None
        (
            write_locs,
            group_positions,
            group_sequence_ids,
            group_member_rows,
            cross_entries,
        ) = (""",
        ),
        (
            """                    extend_rope_matrix = build_rope_position_matrix(
                        rope_source, token_to_batch_idx.numel()
                    )
                else:""",
            """                    extend_rope_matrix = build_rope_position_matrix(
                        rope_source, token_to_batch_idx.numel()
                    )
                    if cross_entries is not None:
                        (
                            prefix_members, first_entries, first_prefix_members
                        ) = cross_entries
                        ring_locs = build_group_ring_slots(
                            req_pool_indices=row_req_pool_indices,
                            group_end_positions=group_positions[first_entries].long(),
                            sequence_ids=group_sequence_ids[first_entries].long(),
                            compress_ratio=self.compress_ratio,
                        )
                        # A row with no crossing group does not need active ring state.
                        ring_locs = torch.where(
                            first_prefix_members[:, None] > 0, ring_locs, 0
                        )
                        cross_prefix_plan = QSACrossPrefixPlan(
                            prefix_members,
                            first_entries,
                            first_prefix_members,
                            ring_locs,
                        )
                else:""",
        ),
        (
            "            compress_member_rows=group_member_rows,\n",
            "            compress_member_rows=group_member_rows,\n"
            "            cross_prefix_plan=cross_prefix_plan,\n",
        ),
    ],
)

patch(
    "qsa/qsa_indexer.py",
    [
        (
            "    def _group_ring_slots(\n",
            """    def _snapshot_cross_prefix_members(
        self, metadata
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        plan = metadata.cross_prefix_plan
        if plan is None:
            return None
        pool = metadata.token_to_kv_pool
        # Advanced indexing copies. Current tail tokens can reuse the prefix
        # members' slots, so this layer's keys must be read before project_qk.
        ring_keys = pool.get_qsa_key_state_buffer(self.layer_id)[plan.ring_locs]
        if plan.rope_snapshot is None:
            # Unlike keys, these slots are shared by all indexers in the batch.
            # Clone once, before the first layer reuses them for the new tail.
            rope = pool.qsa_rope_position_buffer[plan.ring_locs[:, 0]]
            plan.rope_snapshot = torch.where(
                plan.first_prefix_members[:, None] > 0, rope, 0
            )
        return ring_keys, plan.rope_snapshot

    def _group_ring_slots(
""",
        ),
        (
            """        state_stored: bool = False,
    ) -> None:
        \"\"\"Store the pending-group ring and compress each completed group.\"\"\"

        pool = metadata.token_to_kv_pool""",
            """        state_stored: bool = False,
        cross_prefix_state: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        \"\"\"Store the pending-group ring and compress each completed group.\"\"\"

        plan = metadata.cross_prefix_plan
        if plan is not None and cross_prefix_state is None:
            if state_stored:
                raise RuntimeError(
                    "QSA cross-prefix snapshot must precede the ring store"
                )
            cross_prefix_state = self._snapshot_cross_prefix_members(metadata)
        pool = metadata.token_to_kv_pool""",
        ),
        (
            """            # Extend chunks are group-aligned; each planned group lies in this forward,
            # so read its members from the packed chunk tensors.
            member_rows = metadata.compress_member_rows.long()""",
            """            if token_k.shape[0] == 0:
                return
            member_rows = metadata.compress_member_rows.long()""",
        ),
        (
            "            source_keys = token_k\n",
            """            if plan is not None:
                # Do not gather negative indices or read a previous request's
                # packed rows. The correction replaces these placeholder values.
                first_current = member_rows + plan.prefix_members
                group_locs = torch.maximum(group_locs, first_current[:, None])
            source_keys = token_k
""",
        ),
        (
            """            return
        key_groups = source_keys[group_locs]
        pooled = average_pool_qsa_keys(key_groups)
        compressed_rope_positions = self._rope_from_matrix(
            source_rope[group_locs[:, 0]]
        )
        normalized = self.normalize_compressed_keys(pooled, compressed_rope_positions)
        pool.set_qsa_compressed_k_buffer(self.layer_id, compressed_locs, normalized)

    def _compress_decode_cuda_graph(""",
            """        else:
            key_groups = source_keys[group_locs]
            pooled = average_pool_qsa_keys(key_groups)
            compressed_rope_positions = self._rope_from_matrix(
                source_rope[group_locs[:, 0]]
            )
            normalized = self.normalize_compressed_keys(pooled, compressed_rope_positions)
            pool.set_qsa_compressed_k_buffer(self.layer_id, compressed_locs, normalized)

        if plan is not None:
            self._overwrite_cross_prefix_groups(
                token_k,
                metadata,
                group_locs[plan.first_entries],
                compressed_locs[plan.first_entries],
                cross_prefix_state,
            )

    def _overwrite_cross_prefix_groups(
        self, token_k, metadata, current_group_locs, compressed_locs, cross_prefix_state
    ) -> None:
        \"\"\"Recompress only the first group of each row; inactive rows write slot 0.\"\"\"
        ring_keys, ring_rope = cross_prefix_state
        prefix_members = metadata.cross_prefix_plan.first_prefix_members
        pool = metadata.token_to_kv_pool
        current_keys = token_k[current_group_locs]
        use_ring = torch.arange(
            self.compress_ratio, device=token_k.device
        )[None, :] < prefix_members[:, None]
        key_groups = torch.where(
            use_ring[:, :, None, None], ring_keys.to(token_k.dtype), current_keys
        )
        write_locs = torch.where(prefix_members > 0, compressed_locs, 0)
        if self._use_fused_compress(pool):
            # Use the same fused math as the main pass. The scratch rows are
            # already gathered, and only the first row's RoPE is read per group.
            groups = key_groups.shape[0]
            scratch_locs = torch.arange(
                groups * self.compress_ratio, device=token_k.device
            ).view(groups, self.compress_ratio)
            scratch_rope = ring_rope.repeat_interleave(self.compress_ratio, dim=0)
            self._fused_compress_store(
                pool,
                scratch_locs,
                write_locs,
                source_keys=key_groups.flatten(0, 1),
                source_rope=scratch_rope,
            )
        else:
            pooled = average_pool_qsa_keys(key_groups)
            normalized = self.normalize_compressed_keys(
                pooled, self._rope_from_matrix(ring_rope)
            )
            pool.set_qsa_compressed_k_buffer(self.layer_id, write_locs, normalized)

    def _compress_decode_cuda_graph(""",
        ),
        (
            "        q, token_k, state_stored = self.project_qk(\n",
            """        cross_prefix_state = self._snapshot_cross_prefix_members(indexer_metadata)
        q, token_k, state_stored = self.project_qk(
""",
        ),
        (
            "            state_stored=state_stored,\n",
            "            state_stored=state_stored,\n"
            "            cross_prefix_state=cross_prefix_state,\n",
        ),
    ],
)

print("patch_qsa_cross_prefix: ok")
