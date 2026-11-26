#!/usr/bin/env python3
"""CPU regression for QSA cross-prefix compaction, stores, and shared RoPE."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa.metadata import build_rope_position_matrix
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode


RATIO = 4
WIDTH = 16
REQUEST_IDS = [3, 5, 2, 4, 1, 6]


def keys(request, positions):
    values = torch.stack((100 * request + positions, positions.square()), dim=-1)
    return values.float()[:, None]


def rope(request, positions):
    return torch.stack((positions + 17 * request, positions * 3, positions + 9))


def normalized(keys, coordinates):
    # Deliberately depend on all three coordinates and on the pooled key.
    phase = coordinates[0] + coordinates[1] * 2 + coordinates[2] * 3
    return keys * 2 + phase[:, None, None]


class Pool:
    qsa_compress_ratio = RATIO
    qsa_block_topk = 512
    qsa_index_kv_heads = 1
    qsa_index_head_dim = 2
    qsa_compressed_page_size = WIDTH // RATIO

    def __init__(self):
        self.pending = torch.zeros(2, 7 * RATIO, 1, 2)
        self.qsa_rope_position_buffer = torch.zeros(7 * RATIO, 3, dtype=torch.long)
        self.compressed = torch.full((2, 7 * WIDTH // RATIO, 1, 2), float("nan"))
        self.writes = [[], []]

    def get_qsa_key_state_buffer(self, layer):
        return self.pending[layer]

    def set_qsa_key_state_buffer(self, layer, locs, values):
        self.pending[layer, locs] = values

    def set_qsa_rope_position_buffer(self, locs, coordinates):
        self.qsa_rope_position_buffer[locs] = build_rope_position_matrix(coordinates, len(locs))

    def get_qsa_compressed_k_buffer(self, layer):
        return self.compressed[layer]

    def set_qsa_compressed_k_buffer(self, layer, locs, values):
        self.writes[layer].append(locs.tolist())
        self.compressed[layer, locs.long()] = values


class Indexer(QSAIndexer):
    def __init__(self, layer, fused):
        torch.nn.Module.__init__(self)
        self.layer_id = layer
        self.compress_ratio = RATIO
        self.index_n_heads = self.index_kv_heads = 1
        self.index_head_dim = 2
        self.rotary_emb = SimpleNamespace(mrope_section=(1, 1, 1))
        self.fused = fused
        self.fused_gathers = []

    def project_qk(self, hidden_states, positions, pool, cache_loc, **kwargs):
        # Emulate projection's fused ring store, which happens before calling
        # update_key_state_and_compress. In fallback, that method does the store.
        if self.fused:
            pool.set_qsa_key_state_buffer(self.layer_id, cache_loc, hidden_states)
            pool.set_qsa_rope_position_buffer(cache_loc, positions)
        return hidden_states, hidden_states, self.fused

    def _use_fused_compress(self, pool):
        return self.fused

    def normalize_compressed_keys(self, compressed_keys, block_positions):
        return normalized(compressed_keys, block_positions)

    def _fused_compress_store(self, pool, group_locs, write_locs, source_keys, source_rope):
        # CPU stand-in for the kernel contract; validate both gathers and the
        # scratch layout built by production _overwrite_cross_prefix_groups.
        assert not (group_locs < 0).any()
        assert group_locs.max() < source_keys.shape[0]
        assert source_rope.shape == (source_keys.shape[0], 3)
        self.fused_gathers.append(group_locs.tolist())
        values = normalized(
            source_keys[group_locs].mean(dim=1), source_rope[group_locs[:, 0]].T
        )
        pool.set_qsa_compressed_k_buffer(self.layer_id, write_locs, values)

    def select_prefill_tokens(self, q, compressed_keys, *args):
        return compressed_keys


def batch(prefixes, extends, mode):
    positions = [torch.arange(p, p + e) for p, e in zip(prefixes, extends)]
    packed = torch.cat(positions)
    seq_lens = torch.tensor(prefixes) + torch.tensor(extends)
    mrope = [rope(r, ps) for r, ps in zip(REQUEST_IDS, positions)]
    out_locs = [r * WIDTH + ps for r, ps in zip(REQUEST_IDS, positions)]
    forward_batch = SimpleNamespace(
        forward_mode=mode,
        extend_prefix_lens_cpu=prefixes,
        extend_seq_lens_cpu=extends,
        extend_seq_lens=torch.tensor(extends),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        req_pool_indices=torch.tensor(REQUEST_IDS, dtype=torch.int32),
        positions=packed,
        mrope_positions=torch.cat(mrope, dim=1),
        input_ids=torch.zeros_like(packed),
        out_cache_loc=torch.cat(out_locs),
    )
    return forward_batch, torch.cat([keys(r, ps) for r, ps in zip(REQUEST_IDS, positions)])


def make_backend(pool):
    backend = object.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = pool
    backend.compress_ratio = RATIO
    backend.device = torch.device("cpu")
    backend.max_context_len = WIDTH
    backend.req_to_token = torch.arange(7 * WIDTH, dtype=torch.int32).reshape(7, WIDTH)
    backend._mtp_shared_sparse_indices = None
    return backend


def check_state(pool, lengths):
    for request, length in zip(REQUEST_IDS, lengths):
        pending = torch.arange(length // RATIO * RATIO, length)
        slots = request * RATIO + pending % RATIO
        for layer in range(2):
            for start in range(0, length - RATIO + 1, RATIO):
                members = torch.arange(start, start + RATIO)
                expected = normalized(
                    (keys(request, members) + layer).mean(dim=0, keepdim=True),
                    rope(request, members[:1]),
                )
                slot = (request * WIDTH + start) // RATIO
                torch.testing.assert_close(
                    pool.compressed[layer, slot : slot + 1], expected
                )
            torch.testing.assert_close(
                pool.pending[layer, slots], keys(request, pending) + layer
            )
        torch.testing.assert_close(
            pool.qsa_rope_position_buffer[slots], rope(request, pending).T
        )


def exercise(fused, mode):
    pool = Pool()
    backend = make_backend(pool)
    indexers = [Indexer(layer, fused) for layer in range(2)]

    def run(prefixes, extends, mode):
        fb, hidden = batch(prefixes, extends, mode)
        metadata = backend._metadata_from_forward_batch(fb).indexer_metadata
        for layer, indexer in enumerate(indexers):
            indexer.forward_cuda(hidden + layer, fb.mrope_positions, fb, metadata)
        check_state(pool, fb.seq_lens.tolist())
        return metadata

    # Seed the ring from an aligned forward. The next packed rows have every
    # residue; two one-token rows stay pending, and one completes a group.
    prefixes = [5, 6, 7, 4, 5, 9]
    extends = [1, 7, 1, 5, 5, 1]
    seed = run([0] * len(REQUEST_IDS), prefixes, ForwardMode.EXTEND)
    assert seed.cross_prefix_plan is None
    plan = run(prefixes, extends, mode).cross_prefix_plan
    assert plan.first_entries.tolist() == [0, 0, 2, 3, 4, 5]
    assert plan.first_prefix_members.tolist() == [0, 2, 3, 0, 1, 0]
    assert plan.ring_locs[0].tolist() == [0] * RATIO
    assert plan.ring_locs[-1].tolist() == [0] * RATIO
    for layer in range(2):
        # A zero-group row ahead of a crossing row would otherwise borrow
        # and rewrite that other request's live slot.
        assert pool.writes[layer][-1] == [0, 21, 9, 0, 5, 0]
        if fused:
            main_gather = indexers[layer].fused_gathers[-2]
            assert main_gather[0] == [1, 1, 1, 2]
            assert main_gather[2] == [8, 8, 8, 8]
    # Both layers read old RoPE even though layer 0 overwrote request 5's
    # group-start slot with position 12 and request 1's with position 8.
    torch.testing.assert_close(plan.rope_snapshot[1], rope(5, torch.tensor([4]))[:, 0])
    torch.testing.assert_close(
        pool.qsa_rope_position_buffer[20], rope(5, torch.tensor([12]))[:, 0]
    )

    # A forward with only incomplete residual rows takes no snapshot.
    lengths = [p + e for p, e in zip(prefixes, extends)]
    step = run(lengths, [1] * len(REQUEST_IDS), mode)
    assert step.cross_prefix_plan is None
    for layer in range(2):
        assert set(pool.writes[layer][-1]) == {0}


def check_guards():
    backend = make_backend(Pool())
    for mode in (ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL):
        fb, _ = batch([7] * 6, [1] * 6, mode)
        if mode == ForwardMode.EXTEND:
            fb.extend_seq_lens_cpu = None
        elif mode == ForwardMode.MIXED:
            fb.extend_prefix_lens_cpu = None
        try:
            backend._metadata_from_forward_batch(fb)
        except RuntimeError as exc:
            assert "QSA extend write plan requires a ratio-aligned prefix" in str(exc)
        else:
            raise AssertionError(f"missing unaligned-prefix guard for {mode}")
    for mode, paged in ((ForwardMode.DECODE, False), (ForwardMode.TARGET_VERIFY, True)):
        fb, _ = batch([7] * 6, [1] * 6, mode)
        slots = backend.req_to_token[fb.req_pool_indices.long()]
        write_locs, _, _, member_rows, cross_entries = backend._qsa_build_write_plan(
            forward_batch=fb,
            speculative_paged=paged,
            token_slot_table=slots,
            sequence_lengths=fb.seq_lens,
        )
        assert member_rows is None and cross_entries is None
        assert write_locs.tolist() == [r * WIDTH // RATIO + 1 for r in REQUEST_IDS]


if __name__ == "__main__":
    for mode in (ForwardMode.MIXED, ForwardMode.EXTEND):
        for fused in (False, True):
            exercise(fused, mode)
    check_guards()
    print("check_qsa_cross_prefix: ok")
