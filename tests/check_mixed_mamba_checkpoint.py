#!/usr/bin/env python3
"""Exercise the patched SGLang scheduler and Mamba backend metadata on CPU.

Usage: python3 tests/check_mixed_mamba_checkpoint.py [patched-sglang-root]
The default source root is /sgl-workspace/sglang (the image checkout).
"""

import ast
import sys
from enum import Enum, IntEnum, auto
from functools import total_ordering
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple

import torch


def load_symbols(root, path, env, *names, classes=None):
    """Run the real source definitions without importing the serving runtime.

    Only the requested class method bodies are loaded; the fixture supplies the
    constructor state and platform dependencies, not the methods under test.
    """
    file = root / path
    selected = []
    for node in ast.parse(file.read_text()).body:
        if getattr(node, "name", None) in names:
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name in (classes or {}):
            methods = set(classes[node.name])
            body = [n for n in node.body if getattr(n, "name", None) in methods]
            assert {n.name for n in body} == methods, (file, node.name, methods)
            node.body = body
            node.bases = []
            node.decorator_list = []
            selected.append(node)
    assert {n.name for n in selected} == set(names) | set(classes or {}), file
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + selected, type_ignores=[]), str(file), "exec"), env)


TRACK_FIELDS = (
    "mamba_track_indices",
    "mamba_track_mask",
    "mamba_track_seqlens",
    "mamba_prefill_track_mask_cpu",
    "mamba_track_seqlens_cpu",
)
DECODE_FIELDS = (
    "mamba_track_buffer_indices",
    "mamba_track_mask_cpu",
    "mamba_track_mask_next_cpu",
    "mamba_decode_batch_idx_cpu",
    "mamba_lazy_spec_track_positions_cpu",
)
CHUNK = 4


def load_runtime(root):
    settings = NS(
        mamba=NS(enable_mamba_extra_buffer=True, enable_mamba_extra_buffer_lazy=False)
    )
    env = dict(
        __name__=__name__,
        torch=torch,
        Enum=Enum,
        IntEnum=IntEnum,
        auto=auto,
        total_ordering=total_ordering,
        CustomSpecAlgo=object,
        NamedTuple=NamedTuple,
        ForwardMetadata=NS,
        get_exec=lambda: settings,
        mamba_cache_chunk_size=lambda: CHUNK,
        mamba_checkpoint_grid=lambda page_size: page_size,
        is_pin_memory_available=lambda device: False,
        _MAMBA_DEBUG_ASSERTS=False,
    )
    load_symbols(root, "beam_search/batch_tail.py", env, "strip_beam_tail")
    load_symbols(
        root,
        "speculative/spec_info.py",
        env,
        "SpeculativeAlgorithm",
    )
    load_symbols(
        root,
        "model_executor/forward_batch_info.py",
        env,
        "ForwardMode",
        "CaptureHiddenMode",
        classes={"ForwardBatch": ["mamba_track_aligned_lens"]},
    )
    load_symbols(
        root,
        "managers/schedule_batch.py",
        env,
        "_MambaRadixCacheV2TrackEntry",
        "mamba_track_indices_from_reqs",
        "set_mamba_track_indices_from_reqs",
        classes={
            "ScheduleBatch": [
                "_mamba_radix_cache_v2_req_prepare_for_extend",
                "_mamba_track_for_mixed",
                "batch_size",
                "mix_with_running",
                "merge_batch",
            ],
        },
    )
    plan_path = root / "layers/attention/mamba/prefill_track_metadata.py"
    exec(compile(plan_path.read_text(), str(plan_path), "exec"), env)
    load_symbols(
        root,
        "layers/attention/hybrid_linear_attn_backend.py",
        env,
        classes={
            "MambaAttnBackendBase": [
                "mamba_chunk_size",
                "_translate_mamba_indices",
                "_has_cpu_prefill_track_metadata",
                "_forward_metadata",
                "_init_track_conv_indices",
                "_init_track_ssm_indices",
                "_init_track_ssm_indices_from_cpu",
                "_track_indices_to_device",
                "_track_mamba_state_extend",
            ],
        },
    )
    env["Mamba2AttnBackend"] = type("Mamba2AttnBackend", (env["MambaAttnBackendBase"],), {})
    load_symbols(
        root,
        "mem_cache/memory_pool.py",
        env,
        classes={
            "HybridReqToTokenPool": [
                "get_mamba_ping_pong_other_idx",
                "get_mamba_ping_pong_keep_idx",
                "set_mamba_ping_pong_slot",
                "donate_mamba_ping_pong_slot",
            ],
        },
    )
    load_symbols(
        root,
        "mem_cache/unified_cache/components/mamba.py",
        env,
        classes={
            "MambaComponent": ["prepare_for_caching_req", "finalize_match_result_in_cache"],
        },
    )
    return env, settings


def make_req(row, slots, *, length, prefix=0, next_idx=0):
    def set_extend_range(start, end):
        req.extend_range = NS(length=end - start)

    req = NS(
        seqlen=length,
        prefix_indices=range(prefix),
        extend_range=NS(length=length - prefix),
        mamba_branching_seqlen=None,
        _refresh_fill_ids=lambda: None,
        set_extend_range=set_extend_range,
        kv=NS(
            req_pool_idx=row,
            mamba_ping_pong_track_buffer=torch.tensor(slots),
            mamba_next_track_idx=next_idx,
            mamba_last_track_idx=None,
            mamba_last_track_seqlen=None,
        ),
    )
    return req


def make_batch(env, reqs, pool, *, spec, **fields):
    batch = env["ScheduleBatch"]()
    seq = torch.tensor([r.seqlen for r in reqs])
    indices = torch.tensor([r.kv.req_pool_idx for r in reqs])
    batch.__dict__.update(dict.fromkeys(TRACK_FIELDS + DECODE_FIELDS))
    batch.__dict__.update(
        reqs=reqs,
        req_to_token_pool=pool,
        spec_algorithm=spec,
        spec_info=None,
        sampling_info=NS(merge_batch=lambda other: None),
        tree_cache=NS(page_size=CHUNK),
        model_config=NS(is_encoder_decoder=False, hf_text_config=NS(mamba_chunk_size=CHUNK)),
        beam_tail=None,
        input_ids=None,
        seq_lens=seq,
        seq_lens_cpu=seq.clone(),
        orig_seq_lens=seq.clone(),
        req_pool_indices=indices,
        req_pool_indices_cpu=indices.clone(),
        return_logprob=False,
        multimodal_inputs=None,
        has_grammar=False,
        return_hidden_states_mode=env["CaptureHiddenMode"].NULL,
        is_prefill_only=False,
    )
    batch.__dict__.update(fields)
    return batch


def make_batches(env, *, nextn=False):
    mode = env["ForwardMode"]
    spec = env["SpeculativeAlgorithm"].EAGLE if nextn else env["SpeculativeAlgorithm"].NONE
    tracked = make_req(6, [2, 3], length=14, prefix=4)  # track depth 12 from h
    short = make_req(1, [4, 5], length=2)
    d0 = make_req(5, [6, 7], length=16, next_idx=1)
    d1 = make_req(2, [8, 9], length=7, next_idx=0)
    pool = env["HybridReqToTokenPool"]()
    pool.mamba_ping_pong_track_buffer_size = 2
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.full((7, 2), -1)
    for req in (tracked, short, d0, d1):
        pool.req_index_to_mamba_ping_pong_track_buffer_mapping[req.kv.req_pool_idx] = (
            req.kv.mamba_ping_pong_track_buffer
        )
    pool.req_to_token = torch.arange(256).reshape(8, 32)
    pool.get_mamba_indices = lambda indices: indices + 10
    pool.translate_mamba_indices = lambda indices: indices + 32
    extend = make_batch(
        env, [tracked, short], pool, spec=spec, forward_mode=mode.EXTEND,
        out_cache_loc=torch.arange(12), prefix_lens=[4, 0], extend_lens=[10, 2],
        extend_num_tokens=12, extend_logprob_start_lens=[0, 0],
    )
    entries = [extend._mamba_radix_cache_v2_req_prepare_for_extend(r) for r in extend.reqs]
    extend.mamba_prefill_track_mask_cpu = [e.track_mask for e in entries]
    extend.mamba_track_seqlens_cpu = [e.track_seqlen for e in entries]
    extend.mamba_track_indices = torch.tensor([e.track_index for e in entries])
    extend.mamba_track_mask = torch.tensor(extend.mamba_prefill_track_mask_cpu)
    extend.mamba_track_seqlens = torch.tensor(extend.mamba_track_seqlens_cpu)
    running = make_batch(
        env, [d0, d1], pool, spec=spec, forward_mode=mode.DECODE,
        out_cache_loc=torch.tensor([17, 18]), mamba_track_mask=torch.tensor([True, False]),
        mamba_track_buffer_indices=[0, 1], mamba_track_mask_cpu=[True, False],
        mamba_track_mask_next_cpu=[False, False], mamba_decode_batch_idx_cpu=[4, 5],
        mamba_track_seqlens=torch.tensor([16, -1]),
    )
    if nextn:
        running.seq_lens_cpu = None
        running.out_cache_loc = None
    else:
        # Prepared snapshot still owns a slot per request; it intentionally
        # differs from the current next positions so a re-gather is observable.
        running.mamba_track_indices = torch.tensor([6, 9])
    return extend, running


def forward_metadata(env, batch):
    backend = env["MambaAttnBackendBase"]()
    backend.device = "cpu"
    backend._mamba_chunk_size = CHUNK
    backend.conv_states_shape = (1, 3)
    backend.req_to_token_pool = batch.req_to_token_pool
    view = env["ForwardBatch"]()
    lengths = torch.tensor(batch.extend_lens, dtype=torch.int32)
    view.__dict__.update({f: getattr(batch, f) for f in TRACK_FIELDS})
    view.__dict__.update(
        batch_size=batch.batch_size(),
        req_pool_indices=batch.req_pool_indices,
        forward_mode=batch.forward_mode,
        _original_batch_size=None,
        extend_prefix_lens=torch.tensor(batch.prefix_lens, dtype=torch.int32),
        extend_prefix_lens_cpu=batch.prefix_lens,
        extend_seq_lens=lengths,
        extend_seq_lens_cpu=batch.extend_lens,
        extend_start_loc=torch.cat((lengths.new_zeros(1), lengths[:-1].cumsum(0))),
        tbo_parent_token_range=None,
    )
    assert backend._has_cpu_prefill_track_metadata(view) == all(
        getattr(batch, f) is not None for f in TRACK_FIELDS[-2:]
    )
    return backend, view, backend._forward_metadata(view)


def check_donation_restore(env, batch, backend, view, metadata):
    """Use production claim, write, donation and COW restore selection."""
    req = batch.reqs[0]
    depth = req.kv.mamba_last_track_seqlen
    slot = req.kv.mamba_ping_pong_track_buffer[req.kv.mamba_last_track_idx].item()
    assert depth == 12 and slot == 2 and req.kv.mamba_next_track_idx == 1
    states = torch.full((64, 1), -7.0)
    h = torch.arange(6.0).reshape(1, 6, 1)
    backend._track_mamba_state_extend(view, h, states, metadata)
    assert states[slot + 32].item() == 2.0
    assert (states[torch.tensor([3, 4, 5, 6, 7, 8, 9]) + 32] == -7.0).all()

    pool = batch.req_to_token_pool
    component = env["MambaComponent"]()
    component.cache = NS(enable_mamba_extra_buffer=True, req_to_token_pool=pool)
    component.int8_ckpt_pool = None
    component._alloc_mamba_slot = lambda: torch.tensor([10])
    insert = NS()
    assert component.prepare_for_caching_req(req, insert, 14, False) == depth
    assert insert.mamba_value.tolist() == [slot]
    assert req.kv.mamba_ping_pong_track_buffer.tolist() == [10, 3]
    assert pool.req_index_to_mamba_ping_pong_track_buffer_mapping[6].tolist() == [10, 3]

    component.component_type = "mamba"
    component.tree_core = NS(get_component_device_value=lambda node, kind: node[kind])
    reader = NS(kv=NS(holds_mamba=False, mamba_needs_clear=True))
    pool.mamba_allocator = NS(alloc=lambda n: torch.tensor([11]))
    match = NS(best_match_node={"mamba": insert.mamba_value})
    assert component.finalize_match_result_in_cache(NS(cow_mamba=True, req=reader), match) is match
    assert reader.kv.mamba_pool_idx.item() == 11 and not reader.kv.mamba_needs_clear
    assert states[pool.translate_mamba_indices(reader.kv.mamba_cow_src_index)].item() == 2.0


def check_mix(env, *, nextn, absent_mirror=None):
    extend, running = make_batches(env, nextn=nextn)
    if absent_mirror:
        setattr(extend, absent_mirror, None)
    head_mask = extend.mamba_prefill_track_mask_cpu
    head_lens = extend.mamba_track_seqlens_cpu
    tail_mask = running.mamba_track_mask
    mapping = running.req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping
    before = mapping.clone()
    extend.mix_with_running(running)
    assert extend.forward_mode.is_mixed()
    assert extend.req_pool_indices.tolist() == [6, 1, 5, 2]
    assert extend.seq_lens.tolist() == [14, 2, 16, 7]
    assert extend.seq_lens_cpu.tolist() == [14, 2, 16, 7]
    assert extend.prefix_lens == [4, 0, 15, 6]
    assert extend.extend_lens == [10, 2, 1, 1]
    assert extend.out_cache_loc[-2:].tolist() == ([175, 70] if nextn else [17, 18])
    assert extend.mamba_track_mask.tolist() == [True, False, False, False]
    assert extend.mamba_track_seqlens.tolist() == [14, -1, -1, -1]
    for field, expected in zip(TRACK_FIELDS[-2:], ([True, False, False, False], [14, -1, -1, -1])):
        assert getattr(extend, field) == (None if field == absent_mirror else expected)
    assert head_mask is None or head_mask == [True, False]
    assert head_lens is None or head_lens == [14, -1]
    virtual_slots = [2, 4, 7, 8] if nextn else [2, 4, 6, 9]
    assert extend.mamba_track_indices.tolist() == virtual_slots
    assert all(getattr(extend, f) is None for f in DECODE_FIELDS)
    assert running.mamba_track_mask is tail_mask and tail_mask.tolist() == [True, False]
    assert (running.mamba_track_indices is None) == nextn
    assert running.mamba_track_buffer_indices == [0, 1]
    assert [r.kv.mamba_next_track_idx for r in running.reqs] == [1, 0]
    assert torch.equal(mapping, before)

    backend, view, metadata = forward_metadata(env, extend)
    assert metadata.has_mamba_track_mask
    assert metadata.mamba_track_indices.tolist() == [slot + 32 for slot in virtual_slots]
    assert metadata.query_start_loc.tolist() == [0, 10, 12, 13, 14]
    assert metadata.track_conv_indices.tolist() == [[5, 6, 7]]
    assert metadata.track_ssm_final_dst.numel() == 0
    assert metadata.track_ssm_h_dst.tolist() == [34]
    if absent_mirror is None:
        assert metadata.mamba_track_mask_indices.tolist() == [0]
        check_donation_restore(env, extend, backend, view, metadata)


def check_strategy_guard(env, settings):
    for extra, lazy in ((False, False), (True, True)):
        extend, running = make_batches(env, nextn=True)
        settings.mamba.enable_mamba_extra_buffer = extra
        settings.mamba.enable_mamba_extra_buffer_lazy = lazy
        extend.mix_with_running(running)
        assert all(getattr(extend, field) is None for field in TRACK_FIELDS)
        assert running.mamba_track_indices is None


def check_freed_row(env):
    _, running = make_batches(env, nextn=True)
    running.reqs[1].kv.mamba_next_track_idx = None
    lookup = env["mamba_track_indices_from_reqs"]
    positions, indices = lookup(
        req_to_token_pool=running.req_to_token_pool,
        req_pool_indices=running.req_pool_indices,
        reqs=running.reqs,
    )
    assert positions == [1, 0] and indices.tolist() == [7, -1]
    assert running.mamba_track_indices is None
    overrides = [0, 1]
    env["set_mamba_track_indices_from_reqs"](running, overrides)
    assert running.mamba_track_buffer_indices == overrides
    assert running.mamba_track_buffer_indices is not overrides
    assert running.mamba_track_indices.tolist() == [6, -1]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
    env, settings = load_runtime(root / "python/sglang/srt")
    check_mix(env, nextn=False)
    check_mix(env, nextn=True)
    for mirror in TRACK_FIELDS[-2:]:
        check_mix(env, nextn=True, absent_mirror=mirror)
    check_freed_row(env)
    check_strategy_guard(env, settings)
    print("check_mixed_mamba_checkpoint: ok")


if __name__ == "__main__":
    main()
