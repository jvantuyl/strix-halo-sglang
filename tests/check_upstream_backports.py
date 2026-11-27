"""CPU integration checks for patch 59; run in the patched image."""

import ast
import inspect
import tempfile
import textwrap
import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile, HiCacheStorageConfig, PoolName, PoolTransfer, SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as asm
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool


class TestMixedCapacity(unittest.TestCase):
    def test_capacity_after_filter_preserves_decoder_for_next_round(self):
        # Execute the actual final admission block. The harness starts after the
        # prefill has allocated its slots; no model, GPU or policy simulation.
        method = ast.parse(textwrap.dedent(inspect.getsource(Scheduler._get_new_batch_prefill_raw))).body[0]
        branch, ret = method.body[-2:]
        self.assertIsInstance(branch, ast.If)
        self.assertIsInstance(ret, ast.Return)
        method.body = [branch, ret]
        method.args = ast.parse("def f(self, allow_mixed, new_batch, running_batch): pass").body[0].args
        method.returns = None
        namespace = {"ScheduleBatch": NS}
        exec(compile(ast.Module([method], []), "<mixed admission>", "exec"), namespace)
        dispatch = namespace[method.name]
        scheduler = NS(enable_overlap=True)
        pre = Mock(return_logprob=False, input_embeds=None, decoding_reqs=None)
        rows = [NS(beam_group=None)]
        running = Mock(reqs=rows, return_logprob=False)
        running.is_empty.return_value = False
        running.check_decode_mem.side_effect = [False, True]
        self.assertEqual(dispatch(scheduler, True, pre, running), (pre, running))
        running.prepare_for_decode.assert_not_called()
        pre.mix_with_running.assert_not_called()
        self.assertIsNone(pre.decoding_reqs)
        self.assertIs(running.reqs, rows)
        later = Mock(return_logprob=False, input_embeds=None)
        mixed, remainder = dispatch(scheduler, True, later, running)
        self.assertIs(mixed.decoding_reqs, rows)
        self.assertEqual(remainder.reqs, [])
        self.assertEqual([m[0] for m in running.method_calls],
                         ["is_empty", "filter_batch", "is_empty", "check_decode_mem"] * 2
                         + ["prepare_for_decode"])
        later.mix_with_running.assert_called_once_with(running)


def qsa(layers):
    pool = object.__new__(QSATokenToKVPool)
    pool.full_kv_pool = NS(layer_num=len(layers), size=64)
    pool.start_layer = min(layers)
    pool.use_mla = False
    pool.full_attention_layer_id_mapping = {n: i for i, n in enumerate(layers)}
    pool.qsa_compress_ratio = 4
    pool.qsa_compressed_k_buffer_pool = [
        (torch.arange(64, dtype=torch.bfloat16) + 100 * i).reshape(16, 1, 4)
        for i in range(len(layers))
    ]
    return pool


class TestQsaRestore(unittest.TestCase):
    def test_assembly_and_packed_file_restore(self):
        target, draft = qsa([4, 6]), qsa([0])
        anchor = NS(page_size=8, page_num=8, layer_num=1, size=64, logical_size=64,
                    layout="page_first_direct", pin_memory=False, device="cpu",
                    can_use_write_back_jit=False)
        mem = NS(hicache_size=0, hicache_ratio=1, hicache_mem_layout=anchor.layout,
                 hicache_io_backend="direct", hicache_write_policy="write_through",
                 hicache_host_memory_mode=None)
        params = MagicMock(page_size=8, mtp_draft_device_pools=(draft,))
        params.req_to_token_pool.mamba_map = {5: 0}
        cache = NS(cache_controller=NS(mem_pool_host=anchor, page_size=anchor.page_size))
        # The real machine may be below HiCache's multi-GB startup reserve.
        # This test allocates less than a KiB; do not depend on machine load.
        with patch("sglang.srt.mem_cache.memory_pool_host.host_memory_budget_bytes",
                   return_value=1024), patch.multiple(asm, build_kv_host_pool=Mock(return_value=anchor),
                            _build_mha_mla_host_pool=Mock(return_value=anchor),
                            HybridCacheController=Mock(), MambaPoolHost=Mock(),
                            get_memory=lambda: mem, _get_allocator_type=lambda: "default"):
            built = asm._MambaStrategy().build(cache=cache, kvcache=target, params=params,
                                               server_args=None, load_cache_event=None)
            specs, entries = asm.build_full_draft_pools(draft_kv_pool=draft, tree_cache=cache)
        self.assertIn(SidecarPoolSpec(PoolName.DRAFT_INDEXER, PoolName.KV), specs)
        self.assertIs(entries[-1].device_pool, draft)
        self.assertEqual(built.sidecars, [SidecarPoolSpec(PoolName.INDEXER, PoolName.KV)])
        entry = built.host_pool_group.get_entry(PoolName.INDEXER)
        self.assertEqual([entry.layer_mapper(i) for i in range(4)], [0, None, 1, 2])
        host = entry.host_pool
        saved = torch.stack([b[6:8].view(torch.uint8).reshape(-1).clone()
                             for b in target.qsa_compressed_k_buffer_pool + draft.qsa_compressed_k_buffer_pool])
        # Populate host page 2 from device page 3. Only the native DMA is
        # substituted; use the real packed page views, File backend and fence.
        def copy_to_host(*, src_ptrs, dst_ptrs, src_indices, dst_indices, page_size):
            for i, src in enumerate(src_ptrs):
                dst_ptrs[0][dst_indices, i, 0] = src[src_indices]

        with patch.object(memory_pool_host, "transfer_kv_all_layer_direct_lf_pf",
                          copy_to_host, create=True):
            host.backup_from_device_all_layer(target, torch.arange(16, 24),
                                              torch.arange(24, 32), "direct")
        config = HiCacheStorageConfig(0, 1, 0, 1, 0, 1, False, False, True, "cpu-qsa")
        transfer = PoolTransfer(PoolName.INDEXER, host_indices=torch.arange(16, 24), keys=["key"])
        with tempfile.TemporaryDirectory() as directory:
            writer = HiCacheFile(config, file_path=directory)
            writer.register_mem_host_pool_v2(host, PoolName.INDEXER)
            self.assertTrue(writer.set("key", torch.zeros(8, dtype=torch.uint8)))
            self.assertEqual(writer.batch_exists_v2(["key"], [transfer]).kv_hit_pages, 0)
            self.assertTrue(all(writer.batch_set_v2([transfer])[PoolName.INDEXER]))
            reader = HiCacheFile(config, file_path=directory)
            reader.register_mem_host_pool_v2(host, PoolName.INDEXER)
            self.assertEqual(reader.batch_exists_v2(["key"], [transfer]).kv_hit_pages, 1)
            host.kv_buffer.zero_()
            transfer.host_indices = torch.arange(8, 16)
            self.assertTrue(all(reader.batch_get_v2([transfer])[PoolName.INDEXER]))
        torch.testing.assert_close(host.kv_buffer[1, :, 0], saved, rtol=0, atol=0)
        def copy_to_device(*, src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size):
            dst_ptrs[0][dst_indices] = src_ptrs[0][src_indices, layer_id, 0]

        with patch.object(memory_pool_host, "transfer_kv_per_layer_direct_pf_lf",
                          copy_to_device, create=True):
            for i, pool in enumerate((target, target, draft)):
                host.load_to_device_per_layer(pool, torch.arange(8, 16), torch.arange(40, 48),
                                              i, "direct", is_draft=pool is draft)
        for i, (pool, layer) in enumerate(((target, 4), (target, 6), (draft, 0))):
            pool.layer_transfer_counter = Mock()
            keys = pool.get_qsa_compressed_k_buffer(layer)
            pool.layer_transfer_counter.wait_until.assert_called_once_with(layer - pool.start_layer)
            self.assertTrue(torch.equal(keys[10:12].view(torch.uint8).reshape(-1), saved[i]))


if __name__ == "__main__":
    unittest.main()
