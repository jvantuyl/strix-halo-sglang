#!/usr/bin/env python3
"""Patch 57: opt-in service-time cohort selection, native-safe row packing."""
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
    srt = root / "python/sglang/srt"
    changes = {}

    def replace(file, old, new):
        path = srt / file
        src = changes[path] if path in changes else path.read_text()
        assert src.count(old) == 1, (file, src.count(old), old[:100])
        changes[path] = src.replace(old, new, 1)

    sched = "managers/scheduler.py"
    policy = "managers/schedule_policy.py"
    tok = "managers/tokenizer_manager.py"
    io = "managers/io_struct.py"
    replace("arg_groups/fields/schedule.py", '                "shortest-prefill-first",',
            '                "shortest-prefill-first",\n                "fsched",')
    replace(policy, 'and get_schedule().schedule_policy == "shortest-prefill-first"',
            'and get_schedule().schedule_policy in ("shortest-prefill-first", "fsched")')
    # Keep the header separate from native priority and its preemption machinery.
    for anchor in ("    # Extra cache key for caller-defined request classification.\n    extra_key: Optional[Union[List[str], str]]",
                   "    # Extra cache key for caller-defined request classification.\n    extra_key: Optional[str]"):
        replace(io, anchor, "    fsched_weight: float = 1.0\n"+anchor)
    replace(io, "            priority=self.priority,\n            extra_key=",
            "            fsched_weight=self.fsched_weight,\n            priority=self.priority,\n            extra_key=")
    replace(tok, "        # Normalize the request\n        obj.normalize_batch_and_arguments()",
            """        if isinstance(obj, GenerateReqInput) and get_schedule().schedule_policy == "fsched":
            from sglang.srt.managers.fsched import header_weight
            headers = getattr(request, "headers", None)
            obj.fsched_weight = header_weight(headers.getlist("x-sglang-priority") if headers else [])

        # Normalize the request
        obj.normalize_batch_and_arguments()""")
    replace(tok, "                priority=obj.priority,\n                extra_key=",
            "                fsched_weight=obj.fsched_weight,\n                priority=obj.priority,\n                extra_key=")
    replace(sched, "        if recv_req.pp_prefetch_ticketed is True:\n",
            "        if self.fsched is not None:\n            req.fsched_weight = recv_req.fsched_weight\n\n"
            "        if recv_req.pp_prefetch_ticketed is True:\n")
    replace(sched, "        self.policy = SchedulePolicy(\n            self.schedule_policy,",
            """        self.fsched = None
        if self.schedule_policy == "fsched":
            from sglang.srt.managers.fsched import FairScheduler
            from sglang.srt.runtime_context import mamba_checkpoint_grid
            if not (
                self.enable_overlap and self.is_generation and self.is_mixed_chunk
                and self.chunked_prefill_size == 2048 and self.max_running_requests == 12
                and not self.prefill_decode_interval and self.spec_algorithm.is_eagle()
                and get_spec().speculative_num_steps == 3 and get_spec().speculative_eagle_topk == 1
                and get_spec().speculative_num_draft_tokens == 4
                and get_parallel().tp_size == get_parallel().pp_size == get_parallel().dp_size == 1
                and not self.enable_priority_scheduling and not self.enable_lora
                and not self.enable_hierarchical_cache and not self.enable_hisparse
                and not self.dllm_config and not self.enable_pdmux
                and get_disagg().disaggregation_mode == "null"
                and not getattr(self.tree_cache, "disable", True)
            ):
                raise ValueError("fsched requires overlap NEXTN 3/1/4, TP/PP/DP=1, cap12, "
                                 "2048 mixed, interval 0 and native prefix cache without preemption")
            self.fsched = FairScheduler(self.page_size, mamba_checkpoint_grid(self.page_size))
        self.policy = SchedulePolicy(
            "shortest-prefill-first" if self.fsched else self.schedule_policy,""")
    replace(sched, "            if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):",
            """            if (self.fsched is None or (last_batch is not None and
                    last_batch.forward_mode.is_extend() and last_batch.chunked_req is self.chunked_req)
                ) and self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):""")
    replace(sched, "        if self.dllm_config is not None:\n            new_batch = self.get_new_batch_dllm",
            "        if self.fsched is not None:\n            self.fsched.begin()\n"
            "        if self.dllm_config is not None:\n            new_batch = self.get_new_batch_dllm")
    anchor = "        # Prefill policy\n        # Get BLOCK_M"
    replace(sched, anchor, """        candidate_queue = self.waiting_queue
        allow_mixed = self.is_mixed_chunk
        fsched_chunk_limit = None
        if self.fsched is not None:
            from sglang.srt.managers.fsched import position_limit, simple
            pre = ([self.chunked_req] if self.chunked_req else []) + self.waiting_queue
            if all(simple(r) for r in pre) and all(simple(r) for r in running_batch.reqs):
                # calc_priority has refreshed native cache matches. Restore FIFO
                # for eligibility; only feasible work participates in the auction.
                fifo = sorted(self.waiting_queue, key=lambda r: r.time_stats.wait_queue_entry_time)
                slots = self.get_num_allocatable_reqs(running_bs, running_batch=running_batch)
                name, candidate_queue, fsched_chunk_limit = self.fsched.plan(
                    fifo, self.chunked_req, running_batch.reqs, slots)
                if name == "generation":
                    return None, running_batch
                allow_mixed = name == "mixed"
            else:
                # Keep incompatible features on the native pure-prefill path.
                allow_mixed = False
            tails = running_bs if allow_mixed else 0
            chunked_prefill_size = position_limit(tails) + tails

"""+anchor)
    replace(sched, "            running_bs if self.is_mixed_chunk else 0,",
            "            running_bs if allow_mixed else 0,")
    replace(sched, "            self.chunked_req = adder.add_chunked_req(self.chunked_req)",
            "            if fsched_chunk_limit is not None:\n"
            "                adder.chunked_req_limit = fsched_chunk_limit\n"
            "            self.chunked_req = adder.add_chunked_req(self.chunked_req)")
    replace(sched, "        # Get requests from the waiting queue to a new prefill batch\n        for req in self.waiting_queue:",
            """        # Get requests from the waiting queue to a new prefill batch
        for req in candidate_queue:
            if self.fsched is not None and adder.new_chunked_req is not None:
                break""")
    replace(sched, "        # Mixed-style chunked prefill\n        if (\n            self.is_mixed_chunk",
            "        # Mixed-style chunked prefill\n        if (\n            allow_mixed")
    replace(sched, "                self.future_map.resolve_seq_lens_cpu(batch)\n                if self._confidence",
            "                fsched_entry = self.fsched.snapshot(batch) if self.fsched else None\n"
            "                self.future_map.resolve_seq_lens_cpu(batch)\n                if self._confidence")
    anchor = """                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **fwd_kwargs
                        )"""
    replace(sched, anchor, """                        if fsched_entry is not None:
                            fs_start = self.device_module.Event(enable_timing=True)
                            fs_end = self.device_module.Event(enable_timing=True)
                            fs_start.record(stream=self.forward_stream)
"""+anchor+"""
                        if fsched_entry is not None:
                            fs_end.record(stream=self.forward_stream)
                            self.fsched.pending.append((fs_start, fs_end, fsched_entry))""")

    module = srt / "managers/fsched.py"
    assert not module.exists(), module
    changes[module] = (Path(__file__).parent / "fsched/runtime.py").read_text()
    for path, source in changes.items():
        compile(source, str(path), "exec")
    for path, source in changes.items():
        path.write_text(source)
    print("patch 57: installed opt-in fsched policy")


if __name__ == "__main__":
    main()
