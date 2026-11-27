"""Exercise the installed scheduler methods without importing the GPU stack."""

import ast
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock


class Generate:
    def __init__(self, rid):
        self.rid = rid


class Embedding(Generate):
    pass


class Batch(list):
    pass


ROOT = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else Path("/sgl-workspace/sglang")
SRC = ROOT / "python/sglang/srt/managers/scheduler.py"
LOG = Mock()
MESSAGE = "Scheduler idle: no requests in flight or waiting."


def scheduler_class():
    # Keep the real request dispatch, drain predicates, housekeeping and
    # paused path. Substitute only the services they call, not the edge logic.
    cls = next(n for n in ast.parse(SRC.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    methods = {"_tokenized_requests", "process_input_requests", "on_idle",
               "_report_fully_idle", "_record_scheduler_state_for_paused_engine",
               "is_fully_idle", "_pp_microbatches_drained"}
    cls.bases = []
    cls.decorator_list = []
    cls.body = [f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name in methods]
    for f in cls.body:
        f.decorator_list = [d for d in f.decorator_list if isinstance(d, ast.Name) and d.id == "staticmethod"]
    ns = dict(time=Mock(wraps=time), logger=LOG, _CLR_END="", List=list, TokenizedGenerateReqInput=Generate,
              TokenizedEmbeddingReqInput=Embedding, BatchTokenizedGenerateReqInput=Batch,
              BatchTokenizedEmbeddingReqInput=Batch, get_mm=lambda: NS(mm_feature_transport=None),
              get_parallel=lambda: NS(pp_size=1), get_disagg=lambda: NS(enable_pd_role_switch=False),
              DisaggregationMode=NS(NULL=0, PREFILL=1, DECODE=2), LOAD_STALL_REFRESH_S=1,
              SCHEDULER_STAGE_SANITY_CHECK_CACHE="check",
              is_health_check_generate_req=lambda r: getattr(r, "rid", "").startswith("HEALTH_CHECK"))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SRC), "exec"), ns)
    return ns["Scheduler"]


class IdleLog(unittest.TestCase):
    def setUp(self):
        s = self.s = scheduler_class()()
        s._idle_log_pending = False
        s._idle_log_deadline = None
        self.clock = s.on_idle.__globals__["time"].monotonic
        self.clock.return_value = 100.0
        s.running_batch = Mock()
        s.running_batch.is_empty.return_value = True
        s.chunked_req = s.last_batch = s.external_corpus_manager = None
        s.dllm_manager = Mock()
        s.dllm_manager.any_staging_reqs.return_value = False
        s.waiting_queue, s.result_queue = [], []
        s.grammar_manager = NS(grammar_queue=[])
        s.disaggregation_mode = 0
        s.enable_hisparse = s.enable_unified_memory = s.enable_hierarchical_cache = False
        s.enable_hicache_storage = s._engine_paused = False
        s.enable_overlap = True
        s._last_stall_publish_ts = self.clock()
        s.session_controller = Mock()
        s._request_dispatcher = Mock(return_value=None)
        s.flush_wrapper = Mock()
        s.maybe_send_health_check_signal = s.maybe_sleep_on_idle = Mock()
        s.metrics_reporter = Mock(is_stats_logging_rank=True)
        s.scheduler_stage_metrics = MagicMock()
        s.invariant_checker = Mock()
        s.invariant_checker._check_all_pools.return_value = False, []
        s.token_to_kv_pool_allocator = Mock()
        s.token_to_kv_pool_allocator.verify_byte_accounting.return_value = []
        s.pool_stats_observer = s.kv_events_publisher = s.new_token_ratio_tracker = Mock()
        s.publish_load_snapshot = s.load_publisher = s.load_inquirer = Mock()
        LOG.reset_mock()

    def advance(self, seconds):
        self.clock.return_value += seconds

    def test_repeat_idle_health_and_control(self):
        s = self.s
        for req in (None, object(), Generate("HEALTH_CHECK_1"), Batch([Embedding("HEALTH_CHECK_2")])):
            s.process_input_requests([] if req is None else [req])
            s.on_idle()
            self.advance(6)
            s.on_idle()
        LOG.info.assert_not_called()
        for i, req in enumerate((Generate("a"), Embedding("b"), Batch([Generate("HEALTH_CHECK_3"), Generate("c")])), 1):
            s.process_input_requests([req])
            s.on_idle()
            self.assertEqual(LOG.info.call_count, i - 1)
            self.advance(5)
            s.on_idle()
            self.advance(20)
            s.on_idle()
            self.assertEqual(LOG.info.call_count, i)
        LOG.info.assert_called_with(MESSAGE)

    def test_not_idle_until_all_work_drains(self):
        s = self.s
        s.process_input_requests([Generate("a"), Generate("b")])
        for queue in (s.waiting_queue, s.result_queue, s.grammar_manager.grammar_queue):
            queue.append(object())
            s.on_idle()
            queue.clear()
        s.chunked_req = object()
        s.on_idle()
        s.chunked_req = None
        s.last_batch = Mock()
        s.last_batch.is_empty.return_value = False
        s.on_idle()
        s.last_batch = None
        LOG.info.assert_not_called()
        s.on_idle()
        self.advance(5)
        s.on_idle()
        LOG.info.assert_called_once_with(MESSAGE)

    def test_gray_with_reset(self):
        s = self.s
        s._report_fully_idle.__globals__["_CLR_END"] = "\x1b[0m"
        s.process_input_requests([Generate("user")])
        s.on_idle()
        self.advance(5)
        s.on_idle()
        LOG.info.assert_called_once_with("\x1b[90m" + MESSAGE + "\x1b[0m")

    def test_pause_resume_and_rank(self):
        s = self.s
        s.process_input_requests([Generate("user")])
        s.waiting_queue.append(object())
        s._record_scheduler_state_for_paused_engine()
        LOG.info.assert_not_called()
        s.waiting_queue.clear()
        s._record_scheduler_state_for_paused_engine()
        self.advance(5)
        s.on_idle()
        LOG.info.assert_called_once_with(MESSAGE)
        s.metrics_reporter.is_stats_logging_rank = False
        s.process_input_requests([Embedding("second")])
        s.on_idle()
        self.advance(5)
        s.on_idle()
        LOG.info.assert_called_once_with(MESSAGE)
        s.metrics_reporter.record_scheduler_idle.assert_called()

    def test_followup_resets_deadline_even_if_it_finishes_in_one_iteration(self):
        s = self.s
        for request in (Generate("a"), Batch([Embedding("b")]), Generate("c")):
            s.process_input_requests([request])
            s.on_idle()
            self.advance(4.999)
            s.on_idle()
            LOG.info.assert_not_called()
        self.advance(.002)
        s.on_idle()
        self.advance(30)
        s.on_idle()
        LOG.info.assert_called_once_with(MESSAGE)

    def test_control_and_health_do_not_postpone_pending_notice(self):
        s = self.s
        s.process_input_requests([Generate("a")])
        s.on_idle()
        s.metrics_reporter.record_scheduler_idle.assert_called()
        for req in (object(), Generate("HEALTH_CHECK_1"), object(), Batch([Generate("HEALTH_CHECK_2")])):
            self.advance(1)
            s.process_input_requests([req])
            s.on_idle()
            LOG.info.assert_not_called()
        self.advance(1)
        s.on_idle()
        LOG.info.assert_called_once_with(MESSAGE)

    def test_interrupted_drain_requires_a_fresh_five_seconds(self):
        for paused in (False, True):
            with self.subTest(paused=paused):
                s = self.s
                poll = s._record_scheduler_state_for_paused_engine if paused else s.on_idle
                s.process_input_requests([Generate("a")])
                poll()
                self.advance(4)
                s.result_queue.append(object())
                poll()
                self.advance(10)
                s.result_queue.clear()
                poll()
                self.advance(4)
                poll()
                LOG.info.assert_not_called()
                self.advance(1)
                poll()
                LOG.info.assert_called_once_with(MESSAGE)
                LOG.reset_mock()


if __name__ == "__main__":
    unittest.main()
