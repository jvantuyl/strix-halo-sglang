"""Patch 58: run inside the image, with the repo mounted, no GPU or server."""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

# The HIP build probes device properties at import even in a CPU-only container.
if not torch.cuda.is_available():
    torch.version.hip = None

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components import metrics_reporter as mr
from sglang.srt.model_executor.forward_batch_info import ForwardMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import journal_stats


def make_batch(prefill_lens, n_decode):
    lens = [*prefill_lens, *([1] * n_decode)]
    reqs = [NS(rid=str(i), origin_input_ids=range(n), output_ids=[],
               prefix_indices=[], cached_tokens=0, time_stats=NS(wait_queue_entry_time=0))
            for i, n in enumerate(lens)]
    return ScheduleBatch(
        reqs=reqs, forward_mode=ForwardMode.MIXED if n_decode else ForwardMode.EXTEND,
        extend_lens=lens, extend_num_tokens=sum(lens), forward_iter=42,
        decoding_reqs=reqs[len(prefill_lens):] if n_decode else None,
    )


class ReportingTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.multiple(
            mr, LOG_FORWARD_ITERS=True, ENABLE_METRICS_DEVICE_TIMER=False,
            EVENT_STEPS=True, RECORD_STEP_TIME=False, get_disagg=lambda: NS(language_only=False),
            _ple_stats_msg_suffix=lambda: "",
        ))
        self.enterContext(patch.object(mr.time, "perf_counter", return_value=12.0))
        r = self.reporter = object.__new__(mr.SchedulerMetricsReporter)
        pool = Mock(full_token_usage=0.25, mamba_usage=0.2, full_num_used=64)
        pool.get_prefill_usage_msg_parts.return_value = ["token usage: 0.25"]
        pool.get_decode_usage_msg_parts.return_value = ["token usage: 0.25"]
        r.scheduler = NS(pool_stats_observer=NS(get_pool_stats=lambda: pool),
                         running_batch=make_batch([16, 32], 0), waiting_queue=[],
                         forward_ct=99, disaggregation_mode=None, kv_events_publisher=Mock(),
                         spec_algorithm=NS(is_none=lambda: True))
        r.is_stats_logging_rank = True
        r.current_scheduler_metrics_enabled = r.enable_mfu_metrics = False
        r._graph_backend_label = "backend"
        r.stats = NS()
        r._event_pub = Mock(enabled=True)

    def report(self, batch, n_pre=1, new=1856, replay=0):
        r = self.reporter
        stats = mr.PrefillStats(log_input_tokens=new, log_hit_tokens=64,
                                log_replay_tokens=replay, new_token_ratio=0.5,
                                num_running_reqs=NS(total=7), num_new_seqs=n_pre)
        r.last_prefill_stats_tic = 10.0
        r._event_pub.reset_mock()
        with self.assertLogs(mr.logger, "INFO") as logs:
            r.report_prefill_stats(batch, stats, False)
        step, event = r._event_pub.emit.call_args_list
        self.assertEqual((step.args, step.kwargs["kind"], step.kwargs["bs"],
                          step.kwargs["tokens"], step.kwargs["cached"]),
                         (("step",), "P", n_pre, new, 64))
        self.assertEqual(event.args, ("prefill_batch",))
        self.assertEqual(event.kwargs["new_token"], new)
        self.assertEqual(event.kwargs["input_tps"], (new + replay) / 2)
        self.assertEqual(event.kwargs["running_req"], 7)
        return logs.records[0].getMessage(), event.kwargs

    def test_mixed(self):
        line, fields = self.report(make_batch([1856], 3))
        self.assertTrue(line.startswith("Mixed batch [42], prefill rows/positions: 1/1856, "
                                        "decode rows/positions: 3/3, total positions: 1859, "), line)
        expected = dict(batch_mode="mixed", prefill_rows=1, decode_rows=3,
                        prefill_positions=1856, decode_positions=3, total_positions=1859)
        self.assertEqual({k: fields[k] for k in expected}, expected)
        self.assertEqual(journal_stats.PREFILL_RE.search(line).groups(), ("1856", "928.00"))

    def test_overlap_and_one_position_prefill(self):
        batch = make_batch([1, 127], 2)
        result = batch.copy()
        batch.forward_iter, batch.forward_mode = 99, ForwardMode.DECODE
        batch.reqs, batch.decoding_reqs, batch.extend_lens = [], None, None
        self.reporter.scheduler.running_batch = batch
        line, fields = self.report(result, n_pre=2, new=124, replay=4)
        self.assertIn("Mixed batch [42], prefill rows/positions: 2/128, "
                      "decode rows/positions: 2/2, total positions: 130", line)
        self.assertEqual((fields["prefill_positions"], fields["replay_token"],
                          fields["cached_token"]), (128, 4, 64))

    def test_pure_prefill_and_missing_batch(self):
        batch = make_batch([12], 0)
        batch.return_logprob = True  # native pure fallback despite running requests
        for b, it in [(batch, 42), (None, 99)]:
            line, fields = self.report(b, new=12)
            self.assertEqual(line, f"Prefill batch [{it}], #new-seq: 1, #new-token: 12, "
                             f"#cached-token: 64, token usage: 0.25, {mr._CLR_RUN}#running-req: 7"
                             f"{mr._CLR_END}, {mr._CLR_QUEUE}#queue-req: 0{mr._CLR_END}, "
                             f"{mr._CLR_PEND}#pending-token: 0{mr._CLR_END}, backend: False, "
                             f"{mr._CLR_PRE}input throughput (token/s): 6.00{mr._CLR_END}")
            self.assertNotIn("batch_mode", fields)
            self.assertNotIn("total_positions", fields)

    def test_pure_decode(self):
        r = self.reporter
        batch = make_batch([1], 0)
        batch.forward_mode = ForwardMode.DECODE
        r.forward_ct_decode = r.decode_log_interval = 1
        r.last_decode_stats_tic, r.num_generated_tokens = 10, 6
        with self.assertLogs(mr.logger, "INFO") as logs:
            r.report_decode_stats(False, batch, 6)
        line = logs.records[0].getMessage()
        self.assertTrue(line.startswith("Decode batch [42], "), line)
        self.assertNotIn("positions", line)
        self.assertIsNone(journal_stats.PREFILL_RE.search(line))
        step, event = r._event_pub.emit.call_args_list
        self.assertEqual((step.kwargs["kind"], event.args), ("D", ("decode_batch",)))
        self.assertNotIn("batch_mode", event.kwargs)

    def test_journal_totals_with_both_labels(self):
        pure, _ = self.report(make_batch([12], 0), new=12)
        with patch.object(mr, "LOG_FORWARD_ITERS", False):
            mixed, _ = self.report(make_batch([1856], 3))
        self.assertTrue(mixed.startswith("Mixed batch, "), mixed)
        journal = NS(stdout="\n".join([pure, mixed, "Decode batch, accept len: 3.12, "
                                       "accept rate: 0.71, gen throughput (token/s): 25.0"]))
        with patch.object(sys, "argv", ["journal_stats.py", "--json"]), \
             patch.object(journal_stats.subprocess, "run", return_value=journal), \
             redirect_stdout(io.StringIO()) as stdout:
            journal_stats.main()
        epoch, = json.loads(stdout.getvalue())
        self.assertEqual((epoch["prefill_tokens"], epoch["prefill_tp"], epoch["gen_tp"]),
                         (1868, [6.0, 928.0], [25.0]))


if __name__ == "__main__":
    unittest.main()
