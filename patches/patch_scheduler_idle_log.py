#!/usr/bin/env python3
"""Patch 62: report sustained idle once after inference work."""

import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
    path = root / "python/sglang/srt/managers/scheduler.py"
    source = path.read_text()
    for old, new in (
        ("        self.forward_ct = 0\n",
         """        self.forward_ct = 0
        self._idle_log_pending = False
        self._idle_log_deadline = None
"""),
        ("        for recv_req in recv_reqs:\n            vmm_errors = None\n",
         """        for recv_req in recv_reqs:
            # Control RPCs and synthetic health probes may wake an idle scheduler.
            if any(
                not is_health_check_generate_req(req)
                for req in self._tokenized_requests(recv_req)
            ):
                self._idle_log_pending = True
                self._idle_log_deadline = None
            vmm_errors = None
"""),
        ("        fully_idle = self.is_fully_idle()\n        if not fully_idle:\n",
         "        fully_idle = self.is_fully_idle()\n        if not fully_idle:\n            self._idle_log_deadline = None\n"),
        ("        self.metrics_reporter.record_scheduler_idle()\n\n        if self.enable_unified_memory:",
         "        self._report_fully_idle()\n\n        if self.enable_unified_memory:"),
        ("""    def _record_scheduler_state_for_paused_engine(self) -> None:
        if self.is_fully_idle():
            self.metrics_reporter.record_scheduler_idle()
        else:
            self.metrics_reporter.record_scheduler_active()""",
         """    def _report_fully_idle(self) -> None:
        if self._idle_log_pending:
            # The normal interruptible idle poll revisits this gate even
            # without traffic. Never sleep here or delay the idle metrics.
            now = time.monotonic()
            if self._idle_log_deadline is None:
                self._idle_log_deadline = now + 5.0
            elif now >= self._idle_log_deadline:
                self._idle_log_pending = False
                self._idle_log_deadline = None
                if self.metrics_reporter.is_stats_logging_rank:
                    gray = "\\033[90m" if _CLR_END else ""
                    logger.info(f"{gray}Scheduler idle: no requests in flight or waiting.{_CLR_END}")
        self.metrics_reporter.record_scheduler_idle()

    def _record_scheduler_state_for_paused_engine(self) -> None:
        if self.is_fully_idle():
            self._report_fully_idle()
        else:
            self._idle_log_deadline = None
            self.metrics_reporter.record_scheduler_active()"""),
    ):
        assert source.count(old) == 1, f"idle log anchor moved: {old[:80]}"
        source = source.replace(old, new, 1)
    path.write_text(source)
    print("patch 62: log once after five seconds fully idle")


if __name__ == "__main__":
    main()
