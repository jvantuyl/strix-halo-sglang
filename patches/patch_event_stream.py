#!/usr/bin/env python3
"""Low-level scheduler/API event stream over ZMQ PUB (patch 35).

The journal only carries aggregate "Prefill batch" / "Decode batch" lines, so
concurrent requests cannot be told apart, prefill progress per request is
invisible and nothing says when a request arrived or finished. With
SGLANG_EVENT_STREAM_PORT=<port> the server publishes JSON events on two
PUB sockets bound to loopback (one per process; a single SUB socket can
connect to both):

    tcp://127.0.0.1:<port>    scheduler process
    tcp://127.0.0.1:<port+1>  API (tokenizer-manager) process

Every message is one JSON object with "ev" (event name) and "t" (unix time):

  prefill_batch   after each prefill chunk (same cadence as the log line):
                  new_seq, new_token, cached_token, replay_token, running_req,
                  queue_req, pending_token, input_tps, token_usage,
                  mamba_usage, cuda_graph, plus
                  running=[{rid,in,cached,done,out,age}], waiting=[{rid,in,wait}]
  decode_batch    every --decode-log-interval decode steps: running_req,
                  queue_req, gen_tps, accept_len, accept_rate, token_usage,
                  full_num_used, mamba_usage, cuda_graph, running=[...], waiting=[...]
  step            (SGLANG_EVENT_STREAM_STEPS=1) every forward: kind P|D,
                  bs, tokens, cached (P only), gap_ms since the previous step
  req_received    API: rid, in, stream        (prompt dispatched to the scheduler)
  req_first_token API: rid, ttft              (first output reached the API process)
  req_finished    API: rid, in, cached, out, reasoning, finish, ttft, e2e,
                  accept_len (speculative only)

rids are cut to 8 characters; per-request lists hold at most 32 running and
16 waiting entries (+"more": N). Sends are non-blocking with a high-water
mark, so a slow or absent subscriber never stalls the server; a bind failure
only logs a warning. Off when the port is unset. Anchors are asserted so an
upstream rewrite fails the build. See patches/35-event-stream.md.
"""
import os
import sys

SRT = "/sgl-workspace/sglang/python/sglang/srt"
MODULE = f"{SRT}/observability/event_stream.py"
REPORTER = f"{SRT}/managers/scheduler_components/metrics_reporter.py"
TOKENIZER = f"{SRT}/managers/tokenizer_manager.py"


def patch(text, old, new, count=1):
    assert text.count(old) == count, (text.count(old), old[:120])
    return text.replace(old, new)


MODULE_SRC = '''"""gfx1151 patch 35: low-level event stream over ZMQ PUB.

SGLANG_EVENT_STREAM_PORT=<port> enables it: the scheduler publishes on
tcp://127.0.0.1:<port>, the API process on <port+1>. One JSON object per
message with "ev" and "t" (unix time). SGLANG_EVENT_STREAM_STEPS=1 adds a
"step" event per forward. See patches/35-event-stream.md.
"""
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

PORT = int(os.environ.get("SGLANG_EVENT_STREAM_PORT", "0") or 0)
STEPS = PORT > 0 and os.environ.get("SGLANG_EVENT_STREAM_STEPS", "0").lower() in ("1", "true", "yes")
_ROLE_OFFSET = {"scheduler": 0, "api": 1}
_publishers = {}


class EventPublisher:
    def __init__(self, port, role):
        self.sock = None
        if port <= 0:
            return
        try:
            import zmq

            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.PUB)
            sock.setsockopt(zmq.SNDHWM, 20000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(f"tcp://127.0.0.1:{port}")
            self.sock, self.flag = sock, zmq.NOBLOCK
            logger.info("event stream (%s) publishing on tcp://127.0.0.1:%d", role, port)
        except Exception as e:
            logger.warning("event stream (%s) disabled: %r", role, e)

    @property
    def enabled(self):
        return self.sock is not None

    def emit(self, ev, **fields):
        if self.sock is None:
            return
        fields["ev"] = ev
        fields["t"] = time.time()
        try:
            self.sock.send(json.dumps(fields, separators=(",", ":"), default=str).encode(), self.flag)
        except Exception:
            pass  # a slow or absent subscriber must never stall the server


def get_publisher(role):
    """Per-process publisher for `role` ("scheduler" or "api"); created on first use
    so that importing this module in another process binds nothing."""
    pub = _publishers.get(role)
    if pub is None:
        pub = _publishers[role] = EventPublisher(PORT + _ROLE_OFFSET[role] if PORT else 0, role)
    return pub
'''


REPORTER_HELPERS = '''
    # gfx1151 patch 35: event stream (SGLANG_EVENT_STREAM_PORT).
    def _events(self):
        pub = getattr(self, "_event_pub", None)
        if pub is None:
            pub = self._event_pub = get_event_publisher("scheduler")
        return pub

    def _emit_step(self, kind, bs, tokens, cached=0):
        if not EVENT_STEPS or not self.is_stats_logging_rank:
            return
        pub = self._events()
        if not pub.enabled:
            return
        now = time.perf_counter()
        last = getattr(self, "_event_step_tic", None)
        self._event_step_tic = now
        pub.emit("step", kind=kind, bs=bs, tokens=tokens, cached=cached,
                 gap_ms=round((now - last) * 1000, 2) if last is not None else None)

    def _emit_batch_event(self, ev, batch, fields):
        if not self.is_stats_logging_rank:
            return
        pub = self._events()
        if not pub.enabled:
            return
        try:
            now = time.perf_counter()
            held = {}
            for b in (batch, self.scheduler.running_batch):
                for req in (getattr(b, "reqs", None) or []):
                    held.setdefault(req.rid, req)
            chunked = getattr(self.scheduler, "chunked_req", None)
            if chunked is not None:
                held.setdefault(chunked.rid, chunked)
            running = []
            for req in list(held.values())[:32]:
                n_in = len(req.origin_input_ids)
                n_out = len(req.output_ids)
                if n_out:
                    done = n_in
                elif getattr(req, "extend_range", None) is not None:
                    done = min(n_in, req.extend_range.end)
                else:
                    done = min(n_in, len(req.prefix_indices))
                entry = req.time_stats.wait_queue_entry_time
                running.append({"rid": str(req.rid)[:8], "in": n_in, "cached": req.cached_tokens,
                                "done": done, "out": n_out,
                                "age": round(now - entry, 2) if entry else None})
            waiting_q = list(self.scheduler.waiting_queue)
            waiting = []
            for req in waiting_q[:16]:
                entry = req.time_stats.wait_queue_entry_time
                waiting.append({"rid": str(req.rid)[:8], "in": len(req.origin_input_ids),
                                "wait": round(now - entry, 2) if entry else None})
            fields["running"] = running
            fields["waiting"] = waiting
            if len(held) > 32 or len(waiting_q) > 16:
                fields["more"] = max(0, len(held) - 32) + max(0, len(waiting_q) - 16)
            pub.emit(ev, **fields)
        except Exception as e:  # a view must never take the scheduler down
            logger.debug("event %s skipped: %r", ev, e)

    def report_prefill_stats(
'''


def patch_reporter():
    text = open(REPORTER).read()
    assert "_emit_batch_event" not in text, "patch 35 already applied"
    text = patch(
        text,
        "from sglang.srt.utils.device_timer import DeviceTimer\n",
        "from sglang.srt.observability.event_stream import STEPS as EVENT_STEPS\n"
        "from sglang.srt.observability.event_stream import get_publisher as get_event_publisher\n"
        "from sglang.srt.utils.device_timer import DeviceTimer\n",
    )
    text = patch(text, "\n    def report_prefill_stats(\n", REPORTER_HELPERS)
    # prefill: step event at the top (every chunk is a forward), batch event after the log line
    text = patch(
        text,
        "        now = time.perf_counter()\n"
        "        gap_latency = now - self.last_prefill_stats_tic\n"
        "        self.last_prefill_stats_tic = now\n",
        "        self._emit_step(\"P\", prefill_stats.num_new_seqs, prefill_stats.log_input_tokens,\n"
        "                        prefill_stats.log_hit_tokens)\n"
        "        now = time.perf_counter()\n"
        "        gap_latency = now - self.last_prefill_stats_tic\n"
        "        self.last_prefill_stats_tic = now\n",
    )
    text = patch(
        text,
        "        if self.is_stats_logging_rank:\n"
        "            logger.info(msg)\n"
        "        if self.current_scheduler_metrics_enabled:\n"
        "            self.metrics_collector.increment_prefill_cuda_graph_pass(\n",
        "        if self.is_stats_logging_rank:\n"
        "            logger.info(msg)\n"
        "        self._emit_batch_event(\"prefill_batch\", batch, {\n"
        "            \"new_seq\": prefill_stats.num_new_seqs,\n"
        "            \"new_token\": prefill_stats.log_input_tokens,\n"
        "            \"cached_token\": prefill_stats.log_hit_tokens,\n"
        "            \"replay_token\": prefill_stats.log_replay_tokens,\n"
        "            \"running_req\": prefill_stats.num_running_reqs.total,\n"
        "            \"queue_req\": len(self.scheduler.waiting_queue),\n"
        "            \"pending_token\": prefill_stats.num_pending_tokens,\n"
        "            \"input_tps\": round(self.last_input_throughput, 2),\n"
        "            \"token_usage\": round(pool_stats.full_token_usage, 4),\n"
        "            \"mamba_usage\": pool_stats.mamba_usage,\n"
        "            \"cuda_graph\": bool(can_run_cuda_graph),\n"
        "        })\n"
        "        if self.current_scheduler_metrics_enabled:\n"
        "            self.metrics_collector.increment_prefill_cuda_graph_pass(\n",
    )
    # decode: step event every iteration, batch event after the periodic log line
    text = patch(
        text,
        "        batch = running_batch or self.scheduler.running_batch\n"
        "\n"
        "        # Every-iteration work: realtime token counting + status logger\n",
        "        batch = running_batch or self.scheduler.running_batch\n"
        "        self._emit_step(\"D\", len(batch.reqs) if batch is not None else 0, num_generated_tokens)\n"
        "\n"
        "        # Every-iteration work: realtime token counting + status logger\n",
    )
    text = patch(
        text,
        "        if self.is_stats_logging_rank:\n"
        "            logger.info(msg)\n"
        "        if self.current_scheduler_metrics_enabled:\n"
        "            priority_enabled = self.scheduler.enable_priority_scheduling\n",
        "        if self.is_stats_logging_rank:\n"
        "            logger.info(msg)\n"
        "        self._emit_batch_event(\"decode_batch\", batch, {\n"
        "            \"running_req\": num_running_reqs,\n"
        "            \"queue_req\": len(self.scheduler.waiting_queue),\n"
        "            \"gen_tps\": round(self.last_gen_throughput, 2),\n"
        "            \"accept_len\": round(spec_accept_length, 3),\n"
        "            \"accept_rate\": round(spec_accept_rate, 3),\n"
        "            \"token_usage\": round(pool_stats.full_token_usage, 4),\n"
        "            \"full_num_used\": pool_stats.full_num_used,\n"
        "            \"mamba_usage\": pool_stats.mamba_usage,\n"
        "            \"cuda_graph\": bool(can_run_cuda_graph),\n"
        "        })\n"
        "        if self.current_scheduler_metrics_enabled:\n"
        "            priority_enabled = self.scheduler.enable_priority_scheduling\n",
    )
    open(REPORTER, "w").write(text)
    print("patched", REPORTER)


def patch_tokenizer():
    text = open(TOKENIZER).read()
    assert "get_event_publisher" not in text, "patch 35 already applied"
    # import next to the module's own observability imports
    anchor = "from sglang.srt.observability.req_time_stats import"
    assert text.count(anchor) >= 1, anchor
    first = text.index(anchor)
    text = (text[:first]
            + "from sglang.srt.observability.event_stream import get_publisher as get_event_publisher\n"
            + text[first:])
    text = patch(
        text,
        "            self._dispatch_to_scheduler(tokenized_obj)\n"
        "            self._mark_state_dispatched(tokenized_obj.rid)\n",
        "            self._dispatch_to_scheduler(tokenized_obj)\n"
        "            self._mark_state_dispatched(tokenized_obj.rid)\n"
        "            get_event_publisher(\"api\").emit(  # gfx1151 patch 35\n"
        "                \"req_received\", rid=str(tokenized_obj.rid)[:8],\n"
        "                **{\"in\": len(getattr(tokenized_obj, \"input_ids\", None) or ())},\n"
        "                stream=bool(getattr(tokenized_obj, \"stream\", False)))\n",
    )
    text = patch(
        text,
        "            if state.time_stats.first_token_time == 0.0:\n"
        "                state.time_stats.set_first_token_time()\n",
        "            if state.time_stats.first_token_time == 0.0:\n"
        "                state.time_stats.set_first_token_time()\n"
        "                get_event_publisher(\"api\").emit(  # gfx1151 patch 35\n"
        "                    \"req_first_token\", rid=str(rid)[:8],\n"
        "                    ttft=round(state.time_stats.get_first_token_latency(), 3))\n",
    )
    text = patch(
        text,
        "                del self.rid_to_state[rid]\n"
        "\n"
        "                # Mark ongoing LoRA request as finished.\n",
        "                self._emit_req_finished(rid, state, recv_obj, meta_info, i)  # gfx1151 patch 35\n"
        "                del self.rid_to_state[rid]\n"
        "\n"
        "                # Mark ongoing LoRA request as finished.\n",
    )
    text = patch(
        text,
        "    def _calculate_spec_decoding_metrics(\n",
        "    # gfx1151 patch 35: request completion event for the event stream.\n"
        "    def _emit_req_finished(self, rid, state, recv_obj, meta_info, i):\n"
        "        pub = get_event_publisher(\"api\")\n"
        "        if not pub.enabled:\n"
        "            return\n"
        "        try:\n"
        "            def col(name):\n"
        "                v = getattr(recv_obj, name, None)\n"
        "                return v[i] if v is not None else None\n"
        "            finish = col(\"finished_reasons\")\n"
        "            if isinstance(finish, dict):\n"
        "                finish = finish.get(\"type\", str(finish))\n"
        "            ts = state.time_stats\n"
        "            pub.emit(\"req_finished\", rid=str(rid)[:8],\n"
        "                     **{\"in\": col(\"prompt_tokens\")}, cached=col(\"cached_tokens\"),\n"
        "                     out=col(\"completion_tokens\"), reasoning=col(\"reasoning_tokens\"),\n"
        "                     finish=finish,\n"
        "                     ttft=round(ts.get_first_token_latency(), 3) if ts.first_token_time else None,\n"
        "                     e2e=round(ts.get_e2e_latency(), 3) if ts.finished_time else None,\n"
        "                     accept_len=meta_info.get(\"spec_accept_length\"))\n"
        "        except Exception as e:\n"
        "            logger.debug(\"req_finished event skipped: %r\", e)\n"
        "\n"
        "    def _calculate_spec_decoding_metrics(\n",
    )
    open(TOKENIZER, "w").write(text)
    print("patched", TOKENIZER)


def main():
    assert not os.path.exists(MODULE), f"{MODULE} exists"
    with open(MODULE, "w") as f:
        f.write(MODULE_SRC)
    print("wrote", MODULE)
    patch_reporter()
    patch_tokenizer()


if __name__ == "__main__":
    sys.exit(main())
