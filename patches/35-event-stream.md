# Patch 35: low-level event stream over ZMQ

**Files:** `python/sglang/srt/observability/event_stream.py` (new),
`python/sglang/srt/managers/scheduler_components/metrics_reporter.py`,
`python/sglang/srt/managers/tokenizer_manager.py`
**Script:** [`patch_event_stream.py`](patch_event_stream.py)
**Switch:** `SGLANG_EVENT_STREAM_PORT=<port>` (off when unset);
`SGLANG_EVENT_STREAM_STEPS=1` adds a per-forward `step` event

## Why

The journal carries aggregates only: `Prefill batch, ... #running-req: 3,
#queue-req: 1` and `Decode batch, ... gen throughput`. Nothing in it says
which requests are in flight, how far each prompt's prefill has gone, how
many tokens each has produced, when a request arrived or when and why it
finished. Concurrent requests cannot be told apart after the fact and a live
view can only show totals. `/get_load` has the same limitation and only shows
the present.

The server now publishes structured events as they happen. ZMQ PUB/SUB was
chosen over a websocket endpoint because SGLang already moves everything
between its processes over ZMQ, the scheduler lives in a different process
from the HTTP server (so a websocket would have needed this hop anyway), and
a PUB socket costs the publisher nothing when nobody listens.

## Transport

Two PUB sockets bound to loopback, one per process, so no cross-process hop
is needed; a single SUB socket connects to both:

| endpoint | process | events |
|---|---|---|
| `tcp://127.0.0.1:<port>` | scheduler | `prefill_batch`, `decode_batch`, `step` |
| `tcp://127.0.0.1:<port+1>` | API (tokenizer manager) | `req_received`, `req_first_token`, `req_finished` |

With `--network=host` (the launcher's default) the host reaches them at the
same addresses. Sends are `NOBLOCK` behind a 20k-message high-water mark, so a
slow or absent subscriber never stalls the server; a bind failure (port taken)
logs one warning and disables that publisher. Subscribers that connect late
miss earlier events, so a client has to treat rids it first meets inside a
batch snapshot as "already running when I connected".

## Events

One JSON object per message, always with `ev` and `t` (unix time, float).
rids are cut to 8 characters.

`prefill_batch` (after every prefill chunk, same cadence as the log line):
`new_seq, new_token, cached_token, replay_token, running_req, queue_req,
pending_token, input_tps, token_usage, mamba_usage, cuda_graph`, plus the
per-request lists below.

`decode_batch` (every `--decode-log-interval` decode steps, 40 by default):
`running_req, queue_req, gen_tps, accept_len, accept_rate, token_usage,
full_num_used, mamba_usage, cuda_graph`, plus the per-request lists.

Per-request lists on both batch events:

```
running: [{rid, in, cached, done, out, age}, ...]   (at most 32)
waiting: [{rid, in, wait}, ...]                     (at most 16)
more: N                                             (only when truncated)
```

| field | meaning |
|---|---|
| `in` | prompt length (tokens) |
| `cached` | prompt tokens served from the radix cache |
| `done` | prompt tokens prefilled so far (`extend_range.end` while chunking; `in` once decoding) |
| `out` | tokens generated so far |
| `age` | seconds since the request entered the waiting queue |
| `wait` | seconds in the waiting queue so far |

"Running" covers the running batch, the prefill batch just run and the
chunked request stashed between its chunks.

`step` (only with `SGLANG_EVENT_STREAM_STEPS=1`, every forward):
`kind` (`P` or `D`), `bs`, `tokens` (prompt tokens computed for P, tokens
generated for D), `cached` (P only), `gap_ms` since the previous step event.
At bs=1 decode this is ~10 messages/s.

`req_received`: `rid, in, stream`, emitted when the API process dispatches
the tokenized prompt to the scheduler.
`req_first_token`: `rid, ttft` (seconds from request creation to the first
output reaching the API process).
`req_finished`: `rid, in, cached, out, reasoning, finish` (`stop`, `length`,
`abort`, ...), `ttft, e2e, accept_len` (speculative decoding only).

## Cost and safety

Batch events add a dict walk over the held requests per stats line. Step
events add one small `json.dumps` and a non-blocking send per forward. Both
batch-event bodies are wrapped in `try/except` (DEBUG log on failure) so a
surprise in a request object cannot take the scheduler down. Nothing runs
with the port unset: `get_publisher()` returns a disabled publisher and every
`emit` is a single attribute check.

## Deploy

```
SGLANG_DOCKER_ARGS=-e SGLANG_EVENT_STREAM_PORT=8010 -e SGLANG_EVENT_STREAM_STEPS=1
```

in the service env file (the launcher word-splits `SGLANG_DOCKER_ARGS` into
`docker run`). Consumer: `~/scratch/sgl_top.py` (pyzmq SUB on 8010 and 8011,
`--record events.jsonl` keeps the raw stream for later analysis).
