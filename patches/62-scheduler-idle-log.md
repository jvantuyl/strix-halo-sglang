# 62: Log after five quiet seconds

`Scheduler idle: no requests in flight or waiting.` is an INFO journal line,
emitted once by the stats-logging rank after generation or embedding work
has drained and the scheduler has stayed idle for at least five seconds.
The message uses gray (ANSI 90), a color not used by the other
scheduler status fields, and resets the color at its end.
`SGLANG_QWEN4_STATUS_COLOR=0` leaves it plain, like the other status lines.
The monotonic deadline starts at the existing fully-idle check, not at the
last forward or when the waiting queue alone empties. Queued, chunked and
overlapping work must drain first. A follow-up request cancels the deadline,
even if it finishes in one scheduler iteration. Control polling and synthetic
health probes alone do not arm it. A quiet server logs on the first idle
check at or after the deadline; the normal interruptible idle poll already
wakes once per second on cerebro. No timer thread, extra sleep, or delay
to incoming requests or idle metrics is needed.

`tests/check_scheduler_idle_log.py` exercises the actual request intake
and idle methods on a CPU-only fixture during patch replay. The patch
does not change the scheduler's sleep behavior or the service defaults.

**Original immediate-notice verification, 2026-11-27:** On patch 61 an eight-token
request drained to zero with no idle message. On patch 62 the same
request produced exactly one. A subsequent `/health_generate`, status
reads and quiet interval produced no more lines. With a 160-token
stream and an overlapping eight-token request, the short request finished
while the stream was still active: no idle line until both drained, then
exactly one. All three regression cases failed on the old image and
passed on the new one. Both patch profiles passed 128/128 replay checks.

The same restart doubled the deployment's PLE capacity. Both model
attach calls reported 2,097,152 warm-start keys, and the live hotset
grew to 2,130,666 rows during the smoke, above its old ceiling and with
zero cache errors. `/health`, native generation (32 tokens, ten verify
steps), and OpenAI chat passed. All 18 saved serving settings matched;
host RAM available was 6.37 GiB after the smoke. No automatic restart.
This verifies the larger cache operates, not its steady-state hit rate.

The earlier bright-white revision passed all four idle regression cases. The live
journal contained `ESC[97m` immediately before `Scheduler idle:` and
`ESC[0m` immediately after the final period, exactly once after the
eight-token smoke. Probing `/health_generate` did not repeat it. A fresh
process in the same image with the color opt-out emitted the plain text.
Native and OpenAI chat smoke tests passed again; cache sizing was unchanged.

Gray replaced white because white was hard to distinguish from normal text.
The production journal showed `ESC[90m` before the idle message and `ESC[0m`
after it. The idle line appeared once after generation and did not repeat
on health or load probes. The color opt-out still produced plain text.

**Five-second delay, production smoke:** On the immediate-notice image, two
eight-token requests separated by 0.65 s produced two notices, each about
2 ms after completion. On the delayed image, three requests with 0.65 s
and 4.1 s gaps produced only one notice, 5.010 s after the last response.
A separate burst re-armed it (5.011 s). A health probe and a further quiet
interval produced no duplicates. Native generation, chat and all 18 saved
serving settings passed; the 4,194,304-row PLE cache was unchanged.
