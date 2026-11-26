# Running Qwen3.8-Flash-Next on Strix Halo

Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) is a 48-layer hybrid
GDN + sparse-attention (QSA) MoE with a vision tower and a 51B-parameter
n-gram "PLE" embedding table. The resident part of the AWQ-INT4 checkpoint is
~75 GiB; the PLE table alone is 95 GiB in bf16. It fits a 128 GB Strix Halo
only because SGLang can keep the PLE table **off the GPU and out of RAM**,
reading rows on demand from a file. This page is the runbook for that setup
on the fork image (`strix-halo-sglang:dev`).

Tested checkpoint: `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4` (compressed-tensors,
group 32, asymmetric; vision tower unquantized). MTP is optional (see
[Speculative decoding](#speculative-decoding-mtp)); the launcher leaves it
off.

## Current scheduling trial on cerebro

The operator-selected trial for `qwen38-flash-next` on port 8000 is
[fsched](../patches/57-fsched.md), with fixed **2:1 prefill:generation
service time** during contention and a **0.5× switch band**. Adaptive
feedback and per-forward debug tracing are off. There is no expiry timer.
The generic launcher defaults are unchanged.

The user-unit drop-in `96-fsched57.conf` loads `~/.config/qwen38-sglang.env.fsched57`
after the existing draft, TunableOp and native-Q8 settings. Only two values
are overridden:

* `SGLANG_IMAGE` pins the tested local image by ID:
  `sha256:6501a084b5c4bb8a605c13ef3092748ae5a3615592ea4e393c95d1afe0b8f165`
  (`strix-halo-sglang:fsched-57-trial` on this host).
* `QWEN38_EXTRA_ARGS` keeps NEXTN 3/1/4, the presharded draft path and cache
  reporting. Scheduling flags are `--schedule-policy fsched --enable-mixed-chunk
  --chunked-prefill-size 2048 --prefill-decode-interval 0`. Read these back
  from `/get_server_info`; the image tag alone does not select the policy.

Use [the isolated control measurements](../bench/2026-11-26-fsched-server.md)
to interpret live traffic. In the 18+2 benchmark short-request median TTFT
**increased** from 1.48 s to 1.68 s. Compare TTFT, streaming gaps and long
prefill latency separately; an average decode rate cannot distinguish them.

**Rollback:** after checking for connected clients, in-flight requests and
jobs inside `sglang-qwen38`, arrange an approved restart. Archive only
`~/.config/systemd/user/qwen38-sglang.service.d/96-fsched57.conf` outside
the drop-in directory, run `systemctl --user daemon-reload`, then restart
`qwen38-sglang.service`. Leave `10`, `89` and `95` in place. Check
`/health` and verify `shortest-prefill-first`, chunk 1024, mixed off and
interval 1 in `/get_server_info`. No cache rebuild or checkpoint swap is
part of the rollback.

## What the image adds

| Patch | What | Why on gfx1151 |
|---|---|---|
| [11](../patches/11-qwen4-exp-rocm.md) | CPU-side PLE gather, Triton QSA decode route, safe top-k fallback, QSA smem schedule | GPU must not touch host memory; SM121-only kernel gates; JIT top-k reads OOB; 64 KB workgroup smem cap |
| [12](../patches/12-wna16-triton-zp.md) | Pass zero points to the Triton WNA16 MoE kernel; load-time zp transpose; GC between MoE layers in post-load | Asymmetric AWQ; without it every expert weight is off by `(8 - zp) * scale`. Without the GC the old expert weights of every converted layer stay allocated (~1.4 GiB each) and the 48-layer load OOMs |
| [13](../patches/13-ple-table-reuse.md) | Reuse the file-backed PLE table across boots | Upstream rewrites the 48 GiB table from the checkpoint on every start; a fingerprinted marker lets later boots skip the PLE shards |
| [14](../patches/14-cuda-graph-ple.md) | Decode CUDA graphs with the CPU-side PLE gather | The host gather cannot be captured; fill the static PLE prefetch buffer from the host before each replay |
| [15](../patches/15-qsa-graph-scratch.md) | Dedicated QSA packed-KV scratch for captured graphs | Upstream shares one growable scratch between graphs and eager decode; an eager step above the graph range re-allocates it and the graphs write into freed memory (GPU page fault after the next `empty_cache`). Not gfx1151-specific |
| [16](../patches/16-wna16-rocm-dense.md) | Dense compressed-tensors int4 Linear on ROCm: dequantize to bf16 at load, serve with `F.linear` | The dense WNA16 scheme is Marlin-only and Marlin is CUDA-only (`NameError: gptq_marlin_repack`); needed by checkpoints that also quantize attention `q/k/v/o`, e.g. the [abliterated variant](#running-the-abliterated-variant-derisked) |
| [17](../patches/17-moe-config-dir.md) | `SGLANG_MOE_CONFIG_DIR` searched before the builtin MoE tile tree (flat or tree layout, missing dirs skipped) | Upstream's knob replaces the tree and crashes on a missing dir; tuned tiles are now mounted per checkpoint instead of baked into the image |
| [18](../patches/18-hc-mix-rocm.md) | Atomics-free two-launch HyperConnection mix for decode batches, used on HIP | The sm_100 JIT mix is unavailable, so every decode step ran the persistent kernel whose split-K `atomic_add` made greedy decode differ run to run (and whose software grid barrier assumes co-resident CTAs). Same speed, bit-identical |
| [19](../patches/19-moe-wna16-kmask.md) | GPTQ/AWQ MoE kernel masks the packed-weight load on a partial last K block | Unmasked, it read past the last expert's rows: a layout-dependent GPU page fault (killed the tuner on the g128 checkpoint). Runtime shapes are even multiples, so serving is unchanged |
| [20](../patches/20-qsa-decode-topk.md) | Decode QSA block selection uses a graph-capturable torch top-k on HIP instead of the JIT kernel | `select_decode_tokens` bypassed patch 11's guard; the JIT kernel is unsafe here and its output order varies past 512 blocks (long-context decode drift). ~1–1.5 ms per decode step |
| [21](../patches/21-qsa-topk-ties.md) | Tie-stable QSA block selection on HIP: stable sort for prefill rows, top-k over unique score+index keys for decode, ties toward the lower block | `torch.topk` orders tied entries differently per launch on this ROCm build and the indexer's relu scores tie constantly, so prefill above ~1.4k tokens (and patch 20's decode) still drifted bit-wise. Also replaces the per-row Python loop in prefill (119 → 3.5 ms per QSA layer at 1.5k tokens) |
| [22](../patches/22-spec-draft-greedy.md) | The post-prefill draft extend proposes through `sample_draft_proposal` (argmax for greedy rows) under rejection sampling | HIP defaults EAGLE/NEXTN to rejection sampling; that one draft pass called `fast_sample` directly, so the first draft token was random at `temperature=0` and greedy MTP output differed run to run. Not gfx1151-specific |
| [23](../patches/23-qsa-mtp-tail.md) | MTP shared block selection places the drafted positions right after the captured entries, not after the `-1` padding | The KV gather packs valid entries as a prefix (count, not mask), so the drafted tokens were dropped and the packed slot left unwritten: stale scratch here, zeros on the paged path. Draft never saw the newest token; accept length 2.3 → 2.55 on short prompts. Not gfx1151-specific |
| [24](../patches/24-qsa-mqa-triton.md) | Length-bounded Triton kernel for the QSA indexer's decode block scoring, dispatched when TileLang is absent (`SGLANG_QSA_MQA_TRITON=0` forces the reference) | No TileLang here, so the torch reference ran: it gathers the whole `context_length / 4` window per row on every decode step of all 12 QSA layers, 6 ms per layer at bs 20 with a 131k context. That was the whole 99 → 71 tok/s drop at 16–20 streams when the default context grew from 32k; back to 89–105 at 131k. Same on any CUDA build without TileLang |
| [25](../patches/25-qsa-topk-slices.md) | Prefill block selection sorts row slices into a preallocated output | Patch 21's whole-chunk stable sort was ~1 GiB of transient buffers per QSA layer, 40% of the prefill VRAM peak on a box idling at 90 of 96 GiB (MTP, 20 requests). Identical indices |
| [26](../patches/26-qsa-mqa-prefill-triton.md) | `tl.dot` prefill MQA for the indexer, logits the only allocation | Prefill twin of patch 24: the reference `einsum` materialises per-head scores (4× the logits) plus three copies, ~1.15 GiB per layer per chunk. 2.9e-6 from the reference, 5× faster, prefill transient flat in prompt length |
| [27](../patches/27-host-parked-params.md) | Token embedding and vision tower parked in pinned host memory (`SGLANG_HOST_PARKED_PARAMS`) | Both are read a few KB per step or not at all, yet held 2.0 GiB of a carve-out idling at 90 of 96 GiB. Exactly sized `hipHostMalloc` aliased as CUDA tensors; worst-case headroom 1.6 → 3.3 GiB, output and throughput unchanged |
| [28](../patches/28-ple-short-conv-packed.md) | PLE short conv over the packed prefill batch instead of a `[requests, longest, 10240]` padded layout | Chunked prefill mixing many short prompts with a slice of a long one made three copies of that layout: 6.3 GiB for 7,231 tokens, 9.6 GiB possible at the cap, 714 ms per layer; the benchmark suite's mixed scenario reached 117 MiB free. Packed: bit-identical, 431 MiB, 60× faster |
| [29](../patches/29-default-effort-override.md) | A request's `reasoning_effort` beats `--default-chat-template-kwargs` | Upstream pops the request's effort out of its `chat_template_kwargs`, refills the slot with the server default, then merges the kwargs over the request field: with the launcher's `medium` default every request rendered at `medium` (`prompt_tokens` identical for low / medium / xhigh). Not gfx1151-specific |
| [30](../patches/30-draft-mtp-shards.md) | MTP draft loads only the shards that hold `mtp` tensors | The draft's `load_weights` drops every other name, yet its loader walked all 34 shards (222k expert tensors, 26 PLE shards) for 31 tensors in three files: 49 s of the boot. `weight_files_to_skip` on the MTP class, from the safetensors index |
| [31](../patches/31-presharded-host-tables.md) | `--load-format presharded` (post-processed weight dump, copied back on later boots) with the host PLE table | The 47.7 GB file-backed table is a `Parameter` the dump would hash and rewrite; its completion marker is only checked from `load_weights`, which the reload never calls; `load_weights` ends with the GDN in_proj fusion. Skip host tensors, check the marker first (normal load otherwise), fuse after the copy. The dump is named by an explicit cache version (`QWEN38_PRESHARDED_VERSION`, bumped when a patch changes the post-processing) and a hash of the model's `config.json`, so an image rebuild keeps it; an interrupted or mismatched dump is removed and redone, other subfolders only reported; the weight load, hash, write and copy-back log progress every 15 s. See Load time below |
| [32](../patches/32-control-token-encoder.md) | Chat prompts encoded with control tokens and content kept apart: the template marks the control tokens it emits, a split encoder tokenizes everything else with the added-token table emptied | Rendered to one string and tokenized whole, a tool result quoting `<|im_end|>` or `</think>` *is* the control token: strict thinking masks it, the turn structure breaks, the model never sees the quote (the early `</think>` under agents). Ids unchanged when nothing is quoted; inert without the marked template. Not gfx1151-specific |
| [33](../patches/33-control-token-parsers.md) | The qwen3 reasoning parser and the qwen3_coder tool parser switch state on control token ids, not on the same strings spelled out | Both parse detokenized text, so a `</think>` the model writes with ordinary tokens ended the thinking block and a spelled-out `<tool_call>` opened a call. A per-choice stream decodes the output ids beside the text; the parsers search a view with look-alikes defused. Text fallback with one warning if ids and text disagree. Not gfx1151-specific |
| [34](../patches/34-control-token-media.md) | Requests with images or video keep patch 32's prompt ids: the Qwen-VL processor accepts ids, runs the media pipeline on a placeholder-only text derived from them and expands the placeholders inside the original ids with SGLang's id-level builder | With media the endpoint sent the rendered text and the processor re-tokenized it with the added tokens enabled, so a quoted `<|im_end|>` next to a screenshot was the control token again. The id-level expansion is checked token for token against the processor's own result per request; text path with one warning on any disagreement. Not gfx1151-specific |
| [35](../patches/35-event-stream.md) | Low-level event stream over ZMQ PUB (`SGLANG_EVENT_STREAM_PORT`): batch events with per-request lists from the scheduler, request received/first-token/finished from the API process, optional per-forward step events | The journal only has aggregates, so concurrent requests cannot be told apart and a live view shows totals only; non-blocking sends cost nothing without a subscriber. Consumer: `~/scratch/sgl_top.py`. Not gfx1151-specific |
| [36](../patches/36-tool-param-recovery.md) | `qwen3_coder` tool-call parameters: an array/object value that stopped one or two closers short is closed and typed (only when it ends outside a string, opened everything it closes and the result has the schema's type); `get_schema_properties()` merges union branches with top-level properties and resolves local `$ref` | The parser returned such values as strings, so the client reported `Expected object, received string` for a payload whose only fault was a missing `}` (14 of 18 journal cases since 2026-11-01); a property hidden behind `oneOf` or `#/$defs` was "not defined" and returned as text. Values broken mid-structure stay strings, with the same warning as before. `tests/check_tool_param_recovery.py`, 137 checks. Not gfx1151-specific |
| [37](../patches/37-qsa-cpu-seq-lens.md) | QSA declares `needs_cpu_seq_lens`, so the speculative forwards get the scheduler's per-round host copy and `_speculative_max_row_length` takes its sync-free branch | With the opt-out the overlap FutureMap published no `seq_lens_cpu` for spec batches, and every target-verify / draft-extend forward fell back to `int(sequence_lengths.max())` on the device — a stream sync + D2H per forward, 4× per MTP round. The host copy already existed (the eagle worker moves it once per round), so this only routes it where it was missing. Not gfx1151-specific |
| [38](../patches/38-hc-mix-launch-tuning.md) | Split hc-mix down kernel launch: `num_warps` 8→2 + `num_stages=2` (grid constants and math unchanged) | The patch-18 kernel inherited the persistent kernel's launch metadata; at the model's shapes (lowrank 320) the down+up pair ran 56 µs back-to-back vs a ~15 µs/launch floor — five chained `tl.dot`s with 8 warps on a 16×32 tile and no load/dot overlap. Launch retuning alone: 27 µs/pair, same reduction order, bit-identical across repeat launches. Measured end-to-end on production: decode gap p50 135→131 ms, 220-token wall 13.78→13.32 s (−3.3%). `tests/check_hc_mix_launch.py` |
| [39](../patches/39-qsa-host-metadata.md) | QSA speculative metadata builds from host-published lengths (`extend_seq_lens_cpu`, `req_pool_indices_cpu`); no `.item()`/`.tolist()` device readbacks in the hot paths | py-spy on the scheduler put 75% of its busy time in the draft-extend metadata chain, whose `.item()` syncs the plan stream against whatever forward is in flight — ~1 s per 1024-token prefill chunk (thread-bound prefill, the amdgpu_top oscillation) and a share of every decode round. The mapping is now built host-side and moved once; values identical, no numerics change. Measured: 64k prefill 98→82 s (tok/s p50 655→712); decode probes unchanged (GPU-bound). `tests/check_qsa_host_metadata.py` |
| [40](../patches/40-qsa-rope-host-bound.md) | QSA indexer rope cache sized from a monotone host-side bound (`QSA_HOST_MAX_POSITION`); drops `positions.max().item()` from `project_qk`/`apply_rope` | With 39 fixed, py-spy's next prefill dominator was the indexer rope path at 83% of the scheduler thread: a device reduction + D2H sync per QSA layer per prefill chunk just to ask how large the lazy rope cache should be. The backends now record `max(seq_lens_cpu) + draft_window` in both metadata entry points; the indexer ensures against that. Cache contents identical, fallback preserved. `tests/check_qsa_rope_host_bound.py` |
| [41](../patches/41-qsa-prefill-host-mirrors.md) | QSA prefill MQA inputs from host mirrors: `QSAIndexerMetadata` carries `sequence_lengths_cpu` (non-spec branch only), chunk-prefill kernel grid bound comes from `max(extend_seq_lens_cpu)` | Sequential per-layer syncs mask each other — after 40, py-spy's next prefill dominator was `get_prefill_mqa_inputs`'s `sequence_lengths.tolist()` at 82% of the thread, plus the chunk kernel's `cu_q` readback for its grid bound, both per QSA layer per prefill chunk. Both read host-published values now; speculative-paged metadata deliberately keeps the mirror None (its rows are per-token). `tests/check_qsa_prefill_host_mirrors.py` (exercises the real method against a fake pool) |
| [42](../patches/42-presharded-load-readahead.md) | Presharded reload stages each file into an aligned pageable buffer one read ahead (`os.preadv` + O_DIRECT, buffered fallback ladder); large GPU copies go through one fixed 64 MiB pinned stage block | Reload ran at 758 MiB/s vs a 4.2 GiB/s device: `get_tensor` returns zero-copy mmap views, so all page-in happened inside the GPU copy's pin-fault path (0.79 GiB/s measured; H2D from pinned memory is 66 GiB/s; fadvise WILLNEED is a no-op on this kernel). Pageable staging instead of pinning the whole file because pinned host allocations are charged against device memory on this UMA box; the fixed stage block avoids torch's per-size pinned cache, which left ~3 GiB behind after one reload. Reload 101 s → ~26 s. A prepark variant (fill the parking set from the staged views before the reload) was rejected: it broke MTP draft acceptance to 0.00 — the draft shares embedding storage with the target. `tests/check_presharded_load_readahead.py` |
| [43](../patches/43-umem-carve-budget.md) | KV budget from the VRAM carve: when sysfs shows a dedicated carve (`mem_info_vram_total` > 1.5× system RAM), `get_available_gpu_memory` returns carve free minus a 2 GiB reserve; psutil MemAvailable stays the fallback for true-UMA devices | Upstream assumes integrated = sysmem-as-device-mem; here 96 GiB is carved for the iGPU and device allocations never appear in Linux page accounting, while `mem_get_info` inside the container reports the 8 GiB GTT overflow window. The psutil metric made the KV budget track Linux page-cache noise at the instant of measurement (477k mmap vs 270-304k O_DIRECT for identical device footprints). Carve accounting needs `SGLANG_MEM_FRAC=0.985` (the weights are 86.7% of the carve; the old 0.85 default's sanity check was vacuous under the old metric) and lifts `max_total_num_tokens` to the configured 524288 cap (509504 aligned). `tests/check_umem_carve_budget.py` |
| [45](../patches/45-ple-hit-stats.md) | `PLE:` segment on the Prefill status line: rows asked of the table, distinct rows, distinct 4 KiB pages, and how many of those pages `mincore(2)` reports resident in the page cache at gather time | The PLE table is read through a file-backed mmap and prefill cost depends on row-cache residency, which was previously only measurable out-of-band. Residency is kernel ground truth (verified in `tests/check_ple_hit_stats.py`: touched page reads 1, untouched far pages 0; readahead may legitimately populate a fault's neighbour). Per-interval counters drained by the metrics reporter; ≤ ~1 ms of mincore syscalls per prefill chunk; decode replay untouched; recording disables itself on first internal error |
| [46](../patches/46-ple-hotset-shadow.md) | Real read-through hot-set cache in front of the PLE mmap: hottest rows in a dense mlock'd RAM array; the HIP-default CPU gather serves hit rows from RAM, miss rows from the mmap as before; one-chunk-lag fills (a batch's misses are copied at the next batch's prep, when the pages are already resident); `hotset:` segment on the Prefill line reports hits/fills/evictions/size | Iteration 2 of the read-through cache (spec 2026-11-21); the iteration-1 shadow measured 100% warm row-level hits, validating the structure. Flat hash + SIEVE, all vectorized numpy on the scheduler thread, single-owner. Capacity via `SGLANG_QWEN4_PLE_HOTSET_ROWS` (default 1M rows ≈ 152 MiB; 0 disables). Degrades to the stock mmap path on mlock failure or any internal error; bit-exact vs the stock path verified in `tests/check_ple_hotset_shadow.py` (mixed hit/miss/out-of-range batches, bitwise compare — fp8 NaN patterns round-trip identically). Cold on restart; keys-only snapshot is the follow-up |
| [49](../patches/49-flash-attn-probe.md) | The `_patch_flash_attn_availability` HF patch probes flash-attn via `find_spec` + `__init__` source read instead of importing it | The old probe's `import flash_attn` pulled in `flash_attn_interface`, which imports aiter, and aiter's JIT rebuilt `module_aiter_core` (~14 s) in every process start even with `SGLANG_USE_AITER=0` (sglang's own aiter imports are gated; this path was not). Same module resolution, same masquerade verdict (a bare namespace package has no `__init__` origin and still neuters the flag); a v2 dist that exports the name but fails at real import is no longer caught at probe time — accepted, it surfaces loudly at first use. `tests/check_flash_attn_probe.py` |
| [50](../patches/50-mxfp4-aiter-gate.md) | `mxfp4.py` imports aiter at module top only when `SGLANG_USE_AITER` is set (names pre-bound to `None` otherwise) | The quantization package imported aiter on every HIP process start regardless of the flag (`if _is_hip:`, not `if _use_aiter:`), so aiter's JIT rebuilt `module_aiter_core` (~15 s) in every boot/tuner worker/precompile generation even after patch 49 — the server_args resolution chain imports `quantization/__init__` → `mxfp4` ~9 s after start. This checkpoint is INT4, not mxfp4, so the imported functions are never called. An mxfp4 model served with the flag unset now fails loudly at first aiter use instead of working (the flag's documented meaning). Found with a `sys.meta_path` trap; a `builtins.__import__` hook misses it because `importlib.import_module` bypasses it. `tests/check_mxfp4_aiter_gate.py` |
| [51](../patches/51-status-color.md) | Status-line fields color-coded in the journal: gen tps bright green (92), prefill (input) tps bright cyan (96), `#pending-token` yellow (33), `#running-req` bright magenta (95), `#queue-req` red (31) — on every log emitter (prefill/decode status lines, decode tail, flush-cache refusal in scheduler.py) | Wading a live `journalctl -a` stream is the day-to-day use of these lines; color gives instant triage (decode health vs queue backup). Whole label+value segments are wrapped in ANSI escapes so the inner text stays byte-identical — `bench/journal_stats.py`'s regexes and the `grep -o` extraction pipelines keep matching (verified against rendered samples in the test). The attach/detach rejection API messages stay plain (machine-read). `SGLANG_QWEN4_STATUS_COLOR=0` strips all color for file diffs. Recolored + coverage extended 2026-11-23 after live-journal feedback (32 too dark, 35 read as unhighlighted). `tests/check_status_color.py` |
| [52](../patches/52-ple-hotset-mlock-onfault.md) | Hotset `_mlock` uses `mlock2(MLOCK_ONFAULT)` (fallbacks: raw syscall, plain mlock, unlocked) | Plain mlock on the fresh calloc'd payload faulted every page in at attach: 320 MiB × 2 hotsets at 2M rows = two silent ~2-min windows per boot (measured 2026-11-23: 8 s/0 s at 1M rows vs 2:00/1:48 at 2M; boot 65 s → 5m45s). ONFAULT locks pages as the cache touches them — identical steady-state protection, boot cost ~0. Verified in-image by mincore: only the 1 glibc chunk-header page resident at construction (plain mlock faults in all), a far-row write adds its page. `tests/check_ple_hotset_mlock2.py` |
| [53](../patches/53-ple-lazy-warmstart.md) | Hotset warm start restores keys only; payload bytes refill in place on first touch (invalid-bit gate in `prepare`; refill via `_insert_batch`'s resident-key path, no dupes) | Even after patch 52, the first attach window cost 1:44 (2026-11-23): the eager warm start re-gathered ~160 MiB of snapshot rows from the table mmap before serving. Mapping/locking never required those pages — restored keys now act as SIEVE fodder and probe as misses until touched, so the reads amortize over live traffic and only touched rows become resident. Correctness rule (never serve an invalid slot) enforced at the single serving path; bit-exact refill verified against the fill source. `tests/check_ple_lazy_warmstart.py` |
| [54](../patches/54-qsa-native-q8.md) | Opt-in native ROCmFPX Q8 for the 12 target QSA QKV projections on gfx1151 TP=1 | Read-only sidecar, in-register dequant on small batches, transient BF16 unpack for large batches. The draft and all other weights are unchanged. `tests/check_qsa_native_q8.py` |
| [55](../patches/55-qsa-cross-prefix.md) | Complete ratio-4 QSA groups across unaligned eager MIXED/EXTEND prefixes | Reuses pending prefix keys and original group-start RoPE, including single-token MIXED decode rows; leaves aligned, paged and graph compression paths intact. Backport of closed-unmerged SGLang #39575. `tests/check_qsa_cross_prefix.py` |
| [56](../patches/56-mixed-mamba-checkpoint.md) | Preserve non-lazy Mamba prefill checkpoint metadata in MIXED forwards | Keeps checkpoint destinations for prefill rows and masks decode tails off. `tests/check_mixed_mamba_checkpoint.py` |
| [57](../patches/57-fsched.md) | Opt-in `--schedule-policy fsched` | Selects the forward profile by measured cohort debt, then packs feasible rows by request lag. Fixed 2:1 service time, 0.5× switch band. Enabled for the cerebro trial. `tests/check_fsched_server.py` |
| [10](../patches/10-sleep-on-idle-default.md) | Idle scheduler sleeps | unchanged, re-anchored to the new `arg_groups` layout |
| [configs/moe](../configs/moe/) | Tuned fused-MoE Triton tiles for `E=512,N=320,int4_w4a16`, mounted at `/moe-configs` by the launchers | Upstream has no `Radeon_8060S_Graphics` configs; the generic tile is 2.2× slower at decode. See MoE tile tuning below |

Everything else (GDN, QSA prefill attention and indexer projections, the
HyperConnection layers other than the mix, fused sigmoid-mul, n-gram hashing)
is pure Triton upstream and runs unmodified.

## Prerequisites

1. Image: `docker build -t strix-halo-sglang:dev .` (see
   [BUILDING.md](BUILDING.md); the Dockerfile pins upstream
   `70b5b03e78612c94f86ac98eb4d2d8d19ceda738`).
2. Kernel parity check on the box (no model needed, ~1 min):
   ```bash
   docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
       --security-opt seccomp=unconfined \
       -v $PWD/tools/test_qwen38_rocm.py:/test.py:ro strix-halo-sglang:dev python3 /test.py
   ```
   Expect `ALL PARITY TESTS PASSED` (PLE gather bf16/fp8, QSA decode, top-k chain,
   MoE zero points incl. a negative control, dense WNA16 dequant, MoE config
   search path, deterministic HC mix, partial-K MoE tile, dense decode top-k,
   tie-stable block selection, greedy draft proposal, MTP tail placement,
   Triton decode MQA, row-sliced prefill top-k, Triton prefill MQA,
   host-parked parameters, packed PLE short conv). Run
   GPU experiments beside a live server under `--memory 12g` and `timeout`:
   a bad Triton kernel's compile has OOMed the 30 GB host once.
3. Convert the PLE table to fp8 (halves the table to ~48 GiB and is the format
   the file backend expects to keep resident-free):
   ```bash
   docker run --rm -v /path/to/Qwen3.8-Flash-Next-AWQ-INT4:/src:ro \
       -v ~/models/Qwen3.8-Flash-Next-AWQ-INT4-ple-fp8:/dst \
       -v $PWD/tools/convert_ple_fp8.py:/convert.py:ro \
       strix-halo-sglang:dev python3 -u /convert.py /src /dst
   ```
   Two streaming passes (per-table amax, then rewrite). Non-PLE files are
   hardlinked when source and destination share a filesystem, copied otherwise.
   Safe to rerun: finished files are skipped. Output ≈ 128 GiB.

## Launch

```bash
./start-qwen38.sh                                # port 30001, container sglang-qwen38
docker compose --profile qwen38 up -d qwen38     # same thing, detached
systemctl --user start qwen38-sglang             # as a user service, see systemd/README.md
```

All run:

```
python3 -m sglang.launch_server \
    --model-path /models/qwen38 \
    --ple-offload-embedding \
    --ple-offload-backend file --ple-offload-dir /ple \
    --mem-fraction-static 0.85 --context-length 131072 \
    --kv-cache-dtype fp8_e4m3 --max-total-tokens 262144 \
    --attention-backend triton \
    --cuda-graph-max-bs-decode 20 --max-running-requests 20 \
    --max-mamba-cache-size 100 --mamba-ssm-dtype bfloat16 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --enable-strict-thinking \
    --chat-template /chat-template.jinja \
    --default-chat-template-kwargs '{"reasoning_effort": "medium", "default_system_prompt": "If you are unsure or do not know something, say so plainly instead of guessing."}'
```

with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the environment
and [`configs/chat/qwen38.jinja`](../configs/chat/qwen38.jinja) mounted at
`/chat-template.jinja`.

with `SGLANG_FORCE_NATIVE_LAYERNORM=1 SGLANG_USE_AITER=0
SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 PYTORCH_TUNABLEOP_TUNING=0` in the
environment, `~/.cache/strix-halo-sglang-cache` mounted at
`/root/.cache/sglang` (SGLang's Triton / JIT kernel cache) so restarts do not
recompile every kernel, and [`configs/moe/qwen38-flash-next`](../configs/moe/)
mounted at `/moe-configs` with `SGLANG_MOE_CONFIG_DIR=/moe-configs` (the
tuned MoE tiles; `MOE_CONFIG_DIR` / `QWEN38_MOE_CONFIG_DIR` pick another
profile, empty disables it).

The first boot writes the 48 GiB PLE table and takes ~10 min; later boots
reuse it (patch 13) and skip the 128 PLE shards. First-request kernel
compiles add a few more minutes on a cold kernel cache. Keep the PLE directory
on local NVMe: it is random-read during decode.

### Why these flags

| Flag / env | Reason |
|---|---|
| `--ple-offload-embedding` | Must be explicit. Upstream defaults it on only for `is_cuda and dtype==bf16`; on ROCm it resolves to `False`, and `--ple-offload-backend file` refuses to start without it. |
| `--ple-offload-backend file --ple-offload-dir /ple` | The table becomes a sparse file-backed `mmap`; rows are read through the page cache on demand and the resident set is trimmed (8 GiB cap by default). The first boot writes the table (~48 GiB) and arms a completion marker (`<table>.complete.json`, fingerprinted by the checkpoint index and PLE shard sizes); later boots skip the PLE shards while the marker matches. Delete the marker to force a rewrite. |
| `SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1` | Upstream gates the file backend on `cudaDevAttrPageableMemoryAccessUsesHostPageTables` (GB10). Irrelevant here: patch 11 gathers on the CPU, the GPU never dereferences the mapping. |
| `--attention-backend triton` | Same as every other model on this box; aiter's CK paths are CDNA-only. |
| `--context-length 131072` | The model is trained to 262,144 positions (no RoPE scaling needed). 131k costs nothing here: the pools are preallocated and the 8192-token prefill chunks bound activation size, so peak VRAM at a 120k prefill equals idle (88.9 GiB). Measured below: prefill flat at 450–480 tok/s to 125k tokens, decode 14.0 tok/s at 125k vs 14.5 short, exact needle recall at 120k. `QWEN38_CONTEXT` / `SGLANG_CONTEXT` override. |
| `--kv-cache-dtype fp8_e4m3 --max-total-tokens 262144` | fp8 KV halves the pool; the token cap keeps the saving as headroom instead of a larger pool (see Memory). `QWEN38_KV_DTYPE` / `QWEN38_MAX_TOTAL_TOKENS` override. **Measured A/B (2026-11-23, dev53, bf16 KV with the pool halved to 184,896 tokens vs fp8's 358,912, `bench/2026-11-23-bf16kv-ab.jsonl` vs the dev53 verify run): bf16 KV decodes 6–10% *faster* everywhere (26.2 vs 23.8 tok/s at 228; 23.0 vs 22.0 at 8k; 21.8 vs 20.5 at 32k) and leaves more VRAM headroom (3.17 vs 2.87 GiB), but prefills ~10% slower (620 vs 707 tok/s at 32k; TTFT 51.6 vs 45.2 s) and halves the token pool. fp8 KV on this Triton backend is therefore a **capacity decision, not a speed one**: it is what makes 262,144 context possible (a single full-context request needs a pool ≥ its length; bf16's 184,896 cannot hold 262k). If 131,072 context were acceptable, bf16 KV would buy ~7% decode. |
| `--cuda-graph-max-bs-decode 20 --max-running-requests 20` | Decode graphs for bs 1, 2, 4, 8, 12, 16, 20; patch 14 fills the PLE prefetch buffer from the host before each replay. The two numbers are independent (`QWEN38_CUDA_GRAPH_MAX_BS`, `QWEN38_MAX_RUNNING_REQUESTS`); the default keeps them equal because graphs above bs 8 cost 0.3 GB and nothing else. They used to be tied because eager decode above the graph range faulted the next replay; patch 15 fixed that. Note this upstream split the flag: a bare `--cuda-graph-max-bs` is rejected as ambiguous. |
| `--max-mamba-cache-size 100` | The GDN layers keep a fixed-size recurrent state (conv window + SSM matrix) per request instead of per-token KV, in a pool counted in slots. With the radix cache on SGLang reserves 5 slots per request (3 for the live state, prefix-cache branch points and the prefill→decode handoff, plus 2 for the overlap scheduler's ping-pong buffer), so `max_running_requests = slots // 5`. The ratio-sized pool came out at 99 slots and silently capped the server at 19 requests (and the graph list at `[..., 16, 19]`); 100 makes the advertised 20 real. ~54 MB per slot in bf16, so ~270 MB per extra request. `QWEN38_MAMBA_CACHE_SIZE` overrides; keep it at 5 × `QWEN38_MAX_RUNNING_REQUESTS`. |
| `--mamba-ssm-dtype bfloat16` | The GDN recurrent state is fp32 by default (~108 MB per slot); bf16 halves it, so 100 slots cost 5.4 GB instead of 10.8. Upstream's own suggestion in the startup log. |
| `--reasoning-parser qwen3 --tool-call-parser qwen3_coder` | The chat template opens `<think>` in the generation prompt and asks for `<tool_call><function=...><parameter=...>` tool calls. Without these the thinking and the XML come back as plain `content`. `qwen3`, not `qwen3-thinking`: the latter forces "everything before `</think>` is reasoning" for every request, and with `"chat_template_kwargs": {"enable_thinking": false}` the template closes the block in the prompt, the model emits no `</think>`, and the whole answer landed in `reasoning_content` with `content` empty (measured). `qwen3` decides per request from the template's `enable_thinking` toggle: thinking on by default, off when asked, answer in `content` either way. |
| `--enable-strict-thinking` | The model sometimes makes `<|im_end|>` its most likely token while still inside `<think>` ([Qwen3.8 issue 216](https://github.com/QwenLM/Qwen3.8/issues/216): a per-token hazard on list-heavy reasoning, worst at `xhigh`, about one call in five there; deterministic at `temperature 0`; not caused by MTP, the reporter measured it with speculation off). The request then ends with `finish_reason: stop`, `reasoning_content` filled and no `content` or `tool_calls`. sglang#24839 proposed ignoring the EOS in `_check_token_based_finish`; that was never merged, and it would leave the end-of-turn token in the context: an `ignore_eos` request here shows what follows (`OK user assistant <think> </think> OK user assistant …`, a new turn each time). Strict thinking instead wraps every thinking request in `ReasonerGrammarObject`, which masks `<|im_end|>`, `<|endoftext|>` and `<tool_call>` in the logits until `</think>` has been emitted, so the model takes its next-best token and closes the block. Needs `--grammar-backend xgrammar` (the default). Verified live under MTP: natural EOS still stops (28 tokens, thinking on; 88, off), `stop_token_ids` stops, `ignore_eos` runs to the cap, tool calls at `temperature` 0 and 0.7 return `tool_calls`, a 1165-token essay has no token-0 (`!!!!`) runs (sglang#36537), accept length 2.25 to 2.65 (2.5 to 3.3 before, different load). The same object enforces a thinking budget: `/generate` with `"require_reasoning": true, "max_thinking_tokens": N` forced `</think>` at the budget (30 → closed mid-sentence at 149 characters; 150 → 741; unbounded → 2890); the OpenAI chat endpoint has no field for it. The other failure mode in that issue, a repetition loop that runs to `max_tokens` with `finish_reason: length` (reproduced here at `xhigh`, `temperature 0`: `Probiotics: possible.` repeated to the 6k cap), is not addressed by this flag. `QWEN38_STRICT_THINKING=0` disables. |
| `--chat-template /chat-template.jinja` | [`configs/chat/qwen38.jinja`](../configs/chat/qwen38.jinja): the checkpoint's stock template plus a `default_system_prompt` kwarg, byte-identical to stock when the kwarg is empty ([`tests/check_chat_template.py`](../tests/check_chat_template.py), 50 renderings), with every control token it emits wrapped in the private-use marks [patch 32](../patches/32-control-token-encoder.md) encodes by id (the marks are stripped before that comparison; the stock template on an unpatched server would print them). Also makes the served template independent of what sits in the model directory (the DERISKED checkpoint shipped a persona template, see below). `QWEN38_CHAT_TEMPLATE=` (empty) uses the checkpoint's file. |
| `--default-chat-template-kwargs '{"reasoning_effort": "medium", "default_system_prompt": "..."}'` | Server-wide template kwargs; a request's own `chat_template_kwargs` (or top-level `reasoning_effort`) win key by key, which for `reasoning_effort` needs [patch 29](../patches/29-default-effort-override.md): upstream refills the request's popped effort with the server default and every request rendered at `medium`. **`reasoning_effort`**: the template's default is `xhigh`, which prepends "Please think carefully through the task, validate key assumptions, consider plausible alternatives..." to every request and is what every Qwen 3.8 overthinking report traces back to (Simon Willison's 21-minute SVG at 22k reasoning tokens; the model repo's "This model cannot stop thinking" thread, where `medium` measured a third less thinking with no quality drop and a proxy forcing `medium` for everything was ~3× faster and "no longer does stuff I didn't ask for"). `medium` emits no instruction text; `low` emits a "keep your thinking brief" one. The stock template rejects every other value with a 400; clients that pass the OpenAI scale straight through (Hermes sent `minimal`) would fail, so the repo template maps `minimal` to `low` and `high` to `xhigh` (case-insensitively; an empty value means the default; anything else still errors). A top-level `reasoning_effort: "none"` never reaches the template: SGLang turns it into `enable_thinking: false`, the OpenAI meaning; inside `chat_template_kwargs` the template maps it to `low`. Verified: `prompt_tokens` 59 / 71 / 35 for `minimal` / `high` / `none`, matching `low` / `xhigh` / thinking off. Qwen's card cautions that in multi-turn agentic work lower effort can cost more through retries, so clients doing that may pass `xhigh`. **`default_system_prompt`**: one line asking the model to say when it is unsure, rendered after the effort instruction and before the client's own system message (so it is part of the shared radix-cache prefix). `QWEN38_REASONING_EFFORT` / `QWEN38_SYSTEM_PROMPT` set them; empty disables. Note the kwargs live at the top of the prompt: changing them per request invalidates the prefix cache for that conversation. |
| `PYTORCH_TUNABLEOP_TUNING=0` | The image enables PyTorch TunableOp, which benchmarks every GEMM solution for each *new* M (= tokens in the prefill chunk). That is 14–20 s of TTFT for every novel prompt length (measured; the recorded results persist in `~/.cache/strix-halo-sglang-tunableop` so a repeated length is fast). With tuning off the recorded solutions are still used and untuned shapes take hipBLASLt's heuristic pick. `SGLANG_TUNABLEOP_TUNING=1 ./start-qwen38.sh` to deliberately record more. |
| `SGLANG_USE_AITER=0` | Set in the image. |
| `LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1` (via `SGLANG_DOCKER_ARGS` in the service env file) | The ROCr runtime inside the Python wheel (`_rocm_sdk_core`, rocm-sdk-core 7.13.0a20260422) leaves one native thread spinning at 100% CPU for the life of any process that has submitted work to the GPU once (a kernel, an event record, an H2D copy); the HTTP/tokenizer process does that at startup and then burned a core forever while idle. Found by sampling the thread's instruction pointer (always inside `libhsa-runtime64.so.1`, never in a syscall); reproduced with a bare torch probe; no HSA/HIP env knob stops it. The system ROCr in `/opt/rocm` (same 7.13.0, different build) does not spin, and preloading it binds every process in the container to it. Dispatch, sync, matmul and copy timings are identical between the two builds; decode speed and numerics unchanged. Oddly the scheduler process never spun even with the wheel library; unexplained. |

### What to look for in the log

```
Using CompressedTensorsWNA16TritonMoE (ROCm)
Using MoE kernel config from /moe-configs/E=512,N=320,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json.
PLE table: file-backed mmap /ple/ple_table_<rows>x160_float8_e4m3fn_..._rows0-<rows>.bin (47.x GiB, torch.float8_e4m3fn)
PLE table: WILLNEED prefetch on for gathers of >= 2048 rows (row = 160 B)
PLE table: resident set capped at 8.0 GiB, checked every 30 s
Using QSA for sparse full-attention layers.
QSA decode on HIP: using Triton qwen38_qsa kernel (heads=(12,1) head_dim=256)
```

If you see `QSA decode on HIP: using flash_attn varlen fallback`, the QSA
kernel's shape contract was not met; the fallback is correct but slow.

### Live view: event stream and `sgl_top`

Patch 35 gives the server a low-level event stream over ZMQ PUB when
`SGLANG_EVENT_STREAM_PORT` is set (the launcher passes it through
`SGLANG_DOCKER_ARGS`, see the service env file):

```
SGLANG_DOCKER_ARGS=-e SGLANG_EVENT_STREAM_PORT=8010 -e SGLANG_EVENT_STREAM_STEPS=1
```

The scheduler publishes on `tcp://127.0.0.1:8010` (`prefill_batch` and
`decode_batch` with per-request lists: rid, prompt length, cached, prefilled
so far, generated so far, age; the waiting queue with wait times; and, with
`_STEPS=1`, a `step` event per forward), the API process on `8011`
(`req_received`, `req_first_token`, `req_finished` with TTFT, e2e and accept
length). Sends are non-blocking, so nobody listening costs nothing. Event
fields are listed in [patches/35-event-stream.md](../patches/35-event-stream.md).

`~/scratch/sgl_top.py` is the consumer (curses, pyzmq):

```
python3 ~/scratch/sgl_top.py [--record ~/scratch/events-$(date +%F).jsonl]
```

Keys: `q` quit, `p` pause, `1`-`5` sparkline bucket (1/2/5/10/30 s), `[`/`]`
move the split. Requests first seen inside a batch snapshot (already running
when the view connected) are marked `~`. Without the stream (older image or
port unset) the SGLang pane falls back to the journal's aggregate batch lines
and has no per-request matrix. `--record` appends every raw event as JSONL for
later analysis of scheduler experiments. The view also polls `/v1/loads` and
`/server_info` once a second (the older `/get_load` and `/get_server_info`
log a deprecation warning on every call) and hides those access lines from
its log pane and request counts.

## Load time

A warm restart (PLE table already on disk, kernel cache warm) of the
DERISKED deployment with MTP and a cap of 16 measured, from `docker run` to
Uvicorn, with the phases from the journal:

| Phase | Normal load | With patch 30 | Presharded reload | Presharded + patch 42 |
|---|---|---|---|---|
| Container start, imports, tokenizer | 37 s | 37 s | 37 s | 37 s |
| Target weights | 152 s | 152 s | 72 to 98 s | 24 to 31 s |
| MTP draft weights | 49 s | 30 s | 4 to 9 s | 2.5 to 3.4 s |
| CUDA graphs (target verify + draft decode) | 14 s | 14 s | 11 s | 11 s |
| Memory pools | 1.4 s | 1.4 s | 1.6 s | 1.6 s |
| **Uvicorn up** | **259 s** | **~240 s** | **133 to 169 s** | **~95 s** |

The patch 30 column is the normal-load column with the measured draft time
(20 to 30 s over two first boots) substituted; it was not booted on its own.
Two reload boots were measured (`load_weight` 75.7 s and 107 s); the
difference is disk contention, not variance in the code path. Add whatever
the previous server takes to stop: with requests in flight it drains them
until the unit's 120 s stop timeout.

**Why the normal load is slow.** It is not I/O. A direct read of a weight
shard through dm-crypt runs at 4.2 GB/s (the 74 GB of non-PLE weights is
about 18 s of disk), `safe_open().get_tensor()` is a zero-copy mmap view, and
the loader's 8 threads (`enable_multithread_load`, already on) only parse
headers. The main checkpoint holds **222,718 tensors with a median size of
25 KiB** (per-expert INT4 weights, scales and zero points, split into files
of 19 GB), and every one costs a Python `weight_loader` dispatch, a `narrow`
and a tiny host-to-device copy: about 0.68 ms each, 152 s in total. The
draft used to walk all 222k tensors too, because `Qwen4ExpForCausalLMMTP`
had no `weight_files_to_skip`; patch 30 reads the index and skips the 31
shards with no `mtp` key (the 3 `model-mtp-merged-rest*` shards hold all 31
draft tensors).

### First boot after a checkpoint change (measured 2026-11-23, draft-int4 conversion)

The final draft-int4 regeneration took **18 min 59 s** from `docker run`
to `fired up`. This measures boot, not draft correctness: even with zero
routed-expert load warnings, the draft still accepted no tokens in the
post-boot smoke test.

| Phase | Time | Notes |
|---|---|---|
| Target checkpoint load, including PLE rebuild | 677 s | 124.3 GiB over 38 shards at 188 MiB/s; the 47.7 GiB PLE table was marked complete at the end of this phase |
| Target dump hashing | 10 s | 75.7 GiB; hashing is separate from the write |
| Target dump write | 317 s | 75.1 GiB in 91 files at 243 MiB/s |
| Draft load, hashing, and dump write | 4 s / 1 s / 5 s | Two input shards; 3.9 GiB dump |
| Graph capture and pool allocation | 8.83 s verify / 2.29 s draft / 2.41 s pools | Reported by the engine |
| **Whole boot** | **18 min 59 s** | Includes startup, post-processing, and warmup outside the phases above |

The presharded key includes the content and mtime of `config.json`; the PLE
fingerprint includes the index file's mtime. A plain `cp` restore can invalidate
both caches even when the content is identical. Preserve mtimes on restore
with `cp -a` or `rsync -a` when the files are unchanged. Patch 31 leaves
dumps under other keys in place. Check `/opt` free space before starting:
keeping the old target dump alongside the new one left only 331 MiB free
after this boot.

**Someday: speed up the first load of a new checkpoint.** Profile shard reads,
PLE rebuild, and per-tensor post-processing separately before trying to batch
or parallelize the loader. The 677 s measurement combines those steps, so it
does not identify the bottleneck. This is only worth revisiting if checkpoint
turnover increases; restarts that reuse a dump already skip this path.

### Draft-int4 verification (2026-11-23)

The clean shard regeneration fixed the routed-expert key names, but the
draft still accepted zero tokens. The draft model does not declare a packed
module mapping, so the individual BF16 exclusions did not cover the fused
attention `qkv_proj` and shared-expert `gate_up_proj`. Those fused parameters
did not receive their checkpoint weights. The converter now excludes both
fused names explicitly while leaving the routed experts int4. The live
verification used a separate draft config so the target dump and PLE table
could be reused unchanged.

Two 192-token greedy requests accepted 123/207 and 110/243 draft proposals.
The matching two-pass benchmark reported a mean accept length of 2.54
over 20 scheduler samples. Warm-pass results:

| Decode scenario | BF16 draft (dev53) | Int4 draft |
|---|---:|---:|
| Single, 128 target prompt tokens | 23.82 | 20.12 |
| Single, 2,048 | 19.36 | 22.33 |
| Single, 8,192 | 22.01 | 22.27 |
| Single, 32,000 | 20.54 | 20.83 |
| 4 streams, aggregate | 45.46 | 39.81 |
| 8 streams, aggregate | 66.39 | 61.82 |
| 12 streams, aggregate | 75.30 | 74.80 |

Rates are tokens/s. The worst sampled free VRAM in the int4 run was
3,971 MiB, versus 2,846 MiB on the corresponding baseline pass. The
int4 runtime allocated the requested 524,288-token pool. These runs show
more memory headroom, not a broad throughput gain.

Raw results: [BF16 baseline](../bench/2026-11-23-dev53-lazywarm-verify.jsonl)
and [corrected int4 draft](../bench/2026-11-23-draft-int4-fused-bf16.jsonl).

**TunableOp on rocBLAS 5.6.** Keep a separate `TUNABLE_DIR` per ROCm/PyTorch
stack. The old `:dev` cache had a rocBLAS 5.4 validator. `:rocpro10`
rejected it, but tuning still appended results without replacing the old
header. Do not relabel that mixed file.

A clean sweep on `:rocpro10` spent 801 s on target verify graph capture and
450 s on draft capture before tuning the seven request scenarios. After
restarting with tuning off, PyTorch accepted 338 saved GEMM signatures.
Graph capture took 9.27 s target and 2.38 s draft, and mean acceptance length
was 2.55 over 20 decode samples. Warm single-request decode stayed within
6% of the earlier int4 baseline; 4/8/12-stream throughput was 42.5/62.7/76.7
tok/s versus 39.8/61.8/74.8. That suggests a small concurrent gain, not a
broad speedup from this one comparison. Keep tuning off during serving.
See [the cache replay benchmark](../bench/2026-11-23-tunableop-rocblas56.jsonl).

**NEXTN vs. n-gram (2026-11-23).** Keep NEXTN. With the current 64-token
pages, Triton attention rejects n-gram's default branch breadth of 10.
Breadth 1 passed that check, but the target failed during graph capture:
`Qwen4 PLE does not support NGRAM speculation`. No n-gram throughput
was measured. NEXTN was restored and reached readiness after both attempts.

**Options that do not help here.** `--load-format npcache` only handles
`.bin` checkpoints (it asserts `use_safetensors is False`).
`--model-loader-extra-config` for the default loader only knows
`enable_multithread_load` / `num_threads` (both already on) and
`weight_loader_disable_mmap` (needs (workers + 2) × 19 GB of host RAM;
the box has 32 GB). SGLang's weight-cache daemon
(`sglang.srt.weight_cache`) shares GPU tensors over CUDA IPC on the same
machine and only for unquantized or block-FP8 weights; it rejects
`compressed-tensors` and needs `expandable_segments` off. The
`remote_instance` loader copies from a peer GPU over NCCL; over 1 GbE it
would be slower than the disk.

**What does help: `--load-format presharded`** (`PreshardedModelLoader`,
patch 31 makes it work with the host PLE table). The first boot loads
normally, then dumps the post-processed state (a few thousand full-layer
tensors, safetensors files, a `checksum.json` plan, a `READY` marker) into
`<root>/TP-1-v<version>-m<model>/` (`TP-1-sig-<sha1>` without a version); later boots initialise the model, run
`process_weights_after_loading` on the empty parameters and copy the dump
straight in. The DERISKED dump is 69 GiB for the target and 7.3 GiB for the
draft (the 47.7 GB PLE table is not in it); the first boot took 366 s for
the target load plus dump and 20 s for the draft (about 7 min to Uvicorn).
Enable it with `QWEN38_PRESHARDED_DIR=/opt/llm/presharded-<variant>` in the
launcher env (compose: uncomment the mount and the two flags).

Rules the cache follows:

- **Key.** The subfolder is `TP-<tp>-v<version>-m<model>` (patch 31):
  `<model>` is the first 8 hex of the sha1 over the checkpoint's
  `config.json` content and mtime, so a root shared by two checkpoints never
  hands one the other's dump (two fine-tunes of one base share the content,
  not the write time; a per-variant root such as `presharded-swift` is still
  the clean separation). `<version>` is the launcher's
  `QWEN38_PRESHARDED_VERSION` (default `1` in `start-qwen38.sh`;
  `QWEN38_PRESHARDED_DRAFT_VERSION` for the draft, same default), passed as
  `SGLANG_PRESHARDED_VERSION` / `SGLANG_PRESHARDED_DRAFT_VERSION`. An image
  rebuild keeps the dump; bump the version when a patch changes how weights
  are post-processed (quantization unpacking, fusion, parking), which costs
  one slow boot and 76 GiB more disk until the old dump is pruned. The
  stored configuration (quantization, dtype, parallel layout, parameter
  shapes) is still compared on reload. Without a version the name is
  upstream's hash of every setting, with the model hash inside. The boot
  logs the key: `Presharded key TP-1-v1-m…: cache version 1, model
  config.json (content+mtime) sha1 …`.
- **Stale dumps.** The current key's subfolder without `READY` (an
  interrupted dump), or with `READY` but a stored configuration that differs
  from this launch's, is removed at boot and the dump redone.
- **Other subfolders** (other versions, other configurations) are logged at
  boot with size and completeness (`Presharded cache root ... also holds
  TP-1-sig-... (complete, 68.1 GiB)`) and never removed: they may belong to
  another deployment. Prune them by hand; they are root-owned, so
  `docker run --rm -v /opt/llm/presharded-<variant>:/p debian:stable-slim rm -rf /p/target/TP-1-v<old>-m<model>`.
- **Host memory.** The dumper stages a whole output file in host RAM before
  writing it and each hash thread holds a host copy of one tensor (up to
  0.84 GB). With upstream's 20 GiB default file size the scheduler reached
  18.4 GB anon RSS and was OOM-killed on this 32 GB box; the launcher passes
  `max_file_bytes` 1 GiB and `hash_num_threads` 4, and the first boot then
  peaked at about 18 GB used system-wide. Do not run other memory-hungry
  services during the first boot.
- **Disk.** Keep 80 GiB free on the dump's filesystem before a first boot
  (the failure mode is `No space left on device` from `save_file` and a
  restart loop). The dumps compress to about 85% with zstd (INT4 packed
  data), not worth the decompression buffer on reload.
- **Parity.** Greedy output (`temperature 0`, 8 prompts × thinking on/off,
  160 tokens, 1485 tokens in total) from a presharded reload was
  byte-identical to the normally loaded server; patch 27 parks the same
  2.02 GiB.

**Where the big files live on this box.** Local NVMe (`/opt/llm`):
the DERISKED checkpoint, its PLE cache (`ple-cache-derisked`, 48 GiB) and its
presharded dump (`presharded-derisked`). NFS `/scratch/converted/`: the stock
`cyankiwi--Qwen3.8-Flash-Next-AWQ-INT4-ple-fp8` checkpoint (the local copy in
`~/models` was verified byte-identical and removed), its PLE cache
(`ple-cache-stock`, with the completion marker) and the DERISKED source. To
serve the stock model again, copy both back to local disk first (the PLE
table is random-read during decode; the launcher defaults still point at
`~/models/...` and `/opt/llm/ple-cache`).

## Dummy-weights smoke test

To exercise the whole stack without the 175 GiB checkpoint (useful after
rebuilding the image), generate a shrunken config that keeps the production
kernel geometry and launch with `--load-format dummy`:

```bash
python3 tools/make_mini_qwen38_config.py /path/to/Qwen3.8-Flash-Next-AWQ-INT4 /tmp/qwen38-mini
docker run -d --name mini --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    --ipc=host --network=host --security-opt seccomp=unconfined \
    -v /tmp/qwen38-mini:/models/mini:ro -v /tmp/ple-mini:/ple \
    -e SGLANG_FORCE_NATIVE_LAYERNORM=1 -e SGLANG_USE_AITER=0 -e SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 \
    strix-halo-sglang:dev python3 -m sglang.launch_server --model-path /models/mini --load-format dummy \
    --host 0.0.0.0 --port 30002 --ple-offload-embedding --ple-offload-backend file --ple-offload-dir /ple \
    --mem-fraction-static 0.12 --context-length 4096 --attention-backend triton --cuda-graph-max-bs-decode 8
```

Output is noise (random weights) but every path runs: fp8 PLE file table,
WNA16 int4 MoE with zero points, QSA prefill + Triton decode, GDN, vision
tower (send a `data:image/png;base64,...` message), concurrent requests.

## Memory

96 GiB of the 128 GB is carved out as VRAM on this box (the rest is host RAM,
which the PLE table must *not* fill; the file backend's RSS trimmer keeps it
near the 8 GiB cap). Measure with `amdgpu_top --json --dump`
(`VRAM.Total VRAM Usage`).

Measured (driver view sampled every 2 s over a plain-graphs launch of the
patches ≤ 28 image): VRAM peaks at 70.8 GiB while the loader streams the
shards, drops to **67.5 GiB** once it releases its staging buffers, and
that is the weights in VRAM (plus 2.0 GiB parked in host memory since
patch 27, so ~69.5 GiB of weights; the vision tower is ~0.9 GiB of that).
Earlier revisions of this page said 75.9 GiB, read before the release.
The pools then add 8.9 GiB (KV cache 3.0 GiB: 262,144 tokens fp8; only the
12 full-attention layers hold KV, ~12 KB/token in fp8, so one full
262k-token request fits; GDN state 5.33 + 0.21 GB in bf16, 100 slots → 20
requests; ~0.4 GiB of radix tree, request-to-token map and tracking) and
the decode graphs 1.0 GiB, for 77.4 GiB idle without MTP (with MTP see the
accounting under Speculative decoding). Scheduler RSS ≈ 2.9 GB; host
`buff/cache` holds the PLE pages. Since patch
27 the launcher parks the token embedding (1.18 GiB, a `bs`-row gather per
step) and the vision tower (0.9 GiB, idle without images) in pinned host
memory (`QWEN38_HOST_PARKED_PARAMS`, default `embed_tokens.weight,visual.`):
driver VRAM after load drops from 92.2 to 90.3 GB with MTP at cap 20, and
the container's shmem rises by the same 2.0 GiB (exactly sized
`hipHostMalloc`; the first version used torch's pinned allocator, which
rounds to powers of two and cost 5.7 GB). `amdgpu_top` does not show these
buffers under GTT (they are userptr mappings; the scheduler's own GTT from
fdinfo is 8 MiB, and the box's total, ~100 MiB headless or ~400 MiB with a
Wayland desktop up, is the compositor and terminals), so the host side is
the scheduler's RSS. The gather from host memory costs
4.2 vs 3.8 µs at bs 20; greedy output, vision and throughput measured
unchanged.

Host memory over a load and a full benchmark (5 s samples of `free`, the
container cgroup and the top RSS processes, three launches): during weight
load the scheduler's RSS climbs to 10–17 GB for a minute or two (safetensors
mapped and streamed to the GPU, file-backed and reclaimable), page cache
rises to 25–26 GB, and the kernel pushes 2.5–3 GB of idle anonymous pages to
swap (`swappiness` 10). Host `used` never exceeded 7.1 GB and `available`
never dropped below 24.6 GB, so the swap is reclaim preference during the
76 GB stream, not exhaustion. While serving, scheduler RSS is 2.2–2.9 GB and
container anonymous memory ≤ 3.2 GB, flat through 20-stream runs. The one
host OOM this box has had was a tracing hook that imported torch into every
spawned process (see Known limitations), not the server.

The `avail mem` figure in the log is not headroom on this ROCm build (it
reads 24.8 GB before loading 76 GB of weights and 26.6 GB after; a
1,650-token prefill under MTP once OOMed with `avail mem=13 GB` printed).
Measure from the host instead: `sudo amdgpu_top --json -n 1` gives total
VRAM / GTT used and capacity plus per-process VRAM and GTT from fdinfo
(`--dump` omits fdinfo; other users' processes need root). What it reports
is driver-allocated memory, which for the scheduler is PyTorch's reserved
high-water mark: after a long prefill the figure stays up until a
`/flush_cache` even though the tensors are gone. For what is live *inside*
the process, `POST /start_profile {"activities": ["MEM"], "output_dir":
...}`, run the workload, `POST /stop_profile`: the pickle in `output_dir`
holds the allocator trace (100k events by default) and the segment map;
replaying the allocs/frees gives the peak live set by call site. That is
how patches 25 and 26 were found. Lowering `--mem-fraction-static`
does not create headroom either, the budget just moves into the KV/mamba
pools. What does:

| Change | Saves | Cost | Default |
|---|---:|---|---|
| `--kv-cache-dtype fp8_e4m3` | half the KV pool | none measured: identical greedy answers, exact needle recall in a 3,858-token prompt, decode 12.9 tok/s | on |
| `--max-total-tokens 262144` | ~3.2 GiB vs the fraction-sized pool | two full 131k-context requests or 20 × 13k fit at once; beyond that SGLang queues or retracts | on |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | fragmentation | none | on |
| vision tower off (`QWEN38_VISION=0` / `QWEN38_MODEL_OVERRIDE='{"language_model_only": true}'`) | ~0.9 GiB (333 tensors skipped) | no image input | off |
| embedding + vision tower in pinned host memory (`QWEN38_HOST_PARKED_PARAMS`, patch 27) | ~2.0 GiB | the same 2.0 GiB of host RAM (scheduler RSS/shmem); IOMMU-routed GPU reads of a gather-only table | on |
| `--max-mamba-cache-size N` | ~54 MB per slot (bf16) | 5 slots per request with the radix cache on, 1 with `--disable-radix-cache`; under MTP an extra `(requests+1) × draft_tokens` intermediate states | 100 (= 20 requests) |

`--language-model-only` itself is whitelisted to three unrelated
architectures in this upstream; the model code supports it, so the override
sets `language_model_only` on the HF config instead.

Not worth doing: pruning East Asian tokens from the vocabulary. 65,932 of
248,077 tokens (26.6%) contain CJK, kana or hangul, but `embed_tokens` and
`lm_head` are 1.18 GiB each in bf16, so the whole prune saves ~0.63 GiB.
It also needs a checkpoint rewrite, BPE merge surgery, and, because the PLE
n-gram table is indexed by hashes of token ids, an id-remap in front of the
hash (or the 51B table is wrong for every renumbered token).

## Measured performance

Single box, no other GPU tenant, PLE table on local NVMe, radix cache
flushed before every prefill measurement (streaming client, `max_tokens`
128, temperature 0). The same measurements, with memory sampled per
scenario, come from [`tools/bench_qwen38.py`](../tools/bench_qwen38.py)
against a running server:

```bash
tools/bench_qwen38.py --container sglang-qwen38-derisked --label "MTP cap 20" \
    --single 128,2048,8192,32000 --concurrent 4,8,16,20 --concurrent-lengths 128 \
    --mixed 18:26000:2 --passes 2 --out bench.jsonl
```

Single requests × prompt length (TTFT, prefill, decode), N streams × prompt
length (aggregate, per stream, worst TTFT), and N short streams decoding
while M long prompts prefill together (the VRAM worst case). Every row
carries the driver's VRAM peak / minimum free and GTT (from sysfs, no
root), the host's minimum available memory and, with `--container`, the
cgroup's peak current / anonymous / shmem, all over that row's own window,
plus the server's speculative settings, request cap and context from
`/get_server_info`. MTP on or off is a launch option, so run the suite once
per launch and compare the JSONL rows. The first pass after a start is
lower while Triton compiles new shapes; `--passes 2` shows both.

| Prompt tokens | TTFT | Prefill | Decode (bs=1) |
|---:|---:|---:|---:|
| 183 | 0.7 s | 264 tok/s (fixed overhead dominates) | 14.5 tok/s |
| 1,650 | 3.1 s | 535 tok/s | 14.5 tok/s |
| 6,693 | 12.1 s | 554 tok/s | 14.5 tok/s |
| 26,983 | 56.8 s | 475 tok/s | 14.7 tok/s |
| 32,935 | 71.7 s | 459 tok/s | 13.9 tok/s |
| 66,043 | 138.5 s | 477 tok/s | 14.0 tok/s |
| 99,151 | 215.2 s | 461 tok/s | 14.0 tok/s |
| 125,576 | 280.9 s | 447 tok/s | 14.0 tok/s |

Prefill is flat to the context limit (sparse attention and GDN, not dense
attention) and decode loses ~3% between short and 125k contexts. A needle at
60% depth of a 119,911-token prompt was recalled exactly. Two concurrent
59.5k-token requests with distinct prefixes: 119k KV tokens in use (46% of
the pool), 19.3 tok/s aggregate decode; the second request's prefill queued
behind the first (TTFT 151 s / 248 s) while the first decoded at 1.2 tok/s
between prefill chunks, which is chunked prefill working as designed.

| Concurrency (short prompts, 200 tokens each) | Aggregate | Per stream | TTFT (max) |
|---:|---:|---:|---:|
| 4 | 33.3 tok/s | 8.8 tok/s | 1.3 s |
| 8 | 57.9 tok/s | 7.5 tok/s | 1.2 s |
| 12 | 77.5 tok/s | 6.9 tok/s | 2.3 s |
| 16 | 99.4 tok/s | 6.7 tok/s | 2.6 s |
| 20 | 88–97 tok/s | 4.6–5.1 tok/s | 2.3 s |
| 24 | 74–76 tok/s | 5.8 tok/s | 42.6 s (4 queued) |

That table was taken with `--context-length 32768`, the launcher default at
the time. At today's default of 131072 the same image gives 71–78 at 16–20
streams, because the QSA indexer's decode scoring ran a torch reference
whose cost scales with the context limit (patch 24 and the DERISKED section
below). With patch 24, stock at 131k, two passes: 8 / 16 / 20 streams
50.6 / 89.2 / 95.2 then 62.4 / 104.7 / 103.5 tok/s aggregate, bs 1
14.0–15.1.

How it got here, single stream / 8 streams: eager 11.2 / 42.9 tok/s; decode
graphs (patch 14) 12.7 / 43.0; tuned MoE tiles (below) 14.5 / 57.9. Graphs
for bs 12–20 do not change throughput measurably against eager at those
sizes (server-side peak +4% at bs 16, within run-to-run noise end to end);
they were originally captured to keep eager decode out of the picture (see
patch 15) and stay on because they are cheap. Run-to-run spread at 8 streams
is about ±7%. Throughput
plateaus at 16 streams; the 20-request cap is queueing capacity, not speed
(each extra request costs ~270 MB of GDN state and one more graph).

The PLE gather costs about 1 s per 2048 cold tokens (32k rows faulted from
NVMe, ~35 µs each); rows already in the page cache shave that off (440–520
tok/s cold vs 550–585 warm). Decode is GPU-bound: with graphs on, ~98% of
the scheduler's host time is the D2H copy of the n-gram ids waiting for the
previous replay to finish. What is left in the ~69 ms bs=1 step is mostly
the bf16 dense projections (~8.6 GB of weight traffic per token, ~38 ms at
the measured ~225 GB/s) plus a long tail of small kernels; the MoE is now
~5 ms of it.

### MoE tile tuning

Upstream ships no fused-MoE Triton configs for `Radeon_8060S_Graphics`, so
the int4 expert GEMMs ran on a generic tile. [`tools/tune_moe_gfx1151.py`](../tools/tune_moe_gfx1151.py)
drives upstream's `benchmark/kernels/fused_moe_triton` tuner with a gfx1151
search space (1,560 configs) and three bail-outs against the best time so
far: an eager run before graph capture (compared with the winner's own eager
time, since at small M both are mostly launch overhead), the first replay
after capture (dropped at 3×, `MOE_TUNE_BAIL_FACTOR`), and a running total
in the timed loop that stops the moment it reaches what the winner needed
for the whole loop (exact: the average can no longer be lower). The winners
of the nearest finished batch sizes and of `MOE_TUNE_SEED_FILE` (an existing
config file, e.g. the one being re-tuned) are benchmarked first, so the
budget is tight from config #1: a full M=2 sweep with compiles cached bails
1,558 of 1,560 configs and takes ~2.5 minutes. The original 3×-only version
took about 7 GPU-hours for 18 sizes. The server can stay up (the int4 sweep
needs ~1.5 GB of VRAM, the bf16 draft sweep ~4.5 GB).

Each batch size runs in its own process (the Ray worker keeps every compiled
kernel resident and the kernel OOM-killed a single-process sweep at M=64 on
this 31 GB-host-RAM box). Results are written as they happen with fsync and
an atomic rename: `M=<m>.json` when a size finishes (finished sizes are
skipped on rerun) and `progress-M=<m>.json` after every config, so a crash
resumes at the next config and skips the one that killed the worker. The
partial directory defaults to `~/.cache/sglang/moe-tune/<tag>` inside the
container, i.e. `~/.cache/strix-halo-sglang-cache/moe-tune/` on the host
(the launcher's cache mount), and compiled kernels go to the same persisted
Triton cache the server uses, so a container restart loses nothing. Pass
`MOE_TUNE_PARTIAL_DIR` to name the directory yourself.

Cross-run partials inventory (host
`~/.cache/strix-halo-sglang-cache/moe-tune/`, 23 MB total, 2026-11-23):
`8098bf4581` (Nov 22, 12 M-sizes) is the source of the **deployed draft
config**; `out-target-int4-tp1` (Oct 25) is the source of the **deployed
target int4 config**; `draft-bf16`/`out-draft-bf16`/`recheck260`/
`out-recheck260` (Oct 24–25) are superseded by the Nov 22 draft tuning
and kept only as history; `f4f8054f6e` is an empty leftover. Nothing is
safe to delete without checking this table first.

Batch sizes are visited by bisection (smallest, largest, then the midpoint
of each unresolved interval). When the nearest finished sizes on both sides
of an M picked the same tile and are adjacent to it in the size list
(`MOE_TUNE_BISECT_MAX_GAP`, default 2, so no bracket wider than 4× is ever
skipped), that M only gets a quick check of the neighbours' winners instead
of a full sweep; where they disagree the regime is changing and the midpoint
is swept in full. `MOE_TUNE_BISECT=0` restores the ascending sweep. At the
large sizes a full sweep is replay-bound (one bf16 draft invocation at
M=1024 is 23 ms, so a config costs ~3 s and an M about 85 minutes); sizes
above the prefill chunk (`--chunked-prefill-size`, 1024 here) are never
run by the server and are not worth tuning.

Kernel time for one MoE layer (M = tokens in the step; ×48 layers per step):

| M | Default tile | Tuned | Best config |
|---:|---:|---:|---|
| 1 | 223 µs | 103 µs | 16×32×64, 1 warp, `waves_per_eu` 2 |
| 2 | 740 µs | 200 µs | 16×16×64, 1 warp, `waves_per_eu` 2 |
| 4 | 1,184 µs | 403 µs | same |
| 8 | 1,634 µs | 739 µs | same |
| 16 | 2,960 µs | 1,342 µs | same |
| 32 | 5,012 µs | 2,070 µs | same |
| 64 | 7,653 µs | 3,471 µs | same |
| 128 | 9,876 µs | 4,475 µs | same |
| 512 | 11,462 µs | 5,223 µs | same |
| 1,024 | 12,671 µs | 5,814 µs | 32×16×64, 1 warp |
| 2,048 | 14,314 µs | 9,145 µs | 64×16×32, 1 warp |
| 4,096 | 22,483 µs | 14,761 µs | 128×64×32, 4 warps |

(Table from the cyankiwi g32 checkpoint, half-width shard. It does not carry
over to g128: `group_size` is a kernel constexpr, and on the Swift checkpoint
`16×32×64` measures 228 µs at M=1 where `16×16×64` takes 98 µs.)

The same lesson as the Qwen3.5 hand sweep: at decode sizes the kernel is
bound by how many workgroups it can put on 40 CUs, so the smallest tiles
with one wave32 wavefront each win; `waves_per_eu=2` (not in upstream's
space) is worth another few percent; `BLOCK_K` 64 beats the group size 32
(the kernel indexes scales per element, so `BLOCK_K` may be any multiple of
the group). Only past M≈1k does tile efficiency start to matter and
`BLOCK_M` grow. End to end: bs=1 decode 12.6 → 14.5 tok/s (+15%), 4 streams
27.4 → 33.3 (+22%), 8 streams 43.0 → 57.9 (+35%); prefill unchanged within
noise (dominated by GDN/QSA and the PLE gather). Greedy answers, needle
recall in a 7,790-token prompt and generated code were re-checked after the
change; the tuner itself does not verify numerics.

The config lives in `configs/moe/qwen38-flash-next/` and is mounted at
`/moe-configs` by the launcher (patch 17 makes `SGLANG_MOE_CONFIG_DIR` a
search path in front of upstream's tree); nothing is baked into the image.
Re-tune after a Triton or kernel change:

```bash
docker cp tools/tune_moe_gfx1151.py sglang-qwen38:/tmp/
docker exec -w /tmp sglang-qwen38 python3 /tmp/tune_moe_gfx1151.py \
    --model /models/qwen38 --dtype int4_w4a16 --disable-shared-experts-fusion --tune
docker cp "sglang-qwen38:/tmp/E=512,N=320,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json" \
    configs/moe/qwen38-flash-next/
```

The tuner's default `--tp-size 2` yields the `N=320` file name the runtime
looks up, but that is a coincidence of two different conventions: the
runtime's `N=320` is the int4-packed width of the real 640-wide
intermediate, while the tuner's is `640 / tp`, so at the default it
benchmarks a half-width shard. The kernel times in the table above are
therefore for half the real GEMM; `--tp-size 1` measures the true shape
(bs 1 / 8 / 16 / 20 / 32: generic 438 / 3066 / 5685 / 6829 / 9794 µs,
tuned 179 / 1390 / 2544 / 3037 / 4340) and writes an `N=640` file that has
to be renamed to `N=320` to be picked up. Whether a true-shape sweep picks
different tiles has not been tested. Another checkpoint gets its own
profile directory (`QWEN38_MOE_CONFIG_DIR=…`) so the two sets of tuning
data never overwrite each other; see
[`configs/moe/README.md`](../configs/moe/README.md).

### Precompiling the MoE kernels from scratch

After a Triton-cache wipe (or on a fresh box/image), the server's first boot
recompiles every MoE kernel it will ever launch (~75 min of compile inside a
boot). [`tools/moe_precompile.py`](../tools/moe_precompile.py) synthesizes
the exact launch arguments of both fused-MoE GEMMs (provenance comments in
the file header) and compiles them via `JITFunction.warmup` — no launch, no
GPU benchmark — so the cache can be filled offline, per variant, before any
boot or sweep. The catches learned the hard way (each one silently produced
thousands of dead cache entries before the strict-warm check caught them):

- `--model-shape`'s `N` is the **intermediate** size (320 here). The tuner's
  merged-config *file name* says `N=640` because it names the kernel's N arg
  (2× intermediate). Passing 640 doubles every synthesized weight tensor.
- The **block-shape constexprs differ per consumer**. The tuner derives
  `block_shape = [0, group_size]` from the checkpoint's compressed-tensors
  `quantization_config.config_groups` (`[0, 32]` here) and passes it as
  `group_n`/`group_k` constexprs even on the generic bf16 kernel; the
  server's bf16 **draft** MoE has no block quant and passes `(0, 0)`.
  Constexpr values are part of the cache key, so these are two disjoint sets
  of entries: precompile BOTH.
- `MOE_TUNE_STRICT_WARM=1` (tuner knob) verifies a precompile: it snapshots
  the Triton cache before each config's first launch and raises listing any
  config that still compiled. If it fires with *every* config cold, the
  synthesis diverged from the real launch args (see the two items above);
  do not just re-run.

Common container shape (one worker; `--chunk 2000` keeps a generation to one
batch size — recycles cost ~15 s of interpreter startup each and buy nothing
at this worker count; Private_Dirty stays ~1–1.5 GiB):

```
docker run --rm --name moe-warm-<tag> --device=/dev/kfd --device=/dev/dri \
  --ipc=host --security-opt seccomp=unconfined \
  -v /opt/llm/models/Qwen3.8-Flash-Next-AWQ-INT4-ple-fp8:/models/qwen38:ro \
  -v "$HOME/.cache/strix-halo-sglang-cache:/root/.cache/sglang" \
  -v ~/strix-halo-sglang/tools:/tools:ro \
  -v /var/tmp/gfx_space_full.json:/space_full.json:ro \
  -e SGLANG_USE_AITER=0 -e SGLANG_FORCE_NATIVE_LAYERNORM=1 \
  -e PYTORCH_TUNABLEOP_TUNING=0 \
  strix-halo-sglang:rocpro10 \
  python3 /tools/moe_precompile.py --model-shape '<SHAPE>' \
    --space-file /space_full.json \
    --batch-sizes 1 2 4 8 16 32 64 512 1024 4096 --slice 0:1 --chunk 2000 \
    > /var/tmp/moe-warm-<tag>.log 2>&1
```

(Redirect the log from the **host** side; a redirect inside the container
lands in the container's own `/var/tmp` unless that is mounted, and a stale
or foreign-owned file at the target path can fail the redirect at startup —
which kills the run before it starts.)

The required runs, all against `~/.cache/strix-halo-sglang-cache`:

| # | Variant | `--model-shape` | Serves |
|---|---|---|---|
| 1 | bf16 draft, server-facing | `{"kind":"bf16","E":512,"K":2560,"N":320,"topk":10}` | the server's bf16 draft MoE launches (group 0,0) |
| 2 | bf16, tuner-facing | same + `,"block_shape":[0,32]` | strict-warm sweeps (the tuner's launches carry the AWQ block shape) |
| 3 | int4 target | `{"kind":"int4","E":512,"K":2560,"N":320,"topk":10,"group_size":32}` | the deployed int4 target MoE (`int4_w4a16` config files) |

Batch sizes `1 2 4 8 16 32 64 512 1024 4096` are the server's operating
cover; keys share across M through Triton's specialization classes, so the
whole set costs roughly 15–25 min per variant (M=64 is the expensive one,
~10 min of fresh compiles; later Ms are mostly hits). The search-space file
is the wrapper's 1,560-config gfx1151 space (`gfx1151_configs()` in
[`tools/tune_moe_gfx1151.py`](../tools/tune_moe_gfx1151.py)); a copy lives
at `/var/tmp/gfx_space_full.json` on this box, and the wrapper regenerates
the same list whenever it runs without `--search-space-file`. Before
sweeping batch sizes the precompile does not cover (e.g. 24, 48, 96, and the
optional 256 probe), run variants 1 **and** 2 again with `--batch-sizes 24
48 96 256` — otherwise the strict-warm check fails those sweeps by
construction.

Arm a watcher instead of polling (see the job-watch skill): `--ok 'M=4096:
DONE'` (last batch size), `--fail 'FAILED|Traceback'`, `--dead
'moe_precompile'`, and a generous `--timeout`. A clean run prints one
`M=<m>: DONE 1560 configs (…ok) in <t>s` per size and zero `FAILED` lines.

For a **sweep** run (`tune_moe_gfx1151.py`), mount the repo's config
directory and set the container workdir to it (`-w /out -v
~/strix-halo-sglang/configs/moe/qwen38-flash-next:/out`): the wrapper writes
its merged `E=…json` to the **current directory** (there is no
`--output-dir`; passing one is passed through to the inner tuner and fails
the sweep with rc=2). The inner tuner rewrites that file after each
completed batch size, and the per-size winners also persist as
`M=<m>.json` partials under the mounted cache's `moe-tune/<hash>/`
directory (one hash per run; the same hash recurs for identical
arguments), so a crashed or lost merged file can always be rebuilt from
the partials. For the sweep watcher, keep `Traceback` **out** of `--fail`:
a strict-warm failure prints one and the wrapper's retry-from-checkpoint is
the expected recovery; `--fail 'giving up on M=|WORKERS FAILED'` catches
only real failures.

### Speculative decoding (MTP)

The checkpoint ships one MTP layer; `--speculative-algorithm NEXTN` loads it
as the draft model (+45–70 s load; 7.1 GiB of VRAM, not the 0.2 GB the
load log claims, see the accounting below) and captures target-verify and
draft graphs. It needs real headroom: the mamba pool grows a 2.3 GB
`intermediate_ssm_state_cache`, and before the fp8 KV cache and the 262144
token cap became the defaults the eager GDN prefill OOMed on prompts over
~1k tokens (lowering `--mem-fraction-static` does not help, the budget just
moves into the pools; `--max-total-tokens 131072 --max-mamba-cache-size 30`
was the workaround). An earlier note here said the request cap also had to
be halved to 10 because a 26k-token prefill OOMed at 20; that was inferred
from pool arithmetic, not measured, and on the patches ≤ 26 image it is
wrong. Measured with `sudo amdgpu_top` (driver view; the server log's
"avail mem" is meaningless on this ROCm build), 96 GiB total:

| MTP, cap | Idle after load | Worst case seen | Free at worst |
|---|---:|---|---:|
| 10 requests | 84.6 GiB | 9 streams + 99k prefill: 87.6 GiB | 8.6 GiB |
| 20 requests, patches ≤ 24 | 90.1 GiB | 18 streams + two 43k prefills together: 94.8 GiB | 1.2 GiB |
| 20 requests, patches ≤ 26 | 90.1 GiB | same: 94.5 GiB | 1.5 GiB |
| 20 requests, patches ≤ 27 (embedding + vision tower parked in host memory) | 88.1 GiB | same: 92.7 GiB (two 71k prefills: also 92.7) | 3.3 GiB |
| 20 requests, patches ≤ 27, 71-token short prompts instead of 300-token ones | 88.1 GiB | 18 streams + two 26k prefills: 95.9 GiB | **0.1 GiB** |
| 20 requests, patches ≤ 28 | 87.9 GiB | either mix: 90.4 GiB | 5.8 GiB |

Idle at cap 20 (87.9 GiB on the patches ≤ 28 image) is accounted for by
the driver trace of the launch, in four steps: target weights 67.5 GiB
after the loader releases its staging buffers (2.0 GiB more are parked in
host memory by patch 27, see Memory); **the draft module 7.1 GiB**; pools
12.2 GiB; graphs and caches 1.1 GiB. The load log's "MTP mem usage=0.12
GB" is wrong for the same reason its `avail mem` is (see Memory). The 7.1
GiB is the MTP layer built in bf16, 4.7 GiB (512 experts × 3 × 640 × 2560
× 2 B; `mtp.*` is on the checkpoint's quantization ignore list, and the
draft's MoE kernel looks up an `E=512,N=640` bf16 config where the target
uses `N=320 int4_w4a16`), plus two 1.18 GiB placeholders, the draft's own
`embed_tokens` and `lm_head` (248,320 × 2560 bf16), which
`set_embed_and_head` replaces with the target's tensors right after the
load. The placeholders go back to the caching allocator, not to the
driver, so they stay in the idle figure; the small pools (conv states,
draft KV) are carved from them, which is why the pool step is 12.2 GiB
against the log's 13.3 (5.33 ssm + 4.43 intermediate + 0.3 conv + 3.05 KV
+ 0.26 draft KV). Building the draft without the placeholders would trim
up to 2.4 GiB of reserved memory; quantizing the MTP layer would save
~3.5 GiB but means a different checkpoint. Neither is done. On top of
idle: a prefill transient (~2 GiB of per-chunk activations since patches
25–26 and 28, flat in prompt length; it was 1.6–2.7 GiB growing with
length before 25–26 and up to 6.3 GiB with mixed batches before 28), ~0.9
GiB with 20 streams decoding, and the caching allocator's per-stream pools
under the overlap scheduler (reserved 93.5 vs allocated 88.9 GiB at the
end of the mixed run; `garbage_collection_threshold:0.8` did not change
it). Every run completed: 10 / 16 / 20 streams, 43k and 99k prefills alone
and under 19 decoding streams, two 43k prefills at once with 18 streams.
The scheduler's GTT stayed at 8 MiB throughout, nothing spills to system
memory.

The 0.1 GiB row is why the benchmark suite exists: the hand-run mixes all
used 300-token synthetic short prompts, and the suite's 71-token story
prompts left the long prompt a longer row in the chunked-prefill batch,
which the PLE conv padded every request to, three copies of
`[requests, longest, 10240]` (patch 28). With the packed conv the mix
bottoms at 5.8 GiB whichever prompts are used.

So the default cap works with MTP:

```bash
./start-qwen38.sh --speculative-algorithm NEXTN \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

and gives 25.4–25.9 tok/s single-stream *and* 107–111 tok/s at 20 streams,
which removes the trade-off MTP used to carry. The same suite run against
the same image with and without MTP (DERISKED, cap 20, two passes, first
pass / second pass; the host columns are the container cgroup):

| | Plain graphs | MTP (3 steps, 4 draft tokens) |
|---|---:|---:|
| Idle after load (VRAM used / free) | 77.4 / 19.1 GiB | 87.9 / 8.3 GiB |
| Decode bs 1, 183-token prompt | 15.7 tok/s | 25.4 tok/s |
| Decode bs 1 after 2k / 8k / 32k prompts | 14.5–15.7 / 14.2–15.6 / 14.7–15.6 | 22.0–22.6 / 19.2–20.0 / 23.8–24.6 |
| Prefill, 8k / 32k prompt | 754 / 731 tok/s | 712 / 706 tok/s |
| 4 / 8 / 16 / 20 short streams, aggregate | 33.9–36.0 / 58.3–61.2 / 93.9–99.0 / 93.4–97.7 | 43.1–45.0 / 63.5–65.7 / 94.6–101.0 / 106.7–110.9 |
| 18 short streams + two 26k prefills: min VRAM free | 16.9 GiB | 5.7 GiB |
| Container anonymous / shmem while serving | 2.5 / 2.4 GB | 2.8 / 2.4 GB |

MTP is ahead or even at every point; it costs 10.5 GiB of VRAM (7.1 draft
module, 3.3 pools net of the placeholder reuse, 0.1 graphs) and ~0.3 GB of
host memory. One thing both configurations share: in the mixed run the 18
short streams decode at 2.1–2.3 tok/s each and the last of them waits 21 s
for a first token while the two long prompts prefill in 8k chunks. That is
the chunked-prefill scheduler favouring prefill over decode, not memory;
see "Scheduling: prefill against decode" below.

With patches 27 and 28 the worst case measured leaves 5.8 GiB; before
them 1.5 GiB was thin and one prompt mix reached 0.1. `QWEN38_CUDA_GRAPH_MAX_BS=10` remains the conservative choice
(8.6 GiB free at its worst case) if the workload mixes many long prefills
with a full decode batch, and `QWEN38_HOST_PARKED_PARAMS=` (empty) keeps
every weight in VRAM if the 2 GiB of host RAM matters more.

| Prompt tokens | TTFT | Prefill | Decode (bs=1) |
|---:|---:|---:|---:|
| 183 | 0.8 s | 222 tok/s | 21.0 tok/s |
| 1,650 | 2.9 s | 574 tok/s | 21.2 tok/s |
| 6,693 | 12.8 s | 522 tok/s | 21.8 tok/s |
| 26,363 | 54.6 s | 483 tok/s | 22.4 tok/s |

Mean accept length 2.6 of 4 draft tokens (accept rate 0.53); +45–55%
single-stream decode over plain graphs with the tuned MoE tiles. 4
concurrent: 39.5–40.4 tok/s aggregate (vs 33.3); 8 concurrent: 57.9–58.9
(vs 57.9). More than 10 streams queue (12 streams: 51.6 aggregate, 31 s
worst TTFT). `--speculative-num-steps 2
--speculative-num-draft-tokens 3` was tried: accept length 2.2, bs=1 18.4–19.7
tok/s, 8 streams 60.0, 12 requests allowed; not better. Not on by default: it
trades concurrency for single-stream speed.

The draft's own MoE got tuned tiles on 2026-11-22
(`E=512,N=320,device_name=AMD_Radeon_8060S_Graphics.json`, 12 batch sizes,
produced by the no-launch precompile pipeline above and the strict-warm
sweeps). Same server (rocpro10, cap 8, patches ≤ 49), same suite, second
pass:

| Prompt tokens | TTFT | Prefill | Decode (bs=1) |
|---:|---:|---:|---:|
| 228 | 0.6 s | 371 tok/s | 23.8 tok/s |
| 2,187 | 3.3 s | 658 tok/s | 20.6 tok/s |
| 8,214 | 11.7 s | 703 tok/s | 21.9 tok/s |
| 31,988 | 45.5 s | 703 tok/s | 20.5 tok/s |

Against the untiled-draft table above (183→3,372→13,467 rows): decode
+1–3 tok/s at short prompts and roughly flat at long ones; prefill ~1%
lower at 8k/32k (within run-to-run noise). Mean accept length 2.40 over
the suite (48 samples) vs 2.3–2.6 historical. Concurrent (128-token
prompts, cap 8): 4/8 streams 42.7/68.8 tok/s aggregate, per-stream
11.7/9.4; 16/20 streams queue at the cap (63.6/59.3 aggregate, worst TTFT
26/50 s — cap behaviour, not a draft effect). Mixed 18×26k: shorts decode
p50 8.1 tok/s (min 2.2), longs TTFT 93/134 s, minimum VRAM free 3.1 GiB
(the allocator's high-water mark persists until `/flush_cache`; the mix
bottoms at 5.8 GiB fresh per the pre-29 note above — the 3.1 here is the
post-suite high-water reading, not a new low). Data:
`bench/2026-11-22-bf16-draft-baseline.jsonl`.

**MoE down-projection A/B** (`tools/moe_ab_spotcheck.py`, 2026-11-23,
int4 target shape E=512 K=2560 N=320-packed topk 10 group 32).

**Fixed-seed retune:** the spotcheck now shares weights, activations, and
routing across candidates. The asymmetric int4 calibration compared the
identical configs at M=1/4/12/48/128, with at most 0.9% between files and
3.1% between repeats of the baseline. The down-only sweep screened 228
tiles per size, held the production up row tile at 16, and retimed five
finalists and the baseline in alternating order. Best median times were
57.5/228.7/634.1/1744.0/2795.7 µs versus baseline
58.8/233.3/634.1/1744.0/2795.7 µs. The small apparent gains did not clear
the calibration noise floor. No serving tiles changed.

**Correction (same night): the `_down` candidate file is byte-identical
to the shared config** (md5 `a14c31ad…`, every M entry the same tiles) —
the separate down-tuning sweep converged to the same winners, so there
was no separate-down candidate. The first spotcheck pass
reported large per-M differences between the "two" configs (218 vs 759
µs at M=8, 4.7 vs 1.6 ms at M=512) — with identical inputs those were
**measurement noise, not config effects**: the tool regenerated its
random tensors and expert-ID draws per config, and at small M the
`moe_align_block_size` padding depends heavily on which experts the
draw happens to hit (512 experts, tokens×topk rows), so the padded
launch can differ several-fold run to run. Tool fix shipped alongside:
int4 weights were `.normal_()`-ing a uint8 (NotImplementedError on
ROCm); integral weights are now `random_(0,255)`. The bf16 draft path
was also smoked (M=8: 536 µs → M=1024: 4,991 µs with the server's tuned
draft config; requires the server stopped — bf16 weights need 2.5 GB
and the server leaves only ~0.1 free). Answer to the original question:
per-M-keyed tuning means a genuinely separate down config *would* keep
each M's fastest kernel (small-M entries included), so separation could
only ever help — it just hasn't produced different tiles yet on this
shape.

**Draft-head quality dig** (dev53 verify bench decode lines, 2026-11-23,
20 samples): mean accept length 2.51 of 4 (rate 0.50), but split by
context: ~2.75–2.95 at 128-token prompts vs ~1.83–1.92 at the 32k
prompt. The NEXTN draft degrades with context length — at 32k the
speculation buys only ~1.9 tokens/step instead of ~2.9. Relevant when
evaluating n-gram vs NEXTN (n-gram quality is length-independent for
repeated substrings) and when weighing a draft-head retune for long
contexts.

Cap raised 8→12 with graph tiers 1..12 (`QWEN38_CUDA_GRAPH_MAX_BS=12`,
`QWEN38_MAX_RUNNING_REQUESTS=12`, hotsets 2M rows, patches ≤ 52,
`max_total_num_tokens=358912` after the 60 mamba slots; rocpro10,
2026-11-23, `bench/2026-11-23-cap12-dev52.jsonl`, same suite, 2 passes):

| Prompt tokens | TTFT (p1/p2) | Prefill (p2) | Decode (p2) |
|---:|---:|---:|---:|
| 228 | 0.7 / 0.6 s | 366 tok/s | 23.5 tok/s |
| 2,187 | 3.9 / 3.3 s | 654 tok/s | 20.5 tok/s |
| 8,214 | 11.4 / 11.4 s | 718 tok/s | 21.8 tok/s |
| 31,988 | 47.0 / 45.8 s | 698 tok/s | 20.5 tok/s |

Single-stream is flat vs the cap-8 baseline (p2 within noise); p1
(cold-cache) rows are *better* than baseline p1 (prefill 330 vs 194 at
228, 724 vs 600 at 8k) — the 2M hotset catches more of the first-touch
misses. Concurrent 128-token (p2, cap-8 baseline in parens): 4 → 46.4
(42.7) agg; 8 → 65.6 (68.8) agg, per-stream 8.9 (9.4) — slight tax at 8;
12 → **78.9 agg, per-stream 7.2**, no queue (ttft 1.5 s) — the new sweet
spot, +14% aggregate over the best cap-8 point; 16 → 68.3 (63.6),
ttft_max 30 s; 20 → 72.2 (59.3), ttft_max 32 s (baseline 50) — the two
extra slots drain the queue faster even above cap, but aggregate at
16/20 sits *below* n=12, so raising cap beyond 12 trades per-stream and
aggregate for queue latency only. Mixed 18×short + 2×26k: shorts p50
6.6 tok/s with the queued tail at 2.1 (6 shorts wait behind the cap
while both 26k prefills run; ttft_max 33.5 s, longs 75/118 s), zero
errors everywhere. Min VRAM free 2.8 GiB after the suite (3.6 before) —
above the 2 GiB health line but thinner than cap-8's 3.1. **Verdict:
cap 12 stays; 16 rejected** (aggregate down, VRAM margin down).

Patches ≤ 53 (boot-time work: mlock2 ONFAULT hotsets, keys-only lazy
warm start, 5-s progress cadence, recolored status lines) approved for
**permanent rollout** 2026-11-23 after the dev53 verification bench;
deployed as `strix-halo-sglang:dev53` tagged `rocpro10`, env
`~/.config/qwen38-sglang.env` unchanged. Boot 15:06→15:07:29 (about one
minute, restart to ready), both hotsets warm-started 1,071,425 keys
lazily, reload progress lines at 5-s cadence. Verification bench
(`bench/2026-11-23-dev53-lazywarm-verify.jsonl`, same suite, dev52 in
parens): single-stream p2 decode 23.8/19.4/22.0/20.5 tok/s (23.5/20.5/
21.8/20.5), prefill 373/664/724/707 (366/654/718/698); concurrent n=12
agg 75.3 (74.3–78.9 pass-to-pass band), per-stream 7.0 (7.2); zero
errors, min VRAM free 2.8 GiB. Lazy warm start costs nothing measurable
— even pass 1 matches dev52's, so the hotset converges from live
traffic fast enough to be invisible to this suite.

Verified end to end: chat with thinking (`reasoning_content` split out),
structured tool calls (`finish_reason: tool_calls`), vision (exact OCR of
rendered text plus shape/color identification, 128 image tokens), video
(`video_url`, a 16-frame clip described frame-accurately; the image ships
`decord` as the decoder backend, SGLang's `srt_hip` extras install none), 8
concurrent streams, 6302-token prompt with radix-cache reuse. The prompts
that trip, or once tripped, the server over control tokens are collected in
[CONTROL_TOKEN_EDGES.md](CONTROL_TOKEN_EDGES.md) with a live probe
([`tests/live_control_token_edges.py`](../tests/live_control_token_edges.py)).

On HIP upstream switches EAGLE/NEXTN to rejection sampling
(`speculative_use_rejection_sampling`, log line "ROCm needs rejection
sampling for EAGLE spec-decode to sample at all"); the greedy verify kernel
is CUDA-only. Greedy requests still commit the target's argmax at every
step, but until patches 22 and 23 the MTP path was not repeatable: the
first draft token was a random draw (patch 22) and the draft's sparse
attention never saw the tokens drafted since the capture, reading a stale
scratch slot instead (patch 23). Details and the trace that found them are
in the two patch notes.

DERISKED with the same MTP flags (`QWEN38_CUDA_GRAPH_MAX_BS=10`, 3 steps, 4
draft tokens), patches ≤ 23, one 128-token generation per row:

| Prompt tokens | TTFT | Prefill | Decode (bs=1) |
|---:|---:|---:|---:|
| 183 | 0.6–0.7 s | 275–292 tok/s | 21.9–23.0 tok/s |
| 3,372 | 4.6 s | 727 tok/s | 19.5 tok/s |
| 13,467 | 19.2 s | 701 tok/s | 16.7 tok/s |
| 54,015 | 82.5 s | 655 tok/s | 18.3 tok/s |

4 / 8 / 10 streams (200-token stories, warm kernels): 42.0 / 47.3 / 53.3
and 42.0 / 46.9 / 49.6 tok/s aggregate, per-stream 11.3 / 6.5 / 5.6–6.1;
the first pass after a start is lower (35.3 / 40.8 / 45.6) while Triton
compiles the new shapes. Before patches 22 and 23 the same box gave 25.3 /
16.9 / 17.8 / 19.8 tok/s at bs 1 and 37.5 / 46.4 / 50.3 aggregate, so the
fixes cost nothing. Mean accept length 2.3–2.6 on short prompts (2.55 over a
512-token completion, was 2.3–2.8 and varying), 3.3–3.4 on the 2.5k- and
14k-token prompts (was 3.1). Against plain graphs on the same checkpoint
(14.5–14.8 bs 1, 48.1 at 8 streams) that is +50% single stream and about
even at 8; the request cap of 10 is the price.

With patch 24 (same flags, 131k context) the batched steps get cheaper: bs 1
25.3–25.9 tok/s on the 183-token prompt, 22–24 / 17–19 / 21–24 at 3.4k /
13.5k / 54k, and 4 / 8 / 10 streams 48.1 / 56.8 / 65.6 then 52.7 / 64.1 /
69.6 tok/s aggregate (per-stream 13–14 / 8–9 / 7.6–8.1). Determinism holds:
96 tokens 3/3 and the 14k-token prompt 64 tokens 3/3, identical accept
histograms.

Determinism under MTP, every run cold (`/flush_cache`, `temperature=0`,
top-1 logprob compared at every position, accept histogram compared):
19-token prompt 96 tokens 4/4 identical and 512 tokens 3/3, 2.5k-token
prompt 128 tokens 3/3, 14k-token prompt 64 tokens 3/3. Before the two
patches the 96-token probe differed 3/3 (same tokens, logprobs from
position 6–7, a token flip at 79 in one run, `spec_verify_ct` 40–42).

### Scheduling: prefill against decode

On the deployed Q8 QSA stack, the
[2026-11-25 isolated HTTPS trial](../bench/2026-11-25-prefill-blocks.md)
kept 1,024-token chunks. The 16,384-token arm fell below 1 GiB free VRAM,
tripping the 2 GiB guard, and took 69.4 s vs. 49.0 s on the isolated 32k
prompt. A follow-up trial with patches 55 and 56 completed real mixed
forwards with NEXTN at 2,048 and 4,096 tokens. Neither arm cleared the
15% prefill throughput gate, so no new serving configuration was selected.

SGLang's scheduler is prefill-first: while anything is waiting to prefill it
runs prefill chunks back to back and the running decode batch gets no step.
With `--chunked-prefill-size 1024` a chunk is ~1.3 s here, so one 40k
cache-miss prompt freezes every stream that is decoding for about a minute,
and a queue of them for as long as the queue lasts. 512-token chunks cost
~40% of prefill throughput and 1024 none (vs 8192), so the chunk stays at
1024; what fixes the freeze is `--prefill-decode-interval N`, which forces a
decode step after at most N prefill chunks. The policy
(`--schedule-policy`) then decides who waits.

Measured 2026-10-25 with [`~/scratch/run_sched_bank.sh`](../../scratch/run_sched_bank.sh)
(`sglang_workload_bench.py`, 420 s closed loop: 4 agent sessions with
cached-prefix turns, a short 750-token prompt every 40 s, 12 fan-out 40k
cache-miss prompts in two waves, two 50k prompts; ~700k prefill tokens, so
about 20 min of prefill work; client TTFT and token gaps, plus the patch 35
event stream recorded per config with `record_events.py`; summary from
`analyze_bank.py`). Worst decode gap = the longest wait for the next token
of a request that was already decoding:

| config | out tok/s | agent turns | worst decode gap | agent decode tok/s | short TTFT p50/max | agent TTFT p50/p95 |
|---|---|---|---|---|---|---|
| fcfs (old default) | 10.6 | 7 | **275 s** | 1.7 | 1 / 192 s | 174 / 246 s |
| fcfs + interval 10 / 20 / 40 | 9.0–9.2 | 8 | 2.6–2.7 s | 12–20 | 1 / 346–356 s | 346–363 / 408–424 s |
| hrrn / lpm / spf | 12.8 / 11.9 / 11.1 | 5 | 204–273 s | 1.5–2.2 | 1 / 329–135 s | 115–175 / 226–232 s |
| hrrn + interval 20 | 14.4 | 13 | 2.0 s | 13.6 | 71 / 180 s | 67 / 400 s |
| **spf + interval 20** | **15.4** | **14** | 2.2 s | 13.6 | 71 / **86 s** | **66 / 313 s** |
| lpm + interval 20 | 14.7 | 14 | 2.6 s | 13.3 | 1 / 299 s | 81 / 313 s |

The interval is the decisive knob: without it every policy freezes decode
for minutes; with it the worst gap is ~2 s and agent turns complete at
10–20 tok/s. Its cost under a flood is 10–20% slower prefill drain, which
the real traffic here (decode concurrency ~1, 45% of prefills mostly
cached) never sees. The interval value barely matters (10/20/40 within
noise); 20 is roughly equal GPU time for a chunk and a decode step.

Policy matters once the interval is on: fcfs + interval was the slowest
configuration of all, `shortest-prefill-first` the fastest, and the only one
where no short prompt waited more than ~86 s (one 40k prefill: an arrival
waits for the in-progress chunked request to finish, since that request
takes the whole chunk budget each step; new requests join between
requests, not between chunks). hrrn rates a fresh short prompt below a
large one that has been waiting (response ratio ≈ 1 vs 3+), so shorts
waited up to 180 s; lpm let one wait 299 s. The price of spf is that big
uncached prompts queue behind small ones (the 50k prompt waited 476 s in
the flood vs 241 s under fcfs); under sustained short traffic a huge prompt
could in principle wait indefinitely, which the measured traffic does not
produce.

Default now: `--schedule-policy shortest-prefill-first
--prefill-decode-interval 20` (in `QWEN38_EXTRA_ARGS`), with
`--enable-cache-report` so `usage.prompt_tokens_details.cached_tokens` is
returned. Raw results: `~/scratch/workload_results.jsonl`,
`~/scratch/events/<label>.jsonl`, thermal CSVs in `~/scratch/thermal/`
(GPU at the 135 W power limit throughout, 70–80 °C).
## Running the abliterated variant (DERISKED)

`davetha/Qwen3.8-Flash-Next-DERISKED-W4A16-AWQ` is a refusal-ablated requant
of the same architecture: compressed-tensors W4A16, **symmetric, group 128**,
experts *and* the 12 full-attention layers' `q/k/v/o` quantized; indexer,
GDN, PLE, MTP, gates, norms, `lm_head` and the vision tower bf16 (the 333
vision tensors are byte-identical in name and shape to cyankiwi's). Kept
separate from the stock checkpoint end to end: own model directory, own PLE
directory (the table file name is the same for every checkpoint), own
container and served name, and its own benchmark and tuning data.

What had to change to run it, and where:

| Difference | Handling |
|---|---|
| Quantized dense attention projections | [Patch 16](../patches/16-wna16-rocm-dense.md): dequantized to bf16 at load, served with `F.linear` (the dense WNA16 scheme is Marlin-only; the MoE path already had a ROCm Triton kernel). Costs no extra bandwidth over the stock checkpoint's bf16 attention. |
| `model-mtp-merged.safetensors` is 100 GiB (all 128 PLE shards + MTP in one file) | `tools/convert_ple_fp8.py --part-bytes 2GiB` splits any oversized file into PLE-only `-pleNNN` and `-restNNN` parts and rewrites the index, so patch 13's PLE-shard skip still applies. The converter also stopped using `safe_open`: safetensors maps the file `PROT_WRITE|MAP_PRIVATE`, which overcommit mode 0 refuses for 100 GiB on a 30 GB host; it now reads headers and tensors with plain seek/`readinto`. |
| Chat template prepends a "Qwentium" obedience persona to every system block | Renamed to `chat_template.derisked.jinja`; the stock template is used. It is a template, not weights: the model's behaviour was probed without it. |
| No `preprocessor_config.json` / `video_preprocessor_config.json` | Copied from the stock checkpoint (identical processor). |
| Symmetric g128 experts | Same Triton kernel; the stock (g32-tuned) `E=512,N=320` tiles measure the same on g128 as on g32 in the tuner's benchmark mode (bs 1/8/16/20/32: 103/734/1418/1569/2242 µs vs 105/750/1347/1611/2305). A full sweep on the g128 checkpoint (`configs/moe/qwen38-flash-next-derisked/`) wins that microbenchmark by 6–9% but loses 7–9% end to end, so the launcher keeps the stock profile mounted; details below. |

Conversion (the 100 GiB file needs a memory cap only to keep the page cache
honest; RSS stays under 2 GiB):

```bash
docker run --rm --memory 14g \
    -v /scratch/hf-staging/Qwen3.8-Flash-Next-DERISKED-W4A16-AWQ:/src:ro \
    -v /opt/llm/models/Qwen3.8-Flash-Next-DERISKED-W4A16-ple-fp8:/dst \
    -v $PWD/tools/convert_ple_fp8.py:/convert.py:ro \
    strix-halo-sglang:dev python3 -u /convert.py /src /dst --part-bytes 2GiB
cd /opt/llm/models/Qwen3.8-Flash-Next-DERISKED-W4A16-ple-fp8
mv chat_template.jinja chat_template.derisked.jinja
cp /path/to/stock/{chat_template.jinja,preprocessor_config.json,video_preprocessor_config.json} .
```

The template rename is belt and braces now: the launcher serves
[`configs/chat/qwen38.jinja`](../configs/chat/qwen38.jinja) via
`--chat-template`, so the file in the model directory is not read unless
`QWEN38_CHAT_TEMPLATE=` is set empty.

Launch beside (not with: the GPU holds one of these) the stock server:

```bash
SGLANG_CONTAINER=sglang-qwen38-derisked QWEN38_SERVED_NAME=qwen38-flash-next-derisked \
MODEL_DIR=/opt/llm/models/Qwen3.8-Flash-Next-DERISKED-W4A16-ple-fp8 \
PLE_DIR=/opt/llm/ple-cache-derisked ./start-qwen38.sh
```

Loads in ~6 min (first boot writes its own 47.7 GiB table), ~1.5 GB more
free VRAM after load than stock, same graph list and request cap. Verified: 12 greedy probes answered (the stock model
refused none of them either; the difference is in tone, not in refusals, for
that set), identity unchanged, vision exact on the synthetic test image
(160 image tokens). Greedy decode is bit-identical across cold runs since
patches 18, 20 and 21; before them, this checkpoint (and stock) drifted from
the first decode token on, see below.

| | Stock (cyankiwi g32) | DERISKED (g128), patches ≤ 18 | DERISKED (g128), patches ≤ 21 |
|---|---:|---:|---:|
| Prefill ~1.6k / ~7k / ~27k tokens | 535 / 554 / 475 tok/s | 535 / 538 / 504 tok/s | 629 / 695 / 676 tok/s (2.6k / 11k / 44.6k) |
| Decode bs=1 | 14.5 tok/s | 14.3–15.0 tok/s | 14.5–14.8 tok/s |
| 8 streams | 57.9 (sanity re-run 49.6) | 47.1 / 51.8 | 48.1 |
| 16 streams | 99.4 (sanity re-run 89.5) | 70.6 / 73.9 / 72.7 | 71.3 |
| 20 streams | 88–97 | 72.9 | 71.5 |

Single-stream matches stock, and prefill is now ~20–30% ahead of the
stock column: patch 21 removed a per-query-row Python loop (one host sync
per row) from the QSA indexer's block selection, which is worth ~1.4 s per
1.5k-token prefill across the 12 QSA layers.

Stock re-measured on the same patches ≤ 21 image, same day, same prompts:
prefill 263 / 675 / 694 tok/s (183 / 2.6k / 11k-token prompts), decode
13.8 / 13.3 / 13.4 tok/s, 8 / 16 / 20 streams **42.8 / 67.7 / 71.7**
tok/s, greedy repeats bit-identical (96 tokens 3/3, 8.6k-token prompt 3/3).
So stock gained the same prefill and now sits at the same concurrency as
the abliterated build: the 16–20 stream gap in the table is not
checkpoint-specific. Why both were ~25% under the 99.4 / 88–97 figures in
the stock column was found by rebuilding the image at the commit that
recorded them (`d1953c8`; the Dockerfile's upstream pin was unchanged and
the `sgl_kernel` binaries came out identical) and running old and new
images at both context lengths:

| Image, stock checkpoint | Context | 8 | 16 | 20 streams |
|---|---:|---:|---:|---:|
| patches ≤ 14 (`d1953c8`) | 32k | 47.9–53.1 | 83.6–85.5 | 87.3–92.7 |
| patches ≤ 14 (`d1953c8`) | 131k | 42.7–48.0 | 71.4–72.5 | 71.6–77.8 |
| patches ≤ 23 | 32k | 47.5–62.4 | 84.8–96.5 | 89.2–93.5 |
| patches ≤ 24, reference MQA forced | 131k | 40.9–53.4 | 66.8–78.9 | 68.4–77.0 |
| patches ≤ 24 | 131k | 50.6–62.4 | **89.2–104.7** | **95.2–103.5** |

The launcher's default context moved from 32768 to 131072 between the two
measurements, and that is the entire gap; patches 15–23 cost nothing. The
mechanism: without TileLang the QSA indexer's decode block scoring ran
upstream's torch reference, which gathers the whole `context_length / 4`
window per row on every decode step (1.8 ms per layer at bs 20 with 32k,
6.0 ms with 131k, × 12 layers; under 1 ms at bs 1, so single stream never
showed it). Patch 24 replaces it with a Triton kernel bounded by each row's
length. Every decode step in both builds runs in a captured graph. The int4
attention projections are served as bf16 (patch 16), so they cannot be
slower than stock's bf16 ones.

MoE tiles tuned on the g128 checkpoint (full 18-size sweep, 8 h; tiles and
numbers in [`configs/moe/README.md`](../configs/moe/README.md)): in the
tuner's benchmark mode they are 6–9% faster than the stock profile at
batch ≥ 8 (at the true `--tp-size 1` shape: 1285 / 2339 / 2780 / 3942 µs
vs 1390 / 2544 / 3037 / 4340 at bs 8 / 16 / 20 / 32), but end to end,
back to back on the derisked server, 8 / 16 / 20 streams came out
41.8 / 62.4 / 65.4 and 43.3 / 62.9 / 64.3 against 46.5 / 69.0 / 70.2 with
the stock profile, and single stream no better. The launcher therefore
keeps `QWEN38_MOE_CONFIG_DIR` on `configs/moe/qwen38-flash-next/` for this
checkpoint too (the launch command above is unchanged), and the g128
profile stays in the repo as data. Working hypothesis: the tuner routes
tokens uniformly over experts and benchmarks half the real GEMM width, so
its M does not correspond to the per-expert row counts the server sees.

End-to-end determinism on the patches ≤ 21 image, every run cold (cache
flushed between runs, `temperature=0`, top-3 logprobs compared at every
position): 96-token completion of a short prompt 3/3 identical; ~430-token
prompt (12 paragraphs) 3/3; ~1.5k-token prompt (20 paragraphs + question),
48 tokens, 4/4; 8.6k-token prompt, 64 tokens 3/3 and 200 tokens 4/4 (this
prompt diverged at step 37 before patch 20 and differed in the first
logprob before patch 21). The 12 behaviour probes (up to 400 tokens each)
are text-identical between two cold passes. Their texts differ from the
patches ≤ 18 image on 9 of 12 probes, as expected: a different (now fixed)
tie order in the block selection changes the rounding, and long greedy
generations part at the next near-tie. Long-context concurrency stress
(4 × 13k, 8 × 5k and 2 × 50k-token prompts) completed with no GPU faults.
With MTP on, the same holds since patches 22 and 23; numbers in
[Speculative decoding](#speculative-decoding-mtp).

## ROCm 10 base (image `strix-halo-sglang:rocpro10`)

An alternative base built on `kyuz0/vllm-therock-gfx1151:latest`
(`rocm10.0.0-torch2.11.0-vllm0.33.0`, pin `sha256:f8a7433d662b...cfa3966`)
with the same patch chain. Launch it with
`SGLANG_IMAGE=strix-halo-sglang:rocpro10 SGLANG_DOCKER_ARGS="-e SGLANG_QSA_TILELANG=0 -e SGLANG_QSA_DECODE_FLASH_ATTN=1"`.
The two env gates stand down the pieces the new stack cannot run:

- `SGLANG_QSA_TILELANG=0` (patch 48): TileLang codegen is broken on the new
  Triton (`N must be divisible by 16, but got 8` at decode graph capture);
  the gate selects the validated Triton MQA kernels (patches 24/26).
- `SGLANG_QSA_DECODE_FLASH_ATTN=1` (existing patch 11 escape hatch): the
  custom `qwen38_qsa_sm121` Triton kernel needs 73,728 B of shared memory on
  the new compiler (over the 64 KiB RDNA limit) even with the HIP clamp, and
  stage-forcing segfaults at the page level; the gate sends decode back to
  the flash-attn varlen fallback that was already the outside-contract path.

Everything else — PLE hot-set cache, presharded loader — runs
unchanged; the hot-set snapshot restored warm on first prefill.
Verified live (paragraph-shuffle bench, server-reported): 865-tok prefill
664/664/710 tok/s, decode 21.0-21.5 tok/s; 1712-tok prefill 670/659/680.
That is at parity with the 7.13 base (`:dev`) or slightly ahead, and
`rocprofv3` profiling works on this base (it crashes the 7.13 server).
Sanity: greedy 17×23 = 391, coherent 400-token generation.

Triton 3.8 MoE config note (found by kernel trace): the new base's Triton
3.8.0 renames the device to `AMD_Radeon_8060S_Graphics`, so the tuned
configs in `configs/moe/qwen38-flash-next/` (named
`device_name=Radeon_8060S_Graphics`) are not found and the loader falls
back to defaults. Do NOT just copy them to the AMD_ name: they were tuned
under the old Triton and measurably regress on 3.8 (decode 12 vs 20 tok/s,
prefill 457-643 vs 631-635). rocprofv3 on this base attributes ~32% of GPU
kernel time to `fused_moe_kernel_gptq_awq` during a 66k prefill, so a
Triton 3.8 retune is the top candidate for closing the remaining gap. The
AMD_-prefixed copies are parked at `/var/tmp/moe-amd-copies/` and must not
be deployed. rocprofv3 usage on the server: wrap with
`SGLANG_COMMAND_PREFIX="rocprofv3 --kernel-trace -o <out> --output-format
csv --kernel-include-regex (?i)gemm|attn|flash|topk|splitk|fmha --"`; to
flush the trace, SIGTERM the scheduler and detokenizer children inside the
container (`kill -TERM <sched_pid> <detok_pid> <tracker_pid>`) — signaling
the entrypoint alone deadlocks the flush and docker's eventual SIGKILL
loses the buffers. Bind-mount the output directory.

## Known limitations

- The "avail mem" log figure is not headroom; see Memory before adding
  anything that allocates.
- Prefill runs eager (upstream disables prefill graphs for this model).
- First boot writes the 48 GiB PLE table; the `pinned` backend is not an
  option here (host RAM is the same pool).
- Fixed, kept for the record: greedy decode was not repeatable. The same
  prompt at `temperature=0` gave different top-1 logprobs from the first
  decode token on (a handful of recurring values), and long completions
  diverged on near-tie tokens; prefill was bit-identical. Graphs, overlap
  scheduling, the radix cache and the PLE fusion were ruled out one by one,
  as were the QSA, GDN, MoE and GEMM kernels by repeat-and-compare tests.
  The cause was the HyperConnection mix: on ROCm every batch of ≤ 16 rows
  ran upstream's persistent Triton kernel, which accumulates its split-K
  down projection with device-scope atomics. Patch 18 replaces it on HIP
  with a two-launch variant (per-split partials, fixed-order reduction, no
  grid barrier) at the same speed; 96- and 512-token completions are now
  bit-identical across cold runs. Sending the mix to the torch path instead
  would have cost ~35% decode speed (the GEMM shapes are untuned under
  TunableOp).
- Fixed, kept for the record: after patch 18, prompts above ~1.4k tokens
  still drifted: same first token, different logprob, and an 8.6k-token
  prompt diverged on a near-tie token around step 37. Two causes, one per
  path. Decode: `select_decode_tokens` called the JIT `fast_topk` kernel
  directly (bypassing patch 11's guard), and its output order varies past
  512 compressed blocks; patch 20 gives decode a vectorised torch top-k.
  Prefill (and, once exposed, that torch top-k): `torch.topk` on this ROCm
  build orders *tied* scores differently per launch, and the indexer's
  `relu` block scores tie constantly; every other prefill stage (conv, GDN
  chunk kernel, indexer logits, index expansion, sparse attention, GEMMs,
  MoE, HC mix) was bit-stable in repeat-and-compare tests at 1,475 and
  8,192 tokens. Patch 21 breaks ties toward the lower block with a stable
  sort (prefill) or a top-k over unique score+index keys (decode). End-to-end
  numbers are below the DERISKED table: 8.6k-token prompt, 200 greedy
  tokens, 4/4 cold runs identical to the logprob.
- Fixed, kept for the record: with MTP on, greedy decode was still not
  repeatable after patch 21 (same tokens for ~80 positions, logprobs
  differing from position 6–7, a token flip after ~80, the verify count and
  accept histogram changing run to run); without MTP the same image was
  bit-identical. Overlap scheduling and CUDA graphs were ruled out. A
  per-module GPU checksum trace of the target and draft forwards found two
  causes. First (patch 22): HIP defaults EAGLE/NEXTN to rejection sampling,
  and the draft pass after prefill drew its first draft token with
  `fast_sample` instead of the greedy-aware `sample_draft_proposal`, so the
  first draft was random at `temperature=0`; the target still committed its
  argmax, but the accept/reject changed the verify batch shape and with it
  every later rounding. Second (patch 23): the draft's sparse attention
  reuses the block selection captured at draft-extend plus a tail of the
  positions drafted since, and that tail was written after the row's `-1`
  padding while the KV gather packs valid entries as a prefix, so the
  drafted positions were dropped and the packed slot read stale scratch from
  the previous request. Both are upstream bugs, not gfx1151 ones; on CUDA
  the second is silent (zero-filled slot, deterministic, draft blind to the
  newest tokens). A first attempt at the trace, a `sitecustomize` hook that
  imported torch into every spawned process, OOMed the 30 GB host during
  model load and forced a reboot; the working tracer hooks the scheduler's
  model only, sums on the GPU, and the container ran under `--memory 22g`.
- Fixed, kept for the record: 16–20 stream throughput sat at 67–78 tok/s
  on every image after September 21 against 99.4 / 88–97 measured then,
  and was blamed on patches 18–21 or variance. Rebuilding the old image
  from its commit and crossing image × context length showed the launcher's
  context default (32k → 131k) was the whole effect: the QSA indexer's
  decode scoring ran a torch reference whose cost scales with the context
  limit rather than the request (no TileLang in the image). Patch 24
  replaces it with a length-bounded Triton kernel; 89–105 / 95–104 tok/s
  at 16 / 20 streams at the full 131k context, greedy outputs identical
  between the two paths.
- Fixed, kept for the record: eager decode above the largest captured graph
  used to fault the next replay (`Memory access fault by GPU node-1 ... Page
  not present`) after a `/flush_cache`. Two factors: upstream's QSA backend
  shares one growable packed-KV scratch between captured graphs and eager
  decode, so an eager step larger than any graph re-allocated it and the
  graphs kept writing into the freed block; and `empty_cache()` then unmapped
  that block. Patch 15 gives the graphs their own scratch. Verified with
  graphs for bs ≤ 8 and 20 streams, flush between rounds, 3 rounds clean,
  greedy probe identical. Without the flush the stale write was silent.
