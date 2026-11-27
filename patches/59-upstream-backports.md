# 59: Mixed admission, Qwen tool arguments, speculative timing and prefix restore

Five upstream diffs, including their regression tests, rebased on the
September SGLang pin **after** patches 1–58. The [build script](patch_upstream_backports.py)
applies the [series](backports59/) offline. The default build profile,
`production`, includes the mixed KV guard, Qwen parser fix, and timing work.
Only `--build-arg SGLANG_BACKPORT_PROFILE=hicache-trial` adds the PLE and
QSA checkpoint restore diffs as a pair. This is a build-time choice, not a
server flag. Both profiles keep the validated OpenSSL file-hash dependency.

| Upstream | What changes on this fork |
|---|---|
| [Mixed KV guard](https://github.com/sgl-project/sglang/pull/43290) | After prefill admission and filtering, check the allocator before preparing the running decode rows. If they no longer fit, return pure prefill and keep the running batch intact for a later decode. This also applies when `fsched` proposed a mixed batch. It does not retract a decoder to satisfy prefill. |
| [Structured tool arguments](https://github.com/sgl-project/sglang/pull/44124) | If a declared `anyOf`/`oneOf` permits both a string and a container, return a complete JSON object or array only if it validates against the schema. Otherwise preserve the string. Both parser entry points use the enclosing schema for local references; the validator cannot fetch remote schemas. Patch 33's control-token boundaries and patch 36's recovery for malformed, explicitly typed containers and undeclared parameters remain in place. |
| [Speculative forward-pass metrics](https://github.com/sgl-project/sglang/pull/38711) | When `--enable-forward-pass-metrics` is enabled, attach each group of existing device-timer intervals to its result, including PP relay and skip paths. Publish the same-stream span after the events are ready, with decode lengths captured before speculative settlement. An empty publisher queue no longer implies that the scheduler is idle. The port includes the upstream timer's nonblocking readiness API and uses this pin's scheduler idle/active signatures. FPM stays off by default; it neither adds GPU time to journal lines nor changes `fsched` accounting. |
| [PLE checkpoint side state](https://github.com/sgl-project/sglang/pull/39862) | Carry the registered short-convolution state and n-gram history with their Mamba slot through host memory and whole-page storage. Reject a short legacy payload before writing to the destination. Zero-copy storage that cannot represent the side state is refused. |
| [QSA indexer checkpoints](https://github.com/sgl-project/sglang/pull/39893) | Mirror compressed index keys by full-KV page, register the indexer as a required sidecar, and wait on the layer-transfer event before reading restored keys. Includes target, packed MTP draft and separate draft pools. An old file entry lacking the sidecar is a miss, not a usable KV hit. |

**Cache scope:** The last two patches belong together on Qwen3.8-Flash-Next.
They fix restoring prefixes **after device eviction** with HiCache enabled.
They do not change ordinary device radix hits, prefill admission order, or
the file-backed PLE embedding table. HiCache remains off on cerebro.

## Checking the port

Run `TMPDIR=/var/tmp python3 tests/check_patches.py` on the host. Set
`SGLANG_BACKPORT_PROFILE=hicache-trial` to replay the experimental profile.
For in-image CPU checks, mount this repository read-only at `/repo`, omit
the GPU devices, and use `/sgl-workspace/sglang` as the working directory.
The full focused suite below requires a **hicache-trial** image. On the
default production image, select `check_upstream_backports.py::TestMixedCapacity`
instead of that entire module and omit the two `unit + "mem_cache/..."` cases.
The QSA restore implementation and the new Mamba slot test are intentionally
absent. Keep the parser, timing and mixed reporting checks in both profiles.

```sh
python3 - <<'PY'
import sys, torch, pytest
if not torch.cuda.is_available():
    torch.version.hip = None  # avoid import-time architecture probes in the HIP build
unit = "test/registered/unit/"
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider",
    "/repo/tests/check_upstream_backports.py",
    "/repo/tests/check_mixed_batch_logging.py",
    unit + "function_call/test_function_call_parser.py::TestQwen3CoderDetector",
    unit + "observability/test_forward_pass_metrics.py",
    unit + "managers/test_batch_result_processor_spec_grammar.py",
    unit + "managers/test_pp_cp_rank_offsets.py",
    unit + "mem_cache/test_hicache_mamba_slot_side_states.py",
    unit + "mem_cache/test_hybrid_pool_assembler.py",
]))
PY
```

Also run the fork's `tests/check_tool_param_recovery.py` and
`tests/check_control_token_parsers.py` (the latter needs the Qwen tokenizer).
CPU tests use synthetic event completions and CPU tensors: they cover overlapping
results, the actual file-store hit/miss and payload paths, and a refused mixed
step followed by one that fits. They do **not** measure GPU timing accuracy or
exercise native DMA kernels.
