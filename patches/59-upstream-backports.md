# 59: Selected upstream backports

The [build script](patch_upstream_backports.py) applies the [vendored diffs](backports59/) offline, after patch 58.

| Upstream | What changes on this fork |
|---|---|
| [Mixed KV guard](https://github.com/sgl-project/sglang/pull/43290) | After prefill admission and filtering, check the allocator before preparing the running decode rows. If they no longer fit, return pure prefill and keep the running batch intact for a later decode. This also applies when `fsched` proposed a mixed batch. It does not retract a decoder to satisfy prefill. |
| [Structured tool arguments](https://github.com/sgl-project/sglang/pull/44124) | If a declared `anyOf`/`oneOf` permits both a string and a container, return a complete JSON object or array only if it validates against the schema. Otherwise preserve the string. Both parser entry points use the enclosing schema for local references; the validator cannot fetch remote schemas. Patch 33's control-token boundaries and patch 36's recovery for malformed, explicitly typed containers and undeclared parameters remain in place. |
| [Speculative forward-pass metrics](https://github.com/sgl-project/sglang/pull/38711) | When `--enable-forward-pass-metrics` is enabled, attach each group of existing device-timer intervals to its result, including PP relay and skip paths. Publish the same-stream span after the events are ready, with decode lengths captured before speculative settlement. An empty publisher queue no longer implies that the scheduler is idle. The port includes the upstream timer's nonblocking readiness API and uses this pin's scheduler idle/active signatures. FPM stays off by default; it neither adds GPU time to journal lines nor changes `fsched` accounting. |
