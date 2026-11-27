# 59: Selected upstream backports

The [build script](patch_upstream_backports.py) applies the [vendored diffs](backports59/) offline, after patch 58.

| Upstream | What changes on this fork |
|---|---|
| [Mixed KV guard](https://github.com/sgl-project/sglang/pull/43290) | After prefill admission and filtering, check the allocator before preparing the running decode rows. If they no longer fit, return pure prefill and keep the running batch intact for a later decode. This also applies when `fsched` proposed a mixed batch. It does not retract a decoder to satisfy prefill. |
| [Structured tool arguments](https://github.com/sgl-project/sglang/pull/44124) | If a declared `anyOf`/`oneOf` permits both a string and a container, return a complete JSON object or array only if it validates against the schema. Otherwise preserve the string. Both parser entry points use the enclosing schema for local references; the validator cannot fetch remote schemas. Patch 33's control-token boundaries and patch 36's recovery for malformed, explicitly typed containers and undeclared parameters remain in place. |
