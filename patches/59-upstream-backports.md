# 59: Selected upstream backports

The [build script](patch_upstream_backports.py) applies the [vendored diffs](backports59/) offline, after patch 58.

| Upstream | What changes on this fork |
|---|---|
| [Mixed KV guard](https://github.com/sgl-project/sglang/pull/43290) | After prefill admission and filtering, check the allocator before preparing the running decode rows. If they no longer fit, return pure prefill and keep the running batch intact for a later decode. This also applies when `fsched` proposed a mixed batch. It does not retract a decoder to satisfy prefill. |
