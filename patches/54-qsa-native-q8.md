# 54. Native Q8 QSA QKV on gfx1151

Opt-in with `SGLANG_QSA_Q8_DIR` inside the container. The sidecar must be
mounted read-only and built with `tools/prepare_qsa_native_q8.py` from the
TP=1 BF16 presharded dump and the verified Q8 quality overlay. Startup
checks source and packed hashes before graph capture. The draft and all
other layers are untouched. No checkpoint or presharded data is rewritten.
Unset the variable and restart to use the original BF16 projections.

Each 32-weight block has signed int8 codes and a shared unsigned E4M3
half-scale byte. Codes and scales are stored in separate arrays for aligned
GPU loads. At M=1–16, Triton decodes to BF16 in registers and accumulates
the dot product in FP32. Larger batches transiently unpack one projection
and use BF16 `F.linear`. The sidecar replaces the BF16 module after load,
so original weights are still needed during startup. Resident projection
storage falls from 780 MiB to about 402 MiB; peak memory depends on the
allocator and the active graph and prefill paths.

Validation: `tests/check_qsa_native_q8.py` in the patched image on gfx1151.

## Serving opt-in and rollback

On cerebro, the opt-in is `95-qsa-native-q8.conf` in the
`qwen38-sglang.service.d` user drop-in directory. It reads
`~/.config/qwen38-sglang.env.qsa-native-q8`. That file selects the tested
`strix-halo-sglang:qsa-q8-trial` image (image ID
`sha256:fb02d14de02927cf31483ae6b3a86b470ae400e07a777fc5e53e8301d8eaf8fe`)
and extends the existing draft Docker arguments with:

```text
-e SGLANG_QSA_Q8_DIR=/qsa-q8 -v /home/llm/.local/share/qwen38-qsa-q8/TP-1-v2-m1a651b05:/qsa-q8:ro
```

The sidecar is local deployment data, not tracked in Git. Keep the
draft's existing mounts in `SGLANG_DOCKER_ARGS`; an environment file
override replaces the entire value. The drop-in does not change the
serving port, context, KV dtype, or NEXTN settings.

Before restarting, check for active requests. To roll back, archive only
`95-qsa-native-q8.conf` outside the drop-in directory, run
`systemctl --user daemon-reload`, then restart `qwen38-sglang.service`.
The original `rocpro10` image and BF16 QKV weights remain in place.
Verify `/health` on port 8000 and three-step NEXTN in `/get_server_info`.
On a Q8 boot, also check the journal for
`Native QSA Q8 installed on 12 target projections`.
