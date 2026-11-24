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
