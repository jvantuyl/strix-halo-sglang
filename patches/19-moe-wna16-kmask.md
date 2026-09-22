# Patch 19: mask the weight load for a partial last K block in the GPTQ/AWQ MoE kernel

**File:** `python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py`
**Script:** [`patch_moe_wna16_kmask.py`](patch_moe_wna16_kmask.py)

## Symptom

Tuning the fused-MoE tiles for the DERISKED (g128) checkpoint killed the
tuner's Ray worker with a GPU page fault at the same config on every attempt:

```
amdgpu: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:9472)
amdgpu:  Process ray::IDLE pid 207485 ...
amdgpu: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801030  Faulty UTCL2 client ID: TCP
```

The per-config trace the tuner now writes (`partial/trace-M=<m>.log`) put
it at index 60 of the 1,560-config sweep,
`BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, num_warps=1`: the first
config with `BLOCK_SIZE_K=128`. The same sweep had completed on the g32
checkpoint (same `E=512, N=320` shape) a day earlier.

## Cause

In `fused_moe_kernel_gptq_awq`'s K loop, when K is not a multiple of
`BLOCK_SIZE_K` (`not even_Ks`) the kernel masks `a`, `b_scale` and `b_zp`
for the partial last block but loads the packed weight block unmasked:

```python
b = tl.load(b_ptrs)
```

The last iteration therefore reads `BLOCK_SIZE_K - K % BLOCK_SIZE_K` rows
past the expert's weights; for the last expert that is past the end of the
tensor. Whether that is a fault depends on what the caching allocator placed
after the tensor, so it is layout-dependent: the g32 sweep's scale tensors
are 4× larger and the over-read landed in mapped memory. The tuner's default
`--tp-size 2` shard gives the down projection K = 320, and 320 = 2 × 128 + 64.
The masked elements meet zeros in `a`, so results were never affected; only
the read itself was out of bounds. The generic `fused_moe_kernel` in the same
file already masks `b` under `not even_Ks`.

## Fix

Mirror the generic kernel:

```python
if not even_Ks:
    b = tl.load(b_ptrs, mask=k_mask, other=0)
else:
    b = tl.load(b_ptrs)
```

`even_Ks` is a constexpr, so every shape the runtime uses for this model
(K = 2560 and 640, multiples of all candidate tiles) compiles to exactly the
same code as before. Not gfx1151-specific.

## Verification

`tools/test_qwen38_rocm.py` item 8 runs the kernel at K = 320 with
`BLOCK_SIZE_K` 128 and 256 (symmetric int4, group 64) against the
dequantized reference. The tuner run itself is the out-of-bounds check: with
the patch mounted, M = 1 passed index 60 and continued; without it the worker
died there twice in a row (`trace-M=1.faulting.log`).
