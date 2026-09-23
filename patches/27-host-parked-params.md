# Patch 27: park bandwidth-free weights in pinned system memory

**Files:** `python/sglang/srt/model_executor/host_parked_params.py` (new),
`python/sglang/srt/model_executor/model_runner.py`
**Script:** [`patch_host_parked_params.py`](patch_host_parked_params.py)

## Symptom

With MTP at the 20-request cap the server idles at 90.1 of the 96 GiB
carve-out, and the worst mixed load measured (18 decoding streams plus two
43k-token prefills at once) bottomed out at 1.5 GiB free after patches 25
and 26. Everything in that 90 GiB is accounted for (weights 75.9, MTP layer
0.2, KV 3.05, GDN state 9.4, graphs 0.6), so the only way to widen the
margin without shrinking the pools is to stop keeping weights in VRAM that
the GPU hardly reads.

## Cause

Two weight groups have near-zero bandwidth demand. The token embedding
(`model.embed_tokens.weight`, 248,320 × 2,560 bf16, 1.18 GiB) is read as a
`bs`-row gather once per decode step, a few kilobytes. The vision tower
(`visual.*`, 0.9 GiB) is idle unless an image arrives. `lm_head` looks
similar but is a full GEMM against the whole vocabulary every step and
stays in VRAM. On Strix Halo VRAM and system RAM are the same LPDDR5X, but
GPU reads of system memory go through the IOMMU (kept on for the NPU) and
are measurably slower, so only these two qualify.

## Fix

`host_parked_params.py`, no compiled code. `HostParkedBuffer` calls
`hipHostMalloc(hipHostMallocMapped)` through ctypes on the HIP runtime
torch already loaded (found in `/proc/self/maps`) for an allocation of
exactly the parameter's size, takes `hipHostGetDevicePointer`, and exposes
`__cuda_array_interface__`; `torch.as_tensor` then aliases the device
pointer as a CUDA tensor with no copy. `park_model_parameters` walks
`named_parameters()`, and for every name containing one of the
`SGLANG_HOST_PARKED_PARAMS` patterns copies the weight into such a buffer,
repoints the Parameter's `.data` at the alias (every reference to the
Parameter sees the move, `weight_loader` and friends stay attached), and
calls `empty_cache` so the caching allocator returns the VRAM. The buffer
is owned by the new storage: torch holds a reference to the
`__cuda_array_interface__` producer until the storage dies, so nothing
else has to keep it alive, and a parked Parameter a model later replaces
frees its host memory by itself (the module keeps only weak references, by
device address, so `is_parked` can recognise a Parameter shared between
models and skip it). `model_runner.load_model` calls it right after the
loader hands back the model, on CUDA devices and not for draft runners.
Unset or empty, the hook does nothing.

Two things the obvious version got wrong, both seen on the first end-to-end
run. torch's own pinned allocator (`pin_memory=True`) rounds requests up to a
power of two, so the 1.18 GiB embedding locked 2 GiB of host pages (5.7 GB
of shmem for the container); the direct `hipHostMalloc` is exactly sized.
And the MTP draft worker builds a second `ModelRunner`, whose hook logged
"Parked 1 parameters (1.18 GiB) model.embed_tokens.weight": the NextN
checkpoint carries no embedding, the draft's own `embed_tokens.weight` is
an uninitialised placeholder that `init_lm_head` deletes and replaces with
the target's Parameter a moment later, so the copy was 1.18 GiB of garbage
that then sat in a list on the draft model for the life of the server.
Draft runners are skipped now, and the storage-owned lifetime means the
same swap would free the buffer anyway.

`amdgpu_top` does not count these buffers under GTT usage (ROCm maps them
as userptr memory rather than GTT buffer objects), so the effect shows as
driver VRAM dropping and the scheduler's RSS/shmem rising by the same
amount. The launcher parks `embed_tokens.weight,visual.` by default
(`QWEN38_HOST_PARKED_PARAMS`, set empty to keep everything in VRAM) and
notes the ~2.1 GiB host RAM cost.

## Verification

`tools/test_qwen38_rocm.py` item 16: a small model with an embedding, a
two-layer "vision" stack and an lm_head; park the first two and check the
Parameters stay CUDA Parameters at the buffer address, forwards and a CUDA
graph replay equal the VRAM originals, lm_head is untouched, torch's
allocated VRAM drops by the moved bytes, every buffer is exactly its
parameter's size, a second module sharing the Parameter parks nothing,
replacing a parked Parameter frees its buffer, and an empty pattern list
moves nothing. Standalone at the real shapes (248,320 × 2,560 embedding
plus a 1,152 × 4,304 pair): 1.203 GiB moved, driver VRAM −1,233 MiB, host
RSS/shmem +1,232 MiB (exact size 1,231), second model sharing the embedding
moved 0 bytes with no host or VRAM change, replacing a parked weight
returned its 9 MiB, graph replay equal, gather from the parked table 4.2 µs
vs 3.8 µs from VRAM at bs 20 (3.4 vs 2.9 at bs 1).

End to end (DERISKED, MTP, cap 20, patches ≤ 27, launcher default): one
log line, "Parked 334 parameters (2.02 GiB)", nothing from the draft.
Driver VRAM after load 90,258 MiB used / 8,046 free (92,232 / 6,072
without parking); container shmem 2.4 GB (5.7 with the first version).
Greedy repeats identical with the same accept histograms as patch 26;
vision exact on the rendered test image (160 image tokens) with the tower
in host memory; bs 1 25.4 tok/s on the 183-token prompt, 19.0 / 22.0 /
22.6 at 3.4k / 13.5k / 53k, 10 / 20 streams 73.2 / 110.0 tok/s aggregate.
Worst case measured (18 streams plus two 43k-token prefills together): 3.3
GiB free at the bottom, was 1.6 before this patch; two 71k-token prefills
under the same 18 streams also bottomed at 3.3 GiB. GTT stayed at ~110 MiB
(the buffers are userptr mappings, not GTT buffer objects); host used
peaked at 8.9 GB with 22.8 GB available.
