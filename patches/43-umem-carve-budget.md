# Patch 43: budget the KV pool from the VRAM carve, not Linux RAM

**File:** `python/sglang/srt/utils/common.py`
**Script:** [`patch_umem_carve_budget.py`](patch_umem_carve_budget.py)
**Test:** [`../tests/check_umem_carve_budget.py`](../tests/check_umem_carve_budget.py)

## The memory layout (measured, not assumed)

This host (Strix Halo, gfx1151 / Radeon 8060S) is configured with a fixed
BIOS carve:

* `/sys/class/drm/card0/device/mem_info_vram_total` = **96 GiB** — the
  carve, dedicated to the iGPU.
* `/proc/meminfo` MemTotal = **30.2 GiB** — the Linux side.
* `mem_info_gtt_total` = **8 GiB** — not extra RAM: an *overflow window*.
  When the carve runs full (or a mapping explicitly needs it), up to 8 GiB
  of the Linux side can be borrowed. It exists so temporary startup spikes
  (CUDA graph capture) and not-always-resident services (the SDXL model
  under sdcpp) don't force permanent Linux-side headroom; steady-state
  sizing must not plan to borrow from it.
* Device allocations live **entirely in the carve** and never appear in
  Linux page accounting: `mem_info_vram_used` tracked the loaded model
  (88.3 GiB = weights + draft + KV pool) while Linux anon/shmem stayed
  flat through the load.

## What upstream assumes, and why it breaks here

`get_available_gpu_memory()` takes the `is_integrated` branch for this
device and returns **psutil MemAvailable** — a number from the 32 GB
Linux domain. The comment says "integrated = sysmem as device mem"; that
does not hold when a firmware carve exists. Two failure modes follow:

1. **The KV budget is keyed to the wrong domain.** The pool's bytes come
   out of the carve (which has GiB free at boot), but its *size* was
   computed from Linux-side `MemAvailable` — which says nothing about
   device capacity and everything about page-cache noise: the
   patch-42 mmap load path measured ~3 GiB higher than the O_DIRECT
   staged path purely because the freshly-read dump files sat in clean,
   reclaimable page cache at the instant of measurement.
   `max_total_num_tokens`: 477k (mmap) vs 270-304k (staged) for
   **identical device footprints**.
2. **`torch.cuda.mem_get_info()` is no use either** inside the container:
   it reports the GTT window (15.49 GiB total / 7.56 free observed), not
   the carve.

## Fix

In the `is_integrated` branch, before falling back to psutil:

* Read `/sys/class/drm/card<id>/device/mem_info_vram_{total,used}`.
* If `vram_total > 1.5 × system RAM`, a dedicated carve exists: return
  `vram_total - vram_used - 2 GiB reserve` (bytes). The reserve covers
  transient device usage (graph capture, activation scratch); unusual
  spikes may still overflow into the GTT window by design.
* Otherwise (true UMA, no carve), keep upstream's psutil fallback
  unchanged.

The carve number automatically accounts for other device tenants: the
SDXL model under sdcpp raises `vram_used` and shrinks the budget.

## Deployment consequence

`--mem-fraction-static` now operates on the carve. The weights + draft
are 86.7% of the carve-free-at-boot, so the old default 0.85 fails the
configurator's sanity check ("Loaded weights leave no GPU memory for the
KV cache") — the check was vacuous under the old metric. The deployment
sets `SGLANG_MEM_FRAC=0.985`, which keeps the same effective runtime
slack (~1.4 GiB of carve) the old accounting happened to provide.
`QWEN38_MAX_TOTAL_TOKENS=524288` becomes the binding cap: the profiled
capacity (~570k) now exceeds it, and the configurator takes the minimum.

Result: `max_total_num_tokens` **509504** — above the 477k the
psutil-based accounting used to produce, and stable across load paths
because it no longer depends on page-cache state at the moment of
measurement.

## Verification

* `tests/check_umem_carve_budget.py` (no model work): a nonexistent card
  must return None (fallback branch); on this host the helper must agree
  with the sysfs-derived value to the byte and the patched branch must
  consult it before the psutil fallback.
* End-to-end: staged-path boot with the patched accounting shows
  `max_total_num_tokens=509504` and a healthy warmup; the pre-patch
  staged boots showed 270-304k on the same image lineage.
