"""gfx1151 patch 43: budget the KV pool from the VRAM carve, not Linux RAM.

On this host (Strix Halo, gfx1151 / Radeon 8060S) the BIOS carves 96 GiB of
DRAM for the iGPU and leaves 32 GB for Linux. Device allocations (model
weights, KV pool, CUDA graphs) live in the carve and never appear in Linux
page accounting, but SGLang's get_available_gpu_memory() takes the
``is_integrated`` branch for this device and returns psutil's MemAvailable —
a number from the 32 GB Linux domain that says nothing about device capacity.

Upstream's assumption ("integrated = sysmem as device mem") does not hold
when a carve is present. Verified on this host (2026-11-21):

  /sys/class/drm/card0/device/mem_info_vram_total = 96 GiB  (MemTotal 30.2 GiB)
  /sys/class/drm/card0/device/mem_info_vram_used  = 88.3 GiB with the server
      loaded — sysfs tracks exactly the weights + KV pool footprint.
  torch.cuda.mem_get_info() inside the container reports the 8 GiB GTT
      overflow window (15.49 total / 7.56 free observed), NOT the carve.

Consequence of the old accounting: the KV budget moved with Linux page-cache
noise at the instant of measurement. The patch-42 mmap load path measured
~3 GiB higher than the O_DIRECT staged path purely because the freshly read
dump files sat in page cache (clean, reclaimable) — max_total_num_tokens
477k vs 270k for identical device footprints.

New rule, matching the host's design intent:
  - If sysfs reports a dedicated carve (vram_total > 1.5x system RAM),
    budget from the carve's own free space (vram_total - vram_used), minus
    a small reserve for transient device usage (graph capture, activation
    scratch). Unusual spikes may still overflow into the 8 GiB GTT window
    by design; steady-state sizing must not plan to borrow from it.
  - Without a carve (true UMA), keep upstream's psutil fallback.

The carve number also accounts for other device tenants automatically
(e.g. the SDXL model under sdcpp raises vram_used and shrinks the budget).
"""

import argparse
import os
import sys

HELPER = '''_GFX1151_CARVE_RESERVE_GIB = 2.0


def _gfx1151_carve_free_gpu_memory(gpu_id: int):
    """gfx1151 patch 43: free space of the dedicated VRAM carve, in bytes.

    Strix Halo's BIOS carve leaves the iGPU with its own DRAM domain
    (vram_total 96 GiB vs a 32 GB Linux side). Device allocations live in
    the carve; torch.cuda.mem_get_info() inside a container reports the 8 GiB
    GTT overflow window instead, and Linux MemAvailable is yet another
    domain. When sysfs shows a carve larger than system RAM, this returns
    carve_free minus a small reserve for transient device usage (graph
    capture, activation scratch); by design, unusual spikes may overflow
    into the GTT window. Returns None when no carve is detected (plain
    UMA), leaving upstream's psutil-based fallback in charge.
    """
    try:
        base = f"/sys/class/drm/card{gpu_id}/device"
        with open(os.path.join(base, "mem_info_vram_total")) as f:
            vram_total = int(f.read())
        with open(os.path.join(base, "mem_info_vram_used")) as f:
            vram_used = int(f.read())
    except (OSError, ValueError):
        return None
    if vram_total <= psutil.virtual_memory().total * 1.5:
        return None  # no dedicated carve: true UMA, keep the psutil metric
    reserve = int(_GFX1151_CARVE_RESERVE_GIB * (1 << 30))
    return max(vram_total - vram_used - reserve, 0)


def get_available_gpu_memory('''

OLD_BRANCH = '''        props = torch.cuda.get_device_properties(gpu_id)
        if props.is_integrated:
            # On these devices, which use sysmem as device mem, torch.cuda.mem_get_info()
            # only reports "free" memory, which can be lower than what is actually
            # available due to not including cache memory. So we use the system available
            # memory metric instead.
            free_gpu_memory = psutil.virtual_memory().available'''

NEW_BRANCH = '''        props = torch.cuda.get_device_properties(gpu_id)
        if props.is_integrated:
            # On these devices, which use sysmem as device mem, torch.cuda.mem_get_info()
            # only reports "free" memory, which can be lower than what is actually
            # available due to not including cache memory. So we use the system available
            # memory metric instead.
            # gfx1151 patch 43: that assumption breaks when the firmware
            # carves dedicated VRAM for the iGPU (Strix Halo: 96 GiB carve,
            # 32 GB Linux). Device allocations live in the carve, invisible
            # to Linux accounting — budget from the carve itself and keep
            # the psutil metric only for true-UMA devices.
            carve_free = _gfx1151_carve_free_gpu_memory(gpu_id)
            if carve_free is not None:
                free_gpu_memory = carve_free
            else:
                free_gpu_memory = psutil.virtual_memory().available'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=".")
    args = parser.parse_args()
    path = os.path.abspath(os.path.expanduser(args.path))

    p = f"{path}/python/sglang/srt/utils/common.py"
    text = open(p).read()

    anchor = "def get_available_gpu_memory("
    assert text.count(anchor) == 1, "get_available_gpu_memory definition not found"
    text = text.replace(anchor, HELPER)

    assert text.count(OLD_BRANCH) == 1, "integrated-memory branch anchor not found"
    text = text.replace(OLD_BRANCH, NEW_BRANCH)

    open(p, "w").write(text)
    print("patch_umem_carve_budget: ok")


if __name__ == "__main__":
    main()
