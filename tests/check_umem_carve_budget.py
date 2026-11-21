"""Checks for gfx1151 patch 43 (carve-aware device memory budget).

Run in the image against a patched tree:
    PYTHONPATH=<staged>/python python3 tests/check_umem_carve_budget.py

Needs no model work: only sysfs, psutil and the patched helper. The
no-carve branch is exercised through a nonexistent card (card7); the
carve branch is exercised against the real sysfs on this host.
"""

import os

assert (
    os.environ.get("PYTHONPATH", "").startswith("/staged/python")
    or "staged" in os.environ.get("PYTHONPATH", "")
), "run with PYTHONPATH pointing at the patched staged tree"


def _sysfs_gib(name, card=0):
    with open(f"/sys/class/drm/card{card}/device/{name}") as f:
        return int(f.read()) / 2**30


from sglang.srt.utils.common import _gfx1151_carve_free_gpu_memory

# A nonexistent card must fall back to None (no carve detected -> psutil path).
assert _gfx1151_carve_free_gpu_memory(7) is None, "missing card must return None"

# This host has the 96 GiB carve; the helper must see it and reserve 2 GiB.
vram_total = _sysfs_gib("mem_info_vram_total")
vram_used = _sysfs_gib("mem_info_vram_used")
assert vram_total > 1.5 * 30.0, f"expected a carve on this host, got {vram_total} GiB"

free = _gfx1151_carve_free_gpu_memory(0)
expected = (vram_total - vram_used) - 2.0
assert free is not None, "carve not detected on card0"
assert abs(free / 2**30 - expected) < 1e-6, (
    f"carve free {free / 2**30:.2f} GiB != sysfs-derived {expected:.2f} GiB"
)
assert free > 0, "carve free must be positive on the running host"

# The patched branch must consult the helper before the psutil fallback.
import sglang.srt.utils.common as common

src = open(common.__file__).read()
assert "_gfx1151_carve_free_gpu_memory(gpu_id)" in src, "branch not wired to helper"
assert "carve_free is not None" in src, "fallback wiring missing"

print(
    f"check_umem_carve_budget: ok "
    f"(carve {vram_total:.1f} GiB, used {vram_used:.1f}, budget free "
    f"{free / 2**30:.2f} GiB)"
)
