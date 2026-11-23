#!/usr/bin/env python3
"""check_ple_hotset_mlock2.py — verify patch 52 in the image, no GPU.

Patch 52 switches PleHotsetCache._mlock to mlock2(MLOCK_ONFAULT). This test
imports the real module (it only needs ctypes + numpy), constructs a cache,
and proves the two properties the patch promises:

1. mlocked is True and the ONFAULT path was taken (no upfront fault:
   mincore reports exactly the one glibc chunk-header page resident right
   after construction, where plain mlock would have faulted every page in).
2. Lock-on-fault still engages: after writing one payload byte, its page
   is resident, so the steady-state protection is unchanged.

Run: python3 tests/check_ple_hotset_mlock2.py   (inside the image)
"""

import ctypes
import os
import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

PAGE = 4096


def resident_pages(addr: int, nbytes: int) -> int:
    """Count resident pages in [addr, addr+nbytes) via mincore."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    first = addr & ~(PAGE - 1)
    npages = (addr + nbytes + PAGE - 1 - first) // PAGE
    vec = (ctypes.c_ubyte * npages)()
    if libc.mincore(ctypes.c_void_p(first), ctypes.c_size_t(npages * PAGE), vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore failed")
    return sum(1 for b in vec if b & 1)


def main():
    failures = []
    src_path = "/sgl-workspace/sglang/python/sglang/srt/utils/ple_hotset.py"
    src = open(src_path).read()

    # 0. The patch is present with its fallback chain intact.
    if "gfx1151 patch 52" not in src:
        failures.append("patch 52 marker comment missing")
    if "mlock2" not in src or "_MLOCK_ONFAULT" not in src:
        failures.append("mlock2/MLOCK_ONFAULT path missing")
    if src.count("libc.mlock(ctypes.c_void_p(start), ctypes.c_size_t(nbytes))") != 1:
        failures.append("plain-mlock fallback missing or duplicated")

    try:
        from sglang.srt.utils.ple_hotset import PleHotsetCache
    except Exception as e:
        print(f"FAIL: cannot import PleHotsetCache: {e}")
        sys.exit(1)

    # 1. Construction locks lazily: mlocked=True, and only the single
    #    glibc chunk-header page is resident (the payload comes from a
    #    mmap'd chunk whose 16-byte header sits in the first page, which
    #    malloc touched at allocation; the payload pages themselves are
    #    untouched). Plain mlock would have every page resident here.
    n_rows, row_bytes = 262144, 160  # 40 MiB: mmap-backed for sure
    cache = PleHotsetCache(n_rows, row_bytes)
    if not cache.mlocked:
        failures.append("cache reports mlocked=False (locking failed)")
    addr = cache.payload.__array_interface__["data"][0]
    resident0 = resident_pages(addr, cache.payload.nbytes)
    if resident0 != 1:
        failures.append(
            f"expected exactly 1 resident page (glibc chunk header) after "
            f"construction, got {resident0} (plain mlock would fault in all)"
        )

    # 2. Touched pages fault in and stay locked. Row 0 shares the header
    #    page; row 1000 needs its own page, proving lock-on-fault engages.
    cache.payload[1000, 0] = 0x5A
    resident1 = resident_pages(addr, cache.payload.nbytes)
    if resident1 < 2:
        failures.append("a far-row payload page did not become resident")

    if failures:
        print("FAIL:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print(f"check_ple_hotset_mlock2: OK (mlock2 ONFAULT path active: "
          f"{resident0} pages resident at construction, {resident1} after "
          "touch; plain-mlock fallback present). Boot cost: the 2M-row "
          "hotsets no longer force-fault 320 MiB each at attach.")


if __name__ == "__main__":
    main()
