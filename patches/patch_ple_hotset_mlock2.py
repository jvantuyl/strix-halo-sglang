#!/usr/bin/env python3
"""gfx1151 patch 52: mlock2(MLOCK_ONFAULT) for the PLE hot-set payload.

Patch 46's PleHotsetCache locks its whole payload array with plain mlock
right after np.zeros/calloc. Plain mlock on untouched pages faults them all
in synchronously: at 2M rows that is 320 MiB per hotset, and two hotsets
are built per boot (target + draft), so the 2026-11-23 boot spent ~2 min
per attach in silent upfront faulting under this host's RAM pressure
(measured: 8 s / 0 s per attach at 1M rows, 2:00 / 1:48 at 2M; total boot
65 s -> 5m45s).

mlock2(..., MLOCK_ONFAULT) skips the upfront fault: pages become locked
as the cache touches them. Once touched, a page is unevictable exactly as
before, so the steady-state protection is identical; only the boot-time
force-fault disappears. Fallbacks preserve all earlier behavior: plain
mlock (old upfront-fault cost) where mlock2 is unavailable, then unlocked
operation on any failure, exactly like patch 46.

Edit site: sglang/srt/utils/ple_hotset.py, PleHotsetCache._mlock.

Run from the Dockerfile with the target tree as argv[1].
"""

import argparse
import os
import sys

TARGET = "python/sglang/srt/utils/ple_hotset.py"

OLD = """    def _mlock(self) -> None:
        try:
            addr = self.payload.__array_interface__["data"][0]
            page = 4096
            start = addr & ~(page - 1)
            end = (addr + self.payload.nbytes + page - 1) & ~(page - 1)
            if libc.mlock(ctypes.c_void_p(start), ctypes.c_size_t(end - start)) != 0:
                raise OSError(ctypes.get_errno(), "mlock failed")
            self.mlocked = True
        except Exception:
            self.mlocked = False  # cache works unlocked; pressure may evict
"""

NEW = """    # gfx1151 patch 52: lock with mlock2(MLOCK_ONFAULT). Plain mlock on a
    # fresh calloc'd array faults in every page synchronously at boot: at
    # 2M rows that is 320 MiB per hotset, twice per boot (target + draft),
    # measured ~2 min per attach under host-RAM pressure (2026-11-23; the
    # same phase was 8 s / 0 s at 1M rows). ONFAULT skips the upfront
    # fault; pages become unevictable only as the cache touches them, so
    # the steady state is identical and only boot cost disappears.
    _MLOCK_ONFAULT = 0x01  # <linux/mman.h>
    _SYS_MLOCK2_X86_64 = 325  # only used if glibc lacks the mlock2 symbol

    def _mlock(self) -> None:
        try:
            addr = self.payload.__array_interface__["data"][0]
            page = 4096
            start = addr & ~(page - 1)
            end = (addr + self.payload.nbytes + page - 1) & ~(page - 1)
            nbytes = end - start
            locked = False
            mlock2 = getattr(libc, "mlock2", None)
            if mlock2 is not None:
                try:
                    locked = (
                        mlock2(
                            ctypes.c_void_p(start),
                            ctypes.c_size_t(nbytes),
                            ctypes.c_uint(self._MLOCK_ONFAULT),
                        )
                        == 0
                    )
                except Exception:
                    locked = False
            if not locked:
                try:
                    locked = (
                        libc.syscall(
                            self._SYS_MLOCK2_X86_64,
                            ctypes.c_void_p(start),
                            ctypes.c_size_t(nbytes),
                            ctypes.c_uint(self._MLOCK_ONFAULT),
                        )
                        == 0
                    )
                except Exception:
                    locked = False
            if not locked:
                # Pre-4.4 kernel / glibc without mlock2: keep the old
                # upfront-fault behavior rather than serve unlocked.
                if libc.mlock(ctypes.c_void_p(start), ctypes.c_size_t(nbytes)) != 0:
                    raise OSError(ctypes.get_errno(), "mlock failed")
            self.mlocked = True
        except Exception:
            self.mlocked = False  # cache works unlocked; pressure may evict
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/sgl-workspace/sglang")
    args = ap.parse_args()
    path = os.path.join(args.target, TARGET)
    with open(path) as f:
        text = f.read()
    if "gfx1151 patch 52" in text:
        sys.exit("patch 52 already applied")
    assert text.count(OLD) == 1, "ple_hotset.py _mlock anchor not found"
    text = text.replace(OLD, NEW, 1)
    with open(path, "w") as f:
        f.write(text)
    print("patch 52 applied: hotset _mlock uses mlock2(MLOCK_ONFAULT) "
          "with plain-mlock fallback")


if __name__ == "__main__":
    main()
