#!/usr/bin/env python3
"""gfx1151 patch 27: park selected weights in pinned system memory (GTT).

On Strix Halo the 96 GiB VRAM carve-out and system RAM are the same LPDDR5X,
but GPU reads of system memory go through the IOMMU (kept on for the NPU)
and are measurably slower, so only weights with near-zero bandwidth demand
belong there: the token embedding (a `bs`-row gather per step, 1.18 GiB
for this vocabulary) and the vision tower (idle unless an image arrives,
0.9 GiB). With MTP at the 20-request cap the server idles at 90.1 of 96 GiB
and the worst mixed load measured left 1.5 GiB free; this returns ~2.1 GiB
of it.

Mechanism, no compiled code: `hipHostMalloc(hipHostMallocMapped)` called
through ctypes on the HIP runtime torch already loaded gives an exactly
sized host allocation mapped for the device (torch's own pinned allocator
rounds to powers of two, which turned the 1.18 GiB embedding into 2 GiB of
locked pages), and `torch.as_tensor` accepts any object exposing
`__cuda_array_interface__`, aliasing the device pointer as a CUDA tensor
with no copy. After `load_model` the parameters whose names contain one of
the `SGLANG_HOST_PARKED_PARAMS` patterns are copied into such buffers, the
Parameter's `.data` is repointed at the alias (every reference to the
Parameter sees it, `weight_loader` and the like stay attached), and
`empty_cache` releases the VRAM. torch holds the producer object for as
long as the aliasing storage lives, so the buffer's lifetime is the
tensor's: a parked Parameter a model later replaces frees its host memory
by itself. Draft runners are skipped: the MTP checkpoint carries no
embedding, the draft's own `embed_tokens.weight` is a placeholder that
`init_lm_head` swaps for the target's (already parked) Parameter, and
parking it first copied 1.18 GiB of uninitialised memory to the host and
left it there. Parameters already parked (shared between models) are
recognised by device address and skipped. Unset or empty, the patch is a
no-op.

`amdgpu_top` does not count these buffers under GTT usage (ROCm maps them
as userptr memory), so the effect shows as VRAM dropping and the
scheduler's RSS rising by the same amount; the launcher notes the host RAM
cost. Anchors on upstream's `load_model`.
"""
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

helper_path = f"{path}/python/sglang/srt/model_executor/host_parked_params.py"
helper = '''"""gfx1151 patch 27: park selected parameters in pinned system memory."""

import ctypes
import logging
import os
import weakref
from typing import Dict, Iterable, List, Optional

import torch

logger = logging.getLogger(__name__)

ENV = "SGLANG_HOST_PARKED_PARAMS"

_HIP_HOST_MALLOC_MAPPED = 0x2  # hip_runtime_api.h

_hip = None
# Device address -> live buffer, for every parked buffer in this process,
# so a Parameter shared between models is recognised instead of copied
# again. Weak references: the buffer is owned by the tensor storage that
# aliases it (torch holds the __cuda_array_interface__ producer until the
# storage dies), so a parked Parameter that a model later replaces frees
# its host memory on its own.
_PARKED: Dict[int, "weakref.ReferenceType[HostParkedBuffer]"] = {}


def _hip_runtime():
    """The HIP runtime torch already loaded, so the mapping lands in its context."""
    global _hip
    if _hip is None:
        torch.cuda.init()
        loaded = []
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and "libamdhip64.so" in parts[-1]:
                    loaded.append(parts[-1])
        lib = ctypes.CDLL(loaded[0] if loaded else "libamdhip64.so")
        lib.hipHostMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        lib.hipHostMalloc.restype = ctypes.c_int
        lib.hipHostFree.argtypes = [ctypes.c_void_p]
        lib.hipHostFree.restype = ctypes.c_int
        lib.hipHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        lib.hipHostGetDevicePointer.restype = ctypes.c_int
        _hip = lib
    return _hip


class HostParkedBuffer:
    """An exactly sized ``hipHostMalloc`` allocation mapped for the device.

    torch's pinned allocator rounds requests up to a power of two (a 1.18 GiB
    embedding would lock 2 GiB of host pages), so the runtime is called
    directly. Exposes ``__cuda_array_interface__`` so ``torch.as_tensor``
    aliases the mapping as a CUDA tensor without a copy.
    """

    def __init__(self, nbytes: int):
        hip = _hip_runtime()
        self.nbytes = nbytes
        self._host_ptr = None
        host = ctypes.c_void_p()
        err = hip.hipHostMalloc(ctypes.byref(host), max(nbytes, 1), _HIP_HOST_MALLOC_MAPPED)
        if err != 0:
            raise RuntimeError(f"hipHostMalloc({nbytes} bytes) failed: hipError_t {err}")
        self._host_ptr = host.value
        dev = ctypes.c_void_p()
        err = hip.hipHostGetDevicePointer(ctypes.byref(dev), self._host_ptr, 0)
        if err != 0:
            self.free()
            raise RuntimeError(f"hipHostGetDevicePointer failed: hipError_t {err}")
        self._device_ptr = dev.value
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (self._device_ptr, False),
            "strides": None,
            "version": 2,
        }

    def data_ptr(self) -> int:
        return self._device_ptr

    def free(self) -> None:
        if self._host_ptr is not None:
            _PARKED.pop(getattr(self, "_device_ptr", None), None)
            _hip_runtime().hipHostFree(self._host_ptr)
            self._host_ptr = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


def host_parked_patterns(env: Optional[str] = None) -> List[str]:
    raw = os.environ.get(ENV, "") if env is None else env
    return [p.strip() for p in raw.split(",") if p.strip()]


def parked_buffer(param: torch.Tensor) -> Optional[HostParkedBuffer]:
    """The host buffer ``param`` lives in, or None if it is in VRAM."""
    ref = _PARKED.get(param.data.data_ptr())
    return ref() if ref is not None else None


def is_parked(param: torch.Tensor) -> bool:
    return parked_buffer(param) is not None


def park_parameter_in_host_memory(param: torch.nn.Parameter) -> HostParkedBuffer:
    """Repoint ``param.data`` at a host-mapped copy of exactly its size.

    The returned buffer is owned by the new storage; it is freed when the
    last tensor aliasing it goes away.
    """
    if not param.is_cuda:
        raise ValueError("only CUDA parameters can be parked")
    src = param.data.contiguous()
    buf = HostParkedBuffer(src.numel() * src.element_size())
    alias = torch.as_tensor(buf, device=src.device)
    if alias.data_ptr() != buf.data_ptr():
        raise RuntimeError("host-mapped alias does not share the buffer address")
    alias = alias.view(src.dtype).view(src.shape)
    alias.copy_(src)
    param.data = alias
    _PARKED[buf.data_ptr()] = weakref.ref(buf)
    return buf


def park_model_parameters(
    model: torch.nn.Module, patterns: Iterable[str]
) -> int:
    """Park every parameter whose name contains one of ``patterns``.

    Returns the number of bytes moved. Parameters already parked (shared
    with another model in this process) are skipped.
    """
    patterns = list(patterns)
    if not patterns:
        return 0
    moved = 0
    names = []
    shared = []
    for name, param in model.named_parameters():
        if not any(p in name for p in patterns) or not param.is_cuda:
            continue
        if is_parked(param):
            shared.append(name)
            continue
        park_parameter_in_host_memory(param)
        moved += param.numel() * param.element_size()
        names.append(name)
    if shared:
        logger.info(
            "%d parameters already parked by another model in this process: %s",
            len(shared),
            ", ".join(shared[:3]),
        )
    if moved:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        shown = names[:3] + ([f"... {len(names) - 3} more"] if len(names) > 3 else [])
        logger.info(
            "Parked %d parameters (%.2f GiB) in pinned system memory: %s",
            len(names),
            moved / 2**30,
            ", ".join(shown),
        )
    return moved
'''
open(helper_path, "w").write(helper)

p = f"{path}/python/sglang/srt/model_executor/model_runner.py"
text = open(p).read()
old = """        self.loader = loaded.loader
        self.model = loaded.model
        self.startup_weight_load = loaded.startup_weight_load
"""
assert text.count(old) == 1, "model_runner.py: load_model result anchor not found"
new = """        self.loader = loaded.loader
        self.model = loaded.model
        self.startup_weight_load = loaded.startup_weight_load
        if self.device == "cuda" and not self.is_draft_worker:
            # gfx1151 patch 27: weights with no bandwidth demand (token
            # embedding, vision tower) can live in pinned system memory.
            # Not for draft runners: an MTP/EAGLE draft's own embedding is
            # a placeholder that init_lm_head swaps for the target's.
            from sglang.srt.model_executor.host_parked_params import (
                host_parked_patterns,
                park_model_parameters,
            )

            park_model_parameters(self.model, host_parked_patterns())
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("wrote", helper_path)
print("patched", p)
print("patch 27 (host parked params) applied")
