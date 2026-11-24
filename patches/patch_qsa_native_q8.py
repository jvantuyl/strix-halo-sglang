#!/usr/bin/env python3
"""Patch 54: opt-in native ROCmFPX Q8 for the target's 12 QSA QKV linears."""

import sys
from pathlib import Path

RUNTIME = '''
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open

LAYERS = set(range(3, 48, 4))
N, K = 13312, 2560
PIN = "fb08d7cdb67082ddda15a25e64dc7ec413afc41c"
QSA = re.compile(r"model[.]layers[.](\\d+)[.]qkv_proj")


@triton.jit
def ue4m3(e):
    mant = e & 7
    exp = e >> 3
    frac = tl.where(exp == 0, mant, mant + 8).to(tl.float32)
    power = ((tl.maximum(exp, 1) + 116) << 23).to(tl.float32, bitcast=True)
    return tl.where(e <= 126, frac * power, 0.0)


@triton.jit
def unpack(Q, E, W, SIZE: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    q = tl.load(Q + i, i < SIZE, 0).to(tl.float32)
    e = tl.load(E + i // 32, i < SIZE, 0).to(tl.int32)
    tl.store(W + i, q * ue4m3(e), i < SIZE)


@triton.jit
def dense(X, Q, E, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          BN: tl.constexpr, SPLIT: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    part = tl.program_id(1)
    rm = tl.arange(0, 16)
    rk = tl.arange(0, 64)
    acc = tl.full((16, BN), 0, tl.float32)
    for i in range(tl.cdiv(K, 64 * SPLIT)):
        k = (i * SPLIT + part) * 64 + rk
        a = tl.load(X + rm[:, None] * K + k[None, :],
                    (rm[:, None] < M) & (k[None, :] < K), 0)
        q = tl.load(Q + rn[None, :] * K + k[:, None],
                    (rn[None, :] < N) & (k[:, None] < K), 0)
        e = tl.load(E + rn[None, :] * (K // 32) + k[:, None] // 32,
                    (rn[None, :] < N) & (k[:, None] < K), 0)
        b = (q.to(tl.float32) * ue4m3(e.to(tl.int32))).to(tl.bfloat16)
        acc = tl.dot(a, b, acc)
    tl.store(Y + (part * M + rm[:, None]) * N + rn[None, :], acc,
             (rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def merge(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    part = tl.arange(0, SPLIT)
    v = tl.load(P + part[:, None] * SIZE + i[None, :], i[None, :] < SIZE, 0)
    tl.store(Y + i, tl.sum(v, axis=0), i < SIZE)


class QSAQ8Linear(torch.nn.Module):
    def __init__(self, codes, scales):
        super().__init__()
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)

    def forward(self, x):
        x = x.contiguous()
        m = x.numel() // K
        shape = (*x.shape[:-1], N)
        if not m:
            return x.new_empty(shape), None
        if m > 16:
            # Preserve rocBLAS for large prefill and verification batches.
            # This allocation is temporary; no BF16 weights are retained.
            w = x.new_empty((N, K))
            unpack[(triton.cdiv(N*K, 1024),)](self.codes, self.scales, w, N*K, 1024)
            return F.linear(x, w), None
        bn, split, warps = (64, 8, 4) if m == 1 else (64, 1, 4) if m <= 4 else (32, 1, 2)
        y = x.new_empty(shape)
        partial = torch.empty((split, m, N), device=x.device, dtype=torch.float32) if split > 1 else y
        dense[(triton.cdiv(N, bn), split)](
            x, self.codes, self.scales, partial, m, N, K, bn, split, num_warps=warps,
        )
        if split > 1:
            merge[(triton.cdiv(m*N, 256),)](partial, y, m*N, split, 256)
        return y, None


def install_qsa_q8(model, tp_size):
    folder = os.environ.get("SGLANG_QSA_Q8_DIR")
    if not folder:
        return
    if (tp_size != 1 or not torch.version.hip
            or torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx1151"):
        raise ValueError("Native QSA Q8 requires gfx1151 and TP=1")
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["format"] != "qsa-q8-v1" or manifest["reference"] != PIN:
        raise ValueError("Unsupported native QSA Q8 sidecar")
    names = [name for name, _ in model.named_modules() if QSA.fullmatch(name)]
    if ({int(QSA.fullmatch(name)[1]) for name in names} != LAYERS
            or len(names) != len(LAYERS)
            or any(set(manifest[k]) != set(names) for k in ("source_sha256", "packed_sha256"))):
        raise ValueError("Native QSA Q8 sidecar does not cover exactly the 12 target projections")
    for name in names:
        mod = model.get_submodule(name)
        weight = mod.weight
        if (weight.dtype != torch.bfloat16 or tuple(weight.shape) != (N, K)
                or weight.device.type != "cuda" or mod.bias is not None):
            raise ValueError(f"Unexpected native QSA Q8 source projection: {name}")
        source = weight.detach().cpu().contiguous().view(torch.uint16).numpy()
        if hashlib.sha256(source).hexdigest() != manifest["source_sha256"][name]:
            raise ValueError(f"Native QSA Q8 source hash mismatch: {name}")
        with safe_open(folder / (name + ".safetensors"), framework="pt") as f:
            if set(f.keys()) != {"codes", "scales"}:
                raise ValueError(f"Invalid native QSA Q8 tensors: {name}")
            q, e = f.get_tensor("codes"), f.get_tensor("scales")
            if (q.dtype != torch.int8 or tuple(q.shape) != (N, K)
                    or e.dtype != torch.uint8 or tuple(e.shape) != (N, K//32)):
                raise ValueError(f"Invalid native QSA Q8 layout: {name}")
            for tensor, key in ((q, "codes"), (e, "scales")):
                if hashlib.sha256(tensor.numpy()).hexdigest() != manifest["packed_sha256"][name][key]:
                    raise ValueError(f"Native QSA Q8 packed hash mismatch: {name}.{key}")
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, QSAQ8Linear(q.to(weight.device), e.to(weight.device)))
    logging.getLogger(__name__).info("Native QSA Q8 installed on %d target projections", len(names))
'''


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang")
    directory = root / "python/sglang/srt/model_executor"
    runner = directory / "model_runner.py"
    text = runner.read_text()
    anchor = "        self.startup_weight_load = loaded.startup_weight_load\n"
    assert text.count(anchor) == 1, "QSA Q8 post-load anchor moved"
    assert "install_qsa_q8" not in text
    runner.write_text(text.replace(anchor, anchor + """\
        if not self.is_draft_worker:
            import os

            if os.environ.get("SGLANG_QSA_Q8_DIR"):
                from sglang.srt.model_executor.qsa_native_q8 import install_qsa_q8

                install_qsa_q8(self.model, self.ps.tp_size)
"""))
    (directory / "qsa_native_q8.py").write_text(RUNTIME.lstrip())
    print("patched", runner)


if __name__ == "__main__":
    main()
