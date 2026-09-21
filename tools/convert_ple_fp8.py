#!/usr/bin/env python3
"""Convert Qwen3.8-Flash-Next PLE n-gram tables from bf16 to fp8 (e4m3).

Reads the staged checkpoint, rewrites every
``*.ple.ple_embedding.ngram_embedding.shard_*.weight`` as float8_e4m3fn with
one shared per-table ``weight_scale`` (bf16, amax/448) added next to it, sets
``text_config.ple_embedding_dtype="float8_e4m3fn"`` in config.json so the
SGLang loader keeps the table fp8 on the host, and hardlinks every untouched
file so the rest of the checkpoint round-trips byte-identically without
duplicating ~80 GiB.

The loader contract (qwen4_exp.py): fp8 tables gather raw fp8->bf16 (no scale);
the consumer multiplies by the ``weight_scale`` buffer, which
``_load_qwen4_exp_ple_buffer`` fills from ``<module>.ple.ple_embedding.ngram_embedding.weight_scale``
(shape must match the registered [1] bf16 buffer, so the scale is per-table).

Two streaming passes: pass 1 walks the PLE shards to compute each table's
amax; pass 2 rewrites the affected files and hardlinks the rest. Peak RAM is
roughly one safetensors file.

Usage:
  convert_ple_fp8.py <src_checkpoint_dir> <dst_dir>
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0
SHARD_RE = re.compile(r"^(.*)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
INDEX_NAME = "model.safetensors.index.json"
# rows per conversion chunk; bounds the transient fp32 copy to ~a few hundred MiB
CHUNK_ROWS = 262144


def link_or_copy(src: Path, dst: Path) -> str:
    """Hardlink when on the same filesystem, otherwise stream-copy."""
    try:
        os.link(src, dst)
        return "link"
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return "copy"


def chunked_amax(t: torch.Tensor) -> float:
    m = 0.0
    for i in range(0, t.shape[0], CHUNK_ROWS):
        m = max(m, t[i : i + CHUNK_ROWS].float().abs().max().item())
    return m


def to_fp8(t: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty(t.shape, dtype=torch.float8_e4m3fn)
    for i in range(0, t.shape[0], CHUNK_ROWS):
        out[i : i + CHUNK_ROWS] = (
            t[i : i + CHUNK_ROWS].float().div(scale).clamp(-FP8_MAX, FP8_MAX)
            .to(torch.float8_e4m3fn)
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    src: Path = args.src
    dst: Path = args.dst

    files = sorted(src.glob("model-*.safetensors"))
    if not files:
        print(f"no model-*.safetensors under {src}", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: per-table amax ------------------------------------
    amax: dict[str, float] = {}
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for name in sf.keys():
                m = SHARD_RE.match(name)
                if not m:
                    continue
                table = sf.get_tensor(name)
                local_max = chunked_amax(table)
                pref = m.group(1)
                amax[pref] = max(amax.get(pref, 0.0), local_max)
                del table
        print(f"[amax] {f.name}: tables so far {len(amax)}", flush=True)

    scales = {
        pref: torch.tensor([a / FP8_MAX], dtype=torch.bfloat16)
        for pref, a in amax.items()
    }
    for pref, s in scales.items():
        print(f"[scale] {pref}.ple.ple_embedding.ngram_embedding.weight_scale = {s.item():.6g}", flush=True)

    # ---- pass 2: rewrite / hardlink ---------------------------------
    index_path = src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None

    written_scales: set[str] = set()
    scale_file: dict[str, str] = {}
    for f in files:
        target = dst / f.name
        with safe_open(f, framework="pt") as sf:
            keys = list(sf.keys())
            shard_names = [k for k in keys if SHARD_RE.match(k)]
            if not shard_names:
                # pure non-PLE file: byte-identical (hardlink if same FS)
                if target.exists():
                    print(f"[skip] {f.name} (exists)", flush=True)
                else:
                    how = link_or_copy(f, target)
                    print(f"[{how}] {f.name}", flush=True)
                continue
            # the first file carrying a shard of each table also carries
            # that table's weight_scale; track it even when skipping so the
            # index stays right on a resumed run
            for name in shard_names:
                pref = SHARD_RE.match(name).group(1)
                if pref not in written_scales:
                    written_scales.add(pref)
                    scale_file[pref] = f.name
            if target.exists():
                print(f"[skip] {f.name} (exists)", flush=True)
                continue
            tensors = {}
            for name in keys:
                t = sf.get_tensor(name)
                m = SHARD_RE.match(name)
                if m:
                    pref = m.group(1)
                    tensors[name] = to_fp8(t, scales[pref].item())
                    if scale_file[pref] == f.name:
                        tensors[f"{pref}.ple.ple_embedding.ngram_embedding.weight_scale"] = scales[pref]
                else:
                    tensors[name] = t
                del t
            # write via temp name so a crash never leaves a truncated file
            # that a resumed run would then skip
            tmp = target.with_name(target.name + ".tmp")
            save_file(tensors, str(tmp), metadata={"format": "pt"})
            os.replace(tmp, target)
            del tensors
            print(f"[fp8 ] {f.name} ({len(shard_names)} shards)", flush=True)

    # ---- copy aux files, patch config.json --------------------------
    for extra in src.glob("*"):
        if (
            extra.suffix == ".safetensors"
            or extra.name.startswith(".")
            or extra.name == INDEX_NAME  # rewritten below, never linked
            or extra.is_dir()
        ):
            continue
        target = dst / extra.name
        if target.exists():
            continue
        if extra.name == "config.json":
            cfg = json.loads(extra.read_text())
            text_cfg = cfg.setdefault("text_config", {})
            text_cfg["ple_embedding_dtype"] = "float8_e4m3fn"
            target.write_text(json.dumps(cfg, indent=2))
            print("[cfg ] config.json: text_config.ple_embedding_dtype set")
        else:
            link_or_copy(extra, target)

    # ---- update the safetensors index with the new weight_scale rows -
    if index is not None:
        wmap = index.get("weight_map", {})
        for name in sorted(written_scales):
            key = f"{name}.ple.ple_embedding.ngram_embedding.weight_scale"
            if key in wmap:
                continue
            wmap[key] = scale_file[name]
        (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        print("[idx ] index updated with weight_scale entries")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
