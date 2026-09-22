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
amax (cached in ``<dst>/ple_amax.json`` so a resumed run skips it); pass 2
rewrites the affected files and hardlinks the rest. Peak RAM is roughly one
output file.

A source file whose converted contents would exceed ``--part-bytes`` (some
checkpoints pack the whole 100 GiB table plus the MTP head into one file) is
split: the PLE shards go to ``<stem>-pleNNN.safetensors`` parts holding
nothing else, so the loader's PLE-only-file skip (patch 13) still applies,
and every other tensor plus the ``weight_scale`` goes to
``<stem>-restNNN.safetensors``. The index is rewritten to match. Parts are
planned from the safetensors header, so a resumed run skips finished parts
without reading them.

Usage:
  convert_ple_fp8.py <src_checkpoint_dir> <dst_dir> [--part-bytes N]
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

FP8_MAX = 448.0
SHARD_RE = re.compile(r"^(.*)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
INDEX_NAME = "model.safetensors.index.json"
# rows per conversion chunk; bounds the transient fp32 copy to ~a few hundred MiB
CHUNK_ROWS = 262144
READ_CHUNK = 64 << 20

_ST_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "F64": torch.float64, "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16,
    "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


class SafeReader:
    """Minimal safetensors reader: header + positional reads, no mmap.

    ``safetensors.safe_open`` maps the whole file and fails with ENOMEM on
    the 100 GiB files some checkpoints ship, so tensors are read with plain
    seeks into a buffer. Tensor order is header (file) order.
    """

    def __init__(self, path: Path):
        self.f = open(path, "rb")
        (n,) = struct.unpack("<Q", self.f.read(8))
        header = json.loads(self.f.read(n))
        self.meta = {k: v for k, v in header.items() if k != "__metadata__"}
        self.base = 8 + n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.close()

    def keys(self) -> list[str]:
        return list(self.meta)

    def shape(self, name: str) -> list[int]:
        return self.meta[name]["shape"]

    def dtype(self, name: str) -> str:
        return self.meta[name]["dtype"]

    def get_tensor(self, name: str) -> torch.Tensor:
        m = self.meta[name]
        start, end = m["data_offsets"]
        buf = bytearray(end - start)
        view = memoryview(buf)
        self.f.seek(self.base + start)
        pos = 0
        while pos < len(buf):
            got = self.f.readinto(view[pos : pos + READ_CHUNK])
            if not got:
                raise EOFError(f"{name}: short read at {pos}/{len(buf)}")
            pos += got
        dtype = _ST_DTYPES[m["dtype"]]
        if not buf:
            return torch.empty(m["shape"], dtype=dtype)
        return torch.frombuffer(buf, dtype=dtype).reshape(m["shape"])


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


_DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "I32": 4,
    "I64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}


def _out_nbytes(sf, name: str) -> int:
    """Converted size of one tensor, from the header only (no data read)."""
    numel = 1
    for d in sf.shape(name):
        numel *= d
    if SHARD_RE.match(name):
        return numel  # fp8
    return numel * _DTYPE_BYTES[sf.dtype(name)]


def _pack(names: list[str], sizes: dict[str, int], limit: int) -> list[list[str]]:
    """Greedy first-fit in header order; a single oversized tensor gets its own part."""
    parts: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for n in names:
        if cur and cur_bytes + sizes[n] > limit:
            parts.append(cur)
            cur, cur_bytes = [], 0
        cur.append(n)
        cur_bytes += sizes[n]
    if cur:
        parts.append(cur)
    return parts


def plan_outputs(sf, f: Path, part_bytes: int, table_scale_owner: dict[str, str]):
    """Decide which output file(s) a source file becomes.

    Returns a list of (output_name, tensor_names, scale_prefixes). A file that
    fits in ``part_bytes`` keeps its name and carries any weight_scale it owns;
    an oversized file is split into PLE-only ``-pleNNN`` parts and ``-restNNN``
    parts for everything else (the scale rides in the first rest part so the
    PLE-only parts stay skippable).
    """
    keys = list(sf.keys())
    sizes = {k: _out_nbytes(sf, k) for k in keys}
    owned = [pref for pref, owner in table_scale_owner.items() if owner == f.name]
    if sum(sizes.values()) <= part_bytes:
        return [(f.name, keys, owned)]
    stem = f.name[: -len(".safetensors")]
    shard_names = [k for k in keys if SHARD_RE.match(k)]
    rest_names = [k for k in keys if not SHARD_RE.match(k)]
    plan = []
    for i, group in enumerate(_pack(shard_names, sizes, part_bytes)):
        plan.append((f"{stem}-ple{i:03d}.safetensors", group, []))
    rest_parts = _pack(rest_names, sizes, part_bytes) if rest_names else [[]]
    for i, group in enumerate(rest_parts):
        plan.append((f"{stem}-rest{i:03d}.safetensors", group, owned if i == 0 else []))
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument(
        "--part-bytes", type=int, default=4 << 30,
        help="split a converted file larger than this into parts (default 4 GiB)",
    )
    args = ap.parse_args()
    src: Path = args.src
    dst: Path = args.dst

    files = sorted(src.glob("model-*.safetensors"))
    if not files:
        print(f"no model-*.safetensors under {src}", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: per-table amax ------------------------------------
    amax_cache = dst / "ple_amax.json"
    if amax_cache.exists():
        amax = {k: float(v) for k, v in json.loads(amax_cache.read_text()).items()}
        print(f"[amax] loaded {len(amax)} table(s) from {amax_cache.name}", flush=True)
    else:
        amax: dict[str, float] = {}
        for f in files:
            with SafeReader(f) as sf:
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
        amax_cache.write_text(json.dumps(amax, indent=2))

    scales = {
        pref: torch.tensor([a / FP8_MAX], dtype=torch.bfloat16)
        for pref, a in amax.items()
    }
    for pref, s in scales.items():
        print(f"[scale] {pref}.ple.ple_embedding.ngram_embedding.weight_scale = {s.item():.6g}", flush=True)

    # the first source file (sorted order) carrying a shard of each table
    # owns that table's weight_scale
    table_scale_owner: dict[str, str] = {}
    for f in files:
        with SafeReader(f) as sf:
            for name in sf.keys():
                m = SHARD_RE.match(name)
                if m and m.group(1) not in table_scale_owner:
                    table_scale_owner[m.group(1)] = f.name

    # ---- pass 2: rewrite / hardlink ---------------------------------
    index_path = src / INDEX_NAME
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    # tensor name -> output file, for every tensor that did not keep its file
    relocated: dict[str, str] = {}
    scale_file: dict[str, str] = {}

    for f in files:
        with SafeReader(f) as sf:
            keys = list(sf.keys())
            if not any(SHARD_RE.match(k) for k in keys):
                # pure non-PLE file: byte-identical (hardlink if same FS)
                target = dst / f.name
                if target.exists():
                    print(f"[skip] {f.name} (exists)", flush=True)
                else:
                    how = link_or_copy(f, target)
                    print(f"[{how}] {f.name}", flush=True)
                continue

            plan = plan_outputs(sf, f, args.part_bytes, table_scale_owner)
            if len(plan) > 1:
                print(f"[split] {f.name} -> {len(plan)} parts", flush=True)
            for out_name, names, scale_prefs in plan:
                for pref in scale_prefs:
                    scale_file[pref] = out_name
                if out_name != f.name:
                    for n in names:
                        relocated[n] = out_name
                target = dst / out_name
                if target.exists():
                    print(f"[skip] {out_name} (exists)", flush=True)
                    continue
                tensors = {}
                n_shards = 0
                for name in names:
                    t = sf.get_tensor(name)
                    m = SHARD_RE.match(name)
                    if m:
                        tensors[name] = to_fp8(t, scales[m.group(1)].item())
                        n_shards += 1
                    else:
                        tensors[name] = t
                    del t
                for pref in scale_prefs:
                    tensors[f"{pref}.ple.ple_embedding.ngram_embedding.weight_scale"] = scales[pref]
                # write via temp name so a crash never leaves a truncated file
                # that a resumed run would then skip
                tmp = target.with_name(target.name + ".tmp")
                save_file(tensors, str(tmp), metadata={"format": "pt"})
                os.replace(tmp, target)
                del tensors
                print(f"[fp8 ] {out_name} ({n_shards} shards, {len(names) - n_shards} other)", flush=True)

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

    # ---- rewrite the safetensors index: split parts + weight_scale rows
    if index is not None:
        wmap = index.get("weight_map", {})
        for name, out_name in relocated.items():
            wmap[name] = out_name
        for pref, out_name in scale_file.items():
            wmap[f"{pref}.ple.ple_embedding.ngram_embedding.weight_scale"] = out_name
        (dst / INDEX_NAME).write_text(json.dumps(index, indent=2))
        print(
            f"[idx ] index written ({len(relocated)} relocated, "
            f"{len(scale_file)} weight_scale rows)"
        )

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
