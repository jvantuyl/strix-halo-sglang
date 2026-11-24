#!/usr/bin/env python3
"""Make partial presharded overlays for a QSA-only quantization comparison."""
import argparse
import ctypes
import hashlib
import json
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


PIN = "fb08d7cdb67082ddda15a25e64dc7ec413afc41c"
PATTERN = re.compile(r"model\.layers\.(\d+)\.qkv_proj\.weight")
LAYERS = set(range(3, 48, 4))
# TP=1 packing: gated Q (24 heads with Q and gate), then K and V (2 heads each).
QKV_ROWS = (("q", 12288), ("k", 512), ("v", 512))
INPUT_WIDTH = 2560
BLOCK_BYTES = {"q8": 33, "fp4": 17, "fp4-weighted": 17}


def load_imatrix(path):
    from gguf import GGUFReader, GGMLQuantizationType

    reader = GGUFReader(path)
    if reader.fields["general.type"].contents() != "imatrix":
        raise ValueError("Expected a GGUF importance matrix")
    tensors = {t.name: t for t in reader.tensors}
    stats, weights = {}, {}
    for layer in sorted(LAYERS):
        weights[layer] = {}
        for proj, _ in QKV_ROWS:
            name = f"blk.{layer}.attn_{proj}.weight"
            sums = tensors[name + ".in_sum2"]
            counts = tensors[name + ".counts"]
            if (sums.tensor_type != GGMLQuantizationType.F32
                    or counts.tensor_type != GGMLQuantizationType.F32
                    or tuple(sums.shape) not in ((INPUT_WIDTH,), (INPUT_WIDTH, 1))
                    or tuple(counts.shape) not in ((1,), (1, 1))):
                raise ValueError(f"Unexpected importance tensor shape or type: {name}")
            count = counts.data.item()
            if not np.isfinite(count) or count <= 0:
                raise ValueError(f"No calibration samples: {name}")
            # Match the GGUF quantize loader's mean of squared activations.
            vector = np.array(sums.data.reshape(-1) / np.float32(count), dtype=np.float32)
            if not np.isfinite(vector).all() or (vector < 0).any() or not vector.any():
                raise ValueError(f"Invalid activation statistics: {name}")
            weights[layer][proj] = vector
            stats[name] = dict(count=count, min=float(vector.min()), max=float(vector.max()))
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    metadata = {k: v.contents() for k, v in reader.fields.items() if k.startswith("imatrix.")}
    return weights, dict(sha256=digest, metadata=metadata, tensors=stats)


def reference(upstream, out):
    sha = subprocess.check_output([
        "git", "-c", f"safe.directory={upstream}", "-C", str(upstream), "rev-parse", "HEAD"
    ], text=True)
    if sha.strip() != PIN:
        raise ValueError(f"Expected ROCmFPX {PIN}, got {sha.strip()}")
    src = out / "reference.c"
    src.write_text("""
#include <stdint.h>
#include "rocmfp4.c"
#include "rocmfpx.h"
void roundtrip_q8(const float *x, void *q, float *y, int64_t k) {
    rocmfpx_quantize_row_fp8_ref(x, q, k);
    rocmfpx_dequantize_row_fp8(q, y, k);
}
void roundtrip_fp4(const float *x, void *q, float *y, int64_t k) {
    rocmfp4_quantize_row_q4_0_fast_ref(x, q, k);
    rocmfp4_dequantize_row_q4_0_fast(q, y, k);
}
void roundtrip_fp4_weighted(const float *x, block_rocmfp4_fast *q, float *y,
                            int64_t rows, int64_t k, const float *weights) {
    for (int64_t row = 0; row < rows; ++row) {
        rocmfp4_quantize_row_q4_0_fast_weighted(x + row*k, q + row*(k/QK_ROCMFP4), k, weights);
    }
    rocmfp4_dequantize_row_q4_0_fast(q, y, rows*k);
}
""")
    exports = out / "reference.exports"
    exports.write_text("{ global: roundtrip_q8; roundtrip_fp4; roundtrip_fp4_weighted; local: *; };\n")
    lib = out / "reference.so"
    ggml = upstream / "ggml"
    subprocess.run([
        "cc", "-O3", "-ffp-contract=off", "-fPIC", "-shared", "-ffunction-sections",
        f"-I{ggml / 'include'}", f"-I{ggml / 'rocmfp4'}", f"-I{ggml / 'rocmfpx'}",
        str(src), str(ggml / "rocmfpx/rocmfpx.c"),
        "-Wl,--gc-sections", "-Wl,-z,defs", f"-Wl,--version-script={exports}", "-lm",
        "-o", str(lib)
    ], check=True)
    result = ctypes.CDLL(str(lib))
    for variant in ("q8", "fp4"):
        fn = getattr(result, f"roundtrip_{variant}")
        fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64]
        fn.restype = None
    result.roundtrip_fp4_weighted.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2 + [ctypes.c_void_p]
    result.roundtrip_fp4_weighted.restype = None
    return result


def roundtrip(weights, lib, variant, importance=None):
    assert weights.dtype == torch.bfloat16 and weights.is_contiguous() and weights.shape[1] % 32 == 0
    x = weights.float().numpy()
    if not np.isfinite(x).all():
        raise ValueError("QKV has non-finite weight values")
    y = np.empty_like(x)
    block_bytes = BLOCK_BYTES[variant]
    blob = np.empty(x.size // 32 * block_bytes, dtype=np.uint8)
    if importance is None:
        getattr(lib, f"roundtrip_{variant}")(x.ctypes.data, blob.ctypes.data, y.ctypes.data, x.size)
    else:
        start = 0
        k = x.shape[1]
        for proj, rows in QKV_ROWS:
            lib.roundtrip_fp4_weighted(
                x[start:].ctypes.data, blob[start * (k // 32) * block_bytes:].ctypes.data,
                y[start:].ctypes.data, rows, k, importance[proj].ctypes.data,
            )
            start += rows
        assert start == x.shape[0]
    if not np.isfinite(y).all():
        raise ValueError("Reference dequant produced non-finite weights")
    w = torch.from_numpy(y).bfloat16()
    rel_rms = (w.float() - weights.float()).norm() / weights.float().norm()
    return w.view(torch.uint16).numpy().tobytes(), rel_rms.item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path)
    ap.add_argument("upstream", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--imatrix", type=Path, help="Build only the weighted FP4 variant using this GGUF")
    args = ap.parse_args()
    variants = ("q8", "fp4") if args.imatrix is None else ("fp4-weighted",)
    importance, calibration = ({}, None) if args.imatrix is None else load_imatrix(args.imatrix)
    torch.set_num_threads(4)
    args.output.mkdir()
    lib = reference(args.upstream, args.output)
    plan = json.loads((args.dump / "checksum.json").read_text())
    assert plan["world_size"] == 1
    selected = {}
    for row in plan["rank_to_reads"]["0"]:
        match = PATTERN.fullmatch(row["name"])
        if match:
            layer = int(match[1])
            assert layer not in selected and not row["is_extra"]
            selected[layer] = row
    assert set(selected) == LAYERS, selected
    files = sorted({row["filename"] for row in selected.values()})
    records = []
    for variant in variants:
        dest = args.output / variant
        dest.mkdir()
        for file in files:
            shutil.copyfile(args.dump / file, dest / file)
        for layer, row in sorted(selected.items()):
            file, key = row["filename"], row["stored_key"]
            source = args.dump / file
            with source.open("rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            info = header[key]
            assert info["dtype"] == "BF16" and info["shape"] == [sum(n for _, n in QKV_ROWS), INPUT_WIDTH], info
            start, end = info["data_offsets"]
            with safe_open(source, framework="pt") as f:
                w = f.get_tensor(key)
                old = w.view(torch.uint16).numpy().tobytes()
                data, rel_rms = roundtrip(w, lib, variant, importance.get(layer))
            assert len(data) == end - start and data != old
            with (dest / file).open("r+b") as f:
                f.seek(8 + header_len + start)
                assert f.write(data) == len(data)
            records.append(dict(variant=variant, name=row["name"], file=file, key=key,
                                start=8+header_len+start, size=len(data), rel_rms=rel_rms,
                                sha256=hashlib.sha256(data).hexdigest()))
            print(variant, row["name"], f"relative RMS {rel_rms:.5f}", flush=True)
    # Re-read the overlays. Every byte outside the 12 intended ranges must
    # equal the original dump, including headers and all unselected tensors.
    for variant in variants:
        for file in files:
            with (args.dump / file).open("rb") as src, (args.output / variant / file).open("rb") as dst:
                spans = sorted((r["start"], r) for r in records
                               if r["variant"] == variant and r["file"] == file)
                for start, record in spans:
                    while src.tell() < start:
                        size = min(start - src.tell(), 8 << 20)
                        assert src.read(size) == dst.read(size)
                    assert hashlib.sha256(dst.read(record["size"])).hexdigest() == record["sha256"]
                    src.seek(record["size"], 1)
                while chunk := src.read(8 << 20):
                    assert chunk == dst.read(len(chunk))
                assert not dst.read(1)
    manifest = dict(pin=PIN, files=files, records=records)
    if calibration is not None:
        manifest["calibration"] = calibration
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Verified 12 QSA weights per variant; all other bytes identical across {len(files)} files")


if __name__ == "__main__":
    main()
