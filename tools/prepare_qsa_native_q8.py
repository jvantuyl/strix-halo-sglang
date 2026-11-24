#!/usr/bin/env python3
"""Build native Q8 sidecars from the original TP1 BF16 presharded dump."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

from prepare_qsa_quality import INPUT_WIDTH, LAYERS, PATTERN, PIN, QKV_ROWS, reference


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path)
    ap.add_argument("reference_checkout", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--verify-overlay", type=Path, required=True,
                    help="Unweighted Q8 BF16 overlay from the quality gauntlet")
    args = ap.parse_args()
    torch.set_num_threads(2)
    dump = json.loads((args.dump / "checksum.json").read_text())
    rows = [r for r in dump["rank_to_reads"]["0"] if PATTERN.fullmatch(r["name"])]
    if (dump["world_size"] != 1 or len(rows) != len(LAYERS)
            or {int(PATTERN.fullmatch(r["name"])[1]) for r in rows} != LAYERS):
        raise ValueError("Expected all 12 TP1 QSA QKV weights")
    n, k = sum(size for _, size in QKV_ROWS), INPUT_WIDTH
    args.output.mkdir()
    lib = reference(args.reference_checkout, args.output)
    manifest = dict(format="qsa-q8-v1", reference=PIN, source_sha256={}, packed_sha256={})
    for row in rows:
        file, key = row["filename"], row["stored_key"]
        name = row["name"].removesuffix(".weight")
        with safe_open(args.dump / file, framework="pt") as f:
            w = f.get_tensor(key)
            if row["is_extra"] or w.shape != (n, k) or w.dtype != torch.bfloat16:
                raise ValueError(f"Unexpected source projection: {name}")
            manifest["source_sha256"][name] = hashlib.sha256(w.view(torch.uint16).numpy()).hexdigest()
            x = w.float().numpy()
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite source projection: {name}")
        block = np.empty((n, k//32, 33), dtype=np.uint8)
        lib.pack_q8(x.ctypes.data, block.ctypes.data, x.size)
        q = np.ascontiguousarray(block[..., :32].reshape(n, k)).view(np.int8)
        e = np.ascontiguousarray(block[..., 32])
        # Independent E4M3FN decode. Check every packed weight against the
        # BF16 overlay that actually served in the quality experiment.
        with safe_open(args.verify_overlay / file, framework="pt") as f:
            expected = f.get_tensor(key)
            for i in range(0, n, 512):
                scales = torch.from_numpy(e[i:i+512]).view(torch.float8_e4m3fn).float() / 2
                predicted = (torch.from_numpy(q[i:i+512]).reshape(-1, k//32, 32)
                             * scales[..., None]).reshape(-1, k).bfloat16()
                if not torch.equal(predicted, expected[i:i+512]):
                    raise ValueError(f"Q8 dequant disagrees with quality overlay: {name}")
        packed = dict(codes=q, scales=e)
        save_file(packed, args.output / (name + ".safetensors"))
        manifest["packed_sha256"][name] = {key: hashlib.sha256(t).hexdigest() for key, t in packed.items()}
        print(f"{name}: {q.nbytes + e.nbytes} bytes, BF16 overlay match", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
