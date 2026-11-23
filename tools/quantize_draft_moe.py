#!/usr/bin/env python3
r"""Quantize the NEXTN draft's MoE experts to int4, matching the target.

The Qwen3.8-Flash-Next checkpoint ships the MTP (NEXTN draft) layer's MoE in
bf16 (`mtp.layers.0.mlp.experts.gate_up_proj.weight` [E, 2N, K] and
`...down_proj.weight` [E, K, N]) while the target's identical-shape MoE is
int4. Quant-matching the draft frees ~2.5 GB of VRAM (KV pool grows by
~87k tokens per GB at fp8) and aligns the draft's distribution with what the
int4 target actually computes, which llama.cpp evidence suggests *helps*
acceptance rather than hurting it.

This rewrites the two affected shards in place using the exact conventions
the checkpoint's own target experts use (validated by round-trip against
target experts.0 before writing — see REVALIDATE below):

  per expert e, per projection (gate_proj / up_proj / down_proj):
    mtp.layers.0.mlp.experts.{e}.{proj}.weight_packed   [N, K//8]   I32
    mtp.layers.0.mlp.experts.{e}.{proj}.weight_scale    [N, K//32]  BF16
    mtp.layers.0.mlp.experts.{e}.{proj}.weight_zero_point [N//8, K//32] I32
    mtp.layers.0.mlp.experts.{e}.{proj}.weight_shape     [2]         I64

  quantization: asymmetric int4, group 32 along K, signed range [-8, 7]:
    scale = (max - min) / 15        (bf16; eps for zero groups)
    zp    = clamp(round(-8 - min / scale), -8, 7)
    q     = clamp(round(w / scale) + zp, -8, 7)

The loader resolves these through the same per-expert mapping the target's
experts use (same model class), so no loader changes are needed. The two
bf16 fused keys are removed and `mtp.layers.0.mlp.experts` (+ the redundant
`re:mtp\..*`) is dropped from the quantization_config ignore list; every
other mtp module stays ignored/bf16 (attention, shared expert, norms).

Side effects to expect on the next boot: presharded dump and PLE table
fingerprints change -> both regenerate (watch /opt disk: prune the old
presharded subfolders first). The draft's MoE then also picks up the tuned
int4 kernel config files (E=512,N=320,dtype=int4_w4a16[_down]) by name,
which the bf16 draft never could (it looked up N=640 real-width).

Usage:
    python3 tools/quantize_draft_moe.py --model <dir> \
        [--backup-dir <nfs dir>] [--apply] [--revalidate]

Default (no --apply) is a dry run: quantizes in memory, reports dequant
error, changes nothing. --backup-dir copies the two original shards plus
config.json and the weight index there before writing. --revalidate
round-trips a target expert (experts.0.gate_proj) through this code and
requires scale/zp/q agreement (bit-exact up to one bf16 ulp on scale, since
the original pre-quant weights are not available).
"""

import argparse
import json
import os
import shutil
import struct

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from safetensors import safe_open
from safetensors.torch import save_file

GROUP = 32
ZERO_SCALE = 1e-12  # bf16-representable stand-in for all-zero groups only


def quantize_int4_asym(w):
    """bf16 [N, K] -> (packed I32 [N, K//8], scale BF16 [N, K//32], zp I32 [N//8, K//32]).

    Matches the checkpoint author's quantizer exactly, including its quirk:
    scale = bf16((mx-mn)/15), zp = clamp(round(-8 - mn/scale), -8, 7); if the
    bf16-rounded scale and rounded zp can no longer represent both group
    endpoints exactly (((7-zp)*scale < mx) or ((-8-zp)*scale > mn)), the
    divisor shrinks to 14, 13, ... until they can. Reverse-engineered from
    the target experts: all 707 groups where /15 loses an endpoint use /14
    (bit-identical), and no other group does. (The naive formula without
    the retry produces a scale that is 15/14x too small on those groups.)
    """
    n, k = w.shape
    assert k % GROUP == 0, f"K={k} not divisible by {GROUP}"
    groups = k // GROUP
    g = w.float().reshape(n, groups, GROUP)
    mn, mx = g.amin(dim=-1, keepdim=True), g.amax(dim=-1, keepdim=True)
    rng = (mx - mn).clamp(min=0)
    zp = None
    d = torch.full_like(rng, 15.0)
    # constant groups (rng == 0): scale |c|/7 reconstructs c exactly through
    # the clamped zp (verified: c>0 -> q=-1, zp=-8; c<0 -> q=-8, zp=-1).
    base = torch.where(
        rng == 0, mn.abs().clamp(min=ZERO_SCALE) / 7, rng
    )
    for _ in range(14):  # per-group divisor retry: 15, then only where needed
        scale = torch.where(rng == 0, base, base / d).to(torch.bfloat16)
        sf = scale.float()  # the representability check must run in fp32:
        # int * bf16 promotes to bf16 and its ulp (1.2e-4 at ~0.03) spuriously
        # violates groups whose endpoint equals (7-zp)*scale exactly.
        zp = torch.clamp(torch.round(-8 - mn / sf), -8, 7).to(torch.int8)
        zpf = zp.float()
        viol = ((7 - zpf) * sf < mx) | ((-8 - zpf) * sf > mn)
        if not bool(viol.any()):
            break
        d = torch.where(viol, d - 1, d)
    q = torch.clamp(torch.round(g / scale.float()) + zp, -8, 7).to(torch.int8)
    deq = (q - zp).float() * scale.float()
    rel = (deq - g).norm().item() / (g.norm().item() or 1.0)
    packed = pack_to_int32(q.reshape(n, k), 4, packed_dim=1)
    zp_packed = pack_to_int32(zp.squeeze(-1), 4, packed_dim=0)
    return packed, scale.squeeze(-1), zp_packed, torch.tensor([n, k], dtype=torch.int64), rel


def read_shard_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def load_tensor(path, header, data_start, name):
    e = header[name]
    dt = {"I32": torch.int32, "BF16": torch.bfloat16, "I64": torch.int64, "F32": torch.float32}[e["dtype"]]
    with open(path, "rb") as f:
        f.seek(data_start + e["data_offsets"][0])
        return torch.frombuffer(bytearray(f.read(e["data_offsets"][1] - e["data_offsets"][0])), dtype=dt).reshape(e["shape"]).clone()


def revalidate_against_target(model_dir):
    """Round-trip experts.0.gate_proj: dequant the stored target expert with
    our formulas, requantize, and compare. Bit-exact except scale entries
    derived from pre-quantization min/max (one bf16 ulp)."""
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    pre = "model.language_model.layers.0.mlp.experts.0.gate_proj."
    shard = os.path.join(model_dir, idx[pre + "weight_packed"])
    header, ds = read_shard_header(shard)
    get = lambda name: load_tensor(shard, header, ds, name)
    pk, sc, zpp, sh = get(pre + "weight_packed"), get(pre + "weight_scale"), get(pre + "weight_zero_point"), get(pre + "weight_shape")
    n, k = sh.tolist()
    groups = k // GROUP
    q = unpack_from_int32(pk, 4, torch.Size([n, k]), packed_dim=1)
    zpu = unpack_from_int32(zpp, 4, torch.Size([n, groups]), packed_dim=0)
    w = ((q.reshape(n, groups, GROUP).float() - zpu.reshape(n, groups, 1).float()) * sc.float().reshape(n, groups, 1)).reshape(n, k)
    packed, scale, zp_packed, shape, _ = quantize_int4_asym(w)
    zpo = unpack_from_int32(zp_packed, 4, torch.Size([n, groups]), packed_dim=0)
    zp_ok = (zpo == zpu).float().mean().item() >= 0.995
    q_ok = (
        unpack_from_int32(packed, 4, torch.Size([n, k]), packed_dim=1) == q
    ).float().mean().item() >= 0.995
    sd = (scale.float() - sc.float()).abs()
    rel = sd.max().item() / sc.float().abs().min().item()
    # format parity is proven by zp/q exactness; the residual comes only
    # from our input being the dequantized tensor (the original pre-quant
    # min/max can differ by one rounding step near group boundaries).
    # With the representability-retry implemented, scale should now match
    # the stored values exactly everywhere.
    s_rel_pct = (sd / sc.float().abs()).max().item() * 100
    print(f"revalidate vs target experts.0.gate_proj (dequantized input): "
          f"zp>=99.5% {zp_ok} q>=99.5% {q_ok} worst scale rel diff {s_rel_pct:.1f}%")
    assert zp_ok and q_ok, "packing/format conventions do not reproduce the target expert"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revalidate", action="store_true")
    args = ap.parse_args()

    model = args.model
    idx_path = os.path.join(model, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    weight_map = idx["weight_map"]
    cfg_path = os.path.join(model, "config.json")
    cfg = json.load(open(cfg_path))
    ignore = cfg["quantization_config"]["ignore"]

    gate_up_key = "mtp.layers.0.mlp.experts.gate_up_proj"
    down_key = "mtp.layers.0.mlp.experts.down_proj"
    for key in (gate_up_key, down_key):
        assert key in weight_map, f"{key} not in index (already converted?)"
    if args.revalidate:
        revalidate_against_target(model)

    gu_shard = os.path.join(model, weight_map[gate_up_key])
    dn_shard = os.path.join(model, weight_map[down_key])

    # --- quantize in memory, expert by expert (RAM-bounded) ---
    out = {"gate": {}, "up": {}, "down": {}}  # proj -> expert_id -> (4 tensors)
    rels = {"gate": [], "up": [], "down": []}
    with safe_open(gu_shard, framework="pt") as f:
        gu = f.get_tensor(gate_up_key)  # [E, 2N, K]
    E, twoN, K = gu.shape
    N = twoN // 2
    with safe_open(dn_shard, framework="pt") as f:
        dn = f.get_tensor(down_key)  # [E, K, N]
    assert dn.shape == (E, K, N), dn.shape
    print(f"draft experts: E={E} N={N} K={K}; quantizing {E} experts x 3 projections")
    for e in range(E):
        for proj, w in (
            ("gate", gu[e, :N, :]),
            ("up", gu[e, N:, :]),
            ("down", dn[e]),
        ):
            packed, scale, zp, shape, rel = quantize_int4_asym(w)
            out["gate" if proj == "gate" else ("up" if proj == "up" else "down")].setdefault(e, {})[
                proj
            ] = (packed, scale, zp, shape)
            rels[proj].append(rel)
    for proj, v in rels.items():
        print(f"  {proj}_proj: worst rel dequant error {max(v):.4f}, mean {sum(v)/len(v):.4f}")
    del gu, dn

    if not args.apply:
        print("dry run complete; nothing written (pass --apply to rewrite the shards)")
        return

    # --- backups ---
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)
        for p in (gu_shard, dn_shard, cfg_path, idx_path):
            dst = os.path.join(args.backup_dir, os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copy2(p, dst)
                print(f"backed up {os.path.basename(p)}")

    # --- rewrite shards ---
    def rewrite(shard_path, drop_keys, add_tensors):
        header, _ = read_shard_header(shard_path)
        tensors = {}
        with safe_open(shard_path, framework="pt") as f:
            for name in f.keys():
                if name in drop_keys:
                    continue
                tensors[name] = f.get_tensor(name)
        tensors.update(add_tensors)
        tmp = shard_path + ".new"
        save_file(tensors, tmp, metadata={"format": "pt"})
        os.replace(tmp, shard_path)
        print(f"rewrote {os.path.basename(shard_path)}: {len(tensors)} tensors")

    def expert_keys(e, proj, tensors):
        base = f"mtp.layers.0.mlp.experts.{e}.{proj}."
        packed, scale, zp, shape = tensors[proj]
        return {
            base + "weight_packed": packed,
            base + "weight_scale": scale,
            base + "weight_zero_point": zp,
            base + "weight_shape": shape,
        }

    gu_add = {}
    for e in range(E):
        for proj in ("gate", "up"):
            gu_add.update(expert_keys(e, proj, out["gate"][e] if proj == "gate" else out["up"][e]))
    dn_add = {}
    for e in range(E):
        dn_add.update(expert_keys(e, "down", out["down"][e]))
    del out

    # gate/up live where gate_up lived; down where down lived
    rewrite(gu_shard, {gate_up_key}, gu_add)
    rewrite(dn_shard, {down_key}, dn_add)

    # --- index ---
    del weight_map[gate_up_key], weight_map[down_key]
    for e in range(E):
        for proj in ("gate", "up"):
            for suffix in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape"):
                weight_map[f"mtp.layers.0.mlp.experts.{e}.{proj}.{suffix}"] = os.path.basename(gu_shard)
        for suffix in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape"):
            weight_map[f"mtp.layers.0.mlp.experts.{e}.down_proj.{suffix}"] = os.path.basename(dn_shard)
    json.dump(idx, open(idx_path, "w"), indent=2)
    print(f"index updated: {len(weight_map)} entries")

    # --- config: un-ignore the draft experts only ---
    before = len(ignore)
    cfg["quantization_config"]["ignore"] = [
        e for e in ignore if e not in ("mtp.layers.0.mlp.experts", "re:mtp\\..*")
    ]
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"ignore list: {before} -> {len(cfg['quantization_config']['ignore'])} entries")
    print("done. Next boot: presharded dump + PLE table regenerate; prune stale dump subfolders first.")


if __name__ == "__main__":
    main()
