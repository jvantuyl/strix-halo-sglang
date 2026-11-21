#!/usr/bin/env python3
"""gfx1151 patch 40: QSA indexer rope-cache sizing without a per-call sync.

Follow-up to patch 39's py-spy work. With the metadata chain off the hot
path, the scheduler thread's next dominator during 64k-token prefill is the
QSA indexer's rope path (83% of thread samples in a prefill-window
capture): `qsa_indexer.project_qk` and `qsa_indexer.apply_rope` both call

    _ensure_cos_sin_cache_length(int(positions.max().item()))

— a device reduction + D2H sync per call, i.e. per QSA layer per prefill
chunk (12 layers x 63 chunks per 64k-token prompt) and several times per
decode round. As with patches 37/39, the value is host-known: a batch's
largest logical position is bounded by its maximum sequence length plus the
speculative draft window (draft rows may position past the pre-write
length), and the overlap scheduler publishes `seq_lens_cpu` (patch 37
routed it into this backend).

The fix keeps the lazy-grow rope cache (base.py's ensure is grow-only and
incremental) but feeds it a monotone host-side bound instead of a device
readback:

* `qsa/metadata.py` gains a module-level `QSA_HOST_MAX_POSITION` holder and
  a `_host_position_bound(positions)` helper (both files already import from
  this module, so no new import edges);
* the QSA backends record `max(seq_lens_cpu) + draft_window` in
  `_metadata_from_forward_batch` and `_replay_cuda_graph_metadata` — the two
  entry points every serving forward passes through;
* `project_qk` / `apply_rope` ensure against the holder; until a bound is
  recorded (or when `seq_lens_cpu` is absent) they keep the original
  device-readback fallback, so correctness never depends on the hint. An
  over-large bound only grows the grow-only cache slightly earlier; a stale
  bound cannot produce wrong numbers (the cache index would fail loudly).

No numerics change: the cache contents are identical for every position
actually used.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

# --- 1. Holder + helper in qsa/metadata.py -------------------------------
p = f"{path}/python/sglang/srt/layers/attention/qsa/metadata.py"
text = open(p).read()

anchor = "def build_rope_position_matrix("
assert text.count(anchor) == 1, "metadata.py anchor not found (once expected)"
holder = '''# gfx1151 patch 40: monotone host-side bound on the largest logical position
# any QSA forward has scheduled. The attention backends record it from the
# overlap-published seq_lens_cpu; qsa_indexer's rope path reads it so the
# rope cache grows without a positions.max().item() stream sync per call.
QSA_HOST_MAX_POSITION = [0]


def record_qsa_host_max_position(seq_lens_cpu, draft_window: int = 0) -> None:
    """Record max(seq_lens_cpu) + draft_window as the next rope-cache bound."""
    if seq_lens_cpu is None:
        return
    if isinstance(seq_lens_cpu, torch.Tensor):
        if seq_lens_cpu.numel() == 0:
            return
        batch_max = int(seq_lens_cpu.max())
    else:
        values = list(seq_lens_cpu)
        if not values:
            return
        batch_max = max(int(v) for v in values)
    candidate = batch_max + draft_window
    if candidate > QSA_HOST_MAX_POSITION[0]:
        QSA_HOST_MAX_POSITION[0] = candidate


def _host_position_bound(positions: torch.Tensor) -> int:
    """Rope-cache bound for this call: the recorded host bound if any.

    Falls back to the original device readback until a backend has recorded
    a bound (boot paths without the overlap publication), so correctness
    never depends on the hint.
    """
    bound = QSA_HOST_MAX_POSITION[0]
    if bound > 0:
        return bound
    return int(positions.max().item())


def build_rope_position_matrix('''
assert "import torch" in text
text = text.replace(anchor, holder, 1)
open(p, "w").write(text)

# --- 2. Indexer reads the host bound -------------------------------------
p = f"{path}/python/sglang/srt/layers/attention/qsa/qsa_indexer.py"
text = open(p).read()

old = """from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
    build_rope_position_matrix,
)"""
new = """from sglang.srt.layers.attention.qsa.metadata import (
    _host_position_bound,
    build_group_ring_slots,
    build_pending_ring_slots,
    build_rope_position_matrix,
)"""
assert text.count(old) == 1, "indexer metadata import anchor not found"
text = text.replace(old, new)

old = """                self.rotary_emb._ensure_cos_sin_cache_length(
                    int(positions.max().item())
                )"""
new = """                self.rotary_emb._ensure_cos_sin_cache_length(
                    _host_position_bound(positions)
                )"""
assert text.count(old) == 1, "project_qk ensure anchor not found"
text = text.replace(old, new)

old = """            self.rotary_emb._ensure_cos_sin_cache_length(int(positions.max().item()))"""
new = """            self.rotary_emb._ensure_cos_sin_cache_length(_host_position_bound(positions))"""
assert text.count(old) == 1, "apply_rope ensure anchor not found"
text = text.replace(old, new)
open(p, "w").write(text)

# --- 3. Backends record the bound ----------------------------------------
p = f"{path}/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
text = open(p).read()

old = """from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata,
    build_group_ring_slots,
    build_pending_ring_slots,
    build_rope_position_matrix,
    compressed_decode_view,
)"""
new = """from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata,
    build_group_ring_slots,
    build_pending_ring_slots,
    build_rope_position_matrix,
    compressed_decode_view,
    record_qsa_host_max_position,
)"""
assert text.count(old) == 1, "backend metadata import anchor not found"
text = text.replace(old, new)

old = """        speculative_paged = self._is_speculative_paged_mode(forward_batch.forward_mode)
        if speculative_paged:
            logical_positions = forward_batch.positions"""
new = """        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None:
            # gfx1151 patch 40: host-side bound for the indexer's rope-cache
            # ensure; draft rows may position past the pre-write length,
            # hence the draft window (mirrors _speculative_max_row_length).
            spec_info = getattr(forward_batch, "spec_info", None)
            draft_window = (
                int(
                    getattr(
                        spec_info,
                        "draft_token_num",
                        getattr(spec_info, "num_tokens_per_req", 0),
                    )
                    or 0
                )
                if spec_info is not None
                else 0
            )
            record_qsa_host_max_position(
                seq_lens_cpu[: int(forward_batch.seq_lens.numel())], draft_window
            )
        speculative_paged = self._is_speculative_paged_mode(forward_batch.forward_mode)
        if speculative_paged:
            logical_positions = forward_batch.positions"""
assert text.count(old) == 1, "metadata branch anchor not found"
text = text.replace(old, new)

old = """        self._require_chain_speculation(forward_mode, spec_info)
        metadata = self._cuda_graph_metadata[(forward_mode, bs)]
        # Compressed addressing is arithmetic over req_to_token; pure reads below.
        if self._can_replay_with_gpu_kernels(metadata, seq_lens):"""
new = """        self._require_chain_speculation(forward_mode, spec_info)
        if seq_lens_cpu is not None:
            # gfx1151 patch 40: same host-side rope-cache bound on the graph
            # replay path (verify and draft-extend rounds replay here).
            # spec_info is this function's parameter; no forward_batch here.
            draft_window = (
                int(
                    getattr(
                        spec_info,
                        "draft_token_num",
                        getattr(spec_info, "num_tokens_per_req", 0),
                    )
                    or 0
                )
                if spec_info is not None
                else 0
            )
            record_qsa_host_max_position(seq_lens_cpu[:bs], draft_window)
        metadata = self._cuda_graph_metadata[(forward_mode, bs)]
        # Compressed addressing is arithmetic over req_to_token; pure reads below.
        if self._can_replay_with_gpu_kernels(metadata, seq_lens):"""
assert text.count(old) == 1, "graph replay anchor not found"
text = text.replace(old, new)
open(p, "w").write(text)

print("patch_qsa_rope_host_bound: ok")
