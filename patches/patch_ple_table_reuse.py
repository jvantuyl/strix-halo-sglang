#!/usr/bin/env python3
"""gfx1151 patch 13: reuse the file-backed PLE table across boots.

With ``--ple-offload-backend file`` the PLE n-gram table lives in a sparse
file that upstream keeps between restarts (deterministic name, only
re-created when the size changes) -- but the weight loader still reads every
PLE shard of the checkpoint and copies all of it into the mmap on every boot.
For Qwen3.8-Flash-Next that is 48 GiB read from the checkpoint and 48 GiB of
page-faulting writes into the table, every start.

This patch records a completion marker next to the table once a boot has
written every shard, fingerprinted by the checkpoint's safetensors index and
the PLE-only shard files (name + size). On the next boot, if the marker
matches, the model tells the loader to skip those shard files entirely and
nothing is copied. Any change to the checkpoint (a re-quantized table, a
different index) or a re-created table file invalidates the marker and the
boot falls back to the full write, which re-arms it.

Three files:

  * qwen4_exp_ple_table.py -- marker helpers; a re-created table drops a
    stale marker.
  * qwen4_exp.py -- the host embedding remembers its table path; the model
    exposes ``weight_files_to_skip(hf_folder, files)`` and marks the table
    complete at the end of ``load_weights`` when every shard was written.
  * model_loader/loader.py -- ``Source`` carries the model's filter (same
    pattern as ``allow_patterns_overrides``) and both file-resolution paths
    apply it.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

# ---------------------------------------------------------------------------
# qwen4_exp_ple_table.py: completion marker helpers
# ---------------------------------------------------------------------------
p = f"{path}/python/sglang/srt/models/qwen4_exp_ple_table.py"
text = open(p).read()

old = "import ctypes\nimport ctypes.util\nimport logging\n"
assert text.count(old) == 1, "ple_table: import anchor not found"
text = text.replace(old, "import ctypes\nimport ctypes.util\nimport json\nimport logging\n", 1)

old = """        with open(path, "wb") as f:
            f.truncate(nbytes)
"""
assert text.count(old) == 1, "ple_table: truncate anchor not found"
new = """        with open(path, "wb") as f:
            f.truncate(nbytes)
        # A fresh file holds no rows yet; a marker left by a previous file
        # must not let the next boot skip the write.
        try:
            os.remove(ple_table_complete_path(path))
        except FileNotFoundError:
            pass
"""
text = text.replace(old, new, 1)

text += '''

# ---------------------------------------------------------------------------
# gfx1151 patch 13: table reuse across boots
# ---------------------------------------------------------------------------
PLE_TABLE_COMPLETE_SUFFIX = ".complete.json"
_PLE_SHARD_NAME = re.compile(r"\\.ngram_embedding\\.shard_\\d+\\.weight$")
_PLE_INDEX_FILE = "model.safetensors.index.json"


def ple_table_complete_path(table_path: str) -> str:
    return table_path + PLE_TABLE_COMPLETE_SUFFIX


def ple_only_weight_files(hf_folder: str, weight_files: Sequence[str]) -> list[str]:
    """Checkpoint files that hold nothing but PLE n-gram shards.

    Decided from the safetensors index so a checkpoint that interleaves the
    table with other tensors simply yields no skippable files.
    """
    try:
        with open(os.path.join(hf_folder, _PLE_INDEX_FILE)) as f:
            weight_map = json.load(f)["weight_map"]
    except (OSError, KeyError, ValueError):
        return []
    tensors_per_file: dict[str, int] = {}
    ple_per_file: dict[str, int] = {}
    for name, file_name in weight_map.items():
        tensors_per_file[file_name] = tensors_per_file.get(file_name, 0) + 1
        if _PLE_SHARD_NAME.search(name):
            ple_per_file[file_name] = ple_per_file.get(file_name, 0) + 1
    ple_only = {
        file_name
        for file_name, count in tensors_per_file.items()
        if ple_per_file.get(file_name, 0) == count
    }
    return [f for f in weight_files if os.path.basename(f) in ple_only]


def ple_table_fingerprint(hf_folder: str, ple_files: Sequence[str]) -> dict:
    index_stat = os.stat(os.path.join(hf_folder, _PLE_INDEX_FILE))
    return {
        "version": 1,
        "index": {"size": index_stat.st_size, "mtime_ns": index_stat.st_mtime_ns},
        "files": [
            {"name": os.path.basename(f), "size": os.path.getsize(f)}
            for f in sorted(ple_files)
        ],
    }


def ple_table_is_complete(table_path: str, fingerprint: dict) -> bool:
    try:
        with open(ple_table_complete_path(table_path)) as f:
            recorded = json.load(f)
        table_size = os.path.getsize(table_path)
    except (OSError, ValueError):
        return False
    return (
        recorded.get("fingerprint") == fingerprint
        and recorded.get("table_size") == table_size
    )


def mark_ple_table_complete(table_path: str, fingerprint: dict) -> None:
    # The rows were written through a shared mmap; get them onto storage
    # before anything claims the file is complete.
    fd = os.open(table_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    marker = ple_table_complete_path(table_path)
    tmp = marker + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {"fingerprint": fingerprint, "table_size": os.path.getsize(table_path)},
            f,
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, marker)
'''
open(p, "w").write(text)
print("patched", p)

# ---------------------------------------------------------------------------
# qwen4_exp.py: expose the skip list, mark completion
# ---------------------------------------------------------------------------
p = f"{path}/python/sglang/srt/models/qwen4_exp.py"
text = open(p).read()

old = """from sglang.srt.models.qwen4_exp_ple_table import (
    allocate_ple_host_table,
"""
assert text.count(old) == 1, "qwen4_exp: ple_table import anchor not found"
new = """from sglang.srt.models.qwen4_exp_ple_table import (
    allocate_ple_host_table,
    mark_ple_table_complete,
    ple_only_weight_files,
    ple_table_fingerprint,
    ple_table_is_complete,
"""
text = text.replace(old, new, 1)

old = "        self._file_prefetcher = make_ple_file_prefetcher(host_table)\n"
assert text.count(old) == 1, "qwen4_exp: prefetcher anchor not found"
new = """        self._file_prefetcher = make_ple_file_prefetcher(host_table)
        # gfx1151 patch 13: None for the pinned backend.
        self.table_file_path = getattr(host_table, "_sglang_ple_file_path", None)
"""
text = text.replace(old, new, 1)

old = "    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):\n"
assert text.count(old) == 1, "qwen4_exp: load_weights anchor not found"
new = '''    def _ple_file_tables(self) -> list:
        return [
            module
            for module in self.modules()
            if isinstance(module, Qwen4ExpPinnedHostEmbedding)
            and getattr(module, "table_file_path", None)
        ]

    def weight_files_to_skip(self, hf_folder: str, weight_files: list) -> list:
        """Checkpoint files the loader may skip: PLE-only shards whose rows a
        previous boot already wrote into the file-backed table (patch 13)."""
        tables = self._ple_file_tables()
        if not tables:
            return []
        ple_files = ple_only_weight_files(hf_folder, weight_files)
        if not ple_files:
            return []
        fingerprint = ple_table_fingerprint(hf_folder, ple_files)
        if all(
            ple_table_is_complete(table.table_file_path, fingerprint)
            for table in tables
        ):
            logger.info(
                "PLE table: reusing %d table file(s) from a previous boot, "
                "skipping %d checkpoint shard files",
                len(tables),
                len(ple_files),
            )
            self._ple_table_reused = True
            return ple_files
        self._ple_table_fingerprint = fingerprint
        return []

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
'''
text = text.replace(old, new, 1)

old = "        loaded_shard_params: Set[str] = set()\n"
assert text.count(old) == 1, "qwen4_exp: loaded_shard_params anchor not found"
new = """        loaded_shard_params: Set[str] = set()
        ple_shards_written: dict = {}
"""
text = text.replace(old, new, 1)

old = """            loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
            return True
"""
assert text.count(old) == 1, "qwen4_exp: shard bookkeeping anchor not found"
new = """            loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
            ple_shards_written[mod_prefix] = ple_shards_written.get(mod_prefix, 0) + 1
            return True
"""
text = text.replace(old, new, 1)

old = """                module.finalize_fused_in_proj()

        return loaded_params
"""
assert text.count(old) == 1, "qwen4_exp: load_weights return anchor not found"
new = """                module.finalize_fused_in_proj()

        # gfx1151 patch 13: arm table reuse for the next boot once every
        # shard of every file-backed table has been written.
        fingerprint = getattr(self, "_ple_table_fingerprint", None)
        if fingerprint is not None and not getattr(self, "_ple_table_reused", False):
            complete = [
                mod.ngram_embedding
                for mod_prefix, mod in ple_modules.items()
                if isinstance(mod.ngram_embedding, Qwen4ExpPinnedHostEmbedding)
                and getattr(mod.ngram_embedding, "table_file_path", None)
                and ple_shards_written.get(mod_prefix, 0) == ple_num_sync_shards
            ]
            for emb in complete:
                mark_ple_table_complete(emb.table_file_path, fingerprint)
            if complete:
                logger.info(
                    "PLE table: %d table file(s) marked complete; the next boot "
                    "skips the PLE shards",
                    len(complete),
                )

        return loaded_params
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)

# ---------------------------------------------------------------------------
# model_loader/loader.py: let the model drop checkpoint files
# ---------------------------------------------------------------------------
p = f"{path}/python/sglang/srt/model_loader/loader.py"
text = open(p).read()

old = """        model_config: Optional[ModelConfig] = None
        \"\"\"The model configuration (for checking architecture, etc).\"\"\"

        @classmethod
        def init_new(cls, model_config: ModelConfig, model):
            return cls(
                model_config.model_path,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                allow_patterns_overrides=getattr(
                    model, "allow_patterns_overrides", None
                ),
                model_config=model_config,
            )
"""
assert text.count(old) == 1, "loader: Source anchor not found"
new = """        model_config: Optional[ModelConfig] = None
        \"\"\"The model configuration (for checking architecture, etc).\"\"\"

        weight_files_filter: Optional[Any] = None
        \"\"\"``model.weight_files_to_skip(hf_folder, files) -> files`` if the
        model can drop checkpoint files whose contents it already holds
        (gfx1151 patch 13: file-backed PLE table).\"\"\"

        @classmethod
        def init_new(cls, model_config: ModelConfig, model):
            return cls(
                model_config.model_path,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                allow_patterns_overrides=getattr(
                    model, "allow_patterns_overrides", None
                ),
                model_config=model_config,
                weight_files_filter=getattr(model, "weight_files_to_skip", None),
            )

        def apply_weight_files_filter(
            self, hf_folder: str, weight_files: List[str]
        ) -> List[str]:
            if self.weight_files_filter is None:
                return weight_files
            skipped = set(self.weight_files_filter(hf_folder, list(weight_files)))
            if not skipped:
                return weight_files
            return [f for f in weight_files if f not in skipped]
"""
text = text.replace(old, new, 1)

old = """            if use_safetensors and source.model_config is not None:
                hf_weights_files = maybe_add_mtp_safetensors(
                    hf_weights_files,
                    hf_folder,
                    "model.safetensors.index.json",
                    source.model_config.hf_config,
                )
        else:
            hf_folder = resolved_source.hf_folder
"""
assert text.count(old) == 1, "loader: _get_weights_iterator anchor not found"
new = """            if use_safetensors and source.model_config is not None:
                hf_weights_files = maybe_add_mtp_safetensors(
                    hf_weights_files,
                    hf_folder,
                    "model.safetensors.index.json",
                    source.model_config.hf_config,
                )
            hf_weights_files = source.apply_weight_files_filter(
                hf_folder, hf_weights_files
            )
        else:
            hf_folder = resolved_source.hf_folder
"""
text = text.replace(old, new, 1)

old = """            if use_safetensors and source.model_config is not None:
                weight_files = maybe_add_mtp_safetensors(
                    weight_files,
                    hf_folder,
                    "model.safetensors.index.json",
                    source.model_config.hf_config,
                )
            resolved_sources.append(
"""
assert text.count(old) == 1, "loader: resolve_model_weights anchor not found"
new = """            if use_safetensors and source.model_config is not None:
                weight_files = maybe_add_mtp_safetensors(
                    weight_files,
                    hf_folder,
                    "model.safetensors.index.json",
                    source.model_config.hf_config,
                )
            weight_files = source.apply_weight_files_filter(hf_folder, weight_files)
            resolved_sources.append(
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)
print("patch 13 (PLE table reuse) applied")
