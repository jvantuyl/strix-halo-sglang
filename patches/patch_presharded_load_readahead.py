#!/usr/bin/env python3
"""gfx1151 patch 42: presharded reload reads into pinned memory, one file ahead.

Measured on this box (2026-11-20, 75.1 GiB dump, NVMe on vg00-opt):

* the reload ran at 758 MiB/s (101 s) while the device does 4.2+ GiB/s;
* safetensors' get_tensor() returns zero-copy VIEWS into the file's mmap
  (confirmed via /proc/self/maps), so all page-in happens later, inside the
  consumer's copy_() into GPU memory — cudaMemcpy pin-faults file-backed
  pages one 4 KiB page at a time, measured 0.79 GiB/s. The GPU copy itself
  is free once the source is pinned: 66 GiB/s from pinned memory.
* POSIX_FADV_WILLNEED is a no-op on this kernel (returns instantly, warms
  nothing), so a fadvise-based prefetch cannot work;
* os.preadv with O_DIRECT into a pinned buffer: 4.24 GiB/s single stream
  (buffered: 1.15; further concurrency does not scale past ~4.4).

Fix: stage each dump file in a plain 512-aligned CPU buffer on a background
thread that runs one file ahead of the apply loop, and hand the consumer
views into that buffer, so its copy_() transfers from regular pageable
memory at PCIe speed instead of faulting pages. The fallback ladder keeps
the old behavior intact: O_DIRECT -> buffered read into the buffer ->
original safe_open mmap views. Extra host memory: one shard (~1.3 GiB),
truly freed after use (no pinned-allocator cache — see _SAFETENSORS_DTYPES
comment for why pinned was rejected on this UMA box).

Load-time tensor hashing is unaffected: verify_on_load defaults off and this
deployment does not set it; _audit_presharded_root is stat-only.

No numerics change: identical tensors, order, and verification semantics.
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

p = f"{path}/python/sglang/srt/model_loader/loader.py"
text = open(p).read()

old = """            for filename, items in by_file.items():
                stored_keys = list(dict.fromkeys(r["stored_key"] for r in items))
                cached = self._read_presharded_file(
                    os.path.join(presharded_dir, filename), stored_keys
                )
                self._apply_presharded_file(
                    items=items,
                    cached=cached,
                    model=model,
                    state_dict=state_dict,
                    target_device=target_device,
                    loaded_param_keys=loaded_param_keys,
                    verify_hashes=verify_hashes,
                )
                progress.step(file_bytes[filename])"""
new = """            # gfx1151 patch 42: park the bandwidth-idle params BEFORE the
            # reload and fill them from the staged host buffers. The old
            # order copied them into VRAM and back out at park time, and on
            # this iGPU freeing those VRAM-backed pages stopped showing up
            # as available system memory, shrinking the KV budget by ~3 GB
            # (measured 2026-11-20; see patches/42-presharded-load-readahead.md).
            preparked_bytes = 0
            if (
                not model_config.is_draft_model
                and os.environ.get("QWEN38_PRESHARDED_PREPARK", "1") != "0"
            ):
                from sglang.srt.model_executor.host_parked_params import (
                    host_parked_patterns,
                    is_parked,
                    prepark_parameter,
                )

                patterns = host_parked_patterns()
                for name, param in model.named_parameters():
                    if any(p in name for p in patterns) and not is_parked(param):
                        prepark_parameter(param)
                        preparked_bytes += param.numel() * param.element_size()
                if preparked_bytes:
                    logger.info(
                        "Preparked %.2f GiB of parameters before the reload "
                        "(filled from the staged host buffers)",
                        preparked_bytes / 2**30,
                    )

            # gfx1151 patch 42: each file is staged into an aligned CPU
            # buffer one read ahead of the apply loop, so the NVMe transfer
            # overlaps the previous file's GPU copies instead of
            # page-faulting inside them. copy_stage: one fixed pinned block
            # the copies DMA through (see _copy_to_gpu for why).
            copy_stage = torch.empty(
                PreshardedModelLoader._COPY_STAGE_BYTES,
                dtype=torch.uint8,
                pin_memory=True,
            )
            stage_enabled = os.environ.get("QWEN38_PRESHARDED_STAGE", "1") != "0"
            mem_trace = os.environ.get("QWEN38_PRESHARDED_MEM_TRACE") in ("1", "2")

            def _trace_free(tag):
                # On integrated GPUs SGLang's KV budget is psutil's
                # MemAvailable (see get_available_gpu_memory); trace the same
                # metric, not torch.cuda.mem_get_info. QWEN38_PRESHARDED_MEM_TRACE=2
                # additionally dumps the full page accounting plus the
                # device-side view, to separate the VRAM carve from the
                # Linux side when comparing load paths.
                if not mem_trace:
                    return
                import psutil

                vm = psutil.virtual_memory()
                if mem_trace == "1":
                    logger.info(
                        "presharded mem trace %s: avail %.2f GiB",
                        tag,
                        vm.available / 2**30,
                    )
                    return
                mi = {}
                with open("/proc/meminfo") as f:
                    for line in f:
                        key, rest = line.split(":", 1)
                        mi[key] = int(rest.split()[0]) / 2**20  # -> GiB
                free_dev, total_dev = torch.cuda.mem_get_info()
                logger.info(
                    "presharded mem trace %s: avail=%.2f free=%.2f "
                    "cached=%.2f shmem=%.2f anon=%.2f unevict=%.2f "
                    "slab_reclaim=%.2f swap_used=%.2f | dev_free=%.2f "
                    "dev_total=%.2f GiB",
                    tag,
                    vm.available / 2**30,
                    mi.get("MemFree", 0),
                    (mi.get("Cached", 0) - mi.get("Shmem", 0)),
                    mi.get("Shmem", 0),
                    mi.get("AnonPages", 0),
                    mi.get("Unevictable", 0),
                    mi.get("SReclaimable", 0),
                    mi.get("SwapTotal", 0) - mi.get("SwapFree", 0),
                    free_dev / 2**30,
                    total_dev / 2**30,
                )

            serial = os.environ.get("QWEN38_PRESHARDED_SERIAL") == "1"

            def _apply_one(filename, items, cached):
                self._apply_presharded_file(
                    items=items,
                    cached=cached,
                    model=model,
                    state_dict=state_dict,
                    target_device=target_device,
                    loaded_param_keys=loaded_param_keys,
                    verify_hashes=verify_hashes,
                    copy_stage=copy_stage if not serial else None,
                )
                progress.step(file_bytes[filename])

            if not stage_enabled:
                logger.info("QWEN38_PRESHARDED_STAGE=0: using the plain mmap read path")
                for filename, items in by_file.items():
                    stored_keys = list(dict.fromkeys(r["stored_key"] for r in items))
                    cached = self._read_presharded_file(
                        os.path.join(presharded_dir, filename), stored_keys
                    )
                    _apply_one(filename, items, cached)
            elif serial:
                logger.info("QWEN38_PRESHARDED_SERIAL=1: staged reads, no prefetch thread")
                for filename, items in by_file.items():
                    stored_keys = list(dict.fromkeys(r["stored_key"] for r in items))
                    _trace_free(f"pre-stage {filename}")
                    cached = self._staged_file_reads(
                        os.path.join(presharded_dir, filename), stored_keys
                    )
                    _trace_free(f"post-stage {filename}")
                    _apply_one(filename, items, cached)
                    _trace_free(f"post-apply {filename}")
            else:
                for filename, cached in self._prefetched_file_reads(
                    presharded_dir, by_file
                ):
                    _trace_free(f"before apply {filename}")
                    _apply_one(filename, by_file[filename], cached)
                    _trace_free(f"after apply {filename}")
            del copy_stage"""
assert text.count(old) == 1, "presharded read loop anchor not found"
text = text.replace(old, new)

old = """    @staticmethod
    def _read_presharded_file(
        full_path: str, stored_keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        from safetensors.torch import safe_open

        with safe_open(full_path, framework="pt") as fh:
            return {key: fh.get_tensor(key) for key in stored_keys}"""
new = """    @staticmethod
    def _read_presharded_file(
        full_path: str, stored_keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        from safetensors.torch import safe_open

        with safe_open(full_path, framework="pt") as fh:
            return {key: fh.get_tensor(key) for key in stored_keys}

    # gfx1151 patch 42: safetensors' get_tensor returns zero-copy views into
    # the file mmap, deferring all page-in into the consumer's GPU copy,
    # where cudaMemcpy pin-faults pages one by one (~0.8 GiB/s here). These
    # helpers stage the file in a plain aligned CPU buffer instead: O_DIRECT
    # preadv reaches ~4.2 GiB/s on the target NVMe and the consumer's copy_
    # from pageable memory runs at ~4.8 GiB/s (measured), both far above the
    # old path. NOT pinned: on this UMA box pinned host allocations are
    # counted by torch.cuda.mem_get_info (a first draft that staged into
    # pinned memory cost ~9 GB of KV budget — the pinned caching allocator
    # keeps one cached block per distinct file size, and on Strix Halo host
    # pinned pages show up in the device memory accounting). Plain CPU
    # tensors of this size are glibc-mmap'd and truly returned on free.
    # WILLNEED-style readahead hints were measured as no-ops on this kernel,
    # so there is no fadvise path.
    _SAFETENSORS_DTYPES = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    @staticmethod
    def _staged_views_from_buffer(
        full_path: str, stored_keys: List[str], direct: bool
    ):
        \"""Read a safetensors file into one aligned CPU buffer, return views.\"""
        size = os.path.getsize(full_path)
        # O_DIRECT needs 512-byte alignment of the device address as well;
        # the torch CPU allocator guarantees 64, so pad and slice.
        raw = torch.empty(size + 1024, dtype=torch.uint8)
        base = raw.data_ptr()
        pad = (512 - base % 512) % 512
        mv = memoryview(raw.numpy())[pad : pad + size]
        aligned_end = size // 512 * 512 if direct else size
        fd = os.open(
            full_path,
            os.O_RDONLY | (os.O_DIRECT if direct else 0),
        )
        try:
            got = 0
            chunk = 8 << 20 if direct else 32 << 20
            while got < aligned_end:
                want = min(chunk, aligned_end - got)
                r = os.preadv(fd, [mv[got : got + want]], got)
                if r <= 0:
                    raise ValueError(
                        f"short read from {full_path}: {got} < {aligned_end}"
                    )
                got += r
            if got != aligned_end:
                raise ValueError(
                    f"short read from {full_path}: {got} < {aligned_end}"
                )
        finally:
            os.close(fd)
        tail = size - aligned_end
        if tail:
            fd2 = os.open(full_path, os.O_RDONLY)
            try:
                r = os.preadv(fd2, [mv[aligned_end :]], aligned_end)
                if r != tail:
                    raise ValueError(f"short tail read from {full_path}")
            finally:
                os.close(fd2)
        with open(full_path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n))
        data_start = 8 + n
        entries = {k: v for k, v in header.items() if k != "__metadata__"}
        if not set(stored_keys).issubset(entries):
            raise ValueError(
                f"plan keys missing from {full_path}: "
                f"{sorted(set(stored_keys) - set(entries))}"
            )
        views = {}
        for key in stored_keys:
            m = entries[key]
            begin, end = m["data_offsets"]
            dtype = PreshardedModelLoader._SAFETENSORS_DTYPES.get(m["dtype"])
            if dtype is None:
                raise ValueError(f"unsupported dtype {m['dtype']!r} in {full_path}")
            flat = torch.frombuffer(
                mv[data_start + begin : data_start + end], dtype=dtype
            )
            views[key] = flat.reshape(m["shape"])
        return views

    def _staged_file_reads(self, full_path: str, stored_keys: List[str]):
        \"\"\"gfx1151 patch 42: staged views for a dump file, with fallbacks.\"\"\"
        try:
            return self._staged_views_from_buffer(full_path, stored_keys, direct=True)
        except (OSError, ValueError) as e:
            logger.warning(
                "O_DIRECT staging unavailable for %s (%s); using buffered "
                "reads.",
                full_path,
                e,
            )
        try:
            return self._staged_views_from_buffer(
                full_path, stored_keys, direct=False
            )
        except (OSError, ValueError) as e:
            logger.warning(
                "Staging failed for %s (%s); using safetensors mmap views.",
                full_path,
                e,
            )
        return self._read_presharded_file(full_path, stored_keys)

    # gfx1151 patch 42: yield (filename, tensors) one read ahead of the
    # consumer, so file N+1's staging runs while file N's tensors are being
    # copied to the GPU.
    def _prefetched_file_reads(self, presharded_dir: str, by_file: Dict):
        upcoming = iter(list(by_file.items()))

        def _prefetch(entry):
            filename, items = entry
            full_path = os.path.join(presharded_dir, filename)
            stored_keys = list(dict.fromkeys(r["stored_key"] for r in items))
            trace = os.environ.get("QWEN38_PRESHARDED_MEM_TRACE") == "1"
            if trace:
                import psutil

                logger.info(
                    "presharded mem trace pre-stage %s: avail %.2f GiB",
                    filename,
                    psutil.virtual_memory().available / 2**30,
                )
            out = self._staged_file_reads(full_path, stored_keys)
            if trace:
                import psutil

                logger.info(
                    "presharded mem trace post-stage %s: avail %.2f GiB",
                    filename,
                    psutil.virtual_memory().available / 2**30,
                )
            return out

        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="presharded-prefetch"
        )
        pending = None
        try:
            entry = next(upcoming, None)
            if entry is not None:
                pending = (entry, pool.submit(_prefetch, entry))
            while pending is not None:
                entry, fut = pending
                nxt = next(upcoming, None)
                pending = (nxt, pool.submit(_prefetch, nxt)) if nxt else None
                yield entry[0], fut.result()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    # gfx1151 patch 42: H2D copies from a pageable source make cudaMemcpy
    # stage through torch's internal pinned cache — one cached block per
    # distinct tensor size, and on this UMA box those pinned pages count
    # against the device memory the KV budget is computed from (one reload
    # left ~3 GiB behind, unrecoverable by empty_cache). Routing every
    # large copy through ONE fixed pinned stage block avoids that: direct
    # DMA from pinned memory, no per-size caching, ~64 MiB pinned total.
    # The torch.cuda.synchronize() before each stage reuse guarantees the
    # previous async H2D finished reading it.
    _COPY_STAGE_BYTES = 64 << 20
    _COPY_STAGE_MIN = 1 << 20

    def _copy_to_gpu(self, param_data, src, stage):
        # gfx1151 patch 42: a parked parameter's storage is host-mapped
        # memory aliased as a CUDA tensor (patch 27). Fill that mapping
        # directly from the staged host view — pure CPU memcpy — so the
        # bytes never enter VRAM at all.
        from sglang.srt.model_executor.host_parked_params import parked_buffer

        buf = parked_buffer(param_data)
        if buf is not None:
            import ctypes

            nbytes = param_data.numel() * param_data.element_size()
            host = (ctypes.c_uint8 * nbytes).from_address(buf._host_ptr)
            flat = torch.frombuffer(host, dtype=param_data.dtype)
            flat.copy_(src.reshape(-1))
            return
        src_flat = src.reshape(-1)
        dst_flat = param_data.reshape(-1)
        if (
            stage is None
            or src_flat.numel() < self._COPY_STAGE_MIN
            or not src_flat.is_contiguous()
            or not dst_flat.is_contiguous()
        ):
            param_data.copy_(src)
            return
        src_u8 = src_flat.view(torch.uint8) if src_flat.dtype != torch.uint8 else src_flat
        dst_u8 = dst_flat.view(torch.uint8) if dst_flat.dtype != torch.uint8 else dst_flat
        n = src_u8.numel()
        off = 0
        while off < n:
            m = min(stage.numel(), n - off)
            torch.cuda.synchronize()
            stage[:m].copy_(src_u8[off : off + m])
            dst_u8[off : off + m].copy_(stage[:m], non_blocking=True)
            off += m"""
assert text.count(old) == 1, "_read_presharded_file anchor not found"
text = text.replace(old, new)
open(p, "w").write(text)

# --- 2b. prepark_parameter in the host-parked-params helper ---------------
p = f"{path}/python/sglang/srt/model_executor/host_parked_params.py"
text = open(p).read()

old = """def park_parameter_in_host_memory(param: torch.nn.Parameter) -> HostParkedBuffer:"""
new = """def prepark_parameter(param: torch.nn.Parameter) -> HostParkedBuffer:
    \"\"\"gfx1151 patch 42: point ``param.data`` at a fresh host-mapped buffer
    without copying anything. The presharded reload then fills the mapping
    directly from its staged host buffer, so the data never enters VRAM.
    \"\"\"
    buf = HostParkedBuffer(param.numel() * param.element_size())
    device = (
        param.device
        if param.device.type == "cuda"
        else torch.device("cuda", torch.cuda.current_device())
    )
    alias = torch.as_tensor(buf, device=device)
    if alias.data_ptr() != buf.data_ptr():
        raise RuntimeError("host-mapped alias does not share the buffer address")
    alias = alias.view(param.dtype).view(param.shape)
    param.data = alias
    _PARKED[buf.data_ptr()] = weakref.ref(buf)
    return buf


def park_parameter_in_host_memory(param: torch.nn.Parameter) -> HostParkedBuffer:"""
assert text.count(old) == 1, "park_parameter anchor not found in host_parked_params"
text = text.replace(old, new)
open(p, "w").write(text)

# --- 2c. back to the loader: staged fill for parked params ----------------
p = f"{path}/python/sglang/srt/model_loader/loader.py"
text = open(p).read()

old = """        loaded_param_keys: set,
        verify_hashes: List[Tuple[str, str]],
    ) -> None:"""
new = """        loaded_param_keys: set,
        verify_hashes: List[Tuple[str, str]],
        copy_stage: Optional[torch.Tensor] = None,
    ) -> None:"""
assert text.count(old) == 1, "_apply_presharded_file signature anchor not found"
text = text.replace(old, new)

old = """            param_data.copy_(tensor)
            loaded_param_keys.add(r["name"])"""
new = """            # gfx1151 patch 42: route large copies through the fixed pinned
            # stage block; plain copy_ for tiny tensors and the None-stage
            # fallback.
            self._copy_to_gpu(param_data, tensor, copy_stage)
            loaded_param_keys.add(r["name"])"""
assert text.count(old) == 1, "_apply_presharded_file copy anchor not found"
text = text.replace(old, new)
open(p, "w").write(text)

print("patch_presharded_load_readahead: ok")
