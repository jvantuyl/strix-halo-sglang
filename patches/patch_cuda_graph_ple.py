#!/usr/bin/env python3
"""gfx1151 patch 14: CUDA graphs with the CPU-side PLE gather.

Patch 11 replaces the UVA gather of PLE n-gram rows with a host-side
``index_select`` on the mmap'd table (``_gather_ple_embedding_on_cpu``): it
copies the ids to the host, gathers the rows, and copies them back into the
prefetch buffer. None of that can be recorded into a CUDA graph, so Qwen3.8
has had to run with ``--disable-cuda-graph`` and pays the eager dispatch cost
of 48 layers on every decode step.

Upstream already gives every PLE layer a static per-batch-size prefetch
buffer under capture (``_graph_prefetch_buffers``) that the captured graph
reads. This patch fills that buffer from the host *before* each replay:

  * ``Qwen4ExpPinnedHostEmbedding.gather`` skips the host gather while the
    stream is capturing (the buffer is filled outside the graph), and the
    capture-mode buffer is zeroed on allocation so the warmup/capture passes
    do not read uninitialised memory.
  * ``Qwen4ExpPLELayer.fill_graph_prefetch_buffer`` recomputes the n-gram
    ids for the padded replay batch eagerly (the same GPU ops the graph
    records) and runs the host gather into the captured buffer.
  * ``Qwen4ExpModel.prepare_graph_replay`` builds the PLE batch for the
    replay view of the forward batch and fills every PLE layer;
    ``Qwen4ExpForConditionalGeneration`` forwards the call.
  * ``decode_cuda_graph_runner.py`` keeps the capture-time ``ForwardBatch``
    of every batch size (it wraps the static input buffers the graph reads,
    which ``load_batch`` refreshes) and calls the model's
    ``prepare_graph_replay`` with it between ``load_batch`` and
    ``backend.replay``.

The hook is a no-op with the UVA gather (``SGLANG_PLE_UVA_GATHER=1``) or when
the model has no PLE layer. Requires patch 11 (anchors on its gather branch).
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"

# ---------------------------------------------------------------------------
# qwen4_exp.py
# ---------------------------------------------------------------------------
p = f"{path}/python/sglang/srt/models/qwen4_exp.py"
text = open(p).read()

# 1. gather(): no host work while the stream is capturing.
old = """            if _STRIX_PLE_CPU_GATHER:
                _gather_ple_embedding_on_cpu(self, flat_ids, output)
            else:
"""
assert text.count(old) == 1, "qwen4_exp.py: CPU gather branch not found (patch 11 missing?)"
new = """            if _STRIX_PLE_CPU_GATHER:
                if torch.cuda.is_current_stream_capturing():
                    # The runner fills the static prefetch buffer from the host
                    # before each replay; nothing to record here.
                    if out is None:
                        raise RuntimeError(
                            "Qwen4 PLE CPU gather under CUDA graph capture needs "
                            "a prefetch buffer"
                        )
                else:
                    _gather_ple_embedding_on_cpu(self, flat_ids, output)
            else:
"""
text = text.replace(old, new, 1)

# 2. Capture-mode prefetch buffers start zeroed.
old = """        if get_is_capture_mode():
            buffer = self._graph_prefetch_buffers.get(lookup_tokens)
            if buffer is None:
                buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids)
                self._graph_prefetch_buffers[lookup_tokens] = buffer
            return buffer
"""
assert text.count(old) == 1, "qwen4_exp.py: capture prefetch buffer anchor not found"
new = """        if get_is_capture_mode():
            buffer = self._graph_prefetch_buffers.get(lookup_tokens)
            if buffer is None:
                buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids)
                if _STRIX_PLE_CPU_GATHER:
                    # Read by the capture pass before any host fill.
                    buffer.zero_()
                self._graph_prefetch_buffers[lookup_tokens] = buffer
            return buffer
"""
text = text.replace(old, new, 1)

# 3. Host fill of a captured buffer.
old = """    def _consume_prefetched_embeddings(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
"""
assert text.count(old) == 1, "qwen4_exp.py: _consume_prefetched_embeddings anchor not found"
new = """    def fill_graph_prefetch_buffer(
        self,
        batch: Optional[_PLEBatch],
        forward_batch: ForwardBatch,
    ) -> None:
        \"\"\"Host-gather PLE rows into the captured buffer for this batch size.\"\"\"
        if self._prefetch_stream is None or not _STRIX_PLE_CPU_GATHER:
            return
        if batch is None:
            if not self.ple_embedding.gather_dp_tokens:
                return
            physical_tokens = forward_batch.input_ids.numel()
            ngram_ids = forward_batch.input_ids.new_zeros(
                (physical_tokens, self.ple_embedding.ngram_heads)
            )
        else:
            physical_tokens = batch.physical_tokens
            ngram_ids = self.ple_embedding.compute_ngram_ids(batch)

        lookup_ids, _ = self.ple_embedding._prepare_embedding_lookup(
            ngram_ids, forward_batch, physical_tokens
        )
        lookup_tokens = lookup_ids.shape[0]
        if lookup_tokens == 0:
            return
        buffer = self._graph_prefetch_buffers.get(lookup_tokens)
        if buffer is None:
            raise RuntimeError(
                f"no captured PLE prefetch buffer for {lookup_tokens} tokens"
            )
        output_view = buffer.view(lookup_tokens, self.ple_embedding.ngram_heads, -1)
        self.ple_embedding.ngram_embedding.gather(lookup_ids, out=output_view)

    def _consume_prefetched_embeddings(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
"""
text = text.replace(old, new, 1)

# 4. Model-level hook, called by the decode graph runner before replay.
old = """class Qwen4ExpVLModel(Qwen4ExpModel):
"""
assert text.count(old) == 1, "qwen4_exp.py: Qwen4ExpVLModel anchor not found"
new = """    def prepare_graph_replay(self, forward_batch: Optional[ForwardBatch]) -> None:
        \"\"\"Fill the PLE prefetch buffers the captured graph is about to read.\"\"\"
        if not self.has_ple or not _STRIX_PLE_CPU_GATHER:
            return
        if forward_batch is None:
            raise RuntimeError(
                "Qwen4 PLE CPU gather needs the replay forward batch view"
            )
        ple_batch = _prepare_ple_batch(
            forward_batch.input_ids,
            forward_batch,
            ngram_size=self.ple_ngram_size,
            ngram_eos_token_id=self.ple_ngram_eos_token_id,
        )
        for i in range(self.start_layer, self.end_layer):
            ple = getattr(self.layers[i], "ple", None)
            if ple is not None:
                ple.fill_graph_prefetch_buffer(ple_batch, forward_batch)


class Qwen4ExpVLModel(Qwen4ExpModel):
"""
text = text.replace(old, new, 1)

old = """    @torch.no_grad()
    def forward(self, *args, **kwargs):
"""
assert text.count(old) == 1, "qwen4_exp.py: ForConditionalGeneration.forward anchor not found"
new = """    def prepare_graph_replay(self, forward_batch: Optional[ForwardBatch]) -> None:
        self.model.prepare_graph_replay(forward_batch)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)

# ---------------------------------------------------------------------------
# decode_cuda_graph_runner.py: model hook between load_batch and replay
# ---------------------------------------------------------------------------
p = f"{path}/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
text = open(p).read()

# The capture-time ForwardBatch wraps the static input buffers that fill_from
# refreshes on every replay, so it is exactly what the graph reads.
old = """        forward_batch, attn_backend, pp_proxy_tensors = self.capture_prepare(
            bs, stream_idx=stream_idx, num_tokens=num_tokens
        )
"""
assert text.count(old) == 1, "decode runner: capture_prepare call anchor not found"
new = old + """        if not hasattr(self, "_capture_forward_batches"):
            self._capture_forward_batches = {}
        self._capture_forward_batches[bs] = forward_batch
"""
text = text.replace(old, new, 1)

old = """            if shared_read_ends is SharedReadEnds.PRE_REPLAY:
                self._publish_read_done(in_graph=False)

            output = self.backend.replay(self._replay_graph_key, forward_batch)
"""
assert text.count(old) == 1, "decode runner: replay call anchor not found"
new = """            if shared_read_ends is SharedReadEnds.PRE_REPLAY:
                self._publish_read_done(in_graph=False)

            # Models whose graphs read buffers filled from the host (Qwen4 PLE
            # with the CPU gather) refresh them here, after the static input
            # buffers hold this batch and before the graph runs.
            prepare_graph_replay = getattr(
                self.model_runner.model, "prepare_graph_replay", None
            )
            if prepare_graph_replay is not None:
                prepare_graph_replay(
                    getattr(self, "_capture_forward_batches", {}).get(self.bs)
                )

            output = self.backend.replay(self._replay_graph_key, forward_batch)
"""
text = text.replace(old, new, 1)
open(p, "w").write(text)
print("patched", p)
print("patch 14 (CUDA graphs with CPU PLE gather) applied")
