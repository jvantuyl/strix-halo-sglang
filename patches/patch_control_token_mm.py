#!/usr/bin/env python3
"""Requests with images or video keep the split-encoded prompt ids (patch 34).

Patch 32 encodes the chat prompt so that only the template's control tokens
become control token ids. A request that carries media could not use those
ids: the Qwen-VL multimodal processor takes the rendered text, expands every
image or video placeholder into the right number of pad tokens (with
per-frame "<t seconds>" text for video) and tokenizes the whole prompt with
the added tokens enabled, so a quoted "<|im_end|>" in such a request was a
control token again. Handing it the ids did not help either: the processor
decoded them back to text first.

Now the processor accepts the prompt ids. It derives a placeholder-only text
from them (one "<|vision_start|><|image_pad|><|vision_end|>" or video triple
per medium, decoded from the ids themselves), runs the unchanged media
pipeline on that text (image/video preprocessing, pixel values, grids,
timestamps, feature hashing), and then rebuilds the final ids by expanding
the placeholders inside the *original* prompt ids with SGLang's own id-level
expansion (build_input_ids / build_input_ids_with_timestamps). Before the
rebuilt ids are used, the same expansion is run on the placeholder-only ids
and compared with what the processor produced for the placeholder-only text;
they have to agree token for token. The media items' offsets are remapped
onto the rebuilt ids. If anything disagrees, the request falls back to the
text path (today's behaviour) with a warning instead of failing.

serving_chat.py sends the ids with media when the processor declares
accepts_prompt_ids and the ids hold exactly one placeholder per image and
per video; otherwise the text path and its warning remain (audio, or a
template that does not place one placeholder per medium). Anchors are
asserted so an upstream rewrite fails the build.
See patches/34-control-token-media.md.
"""
import sys


def patch(text, old, new, count=1):
    assert text.count(old) == count, (text.count(old), old[:120])
    return text.replace(old, new)


def patch_serving_chat(path):
    text = open(path).read()
    text = patch(
        text,
        "            and not (\n"
        "                processed_messages.image_data\n"
        "                or processed_messages.video_data\n"
        "                or processed_messages.audio_data\n"
        "            )\n"
        "        )\n",
        "            and not (\n"
        "                processed_messages.image_data\n"
        "                or processed_messages.video_data\n"
        "                or processed_messages.audio_data\n"
        "            )\n"
        "        )\n"
        "        # gfx1151 patch 34: with the split encoder the ids already hold one\n"
        "        # placeholder control token per medium, and the MM processor expands\n"
        "        # them in place, so the text is not re-tokenized.\n"
        "        if (\n"
        "            not prefers_prompt_ids\n"
        "            and self._control_token_encoder is not None\n"
        "            and isinstance(processed_messages.prompt_ids, list)\n"
        "            and processed_messages.prompt_ids\n"
        "            and self._media_placeholders_match(processed_messages)\n"
        "        ):\n"
        "            prefers_prompt_ids = True\n",
    )
    text = patch(
        text,
        "                logger.warning(\n"
        "                    \"Request carries media; the prompt text is re-tokenized \"\n"
        "                    \"and quoted control strings are not kept apart\"\n"
        "                )\n",
        "                logger.warning(\n"
        "                    \"Request carries media the prompt ids cannot account for \"\n"
        "                    \"(audio, or placeholders not matching the media count); the \"\n"
        "                    \"prompt text is re-tokenized and quoted control strings are \"\n"
        "                    \"not kept apart\"\n"
        "                )\n",
    )
    text = patch(
        text,
        "    def _convert_to_internal_request(\n",
        "    def _media_placeholders_match(\n"
        "        self, processed_messages: MessageProcessingResult\n"
        "    ) -> bool:\n"
        "        \"\"\"gfx1151 patch 34: True when the MM processor expands prompt ids in\n"
        "        place and the ids hold exactly one placeholder per image and per\n"
        "        video (a vision-start token followed by the pad token).\"\"\"\n"
        "        if processed_messages.audio_data:\n"
        "            return False\n"
        "        if not getattr(self.tokenizer_manager.mm_processor, \"accepts_prompt_ids\", False):\n"
        "            return False\n"
        "        hf_config = self.tokenizer_manager.model_config.hf_config\n"
        "        start_id = getattr(hf_config, \"vision_start_token_id\", None)\n"
        "        image_id = getattr(hf_config, \"image_token_id\", None)\n"
        "        video_id = getattr(hf_config, \"video_token_id\", None)\n"
        "        if start_id is None:\n"
        "            return False\n"
        "        ids = processed_messages.prompt_ids\n"
        "        n_image = n_video = 0\n"
        "        for a, b in zip(ids, ids[1:]):\n"
        "            if a != start_id:\n"
        "                continue\n"
        "            if image_id is not None and b == image_id:\n"
        "                n_image += 1\n"
        "            elif video_id is not None and b == video_id:\n"
        "                n_video += 1\n"
        "        return n_image == len(processed_messages.image_data or []) and n_video == len(\n"
        "            processed_messages.video_data or []\n"
        "        )\n"
        "\n"
        "    def _convert_to_internal_request(\n",
    )
    open(path, "w").write(text)


def patch_qwen_vl(path):
    text = open(path).read()
    text = patch(
        text,
        "    supports_transformers_backend = True\n"
        "    generates_input_ids_from_raw_prompt = True\n"
        "    artifact_modality = Modality.IMAGE\n",
        "    supports_transformers_backend = True\n"
        "    generates_input_ids_from_raw_prompt = True\n"
        "    artifact_modality = Modality.IMAGE\n"
        "    # gfx1151 patch 34: a list of prompt ids is expanded in place, see\n"
        "    # _placeholder_prompt_for_ids / _rebuild_from_prompt_ids.\n"
        "    accepts_prompt_ids = True\n",
    )
    # ids bypass the artifact cache path (which composes on text)
    text = patch(
        text,
        "        if (\n"
        "            not image_data\n"
        "            or request_obj.video_data\n"
        "            or request_obj.audio_data\n"
        "            or any(self._is_preprocessed_input(item) for item in image_data)\n",
        "        if (\n"
        "            isinstance(input_text, list)  # gfx1151 patch 34: prompt ids\n"
        "            or not image_data\n"
        "            or request_obj.video_data\n"
        "            or request_obj.audio_data\n"
        "            or any(self._is_preprocessed_input(item) for item in image_data)\n",
    )
    text = patch(
        text,
        "        entry_time = time.perf_counter()\n"
        "        base_output = await self.load_mm_data(\n"
        "            prompt=input_text,\n",
        "        entry_time = time.perf_counter()\n"
        "        # gfx1151 patch 34: prompt ids are expanded in place; the media\n"
        "        # pipeline runs on a placeholder-only text derived from them.\n"
        "        prompt_ids = input_text if isinstance(input_text, list) else None\n"
        "        if prompt_ids is not None:\n"
        "            input_text = self._placeholder_prompt_for_ids(prompt_ids)\n"
        "            if input_text is None:\n"
        "                return await self._process_mm_data_uncached(\n"
        "                    image_data,\n"
        "                    self._fall_back_to_text(prompt_ids, \"no media placeholder in the ids\"),\n"
        "                    request_obj,\n"
        "                    *args,\n"
        "                    **kwargs,\n"
        "                )\n"
        "        base_output = await self.load_mm_data(\n"
        "            prompt=input_text,\n",
    )
    text = patch(
        text,
        "        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(\n"
        "            base_output, self.mm_tokens, **processor_kwargs\n"
        "        )\n"
        "\n"
        "        self._mark_cuda_ipc_features_for_deferred_reconstruction(mm_items)\n",
        "        mm_items, input_ids, ret = await self.process_and_combine_mm_data_async(\n"
        "            base_output, self.mm_tokens, **processor_kwargs\n"
        "        )\n"
        "\n"
        "        if prompt_ids is not None:  # gfx1151 patch 34\n"
        "            rebuilt = self._rebuild_from_prompt_ids(\n"
        "                prompt_ids, mm_items, input_ids, ret, image_data, request_obj, video_metadata\n"
        "            )\n"
        "            if rebuilt is None:\n"
        "                return await self._process_mm_data_uncached(\n"
        "                    image_data,\n"
        "                    self._fall_back_to_text(\n"
        "                        prompt_ids, \"id-level expansion disagreed with the processor\"\n"
        "                    ),\n"
        "                    request_obj,\n"
        "                    *args,\n"
        "                    **kwargs,\n"
        "                )\n"
        "            mm_items, input_ids, ret = rebuilt\n"
        "\n"
        "        self._mark_cuda_ipc_features_for_deferred_reconstruction(mm_items)\n",
    )
    text = patch(
        text,
        "    def _mark_cuda_ipc_features_for_deferred_reconstruction(self, mm_items):\n",
        '''    # gfx1151 patch 34 -------------------------------------------------------
    _TIMESTAMP_MODEL_TYPES = (
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen4_exp",
        "intern_s2_preview",
        "cosmos3_omni",
    )

    def _placeholder_triples(self, prompt_ids):
        """The (start, pad, end) placeholder triples in prompt ids, in order."""
        start_id = self.vision_start_token_id
        pads = {self.IM_TOKEN_ID, self.VIDEO_TOKEN_ID}
        triples = []
        for i in range(len(prompt_ids) - 2):
            if prompt_ids[i] == start_id and prompt_ids[i + 1] in pads:
                triples.append(tuple(prompt_ids[i : i + 3]))
        return triples

    def _placeholder_prompt_for_ids(self, prompt_ids):
        """A text holding only the media placeholders the ids hold, decoded
        from the ids themselves, or None when there is none."""
        triples = self._placeholder_triples(prompt_ids)
        if not triples:
            return None
        return "".join(
            self._processor.tokenizer.decode(list(t), skip_special_tokens=False)
            for t in triples
        )

    def _fall_back_to_text(self, prompt_ids, why):
        logger.warning(
            "Prompt ids with media: %s; re-tokenizing the decoded prompt text "
            "(quoted control strings are not kept apart for this request).",
            why,
        )
        return self._processor.tokenizer.decode(prompt_ids, skip_special_tokens=False)

    def _expand_prompt_ids(self, prompt_ids, image_grid_thw, video_grid_thw, video_timestamps):
        if video_timestamps is not None and self.model_type in self._TIMESTAMP_MODEL_TYPES:
            ids, _, _ = self.build_input_ids_with_timestamps(
                list(prompt_ids), None, image_grid_thw, video_grid_thw, video_timestamps
            )
        else:
            ids, _, _ = self.build_input_ids(list(prompt_ids), image_grid_thw, video_grid_thw)
        return [int(t) for t in ids]

    def _video_timestamps(self, video_metadata, n_videos):
        """Per-video frame timestamps, as the HF processor computes them."""
        if not n_videos:
            return None
        if not video_metadata or len(video_metadata) != n_videos:
            return None
        calc = getattr(self._processor, "_calculate_timestamps", None)
        video_processor = getattr(self._processor, "video_processor", None)
        if calc is None or video_processor is None:
            return None
        timestamps = []
        for meta in video_metadata:
            if not isinstance(meta, dict):
                meta = {
                    "fps": getattr(meta, "fps", None),
                    "frames_indices": getattr(meta, "frames_indices", None),
                }
            fps = meta.get("fps")
            indices = meta.get("frames_indices")
            if indices is None:
                return None
            if fps is None:
                fps = 24
            timestamps.append(
                calc(list(indices), fps, video_processor.temporal_patch_size)
            )
        return timestamps

    def _rebuild_from_prompt_ids(
        self, prompt_ids, mm_items, input_ids, ret, image_data, request_obj, video_metadata
    ):
        """Expand the placeholders inside the original prompt ids, after checking
        that the same expansion reproduces the processor's ids for the
        placeholder-only text; remap the items' offsets. None on disagreement."""
        synthetic_ids = [int(t) for t in input_ids.flatten().tolist()]
        triples = self._placeholder_triples(prompt_ids)
        placeholder_ids = [t for triple in triples for t in triple]
        image_grid_thw = self._get_grid_from_output_or_items(
            ret, mm_items, "image_grid_thw", Modality.IMAGE, image_data
        )
        video_grid_thw = self._get_grid_from_output_or_items(
            ret, mm_items, "video_grid_thw", Modality.VIDEO, request_obj.video_data
        )
        n_videos = sum(1 for t in triples if t[1] == self.VIDEO_TOKEN_ID)
        video_timestamps = self._video_timestamps(video_metadata, n_videos)
        if n_videos and video_timestamps is None and self.model_type in self._TIMESTAMP_MODEL_TYPES:
            return None
        try:
            check = self._expand_prompt_ids(
                placeholder_ids, image_grid_thw, video_grid_thw, video_timestamps
            )
            if check != synthetic_ids:
                return None
            real_ids = self._expand_prompt_ids(
                prompt_ids, image_grid_thw, video_grid_thw, video_timestamps
            )
        except Exception as exc:  # a grid or timestamp shape the builder rejects
            logger.warning("Prompt ids with media: id-level expansion failed: %s", exc)
            return None

        # The k-th run of a pad token in the synthetic ids is the k-th run of
        # that token in the rebuilt ids; items carry runs, remap them.
        synthetic_t = torch.tensor(synthetic_ids, dtype=torch.long)
        real_t = torch.tensor(real_ids, dtype=input_ids.dtype, device=input_ids.device)
        remap = {}
        for pad_id in {self.IM_TOKEN_ID, self.VIDEO_TOKEN_ID}:
            if pad_id is None:
                continue
            old_runs = self.get_mm_items_offset(synthetic_t, pad_id)
            new_runs = self.get_mm_items_offset(real_t.cpu(), pad_id)
            if len(old_runs) != len(new_runs):
                return None
            for old, new in zip(old_runs, new_runs):
                if old[1] - old[0] != new[1] - new[0]:
                    return None
                remap[(pad_id, old)] = new
        for item in mm_items:
            if not item.offsets:
                continue
            pad_id = self.mm_tokens.get_token_id_by_modality(item.modality)
            try:
                item.offsets = [remap[(pad_id, tuple(run))] for run in item.offsets]
            except KeyError:
                return None
        for key in ("padded_input_ids", "mrope_positions", "mrope_position_delta"):
            if hasattr(ret, "pop") and key in ret:
                ret.pop(key)
        return mm_items, real_t, ret

    def _mark_cuda_ipc_features_for_deferred_reconstruction(self, mm_items):
''',
    )
    open(path, "w").write(text)


def main():
    tree = sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang"
    srt = f"{tree}/python/sglang/srt"
    patch_serving_chat(f"{srt}/entrypoints/openai/serving_chat.py")
    patch_qwen_vl(f"{srt}/multimodal/processors/qwen_vl.py")
    print("patched serving_chat.py and multimodal/processors/qwen_vl.py")


if __name__ == "__main__":
    main()
