from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers.generation.stopping_criteria import StoppingCriteria

from ..base import BaseHFLM
from ..spec import ModelSpec


class BatchedStoppingCriteria(StoppingCriteria):
    """
    Batch-safe stopping criteria wrapper.

    Upstream Libra/LLaVA stopping utilities are commonly written for a single
    sequence and expect input_ids shaped [1, L]. This wrapper applies a
    per-sample stopping criteria to each row and stops generation only when
    ALL samples have reached their stop condition.
    """

    def __init__(self, criterias: List[StoppingCriteria]):
        super().__init__()
        self.criterias = criterias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        all_done = True
        for i, c in enumerate(self.criterias):
            if not c(input_ids[i : i + 1], scores, **kwargs):
                all_done = False
        return all_done


class LLaVARad(BaseHFLM):
    """
    Wrapper for:
      - X-iZhang/libra-llava-rad

    This wrapper supports:
      - single-image inference (current + optional prior)
      - batched inference (N independent conversations in one generate() call)

    Implementation mirrors BaseHFLM style:
      - __call__ delegates to _batch_call when input is a list
      - single-image path prepares inputs, runs generate(), slices generated tokens
    """

    # In this repo, weights are typically complete -> model_base not needed
    model_base: str | None = None

    # Name passed to the builder (used to pick code paths)
    model_name: str = "libra-llava-rad"

    # LLaVA-style conversation template
    conv_mode: str = "v1"

    def _ensure_loaded(self):
        """
        Load (tokenizer, model, image_processor) via Libra upstream builder.
        Cache objects on the instance.
        """
        if getattr(self, "_model", None) is not None and getattr(self, "_tokenizer", None) is not None:
            return self._model, self._tokenizer

        from libra.model.builder import load_pretrained_model

        model_path = snapshot_download(
            repo_id=self.model_id,
            revision="main",
            cache_dir=self.cache_dir,
        )
        model_path = Path(model_path)

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            self.model_base,
            self.model_name,
        )

        # Ensure pad_token_id exists (common upstream behavior)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if self.device is not None:
            model = model.to(self.device)

        self._model = model
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        return self._model, self._tokenizer

    # ------------------------------------------------------------------
    # Prompt + image preparation helpers
    # ------------------------------------------------------------------
    def _build_single_prompt(self, model, prompt_text: str, user_text: str) -> tuple[str, str]:
        """
        Build a single LLaVA-style prompt and return (prompt, stop_str).
        """
        from libra.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from libra.conversation import conv_templates, SeparatorStyle

        qs = (user_text or "").strip()
        if prompt_text:
            qs = f"{prompt_text.strip()}\n\n{qs}" if qs else prompt_text.strip()

        if getattr(model.config, "mm_use_im_start_end", False):
            qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
        else:
            qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        stop_str = conv.sep if conv.sep_style not in {
            SeparatorStyle.TWO,
            SeparatorStyle.LLAMA_3,
            SeparatorStyle.MISTRAL,
        } else conv.sep2

        return prompt, stop_str

    def _prepare_images_batch(
        self,
        images: list[Image.Image],
        priors: Optional[list[Optional[Image.Image]]] = None,
    ) -> torch.Tensor:
        """
        Prepare a batch of (current, prior) image pairs.

        This model wrapper follows the upstream "two-slot" behavior:
          - always build [current, prior] slots
          - if prior is missing, duplicate current

        Returns:
          - Tensor shaped [2, B, C, H, W]
        """
        from libra.mm_utils import process_images

        model = self._model

        if priors is None:
            priors = [None] * len(images)

        prior_imgs = [p if p is not None else im for im, p in zip(images, priors)]

        # process_images returns batch tensor [B, C, H, W] for list input
        cur_t = process_images(images, self._image_processor, model.config)
        prior_t = process_images(prior_imgs, self._image_processor, model.config)

        cur_t = cur_t.to(device=model.device, non_blocking=True)
        prior_t = prior_t.to(device=model.device, non_blocking=True)

        if self._torch_dtype is not None and torch.is_floating_point(cur_t):
            cur_t = cur_t.to(dtype=self._torch_dtype)
            prior_t = prior_t.to(dtype=self._torch_dtype)

        return torch.stack([cur_t, prior_t], dim=0)  # [2, B, C, H, W]

    # ------------------------------------------------------------------
    # Single-image path (BaseHFLM-like)
    # ------------------------------------------------------------------
    def prepare_inputs(
        self,
        image: Image.Image,
        prompt_text: str,
        *,
        user_text: str = "",
        prior_image: Image.Image | None = None,
    ):
        """
        Prepare single-image inputs and return:
          (model, tokenizer, inputs_dict, input_len, stop_str)
        """
        from libra.constants import IMAGE_TOKEN_INDEX
        from libra.mm_utils import tokenizer_image_token

        model, tokenizer = self._ensure_loaded()

        prompt, stop_str = self._build_single_prompt(model, prompt_text, user_text)

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        # Ensure pad token exists and build attention mask
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        images_t = self._prepare_images_batch([image], priors=[prior_image])

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images_t,
        }
        input_len = int(input_ids.shape[1])
        return model, tokenizer, inputs, input_len, stop_str

    def _single_image_call(
        self,
        image: Image.Image,
        prompt_text: str,
        *,
        user_text: str = "",
        prior_image: Image.Image | None = None,
    ) -> dict[str, str]:
        """
        Run single-image inference and return {"generated_text": ...}.
        """
        from libra.mm_utils import KeywordsStoppingCriteria

        model, tokenizer, inputs, input_len, stop_str = self.prepare_inputs(
            image, prompt_text, user_text=user_text, prior_image=prior_image
        )

        stopping = KeywordsStoppingCriteria([stop_str], tokenizer, inputs["input_ids"])

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                images=inputs["images"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
                max_new_tokens=self.generation_max_tokens,
                stopping_criteria=[stopping],
                use_cache=self.caching,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen = output_ids[:, input_len:]
        text = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

        # Extra safeguard: strip stop string if present.
        if stop_str and text.endswith(stop_str):
            text = text[:-len(stop_str)].strip()

        return {"generated_text": text}

    # ------------------------------------------------------------------
    # Batched inference: N independent conversations, one generate() call
    # ------------------------------------------------------------------
    def _batch_call(
        self,
        images: list[Image.Image],
        prompt_text: str,
        *,
        user_text: str = "",
        prior_images: Optional[list[Optional[Image.Image]]] = None,
    ) -> list[dict[str, str]]:
        """
        Batch N independent conversations into a single generate() call.

        This mirrors BaseHFLM._batch_call:
          - build N prompts independently
          - pad into a batch (left padding for consistency with BaseHFLM._manual_batch)
          - run a single model.generate()
          - slice / decode per sample
        """
        from libra.constants import IMAGE_TOKEN_INDEX
        from libra.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

        model, tokenizer = self._ensure_loaded()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        if prior_images is None:
            prior_images = [None] * len(images)

        # 1) Build prompts and tokenize (un-padded)
        ids_list: list[torch.Tensor] = []
        lens: list[int] = []
        stop_strs: list[str] = []

        for _img in images:
            prompt, stop_str = self._build_single_prompt(model, prompt_text, user_text)
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")  # [L]
            ids_list.append(ids)
            lens.append(int(ids.shape[0]))
            stop_strs.append(stop_str)

        # 2) Left-pad to max length (consistent with BaseHFLM._manual_batch)
        max_len = max(lens)
        batch_ids = []
        batch_attn = []

        for ids in ids_list:
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                ids_padded = torch.nn.functional.pad(ids, (pad_len, 0), value=pad_id)  # left pad
                attn = torch.nn.functional.pad(torch.ones_like(ids), (pad_len, 0), value=0)
            else:
                ids_padded = ids
                attn = torch.ones_like(ids)

            batch_ids.append(ids_padded)
            batch_attn.append(attn)

        input_ids = torch.stack(batch_ids, dim=0).to(model.device)              # [B, Lmax]
        attention_mask = torch.stack(batch_attn, dim=0).to(model.device).bool() # [B, Lmax]

        # 3) Build per-sample stopping criteria and wrap them into a batched one
        per_sample = []
        for i in range(len(images)):
            per_sample.append(
                KeywordsStoppingCriteria([stop_strs[i]], tokenizer, input_ids[i : i + 1])
            )
        stopping = BatchedStoppingCriteria(per_sample)

        # 4) Prepare images batch (always two slots)
        images_t = self._prepare_images_batch(images, priors=prior_images)

        # 5) Single generate() call
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=images_t,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=self.generation_max_tokens,
                stopping_criteria=[stopping],
                use_cache=self.caching,
                pad_token_id=pad_id,
            )

        # 6) Decode per sample
        # With left padding, all prompts occupy the first max_len tokens.
        results: list[dict[str, str]] = []
        for i in range(output_ids.shape[0]):
            gen = output_ids[i, max_len:]
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()

            stop_str = stop_strs[i]
            if stop_str and text.endswith(stop_str):
                text = text[:-len(stop_str)].strip()

            results.append({"generated_text": text})

        return results

    def __call__(
        self,
        image: Image.Image | list[Image.Image],
        prompt_text: str,
        *,
        user_text: str = "",
        prior_image: Image.Image | None = None,
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        """
        Run inference on one image or a batch of independent conversations.

        - If image is a list: run a single batched generate() call for efficiency.
        - If image is a single PIL.Image: run single-image path.

        Note:
          drop_config is ignored in this wrapper.
        """
        if isinstance(image, list):
            return self._batch_call(image, prompt_text, user_text=user_text)

        return [self._single_image_call(image, prompt_text, user_text=user_text, prior_image=prior_image)]


MODEL_SPEC = ModelSpec(
    key="llavarad",
    model_id="X-iZhang/libra-llava-rad",
    prompt_key="llavarad_default",
    task="image-to-text",
    generation_max_tokens=300,
    caching=True,
    supports_images=True,
)

MODEL_CLASS = LLaVARad
