from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from huggingface_hub import snapshot_download

from ..base import BaseHFLM
from ..spec import ModelSpec


class Libra(BaseHFLM):
    """
    Wrapper for:
      - X-iZhang/libra-v1.0-3b
    """

    # Libra v1 models typically do not require a separate base model
    model_base: str | None = None

    # Model name string used by the Libra builder
    model_name: str = "libra-v1.0-3b"

    # Upstream selects this for libra-v1.0-3b
    #CHANGE with prompt
    conv_mode: str = "libra_llama_3"

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model, self._tokenizer

        from libra.model.builder import load_pretrained_model

        # Download model snapshot locally
        model_path = snapshot_download(
            repo_id=self.model_id,
            revision="main",
            cache_dir=self.cache_dir,
        )
        model_path = Path(model_path)

        # Load model components
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            self.model_base,
            self.model_name,
        )

        # Ensure pad_token_id exists (upstream behavior)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if self.device is not None:
            model = model.to(self.device)

        self._model = model
        self._tokenizer = tokenizer
        self._image_processor = image_processor
        return self._model, self._tokenizer

    def _build_prompt(self, query: str) -> str:
        from libra.conversation import conv_templates

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _prepare_images(self, cur: Image.Image, prior: Image.Image | None) -> torch.Tensor:
        """
        Upstream behavior:
          - If a temporal projector (TAC) is used, the model expects TWO images:
            current + prior. If prior is missing, duplicate current.
          - Otherwise, use a single image.

        Returns:
          - Non-TAC: Tensor [1, C, H, W]
          - TAC:     Tensor [2, 1, C, H, W]  (same stacking style as upstream get_image_tensors)
        """
        from libra.mm_utils import process_images

        projector = getattr(self._model.config, "mm_projector_type", None)
        is_tac = projector == "TAC"

        if not is_tac:
            images = process_images([cur], self._image_processor, self._model.config)
            if isinstance(images, torch.Tensor) and images.ndim == 3:
                images = images.unsqueeze(0)
            return images

        # TAC path: ensure prior exists (dummy prior = current)
        prior = prior if prior is not None else cur

        cur_t = process_images([cur], self._image_processor, self._model.config)[0]      # [C,H,W]
        prior_t = process_images([prior], self._image_processor, self._model.config)[0]  # [C,H,W]

        cur_t = cur_t.to(device=self._model.device, non_blocking=True)
        prior_t = prior_t.to(device=self._model.device, non_blocking=True)

        if self._torch_dtype is not None and torch.is_floating_point(cur_t):
            cur_t = cur_t.to(dtype=self._torch_dtype)
            prior_t = prior_t.to(dtype=self._torch_dtype)

        return torch.stack([cur_t.unsqueeze(0), prior_t.unsqueeze(0)], dim=0)  # [2,1,C,H,W]

    def __call__(
        self,
        image: Image.Image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        prior_image: Image.Image | None = None,
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        from libra.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from libra.conversation import conv_templates, SeparatorStyle
        from libra.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

        model, tokenizer = self._ensure_loaded()
        cur=image
        prior = prior_image if prior_image is not None else None

        # Build qs like upstream
        qs = (user_text or "").strip()
        if prompt_text:
            qs = f"{prompt_text.strip()}\n\n{qs}"

        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        # Build prompt via conv template
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize (with image token)
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        # Upstream-style attention mask
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        # Prepare images (IMPORTANT: mimic upstream "2-slot" behavior if you want)
        image_tensor = self._prepare_images(cur, prior)  # whatever implementation you decided

        # Stopping criteria like upstream
        stop_str = conv.sep if conv.sep_style not in {
            SeparatorStyle.TWO,
            SeparatorStyle.LLAMA_3,
            SeparatorStyle.MISTRAL,
        } else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=self.generation_max_tokens,
                stopping_criteria=[stopping_criteria],
                use_cache=self.caching,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

        # Strip trailing stop string if present
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        return [{"generated_text": outputs}]



MODEL_SPEC = ModelSpec(
    key="libra",
    model_id="X-iZhang/libra-v1.0-3b",
    prompt_key="libra_default",
    task="image-to-text",
    generation_max_tokens=300,
    caching=True,
    supports_images=True,
)

MODEL_CLASS = Libra
