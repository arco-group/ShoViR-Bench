from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from huggingface_hub import snapshot_download

from ..base import BaseHFLM
from ..spec import ModelSpec


class LLaVARad(BaseHFLM):
    """
    Reference-like wrapper for:
      - X-iZhang/libra-llava-rad

    Mirrors upstream libra_eval style:
      - snapshot_download -> load_pretrained_model
      - build qs with DEFAULT_IMAGE_TOKEN (+ start/end if needed)
      - conv_templates[conv_mode]
      - tokenizer_image_token
      - TAC: two-image tensor (current + prior, dummy prior if missing)
      - stop_str + KeywordsStoppingCriteria
      - generate + decode + strip stop_str
    """

    # In this repo, the weights are typically complete -> model_base not needed
    model_base: str | None = None

    # Name passed to the builder (used to pick code paths)
    model_name: str = "libra-llava-rad"

    # Choose the right conversation template for llava-style models
    conv_mode: str = "v1"

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

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        if self.image_preprocess is not None:
            return self.image_preprocess(image)
        return image.convert("RGB")

    def _prepare_images(self, cur: Image.Image, prior: Image.Image | None) -> torch.Tensor:
        """
        Upstream get_image_tensors behavior:
        - Always build two slots: [current, prior]
        - If prior is missing, duplicate current as dummy prior
        - Return tensor shaped [2, 1, C, H, W]
        """
        from libra.mm_utils import process_images

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

        # Preprocess images
        cur = self.preprocess_image(image)
        prior = self.preprocess_image(prior_image) if prior_image is not None else None

        # Build qs exactly like upstream
        qs = (user_text or "").strip()
        if prompt_text:
            qs = f"{prompt_text.strip()}\n\n{qs}"

        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        # Build prompt via the llava-style conversation template
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize prompt with image token insertion
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        # Upstream-style attention mask
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        # Prepare images (handles TAC vs non-TAC)
        image_tensor = self._prepare_images(cur, prior)

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
                use_cache=True,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:],
            skip_special_tokens=True,
        )[0].strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        return [{"generated_text": outputs}]


MODEL_SPEC = ModelSpec(
    key="llava_rad",
    model_id="X-iZhang/libra-llava-rad",
    prompt_key="llava_rad_default",
    task="image-to-text",
    generation_max_tokens=300,
    caching=True,
    supports_images=True,
)

MODEL_CLASS = LLaVARad
