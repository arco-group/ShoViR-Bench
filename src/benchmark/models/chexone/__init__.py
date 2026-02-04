from __future__ import annotations

import torch
from PIL import Image

from ..base import BaseHFLM
from ..spec import ModelSpec


class ChexOne(BaseHFLM):
    """CheXOne model using Qwen2_5_VL architecture with support for multiple images."""

    def _ensure_loaded(self):
        """Load model and processor without manual device placement (uses device_map='auto')."""
        if self._processor is None:
            from ..chat_template import load_processor
            self._processor = load_processor(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
            )
        if self._model is None:
            from ..chat_template import load_model
            self._model = load_model(
                self.__class__.__name__,
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                torch_dtype=self._torch_dtype,
            )
            # Don't call .to(device) when using device_map="auto"
            # The model handles device placement automatically
        return self._model, self._processor

    def build_chat_inputs(
        self,
        processor,
        image: Image.Image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        device: str | None = None,
        torch_dtype=None,
    ):
        """Build chat inputs following the same pattern as NV-Reason-CXR."""
        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self._torch_dtype

        # Build messages in standard format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{prompt_text}\n{user_text}" if user_text else prompt_text},
                ],
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt")

        # Compute input lengths
        input_lens = None
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            input_lens = inputs["attention_mask"].sum(dim=-1).to(torch.long)
        elif "input_ids" in inputs and inputs["input_ids"] is not None:
            input_lens = torch.full(
                (inputs["input_ids"].shape[0],),
                inputs["input_ids"].shape[-1],
                dtype=torch.long,
            )

        # Prepare inputs (move to device, convert dtype)
        from ..chat_template import _prepare_inputs
        inputs = _prepare_inputs(inputs, device=device, torch_dtype=torch_dtype)
        inputs.pop("num_crops", None)

        if input_lens is not None and input_lens.numel() == 1:
            input_lens = int(input_lens.item())
        return inputs, input_lens


MODEL_SPEC = ModelSpec(
    key="chexone",
    model_id="StanfordAIMI/CheXOne",
    prompt_key="chexone_default",
    task="image-to-text",
    generation_max_tokens=1024,
    caching=True,
    supports_images=True,
)

MODEL_CLASS = ChexOne
