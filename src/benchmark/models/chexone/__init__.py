from __future__ import annotations

import torch
import torch.nn.functional as F
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
            if hasattr(self._processor, "tokenizer") and self._processor.tokenizer is not None:
                # Recommended for decoder-only generation
                self._processor.tokenizer.padding_side = "left"
                if self._processor.tokenizer.pad_token is None:
                    self._processor.tokenizer.pad_token = self._processor.tokenizer.eos_token

        if self._model is None:
            from ..chat_template import load_model
            self._model = load_model(
                self.__class__.__name__,
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                torch_dtype=self._torch_dtype,
            )

            if getattr(self._model, "generation_config", None) is not None:
                tok = getattr(self._processor, "tokenizer", None)
                if tok is not None:
                    self._model.generation_config.pad_token_id = tok.pad_token_id

        return self._model, self._processor

    def build_chat_messages(
        self,
        prompt_text: str,
        image: Image.Image,
        user_text: str = "",
    ) -> list[dict[str, list[dict[str, object]]]]:

        text = prompt_text if not user_text else f"{prompt_text}\n{user_text}"
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def build_chat_inputs(
        self,
        processor,
        image: Image.Image,
        prompt_text: str,
        *,
        user_text: str = "",
        device: str | None = None,
        torch_dtype=None,
    ):

        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self._torch_dtype

        # Ensure tokenizer is configured for decoder-only generation
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token

        messages = self.build_chat_messages(prompt_text, image, user_text)
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt")

        input_lens = None
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            input_lens = inputs["attention_mask"].sum(dim=-1).to(torch.long)
        elif "input_ids" in inputs and inputs["input_ids"] is not None:
            input_lens = torch.full(
                (inputs["input_ids"].shape[0],),
                inputs["input_ids"].shape[-1],
                dtype=torch.long,
            )

        from ..chat_template import _prepare_inputs
        inputs = _prepare_inputs(inputs, device=device, torch_dtype=torch_dtype)
        inputs.pop("num_crops", None)

        if input_lens is not None and input_lens.numel() == 1:
            input_lens = int(input_lens.item())

        return inputs, input_lens

    def __call__(
        self,
        image: Image.Image | list[Image.Image],
        prompt_text: str,
        *,
        user_text: str = "",
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        """Run inference on one image or a batch of independent conversations."""
        if isinstance(image, list):
            return self._batch_call(
                image, prompt_text, user_text=user_text, drop_config=drop_config
            )

        model, processor, inputs, input_len = self.prepare_inputs(
            image,
            prompt_text,
            user_text=user_text,
            drop_config=drop_config,
        )
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.generation_max_tokens,
                do_sample=False,
                use_cache=self.caching,
            )
        output_ids = output_ids[:, input_len:]

        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        return [{"generated_text": text} for text in decoded]

    def _batch_call(
        self,
        images: list[Image.Image],
        prompt_text: str,
        *,
        user_text: str = "",
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        model, processor = self._ensure_loaded()

        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

        texts: list[str] = []
        for img in images:
            messages = self.build_chat_messages(prompt_text, img, user_text)
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text)

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        from ..chat_template import _prepare_inputs
        device = self.device
        if device is None:
            device = next(model.parameters()).device
        inputs = _prepare_inputs(inputs, device=device, torch_dtype=self._torch_dtype)
        inputs.pop("num_crops", None)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.generation_max_tokens,
                do_sample=False,
                use_cache=self.caching,
            )

        max_len = inputs["input_ids"].shape[1]
        gen_only = output_ids[:, max_len:]

        decoded = processor.batch_decode(gen_only, skip_special_tokens=True)
        results: list[dict[str, str]] = [{"generated_text": t} for t in decoded]
        return results
        

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
