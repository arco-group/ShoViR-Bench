import os
import tempfile
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import BaseHFLM
from ..spec import ModelSpec


class CheXagent(BaseHFLM):

    def _ensure_loaded(self):
        if self._processor is None:
            self._processor = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
            )

        if self._model is None:
            model_kwargs = dict(
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
            )
            if self._torch_dtype is not None:
                model_kwargs["torch_dtype"] = self._torch_dtype

            if self.device is None:
                model_kwargs["device_map"] = "auto"

            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

            # If a specific device is requested, move the model there
            # and ensure all parameters are in the target dtype
            if self.device is not None:
                self._model.to(self.device)
                if self._torch_dtype is not None:
                    self._model.to(self._torch_dtype)

            self._model.eval()

        return self._model, self._processor  # (model, tokenizer)

    def _image_to_local_path(self, image) -> str:
        # CheXagent via tokenizer.from_list_format vuole un PATH su disco.
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image path not found: {image}")
            return image

        if isinstance(image, Image.Image):
            fd, path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            image.save(path, format="JPEG")
            return path

        raise TypeError("image must be a file path (str) or a PIL.Image.Image")

    # ---------------------------------------------------------
    # Build inputs exactly like the HF snippet
    # ---------------------------------------------------------
    def build_chat_inputs(
        self,
        processor,  # qui è il tokenizer HF
        image,
        prompt_text: str,
        *,
        user_text: str = "",
        device: str | None = None,
        torch_dtype=None,
    ):
        tokenizer = processor
        model, _ = self._ensure_loaded()

        img_path = self._image_to_local_path(image)

        text = prompt_text.strip()
        if user_text and user_text.strip():
            text = f"{text}\n{user_text.strip()}"

        query = tokenizer.from_list_format([{"image": img_path}, {"text": text}])

        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]

        input_ids = tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )

        attn = torch.ones_like(input_ids)

        # Get target device
        if device is None:
            device = self.device
        if device is None:
            device = next(model.parameters()).device

        # Move to device and convert floating point tensors to model dtype
        inputs = {}
        for key, value in {"input_ids": input_ids, "attention_mask": attn}.items():
            value = value.to(device)
            # Convert floating point tensors to model's dtype
            if self._torch_dtype is not None and torch.is_floating_point(value):
                value = value.to(dtype=self._torch_dtype)
            inputs[key] = value

        input_len = int(input_ids.shape[-1])
        return inputs, input_len

    def _batch_call(
        self,
        images: list[Image.Image],
        prompt_text: str,
        *,
        user_text: str = "",
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        """Batch process multiple images with the same prompt."""
        model, tokenizer = self._ensure_loaded()

        batch_input_ids = []
        batch_attention = []
        input_lens = []

        # Build inputs for each image
        for img in images:
            inputs, input_len = self.build_chat_inputs(
                tokenizer, img, prompt_text, user_text=user_text
            )
            batch_input_ids.append(inputs["input_ids"].squeeze(0))
            batch_attention.append(inputs["attention_mask"].squeeze(0))
            input_lens.append(input_len)

        # Pad to same length
        tok = tokenizer
        pad_id = getattr(tok, "pad_token_id", 0) or 0

        max_len = max(ids.shape[-1] for ids in batch_input_ids)
        padded_ids = []
        padded_attn = []

        for i, ids in enumerate(batch_input_ids):
            pad_len = max_len - ids.shape[-1]
            padded_ids.append(
                torch.nn.functional.pad(ids, (pad_len, 0), value=pad_id)
            )
            padded_attn.append(
                torch.nn.functional.pad(batch_attention[i], (pad_len, 0), value=0)
            )
            input_lens[i] = input_lens[i] + pad_len

        batched = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_attn),
        }

        # Convert floating point tensors to model dtype
        if self._torch_dtype is not None:
            for key, value in batched.items():
                if torch.is_floating_point(value):
                    batched[key] = value.to(dtype=self._torch_dtype)

        with torch.no_grad():
            output_ids = model.generate(
                **batched,
                max_new_tokens=self.generation_max_tokens,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=self.caching,
            )

        results = []
        for i in range(output_ids.shape[0]):
            generated = output_ids[i, input_lens[i]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            results.append({"generated_text": text})
        return results

    def __call__(
        self,
        image,
        prompt_text: str,
        *,
        user_text: str = "",
        drop_config: dict[str, object] | None = None,
    ):
        if isinstance(image, list):
            return self._batch_call(image, prompt_text, user_text=user_text, drop_config=drop_config)

        # Single image case
        model, tokenizer = self._ensure_loaded()
        inputs, input_len = self.build_chat_inputs(
            tokenizer, image, prompt_text, user_text=user_text
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.generation_max_tokens,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=self.caching,
            )

        gen = output_ids[:, input_len:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        text = decoded[0] if decoded else ""
        return [{"generated_text": text}]


MODEL_SPEC = ModelSpec(
    key="chexagent",
    model_id="StanfordAIMI/CheXagent-2-3b-srrg-findings",
    prompt_key="chexagent_default",
    task="image-to-text",
    supports_images=True,
    generation_max_tokens=512,
    caching=True,
)

MODEL_CLASS = CheXagent
