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
            if self.device is not None:
                self._model.to(self.device)

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

        # BaseHFLM si aspetta un dict di input per model.generate(**inputs)
        attn = torch.ones_like(input_ids)

        # device: usa quello del modello se non passato
        if device is None:
            device = self.device
        if device is None:
            device = next(model.parameters()).device

        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attn.to(device),
        }
        input_len = int(input_ids.shape[-1])
        return inputs, input_len

    # ---------------------------------------------------------
    # Minimal call: single image / list -> loop
    # ---------------------------------------------------------
    def _single_image_call(self, image, prompt_text: str, *, user_text: str = ""):
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
        return {"generated_text": text}

    def __call__(
        self,
        image,
        prompt_text: str,
        *,
        user_text: str = "",
        drop_config: dict[str, object] | None = None,  # ignorato qui (non serve)
    ):
        if isinstance(image, list):
            return [self._single_image_call(img, prompt_text, user_text=user_text) for img in image]
        return [self._single_image_call(image, prompt_text, user_text=user_text)]


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
