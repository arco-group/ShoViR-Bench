from __future__ import annotations

import abc
import argparse
import json
import random
import re
from typing import Callable

from PIL import Image
import torch

from ..prompts import PROMPTS

# HF path (default for most models)
from .chat_template import (
    apply_image_token_rules,
    build_chat_inputs,
    drop_pixel_patches,
    load_model,
    load_processor,
    resolve_image_token_id,
    resolve_replacement_token_id,
)

# Libra-backed multimodal path (handles both llava-rad and libra)
from .chat_template_llava import (
    build_chat_inputs_llava,
    load_model_llava,
    load_processor_llava,
)

# --- Hardcoded model IDs you care about (no ModelSpec changes) ---
LLAVA_RAD_ID ="X-iZhang/libra-llava-rad"

LIBRA_3B_ID = "X-iZhang/libra-v1.0-3b"

LLAVA_RAD_BASE = None


def _uses_libra_backend(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return (
        mid == LLAVA_RAD_ID.lower()
        or mid == LIBRA_3B_ID.lower()
        or "libra" in mid
        or "llava-rad" in mid
    )


def _resolve_conv_mode(model_id: str) -> str:
    mid = (model_id or "").lower()
    if "llava-rad" in mid:
        return "v1"
    if "libra" in mid:
        return "libra_v1"
    return "v1"


def _resolve_model_base(model_id: str) -> str | None:
    mid = (model_id or "").lower()
    if "llava-rad" in mid:
        return LLAVA_RAD_BASE
    # Libra models typically don't need model_base when loading from their own repo weights
    return None


class BaseHFLM(abc.ABC):
    def __init__(
        self,
        name: str,
        device: str | None,
        dtype: str | None,
        model_id: str,
        task: str,
        trust_remote_code: bool = False,
        cache_dir: str | None = None,
        image_preprocess: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.name = name
        self.model_id = model_id
        self.task = task
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.image_preprocess = image_preprocess
        self._model = None
        self._processor = None
        self._torch_dtype = _resolve_dtype(dtype) if dtype is not None else None

    def build_pipeline(self):
        from transformers import pipeline

        pipeline_kwargs: dict[str, object] = {
            "task": self.task,
            "model": self.model_id,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.cache_dir is not None:
            pipeline_kwargs["cache_dir"] = self.cache_dir
        if self.device is not None:
            pipeline_kwargs["device"] = self.device
        if self.dtype is not None:
            pipeline_kwargs["torch_dtype"] = _resolve_dtype(self.dtype)
        return pipeline(**pipeline_kwargs)

    def __call__(
        self,
        image: Image.Image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ) -> list[dict[str, str]]:
        model, processor, inputs, input_len = self.prepare_inputs(
            image,
            prompt_text,
            user_text=user_text,
            drop_config=drop_config,
        )
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
            )

        output_ids = output_ids[:, input_len:]
        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        return [{"generated_text": text} for text in decoded]

    def prepare_inputs(
        self,
        image: Image.Image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ):
        image = self.preprocess_image(image)
        model, processor = self._ensure_loaded()

        if _uses_libra_backend(self.model_id):
            conv_mode = _resolve_conv_mode(self.model_id)
            inputs, input_len = build_chat_inputs_llava(
                model,
                processor,
                image,
                prompt_text,
                device=self.device,
                torch_dtype=self._torch_dtype,
                user_text=user_text,
                conv_mode=conv_mode,
            )
        else:
            inputs, input_len = build_chat_inputs(
                processor,
                image,
                prompt_text,
                device=self.device,
                torch_dtype=self._torch_dtype,
                user_text=user_text,
            )

        inputs = self._apply_drop_config(inputs, processor, model, drop_config)
        return model, processor, inputs, input_len

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        if self.image_preprocess is not None:
            return self.image_preprocess(image)
        return image.convert("RGB")

    def _ensure_loaded(self):
        use_libra_backend = _uses_libra_backend(self.model_id)
        model_base = _resolve_model_base(self.model_id) if use_libra_backend else None

        # Load processor
        if self._processor is None:
            if use_libra_backend:
                self._processor = load_processor_llava(
                    self.model_id,
                    model_base=model_base,
                    cache_dir=self.cache_dir,
                    torch_dtype=self._torch_dtype,
                    device=self.device,
                    device_map=None,
                )
            else:
                self._processor = load_processor(
                    self.model_id,
                    trust_remote_code=self.trust_remote_code,
                    cache_dir=self.cache_dir,
                )

        # Load model
        if self._model is None:
            if use_libra_backend:
                self._model = load_model_llava(
                    self.model_id,
                    model_base=model_base,
                    cache_dir=self.cache_dir,
                    torch_dtype=self._torch_dtype,
                    device=self.device,
                    device_map=None,
                )
            else:
                self._model = load_model(
                    self.model_id,
                    trust_remote_code=self.trust_remote_code,
                    cache_dir=self.cache_dir,
                    torch_dtype=self._torch_dtype,
                )

            # Don't .to(device) if the model is already sharded/mapped
            if self.device is not None and getattr(self._model, "hf_device_map", None) is None:
                self._model.to(self.device)

        return self._model, self._processor

    def _apply_drop_config(self, inputs, processor, model, drop_config):
        if not drop_config:
            return inputs

        prepared = dict(inputs)

        image_token_id = drop_config.get("image_token_id")
        if image_token_id is None:
            image_token_id = resolve_image_token_id(processor, model)

        replace_id = drop_config.get("image_token_replacement_id")
        if replace_id is None and drop_config.get("replace_image_tokens"):
            replace_id = resolve_replacement_token_id(processor, model)

        drop_image_tokens = bool(drop_config.get("drop_image_tokens"))
        if image_token_id is not None and (drop_image_tokens or replace_id is not None):
            prepared = apply_image_token_rules(
                prepared,
                image_token_id,
                drop=drop_image_tokens,
                replace_id=replace_id,
            )

        tensor_key = "pixel_values" if "pixel_values" in prepared else ("images" if "images" in prepared else None)
        if tensor_key is not None:
            pixel_values = prepared[tensor_key]
            patch_size = drop_config.get("pixel_patch_size")
            drop_fraction = drop_config.get("pixel_patch_drop_fraction")
            drop_indices = drop_config.get("pixel_patch_drop_indices")
            fill_value = drop_config.get("pixel_patch_fill", 0.0)
            seed = drop_config.get("pixel_patch_seed")

            prepared[tensor_key] = drop_pixel_patches(
                pixel_values,
                patch_size=patch_size,
                drop_fraction=drop_fraction,
                drop_indices=drop_indices,
                fill_value=fill_value,
                seed=seed,
            )

        return prepared

    def process_img(self, paths):
        raise NotImplementedError()

    def get_likelihood_prompt(self, question, options):
        raise NotImplementedError()

    def get_logits(self, pixel_values, prompt_ids, ans_ids):
        raise NotImplementedError()

    def compute_scores(self, likelihood, ans_indices, length_norm):
        raise NotImplementedError()

    def get_prompt(self, question, options):
        raise NotImplementedError()

    def parse_response(self, response, target, options):
        print(f"Response: {response}; Target: {options[target]}")
        choice_style = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        prediction = re.findall(r"\(([A-Z])\)", response)
        if len(prediction) == 0:
            prediction = choice_style[random.choice(list(range(len(options))))]
        else:
            prediction = prediction[0]
        target = choice_style[target]
        return prediction.lower() == target.lower()

    @classmethod
    def run_debug_main(cls, model_spec, description: str) -> None:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("image", help="Path to an image to run.")
        parser.add_argument("--prompt", help="Override the default prompt.")
        parser.add_argument("--device", help="Device string (e.g., cuda:0).")
        parser.add_argument("--dtype", help="Torch dtype (e.g., float16).")
        parser.add_argument("--cache-dir", help="Hugging Face cache directory.")
        parser.add_argument("--trust-remote-code", action="store_true")
        args = parser.parse_args()

        prompt = args.prompt or PROMPTS.get(model_spec.prompt_key, "")
        model = cls(
            name=model_spec.key,
            device=args.device,
            dtype=args.dtype,
            model_id=model_spec.model_id,
            task=model_spec.task,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
        )
        image = Image.open(args.image)
        output = model(image, prompt)
        print(json.dumps(output, ensure_ascii=True, indent=2))


def _resolve_dtype(value: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required when setting --dtype") from exc

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value in mapping:
        return mapping[value]
    if value.startswith("torch."):
        attr = value.split(".", 1)[1]
        return getattr(torch, attr)
    return value
