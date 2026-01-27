from __future__ import annotations

import abc
import argparse
import json
import random
import re
from typing import Callable

import numpy as np
from PIL import Image
import torch

from ..prompts import PROMPTS
from .chat_template import (
    apply_image_token_rules,
    build_chat_messages,
    drop_pixel_patches,
    load_model,
    load_processor,
    _prepare_inputs,
    resolve_image_token_id,
    resolve_replacement_token_id,
)


class BaseHFLM(abc.ABC):
    def __init__(
        self,
        name: str,
        device: str | None,
        dtype: str | None,
        model_id: str,
        task: str,
        generation_max_tokens: int = 512,
        caching: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str | None = None
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.name = name
        self.model_id = model_id
        self.generation_max_tokens = generation_max_tokens
        self.caching = caching
        self.task = task
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
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
                max_new_tokens=self.generation_max_tokens,
                do_sample=False,
                use_cache=self.caching,
            )
        output_ids = output_ids[:, input_len:]

        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        # Convert the model output into plain text
        return [{"generated_text": text} for text in decoded]

    def prepare_inputs(
        self,
        image: Image.Image,
        prompt_text: str,
        *,
        user_text: str = "Analyze this image.",
        drop_config: dict[str, object] | None = None,
    ):
        model, processor = self._ensure_loaded()
        inputs, input_len = self.build_chat_inputs(
            processor,
            image,
            prompt_text,
            user_text=user_text,
        )
        inputs = self._apply_drop_config(inputs, processor, model, drop_config)
        return model, processor, inputs, input_len

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

        if hasattr(processor, "apply_chat_template"):
            messages = build_chat_messages(prompt_text, image, user_text=user_text)
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception:
                inputs = processor(images=image, text=prompt_text, return_tensors="pt")
        else:
            inputs = processor(images=image, text=prompt_text, return_tensors="pt")

        input_lens = None
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            input_lens = inputs["attention_mask"].sum(dim=-1).to(torch.long)
        elif "input_ids" in inputs and inputs["input_ids"] is not None:
            input_lens = torch.full(
                (inputs["input_ids"].shape[0],),
                inputs["input_ids"].shape[-1],
                dtype=torch.long,
            )

        inputs = _prepare_inputs(inputs, device=device, torch_dtype=torch_dtype)
        inputs.pop("num_crops", None)

        if input_lens is not None and input_lens.numel() == 1:
            input_lens = int(input_lens.item())
        return inputs, input_lens

        

    def _ensure_loaded(self):
        if self._processor is None:
            self._processor = load_processor(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
            )
        if self._model is None:
            self._model = load_model(
                self.__class__.__name__,
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                torch_dtype=self._torch_dtype,
            )
            if self.device is not None:
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

        if "pixel_values" in prepared:
            pixel_values = prepared["pixel_values"]
            patch_size = drop_config.get("pixel_patch_size")
            drop_fraction = drop_config.get("pixel_patch_drop_fraction")
            drop_indices = drop_config.get("pixel_patch_drop_indices")
            fill_value = drop_config.get("pixel_patch_fill", 0.0)
            seed = drop_config.get("pixel_patch_seed")
            prepared["pixel_values"] = drop_pixel_patches(
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
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
            model_id=model_spec.model_id,
            name=model_spec.key,
            task=model_spec.task,
            generation_max_tokens=model_spec.generation_max_tokens,
            caching=model_spec.caching,
            
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

