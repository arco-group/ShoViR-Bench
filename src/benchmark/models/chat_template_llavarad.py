from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


# -----------------------------------------------------------------------------
# LLaVA-RAD config
# -----------------------------------------------------------------------------
LLAVA_RAD_BASE = "lmsys/vicuna-7b-v1.5"
LLAVA_RAD_NAME = "llavarad"

# Simple in-module cache to avoid loading the model twice (processor + model)
# Keyed by (model_id, cache_dir, torch_dtype_str)
_LLAVA_CACHE: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}


@dataclass
class LlavaProcessor:
    """
    Minimal wrapper that matches what BaseHFLM expects:
    - exposes batch_decode(...)
    - holds tokenizer and image_processor
    """
    tokenizer: Any
    image_processor: Any

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


def _load_llava_bundle(
    model_id: str,
    *,
    cache_dir: str | None = None,
    torch_dtype=None,
) -> dict[str, Any]:
    """
    Load tokenizer/model/image_processor once via the LLaVA loader and cache them.
    Note: some LLaVA forks do not accept cache_dir / dtype in the builder; we keep
    them only for caching key and apply dtype afterward if possible.
    """
    key = (model_id, cache_dir, str(torch_dtype) if torch_dtype is not None else None)
    if key in _LLAVA_CACHE:
        return _LLAVA_CACHE[key]

    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_id,
        LLAVA_RAD_BASE,
        LLAVA_RAD_NAME,
        device="cpu", #change on gpu (basta levarlo teoricamente)
        device_map=None #change on gpu (basta levarlo teoricamente)
    )

    # Try to apply dtype after loading (best-effort)
    if torch_dtype is not None:
        try:
            model = model.to(dtype=torch_dtype)
        except Exception:
            pass

    bundle = {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "context_len": context_len,
    }
    _LLAVA_CACHE[key] = bundle
    return bundle


def load_processor_llava(
    model_id: str,
    *,
    cache_dir: str | None = None,
    torch_dtype=None,
) -> LlavaProcessor:
    """
    Return a processor-like object for LLaVA: tokenizer + image_processor + batch_decode().
    """
    bundle = _load_llava_bundle(model_id, cache_dir=cache_dir, torch_dtype=torch_dtype)
    return LlavaProcessor(
        tokenizer=bundle["tokenizer"],
        image_processor=bundle["image_processor"],
    )


def load_model_llava(
    model_id: str,
    *,
    cache_dir: str | None = None,
    torch_dtype=None,
):
    """
    Return the LLaVA model instance (cached).
    """
    bundle = _load_llava_bundle(model_id, cache_dir=cache_dir, torch_dtype=torch_dtype)
    return bundle["model"]


def build_llava_prompt(prompt_text: str, user_text: str) -> str:
    """
    Build a simple, robust LLaVA prompt containing the <image> placeholder.
    We intentionally keep it minimal to reduce dependency on fork-specific conv templates.
    """
    prompt_text = (prompt_text or "").strip()
    user_text = (user_text or "Analyze this image.").strip()

    if prompt_text:
        return f"{prompt_text}\n\n<image>\n{user_text}\n"
    return f"<image>\n{user_text}\n"


def build_chat_inputs_llava(
    processor: LlavaProcessor,
    image: Image.Image,
    prompt_text: str,
    *,
    device: str | None,
    torch_dtype=None,
    user_text: str = "Analyze this image.",
):
    # Build query exactly like their example from hugging face
    query = f"<image>\n{(user_text or 'Analyze this image.').strip()}\n"

    # If you want prompt_text as a "system instruction", just prepend it
    # (keeps the same structure as their example)
    if prompt_text:
        query = f"{prompt_text.strip()}\n\n{query}"

    # Build prompt using conv_templates exactly like their example
    from llava.conversation import conv_templates

    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize prompt (LLaVA helper if available)
    try:
        from llava.mm_utils import tokenizer_image_token
        input_ids = tokenizer_image_token(prompt, processor.tokenizer, return_tensors="pt")
    except Exception:
        toks = processor.tokenizer(prompt, return_tensors="pt")
        input_ids = toks["input_ids"]

    attention_mask = torch.ones_like(input_ids)
    input_len = int(attention_mask.sum(dim=-1).item())  # batch=1

    # Image preprocessing exactly like their example:
    # image_processor.preprocess(... )["pixel_values"][0]
    pixel = processor.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    images = pixel.unsqueeze(0)  # add batch dim: (1, C, H, W)

    # Match their .half().cuda() BUT in a safe, device-aware way:
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        images = images.to(device)

    if torch_dtype is not None and torch.is_floating_point(images):
        images = images.to(dtype=torch_dtype)

    # Most LLaVA forks expect the kwarg name `images`
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images,
    }
    return inputs, input_len

