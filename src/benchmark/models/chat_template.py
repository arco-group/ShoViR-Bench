from __future__ import annotations

import torch
from PIL import Image
import random

from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor


def load_processor(
    model_id: str,
    trust_remote_code: bool = False,
    cache_dir: str | None = None,
):
    return AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )


def load_model(
    model_id: str,
    trust_remote_code: bool = False,
    cache_dir: str | None = None,
    torch_dtype=None,
):
    last_error: Exception | None = None
    for loader in (AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            return loader.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
            )
        except ValueError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to load model for {model_id}")


def build_chat_messages(
    prompt_text: str,
    image: Image.Image,
    user_text: str = "Analyze this image.",
) -> list[dict[str, list[dict[str, object]]]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt_text}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": image},
            ],
        },
    ]


def build_chat_inputs(
    processor,
    image: Image.Image,
    prompt_text: str,
    device: str | None,
    torch_dtype=None,
    user_text: str = "Analyze this image.",
):
        # --- Build inputs ---
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

    # --- Compute *true* prompt lengths (per sample) ---
    input_lens = None
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        # attention_mask: (B, T) with 1s on real tokens and 0s on padding
        input_lens = inputs["attention_mask"].sum(dim=-1).to(torch.long)  # (B,)
    elif "input_ids" in inputs and inputs["input_ids"] is not None:
        # Fallback: assumes no padding. If padding exists, this will be wrong.
        input_lens = torch.full(
            (inputs["input_ids"].shape[0],),
            inputs["input_ids"].shape[-1],
            dtype=torch.long,
        )

    # Move to device / dtype
    inputs = _prepare_inputs(inputs, device=device, torch_dtype=torch_dtype)

    # Avoid passing unused kwargs to generate/model forward
    inputs.pop("num_crops", None)

    # For convenience: return int if batch size == 1
    if input_lens is not None and input_lens.numel() == 1:
        input_lens = int(input_lens.item())
    return inputs, input_lens


def generate_with_template(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    device: str | None,
    torch_dtype=None,
    user_text: str = "Analyze this image.",
) -> list[dict[str, str]]:
    inputs, _ = build_chat_inputs(
        processor,
        image,
        prompt_text,
        device=device,
        torch_dtype=torch_dtype,
        user_text=user_text,
    )
    with torch.no_grad():
        output_ids = model.generate(**inputs)
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
    return [{"generated_text": text} for text in decoded]


def resolve_image_token_id(processor, model):
    for candidate in (
        processor,
        getattr(processor, "tokenizer", None),
        getattr(processor, "text_processor", None),
        getattr(model, "config", None),
    ):
        if candidate is None:
            continue
        token_id = getattr(candidate, "image_token_id", None)
        if isinstance(token_id, int):
            return token_id
    return None


def resolve_replacement_token_id(processor, model):
    for candidate in (
        processor,
        getattr(processor, "tokenizer", None),
        getattr(model, "config", None),
    ):
        if candidate is None:
            continue
        token_id = getattr(candidate, "pad_token_id", None)
        if isinstance(token_id, int):
            return token_id
    return None


def apply_image_token_rules(inputs, image_token_id: int, drop: bool, replace_id: int | None):
    if "input_ids" not in inputs:
        return inputs
    input_ids = inputs["input_ids"]
    mask = input_ids == image_token_id
    if replace_id is not None:
        input_ids = input_ids.masked_fill(mask, replace_id)
        inputs["input_ids"] = input_ids
    if drop:
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.masked_fill(mask, 0)
        inputs["attention_mask"] = attention_mask
    return inputs


def drop_pixel_patches(
    pixel_values: torch.Tensor,
    *,
    patch_size,
    drop_fraction: float | None = None,
    drop_indices,
    fill_value: float = 0.0,
    seed: int | None = None,
):
    if patch_size is None:
        return pixel_values

    added_batch = False
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)
        added_batch = True
    if pixel_values.dim() != 4:
        return pixel_values

    if isinstance(patch_size, int):
        patch_h, patch_w = patch_size, patch_size
    else:
        patch_h, patch_w = patch_size

    _, _, height, width = pixel_values.shape
    grid_h = height // patch_h
    grid_w = width // patch_w
    if grid_h == 0 or grid_w == 0:
        return pixel_values

    normalized = []
    if drop_indices:
        for item in drop_indices:
            if isinstance(item, int):
                normalized.append((item // grid_w, item % grid_w))
            else:
                if len(item) != 2:
                    continue
                normalized.append((int(item[0]), int(item[1])))
    elif drop_fraction:
        total = grid_h * grid_w
        drop_count = max(0, min(total, int(round(total * drop_fraction))))
        if drop_count > 0:
            rng = random.Random(seed)
            sampled = rng.sample(range(total), drop_count)
            normalized = [(idx // grid_w, idx % grid_w) for idx in sampled]

    if not normalized:
        return pixel_values

    updated = pixel_values.clone()
    for row, col in normalized:
        if row < 0 or col < 0 or row >= grid_h or col >= grid_w:
            continue
        row_start = row * patch_h
        row_end = min(row_start + patch_h, height)
        col_start = col * patch_w
        col_end = min(col_start + patch_w, width)
        updated[:, :, row_start:row_end, col_start:col_end] = fill_value

    if added_batch:
        return updated[0]
    return updated


def _prepare_inputs(inputs, device: str | None, torch_dtype=None):
    prepared = dict(inputs)
    for key, value in prepared.items():
        if not torch.is_tensor(value):
            continue
        kwargs: dict[str, object] = {}
        if device is not None:
            kwargs["device"] = device
        if torch_dtype is not None and torch.is_floating_point(value):
            kwargs["dtype"] = torch_dtype
        if kwargs:
            prepared[key] = value.to(**kwargs)
    return prepared
