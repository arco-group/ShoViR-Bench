from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from transformers import pipeline

from .config import BenchmarkConfig, InferenceResult
from .models import MODEL_SPECS
from .prompts import PROMPTS


def _build_pipeline(config: BenchmarkConfig):
    spec = MODEL_SPECS[config.model_key]
    pipeline_kwargs = {
        "task": spec.task,
        "model": spec.model_id,
        "trust_remote_code": config.trust_remote_code,
        "cache_dir": str(config.cache_dir),
    }
    if config.device is not None:
        pipeline_kwargs["device"] = config.device
    if config.dtype is not None:
        pipeline_kwargs["torch_dtype"] = _resolve_dtype(config.dtype)
    return pipeline(**pipeline_kwargs)


def _resolve_prompt(config: BenchmarkConfig) -> tuple[str, str]:
    spec = MODEL_SPECS[config.model_key]
    prompt_key = config.prompt_key or spec.prompt_key
    if prompt_key not in PROMPTS:
        raise KeyError(f"Prompt key not found: {prompt_key}")
    return prompt_key, PROMPTS[prompt_key]


def run_inference(
    config: BenchmarkConfig,
    images: Iterable[tuple[str, object]],
) -> list[InferenceResult]:
    spec = MODEL_SPECS[config.model_key]
    prompt_key, prompt_text = _resolve_prompt(config)
    pipe = _build_pipeline(config)

    results: list[InferenceResult] = []
    for image_path, image in images:
        output = pipe(image, prompt=prompt_text)
        response_text = _extract_text(output)
        results.append(
            InferenceResult(
                image_path=image_path,
                model_key=spec.key,
                model_id=spec.model_id,
                prompt_key=prompt_key,
                prompt_text=prompt_text,
                response_text=response_text,
            )
        )
    return results


def write_jsonl(path: Path, results: Iterable[InferenceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item.__dict__, ensure_ascii=True) + "\n")


def _extract_text(output) -> str:
    if isinstance(output, list):
        if not output:
            return ""
        if isinstance(output[0], dict):
            return output[0].get("generated_text", "")
        return str(output[0])
    if isinstance(output, dict):
        return output.get("generated_text", "")
    return str(output)


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
