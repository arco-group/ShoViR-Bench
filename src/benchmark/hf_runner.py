from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import BenchmarkConfig, InferenceResult
from .models import MODEL_CLASSES, MODEL_SPECS
from .prompts import PROMPTS


def _build_model_instance(config: BenchmarkConfig):
    spec = MODEL_SPECS[config.model_key]
    model_cls = MODEL_CLASSES[spec.key]
    return model_cls(
        name=spec.key,
        device=config.device,
        dtype=config.dtype,
        model_id=spec.model_id,
        task=spec.task,
        trust_remote_code=config.trust_remote_code,
        cache_dir=str(config.cache_dir),
    )


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
    model = _build_model_instance(config)

    results: list[InferenceResult] = []
    for image_path, image in images:
        output = model(image, prompt_text)
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
