from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import BenchmarkConfig, InferenceResult
from .models import MODEL_CLASSES, MODEL_SPECS
from .prompts import PROMPTS
from .preprocess import _resolve_preprocess


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
    dataset: Mapping[str, Mapping[str, Any]],
) -> list[InferenceResult]:
    spec = MODEL_SPECS[config.model_key]
    prompt_key, prompt_text = _resolve_prompt(config)
    preprocess_fn = _resolve_preprocess(config.experiment)
    model = _build_model_instance(config)

    results: list[InferenceResult] = []
    processed = 0

    for _sample_id, sample in dataset.items():
        rel_img_path = sample.get("img_path")
        if not rel_img_path:
            continue

        # Create a mutable copy and inject data_dir for the preprocess stage
        sample_with_ctx = dict(sample)
        sample_with_ctx["data_dir"] = str(config.data_dir)

        try:
            # Preprocess is now responsible for reading the image from disk
            image = preprocess_fn(sample_with_ctx)
        except Exception:
            # Skip unreadable samples / preprocessing failures
            continue

        output = model(image, prompt_text)
        response_text = _extract_text(output)

        results.append(
            InferenceResult(
                image_path=str(rel_img_path),
                model_key=spec.key,
                model_id=spec.model_id,
                prompt_key=prompt_key,
                prompt_text=prompt_text,
                response_text=response_text,
            )
        )

        processed += 1
        if config.max_images is not None and processed >= config.max_images:
            break

    return results


def write_jsonl(path: Path, results: Iterable[InferenceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item.__dict__, ensure_ascii=True) + "\n")


def _extract_text(output: Any) -> str:
    if isinstance(output, list):
        if not output:
            return ""
        first = output[0]
        if isinstance(first, dict):
            return first.get("generated_text", "")
        return str(first)

    if isinstance(output, dict):
        return output.get("generated_text", "")

    return str(output)