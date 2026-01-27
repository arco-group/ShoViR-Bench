from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import BenchmarkConfig
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
) -> list[dict[str, str]]:

    _prompt_key, prompt_text = _resolve_prompt(config)
    preprocess_fn = _resolve_preprocess(config.experiment)
    model = _build_model_instance(config)
    results: list[dict[str, str]] = []
    processed = 0

    for _sample_id, sample in dataset.items():
        rel_img_path = sample.get("img_path")
        if not rel_img_path:
            continue

        sample_with_ctx = dict(sample)
        sample_with_ctx["data_dir"] = str(config.data_dir)

        try:
            image = preprocess_fn(sample_with_ctx) 
        except Exception:
            continue

        output = model(image, prompt_text)
        predictions = _extract_text(output)
        references = sample.get("report", "")
        results.append(
            {
                "image_path": str(rel_img_path),
                "predictions": predictions,
                "references": references,
            }
        )

        processed += 1
        if config.max_images is not None and processed >= config.max_images:
            break

    return results


def run_inference_parallel(
    config: BenchmarkConfig,
    image_paths: List[Path],
    num_workers: int = 4,
    prefetch_count: int = 8,
    show_progress: bool = True,
) -> Iterator[InferenceResult]:
    """
    Run inference with parallel image loading.

    Uses a producer-consumer pattern where:
    - Worker threads load images from disk in parallel
    - Main thread runs GPU inference sequentially
    - Images are prefetched to overlap I/O with compute

    Args:
        config: Benchmark configuration
        image_paths: List of image paths to process
        num_workers: Number of parallel image loading workers
        prefetch_count: Number of images to prefetch ahead
        show_progress: Show progress bar

    Yields:
        InferenceResult for each processed image
    """
    from PIL import Image

    spec = MODEL_SPECS[config.model_key]
    prompt_key, prompt_text = _resolve_prompt(config)
    model = _build_model_instance(config)

    results: list[InferenceResult] = []
    for image_path, image in images:
        output = model(image, prompt_text)
        response_text = _extract_text(output)
        results.append(
            {
                "image_path": str(rel_img_path),
                "predictions": predictions,
                "references": references,
            }
        )

        processed += 1
        if config.max_images is not None and processed >= config.max_images:
            break

    return results


def write_json(path: Path, results: Iterable[dict[str, str]]) -> None:
    """
    Write a single JSON array, not JSONL.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(results), handle, ensure_ascii=True, indent=2)



def _extract_text(output: Any) -> str:
    if isinstance(output, list):
        if not output:
            return ""
        first = output[0]
        if isinstance(first, dict):
            return first.get("generated_text", "")
        return str(first)

        if isinstance(output[0], dict):
            item = output[0]
            # Handle generated_text format
            if "generated_text" in item:
                return item["generated_text"]
            # Handle findings/impression format (e.g., CXRMateED)
            if "findings" in item or "impression" in item:
                findings = item.get("findings", "")
                impression = item.get("impression", "")
                return f"Findings: {findings}\nImpression: {impression}".strip()
            return ""
        return str(output[0])
    if isinstance(output, dict):
        if "generated_text" in output:
            return output["generated_text"]
        # Handle findings/impression format
        if "findings" in output or "impression" in output:
            findings = output.get("findings", "")
            impression = output.get("impression", "")
            return f"Findings: {findings}\nImpression: {impression}".strip()
        return ""
        return output.get("generated_text", "")

    return str(output)