from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Tuple

from PIL import Image
from tqdm import tqdm 
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
        caching=spec.caching,
        trust_remote_code=config.trust_remote_code,
        cache_dir=str(config.cache_dir),
    )


def _resolve_prompt(config: BenchmarkConfig) -> tuple[str, str]:
    spec = MODEL_SPECS[config.model_key]
    prompt_key = config.prompt_key or spec.prompt_key
    if prompt_key not in PROMPTS:
        raise KeyError(f"Prompt key not found: {prompt_key}")
    return prompt_key, PROMPTS[prompt_key]


def _batch_samples(
    dataset: Mapping[str, Mapping[str, Any]],
    batch_size: int,
) -> Iterator[List[Tuple[str, Mapping[str, Any]]]]:
    """Yield batches of (sample_id, sample) tuples."""
    items = iter(dataset.items())
    while True:
        batch = list(islice(items, batch_size))
        if not batch:
            break
        yield batch


def run_inference(
    config: BenchmarkConfig,
    dataset: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Run inference on dataset samples.

    Supports both single-image and multi-image inference modes:
    - num_images=1: Process one image at a time (default)
    - num_images>1: Batch multiple images together with the same prompt

    Args:
        config: Benchmark configuration
        dataset: Dataset mapping sample_id -> sample dict

    Returns:
        List of result dictionaries with predictions and references
    """
    _prompt_key, prompt_text = _resolve_prompt(config)
    preprocess_fn = _resolve_preprocess(config.experiment)
    model = _build_model_instance(config)
    results: list[dict[str, Any]] = []
    processed = 0

    num_images = config.num_images
    total_samples = len(dataset)

    if num_images == 1:
        # Single image mode
        for _sample_id, sample in tqdm(dataset.items(), total=total_samples, desc="Inference"):
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
            labels = sample.get("labels", [])

            results.append({
                "image_path": str(rel_img_path),
                "predictions": predictions,
                "references": references,
                "labels": labels,
            })

            processed += 1
            if config.max_images is not None and processed >= config.max_images:
                break
    else:
        # Multi-image mode: batch N independent conversations into
        # a single model.generate() call for GPU efficiency.
        # Each image gets its own conversation with the same prompt.
        total_batches = (total_samples + num_images - 1) // num_images
        for batch in tqdm(_batch_samples(dataset, num_images), total=total_batches, desc=f"Inference (batch={num_images})"):
            images: List[Image.Image] = []
            image_paths: List[str] = []
            references_list: List[str] = []
            labels_list: List[List[int]] = []

            for _sample_id, sample in batch:
                rel_img_path = sample.get("img_path")
                if not rel_img_path:
                    continue

                sample_with_ctx = dict(sample)
                sample_with_ctx["data_dir"] = str(config.data_dir)

                try:
                    image = preprocess_fn(sample_with_ctx)
                    images.append(image)
                    image_paths.append(str(rel_img_path))
                    references_list.append(sample.get("report", ""))
                    labels_list.append(sample.get("labels", []))
                except Exception:
                    continue

            if not images:
                continue

            # Single generate() call with N independent conversations
            outputs = model(images, prompt_text)
            predictions_list = [_extract_text(o) for o in outputs]

            results.append({
                "image_paths": image_paths,
                "predictions": predictions_list,
                "references": references_list,
                "labels": labels_list,
                "num_images": len(images),
            })

            processed += len(images)
            if config.max_images is not None and processed >= config.max_images:
                break

    return results


def write_json(path: Path, results: Iterable[dict[str, Any]]) -> None:
    """Write results as a JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(results), handle, ensure_ascii=False, indent=2)


def _extract_text(output: Any) -> str:
    """Extract text from various model output formats."""
    if isinstance(output, str):
        return output

    if isinstance(output, list):
        if not output:
            return ""
        first = output[0]
        if isinstance(first, dict):
            # Handle generated_text format
            if "generated_text" in first:
                return first["generated_text"]
            # Handle findings/impression format (e.g., CXRMateED)
            if "findings" in first or "impression" in first:
                findings = first.get("findings", "")
                impression = first.get("impression", "")
                return f"Findings: {findings}\nImpression: {impression}".strip()
            return ""
        return str(first)

    if isinstance(output, dict):
        if "generated_text" in output:
            return output["generated_text"]
        # Handle findings/impression format
        if "findings" in output or "impression" in output:
            findings = output.get("findings", "")
            impression = output.get("impression", "")
            return f"Findings: {findings}\nImpression: {impression}".strip()
        return ""

    return str(output)
