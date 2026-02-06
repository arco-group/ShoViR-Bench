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
        generation_max_tokens=spec.generation_max_tokens,
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
                "labels": labels_list
            })

            processed += len(images)
            if config.max_images is not None and processed >= config.max_images:
                break

    return results


from typing import Any, Iterable

def _broadcast_get(value: Any, i: int, n: int) -> Any:
    """
    Return the i-th element of a per-image field, handling common edge cases:
    - If value is a list of length n: return value[i]
    - If value is a list of length 1: broadcast (repeat) value[0] for all i
    - If value is a scalar (e.g., a single string): broadcast it for all i
    - Otherwise: return None
    """
    if value is None:
        return None

    # If it's already a list/tuple (per-image), index or broadcast
    if isinstance(value, (list, tuple)):
        if len(value) == n:
            return value[i]
        if len(value) == 1:
            return value[0]
        # If lengths don't match and it's not a single-item list, fail gracefully
        return None

    # If it's a scalar (e.g., a single prediction string), broadcast it
    return value


def flatten_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert a list of "batched" result dicts (where each field may be a list per image)
    into a flat list of dicts (one dict per image), with broadcasting when needed.
    """
    flat: list[dict[str, Any]] = []

    for r in results:
        image_paths = r.get("image_paths", [])
        n = len(image_paths)

        # Copy any extra metadata into every per-image item
        shared = {
            k: v for k, v in r.items()
            if k not in {"image_paths", "predictions", "references", "labels"}
        }

        preds = r.get("predictions", None)
        refs = r.get("references", None)
        labs = r.get("labels", None)

        for i in range(n):
            item = dict(shared)

            # Store one item per image
            item["image_path"] = image_paths[i]
            item["prediction"] = _broadcast_get(preds, i, n)
            item["reference"] = _broadcast_get(refs, i, n)
            item["label"] = _broadcast_get(labs, i, n)

            flat.append(item)

    return flat


def write_json(path: Path, results: Iterable[dict[str, Any]]) -> None:
    """Write results as a JSON array (flattened per image)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten batched results -> one dict per image
    flat = flatten_results(results)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(flat, handle, ensure_ascii=False, indent=2)



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
