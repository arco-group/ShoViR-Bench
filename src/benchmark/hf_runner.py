from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
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


class _CachedDataset(Dataset):
    """Disk-cached dataset: preprocessed images stored as .npy on disk, metadata in RAM."""

    def __init__(self, cache_dir: str, file_indices: List[int], metadata: List[Tuple[str, str, list, str]]):
        self._cache_dir = cache_dir
        self._file_indices = file_indices
        self._meta = metadata

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, str, list, str]:
        arr = np.load(os.path.join(self._cache_dir, f"{self._file_indices[idx]}.npy"))
        image = Image.fromarray(arr)
        path, report, labels, category = self._meta[idx]
        return image, path, report, labels, category


def _collate_pil(batch: List[Tuple[Image.Image, str, str, list, str]]) -> Tuple[List[Image.Image], List[str], List[str], List[list], List[str]]:
    """Collate that keeps PIL images as-is (no tensor conversion)."""
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    reports = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    categories = [item[4] for item in batch]
    return images, paths, reports, labels, categories


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

    num_images = config.num_images
    total_samples = len(dataset)

    with tempfile.TemporaryDirectory(prefix="benchmark_cache_") as cache_dir:

        # ── Phase 1: preprocess & cache to disk (multi-threaded) ──
        def _preprocess_and_cache(idx_sample: Tuple[int, Tuple[str, Mapping[str, Any]]]) -> Optional[Tuple[int, str, str, list, str]]:
            idx, (_sid, sample) = idx_sample
            rel_img_path = sample.get("img_path")
            if not rel_img_path:
                return None
            sample_with_ctx = dict(sample)
            sample_with_ctx["data_dir"] = str(config.data_dir)
            try:
                image = preprocess_fn(sample_with_ctx)
            except Exception:
                return None
            np.save(os.path.join(cache_dir, f"{idx}.npy"), np.asarray(image, dtype=np.uint8))
            return (idx, str(rel_img_path), sample.get("report", ""), sample.get("labels", []), sample.get("target_category", ""))

        max_items = config.max_images if config.max_images is not None else total_samples
        items_to_load = list(islice(enumerate(dataset.items()), max_items))
        load_workers = min(os.cpu_count() or 4, 16)

        entries: List[Tuple[int, str, str, list, str]] = []
        with ThreadPoolExecutor(max_workers=load_workers) as pool:
            futures = {pool.submit(_preprocess_and_cache, item): item for item in items_to_load}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Preprocessing ({load_workers} workers)"):
                result = future.result()
                if result is not None:
                    entries.append(result)

        # Sort by original dataset order
        entries.sort(key=lambda x: x[0])
        file_indices = [e[0] for e in entries]
        metadata = [(e[1], e[2], e[3], e[4]) for e in entries]

        total_loaded = len(metadata)

        # ── Phase 2: inference via DataLoader (reads from disk cache) ──
        ds = _CachedDataset(cache_dir, file_indices, metadata)
        dl_workers = min(4, load_workers)
        loader = DataLoader(
            ds,
            batch_size=num_images,
            shuffle=False,
            num_workers=dl_workers,
            prefetch_factor=2,
            collate_fn=_collate_pil,
        )

        for images, image_paths, references_list, labels_list, categories_list in tqdm(loader, desc=f"Inference (batch={num_images})"):
            outputs = model(images, prompt_text)
            predictions_list = [_extract_text(o) for o in outputs]

            result = {
                "image_paths": image_paths,
                "predictions": predictions_list,
                "references": references_list,
                "labels": labels_list,
            }
            if any(categories_list):
                result["target_categories"] = categories_list
            results.append(result)

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
            if k not in {"image_paths", "predictions", "references", "labels", "target_categories"}
        }

        preds = r.get("predictions", None)
        refs = r.get("references", None)
        labs = r.get("labels", None)
        cats = r.get("target_categories", None)

        for i in range(n):
            item = dict(shared)

            # Store one item per image
            item["image_path"] = image_paths[i]
            item["prediction"] = _broadcast_get(preds, i, n)
            item["reference"] = _broadcast_get(refs, i, n)
            item["label"] = _broadcast_get(labs, i, n)
            cat = _broadcast_get(cats, i, n)
            if cat:
                item["target_category"] = cat

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
