from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Iterable, Iterator, List

from tqdm import tqdm

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
    """Run inference sequentially (original implementation)."""
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

    # Queue for prefetched images
    image_queue: Queue[tuple[str, Image.Image] | None] = Queue(maxsize=prefetch_count)

    def load_image(path: Path) -> tuple[str, Image.Image]:
        """Load a single image."""
        img = Image.open(path)
        return str(path), img

    def producer():
        """Load images in parallel and put them in the queue."""
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for path in image_paths:
                future = executor.submit(load_image, path)
                futures.append(future)

            for future in futures:
                try:
                    result = future.result()
                    image_queue.put(result)
                except Exception as e:
                    print(f"Error loading image: {e}")
                    continue

        # Signal completion
        image_queue.put(None)

    # Start producer thread
    producer_thread = Thread(target=producer, daemon=True)
    producer_thread.start()

    # Process images as they become available
    progress = tqdm(total=len(image_paths), disable=not show_progress, desc=f"Inference ({spec.key})")

    while True:
        item = image_queue.get()
        if item is None:
            break

        image_path, image = item

        try:
            output = model(image, prompt_text)
            response_text = _extract_text(output)

            yield InferenceResult(
                image_path=image_path,
                model_key=spec.key,
                model_id=spec.model_id,
                prompt_key=prompt_key,
                prompt_text=prompt_text,
                response_text=response_text,
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        finally:
            progress.update(1)

    progress.close()
    producer_thread.join()


def run_inference_streaming(
    config: BenchmarkConfig,
    image_paths: List[Path],
    output_path: Path,
    num_workers: int = 4,
    prefetch_count: int = 8,
    show_progress: bool = True,
) -> int:
    """
    Run inference with parallel loading and stream results to file.

    This is the most efficient method for large datasets:
    - Parallel image loading
    - Streams results to disk (low memory usage)
    - Can resume from partial runs

    Args:
        config: Benchmark configuration
        image_paths: List of image paths to process
        output_path: Path to output JSONL file
        num_workers: Number of parallel image loading workers
        prefetch_count: Number of images to prefetch ahead
        show_progress: Show progress bar

    Returns:
        Number of images processed
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing results to enable resume
    processed_paths = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_paths.add(data.get("image_path", ""))
                except json.JSONDecodeError:
                    continue
        if processed_paths:
            print(f"Resuming: {len(processed_paths)} images already processed")

    # Filter out already processed images
    remaining_paths = [p for p in image_paths if str(p) not in processed_paths]

    if not remaining_paths:
        print("All images already processed")
        return len(processed_paths)

    count = len(processed_paths)

    # Stream results to file
    with output_path.open("a", encoding="utf-8") as handle:
        for result in run_inference_parallel(
            config,
            remaining_paths,
            num_workers=num_workers,
            prefetch_count=prefetch_count,
            show_progress=show_progress,
        ):
            handle.write(json.dumps(result.__dict__, ensure_ascii=True) + "\n")
            handle.flush()  # Ensure data is written
            count += 1

    return count


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
    return str(output)
