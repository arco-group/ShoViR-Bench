from __future__ import annotations

import argparse
from pathlib import Path

from .config import BenchmarkConfig
from .hf_runner import run_inference, run_inference_streaming, write_jsonl
from .io import iter_images, list_images
from .models import MODEL_SPECS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Radiology image benchmark using Hugging Face models."
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--data", required=True, help="Path to radiology image folder")
    parser.add_argument(
        "--experiment", required=True, help="Experiment name used in output path"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit output JSONL path (overrides --output-dir/--experiment)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Base output directory for assembled output paths",
    )
    parser.add_argument("--cache-dir", default="./model_caching")
    parser.add_argument("--prompt-key", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")

    # Parallel inference options
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel image loading (recommended for large datasets)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel image loading workers (default: 4)",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=8,
        help="Number of images to prefetch ahead (default: 8)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    return parser


def _build_output_path(output_dir: str, experiment: str, model_key: str) -> Path:
    return Path(output_dir) / experiment / f"{model_key}.jsonl"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_path = (
        Path(args.output)
        if args.output is not None
        else _build_output_path(args.output_dir, args.experiment, args.model)
    )

    config = BenchmarkConfig(
        model_key=args.model,
        data_dir=Path(args.data),
        output_path=output_path,
        cache_dir=Path(args.cache_dir),
        prompt_key=args.prompt_key,
        max_images=args.max_images,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    image_paths = list_images(config.data_dir)
    if config.max_images is not None:
        image_paths = image_paths[: config.max_images]

    print(f"Model: {args.model}")
    print(f"Images: {len(image_paths)}")
    print(f"Output: {output_path}")
    print(f"Parallel: {args.parallel}")
    print()

    if args.parallel:
        # Use parallel loading with streaming output (supports resume)
        count = run_inference_streaming(
            config,
            image_paths,
            output_path,
            num_workers=args.num_workers,
            prefetch_count=args.prefetch,
            show_progress=not args.no_progress,
        )
        print(f"\nProcessed {count} images")
    else:
        # Original sequential inference
        images = ((str(path), image) for path, image in iter_images(image_paths))
        results = run_inference(config, images)
        write_jsonl(config.output_path, results)
        print(f"\nProcessed {len(results)} images")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
