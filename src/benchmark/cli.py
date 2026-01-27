from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import BenchmarkConfig
from .hf_runner import run_inference, write_json
from .models import MODEL_SPECS
from .preprocess import PREPROCESS, _resolve_preprocess


def _validate_experiment(value: str) -> str:
    """
    Validate experiment string by trying to resolve the preprocess function.
    This supports both static keys (baseline, all_noise, ...) and parametric keys
    like ObjectClassOcclusion_pXX.
    """
    _resolve_preprocess(value)  # raises KeyError if invalid
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Radiology image benchmark using Hugging Face models."
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument(
        "--data_json",
        required=True,
        help="JSON file containing all image info and labels (sample_id -> sample dict).",
    )
    parser.add_argument("--data", required=True, help="Path to radiology image folder")
    static_experiments = sorted(PREPROCESS.keys())
    parser.add_argument(
        "--experiment",
        required=True,
        type=_validate_experiment,
        help=(
            "Experiment name used to select preprocessing. "
            f"Static options: {static_experiments}. "
            "Parametric option: ObjectClassOcclusion_pXX (XX in 0..100)."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Explicit output JSON path (overrides --output-dir/--experiment).",
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
    return parser


def _safe_path_segment(value: str) -> str:
    """Make a string safe to be used inside filenames/paths."""
    return value.replace("/", "__").replace("\\", "__").replace(" ", "_")


def _build_output_path(output_dir: str, experiment: str, model_id: str, prompt_key: str) -> Path:
    model_id_seg = _safe_path_segment(model_id)
    prompt_key_seg = _safe_path_segment(prompt_key)
    filename = f"{model_id_seg}_{prompt_key_seg}.json"
    return Path(output_dir) / experiment / filename



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    spec = MODEL_SPECS[args.model]
    prompt_key = args.prompt_key or spec.prompt_key
    output_path = (
        Path(args.output)
        if args.output is not None
        else _build_output_path(args.output_dir, args.experiment, spec.model_id, prompt_key)
    )

    config = BenchmarkConfig(
        model_key=args.model,
        data_json=Path(args.data_json),
        data_dir=Path(args.data),
        experiment=args.experiment,
        output_path=output_path,
        cache_dir=Path(args.cache_dir),
        prompt_key=args.prompt_key,
        max_images=args.max_images,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    with Path(args.data_json).open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = run_inference(config, dataset)
    write_json(config.output_path, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
