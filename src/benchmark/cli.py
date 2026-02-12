from __future__ import annotations
import argparse
import json
import random
import random
import re
from pathlib import Path
import numpy as np
import random
from .config import BenchmarkConfig, ExperimentType
from .hf_runner import run_inference, write_json
from .io import iter_images, list_images
from .models import MODEL_SPECS
from .preprocess import PREPROCESS, _resolve_preprocess


EXPERIMENT_HELP = """
Experiment types:
  baseline     - No occlusion, process whole image as-is
  oco_pXX      - Object Class Occlusion at XX%% strength (uses annotated bboxes)
  roco_pXX     - Random Object Class Occlusion at XX%% strength (random bboxes)
  all_noise    - Replace entire image with random noise
  all_noise_mean - Replace entire image with matched correlated noise

Examples: baseline, oco_p50, oco_p100, roco_p25, roco_p75
"""


def _validate_experiment(value: str) -> str:
    """
    Validate experiment string by trying to resolve the preprocess function.
    This supports both static keys (baseline, all_noise, ...) and parametric keys
    like oco_pXX or roco_pXX.
    """
    _resolve_preprocess(value)  # raises KeyError if invalid
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Radiology image benchmark using Hugging Face models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXPERIMENT_HELP,
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument(
        "--data-json",
        dest="data_json",
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
            "Experiment name for preprocessing. "
            f"Static: {static_experiments}. "
            "Parametric: oco_pXX, roco_pXX (XX in 0..100)."
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


    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--filter-labels", 
        nargs="+",
        default=None,
        help="Filter dataset to only include samples with these CheXpert labels.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images per inference call (1 = single image, >1 = multi-image). Default: 1.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Random seed for reproducible bbox selection in OCO/ROCO experiments. Default: 3.",
    )

    return parser


def _safe_path_segment(value: str) -> str:
    """Make a string safe to be used inside filenames/paths."""
    return value.replace("/", "__").replace("\\", "__").replace(" ", "_")

_VERSION_RE = re.compile(r"^\d+(?:\.\d+)+$")

def _extract_dataset_name(data_path: str) -> str:
    """
    Extract a dataset name from a filesystem path.

    Heuristic:
    - If the path contains a version-like component (e.g. "2.1.0"), assume the dataset
      name is the directory right before that version component.
      Example: ".../mimic-cxr-jpg/2.1.0/files" -> "mimic-cxr-jpg"

    Fallbacks:
    - If the path contains a "data" directory, return the next component.
      Example: ".../data/mimic-cxr/images" -> "mimic-cxr"
    - Otherwise, return the parent directory name.
    """
    parts = Path(str(data_path).strip()).parts

    for i, p in enumerate(parts):
        if _VERSION_RE.match(p) and i > 0:
            return parts[i - 1]
    try:
        data_idx = next(i for i, p in enumerate(parts) if p == "data")
        return parts[data_idx + 1]
    except StopIteration:
        return Path(data_path).parent.name

def _build_output_path(output_dir: str, experiment: str, model_id: str, prompt_key: str, dataset: str, seed: int) -> Path:
    model_id_seg = _safe_path_segment(model_id)
    prompt_key_seg = _safe_path_segment(prompt_key)
    filename = f"{model_id_seg}_{prompt_key_seg}::seed={seed}.json"

    # For OCO/ROCO experiments, split into experiment_type/percentage format
    # e.g., "oco_p50" -> "oco/p50"
    if experiment.startswith("oco_") or experiment.startswith("roco_") or experiment.startswith("ro_") :
        exp_parts = experiment.split("_", 1)  # Split into ["oco", "p50"] or ["roco", "p75"]
        if len(exp_parts) == 2:
            return Path(output_dir) / exp_parts[0] / exp_parts[1] / dataset / filename

    return Path(output_dir) / experiment / dataset / filename


def _resolve_experiment_dataset(data_json_path: str, data_dir: Path, dataset_name: str, experiment: str) -> dict:
    """
    Resolve and load the dataset based on experiment type and dataset name.

    For padchest-gr dataset with OCO/ROCO experiments, loads all chexpert class files
    excluding categories with less than 50 images. For baseline experiments or other
    datasets, loads the provided data_json file as usual.

    Args:
        data_json_path: Path to the main data JSON file
        data_dir: Path to the data directory
        dataset_name: Name of the dataset (e.g., 'padchest-gr')
        experiment: Experiment type (e.g., 'baseline', 'oco_p50', 'roco_p75')

    Returns:
        Dictionary mapping sample_id to sample data
    """
    if dataset_name == "padchest-gr" and (experiment.startswith("oco") or experiment.startswith("ro")):
        # For OCO/ROCO experiments with padchest-gr, load all chexpert class files
        # excluding categories with less than 50 images
        # Navigate up to padchest-gr level: data_dir is .../BIMCV-Padchest-GR /PadChest_GR_images
        # So we need to go up two levels to reach data/padchest-gr
        chexpert_dir = data_dir.parent.parent / "chexpert-by-label"

        # Categories with >= 50 images (based on summary.json verified_per_category)
        valid_categories = [
            "Atelectasis",
            "Cardiomegaly",
            "Fracture",
            "Lung_Lesion",
            "Lung_Opacity",
            "Pleural_Effusion",
            "Pleural_Other",
            "Support_Devices",
        ]

        dataset = {}
        for category in valid_categories:
            category_file = chexpert_dir / f"{category}_samples.json"
            if category_file.exists():
                with category_file.open("r", encoding="utf-8") as f:
                    category_data = json.load(f)
                    if isinstance(category_data, list):
                        for sample in category_data:
                            composite_key = f"{sample['img_path']}::{category}"
                            sample_copy = dict(sample)
                            sample_copy["target_category"] = category
                            dataset[composite_key] = sample_copy
                    else:
                        for key, sample in category_data.items():
                            composite_key = f"{key}::{category}"
                            sample_copy = dict(sample)
                            sample_copy["target_category"] = category
                            dataset[composite_key] = sample_copy

    elif dataset_name == "mimic-cxr-jpg" and (experiment.startswith("oco") or experiment.startswith("ro")):
        chexpert_dir = Path(data_json_path).parent / "classes_jsons"

        valid_categories = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged_Cardiomediastinum",
            "Fracture",
            "Lung_Lesion",
            "Lung_Opacity",
            "Pleural_Effusion",
            "Pleural_Other",
            "Pneumonia",
            "Pneumothorax"
            "Support_Devices",
        ]

        dataset = {}
        for category in valid_categories:
            category_file = chexpert_dir / f"{category}.json"
            if category_file.exists():
                with category_file.open("r", encoding="utf-8") as f:
                    category_data = json.load(f)
                    if isinstance(category_data, list):
                        for sample in category_data:
                            composite_key = f"{sample['img_path']}::{category}"
                            sample_copy = dict(sample)
                            sample_copy["target_category"] = category
                            dataset[composite_key] = sample_copy
                    else:
                        for key, sample in category_data.items():
                            composite_key = f"{key}::{category}"
                            sample_copy = dict(sample)
                            sample_copy["target_category"] = category
                            dataset[composite_key] = sample_copy
    else:
        # For baseline or other datasets, use the provided data_json file as usual
        with Path(data_json_path).open("r", encoding="utf-8") as f:
            dataset = json.load(f)

    return dataset



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    spec = MODEL_SPECS[args.model]
    prompt_key = args.prompt_key or spec.prompt_key

    # Set random seed for reproducible bbox selection
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Extract dataset name from data path
    dataset_name = _extract_dataset_name(args.data)

    output_path = (
        Path(args.output)
        if args.output is not None
        else _build_output_path(args.output_dir, args.experiment, spec.model_id, prompt_key, dataset_name, args.seed)
    )
    print('remote code: ', args.trust_remote_code)
    config = BenchmarkConfig(
        model_key=args.model,
        data_dir=Path(args.data),
        experiment=args.experiment,
        output_path=output_path,
#        data_json=Path(args.data_json) if args.data_json else None,
        cache_dir=Path(args.cache_dir),
        prompt_key=args.prompt_key,
        max_images=args.max_images,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        filter_labels=args.filter_labels,
        num_images=args.num_images,
    )

    # Set random seed for reproducible bbox selection
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Resolve and load dataset based on experiment type and dataset name
    dataset = _resolve_experiment_dataset(args.data_json, config.data_dir, dataset_name, args.experiment)

    results = run_inference(config, dataset)
    write_json(config.output_path, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())