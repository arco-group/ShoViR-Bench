# flip_rate_cross_disease.py
#
# Cross-disease flip-rate analysis for OCO (Object Class Occlusion) experiments.
#
# Motivation: diseases co-occur statistically in PadChest-GR. Occluding the image
# region of disease A can therefore make the model's mention of a *different*,
# co-occurring disease B disappear from the generated report, even though B's own
# region was never touched. This script quantifies that effect: for every pair
# (A = occluded class, B = co-occurring class present in the ground truth, B != A),
# it measures the fraction of cases where the model correctly mentioned B at
# baseline but stopped mentioning it once A was occluded.

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval_chexbert_class import (  # noqa: E402
    CheXbert,
    CheXbert_CONDITIONS,
    map_target_category_to_condition,
    map_to_binary,
)

COND_TO_IDX = {c: i for i, c in enumerate(CheXbert_CONDITIONS)}
REPO_ROOT = Path(__file__).resolve().parent.parent


def label_reports(model: CheXbert, reports: List[str], uncertain_mode: str = "rrg-") -> np.ndarray:
    """Return an (N, 14) binary matrix, column order = CheXbert_CONDITIONS."""
    if not reports:
        return np.zeros((0, len(CheXbert_CONDITIONS)), dtype=int)
    outputs = model(reports)  # list length 14, each length N
    mode = "rrg+" if uncertain_mode == "rrg+" else "rrg"
    cols = []
    for j in range(len(outputs)):
        cols.append([map_to_binary(x, mode) for x in outputs[j]])
    return np.array(cols, dtype=int).T  # (N, 14)


def load_json(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonical_model_key(filename: str) -> str:
    """Normalize filenames that sometimes drop the '::seed=3' suffix (seen under oco/p00)."""
    stem = Path(filename).stem  # strip .json
    stem = stem.replace("::seed=3", "")
    return stem


def find_model_files(oco_root: Path, dataset: str, strengths: List[str]) -> Dict[str, Dict[str, Path]]:
    """canonical_model_key -> {strength: path}, only keeping models present for ALL requested strengths."""
    per_strength: Dict[str, Dict[str, Path]] = {}
    for s in strengths:
        d = oco_root / s / dataset
        per_strength[s] = {canonical_model_key(p.name): p for p in d.glob("*.json")} if d.exists() else {}

    common_keys = set.intersection(*(set(v.keys()) for v in per_strength.values())) if per_strength else set()
    result: Dict[str, Dict[str, Path]] = {}
    for key in sorted(common_keys):
        result[key] = {s: per_strength[s][key] for s in strengths}
    return result


def resolve_baseline(baseline_dir: Path, model_key: str, p00_path: Path | None) -> Tuple[List[str], List[str]]:
    """Return (image_paths, prediction_texts) for the baseline condition of a model.

    Prefers a real outputs/baseline/<dataset>/ file. Falls back to the OCO p00
    (0% occlusion == unoccluded image) file, deduplicated to one prediction per
    image, when no matching real baseline file exists (e.g. mimic-cxr-jpg, where
    most models were never run as a standalone baseline job).
    """
    candidates = list(baseline_dir.glob("*.json")) if baseline_dir.exists() else []
    match = next((p for p in candidates if canonical_model_key(p.name) == model_key), None)

    if match is not None:
        data = load_json(match)
        images = [item.get("image_path", "") for item in data]
        texts = [item.get("prediction", "") or "" for item in data]
        return images, texts

    if p00_path is not None:
        data = load_json(p00_path)
        seen: Dict[str, str] = {}
        for item in data:
            img = item.get("image_path", "")
            if img not in seen:
                seen[img] = item.get("prediction", "") or ""
        return list(seen.keys()), list(seen.values())

    return [], []


def run(
    dataset: str,
    strengths: List[str],
    models_filter: List[str] | None,
    uncertain_mode: str,
    out_dir: Path,
) -> None:
    oco_root = REPO_ROOT / "outputs" / "oco"
    baseline_dir = REPO_ROOT / "outputs" / "baseline" / dataset

    model_files = find_model_files(oco_root, dataset, strengths)
    if models_filter:
        model_files = {k: v for k, v in model_files.items() if any(m in k for m in models_filter)}

    if not model_files:
        raise SystemExit("No models found with data for all requested strengths.")

    print(f"Found {len(model_files)} model(s) with all strengths {strengths}:")
    for m in model_files:
        print(f"  - {m}")

    chexbert = CheXbert()

    rows: List[dict] = []

    for model_name, strength_paths in model_files.items():
        p00_path = strength_paths.get("p00")
        baseline_images, baseline_texts = resolve_baseline(baseline_dir, model_name, p00_path)
        if not baseline_images:
            print(f"[skip] no baseline (real or p00 fallback) for {model_name}")
            continue
        baseline_source = "real" if next(
            (p for p in (baseline_dir.glob("*.json") if baseline_dir.exists() else [])
             if canonical_model_key(p.name) == model_name), None
        ) else "p00-fallback"

        print(f"\n[{model_name}] Labeling {len(baseline_texts)} baseline reports ({baseline_source})...")
        baseline_labels = label_reports(chexbert, baseline_texts, uncertain_mode)
        baseline_by_image = {img: baseline_labels[i] for i, img in enumerate(baseline_images)}

        for strength, path in strength_paths.items():
            oco_data = load_json(path)
            oco_texts = [item.get("prediction", "") or "" for item in oco_data]
            print(f"[{model_name}][{strength}] Labeling {len(oco_texts)} OCO reports...")
            oco_labels = label_reports(chexbert, oco_texts, uncertain_mode)

            for i, item in enumerate(oco_data):
                image = item.get("image_path", "")
                if image not in baseline_by_image:
                    continue
                gt = item.get("label")
                if gt is None or len(gt) != len(CheXbert_CONDITIONS):
                    continue
                target_category = map_target_category_to_condition(item.get("target_category", ""))
                a_idx = COND_TO_IDX[target_category]

                base_vec = baseline_by_image[image]
                occ_vec = oco_labels[i]

                for b_idx, cond_b in enumerate(CheXbert_CONDITIONS):
                    if b_idx == a_idx:
                        continue
                    if cond_b == "No Finding":
                        continue
                    if int(gt[b_idx]) != 1:
                        continue  # B not present in ground truth for this image -> not a co-occurrence case

                    baseline_pos = int(base_vec[b_idx]) == 1
                    occluded_pos = int(occ_vec[b_idx]) == 1

                    rows.append(
                        {
                            "model": model_name,
                            "dataset": dataset,
                            "baseline_source": baseline_source,
                            "strength": strength,
                            "occluded_class": target_category,
                            "affected_class": cond_b,
                            "image_path": image,
                            "gt_present": True,
                            "baseline_pred": baseline_pos,
                            "occluded_pred": occluded_pos,
                            "flipped_pos_to_neg": bool(baseline_pos and not occluded_pos),
                            "flipped_neg_to_pos": bool((not baseline_pos) and occluded_pos),
                        }
                    )

    if not rows:
        raise SystemExit("No co-occurrence rows collected — check inputs.")

    long_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_dir / "flip_rate_cross_disease_long.csv", index=False)
    print(f"\nSaved per-sample long table to {out_dir / 'flip_rate_cross_disease_long.csv'} ({len(long_df)} rows)")

    # Aggregate: flip rate = flips / (# samples where baseline correctly detected B), per (model, strength, A, B)
    grp = long_df.groupby(["model", "dataset", "strength", "occluded_class", "affected_class"])
    summary = grp.apply(
        include_groups=False,
        func=lambda g: pd.Series(
            {
                "n_cooccur": len(g),
                "n_baseline_positive": int(g["baseline_pred"].sum()),
                "n_flipped_pos_to_neg": int(g["flipped_pos_to_neg"].sum()),
                "n_flipped_neg_to_pos": int(g["flipped_neg_to_pos"].sum()),
                "flip_rate": (
                    g["flipped_pos_to_neg"].sum() / g["baseline_pred"].sum()
                    if g["baseline_pred"].sum() > 0
                    else float("nan")
                ),
            }
        )
    ).reset_index()
    summary.to_csv(out_dir / "flip_rate_cross_disease_summary.csv", index=False)
    print(f"Saved per-(model,strength,pair) summary to {out_dir / 'flip_rate_cross_disease_summary.csv'}")

    # Aggregate over models -> mean flip rate per (strength, A, B), one matrix per strength
    for strength in strengths:
        sub = summary[summary["strength"] == strength]
        if sub.empty:
            continue
        agg = sub.groupby(["occluded_class", "affected_class"]).agg(
            flip_rate=("flip_rate", "mean"),
            n_baseline_positive=("n_baseline_positive", "sum"),
            n_flipped_pos_to_neg=("n_flipped_pos_to_neg", "sum"),
        ).reset_index()
        pivot = agg.pivot(index="occluded_class", columns="affected_class", values="flip_rate")
        pivot.to_csv(out_dir / f"flip_rate_matrix_{strength}_mean_over_models.csv")
        print(f"Saved mean-over-models flip-rate matrix for {strength} to "
              f"{out_dir / f'flip_rate_matrix_{strength}_mean_over_models.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-disease flip-rate analysis for OCO experiments.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="padchest-gr",
        choices=["padchest-gr", "mimic-cxr-jpg"],
        help="Dataset subdirectory under outputs/oco/<strength>/ to analyze.",
    )
    parser.add_argument(
        "--strengths",
        type=str,
        default="p20,p40,p60,p80,p100",
        help="Comma-separated OCO strength dirs under outputs/oco/ (e.g. p20,p40,p60,p80,p100).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated substrings to filter model filenames (default: all models present for all strengths).",
    )
    parser.add_argument(
        "--uncertain-mode",
        type=str,
        default="rrg-",
        choices=["rrg-", "rrg+"],
        help="How to treat CheXbert 'uncertain' labels when binarizing.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (relative to repo root) for CSV results. "
             "Default: results/oco/flip_rate_cross_disease/<dataset>.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    strengths = [s.strip() for s in args.strengths.split(",") if s.strip()]
    models_filter = [m.strip() for m in args.models.split(",") if m.strip()] or None
    out_dir = args.out_dir or f"results/oco/flip_rate_cross_disease/{args.dataset}"
    run(
        dataset=args.dataset,
        strengths=strengths,
        models_filter=models_filter,
        uncertain_mode=args.uncertain_mode,
        out_dir=REPO_ROOT / out_dir,
    )
