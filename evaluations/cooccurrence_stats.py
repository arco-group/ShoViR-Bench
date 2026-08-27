# cooccurrence_stats.py
#
# Ground-truth disease co-occurrence statistics for PadChest-GR / MIMIC-CXR-JPG,
# independent of any model output. Answers: how often do disease A and disease B
# actually co-occur in the same image? This is the population-level statistic that
# motivates the cross-disease flip-rate analysis (see flip_rate_cross_disease.py) --
# it does not require CheXbert or a GPU, since ground-truth labels are already
# present in the OCO json files.

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval_chexbert_class import CheXbert_CONDITIONS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSES = [c for c in CheXbert_CONDITIONS if c != "No Finding"]
COND_TO_IDX = {c: i for i, c in enumerate(CheXbert_CONDITIONS)}


def load_ground_truth(dataset: str) -> pd.DataFrame:
    """One row per unique image with its full binary 14-class GT vector (positive-only)."""
    p00_dir = REPO_ROOT / "outputs" / "oco" / "p00" / dataset
    files = sorted(p00_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No files found under {p00_dir}")

    seen: dict[str, list[int]] = {}
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for item in data:
            img = item.get("image_path", "")
            if img in seen:
                continue
            gt = item.get("label")
            if gt is None or len(gt) != len(CheXbert_CONDITIONS):
                continue
            seen[img] = [1 if int(v) == 1 else 0 for v in gt]

    mat = np.array(list(seen.values()))
    return pd.DataFrame(mat, columns=CheXbert_CONDITIONS, index=list(seen.keys()))


def build_stats(dataset: str, out_dir: Path) -> None:
    gt = load_ground_truth(dataset)
    n_images = len(gt)
    gt = gt[CLASSES]

    prevalence = gt.mean().rename("prevalence")
    count = gt.sum().rename("n_positive")

    # Co-occurrence count: # images where both A and B are positive
    cooccur_count = gt.T.dot(gt).astype(int)
    np.fill_diagonal(cooccur_count.values, 0)

    # Conditional probability P(B present | A present), row=A, col=B
    cond_prob = cooccur_count.div(count, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    prevalence.to_frame().join(count).to_csv(out_dir / "class_prevalence.csv")
    cooccur_count.to_csv(out_dir / "cooccurrence_counts.csv")
    cond_prob.to_csv(out_dir / "cooccurrence_conditional_prob.csv")

    print(f"\n[{dataset}] {n_images} unique images")
    print("Class prevalence (fraction of images positive):")
    print(prevalence.sort_values(ascending=False).map(lambda x: f"{x:.3f}").to_string())
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    for dataset in ["padchest-gr", "mimic-cxr-jpg"]:
        build_stats(dataset, REPO_ROOT / "results" / "oco" / "flip_rate_cross_disease" / dataset)
