"""
count_co_occurrence_by_chexpert.py

For each per-class *_samples.json file in a directory, counts how often each
CheXpert category appears as a co_occurrence_regions label.  Produces a
per-file ranked table and a cross-file summary heatmap-style table.

Usage (from project root):
    python3 src/pre-analysis/padchest/preprocessing/count_co_occurrence_by_chexpert.py

CONFIG block at the bottom of main() controls the input directory.
Works on both the original and the bbox-filtered versions of the files.
"""
import json
import glob
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List


# Canonical CheXpert category order (same as CheXbert label order)
CHEXPERT_CATEGORIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def count_co_occurrences(samples: List[dict]) -> Counter:
    """Return a Counter of chexpert_categories across all co_occurrence_regions."""
    counts: Counter = Counter()
    for sample in samples:
        for region in (sample.get("co_occurrence_regions") or []):
            for cat in (region.get("chexpert_categories") or []):
                counts[cat] += 1
    return counts


def load_file(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def class_name_from_path(path: str) -> str:
    """'Atelectasis_samples.json' → 'Atelectasis'"""
    return os.path.basename(path).replace("_samples.json", "")


def print_per_file_table(class_name: str, counts: Counter, n_samples: int) -> None:
    total = sum(counts.values())
    print(f"  {class_name}  (n={n_samples} samples, {total} co-occurrence regions total)")
    if not counts:
        print("    — none —")
        return
    for cat, cnt in counts.most_common():
        bar = "█" * min(cnt, 40)
        pct = cnt / total * 100
        print(f"    {cat:<35}  {cnt:>5}  ({pct:5.1f}%)  {bar}")
    print()


def print_summary_table(
    class_names: List[str],
    all_counts: Dict[str, Counter],
) -> None:
    # Collect all categories that appear at least once
    all_cats = sorted(
        {cat for c in all_counts.values() for cat in c},
        key=lambda c: CHEXPERT_CATEGORIES.index(c) if c in CHEXPERT_CATEGORIES else 999,
    )

    col_w = 25  # category column width
    num_w = 6   # number column width

    # Header
    header = f"  {'Category':<{col_w}}"
    for name in class_names:
        short = name[:num_w]
        header += f"  {short:>{num_w}}"
    print(header)
    print("  " + "-" * (col_w + (num_w + 2) * len(class_names)))

    for cat in all_cats:
        row = f"  {cat:<{col_w}}"
        for name in class_names:
            cnt = all_counts[name].get(cat, 0)
            row += f"  {cnt:>{num_w}}"
        print(row)
    print()


def main():
    # ============================================================
    # CONFIG
    # ============================================================
    data_dir = "data/padchest-gr/chexpert-by-label"
    # To use filtered files instead:
    # data_dir = "data/padchest-gr/chexpert-by-label-filtered"
    # ============================================================

    input_files = sorted(
        p for p in glob.glob(os.path.join(data_dir, "*_samples.json"))
        if os.path.basename(p)[0].isupper()
    )
    if not input_files:
        print(f"No *_samples.json files found in {data_dir}")
        return

    print(f"Co-occurrence CheXpert category counts")
    print(f"Input: {data_dir}")
    print(f"Files: {len(input_files)}")
    print("=" * 70)
    print()

    class_names: List[str] = []
    all_counts: Dict[str, Counter] = {}

    for path in input_files:
        name = class_name_from_path(path)
        samples = load_file(path)
        counts = count_co_occurrences(samples)
        class_names.append(name)
        all_counts[name] = counts
        print_per_file_table(name, counts, len(samples))

    print("=" * 70)
    print()
    print("SUMMARY TABLE  (co-occurrence region counts per class file)")
    print()
    print_summary_table(class_names, all_counts)


if __name__ == "__main__":
    main()
