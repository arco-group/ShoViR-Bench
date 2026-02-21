#!/usr/bin/env python3
"""
filter_bbox_doco.py — PadChest-GR chexpert-by-label edition.

Filters co_occurrence_regions whose bbox overlaps (IoU > threshold) with any
disease_regions bbox.  Processes all *_samples.json files in a directory and
writes filtered copies to an output directory.

bbox format in the data: [[x1, y1, x2, y2], ...]  (nested list; a region can
have more than one box).  Any overlap between ANY co-bbox and ANY disease-bbox
causes the co-occurrence region to be dropped.
"""
import json
import glob
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

BBox = List[float]  # [x1, y1, x2, y2]


def sanitize_bbox(b: BBox) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, b)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = sanitize_bbox(box_a)
    bx1, by1, bx2, by2 = sanitize_bbox(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _flatten_bboxes(raw) -> List[BBox]:
    """Accept [[x,y,x,y], ...] or [x,y,x,y] and always return a list of flat boxes."""
    if not raw:
        return []
    if isinstance(raw[0], (list, tuple)):
        return [list(b) for b in raw]
    return [list(raw)]


def boxes_overlap(co_bboxes: List[BBox], disease_bboxes: List[BBox], iou_threshold: float) -> bool:
    """True if any co-bbox overlaps (IoU > threshold) any disease-bbox."""
    for cb in co_bboxes:
        for db in disease_bboxes:
            if iou(cb, db) > iou_threshold:
                return True
    return False


def filter_co_occurrence_regions(
    disease_regions: List[Dict[str, Any]],
    co_regions: List[Dict[str, Any]],
    iou_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Keep only co_occurrence regions that do NOT overlap any disease region."""
    disease_bboxes: List[BBox] = []
    for d in (disease_regions or []):
        disease_bboxes.extend(_flatten_bboxes(d.get("bbox")))

    kept = []
    for c in (co_regions or []):
        co_bboxes = _flatten_bboxes(c.get("bbox"))
        if not co_bboxes:
            kept.append(c)  # no bbox → keep (cannot check overlap)
            continue
        if not boxes_overlap(co_bboxes, disease_bboxes, iou_threshold):
            kept.append(c)
    return kept


def process_file(
    json_in_path: str,
    json_out_path: str,
    iou_threshold: float,
) -> Tuple[int, int]:
    """Filter one file and return (co_before, co_after)."""
    with open(json_in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_before = 0
    total_after = 0

    for sample in data:
        disease_regions = sample.get("disease_regions") or []
        co_regions = sample.get("co_occurrence_regions") or []

        total_before += len(co_regions)
        filtered = filter_co_occurrence_regions(disease_regions, co_regions, iou_threshold)
        total_after += len(filtered)
        sample["co_occurrence_regions"] = filtered

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return total_before, total_after


def main():
    # ============================================================
    # CONFIG
    # ============================================================
    data_dir      = "data/padchest-gr/chexpert-by-label"
    out_dir       = "data/padchest-gr/chexpert-by-label-filtered"
    iou_threshold = 0.15   # IoU > this → co-occurrence region is dropped
    # ============================================================

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Per-class files all start with an uppercase letter (e.g. Atelectasis_samples.json).
    # Exclude verified_samples.json / discarded_samples.json which have different formats.
    input_files = sorted(
        p for p in glob.glob(os.path.join(data_dir, "*_samples.json"))
        if os.path.basename(p)[0].isupper()
    )
    if not input_files:
        print(f"No *_samples.json files found in {data_dir}")
        return

    print(f"Processing {len(input_files)} files  |  IoU threshold: {iou_threshold}")
    print(f"Input:  {data_dir}")
    print(f"Output: {out_dir}")
    print()
    print(f"  {'File':<45}  {'Before':>7}  {'After':>7}  {'Removed':>9}")
    print(f"  {'-'*45}  {'-'*7}  {'-'*7}  {'-'*9}")

    grand_before = grand_after = 0

    for in_path in input_files:
        fname = os.path.basename(in_path)
        out_path = os.path.join(out_dir, fname)
        before, after = process_file(in_path, out_path, iou_threshold)
        removed = before - after
        frac = (removed / before) if before else 0.0
        print(f"  {fname:<45}  {before:>7}  {after:>7}  {removed:>6} ({frac:.1%})")
        grand_before += before
        grand_after  += after

    grand_removed = grand_before - grand_after
    grand_frac = (grand_removed / grand_before) if grand_before else 0.0
    print(f"  {'-'*45}  {'-'*7}  {'-'*7}  {'-'*9}")
    print(f"  {'TOTAL':<45}  {grand_before:>7}  {grand_after:>7}  {grand_removed:>6} ({grand_frac:.1%})")
    print()
    print(f"Filtered files saved to: {out_dir}")


if __name__ == "__main__":
    main()
