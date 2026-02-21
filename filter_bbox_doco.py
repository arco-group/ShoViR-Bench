#!/usr/bin/env python3
import json
from typing import List, Dict, Any, Tuple, Optional

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

def has_any_overlap_with_disease(
    co_bbox: BBox,
    disease_regions: List[Dict[str, Any]],
    iou_threshold: float
) -> bool:
    for d in (disease_regions or []):
        db = d.get("bbox")
        if not db:
            continue
        if iou(co_bbox, db) > iou_threshold:
            return True
    return False

def filter_co_occurrence_regions(
    disease_regions: List[Dict[str, Any]],
    co_regions: List[Dict[str, Any]],
    iou_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """Keep only co_occurrence regions that do NOT overlap any disease region."""
    kept = []
    for c in (co_regions or []):
        cb = c.get("bbox")
        if not cb:
            # se manca bbox, per sicurezza la teniamo (o puoi scartarla)
            kept.append(c)
            continue

        if not has_any_overlap_with_disease(cb, disease_regions, iou_threshold):
            kept.append(c)
    return kept

def main():
    # =========================
    # INPUTS (modifica qui)
    # =========================
    json_in_path = "/Users/marcosalme/Desktop/ShortCutRRG/classes_jsons/Support_Devices_samples.json" 
    json_out_path: Optional[str] = "/Users/marcosalme/Desktop/ShortCutRRG/classes_jsons/Support_Devices_samples_filtered.json"  # None per non salvare
    iou_threshold = 0.15  # 0.0 => qualsiasi intersezione conta come overlap
    print_summary = True
    # =========================

    with open(json_in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_co_before = 0
    total_co_after = 0

    for sample_id, sample in data.items():
        disease_regions = sample.get("disease_regions", []) or []
        co_regions = sample.get("co_occurrence_regions", []) or []

        total_co_before += len(co_regions)

        filtered = filter_co_occurrence_regions(
            disease_regions=disease_regions,
            co_regions=co_regions,
            iou_threshold=iou_threshold
        )

        total_co_after += len(filtered)
        sample["co_occurrence_regions"] = filtered  # overwrite in-place

    if print_summary:
        removed = total_co_before - total_co_after
        frac_removed = (removed / total_co_before) if total_co_before else 0.0
        print("===== FILTER SUMMARY =====")
        print(f"Total co_occurrence_regions before: {total_co_before}")
        print(f"Total co_occurrence_regions after:  {total_co_after}")
        print(f"Removed: {removed} ({frac_removed:.2%})")
        print(f"IoU threshold: {iou_threshold}")

    if json_out_path:
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved filtered JSON to: {json_out_path}")

if __name__ == "__main__":
    main()
