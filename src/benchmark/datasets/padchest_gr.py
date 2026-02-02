"""PadChest-GR dataset implementation with CheXpert label mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_dataset import (
    BaseDataset,
    BoundingBox,
    Region,
    Sample,
    CHEXPERT_LABELS,
)


class PadChestGRDataset(BaseDataset):
    """
    PadChest-GR dataset with grounded radiology report annotations.

    The dataset contains chest X-ray images with:
    - Full radiology reports
    - Per-image CheXpert-compatible labels (14 categories)
    - Region annotations with:
      - Anatomical location
      - Bounding boxes (normalized coordinates)
      - Finding descriptions
      - CheXpert category mappings
      - Sentence-level annotations

    Expected annotation format (verified_samples.json):
    {
        "image_filename.png": {
            "img_path": "image_filename.png",
            "report": "Full radiology report text",
            "labels": [0, 0, 1, 0, ...],  # 14 CheXpert labels
            "regions": [
                {
                    "anatomy": "lung",
                    "bbox": [[x1, y1, x2, y2], ...],
                    "findings": ["finding1", "finding2"],
                    "chexpert_categories": ["Lung Opacity"],
                    "sentence": "Sentence describing the finding.",
                    "labels": [0, 0, 1, 0, ...]
                },
                ...
            ]
        },
        ...
    }
    """

    def __init__(
        self,
        data_dir: Path | str,
        annotation_file: Path | str | None = None,
        transform: Optional[Any] = None,
        filter_labels: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the PadChest-GR dataset.

        Args:
            data_dir: Directory containing the image files.
            annotation_file: Path to the JSON annotation file (verified_samples.json).
            transform: Optional transform to apply to images.
            filter_labels: Optional list of CheXpert labels to filter samples by.
                          Only samples with at least one of these labels will be included.
        """
        self.filter_labels = filter_labels
        super().__init__(data_dir, annotation_file, transform)

    @property
    def name(self) -> str:
        return "padchest-gr"

    def _load_annotations(self) -> None:
        """Load annotations from the JSON file."""
        if self.annotation_file is None:
            raise ValueError("annotation_file is required for PadChestGRDataset")

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        self._samples = []
        for image_id, sample_data in data.items():
            sample = self._parse_sample(image_id, sample_data)

            if self.filter_labels:
                if not any(sample.has_label(label) for label in self.filter_labels):
                    continue

            self._samples.append(sample)

    def _parse_sample(self, image_id: str, data: Dict[str, Any]) -> Sample:
        """Parse a single sample from the annotation data."""
        regions = []
        for region_data in data.get("regions", []):
            region = self._parse_region(region_data)
            regions.append(region)

        return Sample(
            image_id=image_id,
            img_path=data["img_path"],
            report=data["report"],
            labels=data["labels"],
            regions=regions,
        )

    def _parse_region(self, data: Dict[str, Any]) -> Region:
        """Parse a region annotation."""
        bboxes = []
        for bbox_coords in data.get("bbox", []):
            bbox = BoundingBox.from_list(bbox_coords)
            bboxes.append(bbox)

        return Region(
            anatomy=data.get("anatomy", ""),
            bboxes=bboxes,
            findings=data.get("findings", []),
            chexpert_categories=data.get("chexpert_categories", []),
            sentence=data.get("sentence", ""),
            labels=data.get("labels", [0] * len(CHEXPERT_LABELS)),
        )

    def get_samples_by_anatomy(self, anatomy: str) -> List[Sample]:
        """Get samples that have regions with the specified anatomy."""
        return [
            sample for sample in self._samples
            if any(region.anatomy == anatomy for region in sample.regions)
        ]

    def get_unique_anatomies(self) -> List[str]:
        """Get all unique anatomy types in the dataset."""
        anatomies = set()
        for sample in self._samples:
            for region in sample.regions:
                anatomies.add(region.anatomy)
        return sorted(anatomies)

    def get_unique_findings(self) -> List[str]:
        """Get all unique findings in the dataset."""
        findings = set()
        for sample in self._samples:
            for region in sample.regions:
                findings.update(region.findings)
        return sorted(findings)

    def get_region_count(self) -> int:
        """Get the total number of region annotations."""
        return sum(len(sample.regions) for sample in self._samples)

    def get_bbox_count(self) -> int:
        """Get the total number of bounding boxes."""
        return sum(
            len(region.bboxes)
            for sample in self._samples
            for region in sample.regions
        )

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the dataset statistics."""
        label_dist = self.get_label_distribution()
        return {
            "name": self.name,
            "num_samples": len(self),
            "num_regions": self.get_region_count(),
            "num_bboxes": self.get_bbox_count(),
            "unique_anatomies": self.get_unique_anatomies(),
            "num_unique_findings": len(self.get_unique_findings()),
            "label_distribution": label_dist,
        }
