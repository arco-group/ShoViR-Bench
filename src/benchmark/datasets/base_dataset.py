"""Base dataset classes for the benchmark."""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

CHEXPERT_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CHEXPERT_LABELS)}


@dataclass
class BoundingBox:
    """Normalized bounding box coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_absolute(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute pixel coordinates."""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
        )

    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])


@dataclass
class Region:
    """A region annotation with bounding boxes and findings."""

    anatomy: str
    bboxes: List[BoundingBox]
    findings: List[str]
    chexpert_categories: List[str]
    sentence: str
    labels: List[int]

    def get_active_chexpert_labels(self) -> List[str]:
        """Return the CheXpert labels that are active (value=1) for this region."""
        return [
            CHEXPERT_LABELS[i]
            for i, val in enumerate(self.labels)
            if val == 1
        ]


@dataclass
class Sample:
    """A single dataset sample with image, report, and region annotations."""

    image_id: str
    img_path: str
    report: str
    labels: List[int]
    regions: List[Region] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_active_chexpert_labels(self) -> List[str]:
        """Return the CheXpert labels that are active (value=1) for this sample."""
        return [
            CHEXPERT_LABELS[i]
            for i, val in enumerate(self.labels)
            if val == 1
        ]

    def has_label(self, label: str) -> bool:
        """Check if a specific CheXpert label is active."""
        idx = CHEXPERT_LABEL_TO_INDEX.get(label)
        if idx is None:
            return False
        return self.labels[idx] == 1


class BaseDataset(abc.ABC):
    """Abstract base class for benchmark datasets."""

    def __init__(
        self,
        data_dir: Path | str,
        annotation_file: Path | str | None = None,
        transform: Optional[Any] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self.transform = transform
        self._samples: List[Sample] = []
        self._load_annotations()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the dataset name."""
        ...

    @abc.abstractmethod
    def _load_annotations(self) -> None:
        """Load annotations from file into self._samples."""
        ...

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Sample]:
        sample = self._samples[idx]
        image_path = self.data_dir / sample.img_path
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, sample

    def __iter__(self) -> Iterator[Tuple[Image.Image, Sample]]:
        for idx in range(len(self)):
            yield self[idx]

    def get_sample(self, idx: int) -> Sample:
        """Get sample metadata without loading the image."""
        return self._samples[idx]

    def get_samples(self) -> List[Sample]:
        """Get all samples without loading images."""
        return self._samples

    def filter_by_label(self, label: str) -> List[Sample]:
        """Filter samples that have a specific CheXpert label active."""
        return [s for s in self._samples if s.has_label(label)]

    def get_label_distribution(self) -> Dict[str, int]:
        """Get the count of samples for each CheXpert label."""
        distribution = {label: 0 for label in CHEXPERT_LABELS}
        for sample in self._samples:
            for label in sample.get_active_chexpert_labels():
                distribution[label] += 1
        return distribution

    def iter_with_paths(self) -> Iterator[Tuple[Path, Image.Image, Sample]]:
        """Iterate yielding (image_path, image, sample) tuples."""
        for idx in range(len(self)):
            sample = self._samples[idx]
            image_path = self.data_dir / sample.img_path
            image = Image.open(image_path)
            if self.transform is not None:
                image = self.transform(image)
            yield image_path, image, sample
