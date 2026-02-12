from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class ExperimentType(str, Enum):
    """Supported experiment types for the benchmark."""

    BASELINE = "baseline"
    OCO = "oco"  # Object Class Occlusion - occlude annotated regions
    ROCO = "roco"  # Random Object Class Occlusion - occlude random regions

    @classmethod
    def choices(cls) -> List[str]:
        return [e.value for e in cls]


@dataclass
class BenchmarkConfig:
    """Configuration for running benchmark experiments."""

    model_key: str
    data_dir: Path
    experiment: str
    output_path: Path
    data_json: Optional[Path] = None
    cache_dir: Path = field(default_factory=lambda: Path("./model_caching"))
    prompt_key: Optional[str] = None
    max_images: Optional[int] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    trust_remote_code: bool = False
    occlusion_strength: float = 1.0  # p value for OCO/ROCO (0.0 to 1.0)
    filter_labels: Optional[List[str]] = None  # Filter dataset by CheXpert labels
    num_images: int = 1  # Number of images per inference (1 = single, >1 = multi-image)
    seed: int = 3  # Random seed for reproducible preprocessing (OCO/ROCO/DOCO)

    @property
    def experiment_type(self) -> ExperimentType:
        """Parse experiment string to ExperimentType."""
        base_exp = self.experiment.split("_p")[0].lower()
        return ExperimentType(base_exp)

    @property
    def parsed_occlusion_strength(self) -> float:
        """Parse occlusion strength from experiment string like 'oco_p50'."""
        if "_p" in self.experiment:
            try:
                p_str = self.experiment.split("_p")[1]
                return min(100, max(0, int(p_str))) / 100.0
            except (IndexError, ValueError):
                pass
        return self.occlusion_strength


@dataclass
class InferenceResult:
    """Result from running inference on a single image."""

    image_path: str
    model_key: str
    model_id: str
    prompt_key: str
    prompt_text: str
    response_text: str
    labels: Optional[List[int]] = None
    ground_truth_report: Optional[str] = None
