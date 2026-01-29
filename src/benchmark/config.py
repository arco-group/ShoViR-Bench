from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkConfig:
    model_key: str
    data_json: Path
    data_dir: Path
    experiment: str
    output_path: Path
    data_json: Optional[Path] = None
    cache_dir: Path = Path("./model_caching")
    prompt_key: Optional[str] = None
    max_images: Optional[int] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    trust_remote_code: bool = False


@dataclass
class InferenceResult:
    image_path: str
    model_key: str
    model_id: str
    prompt_key: str
    prompt_text: str
    response_text: str
