from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

from PIL import Image


def get_tmp_dir() -> Path:
    """Return a suitable temporary directory for large intermediate files.

    Priority:
      1. $TMPDIR  — set by SLURM per job (e.g. /local/tmp.<jobid>)
      2. ./.tmp   — fallback in the current working directory
    """
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        p = Path(tmpdir)
        if p.exists():
            return p
    p = Path(".tmp")
    p.mkdir(parents=True, exist_ok=True)
    return p


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory is not a folder: {data_dir}")
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def open_image(path: Path) -> Image.Image:
    image = Image.open(path)
    return image


def iter_images(paths: Iterable[Path]):
    for path in paths:
        yield path, open_image(path)
