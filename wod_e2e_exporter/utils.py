"""Shared utilities for WOD-E2E exporter (path/json/csv/log helpers)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """Create directory with parents if missing."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    """Safe json dump with ASCII for file system stability."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def listify(x: Any) -> Iterable[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def safe_float(val: Any) -> float | None:
    try:
        return float(val)
    except Exception:
        return None


def win_to_wsl_path(p: str | Path) -> str:
    """
    Convert Windows-style path like C:\\Datasets\\... to /mnt/c/Datasets/... for WSL.
    If already POSIX, return as-is.
    """
    if p is None:
        return p
    p = str(p)
    if p.startswith("/mnt/") or p.startswith("/"):
        return p.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p.replace("\\", "/")
