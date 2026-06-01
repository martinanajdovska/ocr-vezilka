"""Atomic file-write helpers for crash-safe artefact persistence.

Each helper writes to ``path.with_suffix(path.suffix + ".tmp")`` first
and then performs an ``os.replace`` to atomically move it into place.
This kills the "half-written JSON on crash".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable


def _tmp_path(path: Path) -> Path:
    """Return a temp sibling path; same dir so ``os.replace`` is atomic."""
    return path.with_suffix(path.suffix + ".tmp")


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    atomic_write_text(
        path, json.dumps(data, ensure_ascii=False, indent=indent) + "\n"
    )


def atomic_write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write JSONL atomically; return the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    n = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)
    return n


def count_jsonl_lines(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
