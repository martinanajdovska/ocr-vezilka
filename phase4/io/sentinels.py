"""Crash-resume sentinel helpers.

Granularity:

- ``_DONE.json``        -- per (model, regime, seed) inside its model_dir
- ``_FOLD_DONE.json``   -- per k-fold (under ``kfold/foldN/``)
- ``_SWEEP_STANDARD_DONE.json`` / ``_SWEEP_KFOLD_DONE.json`` -- per leg
- ``_PIPELINE_DONE.json`` -- both legs of the chained pipeline

All writes are atomic via :mod:`phase4.io.atomic`.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Any, Dict, Optional

from phase4.io.atomic import atomic_write_text


def force_rerun_active() -> bool:
    """Whether ``PHASE4_FORCE_RERUN=1`` was set on the CLI."""
    return os.environ.get("PHASE4_FORCE_RERUN", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


def read_sentinel(path: Path) -> Optional[Dict[str, Any]]:
    """Return the parsed sentinel dict at ``path``, or ``None`` if absent.

    A malformed sentinel (corrupted JSON) is also treated as absent so
    the stage re-runs and re-stamps a clean sentinel.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_sentinel(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2)
    )


def sentinel_matches(
    sentinel: Optional[Dict[str, Any]],
    *,
    hyperparameter_hash: str,
    split_manifest_hash: str,
) -> bool:
    """True iff the sentinel's stored hashes match the current run's."""
    if sentinel is None:
        return False
    if str(sentinel.get("hyperparameter_hash")) != str(hyperparameter_hash):
        return False
    if str(sentinel.get("split_manifest_hash")) != str(split_manifest_hash):
        return False
    return True
