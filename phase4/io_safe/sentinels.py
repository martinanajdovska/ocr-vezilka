"""Crash-resume sentinel helpers.

Granularity:

- ``_DONE.json``        -- per (model, regime, seed) inside its model_dir
- ``_FOLD_DONE.json``   -- per k-fold (under ``kfold/foldN/``)
- ``_SWEEP_STANDARD_DONE.json`` / ``_SWEEP_KFOLD_DONE.json`` -- per leg
- ``_PIPELINE_DONE.json`` -- both legs of the chained pipeline

All writes are atomic via :mod:`phase4.io_safe.atomic`.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Any, Dict, List, Optional

from phase4.io_safe.atomic import atomic_write_text


def seed_suffix_for(primary_seed: int, seed: int) -> str:
    return "" if seed == primary_seed else f"__seed{seed}"


def expected_run_artefacts(
    out_dir: Path,
    model_name: str,
    regime: str,
    seed: int,
    primary_seed: int,
) -> Dict[str, Path]:
    """On-disk outputs that must exist before a (model, regime, seed) is
    considered complete. Used by crash-resume; do not trust ``_DONE.json``
    alone (an empty ``artefacts`` block used to short-circuit the whole run).
    """
    sfx = seed_suffix_for(primary_seed, seed)
    return {
        "val_predictions": out_dir / "predictions" / "val" / model_name / f"{regime}{sfx}.jsonl",
        "test_predictions": out_dir / "predictions" / "test_blind" / model_name / f"{regime}{sfx}.jsonl",
        "val_metrics": out_dir / "val_metrics" / model_name / f"{regime}{sfx}.json",
        "efficiency": out_dir / "efficiency" / model_name / f"{regime}{sfx}.json",
    }


def missing_artefacts(artefacts: Dict[str, Path]) -> List[str]:
    """Return human-readable descriptions of missing or empty artefact paths."""
    missing: List[str] = []
    for key, path in artefacts.items():
        if not path.exists():
            missing.append(f"{key}: {path}")
            continue
        if path.suffix == ".jsonl":
            try:
                if path.stat().st_size == 0:
                    missing.append(f"{key}: empty {path}")
            except OSError:
                missing.append(f"{key}: unreadable {path}")
    return missing


def row_from_artefacts(
    artefacts: Dict[str, Path],
    *,
    model_name: str,
    regime: str,
    seed: int,
    primary_seed: int,
    checkpoint_dir: Path,
) -> Dict[str, object]:
    return {
        "model": model_name,
        "regime": regime,
        "status": "success",
        "seed": seed,
        "primary_seed": primary_seed,
        "val_predictions": str(artefacts["val_predictions"]),
        "test_predictions": str(artefacts["test_predictions"]),
        "val_metrics": str(artefacts["val_metrics"]),
        "efficiency": str(artefacts["efficiency"]),
        "checkpoint_dir": str(checkpoint_dir),
    }


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
