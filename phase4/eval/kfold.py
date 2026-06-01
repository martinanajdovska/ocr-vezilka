"""k-fold aggregation utilities.

Reads ``phase4_output/kfold/foldN/paper_tables/main_table.csv`` per fold
and aggregates the metrics into a single
``phase4_output/paper_tables/kfold_summary.csv`` with mean +/- std
across folds per (model, regime).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional


_FIELDS_TO_AGG = (
    "cer",
    "cer_reduction_rate",
    "wer",
    "chrf",
    "sentence_accuracy",
    "correction_rate",
    "overcorrection_rate",
    "useful_correction_rate",
)


def _read_main_table(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def build_kfold_table(
    out_dir: Path,
    n_folds: int,
    primary_seed: int = 42,
) -> Path:
    """Aggregate per-fold ``main_table.csv`` into one summary CSV.

    Per (model, regime), reports mean and std across folds for every
    metric in :data:`_FIELDS_TO_AGG`. Skips models/regimes that lack
    any fold rows.
    """
    by_key: Dict[str, Dict[str, List[float]]] = {}
    for fold_idx in range(n_folds):
        main_path = (
            out_dir / "kfold" / f"fold{fold_idx}" / "paper_tables" / "main_table.csv"
        )
        rows = _read_main_table(main_path)
        for row in rows:
            model = str(row.get("model") or "")
            regime = str(row.get("regime") or "")
            if not model or not regime:
                continue
            key = f"{model}|{regime}"
            slot = by_key.setdefault(key, {f: [] for f in _FIELDS_TO_AGG})
            for f in _FIELDS_TO_AGG:
                v = row.get(f)
                if v is None or v == "":
                    continue
                try:
                    slot[f].append(float(v))
                except (TypeError, ValueError):
                    continue

    out_path = out_dir / "paper_tables" / "kfold_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "regime", "n_folds"]
    for f in _FIELDS_TO_AGG:
        fieldnames.extend([f"{f}_mean", f"{f}_std"])

    summary_rows: List[Dict[str, object]] = []
    for key in sorted(by_key.keys()):
        model, regime = key.split("|", 1)
        slot = by_key[key]
        n = len(slot["cer"]) if "cer" in slot else 0
        row: Dict[str, object] = {"model": model, "regime": regime, "n_folds": n}
        for f in _FIELDS_TO_AGG:
            values = slot.get(f) or []
            if not values:
                row[f"{f}_mean"] = None
                row[f"{f}_std"] = None
                continue
            row[f"{f}_mean"] = round(mean(values), 6)
            row[f"{f}_std"] = round(
                pstdev(values) if len(values) > 1 else 0.0, 6
            )
        summary_rows.append(row)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    # Also drop a small manifest pointing at each fold's main_table.csv.
    manifest = {
        "n_folds": int(n_folds),
        "primary_seed": int(primary_seed),
        "folds": [
            {
                "fold": i,
                "main_table": str(
                    out_dir / "kfold" / f"fold{i}" / "paper_tables" / "main_table.csv"
                ),
                "fold_done": str(
                    out_dir / "kfold" / f"fold{i}" / "_FOLD_DONE.json"
                ),
            }
            for i in range(n_folds)
        ],
        "summary": str(out_path),
    }
    (out_dir / "paper_tables" / "kfold_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path
