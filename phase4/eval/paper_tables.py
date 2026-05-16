"""Generate CSV tables from Phase 4 prediction artifacts.

Inputs (read-only):
- ``phase4_output/predictions/val/<model>/<regime>.jsonl``     (val predictions)
- ``phase4_output/predictions/test_blind/<model>/<regime>.jsonl`` (test, blind)
- ``phase4_output/val_metrics/<model>/<regime>.json``           (val metrics)
- ``phase4_output/efficiency/<model>/<regime>.json``            (latency/size)
- per-seed run metadata

Outputs:
- ``phase4_output/paper_tables/main_table.csv``         (val, primary seed)
- ``phase4_output/paper_tables/cross_domain_table.csv`` (subset_domain split)
- ``phase4_output/paper_tables/oracle_headroom.csv``    (oracle vs systems)
- ``phase4_output/paper_tables/calibration.csv``        (reliability bins)
- ``phase4_output/paper_tables/seed_variance.csv``      (mean +/- std across
   seeds when available)
- ``phase4_output/paper_tables/significance.csv``       (paired bootstrap)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

from phase4.eval.metrics import (
    aggregate_metrics,
    calibration_bins,
    expected_calibration_error,
    oracle_per_sentence,
    paired_bootstrap,
    per_domain_breakdown,
    rare_and_name_corruption,
)


MODELS = ("identity", "classical", "neural", "hybrid")
REGIMES = ("real_only", "synthetic_only", "synthetic_plus_real")


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _maybe_read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _val_predictions_path(out_dir: Path, model: str, regime: str, seed: Optional[int] = None) -> Path:
    if seed is None:
        return out_dir / "predictions" / "val" / model / f"{regime}.jsonl"
    return out_dir / "predictions" / "val" / model / f"{regime}__seed{seed}.jsonl"


def _test_predictions_path(out_dir: Path, model: str, regime: str, seed: Optional[int] = None) -> Path:
    if seed is None:
        return out_dir / "predictions" / "test_blind" / model / f"{regime}.jsonl"
    return out_dir / "predictions" / "test_blind" / model / f"{regime}__seed{seed}.jsonl"


def _val_metrics_path(out_dir: Path, model: str, regime: str, seed: Optional[int] = None) -> Path:
    if seed is None:
        return out_dir / "val_metrics" / model / f"{regime}.json"
    return out_dir / "val_metrics" / model / f"{regime}__seed{seed}.json"


def _efficiency_path(out_dir: Path, model: str, regime: str, seed: Optional[int] = None) -> Path:
    if seed is None:
        return out_dir / "efficiency" / model / f"{regime}.json"
    return out_dir / "efficiency" / model / f"{regime}__seed{seed}.json"




def build_main_table(out_dir: Path, primary_seed: int = 42) -> Path:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        for regime in REGIMES:
            preds = _read_jsonl(_val_predictions_path(out_dir, model, regime, seed=primary_seed))
            if not preds:
                preds = _read_jsonl(_val_predictions_path(out_dir, model, regime))
            if not preds:
                continue
            agg = aggregate_metrics(preds)
            rare = rare_and_name_corruption(preds)
            eff = _maybe_read_json(_efficiency_path(out_dir, model, regime, seed=primary_seed))
            if not eff:
                eff = _maybe_read_json(_efficiency_path(out_dir, model, regime))
            row = {
                "model": model,
                "regime": regime,
                "n": agg["n_records"],
                "input_cer": round(agg.get("input_cer", 0.0), 5),
                "cer": round(agg["cer"], 5),
                "cer_reduction_rate": round(agg.get("cer_reduction_rate", 0.0), 5),
                "wer": round(agg["wer"], 5),
                "chrf": round(agg["chrf"], 5),
                "sentence_accuracy": round(agg["sentence_accuracy"], 5),
                "correction_rate": round(agg["correction_rate"], 5),
                "overcorrection_rate": round(agg["overcorrection_rate"], 5),
                "rare_word_corruption_rate": round(rare["rare_word_corruption_rate"], 5),
                "proper_name_corruption_rate": round(rare["proper_name_corruption_rate"], 5),
                "median_ms_per_sentence": eff.get("median_ms_per_sentence"),
                "p95_ms_per_sentence": eff.get("p95_ms_per_sentence"),
                "n_params": eff.get("n_params"),
            }
            rows.append(row)
    out_path = out_dir / "paper_tables" / "main_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return out_path


def build_cross_domain_table(out_dir: Path, primary_seed: int = 42) -> Path:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        for regime in REGIMES:
            preds = _read_jsonl(_val_predictions_path(out_dir, model, regime, seed=primary_seed))
            if not preds:
                preds = _read_jsonl(_val_predictions_path(out_dir, model, regime))
            if not preds:
                continue
            domain_break = per_domain_breakdown(preds)
            for d, m in domain_break.items():
                rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        "subset_domain": d,
                        "n": m["n_records"],
                        "cer": round(m["cer"], 5),
                        "wer": round(m["wer"], 5),
                        "chrf": round(m["chrf"], 5),
                        "sentence_accuracy": round(m["sentence_accuracy"], 5),
                    }
                )
    out_path = out_dir / "paper_tables" / "cross_domain_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return out_path


def build_calibration_table(out_dir: Path, primary_seed: int = 42) -> Path:
    rows: List[Dict[str, object]] = []
    for model in MODELS:
        for regime in REGIMES:
            preds = _read_jsonl(_val_predictions_path(out_dir, model, regime, seed=primary_seed))
            if not preds:
                preds = _read_jsonl(_val_predictions_path(out_dir, model, regime))
            if not preds:
                continue
            bins = calibration_bins(preds, n_bins=10)
            ece = expected_calibration_error(bins)
            for b in bins:
                rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        "ece": round(ece, 5),
                        "bin_lo": b["lo"],
                        "bin_hi": b["hi"],
                        "count": b["count"],
                        "mean_confidence": round(b["mean_confidence"], 5),
                        "mean_accuracy": round(b["mean_accuracy"], 5),
                    }
                )
    out_path = out_dir / "paper_tables" / "calibration.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return out_path


def build_oracle_table(out_dir: Path, primary_seed: int = 42) -> Path:
    rows: List[Dict[str, object]] = []
    for regime in REGIMES:
        per_model_preds: Dict[str, List[Dict[str, object]]] = {}
        ref_inputs: List[str] = []
        noisy_inputs: List[str] = []
        for model in ("classical", "neural", "hybrid"):
            preds = _read_jsonl(_val_predictions_path(out_dir, model, regime, seed=primary_seed))
            if not preds:
                preds = _read_jsonl(_val_predictions_path(out_dir, model, regime))
            if not preds:
                continue
            per_model_preds[model] = preds
            if not ref_inputs:
                ref_inputs = [str(r.get("reference", "")) for r in preds]
                noisy_inputs = [str(r.get("input_noisy", "")) for r in preds]
        if not per_model_preds:
            continue
        oracle = oracle_per_sentence(per_model_preds, ref_inputs, noisy_inputs)
        rows.append(
            {
                "regime": regime,
                "n": oracle["n"],
                "oracle_cer": round(oracle["oracle_cer"], 5),
                "oracle_wer": round(oracle["oracle_wer"], 5),
            }
        )
    out_path = out_dir / "paper_tables" / "oracle_headroom.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return out_path


def build_significance_table(out_dir: Path, primary_seed: int = 42) -> Path:
    rows: List[Dict[str, object]] = []
    for regime in REGIMES:
        identity = _read_jsonl(_val_predictions_path(out_dir, "identity", regime, seed=primary_seed))
        if not identity:
            identity = _read_jsonl(_val_predictions_path(out_dir, "identity", regime))
        if not identity:
            continue
        identity_cer = [float(r.get("cer", 0.0)) for r in identity]
        identity_wer = [float(r.get("wer", 0.0)) for r in identity]
        classical = _read_jsonl(_val_predictions_path(out_dir, "classical", regime, seed=primary_seed))
        if not classical:
            classical = _read_jsonl(_val_predictions_path(out_dir, "classical", regime))
        for system in ("classical", "neural", "hybrid"):
            preds = _read_jsonl(_val_predictions_path(out_dir, system, regime, seed=primary_seed))
            if not preds:
                preds = _read_jsonl(_val_predictions_path(out_dir, system, regime))
            if not preds or len(preds) != len(identity):
                continue
            sys_cer = [float(r.get("cer", 0.0)) for r in preds]
            sys_wer = [float(r.get("wer", 0.0)) for r in preds]
            cer_test = paired_bootstrap(identity_cer, sys_cer)
            wer_test = paired_bootstrap(identity_wer, sys_wer)
            row = {
                "regime": regime,
                "system": system,
                "vs_baseline": "identity",
                "n": cer_test["n"],
                "cer_mean_delta": round(cer_test["mean_delta"], 6),
                "cer_ci_low": round(cer_test["ci_low"], 6),
                "cer_ci_high": round(cer_test["ci_high"], 6),
                "cer_p_value": round(cer_test["p_value"], 5),
                "wer_mean_delta": round(wer_test["mean_delta"], 6),
                "wer_p_value": round(wer_test["p_value"], 5),
            }
            rows.append(row)
            if classical and system != "classical" and len(classical) == len(preds):
                cl_cer = [float(r.get("cer", 0.0)) for r in classical]
                cl_wer = [float(r.get("wer", 0.0)) for r in classical]
                cer_test2 = paired_bootstrap(cl_cer, sys_cer)
                wer_test2 = paired_bootstrap(cl_wer, sys_wer)
                rows.append(
                    {
                        "regime": regime,
                        "system": system,
                        "vs_baseline": "classical",
                        "n": cer_test2["n"],
                        "cer_mean_delta": round(cer_test2["mean_delta"], 6),
                        "cer_ci_low": round(cer_test2["ci_low"], 6),
                        "cer_ci_high": round(cer_test2["ci_high"], 6),
                        "cer_p_value": round(cer_test2["p_value"], 5),
                        "wer_mean_delta": round(wer_test2["mean_delta"], 6),
                        "wer_p_value": round(wer_test2["p_value"], 5),
                    }
                )
    out_path = out_dir / "paper_tables" / "significance.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return out_path


def build_seed_variance_table(out_dir: Path, seeds: List[int]) -> Path:
    if not seeds:
        out_path = out_dir / "paper_tables" / "seed_variance.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for model in MODELS:
        for regime in REGIMES:
            for seed in seeds:
                preds = _read_jsonl(_val_predictions_path(out_dir, model, regime, seed=seed))
                if not preds:
                    continue
                agg = aggregate_metrics(preds)
                key = f"{model}|{regime}"
                grouped[key]["cer"].append(agg["cer"])
                grouped[key]["wer"].append(agg["wer"])
                grouped[key]["chrf"].append(agg["chrf"])
                grouped[key]["sentence_accuracy"].append(agg["sentence_accuracy"])
    rows: List[Dict[str, object]] = []
    for key, metrics in grouped.items():
        model, regime = key.split("|", 1)
        rows.append(
            {
                "model": model,
                "regime": regime,
                "n_seeds": len(metrics.get("cer", [])),
                "cer_mean": round(mean(metrics["cer"]), 5) if metrics.get("cer") else None,
                "cer_std": round(pstdev(metrics["cer"]), 5) if len(metrics.get("cer", [])) > 1 else 0.0,
                "wer_mean": round(mean(metrics["wer"]), 5) if metrics.get("wer") else None,
                "wer_std": round(pstdev(metrics["wer"]), 5) if len(metrics.get("wer", [])) > 1 else 0.0,
                "chrf_mean": round(mean(metrics["chrf"]), 5) if metrics.get("chrf") else None,
                "sent_acc_mean": round(mean(metrics["sentence_accuracy"]), 5) if metrics.get("sentence_accuracy") else None,
            }
        )
    out_path = out_dir / "paper_tables" / "seed_variance.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return out_path


def build_alignment_quality_table(
    phase1_output_dir: Path,
    out_dir: Path,
) -> Path:
    """Export per-book Phase 1 alignment_quality.json rows to CSV."""
    src = phase1_output_dir / "alignment_quality_all.json"
    out_path = out_dir / "paper_tables" / "phase1_alignment_quality.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    if src.exists():
        rows = json.loads(src.read_text(encoding="utf-8"))
    else:
        for aq in phase1_output_dir.glob("*/*/alignment_quality.json"):
            data = json.loads(aq.read_text(encoding="utf-8"))
            data["book"] = aq.parent.name
            data["split"] = aq.parent.parent.name
            rows.append(data)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def build_all_tables(
    out_dir: Path,
    primary_seed: int = 42,
    all_seeds: Optional[List[int]] = None,
    phase1_output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    paths = {
        "main": build_main_table(out_dir, primary_seed=primary_seed),
        "cross_domain": build_cross_domain_table(out_dir, primary_seed=primary_seed),
        "calibration": build_calibration_table(out_dir, primary_seed=primary_seed),
        "oracle": build_oracle_table(out_dir, primary_seed=primary_seed),
        "significance": build_significance_table(out_dir, primary_seed=primary_seed),
        "seed_variance": build_seed_variance_table(out_dir, all_seeds or []),
    }
    if phase1_output_dir is not None:
        paths["alignment_quality"] = build_alignment_quality_table(
            phase1_output_dir, out_dir
        )
    return paths
