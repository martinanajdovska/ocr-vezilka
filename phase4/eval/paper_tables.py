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




def _hybrid_calibration_summary(out_dir: Path, regime: str, seed: int) -> Dict[str, object]:
    """surface hybrid's tuned fusion weights / margins so
    main_table.csv documents which calibration produced each hybrid row.
    """
    candidates = [
        out_dir / "models" / "hybrid" / regime / "hybrid_calibration_summary.json",
        out_dir / "models" / "hybrid" / regime / f"seed{seed}" / "hybrid_calibration_summary.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
    return {}


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
                "useful_correction_rate": round(agg.get("useful_correction_rate", 0.0), 5),
                "edited_count": int(agg.get("edited_count", 0) or 0),
                "rare_word_corruption_rate": round(rare["rare_word_corruption_rate"], 5),
                "proper_name_corruption_rate": round(rare["proper_name_corruption_rate"], 5),
                "median_ms_per_sentence": eff.get("median_ms_per_sentence"),
                "p95_ms_per_sentence": eff.get("p95_ms_per_sentence"),
                "n_params": eff.get("n_params"),
            }
            if model == "hybrid":
                summary = _hybrid_calibration_summary(out_dir, regime, primary_seed)
                selected = summary.get("selected") if isinstance(summary, dict) else None
                if isinstance(selected, dict):
                    row["hybrid_lambda_class"] = selected.get("lambda_class")
                    row["hybrid_lambda_neural"] = selected.get("lambda_neural")
                    row["hybrid_margin"] = selected.get("margin")
            rows.append(row)
    out_path = out_dir / "paper_tables" / "main_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        # Union all keys (some rows may have hybrid-only fields).
        fieldnames: List[str] = []
        seen: set = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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


def _read_temperature_summary(out_dir: Path, model: str, regime: str, seed: int) -> Dict[str, object]:
    """Locate a neural_temperature.json for the given run, regardless of
    whether the model is standalone ``neural`` or the inner neural of
    ``hybrid``. Returns ``{}`` if no calibration was performed.
    """
    candidates = []
    if model == "neural":
        candidates.append(out_dir / "models" / model / regime / "neural_temperature.json")
        candidates.append(out_dir / "models" / model / regime / f"seed{seed}" / "neural_temperature.json")
    elif model == "hybrid":
        candidates.append(out_dir / "models" / model / regime / "neural_assets" / "neural_temperature.json")
        candidates.append(
            out_dir / "models" / model / regime / f"seed{seed}" / "neural_assets" / "neural_temperature.json"
        )
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


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
            temp_summary = _read_temperature_summary(out_dir, model, regime, primary_seed)
            temp_value = temp_summary.get("temperature")
            pre_ece_tok = temp_summary.get("pre_ece")
            post_ece_tok = temp_summary.get("post_ece")
            for b in bins:
                rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        # Sentence-level reliability bins (existing).
                        "ece": round(ece, 5),
                        "bin_lo": b["lo"],
                        "bin_hi": b["hi"],
                        "count": b["count"],
                        "mean_confidence": round(b["mean_confidence"], 5),
                        "mean_accuracy": round(b["mean_accuracy"], 5),
                        # token-level ECE before/after temperature scaling
                        # (populated for neural/hybrid; identity/classical leave NaN).
                        "temperature": round(float(temp_value), 5) if temp_value is not None else None,
                        "token_ece_pre": round(float(pre_ece_tok), 5) if pre_ece_tok is not None else None,
                        "token_ece_post": round(float(post_ece_tok), 5) if post_ece_tok is not None else None,
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
    # the primary seed writes ``<regime>.jsonl`` (no suffix); only
    # secondary seeds write ``<regime>__seed{N}.jsonl``. Reading per-seed
    # always with a ``seed=`` suffix would silently skip the primary seed.
    primary = min(seeds)  # convention: primary == smallest
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for model in MODELS:
        for regime in REGIMES:
            for seed in seeds:
                if seed == primary:
                    preds = _read_jsonl(
                        _val_predictions_path(out_dir, model, regime)
                    )
                    if not preds:
                        # Identity model writes the suffixed form even for
                        # the primary seed (see ``_run_one`` L468).
                        preds = _read_jsonl(
                            _val_predictions_path(out_dir, model, regime, seed=seed)
                        )
                else:
                    preds = _read_jsonl(
                        _val_predictions_path(out_dir, model, regime, seed=seed)
                    )
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


def build_headroom_curve(out_dir: Path, primary_seed: int = 42, n_bins: int = 10) -> Path:
    """per-bin headroom curve for the headline figure.

    Bins val records of the *neural* runs by ``headroom_estimate_cer``
    (equal-count quantile bins), then reports mean ``input_cer``,
    ``cer``, and the model's ``error_reduction`` per bin. Emits a single
    CSV with one row per (regime, bin); ``regime == "*"`` adds an
    aggregated row across all regimes for the headline plot.

    Returns the path even when no neural predictions are available
    (writes a CSV with just the header so downstream plotters can skip
    cleanly).
    """
    out_path = out_dir / "paper_tables" / "headroom_curve.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "regime",
        "bin",
        "bin_lo",
        "bin_hi",
        "n",
        "mean_estimate",
        "mean_input_cer",
        "mean_output_cer",
        "mean_error_reduction",
        "headroom_skip_rate",
    ]
    rows: List[Dict[str, object]] = []

    def _bin_for(regime_label: str, records: List[Dict[str, object]]) -> None:
        kept = [
            r for r in records
            if r.get("headroom_estimate_cer") is not None
            and r.get("input_cer") is not None
            and r.get("cer") is not None
        ]
        if not kept:
            return
        estimates = sorted(float(r["headroom_estimate_cer"]) for r in kept)
        # Equal-count quantile bin edges.
        n = len(estimates)
        edges = [
            estimates[min(n - 1, max(0, int(round(i * n / n_bins))))]
            for i in range(n_bins + 1)
        ]
        # Ensure strictly increasing edges so the bin lookup is well-defined.
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12
        for b in range(n_bins):
            lo = edges[b]
            hi = edges[b + 1]
            members = [
                r for r in kept
                if (lo <= float(r["headroom_estimate_cer"]) <= hi)
                and (b > 0 or float(r["headroom_estimate_cer"]) >= lo)
            ]
            if not members:
                continue
            n_skip = sum(1 for r in members if bool(r.get("headroom_skipped")))
            rows.append({
                "regime": regime_label,
                "bin": b,
                "bin_lo": round(float(lo), 6),
                "bin_hi": round(float(hi), 6),
                "n": len(members),
                "mean_estimate": round(
                    sum(float(r["headroom_estimate_cer"]) for r in members)
                    / len(members),
                    6,
                ),
                "mean_input_cer": round(
                    sum(float(r["input_cer"]) for r in members)
                    / len(members),
                    6,
                ),
                "mean_output_cer": round(
                    sum(float(r["cer"]) for r in members) / len(members),
                    6,
                ),
                "mean_error_reduction": round(
                    sum(
                        float(r["input_cer"]) - float(r["cer"])
                        for r in members
                    ) / len(members),
                    6,
                ),
                "headroom_skip_rate": round(n_skip / len(members), 6),
            })

    all_neural: List[Dict[str, object]] = []
    for regime in REGIMES:
        preds = _read_jsonl(
            _val_predictions_path(out_dir, "neural", regime, seed=primary_seed)
        )
        if not preds:
            preds = _read_jsonl(_val_predictions_path(out_dir, "neural", regime))
        if not preds:
            continue
        _bin_for(regime, preds)
        all_neural.extend(preds)
    if all_neural:
        _bin_for("*", all_neural)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        if rows:
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
        "headroom_curve": build_headroom_curve(out_dir, primary_seed=primary_seed),
    }
    if phase1_output_dir is not None:
        paths["alignment_quality"] = build_alignment_quality_table(
            phase1_output_dir, out_dir
        )
    return paths
