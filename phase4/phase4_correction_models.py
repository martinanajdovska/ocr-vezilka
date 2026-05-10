"""Phase 4 runner.

1. (Re)build train-only Phase 2 statistics + Phase 3 synthetic noise so the
   synthetic regimes never see val/test confusion distributions.
2. (Re)build the Phase 4 manifests with banded two-level alignment.
3. For each (model, regime, seed) triple, train the model under the regime,
   predict on val (with metrics) and test_blind (without metrics), persist
   models / predictions / metrics / efficiency.
4. Generate identity baseline and per-regime cross-domain (A->B) predictions.
5. Aggregate paper-table CSVs and a top-level run manifest.

All training-time decisions read only train rows. Hybrid calibration reads
only val rows. Test rows are predicted, but never inspected for parameter or
threshold selection (asserted at runtime).

Smoke / GPU check (from repo root, after Phase 1--3 train-only artifacts exist):
``python -u phase4/phase4_correction_models.py --skip-phase1 --skip-phase2-stats
--skip-phase3-noise --skip-classical --skip-hybrid --regimes real_only
--seeds 42 --neural-device cuda --predict-sample 200``. Optionally capture
stdout and grep for ``train_loss=nan`` or ``Non-finite training loss``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from phase4.config import (
    SEEDS,
    ClassicalConfig,
    HybridConfig,
    TransformerConfig,
    default_run_config,
    frozen_hparams_dict,
)
from phase4.data.build_phase4_dataset import build_phase4_manifests, load_jsonl
from phase4.data.splits import (
    SPLITS,
    assert_disjoint_splits,
    split_manifest_hash,
)
from phase4.eval.metrics import (
    aggregate_metrics,
    cer,
    cer_reduction_rate,
    chrf_score,
    expected_calibration_error,
    calibration_bins,
    identity_baseline_records,
    is_proper_name,
    is_rare_word,
    per_domain_breakdown,
    rare_and_name_corruption,
    wer,
)
from phase4.eval.paper_tables import build_all_tables
from phase4.io.schemas import validate_prediction_records
from phase4.models.classical import ClassicalCorrector
from phase4.models.hybrid import HybridCorrector
from phase4.models.transformer_seq2seq import ByteTransformerCorrector


def _phase2_train_only(phase2_dir: Path) -> bool:
    return (
        phase2_dir.exists()
        and (phase2_dir / "error_distribution.json").exists()
        and (phase2_dir / "error_confusion_probs.csv").exists()
        and (phase2_dir / "phase2_summary.json").exists()
    )


def _phase3_train_only(phase3_dir: Path) -> bool:
    structure = phase3_dir / "structure_aware_noise"
    return structure.exists() and any(structure.glob("*_synthetic.txt"))


def _ensure_train_only_stats(cfg) -> None:
    p2_ready = _phase2_train_only(cfg.phase2_train_only_dir)
    p3_ready = _phase3_train_only(cfg.phase3_train_only_dir)
    if cfg.force_rebuild_train_only_stats or not p2_ready:
        from phase2.ocr_error_analysis import run_train_only_mode as p2_train_only

        print("[phase4] (delegating to phase2) building train-only stats...", flush=True)
        p2_train_only(output_dir=cfg.phase2_train_only_dir)
    else:
        print(
            f"[phase4] reusing train-only Phase 2 artifacts in {cfg.phase2_train_only_dir}",
            flush=True,
        )
    if cfg.force_rebuild_train_only_stats or not p3_ready:
        from phase3.phase3_synthetic_noise import run_train_only_mode as p3_train_only

        print(
            "[phase4] (delegating to phase3) regenerating synthetic noise from train-only stats...",
            flush=True,
        )
        p3_train_only(
            phase2_train_only_dir=cfg.phase2_train_only_dir,
            output_dir=cfg.phase3_train_only_dir,
            seed=42,
        )
    else:
        print(
            f"[phase4] reusing train-only Phase 3 artifacts in {cfg.phase3_train_only_dir}",
            flush=True,
        )



def _to_sentence_pairs(rows: List[Dict[str, object]]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for row in rows:
        noisy = str(row["noisy"]).strip()
        clean = str(row["clean"]).strip()
        if noisy and clean:
            pairs.append((noisy, clean))
    return pairs


def _split_pairs_by_source(
    rows: List[Dict[str, object]],
    split: str,
    source_type: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Filter manifest rows by ``split`` (and optionally ``source_type``) and
    return ``(noisy, clean)`` pairs.

    Used by the ``synthetic_plus_real`` regime to pull out the real-only
    train rows for the finetune stage and the synthetic-only train rows for
    the pretrain stage.
    """
    sel: List[Dict[str, object]] = []
    for row in rows:
        if str(row.get("split")) != split:
            continue
        if source_type is not None and str(row.get("source_type", "")) != source_type:
            continue
        sel.append(row)
    return _to_sentence_pairs(sel)


def _train_neural(
    neural,
    rows: List[Dict[str, object]],
    regime: str,
    val_pairs: List[Tuple[str, str]],
) -> Dict[str, object]:
    """Centralised neural-training dispatch.

    For ``real_only`` and ``synthetic_only`` we just call ``fit`` once. For
    ``synthetic_plus_real`` we run a two-stage pretrain -> finetune pipeline:

    - Stage 1 (pretrain) sees only ``source_type == "synthetic"`` train rows.
    - Stage 2 (finetune) sees only ``source_type == "real"`` train rows and
      starts from the stage-1 checkpoint.

    The ``val_pairs`` argument always points to the regime's val rows
    (currently identical across regimes by construction; the runner passes
    them in to keep one source of truth). Hyperparameters are constant
    across stages: only the per-stage epoch budget changes.
    """
    if regime == "synthetic_plus_real":
        synth_pairs = _split_pairs_by_source(rows, split="train", source_type="synthetic")
        real_pairs = _split_pairs_by_source(rows, split="train", source_type="real")
        if not synth_pairs:
            print(
                "[neural] regime=synthetic_plus_real has no synthetic train rows; "
                "skipping pretrain stage and falling back to single-stage real fit.",
                flush=True,
            )
            return neural.fit(real_pairs, val_pairs, stage="train")
        if not real_pairs:
            print(
                "[neural] regime=synthetic_plus_real has no real train rows; "
                "running synthetic stage only.",
                flush=True,
            )
            return neural.fit(synth_pairs, val_pairs, stage="train")
        print(
            f"[neural] regime=synthetic_plus_real -> stage 1 (pretrain) "
            f"on {len(synth_pairs)} synthetic pairs",
            flush=True,
        )
        pretrain_metrics = neural.fit(synth_pairs, val_pairs, stage="pretrain")
        print(
            f"[neural] regime=synthetic_plus_real -> stage 2 (finetune) "
            f"on {len(real_pairs)} real pairs",
            flush=True,
        )
        finetune_metrics = neural.fit(real_pairs, val_pairs, stage="finetune")
        return {
            "stage": "pretrain_then_finetune",
            "pretrain": pretrain_metrics,
            "finetune": finetune_metrics,
            "n_params": finetune_metrics.get("n_params"),
            "device": finetune_metrics.get("device"),
            "best_val_loss": finetune_metrics.get("best_val_loss"),
            "total_train_seconds": (
                float(pretrain_metrics.get("total_train_seconds", 0.0))
                + float(finetune_metrics.get("total_train_seconds", 0.0))
            ),
        }
    train_pairs = _to_sentence_pairs([r for r in rows if r["split"] == "train"])
    return neural.fit(train_pairs, val_pairs, stage="train")


def _train_word_counts(rows: List[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        for tok in str(row["clean"]).split():
            low = tok.lower()
            counts[low] = counts.get(low, 0) + 1
    return counts


def _assert_no_disallowed_doc_ids(
    rows: List[Dict[str, object]],
    disallowed_splits: Tuple[str, ...],
    context: str,
) -> None:
    forbidden = set()
    for split in disallowed_splits:
        forbidden.update(SPLITS[split])
    leaked = sorted({str(r["doc_id"]) for r in rows if r["doc_id"] in forbidden})
    if leaked:
        raise AssertionError(
            f"Leak barrier violated in {context}: rows from disallowed splits "
            f"{disallowed_splits} present (docs={leaked})."
        )


def _assert_artifacts_saved_before_test(
    artifact_paths: List[Path],
    context: str,
) -> None:
    missing = [str(p) for p in artifact_paths if not p.exists()]
    if missing:
        raise AssertionError(
            f"Refusing to predict on test split in {context}: required model "
            f"artifacts not yet saved: {missing}."
        )


def _jsonl_write(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _predict_records(
    model_name: str,
    correct_fn,
    rows: List[Dict[str, object]],
    train_word_counts: Dict[str, int],
    blind_test: bool,
    progress_every: int = 200,
) -> Tuple[List[Dict[str, object]], List[float]]:
    outputs: List[Dict[str, object]] = []
    latencies_ms: List[float] = []
    label = "test_blind" if blind_test else "val"
    print(f"[predict] {model_name} {label}: {len(rows)} rows", flush=True)
    t0 = time.perf_counter()
    for idx, row in enumerate(rows, start=1):
        noisy = str(row["noisy"])
        clean = str(row["clean"])
        ts = time.perf_counter()
        prediction, logs = correct_fn(noisy)
        latencies_ms.append((time.perf_counter() - ts) * 1000.0)
        changed = prediction != noisy
        token_was_correct_before = noisy == clean
        ref_value = None if blind_test else clean
        sample_tokens = clean.split()
        rare_flag = (
            any(is_rare_word(tok, train_word_counts) for tok in sample_tokens)
            if sample_tokens
            else False
        )
        proper_flag = (
            any(is_proper_name(tok) for tok in sample_tokens)
            if sample_tokens
            else False
        )
        confidence = max((float(d.get("confidence", 0.0)) for d in logs), default=0.0)
        gate_decision: Optional[str] = None
        if model_name == "hybrid" and logs:
            gate_decision = str(logs[0].get("gate_decision", ""))
        output: Dict[str, object] = {
            "doc_id": row["doc_id"],
            "split": row["split"],
            "subset_domain": row["subset_domain"],
            "sample_id": row["sample_id"],
            "source_type": row.get("source_type"),
            "input_noisy": noisy,
            "prediction": prediction,
            "reference": ref_value,
            "changed_flag": changed,
            "token_was_correct_before": token_was_correct_before if not blind_test else None,
            "confidence": confidence,
            "gate_decision": gate_decision,
            "is_rare_word": rare_flag,
            "is_proper_name": proper_flag,
        }
        if not blind_test:
            ref_tokens = clean.split()
            pred_tokens = prediction.split()
            token_pairs = list(zip(ref_tokens, pred_tokens))
            rare_total = rare_bad = name_total = name_bad = 0
            for ref_tok, pred_tok in token_pairs:
                tok_is_rare = is_rare_word(ref_tok, train_word_counts)
                tok_is_name = is_proper_name(ref_tok)
                changed_tok = ref_tok != pred_tok
                if tok_is_rare:
                    rare_total += 1
                    if changed_tok:
                        rare_bad += 1
                if tok_is_name:
                    name_total += 1
                    if changed_tok:
                        name_bad += 1
            input_cer_val = cer(clean, noisy)
            output_cer_val = cer(clean, prediction)
            output["input_cer"] = input_cer_val
            output["cer"] = output_cer_val
            output["error_reduction"] = cer_reduction_rate(input_cer_val, output_cer_val)
            output["wer"] = wer(clean, prediction)
            output["chrf"] = chrf_score(clean, prediction)
            output["overcorrected"] = bool(token_was_correct_before and changed)
            output["corrupted_rare_word"] = rare_bad > 0
            output["corrupted_proper_name"] = name_bad > 0
            output["rare_token_total"] = rare_total
            output["rare_token_corrupted"] = rare_bad
            output["proper_token_total"] = name_total
            output["proper_token_corrupted"] = name_bad
        outputs.append(output)
        if idx == 1 or idx % progress_every == 0:
            dt = time.perf_counter() - t0
            print(
                f"[predict] {model_name} {label}: {idx}/{len(rows)} "
                f"elapsed={dt:.1f}s",
                flush=True,
            )
    validate_prediction_records(outputs, blind_test=blind_test)
    dt = time.perf_counter() - t0
    print(f"[predict] {model_name} {label}: done in {dt:.1f}s", flush=True)
    return outputs, latencies_ms


def _latency_summary(latencies_ms: List[float], n_params: Optional[int] = None) -> Dict[str, float]:
    if not latencies_ms:
        return {
            "median_ms_per_sentence": None,
            "p95_ms_per_sentence": None,
            "n_params": n_params,
        }
    s = sorted(latencies_ms)
    n = len(s)
    return {
        "median_ms_per_sentence": s[n // 2],
        "p95_ms_per_sentence": s[min(n - 1, int(0.95 * n))],
        "n_params": n_params,
    }


def _tune_classical_on_real_val(
    real_rows: List[Dict[str, object]],
    out_dir: Path,
    phase2_train_only_dir: Path,
) -> ClassicalConfig:
    """Run the classical-threshold grid search once on real_only val, and
    return the tuned ``ClassicalConfig`` to apply across all 3 regimes.
    """
    train_rows = [r for r in real_rows if r["split"] == "train"]
    val_rows = [r for r in real_rows if r["split"] == "val"]
    val_pairs = _to_sentence_pairs(val_rows)
    print(
        f"[classical-tune] fitting baseline classical on {len(train_rows)} "
        f"train rows; tuning on {len(val_pairs)} real val pairs...",
        flush=True,
    )
    base = ClassicalCorrector(
        cfg=ClassicalConfig(), phase2_train_only_dir=phase2_train_only_dir
    )
    base.fit([str(r["clean"]) for r in train_rows])
    tuning = base.tune_thresholds(val_pairs)
    tuned_cfg = base.cfg
    print(
        f"[classical-tune] tuned cfg: correction_margin={tuned_cfg.correction_margin} "
        f"lambda_lm={tuned_cfg.lambda_lm} lambda_channel={tuned_cfg.lambda_channel}",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classical_tuning.json").write_text(
        json.dumps(tuning, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return tuned_cfg


def _run_one(
    model_name: str,
    regime: str,
    rows: List[Dict[str, object]],
    out_dir: Path,
    seed: int,
    phase2_train_only_dir: Path,
    primary_seed: int,
    classical_cfg: Optional[ClassicalConfig] = None,
    predict_sample_limit: Optional[int] = None,
) -> Dict[str, object]:
    print(f"[RUN] model={model_name} regime={regime} seed={seed}", flush=True)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(
        f"[RUN] counts train={len(train_rows)} val={len(val_rows)} "
        f"test={len(test_rows)} total={len(rows)}",
        flush=True,
    )
    _assert_no_disallowed_doc_ids(train_rows, ("val", "test"), context="train_rows")
    _assert_no_disallowed_doc_ids(val_rows, ("train", "test"), context="val_rows")
    _assert_no_disallowed_doc_ids(test_rows, ("train", "val"), context="test_rows")

    train_word_counts = _train_word_counts(train_rows)

    is_primary = seed == primary_seed
    seed_suffix = "" if is_primary else f"__seed{seed}"
    model_dir = out_dir / "models" / model_name / regime
    if not is_primary:
        model_dir = model_dir / f"seed{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "regime": regime,
                "seed": seed,
                "primary_seed": primary_seed,
                "split_manifest_hash": split_manifest_hash(),
                "hparams": frozen_hparams_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    n_params: Optional[int] = None
    eff_classical_cfg = classical_cfg or ClassicalConfig()
    if model_name == "classical":
        model = ClassicalCorrector(
            cfg=eff_classical_cfg, phase2_train_only_dir=phase2_train_only_dir
        )
        print("[train] classical: fitting lexicon + KN word LM on TRAIN...", flush=True)
        t_fit = time.perf_counter()
        model.fit([str(r["clean"]) for r in train_rows])
        print(f"[train] classical: fit done in {time.perf_counter() - t_fit:.1f}s", flush=True)
        model.save(model_dir)
        correct_fn = model.correct_sentence
    elif model_name == "neural":
        cfg = TransformerConfig()
        model = ByteTransformerCorrector(cfg=cfg, seed=seed)
        val_pairs = _to_sentence_pairs(val_rows)
        print(
            f"[train] neural: regime={regime} val_pairs={len(val_pairs)}",
            flush=True,
        )
        t_fit = time.perf_counter()
        training_metrics = _train_neural(model, rows, regime, val_pairs)
        print(f"[train] neural: training done in {time.perf_counter() - t_fit:.1f}s", flush=True)
        model.save(model_dir)
        n_params = int(training_metrics.get("n_params") or 0) or None
        (model_dir / "training_metrics.json").write_text(
            json.dumps(training_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        correct_fn = model.correct_sentence
    elif model_name == "hybrid":
        classical = ClassicalCorrector(
            cfg=eff_classical_cfg, phase2_train_only_dir=phase2_train_only_dir
        )
        print("[train] hybrid: fitting classical...", flush=True)
        t0 = time.perf_counter()
        classical.fit([str(r["clean"]) for r in train_rows])
        print(f"[train] hybrid: classical fit done in {time.perf_counter() - t0:.1f}s", flush=True)
        neural = ByteTransformerCorrector(cfg=TransformerConfig(), seed=seed)
        val_pairs = _to_sentence_pairs(val_rows)
        print(
            f"[train] hybrid: regime={regime} val_pairs={len(val_pairs)}",
            flush=True,
        )
        t1 = time.perf_counter()
        training_metrics = _train_neural(neural, rows, regime, val_pairs)
        print(f"[train] hybrid: neural fit done in {time.perf_counter() - t1:.1f}s", flush=True)
        classical.save(model_dir / "classical_assets")
        neural.save(model_dir / "neural_assets")
        n_params = int(training_metrics.get("n_params") or 0) or None
        (model_dir / "neural_assets" / "training_metrics.json").write_text(
            json.dumps(training_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        hybrid_cfg = HybridConfig()
        model = HybridCorrector(
            classical=classical,
            neural=neural,
            cfg=hybrid_cfg,
            train_word_counts=train_word_counts,
        )
        print("[train] hybrid: calibrating fusion weights on VAL...", flush=True)
        t2 = time.perf_counter()
        cal = model.calibrate_on_val(val_pairs, max_pairs=400)
        print(f"[train] hybrid: calibration done in {time.perf_counter() - t2:.1f}s", flush=True)
        model.save_calibration(model_dir / "hybrid_calibration.json")
        (model_dir / "hybrid_calibration_summary.json").write_text(
            json.dumps(
                {"selected": cal["selected"], "grid_size": cal["grid_size"], "tested": cal["tested"]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        correct_fn = model.correct_sentence
    else:
        raise ValueError(f"Unknown model: {model_name}")

    val_predict_rows = (
        val_rows[:predict_sample_limit]
        if predict_sample_limit is not None
        else val_rows
    )
    test_predict_rows = (
        test_rows[:predict_sample_limit]
        if predict_sample_limit is not None
        else test_rows
    )
    if predict_sample_limit is not None:
        print(
            f"[predict] limiting val/test rows to {predict_sample_limit} each "
            f"(full val={len(val_rows)} test={len(test_rows)})",
            flush=True,
        )

    val_records, val_latencies = _predict_records(
        model_name, correct_fn, val_predict_rows, train_word_counts, blind_test=False
    )
    val_path = out_dir / "predictions" / "val" / model_name / f"{regime}{seed_suffix}.jsonl"
    print(f"[write] val predictions -> {val_path}", flush=True)
    _jsonl_write(val_path, val_records)

    artifact_check_paths: List[Path] = [val_path]
    if model_name == "classical":
        artifact_check_paths.append(model_dir / "classical_model.json")
    elif model_name == "neural":
        artifact_check_paths.append(model_dir / "transformer.pt")
    elif model_name == "hybrid":
        artifact_check_paths.append(model_dir / "neural_assets" / "transformer.pt")
        artifact_check_paths.append(model_dir / "hybrid_calibration.json")
    _assert_artifacts_saved_before_test(artifact_check_paths, context=f"{model_name}/{regime}")

    test_records, _test_latencies = _predict_records(
        model_name, correct_fn, test_predict_rows, train_word_counts, blind_test=True
    )
    test_path = out_dir / "predictions" / "test_blind" / model_name / f"{regime}{seed_suffix}.jsonl"
    print(f"[write] test_blind predictions -> {test_path}", flush=True)
    _jsonl_write(test_path, test_records)

    a_to_b_rows = [r for r in test_predict_rows if r["subset_domain"] == "B"]
    a_to_b_records, _ = _predict_records(
        model_name, correct_fn, a_to_b_rows, train_word_counts, blind_test=True
    )
    cd_path = out_dir / "predictions" / "cross_domain" / model_name / f"{regime}{seed_suffix}_A_to_B.jsonl"
    print(f"[write] cross_domain predictions -> {cd_path}", flush=True)
    _jsonl_write(cd_path, a_to_b_records)

    metric_payload = aggregate_metrics(val_records)
    metric_payload.update(rare_and_name_corruption(val_records))
    metric_payload["per_domain"] = per_domain_breakdown(val_records)
    bins = calibration_bins(val_records, n_bins=10)
    metric_payload["calibration"] = {
        "bins": bins,
        "ece": expected_calibration_error(bins),
    }
    metric_payload["runs_on"] = {"model": model_name, "regime": regime, "seed": seed}
    metrics_path = out_dir / "val_metrics" / model_name / f"{regime}{seed_suffix}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metric_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[write] val metrics -> {metrics_path}", flush=True)

    eff_payload = _latency_summary(val_latencies, n_params=n_params)
    eff_path = out_dir / "efficiency" / model_name / f"{regime}{seed_suffix}.json"
    eff_path.parent.mkdir(parents=True, exist_ok=True)
    eff_path.write_text(json.dumps(eff_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] efficiency -> {eff_path}", flush=True)

    return {
        "model": model_name,
        "regime": regime,
        "status": "success",
        "seed": seed,
        "primary_seed": primary_seed,
        "val_predictions": str(val_path),
        "test_predictions": str(test_path),
        "val_metrics": str(metrics_path),
        "efficiency": str(eff_path),
        "checkpoint_dir": str(model_dir),
    }


def _emit_identity_baseline(
    rows: List[Dict[str, object]],
    out_dir: Path,
    regime: str,
    primary_seed: int,
    train_word_counts: Dict[str, int],
    predict_sample_limit: Optional[int] = None,
) -> Dict[str, object]:
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    if predict_sample_limit is not None:
        val_rows = val_rows[:predict_sample_limit]
        test_rows = test_rows[:predict_sample_limit]
    val_records = identity_baseline_records(val_rows, train_word_counts)
    test_records = []
    for r in test_rows:
        noisy = str(r["noisy"])
        test_records.append(
            {
                "doc_id": r["doc_id"],
                "split": r["split"],
                "subset_domain": r["subset_domain"],
                "sample_id": r["sample_id"],
                "source_type": r.get("source_type"),
                "input_noisy": noisy,
                "prediction": noisy,
                "reference": None,
                "changed_flag": False,
                "token_was_correct_before": None,
                "confidence": 1.0,
                "gate_decision": None,
                "is_rare_word": False,
                "is_proper_name": False,
            }
        )
    val_path = out_dir / "predictions" / "val" / "identity" / f"{regime}__seed{primary_seed}.jsonl"
    test_path = out_dir / "predictions" / "test_blind" / "identity" / f"{regime}__seed{primary_seed}.jsonl"
    _jsonl_write(val_path, val_records)
    _jsonl_write(test_path, test_records)
    metrics = aggregate_metrics(val_records)
    metrics.update(rare_and_name_corruption(val_records))
    metrics["per_domain"] = per_domain_breakdown(val_records)
    metrics["runs_on"] = {"model": "identity", "regime": regime, "seed": primary_seed}
    metrics_path = out_dir / "val_metrics" / "identity" / f"{regime}__seed{primary_seed}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "model": "identity",
        "regime": regime,
        "status": "success",
        "seed": primary_seed,
        "primary_seed": primary_seed,
        "val_predictions": str(val_path),
        "test_predictions": str(test_path),
        "val_metrics": str(metrics_path),
    }



def _run_phase1(repo_root: Path) -> None:
    """Re-run phase 1 alignment to refresh ``matched_pairs.json`` for all docs.

    Phase 1's ``__main__`` writes its outputs relative to the current working
    directory, so we ``chdir`` into ``phase1/`` for the duration of the call.
    """
    import importlib
    import runpy
    cwd = os.getcwd()
    try:
        os.chdir(str(repo_root / "phase1"))
        runpy.run_path(str(repo_root / "phase1" / "phase1_alignment.py"), run_name="__main__")
    finally:
        os.chdir(cwd)
    # Drop any cached imports so subsequent regenerations see the fresh data.
    for modname in list(sys.modules):
        if modname.startswith("phase1") or modname.startswith("phase2") or modname.startswith("phase3"):
            try:
                importlib.reload(sys.modules[modname])
            except Exception:
                pass


def _print_run_header(
    repo_root: Path,
    cfg,
    seeds: List[int],
    models_to_run: List[str],
    regimes_to_run: List[str],
) -> None:
    """Dump the active config + neural device summary at run start."""
    import torch

    if torch.cuda.is_available():
        device_label = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_label = "mps"
    else:
        device_label = "cpu"
    override = os.environ.get("PHASE4_NEURAL_DEVICE", "auto")
    print("=" * 78, flush=True)
    print(f"[phase4] repo_root            = {repo_root}", flush=True)
    print(f"[phase4] output_dir           = {cfg.output_dir}", flush=True)
    print(f"[phase4] seeds                = {seeds} (primary={seeds[0]})", flush=True)
    print(f"[phase4] models_to_run        = {models_to_run}", flush=True)
    print(f"[phase4] regimes_to_run       = {regimes_to_run}", flush=True)
    print(f"[phase4] neural device probe  = {device_label} (override={override})", flush=True)
    nn_cfg = TransformerConfig()
    cls_cfg = ClassicalConfig()
    print(
        f"[phase4] neural model         = {nn_cfg.pretrained_model} "
        f"lr={nn_cfg.learning_rate} warmup_ratio={nn_cfg.warmup_ratio} "
        f"max_input_bytes={nn_cfg.max_input_bytes} effective_batch="
        f"{nn_cfg.batch_size * max(1, nn_cfg.gradient_accumulation_steps)} "
        f"max_epochs={nn_cfg.max_epochs} pretrain_epochs={nn_cfg.pretrain_epochs} "
        f"finetune_epochs={nn_cfg.finetune_epochs} "
        f"identity_pair_ratio={nn_cfg.identity_pair_ratio}",
        flush=True,
    )
    print(
        f"[phase4] classical defaults   = correction_margin={cls_cfg.correction_margin} "
        f"lambda_lm={cls_cfg.lambda_lm} lambda_channel={cls_cfg.lambda_channel} "
        f"max_edit_distance={cls_cfg.max_edit_distance}",
        flush=True,
    )
    print("=" * 78, flush=True)


def run_phase4(
    repo_root: Path,
    seeds: Optional[List[int]] = None,
    skip_classical: bool = False,
    skip_neural: bool = False,
    skip_hybrid: bool = False,
    skip_phase1: bool = False,
    skip_phase2_stats: bool = False,
    skip_phase3_noise: bool = False,
    skip_manifests: bool = False,
    regimes: Optional[List[str]] = None,
    predict_sample_limit: Optional[int] = None,
) -> None:
    cfg = default_run_config(repo_root)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    assert_disjoint_splits()

    seeds = seeds or list(SEEDS)
    primary_seed = seeds[0]

    models_to_run: List[str] = []
    if not skip_classical:
        models_to_run.append("classical")
    if not skip_neural:
        models_to_run.append("neural")
    if not skip_hybrid:
        models_to_run.append("hybrid")
    if not models_to_run:
        raise ValueError(
            "At least one model must run; all of --skip-classical, --skip-neural, "
            "--skip-hybrid are set."
        )

    all_regimes = ("real_only", "synthetic_only", "synthetic_plus_real")
    regimes_to_run: List[str] = list(regimes) if regimes else list(all_regimes)
    bad = [r for r in regimes_to_run if r not in all_regimes]
    if bad:
        raise ValueError(f"Unknown regime(s): {bad}. Allowed: {all_regimes}")

    _print_run_header(repo_root, cfg, seeds, models_to_run, regimes_to_run)
    if predict_sample_limit is not None:
        print(
            f"[phase4] predict_sample_limit={predict_sample_limit} "
            f"(caps val/test prediction JSONL rows per model; training unchanged)",
            flush=True,
        )

    if not skip_phase1:
        print("[phase4] running phase 1 alignment to regenerate matched_pairs.json", flush=True)
        _run_phase1(repo_root)
    else:
        print("[phase4] (skip-phase1) reusing existing phase1_output/", flush=True)

    if skip_phase2_stats:
        if not _phase2_train_only(cfg.phase2_train_only_dir):
            raise FileNotFoundError(
                f"--skip-phase2-stats requested but train-only Phase 2 artifacts "
                f"are missing in {cfg.phase2_train_only_dir}."
            )
        print(
            f"[phase4] (skip-phase2-stats) reusing {cfg.phase2_train_only_dir}",
            flush=True,
        )
    if skip_phase3_noise:
        if not _phase3_train_only(cfg.phase3_train_only_dir):
            raise FileNotFoundError(
                f"--skip-phase3-noise requested but train-only Phase 3 artifacts "
                f"are missing in {cfg.phase3_train_only_dir}."
            )
        print(
            f"[phase4] (skip-phase3-noise) reusing {cfg.phase3_train_only_dir}",
            flush=True,
        )
    if not (skip_phase2_stats and skip_phase3_noise):
        _ensure_train_only_stats(cfg)

    manifests_dir = repo_root / "phase4" / "data" / "manifests"
    if skip_manifests:
        print("[phase4] (skip-manifests) reusing existing manifests/*.jsonl", flush=True)
        manifest_paths = {
            r: manifests_dir / f"{r}.jsonl" for r in all_regimes
        }
        missing = [str(p) for p in manifest_paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"--skip-manifests requested but the following manifests are missing: {missing}"
            )
    else:
        print("[phase4] building manifests from train-only synthetic + real pairs ...", flush=True)
        manifest_paths = build_phase4_manifests(
            phase1_output_dir=cfg.phase1_output_dir,
            phase3_output_dir=cfg.phase3_train_only_dir,
            manifests_dir=manifests_dir,
        )
        print(f"[phase4] manifests: {manifest_paths}", flush=True)

    runs: List[Dict[str, object]] = []
    selected_checkpoints: Dict[str, object] = {}
    failures: List[Dict[str, object]] = []
    print(f"[phase4] models_to_run={models_to_run}", flush=True)
    print(f"[phase4] regimes_to_run={regimes_to_run}", flush=True)


    real_rows_for_tuning = load_jsonl(manifest_paths["real_only"])
    classical_cfg = _tune_classical_on_real_val(
        real_rows=real_rows_for_tuning,
        out_dir=cfg.output_dir,
        phase2_train_only_dir=cfg.phase2_train_only_dir,
    )

    for regime in regimes_to_run:
        rows = load_jsonl(manifest_paths[regime])
        train_word_counts = _train_word_counts([r for r in rows if r["split"] == "train"])
        try:
            id_run = _emit_identity_baseline(
                rows=rows,
                out_dir=cfg.output_dir,
                regime=regime,
                primary_seed=primary_seed,
                train_word_counts=train_word_counts,
                predict_sample_limit=predict_sample_limit,
            )
            runs.append(id_run)
            print(f"[phase4] identity baseline emitted for {regime}", flush=True)
        except Exception as exc:  
            fail = {
                "model": "identity",
                "regime": regime,
                "status": "failed",
                "error": str(exc),
            }
            runs.append(fail)
            failures.append(fail)
            print(f"[phase4] FAILED identity {regime}: {exc}", flush=True)
            if cfg.fail_fast:
                raise
        for model_name in models_to_run:
            for seed in seeds:
                try:
                    result = _run_one(
                        model_name=model_name,
                        regime=regime,
                        rows=rows,
                        out_dir=cfg.output_dir,
                        seed=seed,
                        phase2_train_only_dir=cfg.phase2_train_only_dir,
                        primary_seed=primary_seed,
                        classical_cfg=classical_cfg,
                        predict_sample_limit=predict_sample_limit,
                    )
                    runs.append(result)
                    if seed == primary_seed:
                        selected_checkpoints[f"{model_name}:{regime}"] = {
                            "checkpoint_dir": result["checkpoint_dir"],
                            "selection_rationale": "primary seed; hyperparameters frozen across seeds and regimes",
                        }
                    print(
                        f"[phase4] completed {model_name} {regime} seed={seed}",
                        flush=True,
                    )
                except Exception as exc:
                    fail = {
                        "model": model_name,
                        "regime": regime,
                        "status": "failed",
                        "seed": seed,
                        "error": str(exc),
                    }
                    runs.append(fail)
                    failures.append(fail)
                    print(
                        f"[phase4] FAILED {model_name} {regime} seed={seed}: {exc}",
                        flush=True,
                    )
                    if cfg.fail_fast:
                        raise

    print("[phase4] building paper tables...", flush=True)
    table_paths = build_all_tables(
        out_dir=cfg.output_dir, primary_seed=primary_seed, all_seeds=list(seeds)
    )
    for name, path in table_paths.items():
        print(f"[phase4] paper_table[{name}] -> {path}", flush=True)

    manifest = {
        "split_manifest_hash": split_manifest_hash(),
        "hyperparameter_hash": hashlib.sha256(
            json.dumps(frozen_hparams_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "seeds": seeds,
        "primary_seed": primary_seed,
        "selected_checkpoints": selected_checkpoints,
        "failures": failures,
        "paper_tables": {k: str(v) for k, v in table_paths.items()},
        "leak_fix": {
            "phase2_train_only_dir": str(cfg.phase2_train_only_dir),
            "phase3_train_only_dir": str(cfg.phase3_train_only_dir),
        },
        "models_run": models_to_run,
    }
    metadata_dir = cfg.output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[phase4] wrote {metadata_dir / 'run_manifest.json'}", flush=True)

    table_path = metadata_dir / "run_table.csv"
    fieldnames = [
        "model",
        "regime",
        "status",
        "seed",
        "primary_seed",
        "val_predictions",
        "test_predictions",
        "val_metrics",
        "efficiency",
        "checkpoint_dir",
        "error",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in runs:
            writer.writerow({k: row.get(k) for k in fieldnames})
    print(f"[phase4] wrote {table_path}", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds (overrides SEEDS).",
    )
    parser.add_argument(
        "--skip-classical",
        action="store_true",
        help="Skip the standalone classical model (neural and hybrid are unchanged; "
        "hybrid still fits its own classical head internally).",
    )
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Reuse existing phase1_output/ instead of re-running phase 1 alignment. "
        "Use this on subsequent phase 4 runs once phase 1 is known to be correct.",
    )
    parser.add_argument(
        "--skip-phase2-stats",
        action="store_true",
        help="Reuse existing phase2/phase2_output_train_only/ stats instead of "
        "rebuilding them.",
    )
    parser.add_argument(
        "--skip-phase3-noise",
        action="store_true",
        help="Reuse existing phase3/phase3_output_train_only/ synthetic noise "
        "instead of regenerating it.",
    )
    parser.add_argument(
        "--skip-manifests",
        action="store_true",
        help="Reuse existing phase4/data/manifests/*.jsonl instead of rebuilding.",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default=None,
        help="Comma-separated subset of regimes to run "
        "(real_only,synthetic_only,synthetic_plus_real). Default: all three.",
    )
    parser.add_argument(
        "--neural-device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device for the byte Transformer. 'auto' picks cuda > mps > cpu. "
        "Use 'cpu' as an escape hatch when MPS hits a Metal bug for your "
        "PyTorch / macOS combination (slower but always works).",
    )
    parser.add_argument(
        "--predict-sample",
        type=int,
        default=None,
        metavar="N",
        help="For smoke/dev runs: after training, only write the first N val and "
        "first N test rows to prediction JSONL (and cross-domain A->B from the "
        "capped test set). Training and validation during fit are unchanged.",
    )
    args = parser.parse_args()
    if args.neural_device != "auto":
        os.environ["PHASE4_NEURAL_DEVICE"] = args.neural_device

    seeds: Optional[List[int]] = None
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    regimes: Optional[List[str]] = None
    if args.regimes:
        regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    repo_root = Path(__file__).resolve().parents[1]
    run_phase4(
        repo_root,
        seeds=seeds,
        skip_classical=args.skip_classical,
        skip_neural=args.skip_neural,
        skip_hybrid=args.skip_hybrid,
        skip_phase1=args.skip_phase1,
        skip_phase2_stats=args.skip_phase2_stats,
        skip_phase3_noise=args.skip_phase3_noise,
        skip_manifests=args.skip_manifests,
        regimes=regimes,
        predict_sample_limit=args.predict_sample,
    )
    print("Phase 4 completed.", flush=True)


if __name__ == "__main__":
    main()
