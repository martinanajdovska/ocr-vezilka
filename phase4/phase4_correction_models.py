"""Phase 4 runner.

1. (Re)build train-only Phase 2 statistics + Phase 3 synthetic noise so the
   synthetic regimes never see val/test confusion distributions.
2. (Re)build the Phase 4 manifests with banded two-level alignment.
3. For each (model, regime, seed) triple, train the model under the regime,
   predict on val (with metrics) and test_blind (without metrics), persist
   models / predictions / metrics / efficiency.
4. Generate identity baseline for each regime.
5. Aggregate paper-table CSVs and a top-level run manifest.

All training-time decisions read only train rows. Hybrid calibration reads
only val rows. Test rows are predicted, but never inspected for parameter or
threshold selection (asserted at runtime).

Neural and hybrid ByT5 training writes resumable checkpoints under
``models/<neural|hybrid>/<regime>/_neural_train_checkpoint/`` (epoch files
``byt5_resume_<stage>.pt``, plus ``pretrain_hf/`` after a completed pretrain
for ``synthetic_plus_real``, and ``awaiting_finetune`` between stages).
Disable with ``--no-neural-resume`` or env ``PHASE4_NEURAL_RESUME=0``.

Smoke / GPU check (from repo root, after Phase 1--3 train-only artifacts exist):
``python -u phase4/phase4_correction_models.py --skip-phase1 --skip-phase2-stats
--skip-phase3-noise --skip-classical --skip-hybrid --regimes real_only
--seeds 42 --neural-device cuda --predict-sample 200``. ``200`` caps ByT5
train/val pairs for fitting (1 epoch per stage) and caps val/test prediction
JSONL rows. Optionally grep logs for ``train_loss=nan`` or
``Non-finite training loss``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
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


def _predict_sample_progress(limit: Optional[int], msg: str) -> None:
    """Extra checkpoints when ``--predict-sample`` is set (subprocess log stays alive)."""
    if limit is None:
        return
    wall = time.strftime("%H:%M:%S")
    print(f"[phase4] (predict-sample N={limit}) {wall}  {msg}", flush=True)


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


def _maybe_cap_pairs(
    pairs: List[Tuple[str, str]],
    cap: Optional[int],
    label: str,
) -> List[Tuple[str, str]]:
    """Return ``pairs[:cap]`` when ``cap`` is set and smaller than ``len(pairs)``."""
    if cap is None or len(pairs) <= cap:
        return pairs
    print(
        f"[neural] predict-sample: using first {cap} of {len(pairs)} {label} pairs",
        flush=True,
    )
    return pairs[:cap]


def _neural_resume_effective(cli_neural_resume: bool) -> bool:
    """Honor ``PHASE4_NEURAL_RESUME`` when set; otherwise use the CLI / runner flag."""
    raw = os.environ.get("PHASE4_NEURAL_RESUME", "").strip()
    if not raw:
        return cli_neural_resume
    return raw.lower() not in ("0", "false", "no", "off")


def _neural_train_checkpoint_dir(model_dir: Path) -> Path:
    return model_dir / "_neural_train_checkpoint"


def _train_neural(
    neural,
    rows: List[Dict[str, object]],
    regime: str,
    val_pairs: List[Tuple[str, str]],
    train_pair_cap: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    neural_resume: bool = True,
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

    ``train_pair_cap`` (same ``N`` as ``--predict-sample`` when that flag is set):
    use only the first ``N`` synthetic / real / plain train pairs per stage;
    ``val_pairs`` must already be capped by the caller when ``--predict-sample``
    is in use.

    ``checkpoint_dir`` (typically ``.../models/<neural|hybrid>/<regime>/_neural_train_checkpoint``)
    stores epoch checkpoints and, for two-stage regimes, ``pretrain_hf`` so a
    Colab/Drive reconnect can resume pretrain, finetune, or single-stage training.
    """
    if train_pair_cap is not None:
        print(
            f"[neural] (predict-sample) starting ByT5 fit  regime={regime}  "
            f"val_pairs={len(val_pairs)} (short run: 1 epoch per stage)",
            flush=True,
        )
    ckpt = checkpoint_dir
    use_resume = neural_resume

    def _fit(
        pairs: List[Tuple[str, str]],
        st: str,
        resume_from: Optional[Path] = None,
    ) -> Dict[str, object]:
        return neural.fit(
            pairs,
            val_pairs,
            stage=st,
            resume_from=resume_from,
            checkpoint_dir=ckpt,
            resume=use_resume,
        )

    if regime == "synthetic_plus_real":
        synth_pairs = _split_pairs_by_source(rows, split="train", source_type="synthetic")
        real_pairs = _split_pairs_by_source(rows, split="train", source_type="real")
        synth_pairs = _maybe_cap_pairs(synth_pairs, train_pair_cap, "synthetic train")
        real_pairs = _maybe_cap_pairs(real_pairs, train_pair_cap, "real train")
        if not synth_pairs:
            print(
                "[neural] regime=synthetic_plus_real has no synthetic train rows; "
                "skipping pretrain stage and falling back to single-stage real fit.",
                flush=True,
            )
            return _fit(real_pairs, "train")
        if not real_pairs:
            print(
                "[neural] regime=synthetic_plus_real has no real train rows; "
                "running synthetic stage only.",
                flush=True,
            )
            return _fit(synth_pairs, "train")

        pretrain_hf = ckpt / "pretrain_hf" if ckpt is not None else None
        fin_resume = ckpt / "byt5_resume_finetune.pt" if ckpt is not None else None
        pretrain_resume = ckpt / "byt5_resume_pretrain.pt" if ckpt is not None else None
        pretrain_metrics_path = ckpt / "pretrain_training_metrics.json" if ckpt is not None else None
        awaiting = ckpt / "awaiting_finetune" if ckpt is not None else None

        if use_resume and ckpt is not None and fin_resume is not None and fin_resume.exists():
            pre_metrics: Dict[str, object] = {}
            if pretrain_metrics_path is not None and pretrain_metrics_path.exists():
                pre_metrics = json.loads(pretrain_metrics_path.read_text(encoding="utf-8"))
            print(
                "[neural] regime=synthetic_plus_real -> resuming finetune only "
                f"(checkpoint {fin_resume.name})",
                flush=True,
            )
            finetune_metrics = _fit(real_pairs, "finetune")
            if awaiting is not None and awaiting.exists():
                awaiting.unlink()
            return {
                "stage": "pretrain_then_finetune",
                "pretrain": pre_metrics,
                "finetune": finetune_metrics,
                "n_params": finetune_metrics.get("n_params"),
                "device": finetune_metrics.get("device"),
                "best_val_loss": finetune_metrics.get("best_val_loss"),
                "total_train_seconds": float(finetune_metrics.get("total_train_seconds", 0.0)),
            }

        skipped_pretrain = (
            use_resume
            and ckpt is not None
            and pretrain_hf is not None
            and pretrain_hf.exists()
            and (pretrain_resume is None or not pretrain_resume.exists())
            and awaiting is not None
            and awaiting.exists()
        )
        if not skipped_pretrain:
            print(
                f"[neural] regime=synthetic_plus_real -> stage 1 (pretrain) "
                f"on {len(synth_pairs)} synthetic pairs",
                flush=True,
            )
            pretrain_metrics = _fit(synth_pairs, "pretrain")
            if awaiting is not None:
                awaiting.write_text("1", encoding="utf-8")
            if pretrain_metrics_path is not None:
                pretrain_metrics_path.write_text(
                    json.dumps(pretrain_metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        else:
            print(
                "[neural] regime=synthetic_plus_real -> skipping completed pretrain; "
                "loading pretrain snapshot for finetune",
                flush=True,
            )
            pretrain_metrics = {}
            if pretrain_metrics_path is not None and pretrain_metrics_path.exists():
                pretrain_metrics = json.loads(pretrain_metrics_path.read_text(encoding="utf-8"))

        finetune_resume_from = pretrain_hf if skipped_pretrain else None
        print(
            f"[neural] regime=synthetic_plus_real -> stage 2 (finetune) "
            f"on {len(real_pairs)} real pairs",
            flush=True,
        )
        finetune_metrics = _fit(real_pairs, "finetune", resume_from=finetune_resume_from)
        if awaiting is not None and awaiting.exists():
            awaiting.unlink()
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
    train_pairs = _maybe_cap_pairs(train_pairs, train_pair_cap, "train")
    return _fit(train_pairs, "train")


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


def _val_metric_payload(
    val_records: List[Dict[str, object]],
    model_name: str,
    regime: str,
    seed: int,
) -> Dict[str, object]:
    metric_payload: Dict[str, object] = aggregate_metrics(val_records)
    metric_payload.update(rare_and_name_corruption(val_records))
    metric_payload["per_domain"] = per_domain_breakdown(val_records)
    bins = calibration_bins(val_records, n_bins=10)
    metric_payload["calibration"] = {
        "bins": bins,
        "ece": expected_calibration_error(bins),
    }
    metric_payload["runs_on"] = {"model": model_name, "regime": regime, "seed": seed}
    return metric_payload


def _val_prediction_candidates(
    out_dir: Path,
    model_name: str,
    regime: str,
    seed: int,
    primary_seed: int,
) -> List[Path]:
    val_dir = out_dir / "predictions" / "val" / model_name
    unsuffixed = val_dir / f"{regime}.jsonl"
    suffixed = val_dir / f"{regime}__seed{seed}.jsonl"
    if seed == primary_seed and model_name != "identity":
        return [unsuffixed, suffixed]
    return [suffixed, unsuffixed]


def rebuild_val_metrics_from_predictions(
    repo_root: Path,
    model_name: str,
    regime: str,
    seed: int = SEEDS[0],
    primary_seed: int = SEEDS[0],
    rebuild_paper_tables: bool = False,
) -> Path:
    """Recreate ``val_metrics`` from an existing val prediction JSONL.

    This is a recovery helper for runs that finished prediction but stopped
    before writing the metadata/metrics tail of Phase 4. It does not load,
    train, or run a model; it only reads the saved validation predictions.
    """
    cfg = default_run_config(repo_root)
    candidates = _val_prediction_candidates(
        cfg.output_dir,
        model_name=model_name,
        regime=regime,
        seed=seed,
        primary_seed=primary_seed,
    )
    val_path = next((p for p in candidates if p.exists()), None)
    if val_path is None:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"No val prediction JSONL found for {model_name}/{regime}/seed={seed}. "
            f"Tried: {tried}"
        )
    val_records = load_jsonl(val_path)
    validate_prediction_records(val_records, blind_test=False)
    metric_payload = _val_metric_payload(val_records, model_name, regime, seed)

    seed_suffix = "" if seed == primary_seed and model_name != "identity" else f"__seed{seed}"
    metrics_path = cfg.output_dir / "val_metrics" / model_name / f"{regime}{seed_suffix}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metric_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[recover] read val predictions -> {val_path}", flush=True)
    print(f"[recover] wrote val metrics -> {metrics_path}", flush=True)

    if rebuild_paper_tables:
        table_paths = build_all_tables(
            out_dir=cfg.output_dir,
            primary_seed=primary_seed,
            all_seeds=[seed],
            phase1_output_dir=cfg.phase1_output_dir,
        )
        for name, path in table_paths.items():
            print(f"[recover] paper_table[{name}] -> {path}", flush=True)
    return metrics_path


def _build_pred_record(
    model_name: str,
    row: Dict[str, object],
    prediction: str,
    logs: List[Dict[str, object]],
    train_word_counts: Dict[str, int],
    blind_test: bool,
) -> Dict[str, object]:
    noisy = str(row["noisy"])
    clean = str(row["clean"])
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
    return output


def _predict_records(
    model_name: str,
    correct_fn,
    rows: List[Dict[str, object]],
    train_word_counts: Dict[str, int],
    blind_test: bool,
    progress_every: int = 200,
    batch_correct_fn=None,
    batch_size: int = 1,
) -> Tuple[List[Dict[str, object]], List[float]]:
    """Run model predictions over ``rows``.

    If ``batch_correct_fn`` is provided and ``batch_size > 1``, sentences are
    batched through it (one ``model.generate`` call per chunk) — typically
    ~5-15x faster than calling ``correct_fn`` per sentence on a CUDA model.
    Per-sentence latency is recorded as the chunk wall-time divided by the
    chunk size, so the latency table still reflects amortized cost.
    """
    outputs: List[Dict[str, object]] = []
    latencies_ms: List[float] = []
    label = "test_blind" if blind_test else "val"
    use_batch = batch_correct_fn is not None and batch_size > 1
    mode = f"batch={batch_size}" if use_batch else "single"
    print(
        f"[predict] {model_name} {label}: {len(rows)} rows ({mode})",
        flush=True,
    )
    t0 = time.perf_counter()

    if use_batch:
        # Length-bucket so each chunk's max_new_tokens is tight: short
        # sentences first, long sentences last.
        byte_lengths = [len(str(r["noisy"]).encode("utf-8")) for r in rows]
        total_bytes = max(1, sum(byte_lengths))
        order = sorted(
            range(len(rows)), key=lambda k: byte_lengths[k]
        )
        per_idx_outputs: Dict[int, Dict[str, object]] = {}
        done = 0
        done_bytes = 0
        for start in range(0, len(order), batch_size):
            idxs = order[start : start + batch_size]
            chunk_rows = [rows[i] for i in idxs]
            noisy_list = [str(r["noisy"]) for r in chunk_rows]
            ts = time.perf_counter()
            results = batch_correct_fn(noisy_list)
            chunk_dt = (time.perf_counter() - ts) * 1000.0
            per_call_ms = chunk_dt / max(1, len(idxs))
            for j, src_idx in enumerate(idxs):
                pred, logs = results[j]
                per_idx_outputs[src_idx] = _build_pred_record(
                    model_name,
                    rows[src_idx],
                    pred,
                    logs,
                    train_word_counts,
                    blind_test,
                )
                latencies_ms.append(per_call_ms)
                done_bytes += byte_lengths[src_idx]
            done += len(idxs)
            dt = time.perf_counter() - t0
            # ETA from byte-progress (work-weighted), with a small floor
            # so the very first chunk does not produce a meaningless number.
            byte_rate = dt / max(1, done_bytes)
            eta = byte_rate * max(0, total_bytes - done_bytes)
            print(
                f"[predict] {model_name} {label}: {done}/{len(rows)} "
                f"elapsed={dt:.1f}s eta={eta:.1f}s "
                f"(batch={len(idxs)} bytes_done={done_bytes}/{total_bytes})",
                flush=True,
            )
        outputs = [per_idx_outputs[i] for i in range(len(rows))]
    else:
        for idx, row in enumerate(rows, start=1):
            noisy = str(row["noisy"])
            ts = time.perf_counter()
            prediction, logs = correct_fn(noisy)
            latencies_ms.append((time.perf_counter() - ts) * 1000.0)
            outputs.append(
                _build_pred_record(
                    model_name, row, prediction, logs, train_word_counts, blind_test
                )
            )
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
    neural_resume: bool = True,
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
    if predict_sample_limit is not None:
        wall = time.strftime("%H:%M:%S")
        n = predict_sample_limit
        print(
            f"[RUN] (predict-sample N={n}) {wall}  ByT5 uses ≤{n} train+val pairs per stage "
            f"and 1 epoch/stage; predictions capped to ≤{n} val / ≤{n} test rows.",
            flush=True,
        )
    _assert_no_disallowed_doc_ids(train_rows, ("val", "test"), context="train_rows")
    _assert_no_disallowed_doc_ids(val_rows, ("train", "test"), context="val_rows")
    _assert_no_disallowed_doc_ids(test_rows, ("train", "val"), context="test_rows")

    train_word_counts = _train_word_counts(train_rows)

    nn_pair_cap = predict_sample_limit

    val_pairs_all = _to_sentence_pairs(val_rows)
    val_nn = _maybe_cap_pairs(val_pairs_all, nn_pair_cap, "val (ByT5)")

    base_nn = TransformerConfig()
    nn_cfg = (
        replace(
            base_nn,
            max_epochs=1,
            pretrain_epochs=1,
            finetune_epochs=1,
        )
        if nn_pair_cap is not None
        else base_nn
    )

    is_primary = seed == primary_seed
    seed_suffix = "" if is_primary else f"__seed{seed}"
    model_dir = out_dir / "models" / model_name / regime
    if not is_primary:
        model_dir = model_dir / f"seed{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    nn_ckpt_dir = _neural_train_checkpoint_dir(model_dir)
    nn_ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_nn = _neural_resume_effective(neural_resume)
    (model_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "regime": regime,
                "seed": seed,
                "primary_seed": primary_seed,
                "split_manifest_hash": split_manifest_hash(),
                "hparams": frozen_hparams_dict(),
                "predict_sample_limit": predict_sample_limit,
                "short_neural_fit": predict_sample_limit is not None,
                "neural_resume": neural_resume,
                "neural_resume_effective": resume_nn,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    n_params: Optional[int] = None
    eff_classical_cfg = classical_cfg or ClassicalConfig()
    neural_batch_correct_fn = None
    if model_name == "classical":
        model = ClassicalCorrector(
            cfg=eff_classical_cfg, phase2_train_only_dir=phase2_train_only_dir
        )
        print("[train] classical: fitting lexicon + KN word LM on TRAIN...", flush=True)
        t_fit = time.perf_counter()
        model.fit([str(r["clean"]) for r in train_rows])
        print(f"[train] classical: fit done in {time.perf_counter() - t_fit:.1f}s", flush=True)
        if predict_sample_limit is not None:
            print(
                "[train] classical: (predict-sample) used full train split for LM "
                "(only ByT5 + JSONL outputs are capped).",
                flush=True,
            )
        model.save(model_dir)
        correct_fn = model.correct_sentence
    elif model_name == "neural":
        model = ByteTransformerCorrector(cfg=nn_cfg, seed=seed)
        print(
            f"[train] neural: regime={regime} val_pairs={len(val_nn)} "
            f"(predict_sample_limit={predict_sample_limit})",
            flush=True,
        )
        t_fit = time.perf_counter()
        training_metrics = _train_neural(
            model,
            rows,
            regime,
            val_nn,
            train_pair_cap=nn_pair_cap,
            checkpoint_dir=nn_ckpt_dir,
            neural_resume=resume_nn,
        )
        print(f"[train] neural: training done in {time.perf_counter() - t_fit:.1f}s", flush=True)
        model.save(model_dir)
        if predict_sample_limit is not None:
            print(
                f"[train] neural: (predict-sample) checkpoint saved -> {model_dir / 'transformer.pt'}",
                flush=True,
            )
        n_params = int(training_metrics.get("n_params") or 0) or None
        (model_dir / "training_metrics.json").write_text(
            json.dumps(training_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        correct_fn = model.correct_sentence
        # Neural model supports batched generation; the runner uses it
        # below in ``_predict_records`` for ~5-15x faster val/test passes.
        neural_batch_correct_fn = model.correct_batch_with_logs
    elif model_name == "hybrid":
        classical = ClassicalCorrector(
            cfg=eff_classical_cfg, phase2_train_only_dir=phase2_train_only_dir
        )
        print("[train] hybrid: fitting classical...", flush=True)
        t0 = time.perf_counter()
        classical.fit([str(r["clean"]) for r in train_rows])
        print(f"[train] hybrid: classical fit done in {time.perf_counter() - t0:.1f}s", flush=True)
        neural = ByteTransformerCorrector(cfg=nn_cfg, seed=seed)
        print(
            f"[train] hybrid: regime={regime} val_pairs={len(val_nn)} "
            f"(predict_sample_limit={predict_sample_limit})",
            flush=True,
        )
        t1 = time.perf_counter()
        training_metrics = _train_neural(
            neural,
            rows,
            regime,
            val_nn,
            train_pair_cap=nn_pair_cap,
            checkpoint_dir=nn_ckpt_dir,
            neural_resume=resume_nn,
        )
        print(f"[train] hybrid: neural fit done in {time.perf_counter() - t1:.1f}s", flush=True)
        classical.save(model_dir / "classical_assets")
        neural.save(model_dir / "neural_assets")
        if predict_sample_limit is not None:
            print(
                f"[train] hybrid: (predict-sample) neural checkpoint saved -> "
                f"{model_dir / 'neural_assets' / 'transformer.pt'}",
                flush=True,
            )
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
        cal = model.calibrate_on_val(val_nn, max_pairs=min(400, len(val_nn)))
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
        wall = time.strftime("%H:%M:%S")
        print(
            f"[predict] (predict-sample N={predict_sample_limit}) {wall}  "
            f"scoring val ({len(val_predict_rows)} rows) then test ({len(test_predict_rows)} rows); "
            f"full split sizes val={len(val_rows)} test={len(test_rows)}",
            flush=True,
        )

    predict_batch = int(getattr(TransformerConfig(), "predict_batch_size", 1))
    val_records, val_latencies = _predict_records(
        model_name,
        correct_fn,
        val_predict_rows,
        train_word_counts,
        blind_test=False,
        batch_correct_fn=neural_batch_correct_fn,
        batch_size=predict_batch if neural_batch_correct_fn is not None else 1,
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
        model_name,
        correct_fn,
        test_predict_rows,
        train_word_counts,
        blind_test=True,
        batch_correct_fn=neural_batch_correct_fn,
        batch_size=predict_batch if neural_batch_correct_fn is not None else 1,
    )
    test_path = out_dir / "predictions" / "test_blind" / model_name / f"{regime}{seed_suffix}.jsonl"
    print(f"[write] test_blind predictions -> {test_path}", flush=True)
    _jsonl_write(test_path, test_records)

    if predict_sample_limit is not None:
        wall = time.strftime("%H:%M:%S")
        print(
            f"[predict] (predict-sample) {wall}  prediction pass done for "
            f"{model_name}/{regime}: val={len(val_records)} test={len(test_records)} lines.",
            flush=True,
        )

    metric_payload = _val_metric_payload(val_records, model_name, regime, seed)
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
    neural_resume: bool = True,
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
            f"(caps ByT5 train/val pairs for fit with 1 epoch per stage, and "
            f"caps val/test prediction JSONL rows per model)",
            flush=True,
        )
        _predict_sample_progress(
            predict_sample_limit,
            "Expect: phase1 → train-only stats (if rebuilt) → manifests → "
            "classical tuning → per-regime identity + models.",
        )
        _predict_sample_progress(
            predict_sample_limit,
            f"Output directory: {cfg.output_dir}",
        )

    if not skip_phase1:
        print("[phase4] running phase 1 alignment to regenerate matched_pairs.json", flush=True)
        _run_phase1(repo_root)
    else:
        print("[phase4] (skip-phase1) reusing existing phase1_output/", flush=True)
    _predict_sample_progress(predict_sample_limit, "Phase 1 finished or skipped.")

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
        _predict_sample_progress(
            predict_sample_limit,
            "Ensuring train-only Phase 2 stats + Phase 3 noise (may take several minutes if rebuilding)...",
        )
        _ensure_train_only_stats(cfg)
    _predict_sample_progress(predict_sample_limit, "Train-only Phase 2/3 artifacts ready.")

    manifests_dir = repo_root / "phase4" / "data" / "manifests"
    if skip_manifests:
        print("[phase4] (skip-manifests) reusing existing manifests/*.jsonl", flush=True)
        _predict_sample_progress(predict_sample_limit, "Reusing existing manifests.")
        manifest_paths = {
            r: manifests_dir / f"{r}.jsonl" for r in all_regimes
        }
        missing = [str(p) for p in manifest_paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"--skip-manifests requested but the following manifests are missing: {missing}"
            )
        _predict_sample_progress(predict_sample_limit, "Manifest paths verified (reused).")
    else:
        _predict_sample_progress(
            predict_sample_limit,
            "Building phase4 manifests from train-only data (I/O + alignment)...",
        )
        print("[phase4] building manifests from train-only synthetic + real pairs ...", flush=True)
        manifest_paths = build_phase4_manifests(
            phase1_output_dir=cfg.phase1_output_dir,
            phase3_output_dir=cfg.phase3_train_only_dir,
            manifests_dir=manifests_dir,
            min_pair_sim=cfg.manifest_min_pair_sim,
            max_len_ratio_delta=cfg.manifest_max_len_ratio_delta,
            synthetic_real_oversample_ratio=cfg.synthetic_real_oversample_ratio,
        )
        print(f"[phase4] manifests: {manifest_paths}", flush=True)
        _predict_sample_progress(predict_sample_limit, "Manifests ready.")

    runs: List[Dict[str, object]] = []
    selected_checkpoints: Dict[str, object] = {}
    failures: List[Dict[str, object]] = []
    print(f"[phase4] models_to_run={models_to_run}", flush=True)
    print(f"[phase4] regimes_to_run={regimes_to_run}", flush=True)

    needs_classical_val_tune = any(
        m in models_to_run for m in ("classical", "hybrid")
    )
    if needs_classical_val_tune:
        _predict_sample_progress(
            predict_sample_limit,
            "Loading real_only manifest and running classical threshold tuning (full val; can take a few minutes)...",
        )
        real_rows_for_tuning = load_jsonl(manifest_paths["real_only"])
        classical_cfg = _tune_classical_on_real_val(
            real_rows=real_rows_for_tuning,
            out_dir=cfg.output_dir,
            phase2_train_only_dir=cfg.phase2_train_only_dir,
        )
        _predict_sample_progress(
            predict_sample_limit,
            "Classical tuning done; starting regime / model loop.",
        )
    else:
        classical_cfg = None
        print(
            "[phase4] skipping classical val tuning (only needed for classical/hybrid "
            f"in models_to_run={models_to_run})",
            flush=True,
        )
        _predict_sample_progress(
            predict_sample_limit,
            "Starting regime / model loop (no classical threshold tuning).",
        )

    for regime in regimes_to_run:
        rows = load_jsonl(manifest_paths[regime])
        _predict_sample_progress(
            predict_sample_limit,
            f"--- regime={regime}  ({len(rows)} manifest rows) ---",
        )
        train_word_counts = _train_word_counts([r for r in rows if r["split"] == "train"])
        try:
            _predict_sample_progress(
                predict_sample_limit,
                f"regime={regime}: writing identity baseline...",
            )
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
            _predict_sample_progress(
                predict_sample_limit,
                f"regime={regime}: identity baseline done.",
            )
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
                    _predict_sample_progress(
                        predict_sample_limit,
                        f"regime={regime}  starting {model_name}  seed={seed} ...",
                    )
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
                        neural_resume=neural_resume,
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
                    _predict_sample_progress(
                        predict_sample_limit,
                        f"regime={regime}  finished {model_name}  seed={seed}.",
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
    _predict_sample_progress(predict_sample_limit, "Aggregating paper tables + run manifest...")
    table_paths = build_all_tables(
        out_dir=cfg.output_dir,
        primary_seed=primary_seed,
        all_seeds=list(seeds),
        phase1_output_dir=cfg.phase1_output_dir,
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
    _predict_sample_progress(predict_sample_limit, "All phase4 artifacts written; run complete.")


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
        help="Short sanity run + smaller artifacts: caps ByT5 train and val "
        "pairs to the first N each (1 epoch per stage) and writes at most N val "
        "and N test rows per model to prediction JSONL (and cross-domain A->B "
        "from the capped test set). Omit for full training and full predictions.",
    )
    parser.add_argument(
        "--no-neural-resume",
        action="store_true",
        help="Disable ByT5 mid-training resume (ignore epoch checkpoints under "
        "models/.../_neural_train_checkpoint/). Default is to resume when files exist.",
    )
    parser.add_argument(
        "--rebuild-val-metrics",
        action="store_true",
        help="Recovery mode: rebuild val_metrics JSON from existing val prediction "
        "JSONL files for the selected --regimes/--seeds/--metrics-models, then exit "
        "without training or predicting.",
    )
    parser.add_argument(
        "--metrics-models",
        type=str,
        default=None,
        help="Comma-separated models for --rebuild-val-metrics "
        "(identity,classical,neural,hybrid). Default follows the normal "
        "--skip-classical/--skip-neural/--skip-hybrid selection.",
    )
    parser.add_argument(
        "--rebuild-paper-tables",
        action="store_true",
        help="With --rebuild-val-metrics, refresh phase4_output/paper_tables/*.csv "
        "after rebuilding the selected val metrics.",
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
    if args.rebuild_val_metrics:
        all_regimes = ("real_only", "synthetic_only", "synthetic_plus_real")
        regimes_to_rebuild = regimes or list(all_regimes)
        bad_regimes = [r for r in regimes_to_rebuild if r not in all_regimes]
        if bad_regimes:
            raise ValueError(f"Unknown regime(s): {bad_regimes}. Allowed: {all_regimes}")

        if args.metrics_models:
            models_to_rebuild = [m.strip() for m in args.metrics_models.split(",") if m.strip()]
        else:
            models_to_rebuild = []
            if not args.skip_classical:
                models_to_rebuild.append("classical")
            if not args.skip_neural:
                models_to_rebuild.append("neural")
            if not args.skip_hybrid:
                models_to_rebuild.append("hybrid")
        all_metric_models = ("identity", "classical", "neural", "hybrid")
        bad_models = [m for m in models_to_rebuild if m not in all_metric_models]
        if bad_models:
            raise ValueError(f"Unknown metrics model(s): {bad_models}. Allowed: {all_metric_models}")
        if not models_to_rebuild:
            raise ValueError("No models selected for --rebuild-val-metrics.")

        seeds_to_rebuild = seeds or list(SEEDS)
        primary_seed = seeds_to_rebuild[0]
        print(
            f"[recover] rebuilding val metrics for models={models_to_rebuild} "
            f"regimes={regimes_to_rebuild} seeds={seeds_to_rebuild}",
            flush=True,
        )
        for seed in seeds_to_rebuild:
            for regime in regimes_to_rebuild:
                for model_name in models_to_rebuild:
                    rebuild_val_metrics_from_predictions(
                        repo_root=repo_root,
                        model_name=model_name,
                        regime=regime,
                        seed=seed,
                        primary_seed=primary_seed,
                        rebuild_paper_tables=False,
                    )
        if args.rebuild_paper_tables:
            cfg = default_run_config(repo_root)
            table_paths = build_all_tables(
                out_dir=cfg.output_dir,
                primary_seed=primary_seed,
                all_seeds=seeds_to_rebuild,
            )
            for name, path in table_paths.items():
                print(f"[recover] paper_table[{name}] -> {path}", flush=True)
        print("Phase 4 val metrics recovery completed.", flush=True)
        return

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
        neural_resume=not args.no_neural_resume,
    )
    print("Phase 4 completed.", flush=True)


if __name__ == "__main__":
    main()
