"""Phase 4 evaluation metrics.

Provides:
- ``cer`` / ``wer`` (sentence-level)
- ``chrf_score`` (character n-gram F-score, n=6, beta=2)
- ``aggregate_metrics`` (CER, WER, chrF, sentence accuracy, correction rate,
   overcorrection rate)
- ``rare_and_name_corruption`` (token-level diagnostics)
- ``per_domain_breakdown`` (A vs B subset cross-domain split)
- ``paired_bootstrap`` (significance test on paired sentence-level CER/WER)
- ``calibration_bins`` (reliability diagram inputs)
- ``oracle_per_token`` (per-token oracle: pick closest of {identity, system A,
   system B, ...} to gold)
"""

from __future__ import annotations

import math
import random
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _edit_components(reference: Sequence, hypothesis: Sequence) -> Tuple[int, int, int, int]:
    matches = subs = dels = ins = 0
    ops = SequenceMatcher(None, list(reference), list(hypothesis)).get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            matches += i2 - i1
        elif tag == "replace":
            overlap = min(i2 - i1, j2 - j1)
            subs += overlap
            dels += max(0, (i2 - i1) - overlap)
            ins += max(0, (j2 - j1) - overlap)
        elif tag == "delete":
            dels += i2 - i1
        elif tag == "insert":
            ins += j2 - j1
    return matches, subs, dels, ins


def cer(reference: str, hypothesis: str) -> float:
    matches, subs, dels, ins = _edit_components(reference, hypothesis)
    denom = max(1, matches + subs + dels)
    return (subs + dels + ins) / denom


def wer(reference: str, hypothesis: str) -> float:
    matches, subs, dels, ins = _edit_components(reference.split(), hypothesis.split())
    denom = max(1, matches + subs + dels)
    return (subs + dels + ins) / denom


def cer_reduction_rate(input_cer: float, output_cer: float) -> float:
    """
    ``(input_cer - output_cer) / max(input_cer, eps)``: the fraction of input
    error that the system removed. Positive values mean the system corrected
    more than it broke; negative values mean net regression. We clamp at -1
    so cells where a system doubled the CER show as -1.0 rather than wildly
    negative numbers, which simplifies plots.
    """
    eps = 1e-9
    if input_cer <= eps and output_cer <= eps:
        return 0.0
    if input_cer <= eps:
        return -1.0
    delta = (input_cer - output_cer) / input_cer
    return max(-1.0, min(1.0, delta))



def _char_ngrams(text: str, n: int) -> Counter:
    if not text:
        return Counter()
    if len(text) < n:
        return Counter([text])
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def chrf_score(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (averaged over orders 1..n)."""
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0
    f_scores: List[float] = []
    for k in range(1, n + 1):
        ref_grams = _char_ngrams(reference, k)
        hyp_grams = _char_ngrams(hypothesis, k)
        if not ref_grams or not hyp_grams:
            continue
        overlap = sum((ref_grams & hyp_grams).values())
        precision = overlap / max(1, sum(hyp_grams.values()))
        recall = overlap / max(1, sum(ref_grams.values()))
        if precision == 0.0 and recall == 0.0:
            f_scores.append(0.0)
            continue
        beta_sq = beta * beta
        denom = beta_sq * precision + recall
        if denom == 0.0:
            f_scores.append(0.0)
        else:
            f_scores.append((1 + beta_sq) * precision * recall / denom)
    if not f_scores:
        return 0.0
    return sum(f_scores) / len(f_scores)



def is_rare_word(word: str, train_word_counts: Dict[str, int], threshold: int = 1) -> bool:
    return train_word_counts.get(word.lower(), 0) <= threshold


def is_proper_name(word: str) -> bool:
    return bool(word) and word[0].isupper()


def aggregate_metrics(records: Iterable[Dict[str, object]]) -> Dict[str, float]:
    records = list(records)
    if not records:
        return {
            "cer": 0.0,
            "wer": 0.0,
            "chrf": 0.0,
            "input_cer": 0.0,
            "cer_reduction_rate": 0.0,
            "correction_rate": 0.0,
            "overcorrection_rate": 0.0,
            "sentence_accuracy": 0.0,
            "n_records": 0,
        }
    cer_avg = sum(float(r["cer"]) for r in records) / len(records)
    wer_avg = sum(float(r["wer"]) for r in records) / len(records)
    chrf_vals = [float(r["chrf"]) for r in records if "chrf" in r]
    chrf_avg = sum(chrf_vals) / len(chrf_vals) if chrf_vals else 0.0
    correction_rate = sum(1 for r in records if bool(r.get("changed_flag"))) / len(records)
    overcorr_rate = sum(1 for r in records if bool(r.get("overcorrected"))) / len(records)
    sentence_acc = sum(
        1 for r in records if str(r.get("prediction", "")) == str(r.get("reference", ""))
    ) / len(records)
    # Corpus-level input CER and CER^Δ (derived from per-record ``input_cer``
    # if the runner stamped it; otherwise fall back to identity = noisy as
    # input). The headline post-OCR metric: fraction of input error removed.
    input_cers: List[float] = []
    for r in records:
        if "input_cer" in r and r["input_cer"] is not None:
            input_cers.append(float(r["input_cer"]))
        else:
            ref = str(r.get("reference", "") or "")
            noisy = str(r.get("input_noisy", "") or "")
            input_cers.append(cer(ref, noisy))
    input_cer_avg = sum(input_cers) / max(1, len(input_cers))
    return {
        "cer": cer_avg,
        "wer": wer_avg,
        "chrf": chrf_avg,
        "input_cer": input_cer_avg,
        "cer_reduction_rate": cer_reduction_rate(input_cer_avg, cer_avg),
        "correction_rate": correction_rate,
        "overcorrection_rate": overcorr_rate,
        "sentence_accuracy": sentence_acc,
        "n_records": len(records),
    }


def rare_and_name_corruption(records: List[Dict[str, object]]) -> Dict[str, float]:
    rare_total = rare_bad = 0
    name_total = name_bad = 0
    for r in records:
        if "rare_token_total" in r:
            rare_total += int(r.get("rare_token_total", 0))
            rare_bad += int(r.get("rare_token_corrupted", 0))
        elif bool(r.get("is_rare_word")):
            rare_total += 1
            if bool(r.get("corrupted_rare_word")):
                rare_bad += 1
        if "proper_token_total" in r:
            name_total += int(r.get("proper_token_total", 0))
            name_bad += int(r.get("proper_token_corrupted", 0))
        elif bool(r.get("is_proper_name")):
            name_total += 1
            if bool(r.get("corrupted_proper_name")):
                name_bad += 1
    return {
        "rare_word_corruption_rate": rare_bad / max(1, rare_total),
        "proper_name_corruption_rate": name_bad / max(1, name_total),
        "rare_token_total": rare_total,
        "proper_token_total": name_total,
    }



def per_domain_breakdown(records: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    by_domain: Dict[str, List[Dict[str, object]]] = {}
    for r in records:
        d = str(r.get("subset_domain", ""))
        by_domain.setdefault(d, []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for d, recs in by_domain.items():
        out[d] = aggregate_metrics(recs)
    return out



def paired_bootstrap(
    paired_a: Sequence[float],
    paired_b: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 12345,
) -> Dict[str, float]:
    """Two-sided paired bootstrap test for "system A < system B" on a metric.

    Returns mean delta (B - A), 95% CI of delta, and a two-sided p-value.

    Convention: ``paired_a`` and ``paired_b`` are per-sample metric values for
    the SAME samples (e.g. CER per sentence) for two systems. A negative
    delta means system A has the lower (better) metric.
    """
    if len(paired_a) != len(paired_b):
        raise ValueError("Paired arrays must have equal length.")
    n = len(paired_a)
    if n == 0:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n": 0}
    rng = random.Random(seed)
    deltas: List[float] = []
    observed_delta = sum(b - a for a, b in zip(paired_a, paired_b)) / n
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            s += paired_b[i] - paired_a[i]
        deltas.append(s / n)
    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[int(0.975 * n_resamples) - 1]
    if observed_delta >= 0:
        p_one = sum(1 for d in deltas if d <= 0) / n_resamples
    else:
        p_one = sum(1 for d in deltas if d >= 0) / n_resamples
    p_two = min(1.0, 2.0 * p_one)
    return {
        "mean_delta": observed_delta,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_two,
        "n": n,
    }



def calibration_bins(
    records: List[Dict[str, object]],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    bins = [
        {
            "lo": k / n_bins,
            "hi": (k + 1) / n_bins,
            "count": 0,
            "mean_confidence": 0.0,
            "mean_accuracy": 0.0,
        }
        for k in range(n_bins)
    ]
    bucket_conf = [[] for _ in range(n_bins)]
    bucket_acc = [[] for _ in range(n_bins)]
    for r in records:
        conf = float(r.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        idx = min(n_bins - 1, int(conf * n_bins))
        bucket_conf[idx].append(conf)
        bucket_acc[idx].append(
            1.0 if str(r.get("prediction", "")) == str(r.get("reference", "")) else 0.0
        )
    for k, b in enumerate(bins):
        if bucket_conf[k]:
            b["count"] = len(bucket_conf[k])
            b["mean_confidence"] = sum(bucket_conf[k]) / b["count"]
            b["mean_accuracy"] = sum(bucket_acc[k]) / b["count"]
    return bins


def expected_calibration_error(bins: List[Dict[str, float]]) -> float:
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in bins:
        if b["count"] == 0:
            continue
        ece += (b["count"] / total) * abs(b["mean_confidence"] - b["mean_accuracy"])
    return ece



def oracle_per_sentence(
    aligned_predictions: Dict[str, List[Dict[str, object]]],
    references: List[str],
    noisy_inputs: List[str],
) -> Dict[str, float]:
    """Compute oracle-best CER if we could pick the best system per sentence.

    ``aligned_predictions[name]`` is the list of prediction records (already
    aligned by index), each providing ``prediction``.
    """
    n = len(references)
    if n == 0:
        return {"oracle_cer": 0.0, "oracle_wer": 0.0, "n": 0}
    cer_sum = 0.0
    wer_sum = 0.0
    for i in range(n):
        ref = references[i]
        candidates = [noisy_inputs[i]]
        for _name, recs in aligned_predictions.items():
            if i < len(recs):
                candidates.append(str(recs[i]["prediction"]))
        best = min(candidates, key=lambda c: cer(ref, c))
        cer_sum += cer(ref, best)
        wer_sum += wer(ref, best)
    return {
        "oracle_cer": cer_sum / n,
        "oracle_wer": wer_sum / n,
        "n": n,
    }



def identity_baseline_records(
    rows: List[Dict[str, object]],
    train_word_counts: Dict[str, int],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in rows:
        noisy = str(r["noisy"])
        clean = str(r["clean"])
        ref_tokens = clean.split()
        rare_total = rare_bad = name_total = name_bad = 0
        for ref_tok, pred_tok in zip(ref_tokens, noisy.split()):
            tok_is_rare = is_rare_word(ref_tok, train_word_counts)
            tok_is_name = is_proper_name(ref_tok)
            changed = ref_tok != pred_tok
            if tok_is_rare:
                rare_total += 1
                if changed:
                    rare_bad += 1
            if tok_is_name:
                name_total += 1
                if changed:
                    name_bad += 1
        cer_value = cer(clean, noisy)
        out.append(
            {
                "doc_id": r["doc_id"],
                "split": r["split"],
                "subset_domain": r["subset_domain"],
                "sample_id": r["sample_id"],
                "source_type": r.get("source_type"),
                "input_noisy": noisy,
                "prediction": noisy,
                "reference": clean,
                "changed_flag": False,
                "token_was_correct_before": noisy == clean,
                "confidence": 1.0,
                "is_rare_word": any(is_rare_word(t, train_word_counts) for t in ref_tokens),
                "is_proper_name": any(is_proper_name(t) for t in ref_tokens),
                "input_cer": cer_value,
                "cer": cer_value,
                "error_reduction": 0.0,
                "wer": wer(clean, noisy),
                "chrf": chrf_score(clean, noisy),
                "overcorrected": False,
                "corrupted_rare_word": rare_bad > 0,
                "corrupted_proper_name": name_bad > 0,
                "rare_token_total": rare_total,
                "rare_token_corrupted": rare_bad,
                "proper_token_total": name_total,
                "proper_token_corrupted": name_bad,
            }
        )
    return out
