"""Phase 4 manifest construction.

Builds three Phase 4 training/eval manifests:

* ``real_only``           — Phase 1 verified OCR/GT pairs.
* ``synthetic_only``      — Phase 3 sentence-level (clean, noisy) pairs.
* ``synthetic_plus_real`` — concatenation of the above.

**Synthetic pairs are taken directly from Phase 3 output**: Phase 3 emits one
``<doc>_pairs.jsonl`` per generator subdir, where each row is a true 1:1
``(clean_sentence, noisy_sentence)`` correspondence by construction. 

Per-pair similarity is recomputed here only as QA (``SequenceMatcher.ratio``
between the noisy and clean sides of each row), so we can see how
much corruption the noise model introduced.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

from phase4.data.splits import (
    assert_disjoint_splits,
    assert_no_unknown_docs,
    get_split,
    get_subset_domain,
)



def _read_phase1_real_pairs(phase1_output_dir: Path) -> Dict[str, List[Tuple[str, str]]]:
    pairs_by_doc: Dict[str, List[Tuple[str, str]]] = {}
    for split_dir in phase1_output_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for doc_dir in split_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            pair_path = doc_dir / "matched_pairs.json"
            if not pair_path.exists():
                continue
            data = json.loads(pair_path.read_text(encoding="utf-8"))
            pairs = []
            for r in data:
                noisy = str(r.get("ocr", "")).strip()
                clean = str(r.get("gt", "")).strip()
                if not noisy or not clean:
                    continue
                if noisy == clean:
                    continue
                pairs.append((noisy, clean))
            pairs_by_doc[doc_dir.name] = pairs
    return pairs_by_doc


def _read_phase3_synth_pairs(
    phase3_output_dir: Path,
    synthetic_subdir: str = "structure_aware_noise",
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, Dict[str, object]]]:
    """Read Phase 3's pre-aligned ``<doc>_pairs.jsonl`` files directly.
    Each JSONL file holds rows ``{"clean": ..., "noisy": ...}`` produced by
    Phase 3's ``Phase3SyntheticNoise.generate_with_pairs`` — i.e. each row is
    the noisy version of exactly one clean sentence. 
    """
    pairs_by_doc: Dict[str, List[Tuple[str, str]]] = {}
    qa_by_doc: Dict[str, Dict[str, object]] = {}
    synth_dir = phase3_output_dir / synthetic_subdir
    pair_files = sorted(synth_dir.glob("*_pairs.jsonl"))
    if not pair_files:
        raise FileNotFoundError(
            f"No '*_pairs.jsonl' files under {synth_dir}. Re-run Phase 3 to "
            "emit sentence-level pair files (Phase3SyntheticNoise.run())."
        )
    print(
        f"[manifest] reading synthetic pairs from {synth_dir} "
        f"({len(pair_files)} files)...",
        flush=True,
    )
    for idx, pair_path in enumerate(pair_files, start=1):
        doc_id = pair_path.stem.replace("_pairs", "")
        pairs: List[Tuple[str, str]] = []
        sims: List[float] = []
        with pair_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                clean = str(obj.get("clean", "")).strip()
                noisy = str(obj.get("noisy", "")).strip()
                if not clean or not noisy:
                    continue
                if clean == noisy:
                    # Identity pairs add no learning signal; drop them so we
                    # do not bias the model toward copy.
                    continue
                pairs.append((noisy, clean))
                sims.append(
                    SequenceMatcher(None, noisy.lower(), clean.lower()).ratio()
                )
        pairs_by_doc[doc_id] = pairs
        n = max(1, len(sims))
        qa_by_doc[doc_id] = {
            "aligned_pairs": len(pairs),
            "source_sentences": len(pairs),
            "target_sentences": len(pairs),
            "mismatch_ratio": 0.0,
            "mean_alignment_similarity": round(sum(sims) / n, 4),
            "median_alignment_similarity": round(
                sorted(sims)[len(sims) // 2] if sims else 0.0, 4
            ),
            "min_alignment_similarity": round(min(sims), 4) if sims else 0.0,
            "max_alignment_similarity": round(max(sims), 4) if sims else 0.0,
            "low_alignment_ratio": round(
                sum(1 for s in sims if s < 0.45) / n, 4
            ),
            "very_low_alignment_ratio": round(
                sum(1 for s in sims if s < 0.30) / n, 4
            ),
        }
        qa = qa_by_doc[doc_id]
        print(
            f"[manifest] synthetic {idx}/{len(pair_files)} doc={doc_id!r} "
            f"pairs={qa['aligned_pairs']} "
            f"mean_sim={qa['mean_alignment_similarity']} "
            f"low_sim_ratio={qa['low_alignment_ratio']} "
            f"very_low_sim_ratio={qa['very_low_alignment_ratio']}",
            flush=True,
        )
    return pairs_by_doc, qa_by_doc


def _build_manifest_rows(
    pairs_by_doc: Dict[str, List[Tuple[str, str]]],
    source_type: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    assert_no_unknown_docs(pairs_by_doc.keys())
    for doc_id in sorted(pairs_by_doc):
        split = get_split(doc_id)
        subset_domain = get_subset_domain(doc_id)
        for idx, (noisy, clean) in enumerate(pairs_by_doc[doc_id]):
            rows.append(
                {
                    "doc_id": doc_id,
                    "split": split,
                    "subset_domain": subset_domain,
                    "sample_id": f"{doc_id}:{idx}",
                    "order_key": idx,
                    "token_or_sentence_level": "sentence",
                    "noisy": noisy,
                    "clean": clean,
                    "source_type": source_type,
                }
            )
    return rows


def _check_manifest(rows: List[Dict[str, object]]) -> None:
    seen = set()
    for row in rows:
        doc_id = str(row["doc_id"])
        if get_split(doc_id) != row["split"]:
            raise AssertionError(f"Split mismatch for {doc_id}")
        key = (doc_id, row["sample_id"], row["source_type"])
        if key in seen:
            raise AssertionError(f"Duplicate key in manifest: {key}")
        seen.add(key)


def _validate_manifest_quality(summary: Dict[str, Dict[str, object]]) -> None:
    """Refuse manifests where synthetic supervision is implausibly broken.

    Thresholds operate on **real** SequenceMatcher similarity (per-pair),
    which is meaningful since pairs are 1:1 by construction. ``low_align``
    counts pairs below 0.45 sim; ``very_low_align`` counts pairs below 0.30.
    """
    bad: List[str] = []
    for doc_id, row in summary.items():
        very_low = float(row.get("very_low_alignment_ratio", 0.0))
        low = float(row.get("low_alignment_ratio", 0.0))
        mean_sim = float(row.get("mean_alignment_similarity", 0.0))
        pairs = int(row.get("aligned_pairs", 0))
        if pairs <= 0:
            bad.append(f"{doc_id}: zero pairs")
            continue
        if very_low > 0.40:
            bad.append(
                f"{doc_id}: very_low_alignment_ratio={very_low} (>0.40)"
            )
        if low > 0.75:
            bad.append(f"{doc_id}: low_alignment_ratio={low} (>0.75)")
        if mean_sim < 0.40:
            bad.append(f"{doc_id}: mean_alignment_similarity={mean_sim} (<0.40)")
    if bad:
        msg = "\n".join(bad)
        raise AssertionError(f"Manifest QA failed:\n{msg}")


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_phase4_manifests(
    phase1_output_dir: Path,
    phase3_output_dir: Path,
    manifests_dir: Path,
    synthetic_subdir: str = "structure_aware_noise",
) -> Dict[str, Path]:
    """Assemble the three Phase 4 manifests from Phase 1 and Phase 3 outputs.

    ``phase3_output_dir`` is expected to contain ``<synthetic_subdir>/<doc>_pairs.jsonl``
    files emitted by Phase 3. 
    """
    print("[manifest] build_phase4_manifests: start", flush=True)
    assert_disjoint_splits()
    print(f"[manifest] loading real pairs from {phase1_output_dir} ...", flush=True)
    real_pairs = _read_phase1_real_pairs(phase1_output_dir)
    print(f"[manifest] real docs loaded: {len(real_pairs)}", flush=True)

    synth_pairs, synth_qa = _read_phase3_synth_pairs(
        phase3_output_dir,
        synthetic_subdir=synthetic_subdir,
    )
    print(f"[manifest] synthetic docs loaded: {len(synth_pairs)}", flush=True)
    _validate_manifest_quality(synth_qa)

    print("[manifest] building manifest rows...", flush=True)
    real_rows = _build_manifest_rows(real_pairs, "real")
    synth_rows = _build_manifest_rows(synth_pairs, "synthetic")
    combined_rows = sorted(
        real_rows + synth_rows,
        key=lambda r: (r["doc_id"], r["order_key"], r["source_type"]),
    )

    for rows in (real_rows, synth_rows, combined_rows):
        _check_manifest(rows)

    outputs = {
        "real_only": manifests_dir / "real_only.jsonl",
        "synthetic_only": manifests_dir / "synthetic_only.jsonl",
        "synthetic_plus_real": manifests_dir / "synthetic_plus_real.jsonl",
    }
    print(
        f"[manifest] writing jsonl: real={len(real_rows)} synth={len(synth_rows)} "
        f"combined={len(combined_rows)}",
        flush=True,
    )
    _write_jsonl(outputs["real_only"], real_rows)
    _write_jsonl(outputs["synthetic_only"], synth_rows)
    _write_jsonl(outputs["synthetic_plus_real"], combined_rows)

    summary: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for regime, rows in (
        ("real_only", real_rows),
        ("synthetic_only", synth_rows),
        ("synthetic_plus_real", combined_rows),
    ):
        for row in rows:
            summary[regime][row["split"]]["total"] += 1
            summary[regime][row["split"]][row["source_type"]] += 1
    manifest_summary: Dict[str, object] = {
        regime: {split: dict(stats) for split, stats in split_stats.items()}
        for regime, split_stats in summary.items()
    }
    manifest_summary["synthetic_alignment_qa"] = synth_qa
    manifest_summary["alignment_config"] = {
        "method": "phase3_native_pairs",
        "synthetic_subdir": synthetic_subdir,
    }
    (manifests_dir / "manifest_summary.json").write_text(
        json.dumps(manifest_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[manifest] wrote summary -> {manifests_dir / 'manifest_summary.json'}",
        flush=True,
    )
    print("[manifest] build_phase4_manifests: done", flush=True)
    return outputs


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
