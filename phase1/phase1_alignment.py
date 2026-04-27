import re
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
from collections import Counter
from difflib import SequenceMatcher


PAGE_MARKER_PATTERNS = [
    re.compile(r"-{1,3}\s*[Pp]age\s+\d+\s*-{1,3}"),
    re.compile(r"={3,}\s*PAGE\s*\d+\s*={3,}"),
    re.compile(r"\[Page\s+\d+\]"),
    re.compile(r"^Page\s+\d+\s*$", re.MULTILINE),
    re.compile(r".{1,80}\s*-{3,}\s*$"),
]


# OCR-pipeline batch metadata that contaminates `raw_ocr/*.txt` for some docs
# (notably ``Забите на Ветрот - Томе Арсовски`` which had 261 such lines).
# These markers are NOT in the corresponding ground-truth file, so leaving them
# in pushes phase 1's DP into spurious matches and contaminates val labels.
OCR_BATCH_PATTERNS = [
    re.compile(r"^\s*#\d+\s*$", re.MULTILINE),
    re.compile(r"^\s*BATCH\s+\d+\s*$", re.MULTILINE),
    re.compile(r"^.*МАКЕДОНСКИ\s+OCR\s+ИЗВЕШТАЈ.*$", re.MULTILINE),
    re.compile(r"^\s*Датум:\s*\d.*$", re.MULTILINE),
    re.compile(r"^\s*[#=*]{3,}\s*$", re.MULTILINE),
]


DASH_NOISE_PATTERNS = [
    re.compile(r"[—\-]{3,}"),
    re.compile(r"\s*[—\-]{2,}\s*"),
    re.compile(r"^\s*[—\-]{2,}\s*$", re.MULTILINE),
    re.compile(r"[—\-]{2,}(\s*[—\-]{2,})+"),
]



MIN_DIAGONAL_SIM = 0.30
GAP_PENALTY = -0.5
MIN_PAIR_SIM = 0.30


def remove_dash_noise(text: str) -> str:
    for p in DASH_NOISE_PATTERNS:
        text = p.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def detect_and_strip_page_markers(text: str) -> Tuple[str, bool]:
    found = False
    for p in PAGE_MARKER_PATTERNS:
        if p.search(text):
            text = p.sub("", text)
            found = True
    return text, found


def strip_ocr_batch_metadata(text: str) -> Tuple[str, int]:
    """Remove OCR-pipeline batch metadata lines from raw OCR.

    Returns (cleaned_text, n_lines_removed). The patterns are applied
    line-anchored so we don't accidentally chop content out of body text.
    """
    n_before = text.count("\n")
    for p in OCR_BATCH_PATTERNS:
        text = p.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    n_after = text.count("\n")
    return text, max(0, n_before - n_after)


def is_garbage_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if not re.search(r"[\w\u0400-\u04FF]", s):
        return True
    if len(s) <= 1:
        return True
    if re.fullmatch(r"\d+", s):
        return True
    return False


def extract_sentences(path: str):
    raw = Path(path).read_text(encoding="utf-8")
    cleaned, _ = detect_and_strip_page_markers(raw)
    cleaned, _ = strip_ocr_batch_metadata(cleaned)
    cleaned = remove_dash_noise(cleaned)

    lines = cleaned.splitlines()
    filtered = []
    for line in lines:
        if is_garbage_line(line):
            continue
        filtered.append(line)

    text = re.sub(r"\s+", " ", " ".join(filtered)).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def build_features(paras):
    feats = []
    for p in paras:
        words = set(re.findall(r"\w+", p.lower()))
        shingles = Counter(p[i:i + 5] for i in range(len(p) - 4))
        feats.append((p, words, shingles))
    return feats


def fast_similarity(a, b):
    _, a_words, a_sh = a
    _, b_words, b_sh = b

    word = len(a_words & b_words) / (math.sqrt(len(a_words) * len(b_words)) + 1e-6)

    dot = sum(a_sh[k] * b_sh.get(k, 0) for k in a_sh)
    ma = math.sqrt(sum(v * v for v in a_sh.values()))
    mb = math.sqrt(sum(v * v for v in b_sh.values()))
    char = dot / (ma * mb + 1e-6)

    return 0.75 * char + 0.25 * word


def align_sentences(
    ocr_sents,
    gt_sents,
    min_diagonal_sim: float = MIN_DIAGONAL_SIM,
    gap_penalty: float = GAP_PENALTY,
    min_pair_sim: float = MIN_PAIR_SIM,
):
    """Align OCR sentences to GT sentences via DP with quality guards.

    The diagonal (i+1, j+1) step is only allowed when ``sim(i, j)`` clears
    ``min_diagonal_sim``; otherwise the DP must skip one side (paying
    ``gap_penalty``). After backtracking we additionally drop pairs whose
    final ``SequenceMatcher`` similarity is below ``min_pair_sim`` -
    these are leftover forced matches when no good alignment exists.
    """
    ocr_feats = build_features(ocr_sents)
    gt_feats = build_features(gt_sents)

    m, n = len(ocr_sents), len(gt_sents)
    dp = [[-1e18] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    def sim(i, j):
        return fast_similarity(ocr_feats[i], gt_feats[j])

    for i in range(m + 1):
        for j in range(n + 1):
            if dp[i][j] < -1e17:
                continue

            cur = dp[i][j]

            if i < m and j < n:
                s = sim(i, j)
                if s >= min_diagonal_sim:
                    val = cur + s
                    if val > dp[i + 1][j + 1]:
                        dp[i + 1][j + 1] = val
                        back[i + 1][j + 1] = (i, j)

            if i < m:
                val = cur + gap_penalty
                if val > dp[i + 1][j]:
                    dp[i + 1][j] = val
                    back[i + 1][j] = (i, j)

            if j < n:
                val = cur + gap_penalty
                if val > dp[i][j + 1]:
                    dp[i][j + 1] = val
                    back[i][j + 1] = (i, j)

    if back[m][n] is None:
        min_len = min(len(ocr_sents), len(gt_sents))
        raw_pairs = list(zip(ocr_sents[:min_len], gt_sents[:min_len]))
    else:
        i, j = m, n
        raw_pairs = []
        while (i, j) != (0, 0):
            pi, pj = back[i][j]
            if i > pi and j > pj:
                raw_pairs.append((ocr_sents[i - 1], gt_sents[j - 1]))
            i, j = pi, pj
        raw_pairs.reverse()

    pairs = []
    for ocr, gt in raw_pairs:
        if not ocr.strip() or not gt.strip():
            continue
        ratio = SequenceMatcher(None, ocr, gt).ratio()
        if ratio < min_pair_sim:
            continue
        pairs.append((ocr, gt))
    return pairs


@dataclass
class WordEditOp:
    op: str
    ocr_word: str
    gt_word: str


def word_align(ocr: str, gt: str):
    ow = ocr.split()
    gw = gt.split()
    ops = []

    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ow, gw).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ops.append(WordEditOp("match", ow[i1 + k], gw[j1 + k]))

        elif tag == "replace":
            m = min(i2 - i1, j2 - j1)

            for k in range(m):
                ops.append(WordEditOp("substitution", ow[i1 + k], gw[j1 + k]))

            for k in range(m, i2 - i1):
                ops.append(WordEditOp("insertion", ow[i1 + k], ""))

            for k in range(m, j2 - j1):
                ops.append(WordEditOp("deletion", "", gw[j1 + k]))

        elif tag == "delete":
            for k in range(i1, i2):
                ops.append(WordEditOp("insertion", ow[k], ""))

        elif tag == "insert":
            for k in range(j1, j2):
                ops.append(WordEditOp("deletion", "", gw[k]))

    return ops


@dataclass
class CharEditOp:
    op: str
    ocr_char: str
    gt_char: str


def char_align(ocr: str, gt: str):
    ops = []

    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ocr, gt).get_opcodes():
        if tag == "equal":
            for i in range(i2 - i1):
                ops.append(CharEditOp("match", ocr[i1 + i], gt[j1 + i]))

        elif tag == "replace":
            m = min(i2 - i1, j2 - j1)

            for i in range(m):
                ops.append(CharEditOp("substitution", ocr[i1 + i], gt[j1 + i]))

            for i in range(m, i2 - i1):
                ops.append(CharEditOp("insertion", ocr[i1 + i], ""))

            for i in range(m, j2 - j1):
                ops.append(CharEditOp("deletion", "", gt[j1 + i]))

        elif tag == "delete":
            for i in range(i1, i2):
                ops.append(CharEditOp("insertion", ocr[i], ""))

        elif tag == "insert":
            for i in range(j1, j2):
                ops.append(CharEditOp("deletion", "", gt[i]))

    return ops


def normalize_token(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\u0400-\u04FF]+", "", s)
    return s


def token_similarity(a: str, b: str) -> float:
    a_n = normalize_token(a)
    b_n = normalize_token(b)

    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0

    return SequenceMatcher(None, a_n, b_n).ratio()


def _token_join(tokens):
    return "".join(tokens)



WB_DP_INF = 1e18
WB_DP_DEL = 0.92
WB_DP_INS = 0.92
# Preference for plain 1:1 alignments over merge/split when costs tie.
WB_DP_LAMBDA_PAIR = 0.36
WB_DP_LAMBDA_TRIP = 0.46

WB_DP_JUNK_PREFIX = 0.62

WB_DP_MERGE_SHORT_TAIL = 0.28
WB_DP_MERGE_SHORT_TAIL_SIM_FLOOR = 0.93
# Do not emit merge/split unless token_similarity on the merged spans reaches this
# floor; otherwise structural ops act as cheap ``skip many tokens'' hacks.
WB_DP_MIN_STRUCT_SIM = 0.82


def _substitution_cost(ot: str, gtok: str) -> float:
    if normalize_token(ot) == normalize_token(gtok):
        return 0.0
    return 1.0 - token_similarity(ot, gtok)


def _split_junk_extra(first_ocr: str) -> float:
    if len(normalize_token(first_ocr)) <= 1:
        return WB_DP_JUNK_PREFIX
    return 0.0


def detect_word_boundary_errors(ocr: str, gt: str):
    """Detect merge/split word-boundary errors.

    ``Emitted
    ``similarity`` values are in ``[0, 1]`` (token_similarity on the
    merged spans) so Phase 2 plots and downstream JSON consumers stay stable.

    Returns a list of dicts with ``type`` in ``{"merge","split"}``, ``ocr``,
    ``gt`` and ``similarity``
    """
    ow = ocr.split()
    gw = gt.split()
    if not ow or not gw:
        return []

    n, m = len(ow), len(gw)
    dp = [[WB_DP_INF] * (m + 1) for _ in range(n + 1)]
    pi = [[-1] * (m + 1) for _ in range(n + 1)]
    pj = [[-1] * (m + 1) for _ in range(n + 1)]
    op = [[-1] * (m + 1) for _ in range(n + 1)]

    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + WB_DP_DEL
        pi[i][0] = i - 1
        pj[i][0] = 0
        op[i][0] = 2  # delete ocr
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + WB_DP_INS
        pi[0][j] = 0
        pj[0][j] = j - 1
        op[0][j] = 3  # insert gt

    def relax(i, j, cand, opi, opj, opc):
        nonlocal dp, pi, pj, op
        if cand + 1e-9 < dp[i][j]:
            dp[i][j] = cand
            pi[i][j] = opi
            pj[i][j] = opj
            op[i][j] = opc

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Apply cheaper / more structural transitions first, then plain
            # match/substitution **last** so equal-cost paths prefer 1:1 edits
            # over merge/split

            # MERGE: 1 OCR vs 3 GT
            if j >= 3:
                joined = _token_join(gw[j - 3 : j])
                ms = token_similarity(ow[i - 1], joined)
                if ms >= WB_DP_MIN_STRUCT_SIM:
                    c = 1.0 - ms + WB_DP_LAMBDA_TRIP
                    last = normalize_token(gw[j - 1])
                    if len(last) <= 1 and ms < WB_DP_MERGE_SHORT_TAIL_SIM_FLOOR:
                        c += WB_DP_MERGE_SHORT_TAIL
                    relax(i, j, dp[i - 1][j - 3] + c, i - 1, j - 3, 6)

            # SPLIT: 3 OCR vs 1 GT
            if i >= 3:
                joined = _token_join(ow[i - 3 : i])
                ms = token_similarity(joined, gw[j - 1])
                if ms >= WB_DP_MIN_STRUCT_SIM:
                    c = (
                        1.0
                        - ms
                        + WB_DP_LAMBDA_TRIP
                        + _split_junk_extra(ow[i - 3])
                    )
                    relax(i, j, dp[i - 3][j - 1] + c, i - 3, j - 1, 7)

            # MERGE: 1 OCR word vs 2 GT words
            if j >= 2:
                joined = _token_join(gw[j - 2 : j])
                ms = token_similarity(ow[i - 1], joined)
                if ms >= WB_DP_MIN_STRUCT_SIM:
                    c = 1.0 - ms + WB_DP_LAMBDA_PAIR
                    if len(normalize_token(gw[j - 1])) <= 1 and ms < WB_DP_MERGE_SHORT_TAIL_SIM_FLOOR:
                        c += WB_DP_MERGE_SHORT_TAIL
                    relax(i, j, dp[i - 1][j - 2] + c, i - 1, j - 2, 4)

            # SPLIT: 2 OCR words vs 1 GT word
            if i >= 2:
                joined = _token_join(ow[i - 2 : i])
                ms = token_similarity(joined, gw[j - 1])
                if ms >= WB_DP_MIN_STRUCT_SIM:
                    c = 1.0 - ms + WB_DP_LAMBDA_PAIR + _split_junk_extra(ow[i - 2])
                    relax(i, j, dp[i - 2][j - 1] + c, i - 2, j - 1, 5)

            # delete OCR token (i-1, j) -> (i, j)
            relax(i, j, dp[i - 1][j] + WB_DP_DEL, i - 1, j, 2)

            # insert GT token (missing OCR word): (i, j-1) -> (i, j)
            relax(i, j, dp[i][j - 1] + WB_DP_INS, i, j - 1, 3)

            # match / substitute: (i-1, j-1) -> (i, j)  — last wins on ties
            c = _substitution_cost(ow[i - 1], gw[j - 1])
            opc = 0 if c <= 1e-12 else 1
            relax(i, j, dp[i - 1][j - 1] + c, i - 1, j - 1, opc)

    if dp[n][m] >= WB_DP_INF / 2:
        return []

    # Backtrack from (n, m) to (0, 0); include insert-only / delete-only tails.
    seq = []
    ci, cj = n, m
    while (ci, cj) != (0, 0):
        opc = op[ci][cj]
        pi0, pj0 = pi[ci][cj], pj[ci][cj]
        if pi0 < 0 or pj0 < 0:
            break
        seq.append((opc, pi0, pj0, ci, cj))
        ci, cj = pi0, pj0

    seq.reverse()
    boundary_errors = []
    for rec in seq:
        opc, pi0, pj0, ci, cj = rec
        if opc == 4:  # merge 1 vs 2
            o_tok = ow[ci - 1]
            g_slice = gw[pj0: cj]
            sim = round(token_similarity(o_tok, _token_join(g_slice)), 4)
            boundary_errors.append(
                {
                    "type": "merge",
                    "ocr": o_tok,
                    "gt": list(g_slice),
                    "similarity": sim,
                }
            )
        elif opc == 6:  # merge 1 vs 3
            o_tok = ow[ci - 1]
            g_slice = gw[pj0: cj]
            sim = round(token_similarity(o_tok, _token_join(g_slice)), 4)
            boundary_errors.append(
                {
                    "type": "merge",
                    "ocr": o_tok,
                    "gt": list(g_slice),
                    "similarity": sim,
                }
            )
        elif opc == 5:  # split 2 vs 1
            o_slice = ow[pi0: ci]
            g_tok = gw[cj - 1]
            sim = round(token_similarity(_token_join(o_slice), g_tok), 4)
            boundary_errors.append(
                {
                    "type": "split",
                    "ocr": list(o_slice),
                    "gt": g_tok,
                    "similarity": sim,
                }
            )
        elif opc == 7:  # split 3 vs 1
            o_slice = ow[pi0: ci]
            g_tok = gw[cj - 1]
            sim = round(token_similarity(_token_join(o_slice), g_tok), 4)
            boundary_errors.append(
                {
                    "type": "split",
                    "ocr": list(o_slice),
                    "gt": g_tok,
                    "similarity": sim,
                }
            )

    return boundary_errors


def align_all_pairs(matched_pairs):
    all_word_ops = []
    all_char_ops = []
    all_boundary_errors = []

    for ocr, gt in matched_pairs:
        all_word_ops.extend(word_align(ocr, gt))
        all_char_ops.extend(char_align(ocr, gt))
        all_boundary_errors.extend(detect_word_boundary_errors(ocr, gt))

    return all_char_ops, all_word_ops, all_boundary_errors


def compute_cer(char_ops):
    errors = sum(1 for o in char_ops if o.op != "match")
    total = sum(1 for o in char_ops if o.op in ("match", "substitution", "deletion"))
    return errors / max(total, 1)


def compute_wer(word_ops):
    errors = sum(1 for o in word_ops if o.op != "match")
    total = sum(1 for o in word_ops if o.op in ("match", "substitution", "deletion"))
    return errors / max(total, 1)


def compute_text_statistics(text: str):
    total_characters = len(text)
    tokens = re.findall(r"\w+", text)
    total_tokens = len(tokens)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if s.strip()]
    total_sentences = len(sentences)
    avg_sentence_length = total_tokens / total_sentences if total_sentences > 0 else 0

    return {
        "total_characters": total_characters,
        "total_tokens": total_tokens,
        "total_sentences": total_sentences,
        "avg_sentence_length": round(avg_sentence_length, 4),
    }


def compute_boundary_statistics(boundary_errors):
    split_examples = []
    merge_examples = []
    split_count = 0
    merge_count = 0

    for e in boundary_errors:
        if e["type"] == "split":
            split_count += 1
            if len(split_examples) < 10:
                split_examples.append(e)
        elif e["type"] == "merge":
            merge_count += 1
            if len(merge_examples) < 10:
                merge_examples.append(e)

    return {
        "split_count": split_count,
        "merge_count": merge_count,
        "split_examples": split_examples,
        "merge_examples": merge_examples,
    }


def compute_statistics(ocr_sents, gt_sents, matched_pairs, char_ops, word_ops, boundary_errors):
    ocr_text = " ".join(ocr_sents)
    gt_text = " ".join(gt_sents)

    ocr_stats = compute_text_statistics(ocr_text)
    gt_stats = compute_text_statistics(gt_text)
    boundary_stats = compute_boundary_statistics(boundary_errors)

    return {
        "matched_pairs": len(matched_pairs),
        "ocr_total_characters": ocr_stats["total_characters"],
        "ocr_total_tokens": ocr_stats["total_tokens"],
        "ocr_total_sentences": ocr_stats["total_sentences"],
        "ocr_avg_sentence_length": ocr_stats["avg_sentence_length"],
        "gt_total_characters": gt_stats["total_characters"],
        "gt_total_tokens": gt_stats["total_tokens"],
        "gt_total_sentences": gt_stats["total_sentences"],
        "gt_avg_sentence_length": gt_stats["avg_sentence_length"],
        "CER": round(compute_cer(char_ops), 4),
        "WER": round(compute_wer(word_ops), 4),
        "split_count": boundary_stats["split_count"],
        "merge_count": boundary_stats["merge_count"],
        "split_examples": boundary_stats["split_examples"],
        "merge_examples": boundary_stats["merge_examples"],
    }


def compute_corpus_cer(char_ops):
    errors = 0
    total = 0
    for o in char_ops:
        if o.op != "match":
            errors += 1
        if o.op in ("match", "substitution", "deletion"):
            total += 1
    return errors / max(total, 1)


def compute_corpus_wer(word_ops):
    errors = 0
    total = 0
    for o in word_ops:
        if o.op != "match":
            errors += 1
        if o.op in ("match", "substitution", "deletion"):
            total += 1
    return errors / max(total, 1)


def compute_corpus_boundary_stats(all_boundary_errors):
    split_count = sum(1 for e in all_boundary_errors if e["type"] == "split")
    merge_count = sum(1 for e in all_boundary_errors if e["type"] == "merge")
    return {
        "split_count": split_count,
        "merge_count": merge_count,
        "total_boundary_errors": split_count + merge_count
    }


def compute_alignment_quality(ocr_sents, gt_sents, matched_pairs):
    """Diagnostic alignment-quality metrics for a single document.

    The ratios are over ``len(matched_pairs)``; ``sentence_count_ratio`` is
    raw OCR sentences over GT sentences, useful for spotting docs where one
    side is missing pages.
    """
    sims = [SequenceMatcher(None, o, g).ratio() for o, g in matched_pairs]
    n = len(sims)
    sims_sorted = sorted(sims)
    median = sims_sorted[n // 2] if n else 0.0
    mean_sim = sum(sims) / n if n else 0.0

    def _frac_below(threshold: float) -> float:
        if not n:
            return 0.0
        return sum(1 for s in sims if s < threshold) / n

    return {
        "n_pairs": n,
        "n_ocr_sentences": len(ocr_sents),
        "n_gt_sentences": len(gt_sents),
        "sentence_count_ratio": (
            round(len(ocr_sents) / max(1, len(gt_sents)), 4)
        ),
        "mean_similarity": round(mean_sim, 4),
        "median_similarity": round(median, 4),
        "min_similarity": round(min(sims), 4) if n else 0.0,
        "max_similarity": round(max(sims), 4) if n else 0.0,
        "below_0_30_ratio": round(_frac_below(0.30), 4),
        "below_0_50_ratio": round(_frac_below(0.50), 4),
        "below_0_70_ratio": round(_frac_below(0.70), 4),
    }


def save_outputs(matched_pairs, char_ops, word_ops, boundary_errors, stats, out_dir, alignment_quality=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "matched_pairs.json").write_text(
        json.dumps([{"ocr": o, "gt": g} for o, g in matched_pairs], indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if alignment_quality is not None:
        (out_dir / "alignment_quality.json").write_text(
            json.dumps(alignment_quality, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    (out_dir / "char_ops.json").write_text(
        json.dumps([{"op": o.op, "ocr": o.ocr_char, "gt": o.gt_char} for o in char_ops], indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    (out_dir / "word_ops.json").write_text(
        json.dumps([{"op": o.op, "ocr": o.ocr_word, "gt": o.gt_word} for o in word_ops], indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    (out_dir / "word_boundary_errors.json").write_text(
        json.dumps(boundary_errors, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


if __name__ == "__main__":
    OUTPUT_ROOT = Path("phase1_output")
    raw_dir = "raw_ocr"
    corrected_dir = "corrected_ocr"

    SPLITS = {
        "train": [
            "dnevnik_po_mnogu_godini", "itar_pejo", "Pesni",
            "Prezir", "Samecot", "sina_pesna", "tajnopis",
            "Toj", "viktor_kupidon"
        ],
        "val": [
            "Забите на Ветрот - Томе Арсовски",
            "Провиденија"
        ],
        "test": [
            "Сите лица на смртта",
            "Современост 7"
        ]
    }

    book_to_split = {b: s for s, bs in SPLITS.items() for b in bs}
    BOOKS = []

    for file in os.listdir(raw_dir):
        if file.endswith("_ocr_raw.txt"):
            base = file[:-len("_ocr_raw.txt")]
            gt_file = base + "_corrected.txt"

            ocr_path = os.path.join(raw_dir, file)
            gt_path = os.path.join(corrected_dir, gt_file)

            if os.path.exists(gt_path):
                BOOKS.append({
                    "name": base,
                    "ocr": ocr_path,
                    "gt": gt_path
                })

    all_stats = []
    all_char_ops_global = []
    all_word_ops_global = []
    all_boundary_errors_global = []

    global_ocr_chars = 0
    global_gt_chars = 0
    global_ocr_tokens = 0
    global_gt_tokens = 0
    global_ocr_sentences = 0
    global_gt_sentences = 0

    for book in BOOKS:
        name = book["name"]
        split = book_to_split.get(name, "unassigned")

        print(f"[PROCESSING] {name} -> {split}")

        out_dir = OUTPUT_ROOT / split / name

        ocr_sents = extract_sentences(book["ocr"])
        gt_sents = extract_sentences(book["gt"])

        matched_pairs = align_sentences(ocr_sents, gt_sents)

        char_ops, word_ops, boundary_errors = align_all_pairs(matched_pairs)

        all_char_ops_global.extend(char_ops)
        all_word_ops_global.extend(word_ops)
        all_boundary_errors_global.extend(boundary_errors)

        stats = compute_statistics(
            ocr_sents, gt_sents,
            matched_pairs,
            char_ops, word_ops, boundary_errors
        )

        global_ocr_chars += stats["ocr_total_characters"]
        global_gt_chars += stats["gt_total_characters"]
        global_ocr_tokens += stats["ocr_total_tokens"]
        global_gt_tokens += stats["gt_total_tokens"]
        global_ocr_sentences += stats["ocr_total_sentences"]
        global_gt_sentences += stats["gt_total_sentences"]

        stats["book"] = name
        stats["split"] = split

        alignment_quality = compute_alignment_quality(ocr_sents, gt_sents, matched_pairs)
        print(
            f"  -> pairs={alignment_quality['n_pairs']} "
            f"mean_sim={alignment_quality['mean_similarity']:.3f} "
            f"<0.30={alignment_quality['below_0_30_ratio']:.3f} "
            f"<0.50={alignment_quality['below_0_50_ratio']:.3f}"
        )

        save_outputs(
            matched_pairs,
            char_ops,
            word_ops,
            boundary_errors,
            stats,
            out_dir,
            alignment_quality=alignment_quality,
        )

        all_stats.append(stats)

    corpus_boundary_stats = compute_corpus_boundary_stats(all_boundary_errors_global)

    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps({
            "books": len(all_stats),
            "ocr_total_characters": global_ocr_chars,
            "gt_total_characters": global_gt_chars,
            "ocr_total_tokens": global_ocr_tokens,
            "gt_total_tokens": global_gt_tokens,
            "ocr_total_sentences": global_ocr_sentences,
            "gt_total_sentences": global_gt_sentences,
            "corpus_CER": round(compute_corpus_cer(all_char_ops_global), 6),
            "corpus_WER": round(compute_corpus_wer(all_word_ops_global), 6),
            "corpus_split_count": corpus_boundary_stats["split_count"],
            "corpus_merge_count": corpus_boundary_stats["merge_count"],
            "corpus_total_boundary_errors": corpus_boundary_stats["total_boundary_errors"]
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )