import re
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple
from collections import Counter

PAGE_MARKER_PATTERNS = [
    re.compile(r"---\s*[Pp]age\s+\d+\s*---"),  # --- Page 1 ---
    re.compile(r"={3,}\s*PAGE\s*\d+\s*={3,}"),  # ===== PAGE 1 =====
    re.compile(r"-{3,}\s*PAGE\s*\d+\s*-{3,}"),  # --- PAGE 1 ---
    re.compile(r"\[Page\s+\d+\]"),  # [Page 1]
    re.compile(r"^Page\s+\d+\s*$", re.MULTILINE),  # Page 1
]


def detect_and_strip_page_markers(text: str) -> Tuple[str, bool]:
    """
    Detect whether any known page marker pattern exists in the text.
    If found, strip all markers and return (cleaned_text, True).
    If not found, return (text, False).
    """
    found = False
    for pattern in PAGE_MARKER_PATTERNS:
        if pattern.search(text):
            text = pattern.sub("", text)
            found = True

    if found:
        text = re.sub(r"^\s*[-=]{3,}.*[-=]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text, found


def is_garbage_line(line: str) -> bool:
    """
    Returns True for lines that are clearly OCR noise with no useful content.
    """
    s = line.strip()
    if not s:
        return True

    # no real word characters at all
    if not re.search(r"[\w\u0400-\u04FF]", s):
        return True

    # single character
    if len(s) <= 1:
        return True

    # pure page number
    if re.fullmatch(r"\d+", s):
        return True

    # very few word characters relative to line length
    word_chars = re.findall(r"[\w\u0400-\u04FF]", s)
    if len(word_chars) <= 2 and len(s) >= 4:
        return True
    return False


def extract_paragraphs(path: str) -> Tuple[List[str], bool]:
    """
    Load a file, auto-detect and strip page markers, filter garbage lines,
    then split into paragraphs on blank lines.

    Each paragraph is returned as a single string (lines joined with space)
    so that cross-page sentence fragments merge seamlessly.

    Returns (paragraphs, had_page_markers).
    """
    raw = Path(path).read_text(encoding="utf-8")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    text, had_markers = detect_and_strip_page_markers(text)

    raw_blocks = re.split(r"\n\s*\n", text)

    paragraphs = []
    for block in raw_blocks:
        lines = block.split("\n")
        clean_lines = [l.strip() for l in lines if not is_garbage_line(l)]
        if clean_lines:
            para = " ".join(clean_lines)
            para = re.sub(r"  +", " ", para).strip()
            if para:
                paragraphs.append(para)

    return paragraphs, had_markers


def char_ngrams(text: str, n: int = 3) -> Counter:
    t = text.lower()
    return Counter(t[i:i + n] for i in range(len(t) - n + 1))


def cosine_similarity(a: str, b: str, n: int = 3) -> float:
    na, nb = char_ngrams(a, n), char_ngrams(b, n)
    if not na or not nb:
        return 0.0

    dot = sum(na[k] * nb[k] for k in na if k in nb)
    magnitude_a = math.sqrt(sum(v ** 2 for v in na.values()))
    magnitude_b = math.sqrt(sum(v ** 2 for v in nb.values()))
    return dot / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0.0


def length_ratio_penalty(ocr_text: str, gt_text: str) -> float:
    """
    Returns a multiplier in (0, 1] that penalises pairs where one side is
    much longer than the other.

    The penalty is based on the ratio  R = longer / shorter:
      R <= 1.5  →  no penalty   (multiplier = 1.0)
      R == 2.5  →  ~50% penalty (multiplier ≈ 0.5)
      R == 4.0  →  ~80% penalty (multiplier ≈ 0.2)
      R >= 6.0  →  ~95% penalty (multiplier ≈ 0.05)

    Using an exponential decay so the penalty is smooth and gradual,
    not a hard cutoff that would discard genuine but uneven matches.
    """
    lo = min(len(ocr_text), len(gt_text))
    hi = max(len(ocr_text), len(gt_text))
    if lo == 0:
        return 0.0

    ratio = hi / lo
    if ratio <= 1.5:
        return 1.0

    return math.exp(-0.5 * (ratio - 1.5))


def many_to_many_align(
        ocr_paras: List[str],
        gt_paras: List[str],
        max_merge: int = 4,
        sim_threshold: float = 0.40,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Many-to-many paragraph alignment using dynamic programming (DP) with merge candidates
    and length-ratio penalty.

    At each step the DP considers merging up to `max_merge` consecutive
    paragraphs on the OCR side, the GT side, or both before scoring.
    The raw cosine similarity is multiplied by a length-ratio penalty
    so that pairs where one side is much longer than the other score
    poorly unless there is no better alignment available.

    This handles:
      - Paragraphs split across OCR page boundaries
      - GT paragraphs that correspond to multiple OCR fragments
      - Structurally mismatched pairs (e.g. one OCR fragment matched
        against an entire multi-paragraph GT block)
      - Content gaps (OCR-only or GT-only paragraphs)

    Returns:
      matched  — pairs above sim_threshold, both sides non-empty
      all_pairs — full backtrack including gaps, for coverage analysis
    """
    m, n = len(ocr_paras), len(gt_paras)
    NEG_INF = float("-inf")

    dp = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    back = [[(0, 0)] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    gap_penalty = 0.20  # cost for skipping an unmatched paragraph
    merge_penalty = 0.03  # extra cost per additional paragraph merged

    for i in range(m + 1):
        for j in range(n + 1):
            if dp[i][j] == NEG_INF:
                continue
            current = dp[i][j]

            for di in range(1, max_merge + 1):
                if i + di > m:
                    break
                ocr_merged = " ".join(ocr_paras[i:i + di])

                for dj in range(1, max_merge + 1):
                    if j + dj > n:
                        break
                    gt_merged = " ".join(gt_paras[j:j + dj])

                    sim = cosine_similarity(ocr_merged, gt_merged)
                    len_mult = length_ratio_penalty(ocr_merged, gt_merged)
                    merge_cost = (di - 1 + dj - 1) * merge_penalty
                    score = current + (sim * len_mult) - merge_cost

                    ni, nj = i + di, j + dj
                    if score > dp[ni][nj]:
                        dp[ni][nj] = score
                        back[ni][nj] = (di, dj)

                # skip di OCR paragraphs (gap on OCR side)
                skip = current - di * gap_penalty
                if skip > dp[i + di][j]:
                    dp[i + di][j] = skip
                    back[i + di][j] = (di, 0)

            # skip dj GT paragraphs (gap on GT side)
            for dj in range(1, max_merge + 1):
                if j + dj > n:
                    break
                skip = current - dj * gap_penalty
                if skip > dp[i][j + dj]:
                    dp[i][j + dj] = skip
                    back[i][j + dj] = (0, dj)

    raw_pairs = []
    i, j = m, n
    while i > 0 or j > 0:
        di, dj = back[i][j]
        if di == 0 and dj == 0:
            break

        ocr_text = " ".join(ocr_paras[i - di:i]) if di > 0 else ""
        gt_text = " ".join(gt_paras[j - dj:j]) if dj > 0 else ""
        raw_pairs.append((ocr_text, gt_text))
        i -= di
        j -= dj

    raw_pairs.reverse()

    # keep only pairs where both sides exist and similarity meets threshold
    matched = [
        (o, g) for o, g in raw_pairs
        if o and g and cosine_similarity(o, g) >= sim_threshold
    ]

    return matched, raw_pairs


@dataclass
class CharEditOp:
    op: str  # 'match' | 'substitution' | 'insertion' | 'deletion'
    ocr_char: str
    gt_char: str
    ocr_pos: int
    gt_pos: int


@dataclass
class WordEditOp:
    op: str
    ocr_word: str
    gt_word: str


def levenshtein_align(ocr: str, gt: str) -> List[CharEditOp]:
    m, n = len(ocr), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ocr[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ocr[i - 1] == gt[j - 1]:
            ops.append(CharEditOp("match", ocr[i - 1], gt[j - 1], i - 1, j - 1))
            i -= 1;
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(CharEditOp("substitution", ocr[i - 1], gt[j - 1], i - 1, j - 1))
            i -= 1;
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(CharEditOp("deletion", ocr[i - 1], "", i - 1, j))
            i -= 1
        else:
            ops.append(CharEditOp("insertion", "", gt[j - 1], i, j - 1))
            j -= 1

    ops.reverse()
    return ops


def word_align(ocr: str, gt: str) -> List[WordEditOp]:
    ow, gw = ocr.split(), gt.split()
    m, n = len(ow), len(gw)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ow[i - 1] == gw[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ow[i - 1] == gw[j - 1]:
            ops.append(WordEditOp("match", ow[i - 1], gw[j - 1]))
            i -= 1;
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(WordEditOp("substitution", ow[i - 1], gw[j - 1]))
            i -= 1;
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(WordEditOp("deletion", ow[i - 1], ""))
            i -= 1
        else:
            ops.append(WordEditOp("insertion", "", gw[j - 1]))
            j -= 1

    ops.reverse()
    return ops


def align_all_pairs(matched_pairs):
    all_char_ops, all_word_ops = [], []
    for idx, (ocr_para, gt_para) in enumerate(matched_pairs):
        if len(ocr_para) * len(gt_para) > 4_000_000:
            print(f"    WARNING: pair {idx} too large, skipping char alignment")
            continue

        all_char_ops.extend(levenshtein_align(ocr_para, gt_para))
        all_word_ops.extend(word_align(ocr_para, gt_para))
    return all_char_ops, all_word_ops


def compute_cer(char_ops):
    errors = sum(1 for op in char_ops if op.op != "match")
    gt_len = sum(1 for op in char_ops if op.op in ("match", "substitution", "insertion"))
    return errors / max(gt_len, 1)


def compute_wer(word_ops):
    errors = sum(1 for op in word_ops if op.op != "match")
    gt_len = sum(1 for op in word_ops if op.op in ("match", "substitution", "insertion"))
    return errors / max(gt_len, 1)


def compute_statistics(ocr_paras, gt_paras, matched_pairs, all_pairs,
                       char_ops, word_ops, ocr_had_markers, gt_had_markers):
    ocr_text = " ".join(o for o, g in matched_pairs)
    gt_text = " ".join(g for o, g in matched_pairs)
    ocr_full = " ".join(ocr_paras)
    gt_full = " ".join(gt_paras)

    n_sub = sum(1 for op in char_ops if op.op == "substitution")
    n_ins = sum(1 for op in char_ops if op.op == "insertion")
    n_del = sum(1 for op in char_ops if op.op == "deletion")
    n_err = n_sub + n_ins + n_del

    total_characters = len(ocr_text) + len(gt_text)
    total_tokens = len(ocr_text.split()) + len(gt_text.split())
    total_sentences = len(matched_pairs)
    avg_sentence_length = total_characters / max(total_sentences, 1)

    return {
        "ocr_had_page_markers": ocr_had_markers,
        "gt_had_page_markers": gt_had_markers,
        "ocr_total_paragraphs": len(ocr_paras),
        "gt_total_paragraphs": len(gt_paras),
        "matched_pairs": len(matched_pairs),
        "total_alignment_steps": len(all_pairs),
        "ocr_characters_matched": len(ocr_text),
        "gt_characters_matched": len(gt_text),
        "ocr_tokens_matched": len(ocr_text.split()),
        "gt_tokens_matched": len(gt_text.split()),
        "ocr_coverage_pct": round(len(ocr_text) / max(len(ocr_full), 1) * 100, 1),
        "gt_coverage_pct": round(len(gt_text) / max(len(gt_full), 1) * 100, 1),
        "baseline_CER": round(compute_cer(char_ops), 4),
        "baseline_WER": round(compute_wer(word_ops), 4),
        "total_char_errors": n_err,
        "substitutions": n_sub,
        "insertions": n_ins,
        "deletions": n_del,
        "substitution_pct": round(n_sub / max(n_err, 1) * 100, 2),
        "insertion_pct": round(n_ins / max(n_err, 1) * 100, 2),
        "deletion_pct": round(n_del / max(n_err, 1) * 100, 2),
        "total_characters": total_characters,
        "total_tokens": total_tokens,
        "total_sentences": total_sentences,
        "average_sentence_length": round(avg_sentence_length, 2),
    }


def save_outputs(matched_pairs, all_pairs, char_ops, word_ops, stats, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "matched_pairs.json").write_text(
        json.dumps([{"ocr": o, "gt": g, "sim": round(cosine_similarity(o, g), 3)}
                    for o, g in matched_pairs],
                   ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "full_alignment.json").write_text(
        json.dumps([{"ocr": o, "gt": g,
                     "sim": round(cosine_similarity(o, g), 3) if o and g else 0}
                    for o, g in all_pairs],
                   ensure_ascii=False, indent=2), encoding="utf-8")

    char_log = [asdict(op) for op in char_ops if op.op != "match"]
    (out_dir / "char_alignment_log.json").write_text(
        json.dumps(char_log, ensure_ascii=False, indent=2), encoding="utf-8")

    word_log = [asdict(op) for op in word_ops if op.op != "match"]
    (out_dir / "word_alignment_log.json").write_text(
        json.dumps(word_log, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "phase1_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "ocr_cleaned.txt").write_text("\n".join(o for o, g in matched_pairs), encoding="utf-8")
    (out_dir / "gt_cleaned.txt").write_text("\n".join(g for o, g in matched_pairs), encoding="utf-8")


def run_phase1(ocr_path: str, gt_path: str, out_dir: str):
    out = Path(out_dir)

    print("Loading and extracting paragraphs...")
    ocr_paras, ocr_had_markers = extract_paragraphs(ocr_path)
    gt_paras, gt_had_markers = extract_paragraphs(gt_path)

    print("\nMany-to-many paragraph alignment")
    matched_pairs, all_pairs = many_to_many_align(
        ocr_paras, gt_paras, max_merge=6, sim_threshold=0.35)

    print("\nFixing spillovers...")
    matched_pairs = fix_spillovers(matched_pairs)

    print("\nCharacter-level Levenshtein alignment...")
    char_ops, word_ops = align_all_pairs(matched_pairs)

    stats = compute_statistics(ocr_paras, gt_paras, matched_pairs, all_pairs,
                               char_ops, word_ops, ocr_had_markers, gt_had_markers)

    save_outputs(matched_pairs, all_pairs, char_ops, word_ops, stats, out)
    return matched_pairs, char_ops, word_ops, stats


def fix_spillovers(matched_pairs):
    """
    Post-process matched_pairs to merge pairs where there is spillover 
    (last five words of pair i GT are similar to first five words of pair i+1 OCR).
    """
    fixed = []
    i = 0
    while i < len(matched_pairs):
        ocr, gt = matched_pairs[i]
        if i + 1 < len(matched_pairs):
            next_ocr, next_gt = matched_pairs[i + 1]

            gt_words = gt.split()
            last_gt_words = gt_words[-5:] if len(gt_words) >= 5 else gt_words

            ocr_words = next_ocr.split()
            first_ocr_words = ocr_words[:5] if len(ocr_words) >= 5 else ocr_words

            gt_phrase = " ".join(last_gt_words).lower()
            ocr_phrase = " ".join(first_ocr_words).lower()

            if gt_phrase and ocr_phrase:
                sim = cosine_similarity(gt_phrase, ocr_phrase)
                if sim >= 0.8:
                    merged_ocr = ocr + " " + next_ocr
                    merged_gt = gt + " " + next_gt
                    fixed.append((merged_ocr, merged_gt))
                    i += 2
                    continue
        fixed.append((ocr, gt))
        i += 1
    return fixed


def load_book_data(book_out_dir: Path):
    """
    Load existing alignment data from a book's output directory.
    Returns (matched_pairs, char_ops, word_ops, stats, all_pairs) or None if incomplete.
    """
    matched_pairs_path = book_out_dir / "matched_pairs.json"
    if not matched_pairs_path.exists():
        return None
    matched_pairs_data = json.loads(matched_pairs_path.read_text(encoding="utf-8"))
    matched_pairs = [(p["ocr"], p["gt"]) for p in matched_pairs_data]

    char_log_path = book_out_dir / "char_alignment_log.json"
    char_ops = []
    if char_log_path.exists():
        char_log = json.loads(char_log_path.read_text(encoding="utf-8"))
        char_ops = [CharEditOp(**op) for op in char_log]

    word_log_path = book_out_dir / "word_alignment_log.json"
    word_ops = []
    if word_log_path.exists():
        word_log = json.loads(word_log_path.read_text(encoding="utf-8"))
        word_ops = [WordEditOp(**op) for op in word_log]

    stats_path = book_out_dir / "phase1_statistics.json"
    stats = {}
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

    full_alignment_path = book_out_dir / "full_alignment.json"
    all_pairs = []
    if full_alignment_path.exists():
        all_pairs_data = json.loads(full_alignment_path.read_text(encoding="utf-8"))
        all_pairs = [(p["ocr"], p["gt"]) for p in all_pairs_data]

    return matched_pairs, char_ops, word_ops, stats, all_pairs


if __name__ == "__main__":
    # Assumes files are named as: <name>_ocr_raw.txt in raw_ocr/
    # and <name>_corrected.txt in corrected_ocr/
    # Output folders will be named <name> under phase1_output/

    OUTPUT_ROOT = "phase1_output"

    raw_dir = "raw_ocr"
    corrected_dir = "corrected_ocr"

    BOOKS = []
    for file in os.listdir(raw_dir):
        if file.endswith("_ocr_raw.txt"):
            base = file[:-len("_ocr_raw.txt")]
            corrected_file = base + "_corrected.txt"
            ocr_path = os.path.join(raw_dir, file)
            gt_path = os.path.join(corrected_dir, corrected_file)
            if os.path.exists(gt_path):
                BOOKS.append({
                    "name": base,
                    "ocr": ocr_path,
                    "gt": gt_path,
                })

    print("Detected books:")
    for book in BOOKS:
        print(f"  {book['name']}: OCR={book['ocr']}, GT={book['gt']}")

    all_stats = {}

    for book in BOOKS:
        name = book["name"]
        out_dir = Path(OUTPUT_ROOT) / name

        if out_dir.exists() and (out_dir / "matched_pairs.json").exists():
            data = load_book_data(out_dir)
            if data:
                _, _, _, stats, _ = data
            else:
                _, _, _, stats = run_phase1(book["ocr"], book["gt"], str(out_dir))
        else:
            _, _, _, stats = run_phase1(book["ocr"], book["gt"], str(out_dir))

        all_stats[name] = stats

    # combined summary for all books
    summary_path = Path(OUTPUT_ROOT) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 9-2-2 train-val-test split based on book names
    SPLITS = {
        "train": ["dnevnik_po_mnogu_godini", "itar_pejo", "Pesni", "Prezir", "Samecot", "sina_pesna", "tajnopis", "Toj",
                  "viktor_kupidon"],
        "val": ["Забите на Ветрот - Томе Арсовски", "Провиденија"],
        "test": ["Сите лица на смртта", "Современост 7"]
    }

    for split_name, book_names in SPLITS.items():
        combined_matched = []
        combined_char_ops = []
        combined_word_ops = []
        combined_all_pairs = []
        split_stats = {}

        for book_name in book_names:
            book_out_dir = Path(OUTPUT_ROOT) / book_name
            if not book_out_dir.exists():
                continue

            data = load_book_data(book_out_dir)
            if not data:
                continue

            matched_pairs, char_ops, word_ops, stats, all_pairs = data
            combined_matched.extend(matched_pairs)
            combined_char_ops.extend(char_ops)
            combined_word_ops.extend(word_ops)
            combined_all_pairs.extend(all_pairs)
            split_stats[book_name] = stats

        total_chars = sum(s.get("total_characters", 0) for s in split_stats.values())
        total_tokens = sum(s.get("total_tokens", 0) for s in split_stats.values())
        total_sentences = sum(s.get("total_sentences", 0) for s in split_stats.values())
        avg_sent_len = total_chars / max(total_sentences, 1)

        cer = compute_cer(combined_char_ops)
        wer = compute_wer(combined_word_ops)

        split_overall_stats = {
            "total_characters": total_chars,
            "total_tokens": total_tokens,
            "total_sentences": total_sentences,
            "average_sentence_length": round(avg_sent_len, 2),
            "baseline_CER": round(cer, 4),
            "baseline_WER": round(wer, 4),
            "book_stats": split_stats
        }

        split_out_dir = Path(OUTPUT_ROOT) / split_name
        save_outputs(combined_matched, combined_all_pairs, combined_char_ops, combined_word_ops, split_overall_stats,
                     split_out_dir)
