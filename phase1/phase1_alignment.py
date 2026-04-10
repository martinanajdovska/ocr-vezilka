import re
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
from collections import Counter
import Levenshtein


PAGE_MARKER_PATTERNS = [
    re.compile(r"-{1,3}\s*[Pp]age\s+\d+\s*-{1,3}"),
    re.compile(r"={3,}\s*PAGE\s*\d+\s*={3,}"),
    re.compile(r"\[Page\s+\d+\]"),
    re.compile(r"^Page\s+\d+\s*$", re.MULTILINE),
    re.compile(r".{1,80}\s*-{3,}\s*$")
    ]


DASH_NOISE_PATTERNS = [
    re.compile(r"[—\-]{3,}"),
    re.compile(r"\s*[—\-]{2,}\s*"),
    re.compile(r"^\s*[—\-]{2,}\s*$", re.MULTILINE),
    re.compile(r"[—\-]{2,}(\s*[—\-]{2,})+"),
]

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

    cleaned = remove_dash_noise(cleaned)

    lines = cleaned.splitlines()

    filtered = []
    for i in range(len(lines)):
        if is_garbage_line(lines[i]):
            continue
        filtered.append(lines[i])

    text = re.sub(r"\s+", " ", " ".join(filtered)).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def build_features(paras):
    feats = []

    for p in paras:
        words = set(re.findall(r"\w+", p.lower()))
        shingles = Counter(p[i:i+5] for i in range(len(p)-4))
        feats.append((p, words, shingles))

    return feats


def fast_similarity(a, b):
    _, a_words, a_sh = a
    _, b_words, b_sh = b

    word = len(a_words & b_words) / (math.sqrt(len(a_words) * len(b_words)) + 1e-6)

    dot = sum(a_sh[k] * b_sh.get(k, 0) for k in a_sh)
    ma = math.sqrt(sum(v*v for v in a_sh.values()))
    mb = math.sqrt(sum(v*v for v in b_sh.values()))
    char = dot / (ma * mb + 1e-6)

    return 0.75 * char + 0.25 * word


def align_sentences(ocr_sents, gt_sents):
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

            # match
            if i < m and j < n:
                val = cur + sim(i, j)
                if val > dp[i+1][j+1]:
                    dp[i+1][j+1] = val
                    back[i+1][j+1] = (i, j)

            # skip OCR sentence
            if i < m:
                val = cur - 0.3
                if val > dp[i+1][j]:
                    dp[i+1][j] = val
                    back[i+1][j] = (i, j)

            # skip GT sentence
            if j < n:
                val = cur - 0.3
                if val > dp[i][j+1]:
                    dp[i][j+1] = val
                    back[i][j+1] = (i, j)

    if back[m][n] is None:
        return list(zip(ocr_sents, gt_sents)), list(zip(ocr_sents, gt_sents))

    # backtrack
    i, j = m, n
    pairs = []

    while (i, j) != (0, 0):
        pi, pj = back[i][j]

        if i > pi and j > pj:
            pairs.append((ocr_sents[i-1], gt_sents[j-1]))

        i, j = pi, pj

    pairs.reverse()

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

    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ow, gw):

        if tag == "equal":
            for k in range(i2 - i1):
                ops.append(WordEditOp("match", ow[i1+k], gw[j1+k]))

        elif tag == "replace":
            m = min(i2-i1, j2-j1)

            for k in range(m):
                ops.append(WordEditOp("substitution", ow[i1+k], gw[j1+k]))

            for k in range(m, i2-i1):
                ops.append(WordEditOp("insertion", ow[i1+k], ""))

            for k in range(m, j2-j1):
                ops.append(WordEditOp("deletion", "", gw[j1+k]))

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


def levenshtein_align(ocr: str, gt: str):
    ops = []

    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ocr, gt):

        if tag == "equal":
            for i in range(i2 - i1):
                ops.append(CharEditOp("match", ocr[i1+i], gt[j1+i]))

        elif tag == "replace":
            m = min(i2-i1, j2-j1)

            for i in range(m):
                ops.append(CharEditOp("substitution", ocr[i1+i], gt[j1+i]))

            for i in range(m, i2-i1):
                ops.append(CharEditOp("insertion", ocr[i1+i], ""))

            for i in range(m, j2-j1):
                ops.append(CharEditOp("deletion", "", gt[j1+i]))

        elif tag == "delete":
            for i in range(i1, i2):
                ops.append(CharEditOp("insertion", ocr[i], ""))

        elif tag == "insert":
            for i in range(j1, j2):
                ops.append(CharEditOp("deletion", "", gt[i]))

    return ops


def align_all_pairs(matched_pairs):
    all_word_ops = []
    all_char_ops = []

    for ocr, gt in matched_pairs:
        all_word_ops.extend(word_align(ocr, gt))
        all_char_ops.extend(levenshtein_align(ocr, gt))

    return all_char_ops, all_word_ops


def compute_cer(char_ops):
    errors = sum(1 for o in char_ops if o.op != "match")
    total = sum(1 for o in char_ops if o.op in ("match","substitution","deletion"))
    return errors / max(total, 1)


def compute_wer(word_ops):
    errors = sum(1 for o in word_ops if o.op != "match")
    total = sum(1 for o in word_ops if o.op in ("match","substitution","deletion"))
    return errors / max(total, 1)


def compute_text_statistics(text: str):
    total_characters = len(text)

    tokens = re.findall(r"\w+", text)
    total_tokens = len(tokens)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if s.strip()]
    total_sentences = len(sentences)

    avg_sentence_length = (
        total_tokens / total_sentences if total_sentences > 0 else 0
    )

    return {
        "total_characters": total_characters,
        "total_tokens": total_tokens,
        "total_sentences": total_sentences,
        "avg_sentence_length": round(avg_sentence_length, 4),
    }


def compute_statistics(ocr_sents, gt_sents, matched_pairs, char_ops, word_ops):

    ocr_text = " ".join(ocr_sents)
    gt_text = " ".join(gt_sents)

    ocr_stats = compute_text_statistics(ocr_text)
    gt_stats = compute_text_statistics(gt_text)

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


def save_outputs(matched_pairs, char_ops, word_ops, stats, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "matched_pairs.json").write_text(json.dumps(
        [{"ocr": o, "gt": g} for o, g in matched_pairs],
        indent=2, ensure_ascii=False
    ))

    (out_dir / "char_ops.json").write_text(json.dumps(
        [{"op": o.op, "ocr": o.ocr_char, "gt": o.gt_char} for o in char_ops],
        indent=2, ensure_ascii=False
    ))

    (out_dir / "word_ops.json").write_text(json.dumps(
        [{"op": o.op, "ocr": o.ocr_word, "gt": o.gt_word} for o in word_ops],
        indent=2, ensure_ascii=False
    ))

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))


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

        char_ops, word_ops = align_all_pairs(matched_pairs)

        all_char_ops_global.extend(char_ops)
        all_word_ops_global.extend(word_ops)

        stats = compute_statistics(
            ocr_sents, gt_sents,
            matched_pairs,
            char_ops, word_ops
        )
        global_ocr_chars += stats["ocr_total_characters"]
        global_gt_chars += stats["gt_total_characters"]

        global_ocr_tokens += stats["ocr_total_tokens"]
        global_gt_tokens += stats["gt_total_tokens"]

        global_ocr_sentences += stats["ocr_total_sentences"]
        global_gt_sentences += stats["gt_total_sentences"]

        stats["book"] = name
        stats["split"] = split

        save_outputs(
            matched_pairs,
            char_ops,
            word_ops,
            stats,
            out_dir
        )

        all_stats.append(stats)

    (OUTPUT_ROOT / "summary.json").write_text(json.dumps({
    "books": len(all_stats),

    "ocr_total_characters": global_ocr_chars,
    "gt_total_characters": global_gt_chars,
    "ocr_total_tokens": global_ocr_tokens,
    "gt_total_tokens": global_gt_tokens,
    "ocr_total_sentences": global_ocr_sentences,
    "gt_total_sentences": global_gt_sentences,

    "corpus_CER": round(compute_corpus_cer(all_char_ops_global), 6),
    "corpus_WER": round(compute_corpus_wer(all_word_ops_global), 6),

    }, indent=2, ensure_ascii=False))