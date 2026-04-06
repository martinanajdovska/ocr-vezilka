import re
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple
from collections import Counter
import numpy as np
import Levenshtein


PAGE_MARKER_PATTERNS = [
    re.compile(r"-{1,3}\s*[Pp]age\s+\d+\s*-{1,3}"),  # --- Page 1 ---
    re.compile(r"-{1,3}[Pp]age \d+-{1,3}"),  # -Page 1-
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



def extract_paragraphs(path: str):
    raw = Path(path).read_text(encoding="utf-8")
    cleaned_raw, had_markers = detect_and_strip_page_markers(raw)

    lines = cleaned_raw.splitlines()
    clean_lines = [line for line in lines if not is_garbage_line(line)]
    
    text = " ".join(clean_lines)
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = re.split(r'(?<=[.!?]) +', text)
    
    paragraphs = []
    current_unit = []
    
    for chunk in chunks:
        if len(chunk) > 1000:
            if current_unit:
                paragraphs.append(" ".join(current_unit))
                current_unit = []
            
            for i in range(0, len(chunk), 500):
                paragraphs.append(chunk[i:i+500])
            continue

        current_unit.append(chunk)
        
        if len(" ".join(current_unit)) > 300:
            paragraphs.append(" ".join(current_unit))
            current_unit = []
            
    if current_unit:
        paragraphs.append(" ".join(current_unit))
        
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


def many_to_many_align(ocr_paras, gt_paras, max_merge=6, sim_threshold=0.35):
    print(" Aligning paragraphs...")    

    all_text = " ".join(ocr_paras + gt_paras).lower()
    vocab = {gram: i for i, gram in enumerate(set(all_text[i:i+3] for i in range(len(all_text)-2)))}
    v_size = len(vocab)

    def get_vectors(paras):
        vectors = []
        for p in paras:
            vec = np.zeros(v_size, dtype=np.float32)
            for i in range(len(p) - 2):
                gram = p[i:i+3].lower()
                if gram in vocab:
                    vec[vocab[gram]] += 1
            vectors.append(vec)
        return vectors

    ocr_vecs = get_vectors(ocr_paras)
    gt_vecs = get_vectors(gt_paras)

    def precompute_windows(vecs, max_m):
        windows = {}
        for i in range(len(vecs)):
            current_vec = np.zeros(v_size, dtype=np.float32)
            for size in range(1, max_m + 1):
                if i + size > len(vecs): break
                current_vec += vecs[i + size - 1]
                
                mag = np.linalg.norm(current_vec)
                norm_vec = current_vec / mag if mag > 0 else current_vec
                windows[(i, size)] = (norm_vec, sum(len(ocr_paras[k]) if vecs is ocr_vecs else len(gt_paras[k]) for k in range(i, i+size)))
        return windows

    ocr_win = precompute_windows(ocr_vecs, max_merge)
    gt_win = precompute_windows(gt_vecs, max_merge)

    m, n = len(ocr_paras), len(gt_paras)
    dp = np.full((m + 1, n + 1), -np.inf, dtype=np.float32)
    back = [[(0, 0)] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0

    gap_penalty = 0.20
    
    for i in range(m + 1):
        diag_j = int(i * n / m) if m > 0 else 0
        j_range = range(max(0, diag_j - 150), min(n + 1, diag_j + 150))
        
        for j in j_range:
            curr_score = dp[i, j]
            if curr_score == -np.inf: 
                continue
            
            for di in range(1, max_merge + 1):
                if i + di > m: break
                vec_a, len_a = ocr_win[(i, di)]
                
                for dj in range(1, max_merge + 1):
                    if j + dj > n: break
                    vec_b, len_b = gt_win[(j, dj)]

                    sim = np.dot(vec_a, vec_b)
                    
                    ratio = max(len_a, len_b) / max(min(len_a, len_b), 1)
                    penalty = math.exp(-0.5 * (ratio - 1.5)) if ratio > 1.5 else 1.0
                    
                    score = curr_score + (sim * penalty) - ((di + dj - 2) * 0.02)
                    
                    if score > dp[i + di, j + dj]:
                        dp[i + di, j + dj] = score
                        back[i + di][j + dj] = (di, dj)

            for di in range(1, max_merge + 1):
                if i + di <= m:
                    skip_score = curr_score - di * gap_penalty
                    if skip_score > dp[i + di][j]:
                        dp[i + di][j] = skip_score
                        back[i + di][j] = (di, 0)
            
            for dj in range(1, max_merge + 1):
                if j + dj <= n:
                    skip_score = curr_score - dj * gap_penalty
                    if skip_score > dp[i][j + dj]:
                        dp[i][j + dj] = skip_score
                        back[i][j + dj] = (0, dj)

    raw_pairs = []
    curr_i, curr_j = m, n
    while curr_i > 0 or curr_j > 0:
        di, dj = back[curr_i][curr_j]
        if di == 0 and dj == 0:
            break

        ocr_block = " ".join(ocr_paras[curr_i - di : curr_i]) if di > 0 else ""
        gt_block = " ".join(gt_paras[curr_j - dj : curr_j]) if dj > 0 else ""
        raw_pairs.append((ocr_block, gt_block))
        curr_i -= di
        curr_j -= dj

    raw_pairs.reverse()

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
    ops = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ocr, gt):
        if tag == 'equal':
            for idx in range(i2 - i1):
                ops.append(CharEditOp("match", ocr[i1 + idx], gt[j1 + idx], i1 + idx, j1 + idx))
        elif tag == 'replace':

            for idx in range(min(i2 - i1, j2 - j1)):
                ops.append(CharEditOp("substitution", ocr[i1 + idx], gt[j1 + idx], i1 + idx, j1 + idx))
            if (i2 - i1) > (j2 - j1): 
                for idx in range(j2 - j1, i2 - i1):
                    ops.append(CharEditOp("deletion", ocr[i1 + idx], "", i1 + idx, j2))
            elif (j2 - j1) > (i2 - i1): 
                for idx in range(i2 - i1, j2 - j1):
                    ops.append(CharEditOp("insertion", "", gt[j1 + idx], i2, j1 + idx))
        elif tag == 'delete':
            for idx in range(i1, i2):
                ops.append(CharEditOp("deletion", ocr[idx], "", idx, j1))
        elif tag == 'insert':
            for idx in range(j1, j2):
                ops.append(CharEditOp("insertion", "", gt[idx], i1, idx))
    return ops


def word_align(ocr: str, gt: str) -> List[WordEditOp]:
    ow, gw = ocr.split(), gt.split()
    
    vocab = {word: chr(i + 32) for i, word in enumerate(set(ow + gw))}
    
    ocr_encoded = "".join(vocab[w] for w in ow)
    gt_encoded = "".join(vocab[w] for w in gw)
    
    ops = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ocr_encoded, gt_encoded):
        if tag == 'equal':
            for idx in range(i2 - i1):
                ops.append(WordEditOp("match", ow[i1 + idx], gw[j1 + idx]))
        elif tag == 'replace':
            for idx in range(min(i2 - i1, j2 - j1)):
                ops.append(WordEditOp("substitution", ow[i1 + idx], gw[j1 + idx]))
            if (i2 - i1) > (j2 - j1):
                for idx in range(j2 - j1, i2 - i1):
                    ops.append(WordEditOp("deletion", ow[i1 + idx], ""))
            elif (j2 - j1) > (i2 - i1):
                for idx in range(i2 - i1, j2 - j1):
                    ops.append(WordEditOp("insertion", "", gw[j1 + idx]))
        elif tag == 'delete':
            for idx in range(i1, i2):
                ops.append(WordEditOp("deletion", ow[idx], ""))
        elif tag == "insert":
            for idx in range(j1, j2):
                ops.append(WordEditOp("insertion", "", gw[idx]))
    return ops


def align_all_pairs(matched_pairs):
    all_char_ops, all_word_ops = [], []
    for idx, (ocr_para, gt_para) in enumerate(matched_pairs):
        if len(ocr_para) * len(gt_para) > 100_000_000:
            continue

        all_char_ops.extend(levenshtein_align(ocr_para, gt_para)) 
        all_word_ops.extend(word_align(ocr_para, gt_para))
        
    return all_char_ops, all_word_ops


def compute_cer(char_ops):
    errors = sum(1 for op in char_ops if op.op in ("substitution", "deletion", "insertion"))
    gt_len = sum(1 for op in char_ops if op.op in ("match", "substitution", "deletion"))
    return errors / max(gt_len, 1)


def compute_wer(word_ops):
    errors = sum(1 for op in word_ops if op.op != "match")
    gt_len = sum(1 for op in word_ops if op.op in ("match", "substitution", "deletion"))
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

    gt_characters = sum(1 for op in char_ops if op.op in ("match", "substitution", "deletion"))
    gt_tokens = sum(1 for op in word_ops if op.op in ("match", "substitution", "deletion"))
    
    total_sentences = len(matched_pairs)
    avg_sentence_length = gt_characters / max(total_sentences, 1)

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
        
        "total_characters": gt_characters,
        "total_tokens": gt_tokens,
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
    if not matched_pairs:
        return []
        
    fixed = []
    i = 0
    while i < len(matched_pairs):
        current_ocr, current_gt = matched_pairs[i]
        
        while i + 1 < len(matched_pairs):
            next_ocr, next_gt = matched_pairs[i + 1]

            gt_words = current_gt.split()
            last_gt_words = gt_words[-5:] if len(gt_words) >= 5 else gt_words
            ocr_words = next_ocr.split()
            first_ocr_words = ocr_words[:5] if len(ocr_words) >= 5 else ocr_words

            gt_phrase = " ".join(last_gt_words).lower()
            ocr_phrase = " ".join(first_ocr_words).lower()

            if gt_phrase and ocr_phrase and cosine_similarity(gt_phrase, ocr_phrase) >= 0.8:
                current_ocr += " " + next_ocr
                current_gt += " " + next_gt
                i += 1  
            else:
                break
        
        fixed.append((current_ocr, current_gt))
        i += 1
    return fixed


def load_book_data(book_out_dir: Path):
    """
    Load existing alignment data from a book's output directory.
    """
    matched_pairs_path = book_out_dir / "matched_pairs.json"
    if not matched_pairs_path.exists():
        return None
    matched_pairs_data = json.loads(matched_pairs_path.read_text(encoding="utf-8"))
    matched_pairs = [(p["ocr"], p["gt"]) for p in matched_pairs_data]

    char_ops, word_ops = align_all_pairs(matched_pairs)

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


def calculate_total_statistics(all_split_data, output_root):
    """
    Calculate global statistics across all splits (train, val, test).
    """
    total_chars = 0
    total_tokens = 0
    total_sentences = 0
    total_errors_char = 0
    total_ref_chars = 0
    total_errors_word = 0
    total_ref_words = 0
    
    for split_item in all_split_data:
        stats = split_item['stats']
        total_chars += stats.get('total_characters', 0)
        total_tokens += stats.get('total_tokens', 0)
        total_sentences += stats.get('total_sentences', 0)
        
        c_ops = split_item['char_ops']
        w_ops = split_item['word_ops']
        
        total_errors_char += sum(1 for op in c_ops if op.op in ("substitution", "deletion", "insertion"))
        total_ref_chars += sum(1 for op in c_ops if op.op in ("match", "substitution", "deletion"))
        
        total_errors_word += sum(1 for op in w_ops if op.op != "match")
        total_ref_words += sum(1 for op in w_ops if op.op in ("match", "substitution", "deletion"))

    total_stats = {
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "total_sentences": total_sentences,
        "average_sentence_length": round(total_chars / max(total_sentences, 1), 2),
        "total_CER": round(total_errors_char / max(total_ref_chars, 1), 4),
        "total_WER": round(total_errors_word / max(total_ref_words, 1), 4),
        "total_char_errors": total_errors_char,
        "total_word_errors": total_errors_word,
        "total_ref_chars": total_ref_chars,
        "total_ref_words": total_ref_words
    }

    total_path = Path(output_root) / "total_statistics.json"
    total_path.write_text(json.dumps(total_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Global statistics saved to {total_path}")
    
    return total_stats


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

    # 9-2-2 train-val-test split based on book names
    SPLITS = {
        "train": ["dnevnik_po_mnogu_godini", "itar_pejo", "Pesni", "Prezir", "Samecot", "sina_pesna", "tajnopis", "Toj",
                  "viktor_kupidon"],
        "val": ["Забите на Ветрот - Томе Арсовски", "Провиденија"],
        "test": ["Сите лица на смртта", "Современост 7"]
    }

    book_to_split = {book: split for split, books in SPLITS.items() for book in books}

    for book in BOOKS:
        name = book["name"]

        split_name = book_to_split.get(name, "")
        out_dir = Path(OUTPUT_ROOT) / split_name / name

        if out_dir.exists() and (out_dir / "matched_pairs.json").exists():
            data = load_book_data(out_dir)
            if data:
                _, _, _, stats, _ = data
            else:
                _, _, _, stats = run_phase1(book["ocr"], book["gt"], str(out_dir))
        else:
            _, _, _, stats = run_phase1(book["ocr"], book["gt"], str(out_dir))

        all_stats[name] = stats

    summary_path = Path(OUTPUT_ROOT) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    all_split_results = []

    for split_name, book_names in SPLITS.items():
        combined_matched = []
        combined_char_ops = []
        combined_word_ops = []
        combined_all_pairs = []
        split_stats = {}

        for book_name in book_names:
            book_out_dir = Path(OUTPUT_ROOT) / split_name / book_name            
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

        all_split_results.append({
            "char_ops": combined_char_ops,
            "word_ops": combined_word_ops,
            "stats": split_overall_stats
        })

    if all_split_results:
        calculate_total_statistics(all_split_results, OUTPUT_ROOT)
