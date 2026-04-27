import json
import math
import random
import re
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[\.!\?…])\s+")
_PARAGRAPH_BOUNDARY_RE = re.compile(r"\n\s*\n")


def _split_paragraphs(text: str) -> List[str]:
    """Paragraph-segment using one or more blank lines as the boundary."""
    if not text:
        return []
    parts = _PARAGRAPH_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _split_sentences(text: str) -> List[str]:
    """Sentence-segment a single paragraph; whitespace-only sentences dropped."""
    if not text:
        return []
    flat = re.sub(r"\s+", " ", text.strip())
    if not flat:
        return []
    parts = _SENTENCE_BOUNDARY_RE.split(flat)
    return [p.strip() for p in parts if p and p.strip()]


class Phase3SyntheticNoise:
    def __init__(
        self,
        clean_dir="../phase1/corrected_ocr",
        raw_dir="../phase1/raw_ocr",
        phase2_dir="../phase2/phase2_output",
        output_dir="../phase3/phase3_output",
        seed=42,
        calibration_doc_ids: Optional[Iterable[str]] = None,
    ):
        self.clean_dir = Path(clean_dir)
        self.raw_dir = Path(raw_dir)
        self.phase2_dir = Path(phase2_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random = random.Random(seed)
        self.calibration_doc_ids = (
            set(calibration_doc_ids) if calibration_doc_ids is not None else None
        )

        self.error_dist = None
        self.error_conf_counts = None
        self.error_conf_probs = None
        self.word_boundary_stats = None
        self.phase1_summary = {}
        self.p_del = 0.0
        self.p_sub = 0.0
        self.p_ins = 0.0
        self._alphabet = []
        self.p_split_word = 0.0
        self.p_merge_gap = 0.0
        self.p_struct_diacritic = 0.0
        self.p_struct_homoglyph = 0.0
        self.p_struct_punct = 0.0

        self.diacritic_map = {
            "ѓ": "г", "Ѓ": "Г",
            "ќ": "к", "Ќ": "К",
            "љ": "л", "Љ": "Л",
            "њ": "н", "Њ": "Н",
        }

        self.homoglyph_map = {
            "А": "A", "а": "a",
            "В": "B",
            "Е": "E", "е": "e",
            "К": "K",
            "М": "M",
            "Н": "H",
            "О": "O", "о": "o",
            "Р": "P",
            "С": "C", "с": "c",
            "Т": "T",
            "Х": "X",
            "У": "Y",
            "Ѕ": "S", "ѕ": "s",
        }

        self.punct_shift_map = {
            ".": ",",
            ",": ".",
            ":": ";",
            ";": ":",
            "„": "\"",
            "“": "\"",
            "\"": "„",
            "'": "\"",
            "—": "-",
            "-": "—",
        }

    def load_phase2_statistics(self):
        with open(self.phase2_dir / "error_distribution.json", "r", encoding="utf-8") as f:
            self.error_dist = json.load(f)

        self.error_conf_counts = pd.read_csv(self.phase2_dir / "error_confusion_counts.csv", index_col=0)
        self.error_conf_probs = pd.read_csv(self.phase2_dir / "error_confusion_probs.csv", index_col=0)

        wb_path = self.phase2_dir / "word_boundary_stats.json"
        if wb_path.exists():
            with open(wb_path, "r", encoding="utf-8") as f:
                self.word_boundary_stats = json.load(f)
        else:
            self.word_boundary_stats = {"split_rate_pct": 0.0, "merge_rate_pct": 0.0}

        phase1_summary_path = self.clean_dir.parent / "phase1_output" / "summary.json"
        if phase1_summary_path.exists():
            with open(phase1_summary_path, "r", encoding="utf-8") as f:
                self.phase1_summary = json.load(f)
        else:
            self.phase1_summary = {}


        m = float(self.error_dist.get("match_pct", 0.0)) / 100.0
        s = float(self.error_dist.get("substitution_pct", 0.0)) / 100.0
        d = float(self.error_dist.get("deletion_pct", 0.0)) / 100.0
        ins = float(self.error_dist.get("insertion_pct", 0.0)) / 100.0
        ref_denom = m + s + d
        if ref_denom <= 0.0:
            ref_denom = 1.0
        self.p_del = d / ref_denom
        self.p_sub = s / ref_denom
        self.p_ins = ins / ref_denom

        cols = self.error_conf_counts.columns
        self._alphabet = [
            c for c in cols
            if isinstance(c, str) and len(c) == 1 and c not in ["<DEL>", "<INS>"]
        ]

    def _calibrate_structure_noise_rates(self, clean_texts: dict):
        """Word split/merge rates vs GT tokens; structure char ops vs clean corpus counts.
        """
        split_ct = int(self.error_dist.get("split_count", 0))
        merge_ct = int(self.error_dist.get("merge_count", 0))

        if self.calibration_doc_ids is not None:
            calibration_texts = {
                d: t for d, t in clean_texts.items() if d in self.calibration_doc_ids
            }
            gt_tok = max(sum(len(t.split()) for t in calibration_texts.values()), 1)
        else:
            gt_tok = int(self.phase1_summary.get("gt_total_tokens", 0))
            if gt_tok <= 0:
                gt_tok = max(sum(len(t.split()) for t in clean_texts.values()), 1)

        self.p_split_word = min(1.0, split_ct / gt_tok)
        self.p_merge_gap = min(1.0, merge_ct / max(gt_tok - 1, 1))

        p2_sum_path = self.phase2_dir / "phase2_summary.json"
        if p2_sum_path.exists():
            with open(p2_sum_path, "r", encoding="utf-8") as f:
                p2s = json.load(f)
            dtot = float(p2s.get("diacritic_confusions_total", 0))
            htot = float(p2s.get("homoglyph_confusions_total", 0))
            ptot = float(p2s.get("punctuation_confusions_total", 0))
        else:
            dtot = htot = ptot = 0.0

        diac_hits = 0
        homo_hits = 0
        punct_hits = 0
        for t in clean_texts.values():
            for ch in t:
                if ch in self.diacritic_map:
                    diac_hits += 1
                if ch in self.homoglyph_map:
                    homo_hits += 1
                if ch in self.punct_shift_map:
                    punct_hits += 1

        def cond_rate(total_events: float, eligible_hits: int) -> float:
            if eligible_hits <= 0 or total_events <= 0:
                return 0.0
            return min(1.0, total_events / float(eligible_hits))

        if self.calibration_doc_ids is not None:
            calibration_texts = {
                d: t for d, t in clean_texts.items() if d in self.calibration_doc_ids
            }
            calibration_corpus = "".join(calibration_texts.values())
            cal_diac_hits = sum(1 for ch in calibration_corpus if ch in self.diacritic_map)
            cal_homo_hits = sum(1 for ch in calibration_corpus if ch in self.homoglyph_map)
            cal_punct_hits = sum(1 for ch in calibration_corpus if ch in self.punct_shift_map)
            self.p_struct_diacritic = cond_rate(dtot, cal_diac_hits)
            self.p_struct_homoglyph = cond_rate(htot, cal_homo_hits)
            self.p_struct_punct = cond_rate(ptot, cal_punct_hits)
        else:
            self.p_struct_diacritic = cond_rate(dtot, diac_hits)
            self.p_struct_homoglyph = cond_rate(htot, homo_hits)
            self.p_struct_punct = cond_rate(ptot, punct_hits)

    def load_clean_texts(self):
        texts = {}
        for path in self.clean_dir.glob("*_corrected.txt"):
            texts[path.stem.replace("_corrected", "")] = path.read_text(encoding="utf-8")
        return texts

    def load_raw_texts(self):
        texts = {}
        for path in self.raw_dir.glob("*_ocr_raw.txt"):
            texts[path.stem.replace("_ocr_raw", "")] = path.read_text(encoding="utf-8")
        return texts

    def _sample_inserted_char(self, fallback_char: str) -> str:
        if self.error_conf_probs is not None and "<INS>" in self.error_conf_probs.index:
            row = self.error_conf_probs.loc["<INS>"]
            row = row[row > 0]
            allowed = [c for c in row.index if isinstance(c, str) and len(c) == 1 and c not in ["<INS>", "<DEL>"]]
            if allowed:
                weights = [float(row[c]) for c in allowed]
                return str(self.random.choices(allowed, weights=weights, k=1)[0])
        return fallback_char

    def sample_substitution_from_confusion(self, ch: str) -> str:
        """Sample replacement character from empirical error confusion (substitution mass only)."""
        if not ch.strip():
            return ch
        if self.error_conf_probs is None or ch not in self.error_conf_probs.index:
            return self.random.choice(self._alphabet) if self._alphabet else ch
        row = self.error_conf_probs.loc[ch]
        row = row[row > 0]
        allowed = [
            c for c in row.index
            if isinstance(c, str) and len(c) == 1 and c not in ["<DEL>", "<INS>"]
        ]
        if not allowed:
            return ch
        weights = [float(row[c]) for c in allowed]
        wsum = sum(weights)
        if wsum <= 0.0:
            return ch
        weights = [w / wsum for w in weights]
        return str(self.random.choices(allowed, weights=weights, k=1)[0])

    def _aligned_char_noise(self, text: str, substitution_fn) -> str:
        """One alignment-style pass: per ref char choose del/sub/match; then optional insertion."""
        if not self._alphabet:
            return text
        out = []
        for ch in text:
            r = self.random.random()
            if r < self.p_del:
                pass
            elif r < self.p_del + self.p_sub:
                out.append(substitution_fn(ch))
            else:
                out.append(ch)
            if self.random.random() < self.p_ins:
                out.append(self._sample_inserted_char(ch))
        return "".join(out)

    def weighted_word_split(self, word: str) -> list:
        if len(word) < 4 or not word.isalpha():
            return [word]
        if self.random.random() >= self.p_split_word:
            return [word]
        lo, hi = 2, len(word) - 2
        if lo > hi:
            return [word]
        pos = self.random.randint(lo, hi) if lo < hi else lo
        return [word[:pos], word[pos:]]

    def weighted_word_merge(self, words: list, i: int):
        if i >= len(words) - 1:
            return None
        w1, w2 = words[i], words[i + 1]
        if len(w1) < 1 or len(w2) < 1:
            return None
        if self.random.random() >= self.p_merge_gap:
            return None
        return w1 + w2, i + 2

    def random_edit_noise(self, text: str) -> str:
        def sub_random(ch: str) -> str:
            if not ch.strip():
                return ch
            return self.random.choice(self._alphabet)

        return self._aligned_char_noise(text, sub_random)

    def confusion_matrix_noise(self, text: str) -> str:
        return self._aligned_char_noise(text, self.sample_substitution_from_confusion)

    def structure_aware_noise(self, text: str) -> str:
        base = self.confusion_matrix_noise(text)
        chars = list(base)
        for i, ch in enumerate(chars):
            r = self.random.random()
            if ch in self.diacritic_map and r < self.p_struct_diacritic:
                chars[i] = self.diacritic_map[ch]
            elif ch in self.homoglyph_map and r < self.p_struct_homoglyph:
                chars[i] = self.homoglyph_map[ch]
            elif ch in self.punct_shift_map and r < self.p_struct_punct:
                chars[i] = self.punct_shift_map[ch]

        words = "".join(chars).split()
        merged_words = []
        i = 0
        while i < len(words):
            merged = self.weighted_word_merge(words, i)
            if merged is not None:
                merged_word, new_i = merged
                merged_words.append(merged_word)
                i = new_i
            else:
                merged_words.append(words[i])
                i += 1

        split_words = []
        for w in merged_words:
            split_words.extend(self.weighted_word_split(w))

        return " ".join(split_words)

    def _char_ops_and_confusions(self, clean: str, noisy: str):
        matcher = SequenceMatcher(None, clean, noisy)
        op_counts = Counter()
        confusion_pairs = Counter()

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    confusion_pairs[(clean[i1 + k], noisy[j1 + k])] += 1
                op_counts["match"] += (i2 - i1)
                continue

            if tag == "replace":
                overlap = min(i2 - i1, j2 - j1)
                for k in range(overlap):
                    confusion_pairs[(clean[i1 + k], noisy[j1 + k])] += 1
                op_counts["substitution"] += overlap

                for k in range(overlap, i2 - i1):
                    confusion_pairs[(clean[i1 + k], "<DEL>")] += 1
                op_counts["deletion"] += max(0, (i2 - i1) - overlap)

                for k in range(overlap, j2 - j1):
                    confusion_pairs[("<INS>", noisy[j1 + k])] += 1
                op_counts["insertion"] += max(0, (j2 - j1) - overlap)
                continue

            if tag == "delete":
                for k in range(i1, i2):
                    confusion_pairs[(clean[k], "<DEL>")] += 1
                op_counts["deletion"] += (i2 - i1)
                continue

            if tag == "insert":
                for k in range(j1, j2):
                    confusion_pairs[("<INS>", noisy[k])] += 1
                op_counts["insertion"] += (j2 - j1)

        return op_counts, confusion_pairs

    def _normalize_profile(self, counter_obj: Counter, keys):
        total = sum(counter_obj.get(k, 0) for k in keys)
        if total == 0:
            return {k: 0.0 for k in keys}
        return {k: counter_obj.get(k, 0) / total for k in keys}

    def char_error_profile(self, clean: str, noisy: str):
        counts, _pairs = self._char_ops_and_confusions(clean, noisy)
        return self._normalize_profile(counts, ["substitution", "deletion", "insertion"])

    def length_ratio(self, clean: str, noisy: str) -> float:
        return len(noisy) / max(len(clean), 1)

    def _empirical_ks_distance(self, a_values, b_values) -> float:
        if not a_values or not b_values:
            return 0.0
        all_points = sorted(set(a_values + b_values))
        a_sorted = sorted(a_values)
        b_sorted = sorted(b_values)
        a_n = len(a_sorted)
        b_n = len(b_sorted)
        i = 0
        j = 0
        max_diff = 0.0
        for x in all_points:
            while i < a_n and a_sorted[i] <= x:
                i += 1
            while j < b_n and b_sorted[j] <= x:
                j += 1
            max_diff = max(max_diff, abs((i / a_n) - (j / b_n)))
        return max_diff

    def _quantile_abs_diff(self, a_values, b_values) -> float:
        if not a_values or not b_values:
            return 0.0
        quantiles = [0.1 * k for k in range(1, 10)]
        a_series = pd.Series(a_values)
        b_series = pd.Series(b_values)
        diffs = [abs(float(a_series.quantile(q)) - float(b_series.quantile(q))) for q in quantiles]
        return sum(diffs) / len(diffs)

    def _normalize_token(self, s: str) -> str:
        return "".join(ch for ch in s.lower().strip() if ch.isalnum())

    def _token_similarity(self, a: str, b: str) -> float:
        a_n = self._normalize_token(a)
        b_n = self._normalize_token(b)
        if not a_n and not b_n:
            return 1.0
        if not a_n or not b_n:
            return 0.0
        return SequenceMatcher(None, a_n, b_n).ratio()

    def split_merge_counts(self, noisy: str, clean: str, sim_threshold: float = 0.78):
        noisy_words = noisy.split()
        clean_words = clean.split()
        i = 0
        j = 0
        split_count = 0
        merge_count = 0

        while i < len(noisy_words) and j < len(clean_words):
            o = noisy_words[i]
            g = clean_words[j]
            if self._normalize_token(o) == self._normalize_token(g):
                i += 1
                j += 1
                continue

            best = None
            for g_len in (2, 3):
                if j + g_len <= len(clean_words):
                    joined = "".join(clean_words[j:j + g_len])
                    sim = self._token_similarity(o, joined)
                    if sim >= sim_threshold and (best is None or sim > best[0]):
                        best = (sim, "merge", 1, g_len)

            for o_len in (2, 3):
                if i + o_len <= len(noisy_words):
                    joined = "".join(noisy_words[i:i + o_len])
                    sim = self._token_similarity(joined, g)
                    if sim >= sim_threshold and (best is None or sim > best[0]):
                        best = (sim, "split", o_len, 1)

            if best is not None:
                _sim, typ, o_step, g_step = best
                if typ == "split":
                    split_count += 1
                else:
                    merge_count += 1
                i += o_step
                j += g_step
                continue

            i += 1
            j += 1

        return split_count, merge_count

    def js_divergence(self, p, q) -> float:
        keys = sorted(set(p) | set(q))
        eps = 1e-12
        p_vec = [p.get(k, 0.0) + eps for k in keys]
        q_vec = [q.get(k, 0.0) + eps for k in keys]

        def normalize(vec):
            s = sum(vec)
            return [x / s for x in vec] if s else [0.0 for _ in vec]

        p_vec = normalize(p_vec)
        q_vec = normalize(q_vec)
        m_vec = [(a + b) / 2 for a, b in zip(p_vec, q_vec)]

        def kl(a, b):
            total = 0.0
            for x, y in zip(a, b):
                total += x * math.log2(x / y)
            return total

        return 0.5 * kl(p_vec, m_vec) + 0.5 * kl(q_vec, m_vec)

    def evaluate_generator(self, clean_texts, real_raw_texts, synthetic_texts, generator_name):
        rows = []
        real_error_total = Counter()
        synth_error_total = Counter()
        real_confusion_total = Counter()
        synth_confusion_total = Counter()
        real_lengths = []
        synth_lengths = []
        real_split_total = 0
        real_merge_total = 0
        synth_split_total = 0
        synth_merge_total = 0
        real_word_total = 0
        synth_word_total = 0

        shared_docs = sorted(set(clean_texts) & set(real_raw_texts) & set(synthetic_texts))

        for doc in shared_docs:
            clean = clean_texts[doc]
            real = real_raw_texts[doc]
            synth = synthetic_texts[doc]

            real_ops, real_conf = self._char_ops_and_confusions(clean, real)
            synth_ops, synth_conf = self._char_ops_and_confusions(clean, synth)

            for k in ["substitution", "deletion", "insertion"]:
                real_error_total[k] += real_ops.get(k, 0)
                synth_error_total[k] += synth_ops.get(k, 0)
            real_confusion_total.update(real_conf)
            synth_confusion_total.update(synth_conf)

            real_len = self.length_ratio(clean, real)
            synth_len = self.length_ratio(clean, synth)
            real_lengths.append(real_len)
            synth_lengths.append(synth_len)

            real_split, real_merge = self.split_merge_counts(real, clean)
            synth_split, synth_merge = self.split_merge_counts(synth, clean)
            real_split_total += real_split
            real_merge_total += real_merge
            synth_split_total += synth_split
            synth_merge_total += synth_merge
            real_words = max(len(real.split()), 1)
            synth_words = max(len(synth.split()), 1)
            real_word_total += real_words
            synth_word_total += synth_words

            real_prof = self._normalize_profile(real_ops, ["substitution", "deletion", "insertion"])
            synth_prof = self._normalize_profile(synth_ops, ["substitution", "deletion", "insertion"])

            rows.append({
                "doc": doc,
                "generator": generator_name,
                "real_length_ratio": real_len,
                "synthetic_length_ratio": synth_len,
                "real_split_rate": real_split / real_words,
                "real_merge_rate": real_merge / real_words,
                "synthetic_split_rate": synth_split / synth_words,
                "synthetic_merge_rate": synth_merge / synth_words,
                "real_substitution_prop": real_prof["substitution"],
                "real_deletion_prop": real_prof["deletion"],
                "real_insertion_prop": real_prof["insertion"],
                "synthetic_substitution_prop": synth_prof["substitution"],
                "synthetic_deletion_prop": synth_prof["deletion"],
                "synthetic_insertion_prop": synth_prof["insertion"],
            })

        real_profile = self._normalize_profile(real_error_total, ["substitution", "deletion", "insertion"])
        synth_profile = self._normalize_profile(synth_error_total, ["substitution", "deletion", "insertion"])
        real_conf_profile = {f"{k[0]}->{k[1]}": v for k, v in real_confusion_total.items()}
        synth_conf_profile = {f"{k[0]}->{k[1]}": v for k, v in synth_confusion_total.items()}
        real_avg_length = sum(real_lengths) / max(len(real_lengths), 1)
        synth_avg_length = sum(synth_lengths) / max(len(synth_lengths), 1)

        real_split_rate = real_split_total / max(real_word_total, 1)
        real_merge_rate = real_merge_total / max(real_word_total, 1)
        synth_split_rate = synth_split_total / max(synth_word_total, 1)
        synth_merge_rate = synth_merge_total / max(synth_word_total, 1)

        metrics = {
            "generator": generator_name,
            "documents_compared": len(shared_docs),
            "js_divergence_confusion_distribution": self.js_divergence(real_conf_profile, synth_conf_profile),
            "js_divergence_error_profile": self.js_divergence(real_profile, synth_profile),
            "real_avg_length_ratio": real_avg_length,
            "synthetic_avg_length_ratio": synth_avg_length,
            "length_ratio_abs_diff": abs(real_avg_length - synth_avg_length),
            "length_ratio_ks_distance": self._empirical_ks_distance(real_lengths, synth_lengths),
            "length_ratio_quantile_abs_diff": self._quantile_abs_diff(real_lengths, synth_lengths),
            "real_split_rate": real_split_rate,
            "synthetic_split_rate": synth_split_rate,
            "split_rate_abs_diff": abs(real_split_rate - synth_split_rate),
            "real_merge_rate": real_merge_rate,
            "synthetic_merge_rate": synth_merge_rate,
            "merge_rate_abs_diff": abs(real_merge_rate - synth_merge_rate),
            "real_error_profile": real_profile,
            "synthetic_error_profile": synth_profile,
            "error_profile_abs_deltas": {
                "substitution": abs(real_profile["substitution"] - synth_profile["substitution"]),
                "deletion": abs(real_profile["deletion"] - synth_profile["deletion"]),
                "insertion": abs(real_profile["insertion"] - synth_profile["insertion"]),
            },
        }

        return pd.DataFrame(rows), metrics

    def save_texts(self, synthetic_by_doc, subdir: str):
        out_subdir = self.output_dir / subdir
        out_subdir.mkdir(parents=True, exist_ok=True)
        for doc, text in synthetic_by_doc.items():
            (out_subdir / f"{doc}_synthetic.txt").write_text(text, encoding="utf-8")

    def save_pairs(
        self,
        pairs_by_doc: Dict[str, List[Tuple[str, str]]],
        subdir: str,
    ) -> None:
        """Write sentence-level (clean, noisy) pairs per doc as JSONL.

        Pairs are emitted as ``{"clean": ..., "noisy": ...}`` rows in
        ``<subdir>/<doc>_pairs.jsonl``.
        """
        out_subdir = self.output_dir / subdir
        out_subdir.mkdir(parents=True, exist_ok=True)
        for doc, pairs in pairs_by_doc.items():
            with (out_subdir / f"{doc}_pairs.jsonl").open(
                "w", encoding="utf-8"
            ) as f:
                for clean, noisy in pairs:
                    f.write(
                        json.dumps(
                            {"clean": clean, "noisy": noisy},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    def generate_with_pairs(
        self,
        clean_text: str,
        noise_fn: Callable[[str], str],
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Apply ``noise_fn`` per sentence; return (synth_text, pairs).

        The clean text is paragraph- then sentence-segmented; ``noise_fn`` is
        invoked once per non-empty clean sentence so each emitted pair is a
        true 1:1 (clean_sentence, noisy_sentence) correspondence. The
        reassembled synthetic text re-joins noisy sentences with single spaces
        within paragraphs and preserves paragraph boundaries (``\\n\\n``).
        Pairs whose noisy side becomes empty after stripping are dropped.
        """
        paragraphs = _split_paragraphs(clean_text)
        if not paragraphs:
            flat = re.sub(r"\s+", " ", clean_text or "").strip()
            paragraphs = [flat] if flat else []

        pairs: List[Tuple[str, str]] = []
        out_paras: List[str] = []
        for para in paragraphs:
            sents = _split_sentences(para)
            noisy_sents: List[str] = []
            for s in sents:
                n = noise_fn(s)
                if not n or not n.strip():
                    continue
                pairs.append((s, n))
                noisy_sents.append(n)
            if noisy_sents:
                out_paras.append(" ".join(noisy_sents))
        synth_text = "\n\n".join(out_paras)
        return synth_text, pairs

    def run(self):
        self.load_phase2_statistics()
        clean_texts = self.load_clean_texts()
        self._calibrate_structure_noise_rates(clean_texts)
        raw_texts = self.load_raw_texts()

        random_synth: Dict[str, str] = {}
        confusion_synth: Dict[str, str] = {}
        structure_synth: Dict[str, str] = {}
        random_pairs: Dict[str, List[Tuple[str, str]]] = {}
        confusion_pairs: Dict[str, List[Tuple[str, str]]] = {}
        structure_pairs: Dict[str, List[Tuple[str, str]]] = {}

        for doc, clean in clean_texts.items():
            r_text, r_pairs = self.generate_with_pairs(clean, self.random_edit_noise)
            random_synth[doc] = r_text
            random_pairs[doc] = r_pairs

            c_text, c_pairs = self.generate_with_pairs(clean, self.confusion_matrix_noise)
            confusion_synth[doc] = c_text
            confusion_pairs[doc] = c_pairs

            s_text, s_pairs = self.generate_with_pairs(clean, self.structure_aware_noise)
            structure_synth[doc] = s_text
            structure_pairs[doc] = s_pairs

        self.save_texts(random_synth, "random_edit_noise")
        self.save_pairs(random_pairs, "random_edit_noise")
        self.save_texts(confusion_synth, "confusion_matrix_noise")
        self.save_pairs(confusion_pairs, "confusion_matrix_noise")
        self.save_texts(structure_synth, "structure_aware_noise")
        self.save_pairs(structure_pairs, "structure_aware_noise")

        all_metrics = []
        all_doc_rows = []

        for name, synth in [
            ("random_edit_noise", random_synth),
            ("confusion_matrix_noise", confusion_synth),
            ("structure_aware_noise", structure_synth),
        ]:
            df_rows, metrics = self.evaluate_generator(clean_texts, raw_texts, synth, name)
            all_doc_rows.append(df_rows)
            all_metrics.append(metrics)

        pd.concat(all_doc_rows, ignore_index=True).to_csv(
            self.output_dir / "noise_realism_doc_level.csv",
            index=False,
            encoding="utf-8",
        )

        with open(self.output_dir / "noise_realism_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)

        provenance = {
            "phase2_dir": str(self.phase2_dir),
            "clean_dir": str(self.clean_dir),
            "raw_dir": str(self.raw_dir),
            "calibration_doc_ids": (
                sorted(self.calibration_doc_ids)
                if self.calibration_doc_ids is not None
                else None
            ),
            "calibration_filter_active": self.calibration_doc_ids is not None,
            "generated_doc_ids": sorted(clean_texts.keys()),
            "calibrated_rates": {
                "p_del": self.p_del,
                "p_sub": self.p_sub,
                "p_ins": self.p_ins,
                "p_split_word": self.p_split_word,
                "p_merge_gap": self.p_merge_gap,
                "p_struct_diacritic": self.p_struct_diacritic,
                "p_struct_homoglyph": self.p_struct_homoglyph,
                "p_struct_punct": self.p_struct_punct,
            },
        }
        with open(self.output_dir / "provenance.json", "w", encoding="utf-8") as f:
            json.dump(provenance, f, ensure_ascii=False, indent=2)

        print("Phase 3 synthetic noise modeling completed.")
        print(f"Outputs saved to: {self.output_dir.resolve()}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_full_mode(
    clean_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    phase2_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> Path:
    """Run Phase 3 with the legacy full-corpus statistics (descriptive only)."""
    repo_root = _repo_root()
    clean = clean_dir or (repo_root / "phase1" / "corrected_ocr")
    raw = raw_dir or (repo_root / "phase1" / "raw_ocr")
    p2 = phase2_dir or (repo_root / "phase2" / "phase2_output")
    out = output_dir or (repo_root / "phase3" / "phase3_output")
    print(
        f"[phase3] full mode: clean={clean} raw={raw} phase2={p2} output={out}",
        flush=True,
    )
    analyzer = Phase3SyntheticNoise(
        clean_dir=str(clean),
        raw_dir=str(raw),
        phase2_dir=str(p2),
        output_dir=str(out),
        seed=seed,
    )
    analyzer.run()
    return out


def run_train_only_mode(
    clean_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    phase2_train_only_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> Path:
    """Run Phase 3 using **train-only** Phase 2 statistics.

    Reads from ``phase2/phase2_output_train_only/`` (created by Phase 2's
    ``run_train_only_mode``) and writes synthetic noise to
    ``phase3/phase3_output_train_only/``. Synthetic text is generated for ALL
    clean documents (val/test docs need synthetic versions for evaluation),
    but the noise model itself was fit on TRAIN docs only.
    """
    from phase2.ocr_error_analysis import SPLITS as PHASE2_SPLITS

    repo_root = _repo_root()
    clean = clean_dir or (repo_root / "phase1" / "corrected_ocr")
    raw = raw_dir or (repo_root / "phase1" / "raw_ocr")
    p2 = phase2_train_only_dir or (repo_root / "phase2" / "phase2_output_train_only")
    out = output_dir or (repo_root / "phase3" / "phase3_output_train_only")
    train_docs = set(PHASE2_SPLITS["train"])
    print(
        f"[phase3] train-only mode: clean={clean} raw={raw} "
        f"phase2={p2} output={out} calibration_doc_ids={sorted(train_docs)}",
        flush=True,
    )
    if not (p2 / "error_distribution.json").exists():
        raise FileNotFoundError(
            f"{p2 / 'error_distribution.json'} is missing. "
            "Run phase2.ocr_error_analysis.run_train_only_mode() first."
        )
    analyzer = Phase3SyntheticNoise(
        clean_dir=str(clean),
        raw_dir=str(raw),
        phase2_dir=str(p2),
        output_dir=str(out),
        seed=seed,
        calibration_doc_ids=train_docs,
    )
    analyzer.run()
    if analyzer.calibration_doc_ids & (
        set(PHASE2_SPLITS["val"]) | set(PHASE2_SPLITS["test"])
    ):
        raise AssertionError(
            "Phase 3 train-only calibration includes val/test docs; "
            "this would re-introduce the leak."
        )
    return out


def main():
    run_full_mode()
    run_train_only_mode()


if __name__ == "__main__":
    main()