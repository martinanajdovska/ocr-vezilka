import json
import math
import random
from pathlib import Path
from collections import Counter

import pandas as pd


class Phase3SyntheticNoise:
    def __init__(
        self,
        clean_dir="../phase1/corrected_ocr",
        raw_dir="../phase1/raw_ocr",
        phase2_dir="../phase2/phase2_output",
        output_dir="../phase3/phase3_output",
        seed=42,
    ):
        self.clean_dir = Path(clean_dir)
        self.raw_dir = Path(raw_dir)
        self.phase2_dir = Path(phase2_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random = random.Random(seed)

        self.error_dist = None
        self.error_conf_counts = None
        self.error_conf_probs = None
        self.word_boundary_stats = None
        self.boundary_split_rate = 0.0
        self.boundary_merge_rate = 0.0
        self.sub_rate = 0.0
        self.del_rate = 0.0
        self.ins_rate = 0.0

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

        self.boundary_split_rate = float(self.word_boundary_stats.get("split_rate_pct", 0.0)) / 100.0
        self.boundary_merge_rate = float(self.word_boundary_stats.get("merge_rate_pct", 0.0)) / 100.0
        self.sub_rate = float(self.error_dist.get("substitution_pct", 0.0)) / 100.0
        self.del_rate = float(self.error_dist.get("deletion_pct", 0.0)) / 100.0
        self.ins_rate = float(self.error_dist.get("insertion_pct", 0.0)) / 100.0

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

    def sample_error_confusion(self, ch: str) -> str:
        if self.error_conf_probs is None or ch not in self.error_conf_probs.index:
            return ch
        row = self.error_conf_probs.loc[ch]
        row = row[row > 0]
        if row.empty:
            return ch
        sampled = self.random.choices(list(row.index), weights=list(row.values), k=1)[0]
        if sampled == "<DEL>":
            return ""
        if sampled == "<INS>":
            return ch
        return str(sampled)

    def weighted_word_split(self, word: str) -> list:
        if len(word) < 9 or not word.isalpha():
            return [word]
        if self.random.random() >= max(self.boundary_split_rate * 0.07, 0.002):
            return [word]
        pos = self.random.randint(3, len(word) - 3)
        return [word[:pos], word[pos:]]

    def weighted_word_merge(self, words: list, i: int):
        if i >= len(words) - 1:
            return None
        w1, w2 = words[i], words[i + 1]
        if len(w1) < 2 or len(w2) < 2:
            return None
        if self.random.random() >= max(self.boundary_merge_rate * 0.03, 0.001):
            return None
        return w1 + w2, i + 2

    def random_edit_noise(self, text: str) -> str:
        out = []
        alphabet = [c for c in self.error_conf_counts.columns if isinstance(c, str) and len(c) == 1 and c not in ["<DEL>", "<INS>"]]
        if not alphabet:
            return text
        for ch in text:
            r = self.random.random()
            if r < self.del_rate:
                continue
            if r < self.del_rate + self.sub_rate:
                out.append(self.random.choice(alphabet) if ch.strip() else ch)
            else:
                out.append(ch)
            if self.random.random() < self.ins_rate * 0.6:
                out.append(self.random.choice(alphabet))
        return "".join(out)

    def confusion_matrix_noise(self, text: str) -> str:
        out = []
        trigger_rate = max((self.sub_rate + self.del_rate + self.ins_rate) * 0.7, 0.03)
        for ch in text:
            if self.random.random() < trigger_rate:
                out.append(self.sample_error_confusion(ch))
            else:
                out.append(ch)
        return "".join(out)

    def structure_aware_noise(self, text: str) -> str:
        base = self.confusion_matrix_noise(text)
        chars = list(base)
        for i, ch in enumerate(chars):
            r = self.random.random()
            if ch in self.diacritic_map and r < 0.06:
                chars[i] = self.diacritic_map[ch]
            elif ch in self.homoglyph_map and r < 0.10:
                chars[i] = self.homoglyph_map[ch]
            elif ch in self.punct_shift_map and r < 0.05:
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

    def phase2_aligned_noise(self, text: str) -> str:
        text = self.confusion_matrix_noise(text)
        chars = list(text)
        for i, ch in enumerate(chars):
            r = self.random.random()
            if ch in self.diacritic_map and r < 0.06:
                chars[i] = self.diacritic_map[ch]
            elif ch in self.homoglyph_map and r < 0.10:
                chars[i] = self.homoglyph_map[ch]
            elif ch in self.punct_shift_map and r < 0.05:
                chars[i] = self.punct_shift_map[ch]

        tokens = "".join(chars).split()
        if not tokens:
            return "".join(chars)

        out = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and self.random.random() < max(self.boundary_merge_rate * 0.02, 0.0005):
                out.append(tokens[i] + tokens[i + 1])
                i += 2
                continue
            out.extend(self.weighted_word_split(tokens[i]))
            i += 1

        return " ".join(out)

    def char_error_profile(self, clean: str, noisy: str):
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, clean, noisy)
        counts = Counter()
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace":
                counts["substitution"] += max(i2 - i1, j2 - j1)
            elif tag == "delete":
                counts["deletion"] += i2 - i1
            elif tag == "insert":
                counts["insertion"] += j2 - j1
        total = sum(counts.values())
        if total == 0:
            return {"substitution": 0.0, "deletion": 0.0, "insertion": 0.0}
        return {k: v / total for k, v in counts.items()}

    def length_ratio(self, clean: str, noisy: str) -> float:
        return len(noisy) / max(len(clean), 1)

    def js_divergence(self, p, q) -> float:
        keys = sorted(set(p) | set(q))
        p_vec = [p.get(k, 0.0) for k in keys]
        q_vec = [q.get(k, 0.0) for k in keys]

        def normalize(vec):
            s = sum(vec)
            return [x / s for x in vec] if s else [0.0 for _ in vec]

        p_vec = normalize(p_vec)
        q_vec = normalize(q_vec)
        m_vec = [(a + b) / 2 for a, b in zip(p_vec, q_vec)]

        def kl(a, b):
            total = 0.0
            for x, y in zip(a, b):
                if x > 0 and y > 0:
                    total += x * math.log2(x / y)
            return total

        return 0.5 * kl(p_vec, m_vec) + 0.5 * kl(q_vec, m_vec)

    def evaluate_generator(self, clean_texts, real_raw_texts, synthetic_texts, generator_name):
        rows = []
        real_profile_total = Counter()
        synth_profile_total = Counter()
        real_lengths = []
        synth_lengths = []

        shared_docs = sorted(set(clean_texts) & set(real_raw_texts) & set(synthetic_texts))

        for doc in shared_docs:
            clean = clean_texts[doc]
            real = real_raw_texts[doc]
            synth = synthetic_texts[doc]

            real_prof = self.char_error_profile(clean, real)
            synth_prof = self.char_error_profile(clean, synth)

            for k, v in real_prof.items():
                real_profile_total[k] += v
            for k, v in synth_prof.items():
                synth_profile_total[k] += v

            real_len = self.length_ratio(clean, real)
            synth_len = self.length_ratio(clean, synth)
            real_lengths.append(real_len)
            synth_lengths.append(synth_len)

            rows.append({
                "doc": doc,
                "generator": generator_name,
                "real_length_ratio": real_len,
                "synthetic_length_ratio": synth_len,
            })

        real_profile_avg = {k: v / max(len(shared_docs), 1) for k, v in real_profile_total.items()}
        synth_profile_avg = {k: v / max(len(shared_docs), 1) for k, v in synth_profile_total.items()}

        metrics = {
            "generator": generator_name,
            "documents_compared": len(shared_docs),
            "js_divergence_error_profile": self.js_divergence(real_profile_avg, synth_profile_avg),
            "real_avg_length_ratio": sum(real_lengths) / max(len(real_lengths), 1),
            "synthetic_avg_length_ratio": sum(synth_lengths) / max(len(synth_lengths), 1),
            "length_ratio_abs_diff": abs((sum(real_lengths) / max(len(real_lengths), 1)) - (sum(synth_lengths) / max(len(synth_lengths), 1))),
            "real_error_profile": real_profile_avg,
            "synthetic_error_profile": synth_profile_avg,
        }

        return pd.DataFrame(rows), metrics

    def save_texts(self, synthetic_by_doc, subdir: str):
        out_subdir = self.output_dir / subdir
        out_subdir.mkdir(parents=True, exist_ok=True)
        for doc, text in synthetic_by_doc.items():
            (out_subdir / f"{doc}_synthetic.txt").write_text(text, encoding="utf-8")

    def run(self):
        self.load_phase2_statistics()
        clean_texts = self.load_clean_texts()
        raw_texts = self.load_raw_texts()

        random_synth = {}
        confusion_synth = {}
        structure_synth = {}
        phase2_aligned_synth = {}

        for doc, clean in clean_texts.items():
            random_synth[doc] = self.random_edit_noise(clean)
            confusion_synth[doc] = self.confusion_matrix_noise(clean)
            structure_synth[doc] = self.structure_aware_noise(clean)
            phase2_aligned_synth[doc] = self.phase2_aligned_noise(clean)

        self.save_texts(random_synth, "random_edit_noise")
        self.save_texts(confusion_synth, "confusion_matrix_noise")
        self.save_texts(structure_synth, "structure_aware_noise")
        self.save_texts(phase2_aligned_synth, "phase2_aligned_noise")

        all_metrics = []
        all_doc_rows = []

        for name, synth in [
            ("random_edit_noise", random_synth),
            ("confusion_matrix_noise", confusion_synth),
            ("structure_aware_noise", structure_synth),
            ("phase2_aligned_noise", phase2_aligned_synth),
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

        print("Phase 3 synthetic noise modeling completed.")
        print(f"Outputs saved to: {self.output_dir.resolve()}")


def main():
    analyzer = Phase3SyntheticNoise()
    analyzer.run()


if __name__ == "__main__":
    main()