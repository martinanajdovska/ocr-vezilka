import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


SPLITS = {
    "train": [
        "dnevnik_po_mnogu_godini",
        "itar_pejo",
        "Pesni",
        "Prezir",
        "Samecot",
        "sina_pesna",
        "tajnopis",
        "Toj",
        "viktor_kupidon"
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


class Phase2:
    def __init__(self, output_dir="phase2_output"):
        self.char_ops = []
        self.word_ops = []
        self.boundary_errors = []
        self.matched_pairs = []

        self.doc_index = {}

        self.confusion = defaultdict(lambda: defaultdict(int))
        self.confusion_prob = None
        self.error_confusion = defaultdict(lambda: defaultdict(int))
        self.error_confusion_prob = None

        self.sub_pairs = Counter()
        self.diacritic = Counter()
        self.homoglyph = Counter()
        self.punct = Counter()

        self.word_splits = []
        self.word_merges = []

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cyr = set(
            "АБВГДЃЕЖЗЅИЈКЛЉМНЊОПРСТЌУФХЦЧЏШ"
            "абвгдѓежзѕијклљмнњопрстќуфхцчџш"
        )
        self.lat = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        self.punct_set = set(".,!?;:'\"„“”()-—;:[]{}")
        self.space_markers = {" ", "<SPACE>", "<space>"}

        self.diacritic_pairs = {
            ("Г", "Ѓ"), ("г", "ѓ"),
            ("К", "Ќ"), ("к", "ќ"),
            ("Л", "Љ"), ("л", "љ"),
            ("Н", "Њ"), ("н", "њ"),
        }

        self.homoglyph_pairs = {
            ("А", "A"), ("а", "a"),
            ("В", "B"), ("Е", "E"), ("е", "e"),
            ("К", "K"), ("М", "M"), ("Н", "H"),
            ("О", "O"), ("о", "o"), ("Р", "P"),
            ("С", "C"), ("с", "c"), ("Т", "T"),
            ("Х", "X"), ("У", "Y"), ("Ѕ", "S")
        }

    def load_dataset(self, base_dir):
        base_dir = Path(base_dir)

        for split, docs in SPLITS.items():
            split_dir = base_dir / split
            if not split_dir.exists():
                continue

            for doc_folder in split_dir.iterdir():
                if not doc_folder.is_dir():
                    continue

                doc = doc_folder.name
                if doc not in docs:
                    continue

                char_file = doc_folder / "char_ops.json"
                word_file = doc_folder / "word_ops.json"
                pair_file = doc_folder / "matched_pairs.json"
                boundary_file = doc_folder / "word_boundary_errors.json"

                if not char_file.exists():
                    continue

                with open(char_file, "r", encoding="utf-8") as f:
                    char_ops = json.load(f)

                for op in char_ops:
                    op["doc_id"] = doc
                    op["split"] = split
                self.char_ops.extend(char_ops)

                if word_file.exists():
                    with open(word_file, "r", encoding="utf-8") as f:
                        word_ops = json.load(f)
                    for op in word_ops:
                        op["doc_id"] = doc
                        op["split"] = split
                    self.word_ops.extend(word_ops)
                else:
                    word_ops = []

                if pair_file.exists():
                    with open(pair_file, "r", encoding="utf-8") as f:
                        pairs = json.load(f)
                    for p in pairs:
                        p["doc_id"] = doc
                        p["split"] = split
                    self.matched_pairs.extend(pairs)
                else:
                    pairs = []

                if boundary_file.exists():
                    with open(boundary_file, "r", encoding="utf-8") as f:
                        boundaries = json.load(f)
                    for b in boundaries:
                        b["doc_id"] = doc
                        b["split"] = split
                    self.boundary_errors.extend(boundaries)

                self.doc_index[doc] = {
                    "split": split,
                    "char_ops": len(char_ops),
                    "word_ops": len(word_ops),
                    "matched_pairs": len(pairs)
                }

    def build_confusion_matrix(self):
        for op in self.char_ops:
            gt = str(op.get("gt", ""))
            ocr = str(op.get("ocr", ""))
            typ = op.get("op", "")

            if typ in ["match", "substitution"]:
                self.confusion[gt][ocr] += 1

                if typ == "substitution":
                    self.error_confusion[gt][ocr] += 1
                    self.sub_pairs[(gt, ocr)] += 1
                    self._classify(gt, ocr)

            elif typ == "deletion":
                self.confusion[gt]["<DEL>"] += 1
                self.error_confusion[gt]["<DEL>"] += 1

            elif typ == "insertion":
                self.confusion["<INS>"][ocr] += 1
                self.error_confusion["<INS>"][ocr] += 1

    def normalize_confusion_matrix(self):
        df = pd.DataFrame(self.confusion).fillna(0).T.sort_index()
        self.confusion_prob = df.div(df.sum(axis=1), axis=0).fillna(0)
        return self.confusion_prob

    def normalize_error_confusion_matrix(self):
        df = pd.DataFrame(self.error_confusion).fillna(0).T.sort_index()
        self.error_confusion_prob = df.div(df.sum(axis=1), axis=0).fillna(0)
        return self.error_confusion_prob

    def _classify(self, gt, ocr):
        if (gt, ocr) in self.diacritic_pairs or (ocr, gt) in self.diacritic_pairs:
            self.diacritic[(gt, ocr)] += 1

        if (gt, ocr) in self.homoglyph_pairs or (ocr, gt) in self.homoglyph_pairs:
            self.homoglyph[(gt, ocr)] += 1
        elif (gt in self.cyr and ocr in self.lat) or (gt in self.lat and ocr in self.cyr):
            self.homoglyph[(gt, ocr)] += 1

        if gt in self.punct_set or ocr in self.punct_set:
            self.punct[(gt, ocr)] += 1

    def error_distribution(self):
        c = Counter(op["op"] for op in self.char_ops)
        total = sum(c.values()) if c else 1

        split_merge_stats = self.word_boundary_analysis()
        structural_count = split_merge_stats["split_count"] + split_merge_stats["merge_count"]

        return {
            "total_char_ops": int(total),
            "match_pct": 100 * c.get("match", 0) / total,
            "substitution_pct": 100 * c.get("substitution", 0) / total,
            "deletion_pct": 100 * c.get("deletion", 0) / total,
            "insertion_pct": 100 * c.get("insertion", 0) / total,
            "split_count": int(split_merge_stats["split_count"]),
            "merge_count": int(split_merge_stats["merge_count"]),
            "structural_count": int(structural_count)
        }

    def top_substitutions(self, n=30):
        rows = []
        for (gt, ocr), count in self.sub_pairs.most_common(n):
            rows.append({
                "gt": gt,
                "ocr": ocr,
                "count": count
            })
        return pd.DataFrame(rows)

    def top_counter_to_df(self, counter_obj, n=30, colname="pair"):
        rows = []
        for (a, b), count in counter_obj.most_common(n):
            rows.append({
                "src": a,
                "tgt": b,
                "count": count
            })
        return pd.DataFrame(rows)

    def word_boundary_analysis(self):
        split_count = 0
        merge_count = 0
        split_examples = []
        merge_examples = []

        for op in self.boundary_errors:
            typ = str(op.get("type", "")).lower()
            gt = op.get("gt", "")
            ocr = op.get("ocr", "")

            if typ == "split":
                split_count += 1
                self.word_splits.append(op)
                if len(split_examples) < 20:
                    split_examples.append({"gt": gt, "ocr": ocr, "doc_id": op.get("doc_id", "")})

            elif typ == "merge":
                merge_count += 1
                self.word_merges.append(op)
                if len(merge_examples) < 20:
                    merge_examples.append({"gt": gt, "ocr": ocr, "doc_id": op.get("doc_id", "")})

        total_word_ops = max(len(self.boundary_errors), 1)

        return {
            "total_word_ops": len(self.boundary_errors),
            "split_count": split_count,
            "merge_count": merge_count,
            "split_rate_pct": 100 * split_count / total_word_ops,
            "merge_rate_pct": 100 * merge_count / total_word_ops,
            "split_examples": split_examples,
            "merge_examples": merge_examples
        }

    def stratify_docs(self):
        doc_scores = defaultdict(lambda: {"err": 0, "total": 0, "split": None})

        for op in self.char_ops:
            doc = op["doc_id"]
            split = op["split"]

            doc_scores[doc]["total"] += 1
            doc_scores[doc]["err"] += int(op["op"] != "match")
            doc_scores[doc]["split"] = split

        rows = []
        for d, v in doc_scores.items():
            cer = v["err"] / v["total"] if v["total"] else 0.0
            rows.append({
                "doc": d,
                "split": v["split"],
                "errors": v["err"],
                "total_ops": v["total"],
                "cer_proxy": cer
            })

        df = pd.DataFrame(rows).sort_values("cer_proxy", ascending=False)
        high = df[df["cer_proxy"] >= df["cer_proxy"].quantile(0.75)].copy()
        low = df[df["cer_proxy"] <= df["cer_proxy"].quantile(0.25)].copy()

        return df, high, low

    def split_level_stats(self):
        stats = defaultdict(lambda: {"err": 0, "total": 0, "docs": set()})

        for op in self.char_ops:
            split = op["split"]
            stats[split]["total"] += 1
            stats[split]["err"] += int(op["op"] != "match")
            stats[split]["docs"].add(op["doc_id"])

        rows = []
        for split, v in stats.items():
            rows.append({
                "split": split,
                "docs": len(v["docs"]),
                "total_ops": v["total"],
                "errors": v["err"],
                "cer_proxy": v["err"] / v["total"] if v["total"] else 0.0
            })

        return pd.DataFrame(rows).sort_values("split")

    def plot_confusion(self, top_n=25, use_error_only=True):
        if use_error_only:
            if self.error_confusion_prob is None:
                self.normalize_error_confusion_matrix()
            matrix = self.error_confusion_prob.copy()
            count_matrix = pd.DataFrame(self.error_confusion).fillna(0).T
            title = f"Top {top_n} Error Confusions (Normalized)"
            out_name = "confusion_heatmap_error_only.png"
        else:
            if self.confusion_prob is None:
                self.normalize_confusion_matrix()
            matrix = self.confusion_prob.copy()
            count_matrix = pd.DataFrame(self.confusion).fillna(0).T
            title = f"Top {top_n} Full Confusions (Normalized)"
            out_name = "confusion_heatmap_full.png"

        row_strength = count_matrix.sum(axis=1).sort_values(ascending=False)
        top_rows = row_strength.head(top_n).index.tolist()

        sub = matrix.loc[top_rows]
        top_cols = count_matrix.loc[top_rows].sum(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
        sub = sub[top_cols]

        plt.figure(figsize=(14, 10))
        sns.heatmap(sub, cmap="mako", linewidths=0.3, linecolor="white")
        plt.title(title)
        plt.xlabel("OCR character")
        plt.ylabel("Ground-truth character")
        plt.tight_layout()
        plt.savefig(self.output_dir / out_name, dpi=300)
        plt.close()

    def plot_error_distribution(self, dist):
        keys = ["substitution_pct", "deletion_pct", "insertion_pct"]
        values = [dist[k] for k in keys]

        plt.figure(figsize=(8, 5))
        plt.bar(keys, values)
        plt.title("Character Error Distribution")
        plt.ylabel("Percentage")
        plt.tight_layout()
        plt.savefig(self.output_dir / "error_distribution.png", dpi=300)
        plt.close()

    def plot_word_boundaries(self, stats):
        plt.figure(figsize=(6, 5))
        plt.bar(["splits", "merges"], [stats["split_count"], stats["merge_count"]])
        plt.title("Word Boundary Errors")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.output_dir / "word_boundary_errors.png", dpi=300)
        plt.close()

    def save_outputs(self, dist, wb, doc_df, high_df, low_df, split_df):
        confusion_counts_df = pd.DataFrame(self.confusion).fillna(0).T.sort_index()
        error_confusion_counts_df = pd.DataFrame(self.error_confusion).fillna(0).T.sort_index()

        confusion_counts_df.to_csv(self.output_dir / "confusion_matrix_counts.csv", encoding="utf-8")
        self.confusion_prob.to_csv(self.output_dir / "confusion_matrix_probs.csv", encoding="utf-8")

        error_confusion_counts_df.to_csv(self.output_dir / "error_confusion_counts.csv", encoding="utf-8")
        self.error_confusion_prob.to_csv(self.output_dir / "error_confusion_probs.csv", encoding="utf-8")

        self.top_substitutions(30).to_csv(self.output_dir / "top_30_substitutions.csv", index=False, encoding="utf-8")
        self.top_counter_to_df(self.diacritic, 30).to_csv(self.output_dir / "top_diacritic_confusions.csv", index=False, encoding="utf-8")
        self.top_counter_to_df(self.homoglyph, 30).to_csv(self.output_dir / "top_homoglyph_confusions.csv", index=False, encoding="utf-8")
        self.top_counter_to_df(self.punct, 30).to_csv(self.output_dir / "top_punctuation_confusions.csv", index=False, encoding="utf-8")

        doc_df.to_csv(self.output_dir / "document_error_stats.csv", index=False, encoding="utf-8")
        high_df.to_csv(self.output_dir / "high_error_docs.csv", index=False, encoding="utf-8")
        low_df.to_csv(self.output_dir / "low_error_docs.csv", index=False, encoding="utf-8")
        split_df.to_csv(self.output_dir / "split_level_stats.csv", index=False, encoding="utf-8")

        with open(self.output_dir / "error_distribution.json", "w", encoding="utf-8") as f:
            json.dump(dist, f, ensure_ascii=False, indent=2)

        with open(self.output_dir / "word_boundary_stats.json", "w", encoding="utf-8") as f:
            json.dump(wb, f, ensure_ascii=False, indent=2)

        summary = {
            "documents_loaded": len(self.doc_index),
            "char_ops_total": len(self.char_ops),
            "word_ops_total": len(self.word_ops),
            "boundary_errors_total": len(self.boundary_errors),
            "matched_pairs_total": len(self.matched_pairs),
            "top_substitution_count": int(sum(self.sub_pairs.values())),
            "diacritic_confusions_total": int(sum(self.diacritic.values())),
            "homoglyph_confusions_total": int(sum(self.homoglyph.values())),
            "punctuation_confusions_total": int(sum(self.punct.values()))
        }

        with open(self.output_dir / "phase2_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def run(self, base_dir):
        self.load_dataset(base_dir)
        self.build_confusion_matrix()
        self.normalize_confusion_matrix()
        self.normalize_error_confusion_matrix()

        wb = self.word_boundary_analysis()
        dist = self.error_distribution()

        doc_df, high_df, low_df = self.stratify_docs()
        split_df = self.split_level_stats()

        self.plot_error_distribution(dist)
        self.plot_confusion(top_n=25, use_error_only=True)
        self.plot_confusion(top_n=25, use_error_only=False)
        self.plot_word_boundaries(wb)

        self.save_outputs(dist, wb, doc_df, high_df, low_df, split_df)

        print("Phase 2 analysis completed.")
        print(f"Outputs saved to: {self.output_dir.resolve()}")


def main():
    analyzer = Phase2(output_dir="../phase2/phase2_output")
    analyzer.run(base_dir="../phase1/phase1_output")


if __name__ == "__main__":
    main()