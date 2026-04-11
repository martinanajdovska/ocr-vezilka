import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter

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
    def __init__(self):
        self.char_ops = []
        self.word_ops = []
        self.matched_pairs = []

        self.doc_index = {}

        self.confusion = defaultdict(lambda: defaultdict(int))
        self.confusion_prob = None

        self.sub_pairs = Counter()
        self.diacritic = Counter()
        self.homoglyph = Counter()
        self.punct = Counter()

        self.word_splits = []
        self.word_merges = []

        self.cyr = set(
            "АБВГДЃЕЖЗЅИЈКЛЉМНЊОПРСТЌУФХЦЧЏШ"
            "абвгдѓежзѕијклљмнњопрстќуфхцчџш"
        )
        self.lat = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        self.punct_set = set(".,!?;:'\"„”")

    def load_dataset(self, base_dir):
        base_dir = Path(base_dir)

        total_docs = 0
        total_ops = 0

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

                if not char_file.exists():
                    continue

                char_ops = json.load(open(char_file, "r", encoding="utf-8"))
                for op in char_ops:
                    op["doc_id"] = doc
                    op["split"] = split

                self.char_ops.extend(char_ops)

                word_ops = []
                if word_file.exists():
                    word_ops = json.load(open(word_file, "r", encoding="utf-8"))
                    for op in word_ops:
                        op["doc_id"] = doc
                        op["split"] = split

                    self.word_ops.extend(word_ops)

                pairs = []
                if pair_file.exists():
                    pairs = json.load(open(pair_file, "r", encoding="utf-8"))
                    for p in pairs:
                        p["doc_id"] = doc
                        p["split"] = split

                    self.matched_pairs.extend(pairs)

                self.doc_index[doc] = {
                    "split": split,
                    "ops": len(char_ops)
                }

                total_docs += 1
                total_ops += len(char_ops)


    def build_confusion_matrix(self):
        for op in self.char_ops:
            gt = op["gt"]
            ocr = op["ocr"]
            typ = op["op"]

            if typ in ["match", "substitution"]:
                self.confusion[gt][ocr] += 1
                if typ == "substitution":
                    self.sub_pairs[(gt, ocr)] += 1
                    self._classify(gt, ocr)

            elif typ == "deletion":
                self.confusion[gt]["<DEL>"] += 1

            elif typ == "insertion":
                self.confusion["<INS>"][ocr] += 1


    def normalize_confusion_matrix(self):
        df = pd.DataFrame(self.confusion).fillna(0).T
        self.confusion_prob = df.div(df.sum(axis=1), axis=0).fillna(0)
        return self.confusion_prob


    def _classify(self, gt, ocr):
        diacritic_pairs = {
            ("Г","Ѓ"), ("г","ѓ"),
            ("К","Ќ"), ("к","ќ"),
            ("Л","Љ"), ("л","љ"),
            ("Н","Њ"), ("н","њ"),
        }

        if (gt, ocr) in diacritic_pairs or (ocr, gt) in diacritic_pairs:
            self.diacritic[(gt, ocr)] += 1

        if (gt in self.cyr and ocr in self.lat) or (gt in self.lat and ocr in self.cyr):
            self.homoglyph[(gt, ocr)] += 1

        if gt in self.punct_set or ocr in self.punct_set:
            self.punct[(gt, ocr)] += 1


    def error_distribution(self):
        c = Counter(op["op"] for op in self.char_ops)
        total = sum(c.values())

        return {
            "substitution_pct": 100 * c["substitution"] / total,
            "deletion_pct": 100 * c["deletion"] / total,
            "insertion_pct": 100 * c["insertion"] / total,
        }


    def top_substitutions(self, n=30):
        return pd.DataFrame(
            self.sub_pairs.most_common(n),
            columns=["gt_ocr", "count"]
        )


    def word_boundary_analysis(self):
        #TODO
        pass


    def stratify_docs(self):
        doc_scores = defaultdict(lambda: {"err": 0, "total": 0, "split": None})

        for op in self.char_ops:
            doc = op["doc_id"]
            split = op["split"]

            doc_scores[doc]["total"] += 1
            doc_scores[doc]["err"] += (op["op"] != "match")
            doc_scores[doc]["split"] = split

        df = pd.DataFrame([
            {
                "doc": d,
                "split": v["split"],
                "cer": v["err"] / v["total"] if v["total"] else 0
            }
            for d, v in doc_scores.items()
        ])

        high = df[df["cer"] >= df["cer"].quantile(0.75)]
        low = df[df["cer"] <= df["cer"].quantile(0.25)]

        return high, low


    def plot_confusion(self):
        #TODO
        pass

    def plot_error_distribution(self, dist):
        plt.figure()
        plt.bar(dist.keys(), dist.values())
        plt.title("Error Distribution")
        plt.show()

    def plot_word_boundaries(self, stats):
        plt.figure()
        plt.bar(["splits", "merges"], [stats["split_count"], stats["merge_count"]])
        plt.title("Word Boundary Errors")
        plt.show()

    def run(self, base_dir):
        self.load_dataset(base_dir)

        self.build_confusion_matrix()

        self.normalize_confusion_matrix()

        dist = self.error_distribution()

        wb = self.word_boundary_analysis()

        high, low = self.stratify_docs()

        self.plot_error_distribution(dist)
        self.plot_confusion()
        self.plot_word_boundaries(wb)

def main():
    analyzer = Phase2()
    analyzer.run(
        base_dir="../phase1/phase1_output"
    )

if __name__ == "__main__":
    main()