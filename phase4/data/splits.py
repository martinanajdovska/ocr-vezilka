"""Authoritative train/val/test document splits for phase4 and downstream phases."""

import hashlib
import json
from typing import Dict, Iterable, List, Optional, Set

DOC_GENRE: Dict[str, str] = {
    "dnevnik_po_mnogu_godini": "prose",
    "itar_pejo": "prose",
    "Pesni": "poetry",
    "Prezir": "prose",
    "Samecot": "prose",
    "sina_pesna": "poetry",
    "tajnopis": "prose",
    "Toj": "prose",
    "viktor_kupidon": "prose",
    "Забите на Ветрот - Томе Арсовски": "prose",
    "Провиденија": "prose",
    "Сите лица на смртта": "prose",
    "Современост 7": "prose",
    "Клуч за одредување на рибите и змииорките во Република Македонија": "prose",
}

SPLITS: Dict[str, list] = {
    "train": [
        "dnevnik_po_mnogu_godini",
        "itar_pejo",
        "Pesni",
        "Prezir",
        "sina_pesna",
        "tajnopis",
        "Toj",
        "viktor_kupidon",
        "Забите на Ветрот - Томе Арсовски",
        "Клуч за одредување на рибите и змииорките во Република Македонија",
    ],
    "val": [
        "Провиденија",
        "Samecot",
    ],
    "test": [
        "Сите лица на смртта",
        "Современост 7",
    ],
}

DOC_ID_ALIASES: Dict[str, str] = {
    "Клуч за одредување на рибите и змииорките во Република Македонија поправено": (
        "Клуч за одредување на рибите и змииорките во Република Македонија"
    ),
}

DOC_TO_SPLIT = {doc_id: split for split, docs in SPLITS.items() for doc_id in docs}



KFOLD_TEST_SETS: List[Set[str]] = [
    {"Сите лица на смртта", "Современост 7"},        # fold 0 = current test
    {"Провиденија", "Samecot"},                      # fold 1
    {"dnevnik_po_mnogu_godini", "itar_pejo"},        # fold 2
    {"Pesni", "sina_pesna"},                         # fold 3 (poetry-heavy)
    {"Toj", "viktor_kupidon", "tajnopis"},           # fold 4 (poetry-aware)
]



_OVERRIDE_DOC_TO_SPLIT: Optional[Dict[str, str]] = None


def docs_for_fold(fold_idx: int, k: Optional[int] = None) -> Dict[str, Set[str]]:
    """Return the train/val/test docs for fold ``fold_idx``.

    ``val`` is the next fold's test set (modular indexing); ``train`` is
    every other known doc minus those two. ``k`` defaults to
    ``len(KFOLD_TEST_SETS)``.
    """
    folds = KFOLD_TEST_SETS
    n = int(k) if k is not None else len(folds)
    if not (0 <= fold_idx < n):
        raise ValueError(f"fold_idx={fold_idx} out of range [0, {n})")
    test = set(folds[fold_idx])
    val = set(folds[(fold_idx + 1) % n])
    train = set(DOC_GENRE.keys()) - test - val
    return {"train": train, "val": val, "test": test}


def set_kfold_override(splits: Optional[Dict[str, Set[str]]]) -> None:
    """Toggle the module-level split override for the k-fold driver.

    Passing ``None`` clears the override (restores the canonical splits).
    The override flows through ``get_split``, ``docs_for_split``,
    ``split_manifest_hash`` and ``assert_disjoint_splits`` so downstream
    callers see a consistent fold-aware view.
    """
    global _OVERRIDE_DOC_TO_SPLIT
    if splits is None:
        _OVERRIDE_DOC_TO_SPLIT = None
        return
    flat: Dict[str, str] = {}
    for s in ("train", "val", "test"):
        for d in splits.get(s, set()):
            flat[d] = s
    _OVERRIDE_DOC_TO_SPLIT = flat


def _active_doc_to_split() -> Dict[str, str]:
    return _OVERRIDE_DOC_TO_SPLIT if _OVERRIDE_DOC_TO_SPLIT is not None else DOC_TO_SPLIT


def normalize_doc_id(doc_id: str) -> str:
    return DOC_ID_ALIASES.get(doc_id, doc_id)


def get_split(doc_id: str) -> str:
    doc_id = normalize_doc_id(doc_id)
    active = _active_doc_to_split()
    if doc_id not in active:
        raise ValueError(f"Unknown doc_id '{doc_id}'. Refusing unassigned fallback.")
    return active[doc_id]


def get_subset_domain(doc_id: str) -> str:
    """Linguistic subset: ``prose`` vs ``poetry``."""
    doc_id = normalize_doc_id(doc_id)
    if doc_id not in DOC_GENRE:
        raise ValueError(f"Unknown doc_id '{doc_id}' for subset_domain.")
    return DOC_GENRE[doc_id]


def assert_disjoint_splits() -> None:
    active = _active_doc_to_split()
    if active is DOC_TO_SPLIT:
        train = set(SPLITS["train"])
        val = set(SPLITS["val"])
        test = set(SPLITS["test"])
    else:
        train = {d for d, s in active.items() if s == "train"}
        val = {d for d, s in active.items() if s == "val"}
        test = {d for d, s in active.items() if s == "test"}
    if train & val or train & test or val & test:
        raise AssertionError("Split lists overlap.")


def assert_no_unknown_docs(doc_ids: Iterable[str]) -> None:
    unknown = sorted(set(doc_ids) - set(_active_doc_to_split()))
    if unknown:
        raise AssertionError(f"Unknown docs detected: {unknown}")


def split_manifest_hash() -> str:
    active = _active_doc_to_split()
    if active is DOC_TO_SPLIT:
        payload = json.dumps(SPLITS, ensure_ascii=False, sort_keys=True)
    else:
        # Reconstitute the lists-form so the hash matches the standard
        # ``SPLITS`` layout when an override is active.
        split_lists = {
            "train": sorted(d for d, s in active.items() if s == "train"),
            "val": sorted(d for d, s in active.items() if s == "val"),
            "test": sorted(d for d, s in active.items() if s == "test"),
        }
        payload = json.dumps(split_lists, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def docs_for_split(split: str) -> Set[str]:
    active = _active_doc_to_split()
    if active is DOC_TO_SPLIT:
        return set(SPLITS[split])
    return {d for d, s in active.items() if s == split}
