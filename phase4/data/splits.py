"""Authoritative train/val/test document splits for phase4 and downstream phases."""

import hashlib
import json
from typing import Dict, Iterable, Set

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


def normalize_doc_id(doc_id: str) -> str:
    return DOC_ID_ALIASES.get(doc_id, doc_id)


def get_split(doc_id: str) -> str:
    doc_id = normalize_doc_id(doc_id)
    if doc_id not in DOC_TO_SPLIT:
        raise ValueError(f"Unknown doc_id '{doc_id}'. Refusing unassigned fallback.")
    return DOC_TO_SPLIT[doc_id]


def get_subset_domain(doc_id: str) -> str:
    """Linguistic subset: ``prose`` vs ``poetry`` (replaces hash-based A/B)."""
    doc_id = normalize_doc_id(doc_id)
    if doc_id not in DOC_GENRE:
        raise ValueError(f"Unknown doc_id '{doc_id}' for subset_domain.")
    return DOC_GENRE[doc_id]


def assert_disjoint_splits() -> None:
    train = set(SPLITS["train"])
    val = set(SPLITS["val"])
    test = set(SPLITS["test"])
    if train & val or train & test or val & test:
        raise AssertionError("Split lists overlap.")


def assert_no_unknown_docs(doc_ids: Iterable[str]) -> None:
    unknown = sorted(set(doc_ids) - set(DOC_TO_SPLIT))
    if unknown:
        raise AssertionError(f"Unknown docs detected: {unknown}")


def split_manifest_hash() -> str:
    payload = json.dumps(SPLITS, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def docs_for_split(split: str) -> Set[str]:
    return set(SPLITS[split])
