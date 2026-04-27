import hashlib
import json
from typing import Dict, Iterable, Set


SPLITS: Dict[str, list] = {
    "train": [
        "dnevnik_po_mnogu_godini",
        "itar_pejo",
        "Pesni",
        "Prezir",
        "Samecot",
        "sina_pesna",
        "tajnopis",
        "Toj",
        "viktor_kupidon",
    ],
    "val": [
        "Забите на Ветрот - Томе Арсовски",
        "Провиденија",
    ],
    "test": [
        "Сите лица на смртта",
        "Современост 7",
    ],
}


DOC_TO_SPLIT = {doc_id: split for split, docs in SPLITS.items() for doc_id in docs}


def get_split(doc_id: str) -> str:
    if doc_id not in DOC_TO_SPLIT:
        raise ValueError(f"Unknown doc_id '{doc_id}'. Refusing unassigned fallback.")
    return DOC_TO_SPLIT[doc_id]


def get_subset_domain(doc_id: str) -> str:
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    return "A" if int(digest[:8], 16) % 2 == 0 else "B"


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

