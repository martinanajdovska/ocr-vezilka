"""Macedonian-script constraints for neural OCR correction outputs.

ByT5 is trained on multilingual text and often emits Russian or other
Slavic Cyrillic letters (``ё``, ``й``, ``ы``, …) that are not part of the
official Macedonian alphabet.  Post-decode sanitization maps the common
confusions to Macedonian graphemes and drops any remaining non-Macedonian
Cyrillic so val/test predictions stay in the expected script.
"""

from __future__ import annotations

import unicodedata
from typing import FrozenSet, Tuple

# Official Macedonian alphabet (lower + upper).
MK_LETTERS: str = (
    "абвгдѓежзѕијклљмнњопрстќуфхцчџш"
    "АБВГДЃЕЖЗЅИЈКЛЉМНЊОПРСТЌУФХЦЧЏШ"
)
MK_LETTER_SET: FrozenSet[str] = frozenset(MK_LETTERS)

# Multi-character replacements first (order matters).
_MULTI_CHAR_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("Щ", "Ш"),
    ("щ", "ш"),
    ("Ю", "Ју"),
    ("ю", "ју"),
    ("Я", "Ја"),
    ("я", "ја"),
    ("Ї", "Ји"),
    ("ї", "ји"),
)

# Single-character Russian / Ukrainian / Church-Slavic → Macedonian.
_SINGLE_CHAR_TRANSLATION = str.maketrans(
    {
        "й": "ј",
        "Й": "Ј",
        "ё": "о",
        "Ё": "О",
        "ы": "и",
        "Ы": "И",
        "э": "е",
        "Э": "Е",
        "ъ": "",
        "Ъ": "",
        "ь": "",
        "Ь": "",
        "і": "и",  # Ukrainian
        "І": "И",
        "є": "е",
        "Є": "Е",
        "ґ": "г",
        "Ґ": "Г",
        # Archaic / non-MK Cyrillic often seen when byte constraints misfire.
        "ѣ": "е",
        "Ѣ": "Е",
        "ѳ": "ф",
        "Ѳ": "Ф",
        "ѵ": "и",
        "Ѵ": "И",
        "ѡ": "о",
        "Ѡ": "О",
        "ѧ": "а",
        "Ѧ": "А",
        "ѫ": "у",
        "Ѫ": "У",
        "ѯ": "кс",
        "Ѯ": "Кс",
    }
)

# Punctuation and symbols we keep in OCR book prose / metadata lines.
_ALLOWED_PUNCT: FrozenSet[str] = frozenset(
    ".,;:!?\"'«»„“”–—‐‑-()[]{}/*+=@#%&_<>|\\`~^"
    "№°§·…"
)


def sanitize_macedonian(text: str) -> str:
    """Map common non-MK Cyrillic confusions and drop other Cyrillic letters.

    Latin letters, digits, whitespace, and common punctuation are preserved so
    mixed lines (URLs, phone numbers, catalog codes) survive intact.
    """
    if not text:
        return text
    t = text
    for src, dst in _MULTI_CHAR_REPLACEMENTS:
        t = t.replace(src, dst)
    t = t.translate(_SINGLE_CHAR_TRANSLATION)
    out: list[str] = []
    for ch in t:
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            continue
        if ch in MK_LETTER_SET:
            out.append(ch)
            continue
        if ch.isspace():
            out.append(ch)
            continue
        if ch.isdigit():
            out.append(ch)
            continue
        if ch in _ALLOWED_PUNCT:
            out.append(ch)
            continue
        if ch.isascii() and ch.isalpha():
            out.append(ch)
            continue
        if cat.startswith("L"):
            # Any other alphabetic script (Russian Cyrillic leftovers,
            # Arabic, Armenian, …) after the substitution table — drop.
            continue
        if not ch.isprintable():
            continue
        out.append(ch)
    return "".join(out)


def sanitize_batch(texts: list[str]) -> list[str]:
    return [sanitize_macedonian(t) for t in texts]
