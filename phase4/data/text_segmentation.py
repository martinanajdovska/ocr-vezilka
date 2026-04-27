"""Shared sentence segmentation and chunking utilities.

A single source of truth for how text is split into sentences and chunks. Used
by both the manifest builder (training pair construction) and the inference
runner so that the model never sees a different segmentation policy at train
vs serve time.

Policies:
- Sentence boundary regex matches Macedonian/Cyrillic punctuation by default
  (period, exclamation, question, ellipsis), with a lookbehind to avoid
  splitting on common abbreviations is intentionally NOT added; downstream
  alignment is robust to over-splitting.
- Chunking: paragraphs (blank-line separated) when present, otherwise a fixed
  window of K sentences (with stride K, i.e. non-overlapping).
"""

from __future__ import annotations

import re
from typing import List


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[\.!\?…])\s+")
_PARAGRAPH_BOUNDARY_RE = re.compile(r"\n\s*\n")


def split_sentences(text: str) -> List[str]:
    """Sentence-segment a text. Empty/whitespace-only sentences are dropped."""
    if not text:
        return []
    flat = re.sub(r"\s+", " ", text.strip())
    if not flat:
        return []
    parts = _SENTENCE_BOUNDARY_RE.split(flat)
    return [p.strip() for p in parts if p and p.strip()]


def split_paragraphs(text: str) -> List[str]:
    """Paragraph-segment a text using one or more blank lines as the boundary."""
    if not text:
        return []
    parts = _PARAGRAPH_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_sentences(
    sentences: List[str],
    paragraphs: List[str] | None = None,
    window: int = 20,
    min_paragraphs: int = 8,
) -> List[List[int]]:
    """Group sentence indices into chunks.

    If a usable paragraph segmentation is provided (>= ``min_paragraphs`` parts),
    sentences are partitioned by paragraph boundaries via cumulative length
    matching. Otherwise we fall back to a fixed window of ``window`` sentences
    per chunk (non-overlapping).

    Returns a list of chunks, where each chunk is a list of sentence indices
    into ``sentences``.
    """
    n = len(sentences)
    if n == 0:
        return []

    if paragraphs is not None and len(paragraphs) >= min_paragraphs:
        para_lengths = [len(p) for p in paragraphs]
        para_cum = []
        running = 0
        for length in para_lengths:
            running += length
            para_cum.append(running)
        sent_lengths = [len(s) for s in sentences]
        chunks: List[List[int]] = []
        current: List[int] = []
        running_chars = 0
        para_idx = 0
        for i, length in enumerate(sent_lengths):
            current.append(i)
            running_chars += length
            while para_idx < len(para_cum) and running_chars >= para_cum[para_idx] - 1:
                if current:
                    chunks.append(current)
                    current = []
                para_idx += 1
                if para_idx >= len(para_cum):
                    break
        if current:
            chunks.append(current)
        if chunks:
            return chunks

    return [list(range(i, min(i + window, n))) for i in range(0, n, window)]
