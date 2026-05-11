"""Classical OCR-correction baseline.

Architecture: dictionary + edit distance candidate generation + word n-gram
language model reranking via per-sentence beam search with a channel model.

- Dictionary and unigram counts: TRAIN clean sentences only.
- Word n-gram LM: modified Kneser-Ney smoothing (order configurable).
- Candidate generator: identity + lexicon entries within ``max_edit_distance``
  + a small confusion-aware perturbation set built from a TRAIN-only confusion
  table (diacritic/homoglyph swaps).
- Sentence decoder: beam search over per-token candidate sets, scored as
  ``lambda_lm * log P_lm(sentence) + lambda_channel * log P_channel(noisy | hyp)``.
- Final acceptance: keep identity unless the best beam beats the identity
  sentence by at least ``correction_margin`` joint log-prob; this prevents
  overcorrection of already-correct text.

The model also exposes ``correct_sentence_topk`` so the hybrid system can
rescore the top-k beam outputs with a neural model.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from phase4.config import ClassicalConfig


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: List[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def apply_case_pattern(source: str, target: str) -> str:
    if not source or not target:
        return target
    if source.isupper():
        return target.upper()
    if len(source) > 1 and source[0].isupper() and source[1:].islower():
        return target[:1].upper() + target[1:].lower()
    if source.islower():
        return target.lower()
    return target


def edit_distance(a: str, b: str, cap: Optional[int] = None) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    if cap is not None and abs(m - n) > cap:
        return cap + 1
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        row_min = cur[0]
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if cap is not None and row_min > cap:
            return cap + 1
        prev = cur
    return prev[n]


class CharNGramLM:
    def __init__(self, n: int = 4, add_k: float = 0.1):
        self.n = n
        self.add_k = add_k
        self.counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.vocab: set = set()

    def fit(self, words: Iterable[str]) -> None:
        for word in words:
            s = f"^{word}$"
            for i in range(len(s) - self.n + 1):
                gram = s[i : i + self.n]
                ctx = gram[:-1]
                self.counts[gram] += 1
                self.context_counts[ctx] += 1
                self.vocab.add(gram[-1])

    def score_word(self, word: str) -> float:
        s = f"^{word}$"
        if len(s) < self.n:
            return -5.0
        logp = 0.0
        v = max(1, len(self.vocab))
        for i in range(len(s) - self.n + 1):
            gram = s[i : i + self.n]
            ctx = gram[:-1]
            num = self.counts[gram] + self.add_k
            den = self.context_counts[ctx] + self.add_k * v
            logp += math.log(num / den)
        return logp



_BOS = "<bos>"
_EOS = "<eos>"
_UNK = "<unk>"


class KneserNeyWordLM:
    """Modified Kneser-Ney over word tokens.

    Implementation notes:
    - Uses a single absolute discount ``D``
    - Backs off to a continuation-probability unigram when n-gram contexts
      have not been seen.
    """

    def __init__(self, order: int = 3, discount: float = 0.75):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.discount = discount
        self.ngram_counts: Dict[int, Counter] = {k: Counter() for k in range(1, order + 1)}
        self.context_total: Dict[int, Counter] = {k: Counter() for k in range(2, order + 1)}
        self.context_unique_followers: Dict[int, Counter] = {
            k: defaultdict(set) for k in range(2, order + 1)
        }
        self.continuation_count: Counter = Counter()
        self.continuation_total: int = 0
        self.vocab: set = set()
        self._continuation_unigram: Dict[str, float] = {}

    def fit(self, sentences: Iterable[Sequence[str]]) -> None:
        for raw_tokens in sentences:
            tokens = list(raw_tokens)
            if not tokens:
                continue
            padded = [_BOS] * (self.order - 1) + tokens + [_EOS]
            for tok in padded:
                self.vocab.add(tok)
            for k in range(1, self.order + 1):
                for i in range(len(padded) - k + 1):
                    gram = tuple(padded[i : i + k])
                    self.ngram_counts[k][gram] += 1
                    if k >= 2:
                        ctx = gram[:-1]
                        self.context_total[k][ctx] += 1
                        self.context_unique_followers[k][ctx].add(gram[-1])
        bigram_counter = self.ngram_counts.get(2, Counter())
        prefixed_by: Dict[str, set] = defaultdict(set)
        for (w_prev, w), _c in bigram_counter.items():
            prefixed_by[w].add(w_prev)
        for w, prefixes in prefixed_by.items():
            self.continuation_count[w] = len(prefixes)
        self.continuation_total = sum(self.continuation_count.values())
        if self.continuation_total > 0:
            v = max(1, len(self.vocab))
            for w in self.vocab:
                num = self.continuation_count[w]
                self._continuation_unigram[w] = (num + 1.0) / (self.continuation_total + v)
        else:
            v = max(1, len(self.vocab))
            for w in self.vocab:
                self._continuation_unigram[w] = 1.0 / v

    def _continuation_prob(self, w: str) -> float:
        if w in self._continuation_unigram:
            return self._continuation_unigram[w]
        v = max(1, len(self.vocab))
        return 1.0 / (self.continuation_total + v)

    def _prob(self, gram: Tuple[str, ...]) -> float:
        k = len(gram)
        if k == 1:
            return self._continuation_prob(gram[0])
        ctx = gram[:-1]
        ctx_total = self.context_total[k].get(ctx, 0)
        if ctx_total == 0:
            return self._prob(gram[1:])
        c = self.ngram_counts[k].get(gram, 0)
        unique_followers = len(self.context_unique_followers[k].get(ctx, ()))
        first_term = max(c - self.discount, 0.0) / ctx_total
        lam = (self.discount * unique_followers) / ctx_total if ctx_total > 0 else 0.0
        return first_term + lam * self._prob(gram[1:])

    def log_prob_sentence(self, tokens: Sequence[str]) -> float:
        if not tokens:
            return 0.0
        padded = [_BOS] * (self.order - 1) + list(tokens) + [_EOS]
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            gram = tuple(padded[i - self.order + 1 : i + 1])
            p = self._prob(gram)
            if p <= 0.0:
                p = 1e-12
            total += math.log(p)
        return total

    def to_dict(self) -> Dict[str, object]:
        # Compact serialization: full ngram counts (for tiny corpora) + the
        # continuation table. For very large corpora consider top-k pruning
        # before saving.
        return {
            "order": self.order,
            "discount": self.discount,
            "ngram_counts": {
                str(k): {" ".join(g): int(c) for g, c in counter.items()}
                for k, counter in self.ngram_counts.items()
            },
            "context_total": {
                str(k): {" ".join(g): int(c) for g, c in counter.items()}
                for k, counter in self.context_total.items()
            },
            "context_unique_followers": {
                str(k): {" ".join(g): int(len(s)) for g, s in counter.items()}
                for k, counter in self.context_unique_followers.items()
            },
            "continuation_count": {w: int(c) for w, c in self.continuation_count.items()},
            "continuation_total": self.continuation_total,
            "vocab_size": len(self.vocab),
        }




class EditChannelModel:
    """Crude channel model based on per-class edit-operation log costs."""

    def __init__(self, sub_log_p: float = -2.0, ins_log_p: float = -3.0, del_log_p: float = -3.0):
        self.sub_log_p = sub_log_p
        self.ins_log_p = ins_log_p
        self.del_log_p = del_log_p

    def log_prob(self, noisy_token: str, hyp_token: str) -> float:
        if noisy_token == hyp_token:
            return 0.0
        d = edit_distance(noisy_token.lower(), hyp_token.lower())
        # Distribute the distance roughly evenly among substitutions
        return d * self.sub_log_p + 0.5 * abs(len(noisy_token) - len(hyp_token)) * (self.ins_log_p + self.del_log_p) / 2.0



def _load_diacritic_homoglyph_pairs(phase2_train_only_dir: Optional[Path]) -> List[Tuple[str, str]]:
    if phase2_train_only_dir is None:
        return []
    pairs: List[Tuple[str, str]] = []
    for fname in ("top_diacritic_confusions.csv", "top_homoglyph_confusions.csv"):
        path = phase2_train_only_dir / fname
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                src_idx = header.index("src") if "src" in header else 0
                tgt_idx = header.index("tgt") if "tgt" in header else 1
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) <= max(src_idx, tgt_idx):
                        continue
                    s, t = parts[src_idx], parts[tgt_idx]
                    if s and t and s != t and len(s) == 1 and len(t) == 1:
                        pairs.append((s, t))
                        pairs.append((t, s))
        except Exception:
            continue
    seen = set()
    out: List[Tuple[str, str]] = []
    for s, t in pairs:
        if (s, t) in seen:
            continue
        seen.add((s, t))
        out.append((s, t))
    return out



class ClassicalCorrector:
    def __init__(
        self,
        cfg: Optional[ClassicalConfig] = None,
        phase2_train_only_dir: Optional[Path] = None,
    ):
        self.cfg = cfg or ClassicalConfig()
        self.phase2_train_only_dir = phase2_train_only_dir
        self.lexicon: set = set()
        self.word_counts: Counter = Counter()
        self.lm = CharNGramLM(n=self.cfg.char_ngram_order, add_k=self.cfg.add_k_smoothing)
        self.word_lm = KneserNeyWordLM(
            order=self.cfg.word_lm_order,
            discount=self.cfg.word_lm_discount,
        )
        self.channel = EditChannelModel()
        self.by_len: Dict[int, List[str]] = {}
        self._token_cache: Dict[str, List[Tuple[str, float]]] = {}
        self.confusion_swaps: List[Tuple[str, str]] = _load_diacritic_homoglyph_pairs(
            phase2_train_only_dir
        )


    def fit(self, train_clean_sentences: Iterable[str]) -> None:
        sentences = [s for s in train_clean_sentences if s]
        all_tokens: List[List[str]] = []
        word_iter: List[str] = []
        for sentence in sentences:
            toks = tokenize(sentence)
            all_tokens.append([t for t in toks])
            for t in toks:
                if t.isalpha():
                    word_iter.append(t.lower())
        self.word_counts = Counter(word_iter)
        most_common = self.word_counts.most_common(120000)
        self.lexicon = {w for w, _ in most_common}
        by_len: Dict[int, List[str]] = {}
        for w, _ in most_common:
            by_len.setdefault(len(w), []).append(w)
        self.by_len = by_len
        self.lm.fit(self.lexicon)

        word_token_sentences: List[List[str]] = []
        for toks in all_tokens:
            ws = [t.lower() for t in toks if t.isalpha()]
            if ws:
                word_token_sentences.append(ws)
        self.word_lm.fit(word_token_sentences)


    def _candidates_for_token(self, token: str) -> List[str]:
        if token in self._token_cache:
            return [c for c, _ in self._token_cache[token]]
        if not token.isalpha():
            self._token_cache[token] = [(token, 0.0)]
            return [token]
        low = token.lower()
        pool: List[str] = []
        seen: set = {low}
        pool.append(low)
        if low in self.lexicon:
            pass
        for L in range(
            max(1, len(low) - self.cfg.max_edit_distance),
            len(low) + self.cfg.max_edit_distance + 1,
        ):
            for w in self.by_len.get(L, []):
                if w in seen:
                    continue
                d = edit_distance(low, w, cap=self.cfg.max_edit_distance)
                if d <= self.cfg.max_edit_distance:
                    pool.append(w)
                    seen.add(w)
                    if len(pool) >= self.cfg.candidate_pool_max:
                        break
            if len(pool) >= self.cfg.candidate_pool_max:
                break
        if self.confusion_swaps:
            for s, t in self.confusion_swaps:
                if s in low:
                    swapped = low.replace(s, t)
                    if swapped not in seen and swapped.isalpha():
                        d = edit_distance(low, swapped, cap=self.cfg.max_edit_distance)
                        if d <= self.cfg.max_edit_distance:
                            pool.append(swapped)
                            seen.add(swapped)
        scored: List[Tuple[str, float]] = []
        for cand in pool[: self.cfg.candidate_pool_max]:
            char_score = self.lm.score_word(cand)
            scored.append((cand, char_score))
        scored.sort(key=lambda x: -x[1])
        scored = scored[: self.cfg.top_k_candidates]
        if all(c != low for c, _ in scored):
            scored.append((low, self.lm.score_word(low)))
        self._token_cache[token] = scored
        return [c for c, _ in scored]

 
    def _score_sentence(
        self,
        noisy_tokens: List[str],
        hyp_word_tokens: List[str],
        hyp_full_tokens: List[str],
    ) -> Tuple[float, float]:
        lm_logp = self.word_lm.log_prob_sentence([t.lower() for t in hyp_word_tokens])
        ch_logp = 0.0
        for noisy, hyp in zip(noisy_tokens, hyp_full_tokens):
            ch_logp += self.channel.log_prob(noisy, hyp)
        return lm_logp, ch_logp

    def correct_sentence_topk(
        self,
        sentence: str,
        k: int = 1,
    ) -> List[Dict[str, object]]:
        tokens = tokenize(sentence)
        if not tokens:
            return [
                {
                    "prediction": sentence,
                    "tokens": [],
                    "lm_log_prob": 0.0,
                    "channel_log_prob": 0.0,
                    "joint_score": 0.0,
                    "is_identity": True,
                }
            ]
        beam: List[Dict[str, object]] = [
            {
                "tokens": [],
                "word_tokens": [],
                "lm_log_prob": 0.0,
                "channel_log_prob": 0.0,
                "joint_score": 0.0,
                "any_change": False,
            }
        ]
        for noisy in tokens:
            cands = self._candidates_for_token(noisy)
            if not cands:
                cands = [noisy.lower() if noisy.isalpha() else noisy]
            new_beam: List[Dict[str, object]] = []
            for state in beam:
                for cand in cands:
                    if noisy.isalpha():
                        token_out = apply_case_pattern(noisy, cand) if self.cfg.case_preserving else cand
                    else:
                        token_out = cand
                    new_word_tokens = list(state["word_tokens"])
                    if noisy.isalpha():
                        new_word_tokens.append(cand)
                    new_state = {
                        "tokens": list(state["tokens"]) + [token_out],
                        "word_tokens": new_word_tokens,
                        "lm_log_prob": 0.0,
                        "channel_log_prob": 0.0,
                        "joint_score": 0.0,
                        "any_change": state["any_change"] or token_out != noisy,
                    }
                    new_beam.append(new_state)
            for st in new_beam:
                lm_logp, ch_logp = self._score_sentence(
                    tokens[: len(st["tokens"])], st["word_tokens"], st["tokens"]
                )
                st["lm_log_prob"] = lm_logp
                st["channel_log_prob"] = ch_logp
                st["joint_score"] = (
                    self.cfg.lambda_lm * lm_logp + self.cfg.lambda_channel * ch_logp
                )
            new_beam.sort(key=lambda s: s["joint_score"], reverse=True)
            beam = new_beam[: self.cfg.beam_size]

        identity_word_tokens = [t.lower() for t in tokens if t.isalpha()]
        identity_lm = self.word_lm.log_prob_sentence(identity_word_tokens)
        identity_score = self.cfg.lambda_lm * identity_lm + self.cfg.lambda_channel * 0.0

        results: List[Dict[str, object]] = []
        seen_predictions: set = set()
        for state in beam[: max(1, k)]:
            prediction = detokenize(state["tokens"])
            if prediction in seen_predictions:
                continue
            seen_predictions.add(prediction)
            results.append(
                {
                    "prediction": prediction,
                    "tokens": state["tokens"],
                    "lm_log_prob": state["lm_log_prob"],
                    "channel_log_prob": state["channel_log_prob"],
                    "joint_score": state["joint_score"],
                    "delta_vs_identity": state["joint_score"] - identity_score,
                    "is_identity": prediction == sentence,
                }
            )
        identity_pred = sentence
        if identity_pred not in seen_predictions:
            results.append(
                {
                    "prediction": identity_pred,
                    "tokens": tokens,
                    "lm_log_prob": identity_lm,
                    "channel_log_prob": 0.0,
                    "joint_score": identity_score,
                    "delta_vs_identity": 0.0,
                    "is_identity": True,
                }
            )
        return results

    def correct_sentence(self, sentence: str) -> Tuple[str, List[Dict[str, object]]]:
        tokens = tokenize(sentence)
        topk = self.correct_sentence_topk(sentence, k=self.cfg.beam_size)
        identity = next((t for t in topk if t["is_identity"]), topk[-1])
        non_identity = [t for t in topk if not t["is_identity"]]
        if not non_identity:
            return sentence, [{"token": tok, "prediction": tok, "confidence": 0.0} for tok in tokens]
        best = max(non_identity, key=lambda t: t["joint_score"])
        delta = float(best["joint_score"]) - float(identity["joint_score"])
        chosen = best if delta >= self.cfg.correction_margin else identity
        details: List[Dict[str, object]] = []
        for noisy, hyp in zip(tokens, chosen["tokens"]):
            details.append(
                {
                    "token": noisy,
                    "prediction": hyp,
                    "confidence": max(0.0, delta),
                }
            )
        return chosen["prediction"], details


    def tune_thresholds(
        self,
        val_pairs: Sequence[Tuple[str, str]],
        correction_margin_grid: Optional[Sequence[float]] = None,
        lambda_lm_grid: Optional[Sequence[float]] = None,
        lambda_channel_grid: Optional[Sequence[float]] = None,
        max_pairs: int = 400,
    ) -> Dict[str, object]:
        """Grid-search ``correction_margin``, ``lambda_lm``, ``lambda_channel``
        on ``val_pairs`` and update ``self.cfg`` with the best triple.

        Selection metric: mean character-error-rate. Tie-break: lower
        correction rate (prefer the more conservative model when CERs match).
        Returns a tuning log suitable for JSON serialization.
        """
        from phase4.eval.metrics import cer

        margin_grid = list(correction_margin_grid or (1.0, 1.5, 2.0))
        lm_grid = list(lambda_lm_grid or (0.5, 1.0, 1.5))
        ch_grid = list(lambda_channel_grid or (0.3, 0.6, 1.0))

        if not val_pairs:
            return {"selected": asdict(self.cfg), "grid_size": 0, "tested": 0, "log": []}

        sample = list(val_pairs[:max_pairs])
        original_cfg = self.cfg
        log: List[Dict[str, object]] = []
        best: Optional[Dict[str, object]] = None
        n_combos = len(margin_grid) * len(lm_grid) * len(ch_grid)
        print(
            f"[classical] tune_thresholds: combos={n_combos} val_sample={len(sample)}",
            flush=True,
        )
        t0 = time.perf_counter()
        combos_done = 0
        for margin in margin_grid:
            for lm_w in lm_grid:
                for ch_w in ch_grid:
                    self.cfg = replace(
                        original_cfg,
                        correction_margin=float(margin),
                        lambda_lm=float(lm_w),
                        lambda_channel=float(ch_w),
                    )
                    cer_sum = 0.0
                    change_ct = 0
                    for noisy, clean in sample:
                        pred, _logs = self.correct_sentence(noisy)
                        cer_sum += cer(clean, pred)
                        if pred != noisy:
                            change_ct += 1
                    mean_cer = cer_sum / max(1, len(sample))
                    change_rate = change_ct / max(1, len(sample))
                    entry = {
                        "correction_margin": float(margin),
                        "lambda_lm": float(lm_w),
                        "lambda_channel": float(ch_w),
                        "mean_cer": mean_cer,
                        "change_rate": change_rate,
                    }
                    log.append(entry)
                    if best is None or (
                        mean_cer < float(best["mean_cer"]) - 1e-6
                        or (
                            abs(mean_cer - float(best["mean_cer"])) < 1e-6
                            and change_rate < float(best["change_rate"])
                        )
                    ):
                        best = entry
                    combos_done += 1
                    if combos_done == 1 or combos_done == n_combos or combos_done % 3 == 0:
                        elapsed = time.perf_counter() - t0
                        rate = elapsed / max(1, combos_done)
                        eta = rate * (n_combos - combos_done)
                        print(
                            f"[classical] tune_thresholds: progress {combos_done}/{n_combos} "
                            f"elapsed={elapsed:.1f}s eta={eta:.1f}s current={entry}",
                            flush=True,
                        )
        # Apply the best triple permanently.
        if best is not None:
            self.cfg = replace(
                original_cfg,
                correction_margin=float(best["correction_margin"]),
                lambda_lm=float(best["lambda_lm"]),
                lambda_channel=float(best["lambda_channel"]),
            )
        print(
            f"[classical] tune_thresholds: done in {time.perf_counter() - t0:.1f}s "
            f"selected={best}",
            flush=True,
        )
        return {
            "selected": asdict(self.cfg),
            "grid_size": n_combos,
            "tested": len(log),
            "log": log,
            "best": best,
            "n_val_sample": len(sample),
        }


    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.cfg),
            "lexicon": sorted(self.lexicon),
            "word_counts": dict(self.word_counts),
            "char_lm": {
                "n": self.lm.n,
                "add_k": self.lm.add_k,
            },
            "word_lm": self.word_lm.to_dict(),
            "confusion_swaps": self.confusion_swaps,
        }
        (path / "classical_model.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
