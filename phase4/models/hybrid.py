"""Hybrid OCR-correction system.

Architecture:

  1. Classical candidate generation: top-N sentence-level candidates from the
     classical beam (always includes identity).
  2. Neural refinement: byte-level Transformer scores each candidate via
     forced-decoding (length-normalized log P(candidate | noisy)).
  3. Calibrated fusion: weighted sum of LM, channel, and neural scores using
     weights ``(w_lm, w_channel, w_neural)``. Weights are tuned on VAL ONLY
     by ``calibrate_on_val`` and then frozen.
  4. Gating vs identity: per-token guard rails - any difference between the
     chosen prediction and identity is rejected unless the joint margin
     beats:
       - ``margin_proper_name`` for tokens that look like proper names,
       - ``margin_rare_word`` for tokens unseen in train (rare),
       - ``margin_default`` otherwise.

Performance notes:

* Every neural call (``score_target``) is expensive (~30-50 ms on A100 for
  ByT5-base). To keep ``calibrate_on_val`` tractable across the full
  ``HybridConfig.calibration_grid_*`` cross-product (default 81 cells), we
  precompute *all* weight-independent quantities once per sentence and then
  evaluate the grid purely on cached numbers. This makes calibration ~80-100x
  faster than re-running ``correct_sentence`` per cell.
* During regular prediction we additionally memoize per-sentence neural
  identity scores and per-(position, candidate token) margins so each
  sentence costs O(num_candidates + num_unique_token_changes) neural passes
  instead of O(num_candidates + 4 * num_tokens).
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from phase4.config import HybridConfig
from phase4.eval.metrics import is_proper_name, is_rare_word
from phase4.models.classical import (
    ClassicalCorrector,
    apply_case_pattern,
    detokenize,
    tokenize,
)
from phase4.models.transformer_seq2seq import ByteTransformerCorrector


def _length_norm(score: float, length: int, alpha: float = 0.6) -> float:
    return score / (((5 + length) / 6.0) ** alpha)


class HybridCorrector:
    def __init__(
        self,
        classical: ClassicalCorrector,
        neural: ByteTransformerCorrector,
        cfg: Optional[HybridConfig] = None,
        train_word_counts: Optional[Dict[str, int]] = None,
    ):
        self.classical = classical
        self.neural = neural
        self.cfg = cfg or HybridConfig()
        self.train_word_counts: Dict[str, int] = train_word_counts or {}
        self.calibration: Dict[str, float] = {
            "w_lm": float(self.cfg.fusion_w_lm),
            "w_channel": float(self.cfg.fusion_w_channel),
            "w_neural": float(self.cfg.fusion_w_neural),
            "margin_default": float(self.cfg.margin_default),
            "margin_proper_name": float(self.cfg.margin_proper_name),
            "margin_rare_word": float(self.cfg.margin_rare_word),
        }
        self.calibration_log: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    # Per-sentence preparation (weight-independent)
    # ------------------------------------------------------------------

    def _gather_candidates(self, sentence: str) -> List[Dict[str, object]]:
        topk = self.classical.correct_sentence_topk(
            sentence, k=self.cfg.num_classical_candidates
        )
        if not any(c["is_identity"] for c in topk):
            tokens = tokenize(sentence)
            identity_word_tokens = [t.lower() for t in tokens if t.isalpha()]
            identity_lm = self.classical.word_lm.log_prob_sentence(identity_word_tokens)
            topk.append(
                {
                    "prediction": sentence,
                    "tokens": tokens,
                    "lm_log_prob": identity_lm,
                    "channel_log_prob": 0.0,
                    "joint_score": self.classical.cfg.lambda_lm * identity_lm,
                    "delta_vs_identity": 0.0,
                    "is_identity": True,
                }
            )
        return topk

    def _prepare_sentence(self, sentence: str) -> Dict[str, object]:
        """Compute every quantity that does not depend on calibration weights.

        Returns a dict with:
        * ``tokens``         - the noisy sentence tokenized
        * ``candidates``     - list of dicts (lm/channel/nn log-probs + tokens)
        * ``id_nn_log_prob`` - neural ``score_target(sentence, sentence)``
        * ``token_margin``   - ``{(i, chosen_token): nn_log_prob_delta}``
                               for every (position, replacement) pair found
                               across any candidate (single-position swap).

        The data is large enough that we deliberately do NOT cache across
        sentences; the prediction loop calls this once per sentence and the
        calibration loop snapshots the dict for the whole sample.
        """
        tokens = tokenize(sentence)
        if not tokens:
            return {
                "tokens": [],
                "candidates": [],
                "id_nn_log_prob": 0.0,
                "token_margin": {},
            }
        raw_candidates = self._gather_candidates(sentence)
        id_nn = float(self.neural.score_target(sentence, sentence))
        candidates: List[Dict[str, object]] = []
        for cand in raw_candidates:
            nn = float(self.neural.score_target(sentence, str(cand["prediction"])))
            candidates.append(
                {
                    "prediction": str(cand["prediction"]),
                    "tokens": list(cand["tokens"]),
                    "lm_log_prob": float(cand["lm_log_prob"]),
                    "channel_log_prob": float(cand["channel_log_prob"]),
                    "is_identity": bool(cand.get("is_identity", False)),
                    "nn_log_prob": nn,
                }
            )
        token_margin: Dict[Tuple[int, str], float] = {}
        for cand in candidates:
            cand_tokens = cand["tokens"]
            n = min(len(tokens), len(cand_tokens))
            for i in range(n):
                noisy_tok = tokens[i]
                chosen_tok = cand_tokens[i]
                if chosen_tok == noisy_tok:
                    continue
                key = (i, chosen_tok)
                if key in token_margin:
                    continue
                new_tokens = list(tokens)
                new_tokens[i] = (
                    apply_case_pattern(noisy_tok, chosen_tok)
                    if noisy_tok.isalpha()
                    else chosen_tok
                )
                hyp_sent = detokenize(new_tokens)
                hyp_lp = float(self.neural.score_target(sentence, hyp_sent))
                token_margin[key] = hyp_lp - id_nn
        return {
            "tokens": tokens,
            "candidates": candidates,
            "id_nn_log_prob": id_nn,
            "token_margin": token_margin,
        }

    # ------------------------------------------------------------------
    # Apply calibration to prepared data (no neural calls below this line)
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse(parts: Dict[str, float], calibration: Dict[str, float]) -> float:
        return (
            calibration["w_lm"] * parts["lm_log_prob"]
            + calibration["w_channel"] * parts["channel_log_prob"]
            + calibration["w_neural"] * parts["nn_log_prob"]
        )

    def _required_margin(
        self, noisy_token: str, chosen_token: str, calibration: Dict[str, float]
    ) -> Tuple[float, str]:
        is_known = noisy_token.lower() in self.classical.lexicon
        is_proper = is_proper_name(noisy_token)
        is_rare = is_rare_word(noisy_token, self.train_word_counts)
        required = float(calibration["margin_default"])
        reason = "high_margin"
        if is_proper:
            required = max(required, float(calibration["margin_proper_name"]))
            reason = "proper_name_guard"
        if is_rare:
            required = max(required, float(calibration["margin_rare_word"]))
            reason = "rare_word_guard"
        if is_known and chosen_token.lower() != noisy_token.lower():
            required = max(required, float(calibration["margin_default"]))
        return required, reason

    def _decode_with_calibration(
        self,
        prepared: Dict[str, object],
        sentence: str,
        calibration: Dict[str, float],
    ) -> Tuple[str, List[Dict[str, object]]]:
        tokens = prepared["tokens"]  # type: ignore[assignment]
        candidates = prepared["candidates"]  # type: ignore[assignment]
        token_margin = prepared["token_margin"]  # type: ignore[assignment]
        if not tokens or not candidates:
            return sentence, [{"confidence": 1.0}]
        scored = []
        for cand in candidates:
            fused = self._fuse(cand, calibration)
            scored.append({**cand, "fused_score": fused})
        identity_state = next(
            (s for s in scored if s["is_identity"]),
            min(scored, key=lambda s: s["fused_score"]),
        )
        identity_score = float(identity_state["fused_score"])
        non_identity = [s for s in scored if not s["is_identity"]]
        if not non_identity:
            best = identity_state
            margin_global = 0.0
        else:
            best = max(non_identity, key=lambda s: s["fused_score"])
            margin_global = float(best["fused_score"]) - identity_score
        best_tokens = best["tokens"]
        n = min(len(tokens), len(best_tokens))
        out_tokens: List[str] = []
        logs: List[Dict[str, object]] = []
        any_change = False
        for i in range(n):
            noisy_tok = tokens[i]
            best_tok = best_tokens[i]
            if best_tok == noisy_tok:
                out_tokens.append(noisy_tok)
                logs.append(
                    {
                        "token": noisy_tok,
                        "identity_token": noisy_tok,
                        "classical_neural_token": best_tok,
                        "prediction": noisy_tok,
                        "gate_decision": "keep",
                        "gate_reason": "identity",
                        "confidence": 0.0,
                        "winner_source": "identity"
                        if best.get("is_identity")
                        else "classical+neural",
                    }
                )
                continue
            tok_nn_margin = float(token_margin.get((i, best_tok), 0.0))
            tok_margin = max(margin_global, tok_nn_margin)
            required, reason = self._required_margin(noisy_tok, best_tok, calibration)
            if tok_margin < required:
                chosen_tok, gate = noisy_tok, f"keep_below_margin_{reason}"
            else:
                chosen_tok, gate = best_tok, f"correct_{reason}"
            out_tokens.append(chosen_tok)
            if chosen_tok != noisy_tok:
                any_change = True
            logs.append(
                {
                    "token": noisy_tok,
                    "identity_token": noisy_tok,
                    "classical_neural_token": best_tok,
                    "prediction": chosen_tok,
                    "gate_decision": "keep" if chosen_tok == noisy_tok else "correct",
                    "gate_reason": gate,
                    "confidence": max(
                        0.0,
                        min(
                            1.0,
                            tok_margin
                            / max(1e-6, float(calibration["margin_default"])),
                        ),
                    ),
                    "winner_source": "classical+neural",
                }
            )
        for i in range(n, len(tokens)):
            out_tokens.append(tokens[i])
            logs.append(
                {
                    "token": tokens[i],
                    "prediction": tokens[i],
                    "gate_decision": "keep",
                    "gate_reason": "length_mismatch_keep",
                    "confidence": 0.0,
                }
            )
        if not any_change:
            return sentence, logs
        return detokenize(out_tokens), logs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct_sentence(
        self, sentence: str
    ) -> Tuple[str, List[Dict[str, object]]]:
        if not sentence:
            return sentence, [{"confidence": 1.0}]
        prepared = self._prepare_sentence(sentence)
        return self._decode_with_calibration(prepared, sentence, self.calibration)

    def calibrate_on_val(
        self,
        val_pairs: Sequence[Tuple[str, str]],
        max_pairs: int = 400,
    ) -> Dict[str, object]:
        """Grid-search calibration weights using cached per-sentence neural scores.

        Selection metric: mean character-error-rate. Tie-break: lower
        correction rate (prefers the more conservative model when CER is
        equal). The neural model is invoked only during the precompute pass.
        """
        from phase4.eval.metrics import cer

        if not val_pairs:
            return {"selected": dict(self.calibration), "grid_size": 0, "tested": 0}

        sample = list(val_pairs[:max_pairs])
        grid_w_neural = list(self.cfg.calibration_grid_w_neural)
        grid_w_lm = list(self.cfg.calibration_grid_w_lm)
        grid_w_channel = list(self.cfg.calibration_grid_w_channel)
        grid_margin = list(self.cfg.calibration_grid_margin)
        n_combos = (
            len(grid_w_neural) * len(grid_w_lm) * len(grid_w_channel) * len(grid_margin)
        )

        # 1. Precompute weight-independent quantities for every val sentence.
        print(
            f"[hybrid] calibrate: precomputing neural scores on {len(sample)} val "
            f"pairs (neural calls happen here only)...",
            flush=True,
        )
        t_pre = time.perf_counter()
        prepared_list: List[Tuple[str, str, Dict[str, object]]] = []
        for idx, (noisy, clean) in enumerate(sample, start=1):
            prepared = self._prepare_sentence(noisy)
            prepared_list.append((noisy, clean, prepared))
            if idx == 1 or idx == len(sample) or idx % max(1, len(sample) // 10) == 0:
                elapsed = time.perf_counter() - t_pre
                rate = elapsed / idx if idx > 0 else 0.0
                eta = rate * (len(sample) - idx)
                print(
                    f"[hybrid] calibrate: prepared {idx}/{len(sample)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        print(
            f"[hybrid] calibrate: precompute done in "
            f"{time.perf_counter() - t_pre:.1f}s; running grid (combos={n_combos}, "
            "no further neural calls)",
            flush=True,
        )

        # 2. Iterate the calibration grid purely on cached scores (CPU only).
        log: List[Dict[str, object]] = []
        best: Optional[Dict[str, object]] = None
        t0 = time.perf_counter()
        combos_done = 0
        for w_neural, w_lm, w_channel, margin in itertools.product(
            grid_w_neural, grid_w_lm, grid_w_channel, grid_margin
        ):
            cal = {
                "w_neural": float(w_neural),
                "w_lm": float(w_lm),
                "w_channel": float(w_channel),
                "margin_default": float(margin),
                "margin_proper_name": float(margin) * 1.5,
                "margin_rare_word": float(margin) * 1.3,
            }
            cer_sum = 0.0
            change_ct = 0
            for noisy, clean, prepared in prepared_list:
                pred, _logs = self._decode_with_calibration(prepared, noisy, cal)
                cer_sum += cer(clean, pred)
                if pred != noisy:
                    change_ct += 1
            mean_cer = cer_sum / max(1, len(prepared_list))
            change_rate = change_ct / max(1, len(prepared_list))
            entry = {
                "w_neural": float(w_neural),
                "w_lm": float(w_lm),
                "w_channel": float(w_channel),
                "margin": float(margin),
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
            if combos_done == 1 or combos_done == n_combos or combos_done % 9 == 0:
                elapsed = time.perf_counter() - t0
                rate = elapsed / max(1, combos_done)
                eta = rate * (n_combos - combos_done)
                print(
                    f"[hybrid] calibrate: combo {combos_done}/{n_combos} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s current={entry}",
                    flush=True,
                )
        if best is not None:
            self.calibration["w_neural"] = float(best["w_neural"])
            self.calibration["w_lm"] = float(best["w_lm"])
            self.calibration["w_channel"] = float(best["w_channel"])
            self.calibration["margin_default"] = float(best["margin"])
            self.calibration["margin_proper_name"] = float(best["margin"]) * 1.5
            self.calibration["margin_rare_word"] = float(best["margin"]) * 1.3
        self.calibration_log = log
        total = time.perf_counter() - t_pre
        print(
            f"[hybrid] calibrate done (precompute + grid) in {total:.1f}s "
            f"selected={best}",
            flush=True,
        )
        return {
            "selected": dict(self.calibration),
            "grid_size": n_combos,
            "tested": len(log),
            "log": log,
        }

    def save_calibration(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.cfg),
            "calibration": self.calibration,
            "calibration_log": self.calibration_log,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
