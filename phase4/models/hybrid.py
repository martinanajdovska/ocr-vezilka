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
from phase4.models.classical import ClassicalCorrector, detokenize, tokenize
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


    def _gather_candidates(self, sentence: str) -> List[Dict[str, object]]:
        topk = self.classical.correct_sentence_topk(
            sentence, k=self.cfg.num_classical_candidates
        )
        # Ensure identity always included.
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

    def _score_candidate(
        self, sentence: str, candidate: Dict[str, object]
    ) -> Dict[str, float]:
        nn_score = self.neural.score_target(sentence, str(candidate["prediction"]))
        return {
            "nn_log_prob": nn_score,
            "lm_log_prob": float(candidate["lm_log_prob"]),
            "channel_log_prob": float(candidate["channel_log_prob"]),
        }

    def _fused_score(self, parts: Dict[str, float]) -> float:
        return (
            self.calibration["w_lm"] * parts["lm_log_prob"]
            + self.calibration["w_channel"] * parts["channel_log_prob"]
            + self.calibration["w_neural"] * parts["nn_log_prob"]
        )

    def _token_gate(
        self,
        noisy_token: str,
        chosen_token: str,
        margin: float,
    ) -> Tuple[str, str]:
        if chosen_token == noisy_token:
            return chosen_token, "identity"
        is_known = noisy_token.lower() in self.classical.lexicon
        is_proper = is_proper_name(noisy_token)
        is_rare = is_rare_word(noisy_token, self.train_word_counts)
        required = self.calibration["margin_default"]
        reason = "high_margin"
        if is_proper:
            required = max(required, self.calibration["margin_proper_name"])
            reason = "proper_name_guard"
        if is_rare:
            required = max(required, self.calibration["margin_rare_word"])
            reason = "rare_word_guard"
        if is_known and chosen_token.lower() != noisy_token.lower():
            required = max(required, self.calibration["margin_default"])
        if margin < required:
            return noisy_token, f"keep_below_margin_{reason}"
        return chosen_token, f"correct_{reason}"

    def correct_sentence(
        self, sentence: str
    ) -> Tuple[str, List[Dict[str, object]]]:
        tokens = tokenize(sentence)
        if not tokens:
            return sentence, [{"confidence": 1.0}]
        candidates = self._gather_candidates(sentence)
        scored: List[Dict[str, object]] = []
        for cand in candidates:
            parts = self._score_candidate(sentence, cand)
            fused = self._fused_score(parts)
            scored.append(
                {
                    **cand,
                    **parts,
                    "fused_score": fused,
                    "winner_source": "identity" if cand["is_identity"] else "classical+neural",
                }
            )
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

        identity_tokens = identity_state["tokens"]
        best_tokens = best["tokens"]
        n = min(len(identity_tokens), len(best_tokens), len(tokens))
        out_tokens: List[str] = []
        logs: List[Dict[str, object]] = []
        any_change = False
        for i in range(n):
            noisy_tok = tokens[i]
            id_tok = identity_tokens[i]
            best_tok = best_tokens[i]
            chosen_tok, gate = self._token_gate(noisy_tok, best_tok, margin_global)
            out_tokens.append(chosen_tok)
            if chosen_tok != noisy_tok:
                any_change = True
            logs.append(
                {
                    "token": noisy_tok,
                    "identity_token": id_tok,
                    "classical_neural_token": best_tok,
                    "prediction": chosen_tok,
                    "gate_decision": "keep" if chosen_tok == noisy_tok else "correct",
                    "gate_reason": gate,
                    "confidence": max(0.0, margin_global),
                    "winner_source": str(best.get("winner_source", "")),
                }
            )
        # Carry over any remaining tokens (length difference between candidates)
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


    def calibrate_on_val(
        self,
        val_pairs: Sequence[Tuple[str, str]],
        max_pairs: int = 400,
    ) -> Dict[str, object]:
        """
        Selection metric is mean character-error-rate on the (subsampled) val
        pairs; ties broken by lower correction rate (prefers conservative
        models when CER is equal).
        """
        from phase4.eval.metrics import cer

        if not val_pairs:
            return {"selected": dict(self.calibration), "grid_size": 0, "tested": 0}

        sample = list(val_pairs[: max_pairs])
        best: Optional[Dict[str, object]] = None
        grid_w_neural = list(self.cfg.calibration_grid_w_neural)
        grid_margin = list(self.cfg.calibration_grid_margin)
        log: List[Dict[str, object]] = []
        t0 = time.perf_counter()
        n_combos = len(grid_w_neural) * len(grid_margin)
        print(
            f"[hybrid] calibrate: combos={n_combos} val_sample={len(sample)}",
            flush=True,
        )
        for w_neural, margin in itertools.product(grid_w_neural, grid_margin):
            self.calibration["w_neural"] = float(w_neural)
            self.calibration["margin_default"] = float(margin)
            self.calibration["margin_proper_name"] = float(margin) * 1.5
            self.calibration["margin_rare_word"] = float(margin) * 1.3
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
                "w_neural": float(w_neural),
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
        if best is not None:
            self.calibration["w_neural"] = float(best["w_neural"])
            self.calibration["margin_default"] = float(best["margin"])
            self.calibration["margin_proper_name"] = float(best["margin"]) * 1.5
            self.calibration["margin_rare_word"] = float(best["margin"]) * 1.3
        self.calibration_log = log
        print(
            f"[hybrid] calibrate done in {time.perf_counter() - t0:.1f}s "
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
