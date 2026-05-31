"""
The HeadroomGate is the *first* filter in the neural inference path. It
estimates the noise level of each input sentence from three cheap proxies
and short-circuits the (expensive) neural correction when the input is
already clean enough that the model is statistically more likely to
*overcorrect* than improve it.

- ``z_ppl`` -- z-scored character-trigram log-perplexity against the
  train-only GT distribution. High values mean "this sentence does not
  look like the training language".
- ``oov_fraction`` -- fraction of surface tokens not present in the
  train lexicon.
- ``susp_per_100c`` -- count of suspicious characters per 100 chars,
  using the homoglyph cluster ``{с,c,е,e,о,o,а,a}`` glued to Cyrillic
  context plus the top-30 substitution sources from phase-2 stats.

A closed-form OLS estimator combines those three into a single predicted
input-CER. The threshold ``tau`` is selected on val by maximising
post-gating corpus CER reduction subject to ``overcorrection_rate <
max_overcorrection``.

State is saved as a single JSON file alongside the HF weights and loaded
by the ByT5Corrector at predict time via
:meth:`ByT5Corrector.attach_headroom_gate`.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from phase4.eval.metrics import cer
from phase4.models.classical import CharNGramLM, _load_top_confusions

# Homoglyph cluster -- the bare Latin look-alikes glued
# to Cyrillic words inside scanned OCR output are the clearest signal of
# residual noise.
_HOMOGLYPH_LATIN = set("ceoaCEOA")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_TOKEN_RE = re.compile(r"[\w'’ʼ\-]+", re.UNICODE)


def _is_macedonian_token(tok: str) -> bool:
    """Heuristic: token contains at least one Cyrillic character. Used to
    scope OOV / suspicious-char counts to text that actually should be
    Macedonian script
    """
    return bool(_CYRILLIC_RE.search(tok))


def _suspicious_char_count(s: str, top_src: Iterable[str]) -> int:
    """Count of homoglyph + top-substitution source characters appearing
    inside Cyrillic-context tokens. Pure-Cyrillic tokens with the Latin
    look-alikes are exactly the smoking gun.
    """
    top_src_set = set(top_src)
    count = 0
    for tok in _TOKEN_RE.findall(s):
        if not _is_macedonian_token(tok):
            continue
        for ch in tok:
            if ch in _HOMOGLYPH_LATIN:
                count += 1
            elif ch in top_src_set and not _is_macedonian_token(ch):
                count += 1
    return count


def _char_perplexity(lm: CharNGramLM, sentence: str) -> float:
    """Per-character log-perplexity (negated log-prob per char) of
    ``sentence`` under ``lm``. Lower = more like training text.
    """
    s = sentence.strip()
    if not s:
        return 0.0
    logp = lm.score_word(s)
    return -logp / max(1, len(s))


def _ols(features: List[List[float]], targets: List[float]) -> List[float]:
    """Tiny closed-form OLS with an intercept column prepended.

    Returns the coefficients (length = len(features[0]) + 1). The first
    entry is the intercept.
    """
    if not features:
        return [0.0]
    n = len(features)
    d = len(features[0])
    # Augmented design matrix with intercept column.
    X = [[1.0] + list(row) for row in features]
    XT_X = [[0.0] * (d + 1) for _ in range(d + 1)]
    XT_y = [0.0] * (d + 1)
    for i in range(n):
        for a in range(d + 1):
            xa = X[i][a]
            XT_y[a] += xa * targets[i]
            for b in range(a, d + 1):
                XT_X[a][b] += xa * X[i][b]
    for a in range(d + 1):
        for b in range(a):
            XT_X[a][b] = XT_X[b][a]
    # Solve via Gauss-Jordan with partial pivoting.
    aug = [row[:] + [XT_y[i]] for i, row in enumerate(XT_X)]
    m = d + 1
    for col in range(m):
        # Partial pivot.
        pivot = col
        for r in range(col + 1, m):
            if abs(aug[r][col]) > abs(aug[pivot][col]):
                pivot = r
        if abs(aug[pivot][col]) < 1e-12:
            # Singular system: fall back to zeros.
            return [0.0] * m
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        for k in range(col, m + 1):
            aug[col][k] /= pv
        for r in range(m):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for k in range(col, m + 1):
                aug[r][k] -= factor * aug[col][k]
    return [row[-1] for row in aug]


@dataclass
class HeadroomGateConfig:
    char_lm_order: int = 4
    char_lm_add_k: float = 0.1
    threshold: float = 0.05
    max_overcorrection: float = 0.08
    min_kept_fraction: float = 0.0
    threshold_grid: Tuple[float, ...] = (
        0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05,
        0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30,
    )
    weights: Tuple[float, float, float] = (0.55, 0.35, 0.10)


@dataclass
class HeadroomGate:
    config: HeadroomGateConfig = field(default_factory=HeadroomGateConfig)
    char_lm: Optional[CharNGramLM] = None
    lexicon: Optional[frozenset] = None
    top_src_chars: frozenset = field(default_factory=frozenset)
    # OLS coefficients [intercept, w_zppl, w_oov, w_susp].
    coefficients: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    train_ppl_mean: float = 0.0
    train_ppl_std: float = 1.0
    threshold: float = 0.05
    val_curve: List[Dict[str, float]] = field(default_factory=list)
    selected: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0

    def _features(self, sentence: str) -> Tuple[float, float, float]:
        """Return (z_ppl, oov_frac, susp_per_100c) for a single sentence."""
        if self.char_lm is None:
            ppl_raw = 0.0
        else:
            ppl_raw = _char_perplexity(self.char_lm, sentence)
        z_ppl = (
            (ppl_raw - self.train_ppl_mean) / max(self.train_ppl_std, 1e-6)
            if self.train_ppl_std > 0
            else 0.0
        )
        tokens = _TOKEN_RE.findall(sentence)
        oov_frac = 0.0
        if tokens and self.lexicon is not None:
            mac_toks = [t for t in tokens if _is_macedonian_token(t)]
            if mac_toks:
                oov_count = sum(1 for t in mac_toks if t.lower() not in self.lexicon)
                oov_frac = oov_count / max(1, len(mac_toks))
        susp = _suspicious_char_count(sentence, self.top_src_chars)
        susp_per_100c = 100.0 * susp / max(1, len(sentence))
        return float(z_ppl), float(oov_frac), float(susp_per_100c)

    def estimate_cer(self, sentence: str) -> float:
        """Predicted input CER from the OLS regression. Returns a value
        bounded to ``[0.0, 1.0]`` for robustness
        """
        feats = self._features(sentence)
        c = self.coefficients
        if len(c) < 4:
            return 0.0
        x = c[0] + c[1] * feats[0] + c[2] * feats[1] + c[3] * feats[2]
        return float(max(0.0, min(1.0, x)))

    def should_correct(self, sentence: str) -> bool:
        if not sentence:
            return False
        return self.estimate_cer(sentence) >= float(self.threshold)

    def fit(
        self,
        train_sentences: Sequence[str],
        lexicon: Optional[Iterable[str]] = None,
        phase2_train_only_dir: Optional[Path] = None,
        top_src_k: int = 30,
    ) -> None:
        """Fit the char-LM and z-score stats from train-only data.

        ``lexicon`` is the lowercased train surface lexicon (used for the
        OOV proxy). ``phase2_train_only_dir`` points at the directory
        produced by phase-2 train-only stats; the top-K substitution
        sources are pulled from ``error_confusion_counts.csv`` there.
        """
        self.char_lm = CharNGramLM(
            n=int(self.config.char_lm_order),
            add_k=float(self.config.char_lm_add_k),
        )
        # Fit on sentence-as-word so the model has sentence boundaries.
        # The CharNGramLM uses ``^...$`` wrappers internally.
        self.char_lm.fit(s for s in train_sentences if s)
        sample = [s for s in train_sentences if s]
        if sample:
            ppls = [_char_perplexity(self.char_lm, s) for s in sample]
            self.train_ppl_mean = float(statistics.fmean(ppls))
            self.train_ppl_std = float(
                statistics.pstdev(ppls) if len(ppls) > 1 else 1.0
            )
            if self.train_ppl_std < 1e-6:
                self.train_ppl_std = 1.0
        else:
            self.train_ppl_mean = 0.0
            self.train_ppl_std = 1.0
        if lexicon is not None:
            self.lexicon = frozenset(t.lower() for t in lexicon if t)
        else:
            self.lexicon = None
        # Top substitution sources (first column of the confusion matrix).
        wl = _load_top_confusions(phase2_train_only_dir, top_k=top_src_k) if phase2_train_only_dir else frozenset()
        self.top_src_chars = frozenset(src for src, _ in wl)
        self.n_train = len(sample)

    def fit_on_val(
        self,
        val_pairs: Sequence[Tuple[str, str]],
        val_preds: Optional[Sequence[str]] = None,
        threshold_grid: Optional[Sequence[float]] = None,
        max_overcorrection: Optional[float] = None,
    ) -> Dict[str, object]:
        """Fit OLS coefficients on val and sweep the threshold.

        ``val_pairs`` is the list of ``(noisy, clean)`` val pairs.
        ``val_preds`` is the ungated model prediction per record; it
        is required for the threshold sweep. If omitted, the threshold sweep
        falls back to a CER-improvement estimate based on the OLS prediction alone.
        """
        if not val_pairs:
            return {"selected": None, "curve": [], "n_val": 0}
        n = len(val_pairs)
        max_overcorr = float(
            max_overcorrection
            if max_overcorrection is not None
            else self.config.max_overcorrection
        )

        feats_list: List[List[float]] = []
        targets: List[float] = []
        noisy_cers: List[float] = []
        pred_cers: List[float] = []
        for k, (noisy, clean) in enumerate(val_pairs):
            z, o, s = self._features(noisy)
            feats_list.append([z, o, s])
            in_cer = cer(clean, noisy)
            targets.append(in_cer)
            noisy_cers.append(in_cer)
            pred_cers.append(cer(clean, val_preds[k]) if val_preds else float("nan"))

        self.coefficients = _ols(feats_list, targets)

        estimates = [
            max(
                0.0,
                min(
                    1.0,
                    self.coefficients[0]
                    + self.coefficients[1] * f[0]
                    + self.coefficients[2] * f[1]
                    + self.coefficients[3] * f[2],
                ),
            )
            for f in feats_list
        ]

        grid = list(
            threshold_grid
            if threshold_grid is not None
            else list(self.config.threshold_grid)
        )
        min_kept = float(getattr(self.config, "min_kept_fraction", 0.0) or 0.0)
        curve: List[Dict[str, float]] = []
        best: Optional[Dict[str, float]] = None
        best_relaxed: Optional[Dict[str, float]] = None
        baseline_pred_cer = (
            sum(pred_cers) / max(1, n) if val_preds else float("nan")
        )
        baseline_input_cer = sum(noisy_cers) / max(1, n)
        for tau in grid:
            kept = 0
            edited = 0
            harmful = 0
            useful = 0
            cer_sum = 0.0
            for k in range(n):
                run_model = estimates[k] >= tau
                if run_model:
                    kept += 1
                    if val_preds is not None and not math.isnan(pred_cers[k]):
                        if pred_cers[k] < noisy_cers[k]:
                            useful += 1
                            edited += 1
                        elif pred_cers[k] > noisy_cers[k]:
                            harmful += 1
                            edited += 1
                        elif val_preds[k] != val_pairs[k][0]:
                            edited += 1
                        cer_sum += pred_cers[k]
                    else:
                        cer_sum += noisy_cers[k]
                else:
                    cer_sum += noisy_cers[k]
            mean_cer = cer_sum / max(1, n)
            overcorr = harmful / edited if edited else 0.0
            useful_rate = useful / edited if edited else 0.0
            cer_reduction_rate = (baseline_input_cer - mean_cer) / max(
                baseline_input_cer, 1e-9
            )
            row = {
                "threshold": float(tau),
                "mean_cer": float(mean_cer),
                "cer_reduction_rate": float(cer_reduction_rate),
                "kept_fraction": float(kept / n),
                "overcorrection_rate": float(overcorr),
                "useful_correction_rate": float(useful_rate),
                "edited": int(edited),
            }
            curve.append(row)
            within_budget = (
                overcorr <= max_overcorr if val_preds is not None else True
            )
            if not within_budget:
                continue

            def _better(candidate: Dict[str, float], incumbent: Dict[str, float]) -> bool:
                if candidate["cer_reduction_rate"] > incumbent["cer_reduction_rate"] + 1e-6:
                    return True
                if abs(candidate["cer_reduction_rate"] - incumbent["cer_reduction_rate"]) <= 1e-6:
                    return candidate["threshold"] < incumbent["threshold"]
                return False

            if best_relaxed is None or _better(row, best_relaxed):
                best_relaxed = row
            if kept / n + 1e-9 < min_kept:
                continue
            if best is None or _better(row, best):
                best = row

        if best is None:
            best = best_relaxed
        if best is None:
            best = min(curve, key=lambda r: (r["mean_cer"], r["threshold"]))
        self.threshold = float(best["threshold"])
        self.val_curve = curve
        self.selected = dict(best)
        self.n_val = n
        return {
            "selection_version": 2,
            "selection_rule": (
                "maximize_cer_reduction subject to "
                f"overcorrection<={max_overcorr} and "
                f"kept_fraction>={min_kept}"
            ),
            "selected": best,
            "curve": curve,
            "baseline_input_cer": float(baseline_input_cer),
            "baseline_pred_cer": float(baseline_pred_cer)
            if not math.isnan(baseline_pred_cer) else None,
            "n_val": n,
            "coefficients": list(self.coefficients),
            "train_ppl_mean": float(self.train_ppl_mean),
            "train_ppl_std": float(self.train_ppl_std),
            "max_overcorrection": float(max_overcorr),
            "min_kept_fraction": float(min_kept),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {
            "version": 1,
            "config": {
                "char_lm_order": int(self.config.char_lm_order),
                "char_lm_add_k": float(self.config.char_lm_add_k),
                "threshold": float(self.config.threshold),
                "max_overcorrection": float(self.config.max_overcorrection),
                "min_kept_fraction": float(self.config.min_kept_fraction),
                "threshold_grid": [float(x) for x in self.config.threshold_grid],
                "weights": list(self.config.weights),
            },
            "coefficients": [float(x) for x in self.coefficients],
            "train_ppl_mean": float(self.train_ppl_mean),
            "train_ppl_std": float(self.train_ppl_std),
            "threshold": float(self.threshold),
            "lexicon": sorted(self.lexicon) if self.lexicon is not None else None,
            "top_src_chars": sorted(self.top_src_chars),
            "val_curve": list(self.val_curve),
            "selected": dict(self.selected),
            "n_train": int(self.n_train),
            "n_val": int(self.n_val),
        }
        if self.char_lm is not None:
            payload["char_lm"] = {
                "n": int(self.char_lm.n),
                "add_k": float(self.char_lm.add_k),
                "counts": {k: int(v) for k, v in self.char_lm.counts.items()},
                "context_counts": {k: int(v) for k, v in self.char_lm.context_counts.items()},
                "vocab": sorted(self.char_lm.vocab),
            }
        (path / "headroom_gate.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "HeadroomGate":
        path = Path(path)
        # Support both ``path/headroom_gate.json`` and a direct file path.
        if path.is_dir():
            json_path = path / "headroom_gate.json"
        else:
            json_path = path
        data = json.loads(json_path.read_text(encoding="utf-8"))
        cfg_raw = data.get("config", {})
        grid_raw = cfg_raw.get("threshold_grid")
        if isinstance(grid_raw, list) and grid_raw:
            threshold_grid = tuple(float(x) for x in grid_raw)
        else:
            threshold_grid = HeadroomGateConfig().threshold_grid
        cfg = HeadroomGateConfig(
            char_lm_order=int(cfg_raw.get("char_lm_order", 4)),
            char_lm_add_k=float(cfg_raw.get("char_lm_add_k", 0.1)),
            threshold=float(cfg_raw.get("threshold", 0.05)),
            max_overcorrection=float(cfg_raw.get("max_overcorrection", 0.08)),
            min_kept_fraction=float(cfg_raw.get("min_kept_fraction", 0.0)),
            threshold_grid=threshold_grid,
            weights=tuple(cfg_raw.get("weights", (0.55, 0.35, 0.10))),
        )
        gate = cls(config=cfg)
        gate.coefficients = [float(x) for x in data.get("coefficients", [])]
        gate.train_ppl_mean = float(data.get("train_ppl_mean", 0.0))
        gate.train_ppl_std = float(data.get("train_ppl_std", 1.0))
        gate.threshold = float(data.get("threshold", cfg.threshold))
        lex = data.get("lexicon")
        gate.lexicon = frozenset(lex) if lex is not None else None
        gate.top_src_chars = frozenset(data.get("top_src_chars", []))
        gate.val_curve = list(data.get("val_curve", []))
        gate.selected = dict(data.get("selected", {}))
        gate.n_train = int(data.get("n_train", 0))
        gate.n_val = int(data.get("n_val", 0))
        lm_payload = data.get("char_lm")
        if lm_payload:
            lm = CharNGramLM(
                n=int(lm_payload.get("n", cfg.char_lm_order)),
                add_k=float(lm_payload.get("add_k", cfg.char_lm_add_k)),
            )
            for k, v in lm_payload.get("counts", {}).items():
                lm.counts[k] = int(v)
            for k, v in lm_payload.get("context_counts", {}).items():
                lm.context_counts[k] = int(v)
            lm.vocab = set(lm_payload.get("vocab", []))
            gate.char_lm = lm
        return gate
