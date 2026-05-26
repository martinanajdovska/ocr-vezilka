from __future__ import annotations

import contextlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union  # noqa: F401

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from phase4.config import TransformerConfig
from phase4.eval.metrics import cer, is_proper_name, is_rare_word
from phase4.models.classical import (
    _load_confusion_sub_log_probs,
    _load_top_confusions,
)
from phase4.models.macedonian_script import sanitize_batch, sanitize_macedonian

# Pre-compiled tokenizer for proper-name / rare-word lookups. Matches the
# same word boundary heuristic used by the classical gate so the two
# correctors agree on what a "token" is for purposes of margin boosts.
_TOKEN_RE = re.compile(r"[\w'’ʼ\-]+", re.UNICODE)


def _import_hf():
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer 

    return T5ForConditionalGeneration, ByT5Tokenizer


def _select_device() -> torch.device:
    """
    Override via ``PHASE4_NEURAL_DEVICE={cpu,mps,cuda,auto}`` env var (set by
    ``--neural-device`` on the runner). ``auto`` (default) falls through to
    cuda > mps > cpu.
    """
    override = os.environ.get("PHASE4_NEURAL_DEVICE", "").strip().lower()
    if override == "cpu":
        return torch.device("cpu")
    if override == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[byt5] CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    if override == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[byt5] MPS requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cuda_amp_autocast_dtype(
    cfg: TransformerConfig, device: torch.device
) -> tuple[bool, Optional[torch.dtype]]:
    """CUDA mixed precision: prefer bfloat16 (Ampere+); else float16 + GradScaler."""
    if not (cfg.use_amp and device.type == "cuda"):
        return False, None
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except (AttributeError, RuntimeError):
            pass


def _training_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            state["mps"] = torch.mps.get_rng_state()
        except (AttributeError, RuntimeError):
            pass
    return state


def _restore_training_rng_state(bundle: Dict[str, Any]) -> None:
    rs = bundle.get("rng_state")
    if not isinstance(rs, dict):
        return
    pr = rs.get("python_random")
    if pr is not None:
        random.setstate(pr)
    tr = rs.get("torch")
    if tr is not None:
        torch.set_rng_state(tr)
    if torch.cuda.is_available():
        cu = rs.get("cuda")
        if cu is not None:
            torch.cuda.set_rng_state_all(cu)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        mp = rs.get("mps")
        if mp is not None:
            try:
                torch.mps.set_rng_state(mp)
            except (AttributeError, RuntimeError):
                pass


def _atomic_torch_save(obj: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _torch_load_checkpoint(path: Path, map_location: Union[str, torch.device]) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location=map_location)


def _pairs_with_context(
    pairs: Sequence[Tuple[str, str]], sep: str
) -> List[Tuple[str, str]]:
    listed = list(pairs)
    out: List[Tuple[str, str]] = []
    for i, (src, tgt) in enumerate(listed):
        prev_s = listed[i - 1][0] if i > 0 else ""
        next_s = listed[i + 1][0] if i < len(listed) - 1 else ""
        ctx = sep.join(p for p in (prev_s, src, next_s) if p)
        out.append((ctx, tgt))
    return out


def _confusion_swap_table(
    sub_log_probs: Dict[Tuple[str, str], float],
) -> List[Tuple[str, str, float]]:
    rows: List[Tuple[str, str, float]] = []
    for (a, b), lp in sub_log_probs.items():
        if a != b:
            rows.append((a, b, math.exp(lp)))
    return rows


def _min_new_tokens_value(
    cfg: TransformerConfig, input_bytes: int, max_new: int
) -> int:
    """Floor ``min_new_tokens`` at a fraction of the *input* byte length so the
    beam cannot collapse to a 5-byte stub when the model is uncertain. The
    untrained ByT5 head has been observed to terminate after a handful of
    tokens on hard inputs, which produced predictions that were ~1/3 the input
    length on average; this prevents that failure mode without bounding the
    upper length (``max_new_tokens`` still does).
    """
    ratio = float(getattr(cfg, "min_new_token_ratio", 0.0) or 0.0)
    if ratio <= 0 or input_bytes <= 0:
        return 0
    floor = int(input_bytes * ratio)
    return max(1, min(floor, max(0, max_new - 1)))


def _min_new_tokens_for_batch(
    cfg: TransformerConfig,
    sentences: Sequence[str],
    non_empty_idx: Sequence[int],
    max_new: int,
) -> int:
    """Return a single ``min_new_tokens`` value for a batched ``generate``.

    HF generate expects a scalar, but inputs in a batch may have very different
    lengths. Use the *shortest* non-empty input as the floor so we never force
    long generation on a short sentence, but still keep the floor high enough
    to stop catastrophic truncation.
    """
    if not non_empty_idx:
        return 0
    min_in_bytes = min(len(sentences[i].encode("utf-8")) for i in non_empty_idx)
    return _min_new_tokens_value(cfg, min_in_bytes, max_new)


class _ByT5Dataset(Dataset):
    """Source/target pairs for ByT5 fine-tuning.

    ByT5 tokenizer maps each UTF-8 byte to ``token_id = byte + 3`` (slots
    0/1/2 are reserved for ``pad``, ``eos``, ``unk``). We rely on the
    tokenizer's own truncation so character/byte boundaries are respected,
    matching the cfg ``max_input_bytes`` / ``max_target_bytes`` policy.
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        tokenizer,
        max_input_bytes: int,
        max_target_bytes: int,
        identity_pair_ratio: float = 0.0,
        task_prefix: str = "",
        confusion_swaps: Optional[List[Tuple[str, str, float]]] = None,
        training_confusion_noise_prob: float = 0.0,
        rng: Optional[random.Random] = None,
    ):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_input_bytes = max_input_bytes
        self.max_target_bytes = max_target_bytes
        self.identity_pair_ratio = float(identity_pair_ratio)
        self.task_prefix = task_prefix or ""
        self.confusion_swaps = confusion_swaps or []
        self.training_confusion_noise_prob = float(training_confusion_noise_prob)
        self.rng = rng or random.Random(0)

    def __len__(self) -> int:
        return len(self.pairs)

    def _apply_confusion_noise(self, text: str) -> str:
        if not self.confusion_swaps or not text:
            return text
        chars = list(text)
        alpha_idx = [i for i, c in enumerate(chars) if c.isalpha()]
        if not alpha_idx:
            return text
        pos = self.rng.choice(alpha_idx)
        ch = chars[pos]
        candidates = [(a, b, w) for a, b, w in self.confusion_swaps if a == ch]
        if not candidates:
            return text
        total = sum(w for _, _, w in candidates)
        r = self.rng.random() * total
        acc = 0.0
        for a, b, w in candidates:
            acc += w
            if r <= acc:
                chars[pos] = b
                break
        return "".join(chars)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src, tgt = self.pairs[idx]
        # Identity-pair augmentation: with probability ``identity_pair_ratio``
        # replace ``(noisy, clean)`` with ``(clean, clean)``. This curbs the
        # neural model's tendency to *always* edit the input(overcorrection on already-correct sentences). 
        # Applied at train time only (val/test datasets pass ``identity_pair_ratio=0``).
        if self.identity_pair_ratio > 0 and self.rng.random() < self.identity_pair_ratio:
            src = tgt
        if (
            self.training_confusion_noise_prob > 0
            and self.rng.random() < self.training_confusion_noise_prob
        ):
            src = self._apply_confusion_noise(src)
        if self.task_prefix and not src.startswith(self.task_prefix):
            src = self.task_prefix + src
        enc = self.tokenizer(
            src,
            max_length=self.max_input_bytes,
            truncation=True,
            return_tensors=None,
        )
        dec = self.tokenizer(
            text_target=tgt,
            max_length=self.max_target_bytes,
            truncation=True,
            return_tensors=None,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(dec["input_ids"], dtype=torch.long),
        }


def _collate_byt5(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        seqs = [b[k] for b in batch]
        if k == "labels":
            # T5 convention: -100 on padded positions so they are ignored by the
            # CrossEntropyLoss inside the model.
            padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=-100)
        elif k == "attention_mask":
            padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        else:
            padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
        out[k] = padded
    return out


class ByT5Corrector:
    def __init__(self, cfg: Optional[TransformerConfig] = None, seed: int = 42):
        self.cfg = cfg or TransformerConfig()
        self.seed = seed
        self.device = _select_device()
        self.tokenizer = None
        self.model = None
        self.training_metrics: Dict[str, object] = {}
        self.checkpoint_path: Optional[Path] = None
        # Filled in by ``calibrate_gate_on_val`` (or restored from
        # ``transformer_meta.json``). Takes precedence over
        # ``cfg.gate_log_prob_margin`` whenever the gate runs.
        self.tuned_gate_margin: Optional[float] = None
        self.tuned_per_edit_margin: Optional[float] = None
        self.gate_calibration: Optional[Dict[str, object]] = None
        # temperature scaling. ``1.0`` means logits are unmodified;
        # ``calibrate_temperature`` sets this to the val-NLL optimum (typically
        # in [1.1, 1.5] for a slightly over-confident ByT5 fine-tune). Every
        # log_softmax in ``score_targets_batch`` / per-edit gate divides
        # logits by this value before softmaxing.
        self.temperature: float = 1.0
        self.temperature_calibration: Optional[Dict[str, object]] = None
        self.confusion_whitelist: Optional[frozenset] = None
        self.train_word_counts: Optional[Dict[str, int]] = None
        self.lexicon: Optional[set] = None
        self.headroom: Optional[object] = None  # HeadroomGate; typed weakly to avoid import cycle


    def _load_pretrained(self, source: Optional[Path] = None) -> None:
        T5ForConditionalGeneration, ByT5Tokenizer = _import_hf()
        src = str(source) if source is not None else self.cfg.pretrained_model
        print(f"[byt5] loading {src} on device={self.device}", flush=True)
        self.tokenizer = ByT5Tokenizer.from_pretrained(src)
        self.model = T5ForConditionalGeneration.from_pretrained(src).to(self.device)
        self._configure_model_for_training()
        self._maybe_torch_compile()

    def _maybe_torch_compile(self) -> None:
        """Optionally torch.compile the model.

        Opt-in via ``PHASE4_TORCH_COMPILE=1`` so deterministic timing tables
        are unaffected by default.

        ``PHASE4_TORCH_COMPILE_MODE`` overrides the compile mode (default,
        reduce-overhead, max-autotune). ``default`` is the safest with
        dynamic ByT5 padding lengths; ``reduce-overhead`` uses CUDA graphs
        and recompiles on every new shape, which thrashes on variable-length
        batches.
        """
        if self.model is None:
            return
        if os.environ.get("PHASE4_TORCH_COMPILE", "").strip().lower() not in (
            "1", "true", "yes", "on"
        ):
            return
        if self.device.type != "cuda":
            print("[byt5] PHASE4_TORCH_COMPILE set but device is not CUDA; skipping.", flush=True)
            return
        mode = os.environ.get("PHASE4_TORCH_COMPILE_MODE", "default").strip() or "default"
        try:
            print(
                f"[byt5] torch.compile(mode={mode!r}, dynamic=True) ... "
                "(first batch is slow while the graph is captured)",
                flush=True,
            )
            self.model = torch.compile(self.model, mode=mode, dynamic=True)
        except Exception as exc:
            print(f"[byt5] torch.compile failed ({exc}); continuing without it.", flush=True)

    def _configure_model_for_training(self) -> None:
        """Enable gradient checkpointing / disable KV cache for training memory.

        Must be called before each ``fit`` and re-disabled before generation
        (which we do in ``correct_sentence`` / ``score_target`` if needed).
        Gradient checkpointing requires ``use_cache=False`` (T5 raises an
        error otherwise), and we prefer the non-reentrant variant to silence
        torch deprecation warnings + play nicely with AMP.
        """
        if self.model is None:
            return
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
        if bool(getattr(self.cfg, "gradient_checkpointing", False)):
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                try:
                    self.model.gradient_checkpointing_enable()
                except (AttributeError, ValueError):
                    pass
            except (AttributeError, ValueError):
                pass

    def _filter_pairs(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], int]:
        kept: List[Tuple[str, str]] = []
        skipped = 0
        # emit a one-time warning when input or target
        # length exceeds ``max_input_bytes`` / ``max_target_bytes``.
        max_in = int(getattr(self.cfg, "max_input_bytes", 0) or 0)
        max_tg = int(getattr(self.cfg, "max_target_bytes", 0) or 0)
        n_truncated_in = 0
        n_truncated_tg = 0
        for src, tgt in pairs:
            if not src or not tgt:
                skipped += 1
                continue
            if max_in and len(src.encode("utf-8")) > max_in:
                n_truncated_in += 1
            if max_tg and len(tgt.encode("utf-8")) > max_tg:
                n_truncated_tg += 1
            kept.append((src, tgt))
        if n_truncated_in or n_truncated_tg:
            print(
                f"[byt5] WARN: truncated {n_truncated_in} input / "
                f"{n_truncated_tg} target pairs to max_input_bytes={max_in} "
                f"/ max_target_bytes={max_tg}. "
                "Consider bumping those config values if the figure is non-trivial.",
                flush=True,
            )
        return kept, skipped


    def fit(
        self,
        train_pairs: Sequence[Tuple[str, str]],
        val_pairs: Sequence[Tuple[str, str]],
        stage: str = "train",
        resume_from: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        resume: bool = True,
    ) -> Dict[str, object]:
        """Fine-tune ByT5 on (noisy, clean) sentence pairs.

        ``stage`` selects the per-stage epoch budget:
        - ``train`` (default) -> ``cfg.max_epochs`` epochs, single stage.
        - ``pretrain`` -> ``cfg.pretrain_epochs`` epochs.
        - ``finetune`` -> ``cfg.finetune_epochs`` epochs (load weights from
          ``resume_from`` when starting finetune in a new process, if no
          mid-stage checkpoint exists).

        When ``checkpoint_dir`` is set, an epoch-end checkpoint is written to
        ``checkpoint_dir / f"byt5_resume_{stage}.pt"`` so training can resume
        after an interrupt. On successful completion that file is removed.
        For ``stage="pretrain"``, weights are also saved under
        ``checkpoint_dir / "pretrain_hf"`` for the finetune stage.

        If a training checkpoint exists and ``resume`` is true, it takes
        precedence over ``resume_from`` for initial weights.
        """
        if not train_pairs:
            raise ValueError("No training pairs for ByT5 corrector.")

        stage = stage.lower()
        if stage == "pretrain":
            n_epochs = int(self.cfg.pretrain_epochs)
        elif stage == "finetune":
            n_epochs = int(self.cfg.finetune_epochs)
        else:
            n_epochs = int(self.cfg.max_epochs)

        ckpt_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
        train_ckpt_path = (
            (ckpt_root / f"byt5_resume_{stage}.pt") if ckpt_root is not None else None
        )

        resume_bundle: Optional[Dict[str, Any]] = None
        if resume and train_ckpt_path is not None and train_ckpt_path.exists():
            resume_bundle = _torch_load_checkpoint(train_ckpt_path, map_location="cpu")
            if (
                resume_bundle.get("stage") != stage
                or int(resume_bundle.get("n_epochs", -1)) != n_epochs
            ):
                print(
                    "[byt5] ignoring stale training checkpoint "
                    f"(stage/n_epochs mismatch): {train_ckpt_path}",
                    flush=True,
                )
                resume_bundle = None
            else:
                print(
                    f"[byt5] resuming training from checkpoint -> {train_ckpt_path}",
                    flush=True,
                )

        if self.device.type == "cuda" and not self.cfg.deterministic:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass
        if resume_bundle is None:
            _set_seed(self.seed)

        if resume_bundle is not None:
            T5ForConditionalGeneration, ByT5Tokenizer = _import_hf()
            self.tokenizer = ByT5Tokenizer.from_pretrained(self.cfg.pretrained_model)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.cfg.pretrained_model
            ).to(self.device)
            sd = resume_bundle["model_state_dict"]
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in sd.items()}, strict=True
            )
            self._configure_model_for_training()
            _restore_training_rng_state(resume_bundle)
        elif self.model is None:
            if resume_from is not None and Path(resume_from).exists():
                self._load_pretrained(Path(resume_from))
            else:
                self._load_pretrained(None)
        elif resume_from is not None and Path(resume_from).exists():
            self._load_pretrained(Path(resume_from))
        else:
            self._configure_model_for_training()

        train_kept, train_skipped = self._filter_pairs(train_pairs)
        val_kept, val_skipped = self._filter_pairs(val_pairs)
        if not train_kept:
            raise ValueError("All training pairs were empty.")

        if resume_bundle is not None:
            ct = resume_bundle.get("kept_train_pairs")
            cv = resume_bundle.get("kept_val_pairs")
            if ct is not None and cv is not None and (
                int(ct) != len(train_kept) or int(cv) != len(val_kept)
            ):
                print(
                    "[byt5] training checkpoint train/val size mismatch vs current data; "
                    "starting this stage from scratch (new seed init).",
                    flush=True,
                )
                resume_bundle = None
                _set_seed(self.seed)
                self._load_pretrained(
                    Path(resume_from)
                    if resume_from is not None and Path(resume_from).exists()
                    else None
                )

        n_params = sum(p.numel() for p in self.model.parameters())
        eff_batch = self.cfg.batch_size * max(1, self.cfg.gradient_accumulation_steps)
        print(
            f"[byt5] stage={stage} epochs={n_epochs} "
            f"train={len(train_kept)} (skip {train_skipped}) "
            f"val={len(val_kept)} (skip {val_skipped}) "
            f"params={n_params:,} effective_batch={eff_batch}",
            flush=True,
        )

        if self.cfg.use_window_context:
            sep = self.cfg.window_context_sep
            train_kept = _pairs_with_context(train_kept, sep)
            val_kept = _pairs_with_context(val_kept, sep)

        sub_lp = _load_confusion_sub_log_probs(
            Path(__file__).resolve().parents[2] / "phase2" / "phase2_output_train_only"
        )
        confusion_swaps = _confusion_swap_table(sub_lp)

        rng = random.Random(self.seed)
        train_ds = _ByT5Dataset(
            train_kept,
            self.tokenizer,
            self.cfg.max_input_bytes,
            self.cfg.max_target_bytes,
            identity_pair_ratio=float(self.cfg.identity_pair_ratio),
            task_prefix=self.cfg.task_prefix,
            confusion_swaps=confusion_swaps,
            training_confusion_noise_prob=float(
                self.cfg.training_confusion_noise_prob
            ),
            rng=rng,
        )
        val_ds = _ByT5Dataset(
            val_kept,
            self.tokenizer,
            self.cfg.max_input_bytes,
            self.cfg.max_target_bytes,
            identity_pair_ratio=0.0,
            task_prefix=self.cfg.task_prefix,
            confusion_swaps=[],
            training_confusion_noise_prob=0.0,
            rng=random.Random(self.seed + 1),
        )
        pad_id = int(self.tokenizer.pad_token_id)

        def _collate(batch):
            return _collate_byt5(batch, pad_id)

        loader_workers = max(0, int(self.cfg.dataloader_num_workers))
        pin_memory = bool(self.cfg.pin_memory and self.device.type == "cuda")
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=_collate,
            num_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=loader_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=loader_workers,
            pin_memory=pin_memory,
            persistent_workers=loader_workers > 0,
        )

        learning_rate = float(self.cfg.learning_rate)
        warmup_ratio = float(self.cfg.warmup_ratio)
        if stage == "finetune":
            learning_rate = float(self.cfg.learning_rate) * float(
                self.cfg.finetune_lr_scale
            )
            warmup_ratio = float(self.cfg.finetune_warmup_ratio)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        accum = max(1, int(self.cfg.gradient_accumulation_steps))
        steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
        total_steps = max(1, steps_per_epoch * n_epochs)
        warmup_steps = max(1, int(round(total_steps * warmup_ratio)))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.05, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=float(self.cfg.label_smoothing),
        )
        early_metric = str(self.cfg.early_stop_metric).lower()

        use_amp, amp_dtype = _cuda_amp_autocast_dtype(self.cfg, self.device)
        use_grad_scaler = bool(use_amp and amp_dtype == torch.float16)
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
        if use_amp and amp_dtype is not None:
            print(
                f"[byt5] CUDA AMP: autocast dtype={amp_dtype}, "
                f"GradScaler={'on' if use_grad_scaler else 'off'}",
                flush=True,
            )

        best_val_loss = float("inf")
        best_val_cer = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience = 0
        history: List[Dict[str, float]] = []
        global_step = 0
        start_epoch = 0

        if resume_bundle is not None:
            optimizer.load_state_dict(resume_bundle["optimizer_state_dict"])
            scheduler.load_state_dict(resume_bundle["scheduler_state_dict"])
            scaler.load_state_dict(resume_bundle["scaler_state_dict"])
            best_val_loss = float(resume_bundle["best_val_loss"])
            best_val_cer = float(resume_bundle.get("best_val_cer", best_val_cer))
            bs = resume_bundle.get("best_state_dict")
            if isinstance(bs, dict):
                best_state = {k: v.clone() for k, v in bs.items()}
            patience = int(resume_bundle["patience"])
            history = list(resume_bundle["history"])
            global_step = int(resume_bundle["global_step"])
            start_epoch = int(resume_bundle["next_epoch"])

        t_all = time.perf_counter()

        def _write_train_checkpoint(next_epoch: int) -> None:
            if train_ckpt_path is None:
                return
            bundle: Dict[str, Any] = {
                "format_version": 1,
                "stage": stage,
                "n_epochs": n_epochs,
                "next_epoch": next_epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "best_val_cer": best_val_cer,
                "best_state_dict": best_state,
                "patience": patience,
                "history": history,
                "model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_state": _training_rng_state(),
                "kept_train_pairs": len(train_kept),
                "kept_val_pairs": len(val_kept),
            }
            _atomic_torch_save(bundle, train_ckpt_path)
            print(f"[byt5] wrote training checkpoint -> {train_ckpt_path}", flush=True)

        log_every = max(1, steps_per_epoch // 7)
        early_stop = False
        for epoch in range(start_epoch, n_epochs):
            t_ep = time.perf_counter()
            self.model.train()
            running_loss = 0.0
            n_batches = 0
            optimizer.zero_grad(set_to_none=True)
            train_opt_steps = 0
            for batch_idx, batch in enumerate(train_loader):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                amp_ctx = (
                    torch.amp.autocast("cuda", dtype=amp_dtype)
                    if use_amp and amp_dtype is not None
                    else contextlib.nullcontext()
                )
                with amp_ctx:
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    labels = batch["labels"]
                    flat_logits = logits.reshape(-1, logits.size(-1))
                    flat_labels = labels.reshape(-1)
                    raw_loss = loss_fn(flat_logits, flat_labels)
                    loss = raw_loss / accum
                batch_loss = float(raw_loss.detach().item())
                scaler.scale(loss).backward()
                running_loss += float(loss.item()) * accum
                n_batches += 1
                if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    train_opt_steps += 1
                    if (
                        train_opt_steps == 1
                        or train_opt_steps == steps_per_epoch
                        or train_opt_steps % log_every == 0
                    ):
                        print(
                            f"[byt5] {stage} epoch {epoch + 1}/{n_epochs} "
                            f"opt_step {train_opt_steps}/{steps_per_epoch} "
                            f"(global_step={global_step}) batch_loss={batch_loss:.4f} "
                            f"epoch_elapsed={time.perf_counter() - t_ep:.1f}s",
                            flush=True,
                        )
            train_loss = running_loss / max(1, n_batches)
            if not math.isfinite(train_loss):
                print(
                    f"[byt5] Non-finite training loss at {stage} epoch {epoch + 1}: "
                    f"train_loss={train_loss}",
                    flush=True,
                )

            print(
                f"[byt5] {stage} epoch {epoch + 1}/{n_epochs}: "
                f"train steps done; computing val_loss on {len(val_loader)} batches...",
                flush=True,
            )
            val_loss = self._eval_loss(val_loader)
            n_mini = min(len(val_kept), max(1, int(self.cfg.eval_cer_pairs)))
            print(
                f"[byt5] {stage} epoch {epoch + 1}/{n_epochs}: "
                f"computing mini-CER on {n_mini} val sentences "
                f"(beam={int(self.cfg.eval_cer_beam)}) — can take several minutes...",
                flush=True,
            )
            val_cer = self._eval_mini_cer(
                val_kept,
                max_pairs=int(self.cfg.eval_cer_pairs),
                beam_size=int(self.cfg.eval_cer_beam),
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "stage": stage,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_cer": val_cer,
                    "epoch_time_s": time.perf_counter() - t_ep,
                }
            )


            if math.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
            if math.isfinite(val_cer) and val_cer < best_val_cer:
                pass  # ``best_val_cer`` is updated below when it drives selection
            if early_metric == "val_cer":
                improved = val_cer < best_val_cer - 1e-4
                if improved:
                    best_val_cer = val_cer
            else:
                improved = val_loss < best_val_loss - 1e-4
            if improved:
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
            print(
                f"[byt5] {stage} epoch {epoch + 1}/{n_epochs}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_cer={val_cer:.4f} "
                f"best_val_cer={best_val_cer:.4f} patience={patience}/{self.cfg.early_stopping_patience} "
                f"time={time.perf_counter() - t_ep:.1f}s",
                flush=True,
            )

            _write_train_checkpoint(epoch + 1)

            if patience >= self.cfg.early_stopping_patience:
                print("[byt5] early stopping", flush=True)
                early_stop = True
                break

        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}, strict=True
            )
        self.training_metrics = {
            "stage": stage,
            "epochs_run": len(history),
            "best_val_loss": best_val_loss,
            "best_val_cer": best_val_cer,
            "early_stop_metric": early_metric,
            "skipped_train_pairs": train_skipped,
            "skipped_val_pairs": val_skipped,
            "kept_train_pairs": len(train_kept),
            "kept_val_pairs": len(val_kept),
            "total_train_seconds": time.perf_counter() - t_all,
            "history": history,
            "n_params": n_params,
            "device": str(self.device),
            "effective_batch_size": eff_batch,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "resumed": resume_bundle is not None,
            "early_stopped": early_stop,
        }
        print(
            f"[byt5] {stage} finished in {self.training_metrics['total_train_seconds']:.1f}s "
            f"best_val_cer={best_val_cer:.4f} best_val_loss={best_val_loss:.4f} "
            f"early_stop_metric={early_metric}",
            flush=True,
        )

        if train_ckpt_path is not None and train_ckpt_path.exists():
            train_ckpt_path.unlink()
            print(f"[byt5] removed completed training checkpoint -> {train_ckpt_path}", flush=True)

        if ckpt_root is not None and stage == "pretrain":
            pre_dir = ckpt_root / "pretrain_hf"
            pre_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(pre_dir)
            self.tokenizer.save_pretrained(pre_dir)
            print(f"[byt5] wrote pretrain snapshot for finetune/resume -> {pre_dir}", flush=True)

        return self.training_metrics

    def _eval_mini_cer(
        self,
        val_pairs: Sequence[Tuple[str, str]],
        max_pairs: int,
        beam_size: int,
    ) -> float:
        if not val_pairs:
            return float("inf")
        sample = list(val_pairs[: max(1, max_pairs)])
        cer_sum = 0.0
        self.model.eval()
        t0 = time.perf_counter()
        gen_batch = max(1, int(getattr(self.cfg, "eval_gen_batch_size", 16)))

        order = sorted(range(len(sample)), key=lambda k: len(sample[k][0]))
        try:
            done = 0
            for start in range(0, len(order), gen_batch):
                idxs = order[start : start + gen_batch]
                noisy = [sample[i][0] for i in idxs]
                clean = [sample[i][1] for i in idxs]
                preds = self.correct_batch(noisy, beam_size=beam_size)
                for ref, pred in zip(clean, preds):
                    cer_sum += cer(ref, pred)
                done += len(idxs)
                elapsed = time.perf_counter() - t0
                rate = elapsed / max(1, done)
                eta = rate * (len(order) - done)
                print(
                    f"[byt5] mini-CER: {done}/{len(order)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s "
                    f"(batch={len(idxs)})",
                    flush=True,
                )
        finally:
            # ``correct_sentence`` toggles use_cache=True and disables
            # gradient checkpointing for fast generation. Restore the
            # training-time configuration before the next training epoch
            # so we keep the activation-memory savings.
            self._configure_model_for_training()
        return cer_sum / max(1, len(sample))

    def _eval_loss(self, loader: DataLoader) -> float:
        """Compute mean validation loss with a CPU fallback on NaN.

        ByT5 on MPS occasionally produces ``NaN`` losses during evaluation due
        to numerical precision in T5's attention/RMSNorm kernels (this is a
        well-known PyTorch MPS issue with the T5 family). Train-time loss is
        rarely affected because gradients still update the weights, but a NaN
        validation loss kills early stopping. We mitigate that here:

        1. Try the batch on the training device.
        2. If the loss is NaN, re-run *that single batch* on CPU using a
           shadow copy of the model. The re-run is exact (fp32) and the model
           is left on its original device; only the per-batch eval is moved.
        """
        self.model.eval()
        losses: List[float] = []
        cpu_model = None
        use_amp, amp_dtype = _cuda_amp_autocast_dtype(self.cfg, self.device)
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if use_amp and amp_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.no_grad():
            for batch in loader:
                gpu_batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }
                with amp_ctx:
                    outputs = self.model(**gpu_batch)
                loss_val = float(outputs.loss.item())
                if not math.isfinite(loss_val) and self.device.type == "mps":
                    if cpu_model is None:
                        T5ForConditionalGeneration, _ = _import_hf()
                        cpu_model = T5ForConditionalGeneration.from_pretrained(
                            self.cfg.pretrained_model
                        )
                        cpu_model.load_state_dict(
                            {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                        )
                        cpu_model.eval()
                    cpu_batch = {k: v.detach().cpu() for k, v in batch.items()}
                    cpu_out = cpu_model(**cpu_batch)
                    loss_val = float(cpu_out.loss.item())
                if math.isfinite(loss_val):
                    losses.append(loss_val)
        if not losses:
            return float("inf")
        return sum(losses) / len(losses)


    def _configure_model_for_inference(self) -> None:
        """Disable gradient checkpointing and re-enable KV cache for fast generation."""
        if self.model is None:
            return
        try:
            self.model.gradient_checkpointing_disable()
        except (AttributeError, ValueError):
            pass
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True

    def _sanitize_decoded(self, texts: Sequence[str]) -> List[str]:
        """Optional post-decode Macedonian script filter (see ``macedonian_script``)."""
        if not bool(getattr(self.cfg, "sanitize_macedonian_output", True)):
            return list(texts)
        return sanitize_batch(list(texts))

    def _maybe_apply_headroom_skip(
        self,
        sentences: Sequence[str],
        apply_gate: Optional[bool],
    ) -> Tuple[List[bool], List[Dict[str, object]]]:
        """Decide which sentences to short-circuit before generate.

        Returns (skip_mask, per_sentence_headroom_logs). When
        ``apply_gate=False`` (calibration paths) the headroom is forced
        off so calibration always sees the model's behaviour on the full
        distribution. When ``self.headroom is None`` no sentence is
        skipped and estimates are ``None``.
        """
        n = len(sentences)
        if apply_gate is False or self.headroom is None:
            return [False] * n, [
                {"headroom_estimate_cer": None, "headroom_skipped": False}
                for _ in range(n)
            ]
        skip: List[bool] = []
        logs: List[Dict[str, object]] = []
        threshold = float(getattr(self.headroom, "threshold", 0.0) or 0.0)
        for s in sentences:
            if not s:
                skip.append(False)
                logs.append({
                    "headroom_estimate_cer": None,
                    "headroom_skipped": False,
                })
                continue
            try:
                est = float(self.headroom.estimate_cer(s)) 
            except Exception:
                est = float("nan")
            if math.isfinite(est) and est < threshold:
                skip.append(True)
                logs.append({
                    "headroom_estimate_cer": est,
                    "headroom_skipped": True,
                })
            else:
                skip.append(False)
                logs.append({
                    "headroom_estimate_cer": est if math.isfinite(est) else None,
                    "headroom_skipped": False,
                })
        return skip, logs

    def correct_batch(
        self,
        sentences: Sequence[str],
        beam_size: Optional[int] = None,
        apply_gate: Optional[bool] = None,
    ) -> List[str]:
        """Batched generation. ~10-20x faster than calling correct_sentence
        in a loop on CUDA, because all sequences share one ``model.generate``
        call (KV cache + matmul utilization scale with batch).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not fitted.")
        if not sentences:
            return []
        if self.model.training:
            self.model.eval()
        self._configure_model_for_inference()
        # headroom gate -- skipped sentences pass through as identity
        # without ever touching the GPU. Calibration paths force this off.
        skip_mask, _hr_logs = self._maybe_apply_headroom_skip(sentences, apply_gate)
        prefixed: List[str] = []
        for i, s in enumerate(sentences):
            if not s or skip_mask[i]:
                prefixed.append("")
                continue
            src = s
            if self.cfg.task_prefix and not src.startswith(self.cfg.task_prefix):
                src = self.cfg.task_prefix + src
            prefixed.append(src)
        non_empty_idx = [i for i, s in enumerate(prefixed) if s]
        if not non_empty_idx:
            return list(sentences)
        enc = self.tokenizer(
            [prefixed[i] for i in non_empty_idx],
            max_length=self.cfg.max_input_bytes,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        max_in_bytes = max(
            len(s.encode("utf-8")) for s in (sentences[i] for i in non_empty_idx)
        )
        max_new = min(
            int(self.cfg.max_target_bytes),
            max(32, int(max_in_bytes * 1.3) + 16),
        )
        beams = int(beam_size if beam_size is not None else self.cfg.beam_size)
        gen_kwargs = {
            "max_new_tokens": max_new,
            "min_new_tokens": _min_new_tokens_for_batch(
                self.cfg, sentences, non_empty_idx, max_new
            ),
            "num_beams": beams,
            "length_penalty": float(self.cfg.length_penalty),
            "early_stopping": True,
        }

        if beams > 1:
            gen_kwargs["length_penalty"] = float(self.cfg.length_penalty)
            gen_kwargs["early_stopping"] = True
        if self.cfg.no_repeat_ngram_size:
            gen_kwargs["no_repeat_ngram_size"] = int(self.cfg.no_repeat_ngram_size)
        use_amp, amp_dtype = _cuda_amp_autocast_dtype(self.cfg, self.device)
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if use_amp and amp_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.no_grad(), amp_ctx:
            out = self.model.generate(**enc, **gen_kwargs)
        decoded = self._sanitize_decoded(
            self.tokenizer.batch_decode(out, skip_special_tokens=True)
        )
        result = list(sentences)
        for j, idx in enumerate(non_empty_idx):
            result[idx] = decoded[j]
        # Neural log-prob gate: drop edits that the model itself rates lower
        # than the original noisy input. Operates on the full batch (including
        # the no-op empty sentences) so indices stay aligned.
        gated, _decisions = self._maybe_apply_gate(
            list(sentences), result, enabled_override=apply_gate
        )
        return gated

    def correct_batch_with_logs(
        self,
        sentences: Sequence[str],
        beam_size: Optional[int] = None,
        apply_gate: Optional[bool] = None,
    ) -> List[Tuple[str, List[Dict[str, object]]]]:
        """Same return shape as ``correct_sentence`` but for a whole batch.

        Sequence-level beam scores are returned via
        ``out.sequences_scores`` (one per *batch* item, already
        length-normalised by HF using ``length_penalty``); we squash to a
        sigmoid-style confidence in [0, 1] consistent with the per-sentence
        path. Empty inputs short-circuit with ``confidence=1.0``.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not fitted.")
        if not sentences:
            return []
        if self.model.training:
            self.model.eval()
        self._configure_model_for_inference()
        # headroom gate -- skipped sentences pass through as identity
        # and get tagged with ``gate_decision="headroom_skip"`` in their log.
        skip_mask, hr_logs = self._maybe_apply_headroom_skip(sentences, apply_gate)
        prefixed: List[str] = []
        for i, s in enumerate(sentences):
            if not s or skip_mask[i]:
                prefixed.append("")
                continue
            src = s
            if self.cfg.task_prefix and not src.startswith(self.cfg.task_prefix):
                src = self.cfg.task_prefix + src
            prefixed.append(src)
        non_empty_idx = [i for i, s in enumerate(prefixed) if s]
        # Pre-populate per-sentence results with identity + headroom log so
        # short-circuited (empty or headroom-skipped) sentences still get
        # rich log rows.
        results: List[Tuple[str, List[Dict[str, object]]]] = []
        for i, s in enumerate(sentences):
            base_log: Dict[str, object] = {
                "confidence": 1.0,
                "score": 0.0,
                "gate_decision": (
                    "headroom_skip" if skip_mask[i] else "identity"
                ),
            }
            base_log.update(hr_logs[i])
            results.append((s, [base_log]))
        if not non_empty_idx:
            return results
        enc = self.tokenizer(
            [prefixed[i] for i in non_empty_idx],
            max_length=self.cfg.max_input_bytes,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        max_in_bytes = max(
            len(sentences[i].encode("utf-8")) for i in non_empty_idx
        )
        max_new = min(
            int(self.cfg.max_target_bytes),
            max(32, int(max_in_bytes * 1.3) + 16),
        )
        beams = int(beam_size if beam_size is not None else self.cfg.beam_size)
        gen_kwargs = {
            "max_new_tokens": max_new,
            "min_new_tokens": _min_new_tokens_for_batch(
                self.cfg, sentences, non_empty_idx, max_new
            ),
            "num_beams": beams,
            "length_penalty": float(self.cfg.length_penalty),
            "early_stopping": True,
            "return_dict_in_generate": True,
            "output_scores": beams > 1,
        }
        if beams > 1:
            gen_kwargs["length_penalty"] = float(self.cfg.length_penalty)
            gen_kwargs["early_stopping"] = True
        if self.cfg.no_repeat_ngram_size:
            gen_kwargs["no_repeat_ngram_size"] = int(self.cfg.no_repeat_ngram_size)
        use_amp, amp_dtype = _cuda_amp_autocast_dtype(self.cfg, self.device)
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if use_amp and amp_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.no_grad(), amp_ctx:
            out = self.model.generate(**enc, **gen_kwargs)
        sequences = out.sequences
        decoded = self._sanitize_decoded(
            self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        )
        seq_scores = None
        if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            seq_scores = out.sequences_scores.detach().to("cpu").tolist()

        raw_preds: List[str] = list(sentences)
        per_log: List[Dict[str, object]] = [
            {"confidence": 1.0, "score": 0.0, "gate_decision": "identity"}
            for _ in range(len(sentences))
        ]
        for j, idx in enumerate(non_empty_idx):
            text = decoded[j]
            raw_preds[idx] = text
            if seq_scores is not None and j < len(seq_scores):
                score = float(seq_scores[j])
                confidence = float(min(1.0, math.exp(min(0.0, score))))
            else:
                # Greedy path has no per-sequence score; use a neutral
                # confidence so calibration bins still get populated.
                score = 0.0
                confidence = 0.5
            per_log[idx] = {
                "confidence": confidence,
                "score": score,
                "gate_decision": "identity",
            }
        # Apply the neural log-prob gate over the full batch (so indices
        # stay aligned). Empty inputs short-circuit inside the gate.
        gated, decisions = self._maybe_apply_gate(
            list(sentences), raw_preds, enabled_override=apply_gate
        )
        for idx in range(len(sentences)):
            decision = decisions[idx] if idx < len(decisions) else {}
            log_row: Dict[str, object] = dict(per_log[idx])
            log_row.update(decision)
            # Stamp headroom metadata first, then let the per-edit gate
            # decision override ``gate_decision`` for sentences that were
            # actually fed through the model. Headroom-skipped sentences
            # keep ``gate_decision="headroom_skip"`` (their pred == noisy).
            log_row.update(hr_logs[idx])
            if skip_mask[idx]:
                log_row["gate_decision"] = "headroom_skip"
                gated[idx] = sentences[idx]
            results[idx] = (gated[idx], [log_row])
        return results

    def correct_sentence(
        self,
        sentence: str,
        beam_size: Optional[int] = None,
        apply_gate: Optional[bool] = None,
    ) -> Tuple[str, List[Dict[str, object]]]:
        if not sentence:
            return sentence, [{"confidence": 1.0}]
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not fitted.")
        if self.model.training:
            self.model.eval()
        self._configure_model_for_inference()
        # single-sentence headroom check. Skipping means we return
        # the input unchanged with ``gate_decision="headroom_skip"``.
        skip_mask, hr_logs = self._maybe_apply_headroom_skip([sentence], apply_gate)
        if skip_mask[0]:
            log = {
                "confidence": 1.0,
                "score": 0.0,
                "gate_decision": "headroom_skip",
            }
            log.update(hr_logs[0])
            return sentence, [log]
        src = sentence
        if self.cfg.task_prefix and not src.startswith(self.cfg.task_prefix):
            src = self.cfg.task_prefix + src
        enc = self.tokenizer(
            src,
            max_length=self.cfg.max_input_bytes,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        in_bytes = len(sentence.encode("utf-8"))
        max_new = min(
            int(self.cfg.max_target_bytes),
            max(32, int(in_bytes * 1.3) + 16),
        )
        min_new = _min_new_tokens_value(self.cfg, in_bytes, max_new)
        beams = int(beam_size if beam_size is not None else self.cfg.beam_size)
        gen_kwargs = {
            "max_new_tokens": max_new,
            "min_new_tokens": min_new,
            "num_beams": beams,
            "return_dict_in_generate": True,
            "output_scores": beams > 1,
        }
        if beams > 1:
            gen_kwargs["length_penalty"] = float(self.cfg.length_penalty)
            gen_kwargs["early_stopping"] = True
        if self.cfg.no_repeat_ngram_size:
            gen_kwargs["no_repeat_ngram_size"] = int(self.cfg.no_repeat_ngram_size)
        use_amp, amp_dtype = _cuda_amp_autocast_dtype(self.cfg, self.device)
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if use_amp and amp_dtype is not None
            else contextlib.nullcontext()
        )
        with torch.no_grad(), amp_ctx:
            out = self.model.generate(**enc, **gen_kwargs)
        seq = out.sequences[0]
        text = self._sanitize_decoded(
            [self.tokenizer.decode(seq, skip_special_tokens=True)]
        )[0]
        score = 0.0
        if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            score = float(out.sequences_scores[0].item())
            confidence = float(min(1.0, math.exp(min(0.0, score))))
        else:
            lp = self.score_target(sentence, text)
            confidence = float(1.0 / (1.0 + math.exp(-lp / max(1.0, len(text) / 8.0))))
        gated_batch, gate_logs = self._maybe_apply_gate(
            [sentence], [text], enabled_override=apply_gate
        )
        log: Dict[str, object] = {
            "confidence": confidence,
            "score": score,
        }
        if gate_logs:
            log.update(gate_logs[0])
        # Headroom metadata flows through every predict path so downstream
        # JSONLs always carry the estimate (or ``None`` when no gate is
        # attached / calibration disabled).
        if hr_logs:
            log.update(hr_logs[0])
        return gated_batch[0], [log]

    def score_target(self, noisy: str, target: str) -> float:
        """Length-normalized log P(target | noisy). Used by the hybrid head."""
        scores = self.score_targets_batch([noisy], [target])
        return scores[0] if scores else float("-inf")

    def _score_targets_internal(
        self,
        noisy_list: Sequence[str],
        target_list: Sequence[str],
        chunk_size: Optional[int] = None,
        return_per_pos: bool = False,
    ) -> Tuple[List[float], Optional[List[List[float]]]]:
        """Teacher-forced log P(target | noisy) — internal core of both
        :meth:`score_targets_batch` and the per-edit gate.

        Returns (sentence_scores, per_pos_logps_or_None). When
        ``return_per_pos`` is True, ``per_pos_logps[i]`` is a list of
        per-target-byte log-probabilities for the i-th pair (length =
        number of non-pad target bytes; floats; empty list for skipped
        rows). Temperature scaling is applied identically in
        both views.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not fitted.")
        if len(noisy_list) != len(target_list):
            raise ValueError(
                f"score_targets_internal: noisy/target length mismatch "
                f"({len(noisy_list)} vs {len(target_list)})"
            )
        if not noisy_list:
            return [], ([] if return_per_pos else None)
        if self.model.training:
            self.model.eval()
        n = len(noisy_list)
        cs = int(chunk_size) if chunk_size is not None else int(
            getattr(self.cfg, "predict_batch_size", 32) or 32
        )
        cs = max(1, cs)
        alpha = float(self.cfg.length_norm_alpha)
        use_amp, amp_dtype = _cuda_amp_autocast_dtype(self.cfg, self.device)
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if use_amp and amp_dtype is not None
            else contextlib.nullcontext()
        )
        out_scores: List[float] = [float("-inf")] * n
        out_perpos: Optional[List[List[float]]] = (
            [[] for _ in range(n)] if return_per_pos else None
        )
        for start in range(0, n, cs):
            stop = min(n, start + cs)
            chunk_noisy = list(noisy_list[start:stop])
            chunk_target = list(target_list[start:stop])
            keep_idx: List[int] = []
            srcs: List[str] = []
            tgts: List[str] = []
            for j, (src, tgt) in enumerate(zip(chunk_noisy, chunk_target)):
                if not tgt:
                    continue
                if self.cfg.task_prefix and not src.startswith(self.cfg.task_prefix):
                    src = self.cfg.task_prefix + src
                keep_idx.append(j)
                srcs.append(src)
                tgts.append(tgt)
            if not srcs:
                continue
            enc = self.tokenizer(
                srcs,
                max_length=self.cfg.max_input_bytes,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            dec = self.tokenizer(
                text_target=tgts,
                max_length=self.cfg.max_target_bytes,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            labels = dec["input_ids"].to(self.device)
            # ByT5 pads target ids with ``pad_token_id``; the T5 loss expects
            # ``-100`` on padded positions, so we mask before forwarding.
            pad_id = int(self.tokenizer.pad_token_id)
            labels_for_loss = labels.clone()
            labels_for_loss[labels_for_loss == pad_id] = -100
            with torch.no_grad(), amp_ctx:
                outputs = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    labels=labels_for_loss,
                )
                logits = outputs.logits  
                # apply temperature scaling before log_softmax. With
                # ``self.temperature == 1.0`` (default) this is a no-op so
                # back-compat with already-trained checkpoints is preserved.
                temperature = max(float(self.temperature), 1e-3)
                log_probs = torch.log_softmax(logits.float() / temperature, dim=-1)
                gathered = log_probs.gather(
                    2, labels.unsqueeze(-1).clamp(min=0)
                ).squeeze(-1)
                mask = (labels_for_loss != -100).float()
                totals = (gathered * mask).sum(dim=1)
                lengths = mask.sum(dim=1)
            totals_l = totals.detach().to("cpu").tolist()
            lengths_l = lengths.detach().to("cpu").tolist()
            if return_per_pos:
                # Mask out pad positions so the downstream window sums are well-defined.
                gathered_l = gathered.detach().to("cpu").tolist()
                mask_l = mask.detach().to("cpu").tolist()
            else:
                gathered_l = None
                mask_l = None
            for k, j in enumerate(keep_idx):
                tok_len = int(lengths_l[k])
                if tok_len <= 0:
                    out_scores[start + j] = float("-inf")
                    if return_per_pos and out_perpos is not None:
                        out_perpos[start + j] = []
                    continue
                denom = ((5 + tok_len) / 6.0) ** alpha
                out_scores[start + j] = float(totals_l[k]) / denom
                if return_per_pos and out_perpos is not None and gathered_l is not None:
                    row = gathered_l[k]
                    m_row = mask_l[k] if mask_l is not None else None
                    # Trim trailing pad positions so the returned vector
                    # has length == number of non-pad target tokens.
                    if m_row is not None:
                        # m_row is a list of 0/1 floats; find last 1.
                        last = -1
                        for idx, v in enumerate(m_row):
                            if v > 0.0:
                                last = idx
                        if last >= 0:
                            out_perpos[start + j] = [float(x) for x in row[: last + 1]]
                        else:
                            out_perpos[start + j] = []
                    else:
                        out_perpos[start + j] = [float(x) for x in row]
        return out_scores, out_perpos

    def score_targets_batch(
        self,
        noisy_list: Sequence[str],
        target_list: Sequence[str],
        chunk_size: Optional[int] = None,
    ) -> List[float]:
        """Length-normalized log P(target | noisy) for many pairs at once.

        Thin wrapper over :meth:`_score_targets_internal` that returns
        sentence-level sums only. Used by the hybrid head and by the
        sentence-level half of the gate.
        """
        scores, _ = self._score_targets_internal(
            noisy_list, target_list, chunk_size=chunk_size, return_per_pos=False
        )
        return scores

    def score_targets_with_perpos_batch(
        self,
        noisy_list: Sequence[str],
        target_list: Sequence[str],
        chunk_size: Optional[int] = None,
    ) -> Tuple[List[float], List[List[float]]]:
        """Both length-normalized sentence scores AND per-target-byte log
        probabilities. Used by the per-edit gate.

        Per-position logps are aligned to UTF-8 byte positions of the
        target string plus a trailing EOS. ``len(per_pos_logps[i])`` may
        be slightly larger than ``len(target_list[i].encode('utf-8'))``
        due to EOS, and may be smaller if the target was truncated by
        ``max_target_bytes``.
        """
        scores, perpos = self._score_targets_internal(
            noisy_list, target_list, chunk_size=chunk_size, return_per_pos=True
        )
        return scores, (perpos or [[] for _ in noisy_list])

    def calibrate_temperature(
        self,
        val_pairs: Sequence[Tuple[str, str]],
        max_pairs: int = 2000,
        grid: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        """Task C: fit a single scalar temperature ``T`` on val NLL.

        Strategy: stream the val sample once through teacher-forced forward
        passes; for every candidate T, accumulate (a) total negative log
        likelihood of the gold tokens and (b) per-bin (confidence, accuracy)
        histograms for token-level Expected Calibration Error. Pick the T
        with the lowest NLL (tie-break: T closest to 1.0). Store on
        ``self.temperature`` and persist alongside HF weights in
        ``transformer_meta.json``.

        Output dict reports ``temperature``, ``pre_ece``, ``post_ece``,
        ``pre_nll``, ``post_nll``, plus the full grid log for the paper
        table. The chosen ``T`` then applies in every place
        :meth:`score_targets_batch` runs (the gate at predict time, the
        gate calibration sweep, the per-edit gate in
        :meth:`_maybe_apply_gate`).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not fitted; cannot calibrate temperature.")
        if not val_pairs:
            print("[byt5] calibrate-temperature: empty val_pairs, skipping", flush=True)
            return {"temperature": 1.0, "tested": 0, "grid": [], "log": []}

        grid_list = list(
            grid
            if grid is not None
            else [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5]
        )
        sample = list(val_pairs[: max(1, int(max_pairs))])
        n_total = len(sample)
        if self.model.training:
            self.model.eval()

        # Always include 1.0 in the grid so we can report pre-calibration ECE.
        if 1.0 not in grid_list:
            grid_list = [1.0] + grid_list

        cs = max(1, int(getattr(self.cfg, "predict_batch_size", 32) or 32))
        use_amp, amp_dtype = _cuda_amp_autocast_dtype(self.cfg, self.device)
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if use_amp and amp_dtype is not None
            else contextlib.nullcontext()
        )

        nll_sum: Dict[float, float] = {T: 0.0 for T in grid_list}
        tok_total = 0
        n_bins = 10
        # Per-T calibration histograms: (conf_sum, correct_sum, count) per bin.
        bins_conf: Dict[float, List[float]] = {T: [0.0] * n_bins for T in grid_list}
        bins_correct: Dict[float, List[float]] = {T: [0.0] * n_bins for T in grid_list}
        bins_count: Dict[float, List[int]] = {T: [0] * n_bins for T in grid_list}

        print(
            f"[byt5] calibrate-temperature: forwarding {n_total} val pairs "
            f"over grid={grid_list} ...",
            flush=True,
        )
        t0 = time.perf_counter()

        for start in range(0, n_total, cs):
            stop = min(n_total, start + cs)
            chunk = sample[start:stop]
            srcs: List[str] = []
            tgts: List[str] = []
            for noisy, clean in chunk:
                if not clean:
                    continue
                src = (
                    self.cfg.task_prefix + noisy
                    if self.cfg.task_prefix and not noisy.startswith(self.cfg.task_prefix)
                    else noisy
                )
                srcs.append(src)
                tgts.append(clean)
            if not srcs:
                continue
            enc = self.tokenizer(
                srcs,
                max_length=self.cfg.max_input_bytes,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            dec = self.tokenizer(
                text_target=tgts,
                max_length=self.cfg.max_target_bytes,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            labels = dec["input_ids"].to(self.device)
            pad_id = int(self.tokenizer.pad_token_id)
            labels_for_loss = labels.clone()
            labels_for_loss[labels_for_loss == pad_id] = -100
            with torch.no_grad(), amp_ctx:
                outputs = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    labels=labels_for_loss,
                )
                logits = outputs.logits.float()  # (B, T, V)
            mask = (labels_for_loss != -100).float()
            gold_safe = labels.clamp(min=0)
            mask_flat = mask.reshape(-1)
            kept = mask_flat > 0
            n_kept_tokens = int(kept.sum().item())
            if n_kept_tokens <= 0:
                continue
            tok_total += n_kept_tokens
            for T in grid_list:
                scaled = logits / max(float(T), 1e-3)
                log_probs = torch.log_softmax(scaled, dim=-1)
                gold_logp = log_probs.gather(2, gold_safe.unsqueeze(-1)).squeeze(-1)
                nll_sum[T] += float(-(gold_logp * mask).sum().item())
                # Per-token confidence / correctness for ECE binning.
                probs = log_probs.exp()
                conf, pred = probs.max(dim=-1)
                correct = (pred == labels).float() * mask
                conf_flat = conf.reshape(-1)[kept]
                correct_flat = correct.reshape(-1)[kept]
                # Bin index on CPU (the per-bin assignment is tiny).
                conf_cpu = conf_flat.detach().to("cpu")
                correct_cpu = correct_flat.detach().to("cpu")
                idx = (conf_cpu * n_bins).clamp_(0, n_bins - 1).long()
                for b in range(n_bins):
                    sel = idx == b
                    if not sel.any():
                        continue
                    bins_conf[T][b] += float(conf_cpu[sel].sum().item())
                    bins_correct[T][b] += float(correct_cpu[sel].sum().item())
                    bins_count[T][b] += int(sel.sum().item())

        if tok_total == 0:
            print(
                "[byt5] calibrate-temperature: no labelled tokens collected, skipping.",
                flush=True,
            )
            return {"temperature": 1.0, "tested": 0, "grid": grid_list, "log": []}

        nll_mean = {T: nll_sum[T] / max(1, tok_total) for T in grid_list}

        def _ece_for(T: float) -> float:
            total = sum(bins_count[T])
            if total == 0:
                return 0.0
            err = 0.0
            for b in range(n_bins):
                cnt = bins_count[T][b]
                if cnt == 0:
                    continue
                mean_conf = bins_conf[T][b] / cnt
                mean_acc = bins_correct[T][b] / cnt
                err += (cnt / total) * abs(mean_conf - mean_acc)
            return err

        ece = {T: _ece_for(T) for T in grid_list}
        best_T = min(
            grid_list, key=lambda T: (nll_mean[T], abs(T - 1.0))
        )

        elapsed = time.perf_counter() - t0
        log = [
            {
                "temperature": float(T),
                "nll": float(nll_mean[T]),
                "ece": float(ece[T]),
                "n_tokens": int(tok_total),
            }
            for T in grid_list
        ]

        self.temperature = float(best_T)
        result = {
            "temperature": float(best_T),
            "pre_temperature": 1.0,
            "pre_nll": float(nll_mean[1.0]),
            "post_nll": float(nll_mean[best_T]),
            "pre_ece": float(ece[1.0]),
            "post_ece": float(ece[best_T]),
            "n_pairs": int(n_total),
            "n_tokens": int(tok_total),
            "n_bins": n_bins,
            "grid": grid_list,
            "log": log,
            "elapsed_s": float(elapsed),
        }
        self.temperature_calibration = result
        print(
            f"[byt5] calibrate-temperature: T={best_T} "
            f"NLL {nll_mean[1.0]:.4f} -> {nll_mean[best_T]:.4f}  "
            f"ECE {ece[1.0]:.4f} -> {ece[best_T]:.4f} "
            f"in {elapsed:.1f}s",
            flush=True,
        )
        return result

    def _effective_gate_margin(self) -> float:
        """Margin actually used by the gate.

        ``tuned_gate_margin`` (set by :meth:`calibrate_gate_on_val` and restored
        from ``transformer_meta.json``) takes precedence over the static
        ``cfg.gate_log_prob_margin`` fallback.
        """
        if self.tuned_gate_margin is not None:
            return float(self.tuned_gate_margin)
        return float(getattr(self.cfg, "gate_log_prob_margin", 0.0) or 0.0)

    def _effective_per_edit_margin(self) -> float:
        """Per-edit margin. Mirrors :meth:`_effective_gate_margin`
        but for the per-edit-window log-prob delta. The tuned value from
        :meth:`calibrate_gate_on_val` takes precedence over
        ``cfg.gate_per_edit_margin``.
        """
        if self.tuned_per_edit_margin is not None:
            return float(self.tuned_per_edit_margin)
        return float(getattr(self.cfg, "gate_per_edit_margin", 0.0) or 0.0)

    @staticmethod
    def _utf8_char_offsets(s: str) -> List[int]:
        """Cumulative UTF-8 byte offsets for each character boundary.

        For a string ``s`` with ``L`` characters returns a list of length
        ``L + 1`` such that ``offsets[i]`` is the byte offset of character
        ``i`` (or end-of-string for ``i == L``). Used to translate
        ``SequenceMatcher`` opcodes (character-indexed) to the byte-indexed
        per-position log-prob arrays returned by
        :meth:`score_targets_with_perpos_batch`.
        """
        offsets = [0]
        cum = 0
        for ch in s:
            cum += len(ch.encode("utf-8"))
            offsets.append(cum)
        return offsets

    def _window_logp(
        self,
        per_pos: Sequence[float],
        byte_start: int,
        byte_stop: int,
    ) -> float:
        """Sum per-byte log-probs over ``[byte_start, byte_stop)``.

        Returns ``-inf`` when the requested window lies fully outside the
        available per-position vector (e.g. tail of a truncated target).
        Partial overlaps sum the available bytes; the caller treats
        ``-inf`` as "unscored, keep edit conservatively".
        """
        if byte_stop <= byte_start:
            # Pure insert/delete on this side: the window has zero length on
            # this target. The caller cancels the comparison and uses a
            # neutral (=0) contribution from the missing side.
            return 0.0
        if not per_pos:
            return float("-inf")
        lo = max(0, int(byte_start))
        hi = min(len(per_pos), int(byte_stop))
        if hi <= lo:
            return float("-inf")
        return float(sum(per_pos[lo:hi]))

    def _token_containing(self, s: str, char_pos: int) -> str:
        """Return the surface token in ``s`` that contains character offset
        ``char_pos``. Empty string if the position falls on whitespace
        between tokens. Used by the proper-name / rare-word margin
        boost.
        """
        if not s or char_pos < 0 or char_pos >= len(s):
            return ""
        # Walk regex matches; pick the one straddling the position.
        for m in _TOKEN_RE.finditer(s):
            if m.start() <= char_pos < m.end():
                return m.group(0)
        return ""

    def _per_edit_extra_margin_for(self, surface_token: str) -> float:
        """extra log-prob margin required to accept an edit inside
        ``surface_token``. Mirrors classical's proper-name / rare-word
        margin discipline. Returns 0.0 when no lexicon is attached.
        """
        if not surface_token:
            return 0.0
        extra = 0.0
        cfg = self.cfg
        if is_proper_name(surface_token):
            extra += float(getattr(cfg, "gate_proper_name_extra_margin", 0.0) or 0.0)
        if self.train_word_counts is not None and is_rare_word(
            surface_token, self.train_word_counts
        ):
            extra += float(getattr(cfg, "gate_rare_word_extra_margin", 0.0) or 0.0)
        return float(extra)

    def _maybe_apply_gate(
        self,
        noisy_list: Sequence[str],
        predictions: Sequence[str],
        enabled_override: Optional[bool] = None,
    ) -> Tuple[List[str], List[Dict[str, object]]]:
        """Run the neural log-prob gate over a batch of generations.

        For each pair we compute the (length-normalized) model log-probability
        of the generated prediction *and* of the original noisy input as if it
        had been emitted unchanged, then keep the edit only when
        ``log P(pred | noisy) - log P(noisy | noisy) >= effective_margin``,
        where ``effective_margin`` is the tuned per-model margin if present,
        else ``cfg.gate_log_prob_margin``. Otherwise we revert to the noisy
        input. The gate is a no-op when the prediction already equals the
        input (``identity`` branch) or when gating is disabled (via
        ``cfg.gate_enabled`` or ``enabled_override=False``).
        """
        n = len(noisy_list)
        default_decision = {
            "gate_decision": "identity",
            "pred_score": 0.0,
            "input_score": 0.0,
            "score_margin": 0.0,
            "n_windows": 0,
            "n_kept_windows": 0,
            "n_reverted_windows": 0,
        }
        decisions: List[Dict[str, object]] = [
            dict(default_decision) for _ in range(n)
        ]
        if not n:
            return list(predictions), decisions
        gate_enabled = (
            bool(enabled_override)
            if enabled_override is not None
            else bool(getattr(self.cfg, "gate_enabled", True))
        )
        margin = self._effective_gate_margin()
        per_edit_margin = self._effective_per_edit_margin()
        results = list(predictions)
        if not gate_enabled:
            for j in range(n):
                if results[j] != noisy_list[j]:
                    decisions[j] = {
                        "gate_decision": "kept_no_gate",
                        "pred_score": 0.0,
                        "input_score": 0.0,
                        "score_margin": 0.0,
                        "n_windows": 0,
                        "n_kept_windows": 0,
                        "n_reverted_windows": 0,
                    }
            return results, decisions
        diff_idx = [
            j for j in range(n)
            if results[j] and results[j] != noisy_list[j]
        ]
        if not diff_idx:
            return results, decisions
        diff_noisy = [noisy_list[j] for j in diff_idx]
        diff_preds = [results[j] for j in diff_idx]
        # collect per-byte log-probs alongside sentence sums in the
        # SAME forward pass so the per-edit walk pays no extra GPU cost.
        pred_scores, pred_perpos = self.score_targets_with_perpos_batch(
            diff_noisy, diff_preds
        )
        input_scores, input_perpos = self.score_targets_with_perpos_batch(
            diff_noisy, diff_noisy
        )
        whitelist = self.confusion_whitelist
        for k, j in enumerate(diff_idx):
            ps = pred_scores[k]
            is_ = input_scores[k]
            delta = ps - is_
            noisy_str = noisy_list[j]
            pred_str = diff_preds[k]
            if not math.isfinite(ps) or not math.isfinite(is_):
                # Sentence-level scoring failed (truncated to nothing,
                # tokenizer hiccup, etc.). Keep the edit but log it for
                # post-hoc analysis; the per-edit pass cannot run reliably.
                decisions[j] = {
                    "gate_decision": "kept_unscored",
                    "pred_score": ps,
                    "input_score": is_,
                    "score_margin": delta,
                    "n_windows": 0,
                    "n_kept_windows": 0,
                    "n_reverted_windows": 0,
                }
                continue
            if delta < margin:
                # Sentence-level reject: cheaper to revert the whole sentence
                # than to walk the edits.
                results[j] = noisy_str
                decisions[j] = {
                    "gate_decision": "reverted_to_input",
                    "pred_score": ps,
                    "input_score": is_,
                    "score_margin": delta,
                    "n_windows": 0,
                    "n_kept_windows": 0,
                    "n_reverted_windows": 0,
                }
                continue

            # Per-edit walk.
            matcher = SequenceMatcher(None, noisy_str, pred_str, autojunk=False)
            noisy_offsets = self._utf8_char_offsets(noisy_str)
            pred_offsets = self._utf8_char_offsets(pred_str)
            pp_pred = pred_perpos[k] or []
            pp_input = input_perpos[k] or []
            out_chars: List[str] = []
            n_windows = 0
            n_kept = 0
            n_reverted = 0
            window_logs: List[Dict[str, object]] = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    out_chars.append(noisy_str[i1:i2])
                    continue
                n_windows += 1
                noisy_window = noisy_str[i1:i2]
                pred_window = pred_str[j1:j2]
                b_n1, b_n2 = noisy_offsets[i1], noisy_offsets[i2]
                b_p1, b_p2 = pred_offsets[j1], pred_offsets[j2]
                pred_window_logp = self._window_logp(pp_pred, b_p1, b_p2)
                input_window_logp = self._window_logp(pp_input, b_n1, b_n2)
                # Compose a per-edit delta that compares the model's
                # confidence in the *edit* against its confidence in *keeping
                # the input*. For pure inserts (i1 == i2) input_window_logp
                # is 0; for pure deletes (j1 == j2) pred_window_logp is 0.
                delta_pos = float(pred_window_logp) - float(input_window_logp)
                # surface-token-aware extra margin. Use the noisy
                # token at i1 because that is what the human-readable
                # alignment intuitively maps to.
                surface_token = self._token_containing(noisy_str, i1)
                extra = self._per_edit_extra_margin_for(surface_token)
                effective_margin = per_edit_margin + extra
                reject_reason = None
                # empirical confusion-matrix constraint for
                # single-character substitutions. Multi-char windows and
                # word-boundary inserts/deletes bypass this check.
                if (
                    whitelist is not None
                    and tag == "replace"
                    and (i2 - i1) == 1
                    and (j2 - j1) == 1
                ):
                    src_ch = noisy_window
                    tgt_ch = pred_window
                    if src_ch.isalpha() and tgt_ch.isalpha():
                        if (src_ch, tgt_ch) not in whitelist:
                            reject_reason = "off_whitelist"
                if reject_reason is None and (
                    not math.isfinite(delta_pos)
                    or delta_pos < effective_margin
                ):
                    reject_reason = "below_per_edit_margin"
                if reject_reason is None:
                    out_chars.append(pred_window)
                    n_kept += 1
                    window_logs.append({
                        "tag": tag,
                        "src": noisy_window,
                        "tgt": pred_window,
                        "delta_pos": float(delta_pos)
                        if math.isfinite(delta_pos) else None,
                        "extra_margin": float(extra),
                        "kept": True,
                    })
                else:
                    out_chars.append(noisy_window)
                    n_reverted += 1
                    window_logs.append({
                        "tag": tag,
                        "src": noisy_window,
                        "tgt": pred_window,
                        "delta_pos": float(delta_pos)
                        if math.isfinite(delta_pos) else None,
                        "extra_margin": float(extra),
                        "kept": False,
                        "gate_reject_reason": reject_reason,
                    })

            new_pred = "".join(out_chars)
            results[j] = new_pred
            if new_pred == noisy_str:
                final_tag = "reverted_to_input"
            elif new_pred == pred_str:
                final_tag = "kept_pred"
            else:
                final_tag = "kept_pred_partial"
            decisions[j] = {
                "gate_decision": final_tag,
                "pred_score": ps,
                "input_score": is_,
                "score_margin": delta,
                "n_windows": n_windows,
                "n_kept_windows": n_kept,
                "n_reverted_windows": n_reverted,
                "windows": window_logs,
            }
        return results, decisions


    def calibrate_gate_on_val(
        self,
        val_pairs: Sequence[Tuple[str, str]],
        max_pairs: Optional[int] = None,
        candidate_margins: Optional[Sequence[float]] = None,
        beam_size: Optional[int] = None,
    ) -> Dict[str, object]:
        """Sweep ``gate_log_prob_margin`` on val and set the best value.

        Strategy: generate **ungated** predictions on the val subset (one
        ``generate`` pass), score ``(noisy, pred)`` and ``(noisy, noisy)``
        once per edited record (two batched forward passes total), then
        evaluate every margin in ``candidate_margins`` against val CER purely
        from the cached ``delta = pred_score - input_score`` values. The grid
        sweep itself is CPU-only.

        Selection metric: mean character-error-rate.
        Tie-break: prefer the **more conservative** margin (higher value).

        The chosen margin is stored on ``self.tuned_gate_margin``, takes
        precedence over ``cfg.gate_log_prob_margin`` at inference time, and is
        persisted in ``transformer_meta.json`` by :meth:`save` so it survives
        save/load.
        """
        from phase4.eval.metrics import cer  # local to avoid import cycle

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not fitted; cannot calibrate gate.")
        if not bool(getattr(self.cfg, "gate_enabled", True)):
            print("[byt5] calibrate-gate: cfg.gate_enabled=False, skipping", flush=True)
            return {"selected": None, "tested": 0, "grid_size": 0, "log": []}
        if not val_pairs:
            print("[byt5] calibrate-gate: empty val_pairs, skipping", flush=True)
            return {"selected": None, "tested": 0, "grid_size": 0, "log": []}

        grid = (
            list(candidate_margins)
            if candidate_margins is not None
            else list(getattr(self.cfg, "gate_calibration_grid", [0.0]))
        )
        if not grid:
            grid = [float(getattr(self.cfg, "gate_log_prob_margin", 0.0) or 0.0)]

        n_max = (
            int(max_pairs)
            if max_pairs is not None
            else int(getattr(self.cfg, "gate_calibration_max_pairs", 400))
        )
        sample = list(val_pairs[: max(1, n_max)])
        noisy_all = [n for n, _ in sample]
        clean_all = [c for _, c in sample]
        n_total = len(sample)

        print(
            f"[byt5] calibrate-gate: generating raw (ungated) predictions on "
            f"{n_total} val pairs and scoring (noisy, pred) + (noisy, noisy) "
            "for edited records...",
            flush=True,
        )
        t0 = time.perf_counter()

        # 1. Generate raw predictions with the gate disabled. Length-bucket so
        #    each chunk's ``max_new_tokens`` is tight, identical to the
        #    runner's prediction path.
        gen_batch = max(1, int(getattr(self.cfg, "eval_gen_batch_size", 16)))
        order = sorted(range(n_total), key=lambda k: len(noisy_all[k]))
        raw_preds: List[str] = [""] * n_total
        beam_confs: List[float] = [0.0] * n_total
        done = 0
        for start in range(0, len(order), gen_batch):
            idxs = order[start : start + gen_batch]
            chunk_noisy = [noisy_all[i] for i in idxs]
            with_logs = self.correct_batch_with_logs(
                chunk_noisy, beam_size=beam_size, apply_gate=False
            )
            for j, src_idx in enumerate(idxs):
                pred, logs = with_logs[j]
                raw_preds[src_idx] = pred
                if logs:
                    beam_confs[src_idx] = float(logs[0].get("confidence", 0.0))
            done += len(idxs)
            elapsed = time.perf_counter() - t0
            rate = elapsed / max(1, done)
            eta = rate * (n_total - done)
            print(
                f"[byt5] calibrate-gate: generated {done}/{n_total} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

        # 2. Score (noisy, pred) and (noisy, noisy) for records the beam edited.
        #    capture per-byte log-probs alongside sentence sums so the
        #    2-D sweep can simulate the per-edit gate from cached deltas only.
        diff_idx = [
            j for j in range(n_total)
            if raw_preds[j] and raw_preds[j] != noisy_all[j]
        ]
        t_score = time.perf_counter()
        if diff_idx:
            diff_noisy = [noisy_all[j] for j in diff_idx]
            diff_preds = [raw_preds[j] for j in diff_idx]
            pred_scores, pred_perpos = self.score_targets_with_perpos_batch(
                diff_noisy, diff_preds
            )
            input_scores, input_perpos = self.score_targets_with_perpos_batch(
                diff_noisy, diff_noisy
            )
        else:
            pred_scores = []
            input_scores = []
            pred_perpos = []
            input_perpos = []
        score_elapsed = time.perf_counter() - t_score

        # 3. Cache per-record info used by both the sentence-level and the
        #    per-edit gate simulations. Treat non-finite sentence deltas as
        #    "always keep at the sentence level" so the sweep keeps them.
        whitelist = self.confusion_whitelist
        per_record_cache: Dict[int, Dict[str, object]] = {}
        for k, j in enumerate(diff_idx):
            ps = pred_scores[k]
            is_ = input_scores[k]
            sentence_delta = (
                float(ps - is_) if math.isfinite(ps) and math.isfinite(is_)
                else float("inf")
            )
            noisy_str = noisy_all[j]
            pred_str = raw_preds[j]
            noisy_offsets = self._utf8_char_offsets(noisy_str)
            pred_offsets = self._utf8_char_offsets(pred_str)
            pp_pred = pred_perpos[k] if pred_perpos else []
            pp_input = input_perpos[k] if input_perpos else []
            matcher = SequenceMatcher(None, noisy_str, pred_str, autojunk=False)
            windows: List[Dict[str, object]] = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    windows.append({"tag": "equal", "i1": i1, "i2": i2,
                                    "j1": j1, "j2": j2})
                    continue
                b_n1, b_n2 = noisy_offsets[i1], noisy_offsets[i2]
                b_p1, b_p2 = pred_offsets[j1], pred_offsets[j2]
                pred_window_logp = self._window_logp(pp_pred, b_p1, b_p2)
                input_window_logp = self._window_logp(pp_input, b_n1, b_n2)
                delta_pos = float(pred_window_logp) - float(input_window_logp)
                surface_token = self._token_containing(noisy_str, i1)
                extra = self._per_edit_extra_margin_for(surface_token)
                # whitelist precheck (does not depend on margin grid).
                off_whitelist = False
                if (
                    whitelist is not None
                    and tag == "replace"
                    and (i2 - i1) == 1
                    and (j2 - j1) == 1
                ):
                    src_ch = noisy_str[i1:i2]
                    tgt_ch = pred_str[j1:j2]
                    if src_ch.isalpha() and tgt_ch.isalpha():
                        off_whitelist = (src_ch, tgt_ch) not in whitelist
                windows.append({
                    "tag": tag, "i1": i1, "i2": i2, "j1": j1, "j2": j2,
                    "delta_pos": delta_pos,
                    "extra": float(extra),
                    "off_whitelist": bool(off_whitelist),
                })
            per_record_cache[j] = {
                "noisy": noisy_str,
                "pred": pred_str,
                "sentence_delta": sentence_delta,
                "windows": windows,
            }

        # Precompute identity CER and full-edit CER so the sweep only has to
        # branch per-record between the two when the sentence-level gate
        # short-circuits.
        pred_cer = [cer(clean_all[j], raw_preds[j]) for j in range(n_total)]
        noisy_cer = [cer(clean_all[j], noisy_all[j]) for j in range(n_total)]
        identity_only_cer = sum(noisy_cer) / max(1, n_total)
        all_edits_cer = sum(pred_cer) / max(1, n_total)

        # 2-D grid: same candidates on both axes by default. The grid is
        # already shared by ``cfg.gate_calibration_grid`` for both
        # ``gate_log_prob_margin`` and ``gate_per_edit_margin``.
        per_edit_grid = list(grid)

        # 4. Precompute the per-edit-margin gated string per record per
        #    candidate ``pe_margin``. Per-edit walking is cheap; we cache
        #    the CER so the outer sentence-margin loop is O(1) per record.
        pe_cer_cache: Dict[Tuple[int, float], float] = {}
        for j_idx, cache in per_record_cache.items():
            clean_str = clean_all[j_idx]
            noisy_str_j = str(cache["noisy"])
            pred_str_j = str(cache["pred"])
            for pe_margin in per_edit_grid:
                out_chars: List[str] = []
                for w in cache["windows"]:  
                    if w["tag"] == "equal":
                        out_chars.append(noisy_str_j[w["i1"]:w["i2"]]) 
                        continue
                    noisy_window = noisy_str_j[w["i1"]:w["i2"]]  
                    pred_window = pred_str_j[w["j1"]:w["j2"]]  
                    if bool(w["off_whitelist"]):
                        out_chars.append(noisy_window)
                        continue
                    dp = float(w["delta_pos"]) 
                    eff = float(pe_margin) + float(w["extra"])  
                    if not math.isfinite(dp):
                        # Truncated/unscored window: keep edit conservatively.
                        out_chars.append(pred_window)
                    elif dp >= eff:
                        out_chars.append(pred_window)
                    else:
                        out_chars.append(noisy_window)
                gated_str = "".join(out_chars)
                pe_cer_cache[(j_idx, float(pe_margin))] = cer(clean_str, gated_str)

        print(
            f"[byt5] calibrate-gate: precompute done in "
            f"{time.perf_counter() - t0:.1f}s "
            f"(scoring took {score_elapsed:.1f}s); "
            f"sweeping 2-D grid (|sentence|={len(grid)}, "
            f"|per_edit|={len(per_edit_grid)}) on {n_total} pairs "
            f"({len(diff_idx)} edited).  baseline mean_cer="
            f"{identity_only_cer:.4f} (identity), {all_edits_cer:.4f} (no-gate)",
            flush=True,
        )

        log: List[Dict[str, object]] = []
        best: Optional[Dict[str, object]] = None
        for s_margin in grid:
            for pe_margin in per_edit_grid:
                cer_sum = 0.0
                kept_ct = 0
                for j in range(n_total):
                    rec = per_record_cache.get(j)
                    if rec is None:
                        # Beam left this record unchanged; identity row.
                        cer_sum += noisy_cer[j]
                        continue
                    sd = float(rec["sentence_delta"])
                    if sd < float(s_margin):
                        cer_sum += noisy_cer[j]
                    else:
                        cer_sum += pe_cer_cache[(j, float(pe_margin))]
                        kept_ct += 1
                mean_cer = cer_sum / max(1, n_total)
                change_rate = kept_ct / max(1, n_total)
                entry = {
                    "gate_log_prob_margin": float(s_margin),
                    "gate_per_edit_margin": float(pe_margin),
                    "mean_cer": float(mean_cer),
                    "kept": int(kept_ct),
                    "n_pairs": int(n_total),
                    "n_generated_edits": int(len(diff_idx)),
                    "change_rate": float(change_rate),
                }
                log.append(entry)
                # Tie-break (in order): lower CER, then higher per-edit
                # margin (more conservative per-edit), then higher
                # sentence margin (more conservative sentence-level).
                if (
                    best is None
                    or float(entry["mean_cer"]) < float(best["mean_cer"]) - 1e-6
                    or (
                        abs(float(entry["mean_cer"]) - float(best["mean_cer"])) < 1e-6
                        and (
                            float(entry["gate_per_edit_margin"])
                            > float(best["gate_per_edit_margin"])
                            or (
                                float(entry["gate_per_edit_margin"])
                                == float(best["gate_per_edit_margin"])
                                and float(entry["gate_log_prob_margin"])
                                > float(best["gate_log_prob_margin"])
                            )
                        )
                    )
                ):
                    best = entry

        if best is None:
            print("[byt5] calibrate-gate: no valid margin found; leaving tuned_gate_margin unset", flush=True)
            return {
                "selected": None,
                "baseline_identity_cer": float(identity_only_cer),
                "baseline_no_gate_cer": float(all_edits_cer),
                "tested": 0,
                "grid_size": len(grid) * len(per_edit_grid),
                "n_generated_edits": int(len(diff_idx)),
                "log": [],
            }

        self.tuned_gate_margin = float(best["gate_log_prob_margin"])
        self.tuned_per_edit_margin = float(best["gate_per_edit_margin"])
        self.gate_calibration = {
            "selected": dict(best),
            "tested": len(log),
            "grid_size": len(grid) * len(per_edit_grid),
            "n_generated_edits": int(len(diff_idx)),
            "baseline_identity_cer": float(identity_only_cer),
            "baseline_no_gate_cer": float(all_edits_cer),
            "fallback_margin": float(
                getattr(self.cfg, "gate_log_prob_margin", 0.0) or 0.0
            ),
            "fallback_per_edit_margin": float(
                getattr(self.cfg, "gate_per_edit_margin", 0.0) or 0.0
            ),
            "log": log,
        }
        print(
            f"[byt5] calibrate-gate: selected sentence_margin="
            f"{self.tuned_gate_margin:.4f}  per_edit_margin="
            f"{self.tuned_per_edit_margin:.4f}  "
            f"mean_cer={best['mean_cer']:.4f}  "
            f"kept={best['kept']}/{best['n_pairs']}  "
            f"(identity baseline={identity_only_cer:.4f}, "
            f"no-gate={all_edits_cer:.4f})  "
            f"total={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return dict(self.gate_calibration)


    def attach_lexicon(
        self,
        train_word_counts: Optional[Dict[str, int]],
        lexicon: Optional[set] = None,
    ) -> None:
        """attach the train lexicon / word counts.

        Used by the per-edit gate (proper-name / rare-word extra margins)
        and by the HeadroomGate (OOV proxy). Idempotent: passing ``None``
        clears the cached lexicon.
        """
        if train_word_counts is None:
            self.train_word_counts = None
            self.lexicon = None
            return
        self.train_word_counts = dict(train_word_counts)
        if lexicon is None:
            # Derive a lexicon from the counts (words seen at least once).
            lexicon = {w for w, c in train_word_counts.items() if c >= 1}
        self.lexicon = set(lexicon)

    def attach_confusion_whitelist(
        self,
        whitelist: Optional[Union[frozenset, Sequence[Tuple[str, str]]]],
    ) -> None:
        """attach the empirical confusion-matrix whitelist.

        ``whitelist`` is a set of ``(src, tgt)`` single-character
        substitutions sourced from train-only phase-2 stats; passing
        ``None`` (or ``frozenset()``) disables the constraint. The gate
        is permissive when there is no whitelist (all single-char
        substitutions are allowed), so omitting this hook on a cold-loaded
        model is safe but loses the discipline.
        """
        if whitelist is None:
            self.confusion_whitelist = None
            return
        if isinstance(whitelist, frozenset):
            self.confusion_whitelist = whitelist
        else:
            self.confusion_whitelist = frozenset(whitelist)

    def attach_headroom_gate(self, gate: Optional[object]) -> None:
        """attach the HeadroomGate that pre-filters predict calls.

        When attached, the public correct_* paths short-circuit and return
        the input unchanged for sentences the gate judges to have too
        little correctable noise. Calibration paths (calibrate_gate_on_val,
        calibrate_temperature, score_targets_batch) never short-circuit.
        """
        self.headroom = gate

    def save(self, output_dir: Path) -> None:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Cannot save unfitted model.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # The HF model may be wrapped by ``torch.compile``. Save
        # the underlying ``_orig_mod`` so the on-disk artefact is loadable
        # by environments without the same compile graph.
        save_target = getattr(self.model, "_orig_mod", self.model)
        save_target.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        meta = {
            "config": asdict(self.cfg),
            "seed": self.seed,
            "training_metrics": self.training_metrics,
            "model_kind": "byt5",
            "tuned_gate_margin": (
                float(self.tuned_gate_margin)
                if self.tuned_gate_margin is not None
                else None
            ),
            "tuned_per_edit_margin": (
                float(self.tuned_per_edit_margin)
                if self.tuned_per_edit_margin is not None
                else None
            ),
            "temperature": float(self.temperature),
            "temperature_calibration_summary": (
                {
                    k: self.temperature_calibration[k]
                    for k in (
                        "temperature",
                        "pre_nll",
                        "post_nll",
                        "pre_ece",
                        "post_ece",
                        "n_pairs",
                        "n_tokens",
                    )
                    if isinstance(self.temperature_calibration, dict)
                    and k in self.temperature_calibration
                }
                if isinstance(self.temperature_calibration, dict)
                else None
            ),
            "gate_calibration_summary": (
                {
                    k: self.gate_calibration[k]
                    for k in (
                        "selected",
                        "baseline_identity_cer",
                        "baseline_no_gate_cer",
                        "fallback_margin",
                        "grid_size",
                        "tested",
                        "n_generated_edits",
                    )
                    if isinstance(self.gate_calibration, dict)
                    and k in self.gate_calibration
                }
                if isinstance(self.gate_calibration, dict)
                else None
            ),
        }
        (output_dir / "transformer_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        (output_dir / "transformer.pt").write_text(
            "byt5 weights saved via huggingface (model.safetensors). "
            "This sentinel file exists to satisfy the legacy artifact check.\n",
            encoding="utf-8",
        )
        # persist the HeadroomGate state alongside HF weights so
        # ``load`` can restore it without re-running the val sweep.
        if self.headroom is not None:
            try:
                self.headroom.save(output_dir / "headroom_gate")  # type: ignore[attr-defined]
            except Exception as exc:
                print(
                    f"[byt5] save: HeadroomGate.save failed ({exc}); "
                    "continuing without it.",
                    flush=True,
                )
        self.checkpoint_path = output_dir

    def load(self, output_dir: Path) -> None:
        T5ForConditionalGeneration, ByT5Tokenizer = _import_hf()
        output_dir = Path(output_dir)
        meta_path = output_dir / "transformer_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cfg_dict = dict(meta.get("config") or {})
            # Drop unknown keys so checkpoints saved by older configs still
            # load when new fields are added.
            valid_field_names = {f.name for f in TransformerConfig.__dataclass_fields__.values()}
            cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_field_names}
            self.cfg = TransformerConfig(**cfg_dict)
            self.training_metrics = meta.get("training_metrics", {})
            tuned = meta.get("tuned_gate_margin")
            self.tuned_gate_margin = float(tuned) if tuned is not None else None
            tuned_pe = meta.get("tuned_per_edit_margin")
            self.tuned_per_edit_margin = (
                float(tuned_pe) if tuned_pe is not None else None
            )
            temp = meta.get("temperature")
            self.temperature = float(temp) if temp is not None else 1.0
            temp_summary = meta.get("temperature_calibration_summary")
            if isinstance(temp_summary, dict):
                self.temperature_calibration = dict(temp_summary)
            gate_summary = meta.get("gate_calibration_summary")
            if isinstance(gate_summary, dict):
                self.gate_calibration = dict(gate_summary)
        self.tokenizer = ByT5Tokenizer.from_pretrained(output_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(output_dir).to(self.device)
        self._maybe_torch_compile()
        # restore HeadroomGate state (if persisted) so the
        # predict path can re-gate without recalibrating on val.
        hr_dir = output_dir / "headroom_gate"
        if hr_dir.exists() and (hr_dir / "headroom_gate.json").exists():
            try:
                from phase4.models.headroom_gate import HeadroomGate  # local import to avoid cycle
                self.headroom = HeadroomGate.load(hr_dir)
            except Exception as exc:
                print(
                    f"[byt5] load: HeadroomGate.load failed ({exc}); "
                    "predict will run without headroom.",
                    flush=True,
                )
                self.headroom = None
        self.checkpoint_path = output_dir
