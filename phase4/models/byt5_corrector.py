from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union  # noqa: F401

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from phase4.config import TransformerConfig
from phase4.eval.metrics import cer
from phase4.models.classical import _load_confusion_sub_log_probs


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


    def _load_pretrained(self, source: Optional[Path] = None) -> None:
        T5ForConditionalGeneration, ByT5Tokenizer = _import_hf()
        src = str(source) if source is not None else self.cfg.pretrained_model
        print(f"[byt5] loading {src} on device={self.device}", flush=True)
        self.tokenizer = ByT5Tokenizer.from_pretrained(src)
        self.model = T5ForConditionalGeneration.from_pretrained(src).to(self.device)
        self._configure_model_for_training()

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
        for src, tgt in pairs:
            if not src or not tgt:
                skipped += 1
                continue
            kept.append((src, tgt))
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

    def correct_batch(
        self,
        sentences: Sequence[str],
        beam_size: Optional[int] = None,
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
        prefixed: List[str] = []
        for s in sentences:
            if not s:
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
            "num_beams": beams,
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
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        result = list(sentences)
        for j, idx in enumerate(non_empty_idx):
            result[idx] = decoded[j]
        return result

    def correct_batch_with_logs(
        self,
        sentences: Sequence[str],
        beam_size: Optional[int] = None,
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
        prefixed: List[str] = []
        for s in sentences:
            if not s:
                prefixed.append("")
                continue
            src = s
            if self.cfg.task_prefix and not src.startswith(self.cfg.task_prefix):
                src = self.cfg.task_prefix + src
            prefixed.append(src)
        non_empty_idx = [i for i, s in enumerate(prefixed) if s]
        results: List[Tuple[str, List[Dict[str, object]]]] = [
            (s, [{"confidence": 1.0}]) for s in sentences
        ]
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
        sequences = out.sequences
        decoded = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        seq_scores = None
        if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            seq_scores = out.sequences_scores.detach().to("cpu").tolist()
        for j, idx in enumerate(non_empty_idx):
            text = decoded[j]
            if seq_scores is not None and j < len(seq_scores):
                score = float(seq_scores[j])
                confidence = float(min(1.0, math.exp(min(0.0, score))))
            else:
                # Greedy path has no per-sequence score; use a neutral
                # confidence so calibration bins still get populated.
                score = 0.0
                confidence = 0.5
            results[idx] = (text, [{"confidence": confidence, "score": score}])
        return results

    def correct_sentence(
        self,
        sentence: str,
        beam_size: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, object]]]:
        if not sentence:
            return sentence, [{"confidence": 1.0}]
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not fitted.")
        if self.model.training:
            self.model.eval()
        self._configure_model_for_inference()
        src = sentence
        if self.cfg.task_prefix and not src.startswith(self.cfg.task_prefix):
            src = self.cfg.task_prefix + src
        enc = self.tokenizer(
            src,
            max_length=self.cfg.max_input_bytes,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        max_new = min(
            int(self.cfg.max_target_bytes),
            max(32, int(len(sentence.encode("utf-8")) * 1.3) + 16),
        )
        beams = int(beam_size if beam_size is not None else self.cfg.beam_size)
        gen_kwargs = {
            "max_new_tokens": max_new,
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
        text = self.tokenizer.decode(seq, skip_special_tokens=True)
        score = 0.0
        if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            score = float(out.sequences_scores[0].item())
            confidence = float(min(1.0, math.exp(min(0.0, score))))
        else:
            lp = self.score_target(sentence, text)
            confidence = float(1.0 / (1.0 + math.exp(-lp / max(1.0, len(text) / 8.0))))
        return text, [{"confidence": confidence, "score": score}]

    def score_target(self, noisy: str, target: str) -> float:
        """Length-normalized log P(target | noisy). Used by the hybrid head."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model is not fitted.")
        if not target:
            return float("-inf")
        if self.model.training:
            self.model.eval()
        src = noisy
        if self.cfg.task_prefix and not src.startswith(self.cfg.task_prefix):
            src = self.cfg.task_prefix + src
        enc = self.tokenizer(
            src,
            max_length=self.cfg.max_input_bytes,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        dec = self.tokenizer(
            text_target=target,
            max_length=self.cfg.max_target_bytes,
            truncation=True,
            return_tensors="pt",
        )
        labels = dec["input_ids"].to(self.device)
        # T5 ignores -100 in the loss; we want the per-token NLL summed and
        # then length-normalised, so manually compute via a forward pass.
        with torch.no_grad():
            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=labels,
            )
            logits = outputs.logits  # (1, T, V)
            log_probs = torch.log_softmax(logits, dim=-1)
            # Drop EOS-pad slots: the gathered NLL is over the *labels* which
            # include the final EOS but not any padding (we only padded across
            # batch in the loader; here batch=1 so labels have no -100).
            target_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            mask = (labels != -100).float()
            total = float((target_log_probs * mask).sum().item())
            n = float(mask.sum().item())
        if n <= 0:
            return float("-inf")
        # Same length-norm formula as the legacy byte Transformer so the
        # hybrid fusion weights stay calibrated to comparable magnitudes.
        return total / (((5 + int(n)) / 6.0) ** float(self.cfg.length_norm_alpha))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: Path) -> None:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Cannot save unfitted model.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        meta = {
            "config": asdict(self.cfg),
            "seed": self.seed,
            "training_metrics": self.training_metrics,
            "model_kind": "byt5",
        }
        (output_dir / "transformer_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # ``transformer.pt`` sentinel keeps backwards compatibility with the
        # runner's ``_assert_artifacts_saved_before_test`` check (which looks
        # for that file as evidence the neural model was persisted).
        (output_dir / "transformer.pt").write_text(
            "byt5 weights saved via huggingface (model.safetensors). "
            "This sentinel file exists to satisfy the legacy artifact check.\n",
            encoding="utf-8",
        )
        self.checkpoint_path = output_dir

    def load(self, output_dir: Path) -> None:
        T5ForConditionalGeneration, ByT5Tokenizer = _import_hf()
        output_dir = Path(output_dir)
        meta_path = output_dir / "transformer_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.cfg = TransformerConfig(**meta["config"])
            self.training_metrics = meta.get("training_metrics", {})
        self.tokenizer = ByT5Tokenizer.from_pretrained(output_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(output_dir).to(self.device)
        self.checkpoint_path = output_dir
