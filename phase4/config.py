"""Phase 4 configuration.

All hyperparameters that influence model behavior live here so they can be
hashed once into ``hyperparameter_hash`` and reused identically across the
3 (model) x 3 (regime) x N (seed) grid
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List


PRIMARY_SEED = 42
SECONDARY_SEEDS = []
ALL_SEEDS = [PRIMARY_SEED]
SEEDS = ALL_SEEDS


@dataclass(frozen=True)
class ClassicalConfig:
    max_edit_distance: int = 2
    top_k_candidates: int = 8
    char_ngram_order: int = 4
    add_k_smoothing: float = 0.1
    word_lm_order: int = 3
    word_lm_discount: float = 0.75
    candidate_pool_max: int = 64
    beam_size: int = 6
    lambda_lm: float = 1.0
    lambda_channel: float = 0.6
    lambda_char_lm: float = 0.3
    correction_margin: float = 1.5
    proper_name_extra_margin: float = 1.0
    rare_word_extra_margin: float = 1.0
    case_preserving: bool = True



@dataclass(frozen=True)
class TransformerConfig:
    pretrained_model: str = "google/byt5-small"

    learning_rate: float = 1.0e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0

    max_input_bytes: int = 512
    max_target_bytes: int = 512


    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    dataloader_num_workers: int = 4
    pin_memory: bool = True

    gradient_checkpointing: bool = False

    max_epochs: int = 15
    pretrain_epochs: int = 10
    finetune_epochs: int = 5
    early_stopping_patience: int = 3
    early_stop_metric: str = "val_cer"

    beam_size: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    length_norm_alpha: float = 0.6
    min_new_token_ratio: float = 0.5

    # ``gate_log_prob_margin`` is the *fallback* value used when no per-model
    # calibration has run. After training, the runner calls
    # ``ByT5Corrector.calibrate_gate_on_val`` which sweeps
    # ``gate_calibration_grid`` on a capped val subset, picks the entry with
    # the lowest mean CER (tie-break: prefer the more conservative margin),
    # and stores it on the corrector as ``tuned_gate_margin``. The tuned value
    # takes precedence over this default at inference
    gate_enabled: bool = True
    gate_log_prob_margin: float = 0.0
    gate_calibration_grid: List[float] = field(
        default_factory=lambda: [-0.2, -0.1, 0.0, 0.05, 0.1, 0.2, 0.3]
    )
    gate_calibration_max_pairs: int = 400

    # Post-decode script filter: map Russian/Ukrainian Cyrillic confusions
    # (``ё``, ``й``, ``ы``, …) to Macedonian graphemes and drop any remaining
    # non-MK Cyrillic.  ByT5 is multilingual and often emits these even when
    # the training targets are Macedonian-only.
    sanitize_macedonian_output: bool = True

    # ``identity_pair_ratio`` bumped 0.2 -> 0.4: the prior run still emitted an
    # edit for ~20% of inputs even though only ~5% of input chars were wrong,
    # so teach the model that "input was already clean" is a frequent target.
    # Combined with the inference-time gate this lowers both spurious edits and
    # the wasted gate work.
    identity_pair_ratio: float = 0.4
    finetune_lr_scale: float = 0.3
    finetune_warmup_ratio: float = 0.0
    task_prefix: str = "correct OCR: "

    use_amp: bool = True
    # False enables cudnn.benchmark + TF32 on CUDA (faster on A100).
    deterministic: bool = False

    eval_cer_pairs: int = 96
    eval_cer_beam: int = 1

    eval_gen_batch_size: int = 128
    predict_batch_size: int = 128

    use_window_context: bool = False
    window_context_sep: str = " <sep> "
    # Bumped 0.1 -> 0.25 so that training batches see noise distributions
    # closer to the Phase-2 confusion stats. This improves edit precision when
    # the gate does let an edit through (the gate is only as good as the
    # model's per-character probability calibration).
    training_confusion_noise_prob: float = 0.25



@dataclass(frozen=True)
class HybridConfig:
    num_classical_candidates: int = 5
    fusion_w_lm: float = 1.0
    fusion_w_channel: float = 0.5
    fusion_w_neural: float = 1.5
    margin_default: float = 0.8
    margin_proper_name: float = 1.6
    margin_rare_word: float = 1.4
    calibration_grid_w_neural: List[float] = field(default_factory=lambda: [0.5, 1.5, 2.5])
    calibration_grid_w_lm: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    calibration_grid_w_channel: List[float] = field(default_factory=lambda: [0.3, 0.6, 1.0])
    calibration_grid_margin: List[float] = field(default_factory=lambda: [0.4, 1.2, 2.0])


@dataclass(frozen=True)
class ManifestConfig:
    band: int = 8
    chunk_window: int = 20
    min_align_sim: float = 0.30
    qa_max_mismatch_ratio: float = 0.55
    qa_max_low_alignment_ratio: float = 0.55



@dataclass(frozen=True)
class RunConfig:
    repo_root: Path
    phase1_output_dir: Path
    phase1_clean_dir: Path
    phase1_raw_dir: Path
    phase2_train_only_dir: Path
    phase3_train_only_dir: Path
    output_dir: Path
    fail_fast: bool = False
    force_rebuild_train_only_stats: bool = False
    force_rebuild_manifests: bool = False
    synthetic_real_oversample_ratio: float = 4.0
    manifest_min_pair_sim: float = 0.5
    manifest_max_len_ratio_delta: float = 0.5


def default_run_config(repo_root: Path) -> RunConfig:
    return RunConfig(
        repo_root=repo_root,
        phase1_output_dir=repo_root / "phase1" / "phase1_output",
        phase1_clean_dir=repo_root / "phase1" / "corrected_ocr",
        phase1_raw_dir=repo_root / "phase1" / "raw_ocr",
        phase2_train_only_dir=repo_root / "phase2" / "phase2_output_train_only",
        phase3_train_only_dir=repo_root / "phase3" / "phase3_output_train_only",
        output_dir=repo_root / "phase4" / "phase4_output",
    )


def frozen_hparams_dict() -> Dict[str, object]:
    return {
        "seeds": ALL_SEEDS,
        "classical": asdict(ClassicalConfig()),
        "transformer": asdict(TransformerConfig()),
        "hybrid": asdict(HybridConfig()),
        "manifest": asdict(ManifestConfig()),
    }


@dataclass(frozen=True)
class Seq2SeqConfig:
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 20
    early_stopping_patience: int = 4
    teacher_forcing_ratio: float = 0.5
    beam_size: int = 1
    max_decode_len: int = 48


MAX_EDIT_DISTANCE = ClassicalConfig().max_edit_distance
TOP_K_CANDIDATES = ClassicalConfig().top_k_candidates
CHAR_NGRAM_ORDER = ClassicalConfig().char_ngram_order
ADD_K_SMOOTHING = ClassicalConfig().add_k_smoothing
