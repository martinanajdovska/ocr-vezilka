# OCR Vezilka: Macedonian OCR Correction Pipeline

This repository implements a 4-phase pipeline for analyzing OCR errors in Macedonian Cyrillic text, generating realistic synthetic OCR noise, and training/evaluating correction models.

The intended workflow is:
1. Align noisy OCR text with corrected text (`phase1`).
2. Analyze observed OCR error patterns (`phase2`).
3. Generate synthetic OCR noise from those patterns (`phase3`).
4. Train/evaluate correction models with real/synthetic regimes (`phase4`).

---

## Quick Start

### Requirements
- Python 3.10+ recommended.
- Install dependencies:
  - `python -m pip install -r requirements.txt`

If you are on Linux with CUDA, install PyTorch CUDA first (as noted in `requirements.txt` comments), then run:
- `pip install -r requirements.txt`

### Minimal run sequence
From repository root:
- `python phase1/phase1_alignment.py`
- `python phase2/ocr_error_analysis.py`
- `python phase3/phase3_synthetic_noise.py`
- `python phase4/phase4_correction_models.py`

Notes:
- `phase2` and `phase3` default `main()` runs both full and train-only modes.
- `phase4` can rebuild dependencies automatically unless skip flags are used.

---

## Repository Map

- `phase1/`
  - `phase1_alignment.py`: sentence alignment + character/word edit extraction + boundary error detection.
  - `raw_ocr/`: noisy OCR inputs (`*_ocr_raw.txt`).
  - `corrected_ocr/`: corrected references (`*_corrected.txt`).
  - `phase1_output/`: split/book outputs + corpus summaries.

- `phase2/`
  - `ocr_error_analysis.py`: confusion matrices, substitution categories (diacritic/homoglyph/punctuation), split/merge rates, visualizations.
  - `phase2_output_train_only/`: train-only artifacts for leak-safe downstream phases.

- `phase3/`
  - `phase3_synthetic_noise.py`: synthetic noise generators and realism evaluation.
  - `phase3_output_train_only/`: generated synthetic corpora and sentence pairs (used in phase4).

- `phase4/`
  - `phase4_correction_models.py`: end-to-end training/evaluation orchestrator.
  - `config.py`: frozen hyperparameters and run configuration.
  - `data/`
    - `splits.py`: authoritative train/val/test doc split and split integrity checks.
    - `build_phase4_dataset.py`: manifest construction for training regimes.
    - `text_segmentation.py`: segmentation/chunking helpers.
  - `models/`
    - `classical.py`: lexicon + channel + language model baseline.
    - `byt5_corrector.py`: ByT5 fine-tuning and inference.
    - `hybrid.py`: classical candidate generation + neural reranking/fusion.
    - `transformer_seq2seq.py`: alias wrapper used by phase4 runner.
  - `eval/`
    - `metrics.py`: CER/WER/chrF, calibration, bootstrap, diagnostics.
    - `paper_tables.py`: aggregate outputs into publication-ready tables.
  - `io/schemas.py`: prediction record schema validation.
  - `phase4_output/`: predictions, metrics, checkpoints, efficiency metrics, metadata.

- `requirements.txt`: dependency pin ranges for analysis + neural training stack.

---

## Data Contract

### Input naming conventions (phase1)
- OCR file: `<book_name>_ocr_raw.txt` in `phase1/raw_ocr/`
- Corrected file: `<book_name>_corrected.txt` in `phase1/corrected_ocr/`
- Pairing is by `<book_name>` stem.

### Split assignment
Split membership is hardcoded in phase scripts (`phase1` and `phase2`) and reused in phase4 split checks. Current split IDs:
- Train: `dnevnik_po_mnogu_godini`, `itar_pejo`, `Pesni`, `Prezir`, `Samecot`, `sina_pesna`, `tajnopis`, `Toj`, `viktor_kupidon`
- Val: `Забите на Ветрот - Томе Арсовски`, `Провиденија`
- Test: `Сите лица на смртта`, `Современост 7`

---

## End-to-End Pipeline

### Phase 1: Alignment and edit extraction
File: `phase1/phase1_alignment.py`

What it does:
- Cleans OCR/reference text (page markers, batch metadata, dash noise, garbage lines).
- Splits text into sentences.
- Aligns OCR-vs-GT sentence sequences with dynamic programming.
- Extracts:
  - character edit operations (`match/substitution/deletion/insertion`)
  - word edit operations
  - word boundary errors (`split` / `merge`)
- Computes per-book metrics and corpus-level statistics.

Run:
- `python phase1/phase1_alignment.py`

Important:
- Script expects current working directory to be `phase1/` when using default relative paths (`raw_ocr`, `corrected_ocr`, `phase1_output`).
- `phase4` handles this internally when it re-runs phase1.

Main outputs:
- `phase1/phase1_output/summary.json` (corpus summary)
- `phase1/phase1_output/<split>/<book>/matched_pairs.json`
- `phase1/phase1_output/<split>/<book>/char_ops.json`
- `phase1/phase1_output/<split>/<book>/word_ops.json`
- `phase1/phase1_output/<split>/<book>/word_boundary_errors.json`
- `phase1/phase1_output/<split>/<book>/stats.json`
- `phase1/phase1_output/<split>/<book>/alignment_quality.json`

### Phase 2: OCR error analysis
File: `phase2/ocr_error_analysis.py`

What it does:
- Loads phase1 outputs.
- Builds confusion matrices and normalized probabilities.
- Computes error distributions and structural boundary rates.
- Extracts top substitutions and confusion families (diacritic/homoglyph/punctuation).
- Produces plots and summary tables.

Run options:
- Full corpus mode:
  - `python -c "from phase2.ocr_error_analysis import run_full_mode; run_full_mode()"`
- Train-only mode (leak-safe downstream stats):
  - `python -c "from phase2.ocr_error_analysis import run_train_only_mode; run_train_only_mode()"`
- Script default (`main`) runs both:
  - `python phase2/ocr_error_analysis.py`

Main outputs:
- `phase2/phase2_output/` (full mode)
- `phase2/phase2_output_train_only/` (train-only mode), including:
  - `error_distribution.json`
  - `error_confusion_counts.csv`
  - `error_confusion_probs.csv`
  - `word_boundary_stats.json`
  - `phase2_summary.json`
  - plots and doc/split tables

### Phase 3: Synthetic OCR noise generation
File: `phase3/phase3_synthetic_noise.py`

What it does:
- Reads phase2 error statistics.
- Calibrates event rates and structure-aware transformations.
- Generates synthetic noisy text using three methods:
  - `random_edit_noise`
  - `confusion_matrix_noise`
  - `structure_aware_noise`
- Exports synthetic text and sentence-level noisy/clean pairs.
- Compares synthetic error profile to real OCR profile.

Run options:
- Full mode:
  - `python -c "from phase3.phase3_synthetic_noise import run_full_mode; run_full_mode()"`
- Train-only-statistics mode (recommended for modeling):
  - `python -c "from phase3.phase3_synthetic_noise import run_train_only_mode; run_train_only_mode()"`
- Script default (`main`) runs both:
  - `python phase3/phase3_synthetic_noise.py`

Main outputs:
- `phase3/phase3_output/` (full mode)
- `phase3/phase3_output_train_only/` (train-only mode), containing per-regime synthetic docs and `*_pairs.jsonl`.

### Phase 4: Model training and evaluation
File: `phase4/phase4_correction_models.py`

What it does:
- Optionally re-runs phase1 and (re)builds train-only phase2/phase3 artifacts.
- Builds phase4 manifests for regimes:
  - `real_only`
  - `synthetic_only`
  - `synthetic_plus_real` (two-stage neural pretrain->finetune)
- Trains/evaluates:
  - `classical`
  - `neural` (ByT5)
  - `hybrid`
- Writes val predictions + metrics, blind test predictions, efficiency metrics, paper tables, run manifest.
- Emits identity baseline for each regime.

Run:
- `python phase4/phase4_correction_models.py`

Useful flags:
- `--seeds 42,1337,2026`
- `--regimes real_only,synthetic_only,synthetic_plus_real`
- `--skip-classical`
- `--skip-neural`
- `--skip-hybrid`
- `--skip-phase1`
- `--skip-phase2-stats`
- `--skip-phase3-noise`
- `--skip-manifests`
- `--neural-device auto|cpu|mps|cuda`

Output root:
- `phase4/phase4_output/`

Key phase4 artifacts:
- `predictions/val/<model>/<regime>[__seedX].jsonl`
- `predictions/test_blind/<model>/<regime>[__seedX].jsonl`
- `val_metrics/<model>/<regime>[__seedX].json`
- `efficiency/<model>/<regime>[__seedX].json`
- `models/<model>/<regime>/...`
- `metadata/run_manifest.json`
- `metadata/run_table.csv`

---

## Modeling and Evaluation Notes

- Primary seed is `42`; additional seeds are configured in `phase4/config.py`.
- Hyperparameters are centralized and hashable via `frozen_hparams_dict()`.
- Train/val/test leak barriers are enforced at runtime in phase4.
- Test split is treated as blind for metrics in main prediction flow.
- Validation metrics include CER, WER, chrF, overcorrection indicators, rare/proper-name corruption checks, and calibration summaries.

---

## Reproducibility and Safety Constraints

- Use train-only phase2/phase3 outputs for any training-time decisions to avoid leakage from val/test statistics.
- Do not tune thresholds on test outputs.
- Keep split definitions synchronized with `phase4/data/splits.py` and phase scripts.
- If running phase1 standalone, run it from the `phase1/` directory or adapt paths.
- Neural training runtime depends on hardware; use `--neural-device cpu` as fallback if MPS/CUDA issues occur.

---

## What a New Agent Should Check First

1. Read `phase4/phase4_correction_models.py` to understand the orchestrated full pipeline.
2. Read `phase4/config.py` for all frozen hyperparameters and path defaults.
3. Inspect `phase4/data/splits.py` and `phase2/ocr_error_analysis.py` split constants for data partitioning assumptions.
4. Verify that `phase1/raw_ocr/` and `phase1/corrected_ocr/` contain correctly named file pairs.
5. Confirm required train-only artifacts exist:
   - `phase2/phase2_output_train_only/error_distribution.json`
   - `phase3/phase3_output_train_only/structure_aware_noise/*_synthetic.txt`

---

## Current Gaps (Important)

- No automated test suite is included.
- No CI/lint configuration is included.
- `.gitignore` is minimal (`.idea` only), so generated artifacts may appear in git status unless ignored manually.

---

## One-command practical run (after first successful build)

For quick iteration on model experiments while reusing artifacts:

- `python phase4/phase4_correction_models.py --skip-phase1 --skip-phase2-stats --skip-phase3-noise --skip-manifests`

Use this only when you trust existing phase1/2/3 outputs and manifests.
