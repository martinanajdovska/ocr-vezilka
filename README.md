# OCR Vezilka: Macedonian OCR Correction Pipeline

This repository implements a 4-phase pipeline for analyzing OCR errors in Macedonian Cyrillic text, generating realistic synthetic OCR noise, and training/evaluating correction models.

The dataset used and outputs generated can be found at: https://huggingface.co/martinanajdovska/ocr-vezilka-outputs

The models can be found at: https://huggingface.co/martinanajdovska/ocr-vezilka-models

**Workflow**

1. Align noisy OCR text with corrected references (`phase1`).
2. Analyze observed OCR error patterns (`phase2`).
3. Generate synthetic OCR noise from those patterns (`phase3`).
4. Train and evaluate correction models under real / synthetic data regimes (`phase4`).

---

## Quick Start

### Requirements

- Python **3.10+** recommended.
- Install **PyTorch first**, then the rest (see `requirements.txt` header). On CUDA 12.x:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -r requirements.txt
huggingface-cli download google/byt5-small   
```

On Mac / CPU-only, install a matching `torch` wheel first, then `python3 -m pip install -r requirements.txt`.

### Minimal run sequence

From the repository root:

```bash
cd phase1 && python3 phase1_alignment.py && cd ..
python3 phase2/ocr_error_analysis.py
python3 phase3/phase3_synthetic_noise.py
python3 phase4/phase4_correction_models.py
```

**Notes**

- `phase2` and `phase3` default `main()` runs **both** full-corpus and train-only modes.
- `phase4` can rebuild upstream artifacts automatically unless skip flags are set (see below).
- For iteration after a successful full build, reuse artifacts:

```bash
python3 phase4/phase4_correction_models.py \
  --skip-phase1 --skip-phase2-stats --skip-phase3-noise --skip-manifests
```

### Full pipeline (standard sweep + k-fold)

The default Phase 4 run trains **3 model families × 3 regimes × 3 seeds** (seeds `42`, `1337`, `2024`), tunes classical thresholds once, calibrates neural gates/headroom on val, and writes paper tables. To also run the **5-fold cross-document** evaluation in one resumable process:

```bash
python3 -u phase4/phase4_correction_models.py \
  --skip-phase1 --skip-phase2-stats --skip-phase3-noise \
  --also-kfold 5
```

Use `--skip-standard-sweep` with `--also-kfold` when only the k-fold harness changed and the standard sweep is already cached. Use `--force-rerun` to wipe sentinels and retrain from scratch.

K-fold only (primary seed, ignores extra `--seeds`):

```bash
python3 phase4/phase4_correction_models.py \
  --skip-phase1 --skip-phase2-stats --skip-phase3-noise \
  --kfold 5
```

### GPU smoke test (neural only)

After train-only Phase 2/3 artifacts exist:

```bash
python3 -u phase4/phase4_correction_models.py \
  --skip-phase1 --skip-phase2-stats --skip-phase3-noise \
  --skip-classical --skip-hybrid --regimes real_only --seeds 42 \
  --neural-device cuda --predict-sample 200
```

`--predict-sample N` caps ByT5 training pairs, uses one epoch per stage, and writes at most `N` val/test prediction rows per model.

---

## Repository Map

| Path | Role |
|------|------|
| `phase1/phase1_alignment.py` | Sentence alignment, char/word edits, boundary errors |
| `phase1/raw_ocr/` | Noisy OCR inputs (`<book>_ocr_raw.txt`) |
| `phase1/corrected_ocr/` | Corrected references (`<book>_corrected.txt`) |
| `phase1/phase1_output/` | Per-split/book JSON artifacts + corpus summary |
| `phase2/ocr_error_analysis.py` | Confusion matrices, substitution families, plots |
| `phase2/phase2_output_train_only/` | **Train-only** stats (leak-safe; used by phase3/4) |
| `phase3/phase3_synthetic_noise.py` | Three synthetic noise generators + realism eval |
| `phase3/phase3_output_train_only/` | Train-only synthetic corpora and `*_pairs.jsonl` |
| `phase4/phase4_correction_models.py` | End-to-end training/evaluation orchestrator |
| `phase4/config.py` | Frozen hyperparameters (hashed into run metadata) |
| `phase4/data/splits.py` | **Authoritative** train/val/test splits + k-fold rotations |
| `phase4/data/build_phase4_dataset.py` | Manifest construction for the three regimes |
| `phase4/data/text_segmentation.py` | Segmentation/chunking helpers |
| `phase4/data/manifests/` | Built `real_only`, `synthetic_only`, `synthetic_plus_real` JSONL |
| `phase4/models/classical.py` | Lexicon + channel + Kneser–Ney LM baseline |
| `phase4/models/byt5_corrector.py` | ByT5 fine-tuning, gate/headroom calibration, inference |
| `phase4/models/headroom_gate.py` | Pre-neural headroom filter (cheap OOV/perplexity proxies) |
| `phase4/models/transformer_seq2seq.py` | Thin alias → `ByT5Corrector` (used by the runner) |
| `phase4/models/hybrid.py` | Classical candidates + neural reranking/fusion |
| `phase4/models/macedonian_script.py` | Post-decode Macedonian Cyrillic sanitization |
| `phase4/eval/metrics.py` | CER, WER, chrF, calibration, bootstrap, diagnostics |
| `phase4/eval/paper_tables.py` | Aggregate CSV tables for publication |
| `phase4/eval/kfold.py` | Mean ± std aggregation across k-fold paper tables |
| `phase4/io_safe/sentinels.py` | Crash-resume sentinel helpers (`_DONE.json`, etc.) |
| `phase4/io_safe/schemas.py` | Prediction record schema validation |
| `phase4/phase4_output/` | Predictions, metrics, checkpoints, paper tables |
| `requirements.txt` | Pinned dependency ranges |

---

## Corpus and Splits

**Authoritative source:** `phase4/data/splits.py` (imported by phase1, phase2, and phase4).

The corpus comprises **14 Macedonian Cyrillic documents** (prose and poetry). OCR/corrected file pairing uses the shared `<book_name>` stem (see [Data contract](#data-contract)).

| Split | Documents | Genre |
|-------|-----------|-------|
| **train** (10) | `dnevnik_po_mnogu_godini`, `itar_pejo`, `Pesni`, `Prezir`, `sina_pesna`, `tajnopis`, `Toj`, `viktor_kupidon`, `Забите на Ветрот - Томе Арсовски`, `Клуч за одредување на рибите и змииорките во Република Македонија` | prose except `Pesni`, `sina_pesna` (poetry) |
| **val** (2) | `Провиденија`, `Samecot` | prose |
| **test** (2) | `Сите лица на смртта`, `Современост 7` | prose |

**Doc ID alias:** corrected file  
`Клуч за одредување на рибите и змииорките во Република Македонија поправено_corrected.txt`  
maps to doc id `Клуч за одредување на рибите и змииорките во Република Македонија`.

**Subset domain** (`prose` vs `poetry`) replaces a hash-based A/B split for cross-domain evaluation in phase4 metrics and paper tables.

### K-fold cross-document splits

`KFOLD_TEST_SETS` in `phase4/data/splits.py` defines **5** held-out test pairs (2 documents each). For fold `i`, that pair is test; the next pair (mod 5) is val; the remaining 10 documents are train. Fold 0 matches the canonical split above. The k-fold driver calls `set_kfold_override()` so manifests, metrics, and split hashes stay consistent within each fold.

| Fold | Test documents |
|------|----------------|
| 0 | `Сите лица на смртта`, `Современост 7` (canonical test) |
| 1 | `Провиденија`, `Samecot` |
| 2 | `dnevnik_po_mnogu_godini`, `itar_pejo` |
| 3 | `Pesni`, `sina_pesna` (poetry-heavy) |
| 4 | `Toj`, `viktor_kupidon`, `tajnopis` |

Each fold writes an isolated tree under `phase4/phase4_output/kfold/foldN/` (manifests, models, predictions, paper tables). Aggregated metrics land in `phase4/phase4_output/paper_tables/kfold_summary.csv`.

---

## Data Contract

### Input naming (phase1)

- OCR: `phase1/raw_ocr/<book_name>_ocr_raw.txt`
- Corrected: `phase1/corrected_ocr/<book_name>_corrected.txt`
- Pairing is by `<book_name>` stem.

### Leakage rules

- **Phase 2 train-only** (`phase2_output_train_only/`): confusion statistics from **train** documents only.
- **Phase 3 train-only** (`phase3_output_train_only/`): noise calibrated from train-only phase2 stats; synthetic text generated from **train** clean sources.
- **Phase 4**: all training-time decisions use **train** manifest rows only; classical thresholds, hybrid fusion weights, neural **gate** margins, **headroom** thresholds, and **temperature** scaling are tuned on **val** only; **test** is blind (predictions written, no threshold tuning).

---

## End-to-End Pipeline

### Phase 1: Alignment and edit extraction

**File:** `phase1/phase1_alignment.py`

- Cleans OCR and reference text (page markers, OCR batch metadata, dash noise, garbage lines).
- Splits into sentences and aligns OCR vs. GT with dynamic programming (`MIN_PAIR_SIM=0.55` by default).
- Extracts character edits, word edits, and word-boundary errors (`split` / `merge`).
- Assigns each book to train/val/test via `phase4.data.splits.SPLITS`.

**Run** (from `phase1/`, default relative paths):

```bash
cd phase1
python3 phase1_alignment.py
# optional: --output-root, --raw-dir, --corrected-dir, --min-pair-sim
```

Phase 4 re-runs phase1 by `chdir` into `phase1/` when `--skip-phase1` is not set.

**Main outputs**

- `phase1/phase1_output/summary.json`
- `phase1/phase1_output/<split>/<book>/matched_pairs.json` — fields `ocr`, `gt`
- `char_ops.json`, `word_ops.json`, `word_boundary_errors.json`, `stats.json`, `alignment_quality.json`

### Phase 2: OCR error analysis

**File:** `phase2/ocr_error_analysis.py`

- Loads phase1 outputs; builds confusion matrices and normalized probabilities.
- Computes error distributions, boundary rates, and top substitution families (diacritic / homoglyph / punctuation).
- Writes CSV/JSON summaries and matplotlib/seaborn plots.

**Run**

```bash
# default: full + train-only
python3 phase2/ocr_error_analysis.py

# or individually:
python3 -c "from phase2.ocr_error_analysis import run_full_mode; run_full_mode()"
python3 -c "from phase2.ocr_error_analysis import run_train_only_mode; run_train_only_mode()"
```

**Main outputs**

- `phase2/phase2_output/` — full corpus
- `phase2/phase2_output_train_only/` — **use this for modeling**, including:
  - `error_distribution.json`, `error_confusion_probs.csv`, `phase2_summary.json`
  - `word_boundary_stats.json`, plots, per-document tables

### Phase 3: Synthetic OCR noise generation

**File:** `phase3/phase3_synthetic_noise.py`

- Reads phase2 error statistics; calibrates event rates and structure-aware transforms.
- Three generators per document:
  - `random_edit_noise`
  - `confusion_matrix_noise`
  - `structure_aware_noise` ← **used for phase4 manifests**
- Emits `<doc>_synthetic.txt` and aligned `<doc>_pairs.jsonl` (`clean` / `noisy` per sentence).
- Compares synthetic error profiles to real OCR.

**Run**

```bash
python3 phase3/phase3_synthetic_noise.py

# train-only (recommended for phase4):
python3 -c "from phase3.phase3_synthetic_noise import run_train_only_mode; run_train_only_mode()"
```

**Main outputs**

- `phase3/phase3_output_train_only/<regime>/` — synthetic text + `*_pairs.jsonl`

### Phase 4: Model training and evaluation

**File:** `phase4/phase4_correction_models.py`

**Orchestration**

1. Optionally re-run phase1; ensure train-only phase2 + phase3 artifacts exist.
2. **Classical tuning (once per output root):** fit a baseline classical model on real train rows and grid-search `correction_margin`, `lambda_lm`, `lambda_channel` on real val pairs; cache in `classical_tuning.json` (reused by all regimes/seeds until hashes change).
3. Build manifests (`real_only`, `synthetic_only`, `synthetic_plus_real`) from phase1 pairs and phase3 `structure_aware_noise` pairs (with real oversampling for `synthetic_plus_real`, ratio `4.0` in `RunConfig`).
4. For each **(model × regime × seed)**:
   - **classical** — lexicon + confusion channel + word LM; uses globally tuned thresholds
   - **neural** — fine-tuned `google/byt5-small` with regime-specific identity-pair mixing, val **temperature** scaling, val **log-prob gate** (sentence + per-edit), val **headroom gate**, Macedonian script sanitization
   - **hybrid** — classical candidate pool + ByT5 rerank; fusion weights calibrated on val
5. Emit **identity** baseline per regime; write val metrics, blind test predictions, efficiency JSON, paper tables, run manifest.

**Default seeds:** `42`, `1337`, `2024` (`SEEDS` in `phase4/config.py`; override with `--seeds`).

**Crash-resume sentinels** (see `phase4/io_safe/sentinels.py`):

| Sentinel | Scope |
|----------|--------|
| `models/<model>/<regime>/_DONE.json` | One (model, regime, seed) run |
| `classical_tuning.json` | Shared classical val tuning |
| `_SWEEP_STANDARD_DONE.json` | Full standard sweep |
| `kfold/foldN/_FOLD_DONE.json` | One k-fold leg |
| `kfold/_SWEEP_KFOLD_DONE.json` | All k-folds |
| `_PIPELINE_DONE.json` | Standard + k-fold chained run |

Sentinels store `hyperparameter_hash` and `split_manifest_hash`; stale sentinels are ignored automatically. `--force-rerun` clears sweep/fold/pipeline markers and per-run `_DONE.json` files.

**Run**

```bash
# Standard 3-seed sweep only
python3 phase4/phase4_correction_models.py

# Standard sweep, then 5-fold cross-document eval
python3 phase4/phase4_correction_models.py --also-kfold 5

# K-fold only (primary seed)
python3 phase4/phase4_correction_models.py --kfold 5 \
  --skip-phase1 --skip-phase2-stats --skip-phase3-noise
```

**CLI flags**

| Flag | Purpose |
|------|---------|
| `--seeds 42,1337,2024` | Override seed list |
| `--regimes real_only,synthetic_only,synthetic_plus_real` | Subset of training regimes |
| `--skip-classical` / `--skip-neural` / `--skip-hybrid` | Skip model families |
| `--skip-phase1` | Reuse `phase1/phase1_output/` |
| `--skip-phase2-stats` | Reuse `phase2/phase2_output_train_only/` |
| `--skip-phase3-noise` | Reuse `phase3/phase3_output_train_only/` |
| `--skip-manifests` | Reuse `phase4/data/manifests/*.jsonl` (standard sweep only; k-fold always rebuilds per fold) |
| `--kfold N` | K-fold cross-document sweep only (primary seed) |
| `--also-kfold N` | Standard sweep, then k-fold sweep (resumable) |
| `--skip-standard-sweep` | With `--also-kfold`, skip the standard leg |
| `--force-rerun` | Delete sentinels and retrain from scratch |
| `--neural-device auto\|cpu\|mps\|cuda` | ByT5 device (`auto`: cuda → mps → cpu) |
| `--predict-sample N` | Short sanity run (capped train/predict) |
| `--no-neural-resume` | Ignore ByT5 epoch checkpoints (or set `PHASE4_NEURAL_RESUME=0`) |
| `--rebuild-val-metrics` | Recompute val metrics from existing prediction JSONL |
| `--rebuild-paper-tables` | With above, refresh `paper_tables/*.csv` |
| `--metrics-models identity,classical,neural,hybrid` | Models for metrics recovery |

**Neural resume:** checkpoints under  
`phase4/phase4_output/models/<neural|hybrid>/<regime>/_neural_train_checkpoint/`  
(`byt5_resume_<stage>.pt`, `pretrain_hf/` for `synthetic_plus_real`, etc.).

**Output root:** `phase4/phase4_output/`

| Artifact | Path pattern |
|----------|----------------|
| Val predictions | `predictions/val/<model>/<regime>__seed<seed>.jsonl` (primary seed omits suffix) |
| Blind test predictions | `predictions/test_blind/<model>/<regime>__seed<seed>.jsonl` |
| Val metrics | `val_metrics/<model>/<regime>__seed<seed>.json` |
| Efficiency | `efficiency/<model>/<regime>__seed<seed>.json` |
| Model weights | `models/<model>/<regime>/...` |
| Classical tuning | `classical_tuning.json` |
| Neural calibrations | `models/neural/<regime>/neural_temperature.json`, `neural_gate_calibration.json`, `headroom_calibration.json`, `headroom_gate/headroom_gate.json` |
| Paper tables | `paper_tables/*.csv` (see [Evaluation](#evaluation)) |
| K-fold per fold | `kfold/foldN/` (full mirror of the above) |
| K-fold aggregate | `paper_tables/kfold_summary.csv`, `paper_tables/kfold_manifest.json` |
| Run metadata | `metadata/run_manifest.json`, `metadata/run_table.csv` |

#### Latest standard run (primary seed 42, val)

Results from the completed 3-seed standard sweep (`metadata/run_manifest.json`). Val CER / CER reduction vs. identity:

| Regime | Identity CER | Classical | Neural | Hybrid |
|--------|-------------|-----------|--------|--------|
| `real_only` | 0.0533 | 0.0533 (−0.05%) | 0.0533 (0%; headroom skips all val sentences) | 0.0533 (−0.05%) |
| `synthetic_only` | 0.1049 | 0.1047 (−0.24%) | **0.0761 (−27.5%)** | 0.0993 (−5.4%) |
| `synthetic_plus_real` | 0.0950 | 0.0947 (−0.24%) | **0.0758 (−20.2%)** | 0.0925 (−2.6%) |

On `real_only` val the headroom gate correctly short-circuits essentially all sentences (already clean); synthetic regimes show large neural gains with controlled overcorrection (see `headroom_curve.csv` and `significance.csv`).

---

## Models and Configuration

Hyperparameters live in `phase4/config.py` and are hashed via `frozen_hparams_dict()` into run metadata.

| Component | Highlights |
|-----------|------------|
| **Classical** | `max_edit_distance=2`, beam search, λ_lm / λ_channel / λ_char_lm; **global** val tuning → `classical_tuning.json` |
| **Neural (ByT5)** | `google/byt5-small`, regime-specific `identity_pair_ratio` (0.6 real / 0.4 synthetic), `training_confusion_noise_prob=0.25`, three-stage val calibration (below), `sanitize_macedonian_output=True`, `synthetic_plus_real`: 10 pretrain + 5 finetune epochs |
| **Hybrid** | Top classical candidates + neural score fusion; grid search on val |
| **Manifests** | Band/chunk settings in `ManifestConfig`; pair QA thresholds in `RunConfig` |

### Neural inference stack (three calibrated layers)

Applied in order at predict time (`phase4/models/byt5_corrector.py`, `phase4/models/headroom_gate.py`):

1. **Headroom gate (pre-filter)** — Estimates input CER from cheap proxies (char-trigram perplexity z-score, OOV fraction, suspicious homoglyph density). Sentences below a val-tuned threshold pass through unchanged (`gate_decision="headroom_skip"`). Threshold is chosen on val to maximize CER reduction subject to regime-specific overcorrection and minimum-kept-fraction constraints (`HEADROOM_*_BY_REGIME` in `config.py`). State: `headroom_gate/headroom_gate.json`, `headroom_calibration.json`.

2. **Decode + Macedonian sanitization** — Beam search with optional length normalization; post-decode script filter maps non-MK Cyrillic to Macedonian graphemes.

3. **Per-edit + sentence log-prob gate** — After decoding, each proposed edit is scored by log-prob delta between corrected and noisy windows. A joint grid over `gate_log_prob_margin` and `gate_per_edit_margin` is swept on val (`gate_calibration_grid`); the pair with lowest mean CER wins (tie-break: more conservative). Proper-name and rare-word edits require extra margin (mirroring classical). A phase-2 confusion whitelist caps accepted single-char substitutions. State: `neural_gate_calibration.json`.

**Temperature scaling** (`calibrate_temperature`) fits a scalar `T` on val token NLL before gate scoring; persisted as `neural_temperature.json`.

Secondary seeds store calibrations under `models/neural/<regime>/seed<seed>/`.

Key behaviors:

- **`real_only`:** higher identity-pair ratio (0.6) + aggressive headroom → model rarely edits already-clean val text.
- **`synthetic_*`:** headroom allows more corrections; per-edit gate limits hallucinated swaps on proper names / rare words.

---

## Evaluation

Validation metrics (per model/regime/seed) include:

- **CER**, **WER**, **chrF** (character n-gram F-score, implemented in `metrics.py`)
- Sentence accuracy, correction / overcorrection rates, useful correction rate
- Rare-word and proper-name corruption diagnostics
- Calibration bins and expected calibration error (when confidence is available)
- **Per-domain** breakdown (`prose` vs `poetry`)
- Paired bootstrap significance tests

Test split predictions are written without using test labels for any tuning.

**Paper tables** (`phase4/eval/paper_tables.py`, written to `phase4/phase4_output/paper_tables/`):

| Table | Contents |
|-------|----------|
| `main_table.csv` | Primary-seed val metrics for all models/regimes |
| `cross_domain_table.csv` | Prose vs. poetry breakdown |
| `calibration.csv` | Reliability bins |
| `significance.csv` | Paired bootstrap vs. identity / classical |
| `seed_variance.csv` | Mean ± std across seeds |
| `headroom_curve.csv` | Binned headroom estimates vs. error reduction (neural) |
| `oracle_headroom.csv` | Oracle per-sentence lower bound vs. systems |
| `phase1_alignment_quality.csv` | Alignment quality summary |
| `kfold_summary.csv` | Mean ± std across k-fold folds |
| `kfold_manifest.json` | Pointers to per-fold `main_table.csv` files |

---

## Reproducibility Checklist

1. Keep splits synchronized — edit only `phase4/data/splits.py` (phase1/2 import it); k-fold rotations live in `KFOLD_TEST_SETS`.
2. Use **train-only** phase2/phase3 outputs for any statistic that informs training or synthetic data.
3. Do not tune thresholds on test predictions.
4. Run phase1 from `phase1/` for standalone use, or let phase4 invoke it.
5. If MPS/CUDA fails, use `--neural-device cpu`.
6. After a crash, re-run the same command; sentinels resume at the finest completed granularity (per model, per fold, or per sweep leg).

**Artifacts phase4 expects before a skip-heavy run:**

- `phase2/phase2_output_train_only/error_distribution.json`
- `phase3/phase3_output_train_only/structure_aware_noise/*_pairs.jsonl`
- `phase4/data/manifests/{real_only,synthetic_only,synthetic_plus_real}.jsonl` (if using `--skip-manifests`)

---

## What to Read First (new contributors)

1. `phase4/phase4_correction_models.py` — full orchestration, k-fold driver, CLI
2. `phase4/config.py` — hyperparameters and paths
3. `phase4/data/splits.py` — document splits, genres, k-fold rotations
4. `phase4/models/headroom_gate.py` + `phase4/models/byt5_corrector.py` — neural gating stack
5. `phase4/data/build_phase4_dataset.py` — manifest construction
6. `phase1/phase1_alignment.py` — alignment and cleaning logic

---

## Known Gaps

- No automated test suite or CI configuration.
- `.gitignore` is minimal; generated artifacts under `phase*_output/` may appear in `git status` unless ignored locally.
- Some books may exist only under `phase1_output/` if raw/corrected inputs were removed from `raw_ocr/` / `corrected_ocr/` after an earlier alignment run.
- K-fold runs are expensive (full retrain per fold); partial fold progress is resumable via `_FOLD_DONE.json` but aggregated `kfold_summary.csv` only includes folds whose `main_table.csv` exists.
