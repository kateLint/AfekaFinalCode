# Kickstarter Paraphrasing & Scoring Pipeline

This project evaluates and optimizes Kickstarter project texts (Story + Risks) by:
- Generating paraphrases with a T5-based model
- Embedding long texts using a RoBERTa sentence transformer with overlap-aware chunking
- Predicting success probability with a trained XGBoost classifier
- Searching sampling hyperparameters with Optuna to maximize predicted success
- Providing quick suggestions and an optional optimization routine

The repository includes robust logging, coherence scoring, and keyphrase extraction to help you iterate toward texts with higher projected success probability.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Environment & Dependencies](#environment--dependencies)
- [Configuration](#configuration)
- [Data & Features](#data--features)
- [Execution Flow](#execution-flow)
- [Models Loaded](#models-loaded)
- [Output Walkthrough (Your Run)](#output-walkthrough-your-run)
- [Common Issues & Debugging](#common-issues--debugging)
- [Quality Review & Improvements](#quality-review--improvements)
- [Project Organization](#project-organization)
- [FAQ](#faq)
- [Changelog](#changelog)

---

## Architecture Overview

### Main Components
1. **Paraphraser**: `humarin/chatgpt_paraphraser_on_T5_base` via `transformers` to produce multiple paraphrases.
2. **Embedder**: `sentence-transformers/roberta-base-nli-mean-tokens` for 768-d embeddings, with:
   - Tokenization to IDs
   - Overlap-aware chunking (max tokens = 512, stride = 64)
   - Decoding to text chunks
   - Batch embedding and **weighted pooling** by chunk length
   - Optional L2 normalization
3. **Classifier**: Pretrained XGBoost model (loaded from `*.pkl`) that expects a specific feature set loaded from `xgboost_feature_columns.json`.
4. **Keyphrase Extraction**: KeyBERT (using the same embedding space) for quick "theme" labeling.
5. **Coherence Scoring**: Sentence-level cosine similarity between original and paraphrase (mean across aligned sentences).
6. **Optimization**: Optuna TPE sampler sweeps `top_k`, `top_p`, and `temperature` to maximize classifier probability, with a coherence threshold gate.

---

## Environment & Dependencies

- Python 3.10+ recommended
- GPU optional (code runs on CPU as well)

**Core libraries**
- `transformers`, `sentence-transformers`, `torch`, `scikit-learn`, `xgboost` (through joblib-loaded model), `pandas`, `numpy`, `optuna`, `keybert`, `nltk`

**Optional**
- `psutil` (used in health check to report memory usage). Missing `psutil` only downgrades the health report; pipeline still runs.

**NLTK Data**
- Requires `punkt` for sentence tokenization. The code attempts to download if missing.

Install example:
```bash
pip install torch transformers sentence-transformers scikit-learn joblib pandas numpy optuna keybert nltk psutil
python -c "import nltk; nltk.download('punkt')"
```

---

## Configuration

All configuration is centralized in `Config`:
- **Model paths**
  - `model_path`: XGBoost classifier (`xgboost_kickstarter_success_model.pkl`)
  - `features_path`: JSON with **exact** feature order expected by the classifier
- **RoBERTa settings**: model name, embedding dimension (768), column prefixes for Story/Risks embeddings
- **Paraphraser settings**: maximum input tokens per chunk, stride, join mode (`sentence` or `paragraph`), `max_new_tokens`
- **Optimization**: number of Optuna trials, coherence cutoff

Validation enforces:
- Existence of `model_path` and `features_path`
- Positive `embed_dim`
- Coherence threshold in [0, 1]

Environment variables allow overrides (`KS_MODEL_PATH`, `KS_FEATURES_PATH`, `KS_ROBERTA_NAME`).

---

## Data & Features

### Input dictionary (`project_input`)
Minimal numeric keys expected:
- `goal`, `rewardscount`, `projectFAQsCount`, `project_length_days`, `preparation_days`

Example category flags:
- `category_Web_Development` (extend to your full category set)

### Text fields
- `story`: Long text describing the project
- `risks`: Long text describing risks

### Feature alignment
`build_base_dataframe` constructs a **single-row** DataFrame aligned to the classifier’s expected columns:
- Uses provided numeric & category features
- Adds **missing** expected columns with zeros
- Reorders to exact `clf_features` column order

### Embeddings
`fill_roberta_embeddings` writes Story/Risks embeddings into `story_roberta_embedding_{{i}}` and `risk_roberta_embedding_{{i}}` columns **if** they exist among `clf_features`.

---

## Execution Flow

1. **Logging & seeds**: Logging to console and `kickstarter_ai.log`, seeds set for reproducibility.
2. **Model loading**: Paraphraser, sentence embedder, classifier, feature list, KeyBERT, tokenizer.
3. **Health check** (best-effort): Tries embedding, a dummy prediction, memory stats, GPU info.
4. **Base DF**: Create aligned DataFrame and fill embeddings from original story/risks.
5. **Base prediction**: `predict_success_probability` on the original text.
6. **Quick suggestions**: Three preset sampling configs → paraphrase → filter by coherence → predict → rank.
7. **(Optional) Optuna**: Sweeps hyperparameters; keeps best paraphrase that passes coherence threshold.

---

## Models Loaded

- **Paraphraser**: `humarin/chatgpt_paraphraser_on_T5_base`
- **Embedder**: `sentence-transformers/roberta-base-nli-mean-tokens` (768-d)
- **Classifier**: XGBoost (loaded via `joblib`) using `{{features}}.json` column list

Hardware autodetection selects **CUDA** if available; otherwise CPU.

---

## Output Walkthrough (Your Run)

Below is a summary of the key events extracted from your console log:

### 1) Dependency / Health
- `psutil` missing → health report warns but continues.
- Health check initially failed due to function order: `predict_success_probability` not yet defined when `health_check()` ran. **Outcome**: printed a warning but the rest of the pipeline executed.

### 2) Model Loading
- All models loaded in ~8 seconds on CPU.
- Feature columns: **1,583** expected columns.

### 3) Base Prediction
- Original story success probability: **12.16%** (visual bar rendered).

### 4) Quick Suggestions
Three suggestions were generated; top candidate reached **14.42%**, with coherence ≥ 0.60.

### 5) Optuna Optimization (10 trials)
- Best trial achieved **19.60%** success probability with:
  - `top_k=35`, `top_p=0.9002`, `temperature=1.2399`
- Coherence check applied; output included a new paraphrase and theme tags.
- Explanations clarify trade-offs for `top_k`, `top_p`, `temperature`.

> **Interpretation**: The best paraphrase improved predicted success by **+7.44 pp** over the original (12.16% → 19.60%).

---

## Common Issues & Debugging

1. **`Health check failed: name 'predict_success_probability' is not defined`**
   - **Cause**: `health_check()` is called before the function is defined.
   - **Fix**: Move the definition of `predict_success_probability` **above** `health_check()`, or wrap the health check in a `try/except` that defers prediction until the function is available.

2. **Missing `psutil`**
   - **Symptom**: Warning only; memory part of health report disabled.
   - **Fix**: `pip install psutil` (optional).

3. **NLTK `punkt` missing**
   - **Symptom**: Errors in sentence tokenization.
   - **Fix**: `python -c "import nltk; nltk.download('punkt')"`

4. **Feature mismatch**
   - **Symptom**: `KeyError` or shape mismatch during prediction.
   - **Fix**: Ensure `features_path` JSON aligns exactly with the trained model; `build_base_dataframe` adds missing columns but **cannot** reconcile semantic mismatches.

5. **GPU memory errors**
   - **Fix**: Lower batch size in `_embed_text_chunks` (e.g., `batch_size=8`), or run on CPU.

6. **Slow paraphrase generation**
   - **Fix**: Reduce `num_return`, shorten `max_new_tokens`, or skip Optuna for quick runs.

---

## Quality Review & Improvements

### 1) Function & Module Ordering
- **Issue**: `health_check()` references `predict_success_probability` before it’s defined.
- **Action**: Move `predict_success_probability` **above** `health_check`, or inject a guarded stub during early initialization.

### 2) Naming & Consistency
- Keep consistent prefixes for embedding columns: `story_roberta_embedding_{{i}}`, `risk_roberta_embedding_{{i}}` (already consistent).
- Consider centralizing **all** magic numbers (512, 64, 16, 0.60) in `Config` (some are already there—complete the move).

### 3) Error Handling & Validation
- Add explicit checks in `fill_roberta_embeddings` to **verify presence** of expected embedding columns and warn once if missing.
- In `generate_paraphrases`, you call `model.generate` and then **ignore** its outputs by calling `paraphrase_long_text` unconditionally. Prefer one code path:
  - **Option A**: For **long** texts, **only** use `paraphrase_long_text` (chunked).  
  - **Option B**: If `len(tokens) <= threshold`, use the simple path; else use chunked path.
- Add timeouts to Optuna trials (e.g., `RDBSampler` + `timeout`) to prevent runaway trials.

### 4) Performance
- Cache sentence embeddings in `multi_sentence_coherence_score` to avoid recomputing for the same sentences across candidates.
- Reuse `df_template` more aggressively and only overwrite embedding columns to avoid DataFrame allocations.
- Consider using **FP16** for embedding on GPU to reduce memory and increase throughput (`SentenceTransformer` supports `torch_dtype=torch.float16` on CUDA).

### 5) Reproducibility
- You set seeds; also pin versions in `requirements.txt` and record `git` commit SHA and config snapshot at runtime in the logs.

### 6) Testing
- Add minimal tests:
  - Tokens → chunking (edge cases: very long single sentence, zero-length, extreme stride)
  - Pooling correctness (weights sum to 1)
  - Feature alignment and prediction shape
  - Paraphrase coherence gating behavior

### 7) CLI / API Structure
- Wrap main steps into a `main()`; expose CLI flags:
  - `--quick-only`, `--optuna-trials`, `--coherence-threshold`, `--num-return`.
- Return machine-readable JSON with the best candidate, probability, and parameters for downstream tooling.

### 8) Logging
- Avoid duplicated handlers on repeated runs by checking `logger.handlers` before adding.
- Use `logger.debug` for verbose traces and condition on `CONFIG.enable_logging`/`log_level`.

---

## Project Organization

Suggested structure:
```
project/
├─ src/
│  ├─ config.py
│  ├─ models.py               # load_models(), Models dataclass
│  ├─ embeddings.py           # tokenize/chunk/pool utilities
│  ├─ paraphrase.py           # paraphrase_long_text(), generate_paraphrases()
│  ├─ features.py             # build_base_dataframe(), fill_roberta_embeddings()
│  ├─ predict.py              # predict_success_probability()
│  ├─ optimize.py             # get_quick_suggestions(), optimize_paraphrase_optuna()
│  └─ health.py               # health_check()
├─ runs/                      # trained models + feature JSONs
├─ notebooks/                 # experiments
├─ tests/                     # unit tests (pytest)
├─ logs/
│  └─ kickstarter_ai.log
├─ README.md
└─ requirements.txt
```

Add a `Makefile` with shortcuts:
```makefile
quick:
	python -m src.main --quick-only

optuna:
	python -m src.main --optuna-trials=25
```

Track experiments with a simple CSV or `mlflow` (optional).

---

## FAQ

**Q: Why coherence threshold 0.60?**  
A: Empirically balances novelty vs. semantic drift; adjust per dataset using validation.

**Q: Can I swap the embedder?**  
A: Yes, but keep `EMBED_DIM` and feature columns in sync. Re-train or re-extract embeddings if the space changes.

**Q: Is KeyBERT mandatory?**  
A: No. It’s for quick interpretability; you can disable if speed is critical.

**Q: Can I optimize the Risks section too?**  
A: Yes—duplicate the flow for `risks` and merge both embeddings into the feature vector before prediction.

---

## Changelog

- **2025-08-10**: Initial README generated from a working run; documents pipeline, output, and known issues (health-check order).

---

## Next Steps (Actionable)

- [ ] Reorder definitions so `health_check` does not reference undefined functions.
- [ ] Unify paraphrase generation path (short vs. long text) to avoid duplicate work.
- [ ] Cache sentence embeddings in coherence scoring.
- [ ] Add CLI flags and JSON output for automation.
- [ ] Create unit tests for chunking, pooling, feature alignment, and gating.
- [ ] Consider experiment tracking (optuna storage, mlflow).
