# KickstarterStructure

[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/) [![LightGBM](https://img.shields.io/badge/LightGBM-1.6-orange)](https://lightgbm.readthedocs.io/en/stable/)

# AfekaFinalCode – The Impact of Textual Characterization on Project Success

This repository analyzes how textual and structural features of Kickstarter projects affect their success. It includes pipelines for hypothesis testing, machine learning classification, and text analysis.

## Structure

- **hypothesis/**: Scripts testing various hypotheses about project success, including sentiment, readability, risk analysis, and category effects.
- **training/**: LightGBM training pipeline with preprocessing for structured features, embeddings, sentiment, and readability.
- **classiffication_comparison/**: Compares machine learning models (XGBoost, RandomForest, LightGBM, CatBoost) using AutoML and SHAP for feature importance.
- **paraphrase/**: Tools for paraphrasing project narratives using T5 and GPT models.

## Main Files

- `hypothesis/h1.py` – Sentiment analysis and hypothesis testing.
- `hypothesis/h2.py` – Topic modeling with BERTopic and logistic regression.
- `hypothesis/h3_story.py` – Readability metrics and their impact on success.
- `hypothesis/h3_risks.py` – Risk section readability and project outcome.
- `hypothesis/h4.py` – Category-based analysis and visualizations.
- `hypothesis/h5.py` – Analyzes passive voice usage in project narratives and risks, outputs sentence-level statistics as CSV.

- `training/train_lighgbm.py` – LightGBM model training with embeddings and feature extraction.
- `classiffication_comparison/xgboost_rf_automl_with_shap.py` – AutoML experiments and SHAP explainability.
- `paraphrase/paraphrase_t5_gpt.py` – Paraphrase generation and keyphrase extraction.

## How to Use

1. See each subfolder’s README for details (e.g., `training/README.md`, `classiffication_comparison/README.md`).
2. Scripts require input datasets as JSON files (see code for paths).
3. Outputs include trained models, feature lists, classification reports, and visualizations.

## Requirements

- Python (various packages: pandas, numpy, sklearn, lightgbm, xgboost, catboost, transformers, bertopic, SHAP, etc.)
- See individual script headers and `requirements.txt` for details.

---

*This summary covers only a portion of the files. For the complete file list, browse the repository here: [AfekaFinalCode on GitHub](https://github.com/kateLint/AfekaFinalCode).*
