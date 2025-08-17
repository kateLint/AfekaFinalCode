# Kickstarter Project

# AfekaFinalCode – The Impact of Textual Characterization on Project Success

This repository analyzes how textual and structural features of Kickstarter projects affect their success. It includes pipelines for hypothesis testing, machine learning classification, and text analysis.

## Structure

- **hypothesis/**: Scripts testing various hypotheses about project success, including sentiment, readability, risk analysis, and category effects.
- **training/**: XgBooost training pipeline with preprocessing for structured features, embeddings, sentiment, and readability.
- **classiffication_comparison/**: Compares machine learning models (XGBoost, RandomForest, LightGBM, CatBoost) using AutoML and SHAP for feature importance.
- **paraphrase/**: Tools for paraphrasing project narratives using T5 and GPT models.

## How to Use

1. See each subfolder’s README for details (e.g., `training/README.md`, `classiffication_comparison/README.md`).
2. Scripts require input datasets as JSON files (see code for paths).
3. Outputs include trained models, feature lists, classification reports, and visualizations.

## Requirements

- Python (various packages: pandas, numpy, sklearn, lightgbm, xgboost, catboost, transformers, bertopic, SHAP, etc.)
- See individual script headers and `requirements.txt` for details.

## 👩‍🔬 Research Context

This project is part of a Master's thesis:

> **Title**: *Crowdfunding Technology Projects: The Impact of Textual Characterization on Project Success*  
> **Author**: Keren Lint  
> **Institution**: Afeka College of Engineering, 2025


