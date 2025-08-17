# Kickstarter Project Success Classifier 🎯

This repository contains a full-featured machine learning pipeline for predicting the success of Kickstarter technology projects. Built for research-grade analysis, it combines traditional structured features, transformer-based embeddings, Bayesian optimization, and SHAP explainability into a single powerful framework.

---

## ✨ Highlights

- **Multi-model comparison**: RandomForest, XGBoost, LightGBM, CatBoost
- **Advanced features**: Readability metrics, sentiment scores, GCI, and transformer embeddings (MiniLM, RoBERTa, ModernBERT)
- **Bayesian Optimization**: Efficient hyperparameter tuning using `BayesSearchCV`
- **Explainability**: SHAP summary and force plots per model
- **Visualization**: Prediction count comparisons, feature importance, success rate per model
- **Robust Evaluation**: Stratified k-fold CV with detailed metrics and result export

---

## 🚀 How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Place your Kickstarter dataset (`.json`) in the correct path.
    > Example path used: `/Users/kerenlint/Projects/Afeka/models/projects_with_short_stories_and_risks_cleaned_512_tokens.json`

3. Run the pipeline:
    ```bash
    python xgboost_rf_automl_with_shap.py
    ```

---

## 📂 Outputs

All results are saved under:

```bash
classification_results_weighted_compare_final/
```

Includes:

- CSVs with metrics, correlations, feature importance
- SHAP visualizations (summary + force plots)
- Markdown result tables per model configuration
- Aggregated prediction count visualizations

---

## 🧠 Feature Configurations

Each run evaluates several combinations of:

- Feature types:
    - Traditional
    - MiniLM / RoBERTa / ModernBERT embeddings
    - Embeddings only / All / Without embeddings
- Classifiers:
    - RandomForest
    - XGBoost
    - LightGBM
    - CatBoost

---

## 🧪 SHAP & Explainability

Models with SHAP enabled will generate:

- Summary plot of global feature importance
- Force plot for sample prediction
- CSV with SHAP values
- Top 20 features with positive/negative impact

---

## 📊 Performance Metrics

Tracked metrics include:

- F1 (weighted)
- Recall / Precision
- AUC (ROC)
- Specificity
- Prediction counts (successful vs failed)
- Fit time

## ⚠️ Notes

- Only structured and preprocessed JSON files are supported.
- Excludes known leakage features (e.g., pledged amount, backers).
- Default config: 5-fold outer CV, 5-fold tuning, 5 BayesSearch iterations.
