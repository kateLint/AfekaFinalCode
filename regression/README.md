# Kickstarter ML Pipelines 🧠📊

This project provides a set of powerful, scalable machine learning pipelines for classification and regression on Kickstarter project data. The pipelines include support for structured features, textual embeddings (MiniLM, RoBERTa, ModernBERT), readability and sentiment metrics, and SHAP-based explainability.

---

## 📁 Contents

### 1. `regression_pipeline_with_embeddings_and_shap.py`
- Predicts numerical outcomes like `pledged` or `pledged_ratio`.
- Supports LightGBM, XGBoost, RandomForest, and linear models.
- Includes:
  - PCA on MiniLM embeddings
  - Stratified train/test split
  - SHAP value visualizations
  - `BayesSearchCV` optimization

### 2. `xgboost_rf_automl_with_shap.py`
- Classification pipeline for `successful` vs `failed` projects.
- Models: LightGBM, XGBoost, CatBoost, RandomForest.
- Key Features:
  - Clean handling of embeddings
  - Extensive Bayesian optimization
  - Modular design for evaluation, SHAP, and metrics
  - Feature importance extraction and visualization

### 3. `ols_regrassion.py`
- Classic linear regression variants: OLS, Ridge, Lasso, ElasticNet.
- Used for interpretable modeling of pledged amount.
- SHAP and Lasso-based feature selection supported.

---

## 🧪 Example Input

```json
{
  "state": "successful",
  "story_miniLM": [0.123, -0.234, ...],  # 384-dim
  "Risks and Challenges Analysis": {
    "Readability Scores": {
      "Flesch Reading Ease": 58.4
    },
    ...
  },
  "projectFAQsCount": 2,
  "rewardscount": 5,
  "category_Web": true,
  ...
}
```

---

## ✅ Output

- Trained model `.pkl` files (e.g., `lightgbm_model.pkl`)
- Feature importance `.csv`
- SHAP `.png` explanations
- Summary results `.md`

---

## 🚀 Run Examples

```bash
python regression_pipeline_with_embeddings_and_shap.py
python xgboost_rf_automl_with_shap.py
python ols_regrassion.py
```

---

## 📦 Installation

Use the `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 📈 Requirements

- Python ≥ 3.8
- LightGBM, XGBoost, scikit-learn
- imbalanced-learn, SHAP, Optuna, scikit-optimize

---

## 🔍 Notes

- JSON structure must include consistent embedding and sentiment keys.
- Automatically handles missing values, constant features, and type issues.

