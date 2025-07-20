# 🧠 Kickstarter Success Classifier – Training Pipeline

This script trains a LightGBM classifier to predict Kickstarter project success (`successful` vs `failed`) using structured features, MiniLM embeddings, readability metrics, sentiment scores, and category indicators.

---

## 📥 Input

- JSON file containing project data with:
  - `state`, `story_miniLM`, `risks_miniLM`
  - Story and risk readability analysis
  - Sentiment scores
  - Metadata (goal, duration, FAQ count, etc.)
  - Category one-hot encodings

---

## 🔧 Preprocessing

1. **Target Label**: Binary encoding of `state` (`1 = successful`, `0 = failed`)
2. **Embedding Expansion**: `story_miniLM` and `risks_miniLM` are expanded into 384-dim features
3. **Readability & Sentiment**: Extracted from nested structures
4. **Preparation Days**: Computed from `created_at` and `launched_at`
5. **Combined Categories**: e.g., `category_Web_Combined`
6. **Final Feature List**: Includes base, category, and embedding features

---

## ⚙️ Model Training

- Algorithm: `LightGBMClassifier`
- Parameters:
  - `learning_rate=0.02742`
  - `n_estimators=363`
  - `num_leaves=41`
  - `subsample=0.9615`
  - `colsample_bytree=0.91198`

- Train/Test Split: 80/20 with stratification
- Metrics:
  - Accuracy
  - ROC-AUC
  - Classification report

---

## 📦 Output

- `lightgbm_kickstarter_success_model.pkl` – trained classifier
- `lightgbm_feature_columns.json` – ordered list of features used

---

## 📝 Example Usage

```bash
python train_kickstarter_classifier.py
```

---

## 📁 Notes

- Ensure input JSON includes valid MiniLM embeddings as lists
- All required fields must exist or will raise a `KeyError`
- Feature list is automatically validated and cleaned

---
