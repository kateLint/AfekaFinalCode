# AutoML Classifier Comparison with SHAP

This project trains and evaluates multiple machine learning models (XGBoost, RandomForest) using Bayesian optimization and SHAP for interpretability. It supports model comparison with metrics like F1-score, precision, recall, and AUC.

## Features

- Supports multiple classifiers (XGBoost, RandomForest)
- Automated hyperparameter tuning with BayesSearchCV
- SHAP value analysis and feature importance plots
- Model evaluation: classification report, ROC curves, confusion matrix
- Easy customization for additional models

## Requirements

See `requirements.txt`.

## How to Run

```bash
python xgboost_rf_automl_with_shap.py
```

Make sure to update the dataset path and any config parameters at the top of the script.

## Output

- Trained models
- Classification metrics
- SHAP plots
- Visual comparison charts

## Author

Generated with help from ChatGPT.
