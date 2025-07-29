# Kickstarter Regression Analysis Pipeline

## Overview

This project performs a comprehensive regression-based analysis of Kickstarter campaigns, with a specific focus on modeling the `pledged_to_goal_ratio`. The pipeline includes:

* Feature extraction from JSON data (including embeddings)
* Preprocessing: imputation, scaling, feature selection
* Model training with Bayesian hyperparameter optimization (BayesSearchCV)
* Evaluation with both standard regression metrics and custom band accuracy
* SHAP analysis for feature explainability
* Visualizations of predicted and actual outcome distributions

## Features

* Support for LightGBM, XGBoost, RandomForest, and Ridge regression models
* Target variable transformations (log1p, capping, winsorizing)
* Stratified cross-validation using custom quantile binning
* SHAP importance plots (bar, beeswarm) + saved CSVs and summaries
* Custom band-based accuracy: Failed / Underfunded / Just Funded / Overfunded
* Clear and interpretable business insights

## Requirements

* Python 3.8+
* Key packages:

  * `scikit-learn`
  * `xgboost`
  * `lightgbm`
  * `shap`
  * `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`
  * `tqdm`, `joblib`, `skopt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
├── main.py                  # Entry point for running the full pipeline
├── results_regression_v4/   # Output directory with results and visualizations
├── all_good_projects_with_modernbert_embeddings_enhanced_with_miniLM12.json
├── model_comparison.csv     # Summary of all trained models and their metrics
├── model_summary.json       # Best model and R2-based rankings
├── actual_outcomes_summary_detailed.csv
├── predicted_outcomes_bands_detailed.png
├── shap_summary_<model>.json
├── shap_feature_importance_<model>.csv
...
```

## Running the Pipeline

To run the full pipeline on a dataset:

```bash
python main.py
```

Edit the file path inside the script (`data_path`) to point to your `.json` file.

## Input Data Format

* JSON list of Kickstarter projects
* Each entry must include:

  * `pledged_to_goal_ratio`
  * `goal`, `rewardscount`, `projectFAQsCount`, etc.
  * Sentiment and readability scores
  * MiniLM-L12 embeddings for story and risks sections

## Output Artifacts

* Trained model files (`best_<model>.joblib`)
* Hyperparameters (`best_<model>_params.json`)
* SHAP plots and CSVs for interpretability
* Outcome band bar charts (absolute + percentage)
* Model comparison CSV and summary JSON
* Full predictions on entire dataset

## Custom Functions & Utilities

* `DataPreprocessor`: Imputation, scaling, encoding, feature selection
* `make_stratified_bins`: Custom binning for regression stratification
* `evaluate_model`: Generates metrics and prediction plots
* `generate_enhanced_shap_analysis`: Computes and saves SHAP plots
* `create_detailed_outcome_bands_visualization`: For grouped band evaluation

## Notes

* SHAP explanations are restricted to non-embedding features
* Models are trained using log1p-transformed targets (with optional capping)
* Predictions are inverse-transformed with `np.expm1`
* Outlier detection is implemented but non-destructive by default

