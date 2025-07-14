import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline as ImbPipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from collections import defaultdict

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    make_scorer,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone

import time
import os
import warnings
import traceback
import gc

# --- Basic Configuration ---
warnings.filterwarnings('ignore')
gc.enable()

# ==============================================================================
#                          HELPER FUNCTIONS
# ==============================================================================

def get_bayes_search_space(model_key):
    """
    Returns search space for BayesSearchCV for supported models.
    Supports: RandomForest, LightGBM, XGBoost, CatBoost.
    """
    if model_key == 'RandomForest':
        return {
            'classifier__n_estimators': Integer(100, 600),
            'classifier__max_depth': Integer(10, 30),
            'classifier__min_samples_split': Integer(2, 10),
            'classifier__min_samples_leaf': Integer(1, 5),
            'classifier__max_features': Categorical(['sqrt', 0.3, 0.5])
        }

    elif model_key == 'LightGBM':
        return {
            'classifier__n_estimators': Integer(100, 600),
            'classifier__learning_rate': Real(0.01, 0.1, prior='log-uniform'),
            'classifier__num_leaves': Integer(20, 50),
            'classifier__colsample_bytree': Real(0.6, 1.0),
            'classifier__subsample': Real(0.6, 1.0)
        }

    elif model_key == 'XGBoost':
        return {
            'classifier__n_estimators': Integer(100, 600),
            'classifier__learning_rate': Real(0.01, 0.1, prior='log-uniform'),
            'classifier__max_depth': Integer(3, 9),
            'classifier__subsample': Real(0.6, 1.0),
            'classifier__colsample_bytree': Real(0.6, 1.0),
            'classifier__gamma': Real(0, 1)
        }

    elif model_key == 'CatBoost':
        return {
            'classifier__iterations': Integer(100, 600),
            'classifier__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'classifier__depth': Integer(4, 10),
            'classifier__l2_leaf_reg': Integer(1, 9),
            'classifier__subsample': Real(0.6, 1.0),
            'classifier__border_count': Integer(32, 128)
        }

    # Default empty dict if model not found
    return {}

# --- Helper function for safe float conversion ---
def safe_float(value, default=np.nan):
    """Safely converts a value to float, handling None, non-numeric, and infinities."""
    if value is None: return default
    try:
        f_val = float(value)
        return default if not np.isfinite(f_val) else f_val
    except (ValueError, TypeError): return default

# --- 1. Extract Features (Corrected for Data Leakage, includes Goal) ---
def extract_features(data, embedding_type=None):
    features = []
    embedding_dimensions = 0
    target_counts = {'successful': 0, 'other': 0, 'skipped_no_state': 0}
    processed_ids = set()
    embedding_keys = {
        'minilm': ('story_miniLM', 'risks_miniLM'),
        'roberta': ('story_roberta', 'risks-and-challenges_roberta'),
        'modernbert': ('story_modernbert', 'risks_modernbert')
    }
    print(f"Starting feature extraction for {len(data)} items...")
    extraction_errors = 0

    for idx, item in enumerate(data):
        if idx > 0 and idx % 10000 == 0: print(f"  Processed {idx}/{len(data)} items...")
        try:
            project_id = item.get('id')
            if project_id is None or not isinstance(project_id, (int, float, str)) or str(project_id).strip() == "":
                extraction_errors += 1; continue

            project_state = item.get('state')
            if project_state == 'successful': target_class = 1; target_counts['successful'] += 1
            elif project_state in ['failed', 'canceled', 'suspended']: target_class = 0; target_counts['other'] += 1
            else: target_counts['skipped_no_state'] += 1; extraction_errors += 1; continue

            feature_dict = {
                'projectFAQsCount': safe_float(item.get('projectFAQsCount'), default=0.0),
                #'commentsCount': safe_float(item.get('commentsCount'), default=0.0),
                #'updateCount': safe_float(item.get('updateCount'), default=0.0),
                'rewardscount': safe_float(item.get('rewardscount'), default=0.0),
                'project_length_days': safe_float(item.get('project_length_days'), default=np.nan),
                'preparation_days': safe_float(item.get('preparation_days'), default=np.nan),
                'Story_Avg_Sentence_Length': safe_float(item.get('Story Analysis', {}).get('Average Sentence Length'), default=np.nan),
                'Story_Flesch_Reading_Ease': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Flesch Reading Ease'), default=np.nan),
                'Story_Flesch_Kincaid_Grade': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Flesch-Kincaid Grade Level'), default=np.nan),
                'Story_Gunning_Fog': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Gunning Fog Index'), default=np.nan),
                'Story_SMOG': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('SMOG Index'), default=np.nan),
                'Story_ARI': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Automated Readability Index'), default=np.nan),
                'Risks_Avg_Sentence_Length': safe_float(item.get('Risks and Challenges Analysis', {}).get('Average Sentence Length'), default=np.nan),
                'Risks_Flesch_Reading_Ease': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Flesch Reading Ease'), default=np.nan),
                'Risks_Flesch_Kincaid_Grade': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Flesch-Kincaid Grade Level'), default=np.nan),
                'Risks_Gunning_Fog': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Gunning Fog Index'), default=np.nan),
                'Risks_SMOG': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('SMOG Index'), default=np.nan),
                'Risks_ARI': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Automated Readability Index'), default=np.nan),
                'Story_Positive': safe_float(item.get('Story_Positive'), default=np.nan), 'Story_Neutral': safe_float(item.get('Story_Neutral'), default=np.nan),
                'Story_Negative': safe_float(item.get('Story_Negative'), default=np.nan), 'Story_Compound': safe_float(item.get('Story_Compound'), default=np.nan),
                'Risks_Positive': safe_float(item.get('Risks_Positive'), default=np.nan), 'Risks_Neutral': safe_float(item.get('Risks_Neutral'), default=np.nan),
                'Risks_Negative': safe_float(item.get('Risks_Negative'), default=np.nan), 'Risks_Compound': safe_float(item.get('Risks_Compound'), default=np.nan),
                'Story_GCI': safe_float(item.get('Story GCI'), default=np.nan), 'Risks_GCI': safe_float(item.get('Risks and Challenges GCI'), default=np.nan),
                'target_class': target_class
            }
            feature_dict["category_Web_Combined"] = 1 if item.get("category_Web", False) or item.get("category_Web Development", False) else 0
            for key, value in item.items():
                if key.startswith("category_") and key not in ["category_Technology", "category_Web", "category_Web Development"]:
                    is_true = value is True or (isinstance(value, (str, int)) and str(value).lower() in ['true', '1'])
                    feature_dict[key] = 1 if is_true else 0

            # --- Handle Embeddings if requested ---
            current_item_emb_dim = 0
            embedding_type_lower = embedding_type.lower() if embedding_type else None
            if embedding_type_lower in embedding_keys:
                story_key, risks_key = embedding_keys[embedding_type_lower]
                story_emb = item.get(story_key, [])
                risks_emb = item.get(risks_key, [])

                def add_embeddings_to_dict(prefix, embeddings, target_dict):
                    nonlocal current_item_emb_dim
                    if embeddings and isinstance(embeddings, list) and len(embeddings) > 0:
                        dim = len(embeddings); emb_vals = []; error = False
                        for x in embeddings:
                            try:
                                val = float(x) if x is not None else 0.0
                                emb_vals.append(0.0 if not np.isfinite(val) else val)
                            except (ValueError, TypeError): emb_vals.append(0.0); error = True
                        if len(emb_vals) != dim: emb_vals.extend([0.0] * (dim - len(emb_vals)))
                        if not emb_vals: return False
                        current_item_emb_dim = dim
                        for i, val in enumerate(emb_vals): target_dict[f'{prefix}_emb_{i}'] = val
                        return True
                    return False

                story_added = add_embeddings_to_dict(f'story_{embedding_type_lower}', story_emb, feature_dict)
                risks_added = add_embeddings_to_dict(f'risks_{embedding_type_lower}', risks_emb, feature_dict)
                if embedding_dimensions == 0 and current_item_emb_dim > 0 and (story_added or risks_added):
                    embedding_dimensions = current_item_emb_dim
            # --- End Embeddings ---
            features.append(feature_dict)
        except Exception as e: extraction_errors += 1; continue

    print(f"Finished extraction. Processed: {len(data)-extraction_errors}, Skipped/Errors: {extraction_errors}")
    print(f"Target counts: Success={target_counts['successful']}, Other={target_counts['other']}, Skipped={target_counts['skipped_no_state']}")
    df = pd.DataFrame(features)
    if df.empty or 'target_class' not in df.columns: print("CRITICAL: No data or target after extraction."); return pd.DataFrame(), 0

    df['target_class'] = pd.to_numeric(df['target_class'], errors='coerce')
    df.dropna(subset=['target_class'], inplace=True); df['target_class'] = df['target_class'].astype(int)
    if df.empty: print("CRITICAL: No data after target cleaning."); return pd.DataFrame(), 0

    print("Starting feature post-processing...")
    numeric_feature_cols = [col for col in df.columns if col != 'target_class']
    for col in numeric_feature_cols:
        if df[col].dtype == 'object': df[col] = pd.to_numeric(df[col], errors='coerce')
        if np.isinf(df[col].values).any(): df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isnull().any():
            fill_value = 0.0
            if not ('_emb_' in col or 'Count' in col or 'count' in col or col.startswith('category_')):
                finite_vals = df[col][np.isfinite(df[col])]
                if not finite_vals.empty: fill_value = finite_vals.median()
            df[col].fillna(fill_value, inplace=True)
        try: df[col] = df[col].astype(float)
        except ValueError: df[col] = 0.0
    print("Checking for constant columns...")
    constant_cols = [col for col in df.columns if col != 'target_class' and df[col].nunique(dropna=False) <= 1]
    if constant_cols: df.drop(columns=constant_cols, inplace=True); print(f"Dropped {len(constant_cols)} constant columns.")
    else: print("No constant columns found.")
    print(f"Final DataFrame shape: {df.shape}")
    if df.shape[1] <= 1: print("CRITICAL: No features remaining."); return pd.DataFrame(), 0

    final_emb_cols = [col for col in df.columns if '_emb_' in col]
    approx_dim = 0
    if final_emb_cols:
        prefixes = set(c.split('_emb_')[0] for c in final_emb_cols)
        approx_dim = len(final_emb_cols) // len(prefixes) if prefixes else len(final_emb_cols)

    return df, approx_dim

# --- 2. Preprocessing Function (No SMOTE) ---
# (Keep preprocess_data_base as previously defined)
def preprocess_data_base(X, y):
    """
    Base preprocessing: Impute missing values using mean. No scaling or normalization.
    Returns imputed X as numpy array, y as array, and original column names.
    """
    print(f"Base Preprocessing data shape: X={X.shape}, y={len(y)}")

    if not isinstance(X, pd.DataFrame):
        try:
            X = pd.DataFrame(X)
            original_columns = [f"feature_{i}" for i in range(X.shape[1])]
        except Exception as e:
            print(f"CRITICAL ERROR: Could not convert X to DataFrame: {e}")
            return None, None, None
    else:
        original_columns = X.columns.tolist()

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=np.number).columns
    X = X[numeric_cols]

    if X.empty:
        print("CRITICAL ERROR: No numeric columns left after filtering.")
        return None, None, None

    if np.isinf(X.values).any():
        print("Found infinity values in X. Replacing with NaN.")
        X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().sum().sum() > 0:
        print(f"NaN values in X before imputation: {X.isna().sum().sum()}")

    # Impute missing values with column mean
    imputer = SimpleImputer(strategy='mean')
    try:
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    except Exception as e:
        print(f"Error during imputation: {e}. Filling NaNs with 0.")
        X_imputed = X.fillna(0)

    y_array = y.values if isinstance(y, pd.Series) else np.array(y)

    print(f"Preprocessing complete. Final X shape: {X_imputed.shape}")
    return X_imputed.values, y_array, original_columns


# --- 3. Define Specificity Scorer ---
# (Keep specificity function as previously defined)
def specificity(y_true, y_pred):
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2): tn, fp, fn, tp = cm.ravel(); return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
             unique_true = np.unique(y_true)
             if len(unique_true) == 1: return 1.0 if unique_true[0] == 0 and np.all(y_pred == 0) else 0.0
             elif 0 in unique_true:
                 is_true_neg = (np.array(y_true) == 0); is_pred_neg = (np.array(y_pred) == 0)
                 tn = np.sum(is_true_neg & is_pred_neg); fp = np.sum(is_true_neg & ~is_pred_neg)
                 return tn / (tn + fp) if (tn + fp) > 0 else 0.0
             else: return 0.0
    except Exception as e: print(f"Error calculating specificity: {e}. Returning 0."); return 0.0
specificity_scorer = make_scorer(specificity, greater_is_better=True)


# --- 4. Get Feature Importance ---
# (Keep get_feature_importance function as previously defined)
def get_feature_importance(model, feature_names, model_name_full, output_dir):
    """ Calculates feature importance for fitted models (handles pipelines). """
    print(f"\nCalculating feature importance for {model_name_full}...")
    importances = None; final_estimator = model; final_estimator_name = "N/A"
    if hasattr(model, 'steps'):
        try:
            if isinstance(model, (Pipeline, ImbPipeline)):
                 estimator_step = None
                 for step_name, step_estimator in reversed(model.steps):
                     if hasattr(step_estimator, 'feature_importances_') or hasattr(step_estimator, 'coef_'):
                         estimator_step = (step_name, step_estimator); break
                 if estimator_step:
                     final_estimator_step_name, final_estimator = estimator_step
                     final_estimator_name = final_estimator.__class__.__name__
                     print(f"  Found estimator step in pipeline: '{final_estimator_step_name}' ({final_estimator_name})")
                 else:
                     final_estimator_step_name, final_estimator = model.steps[-1]
                     final_estimator_name = final_estimator.__class__.__name__
                     print(f"  Using last step as estimator: '{final_estimator_step_name}' ({final_estimator_name}) - May lack importance.")
            else: final_estimator_name = model.__class__.__name__
        except Exception as e: print(f"  Error accessing steps: {e}. Using model directly."); final_estimator_name = model.__class__.__name__
    else: final_estimator_name = model.__class__.__name__
    try:
        if hasattr(final_estimator, 'feature_importances_'): importances = final_estimator.feature_importances_; importance_type = "feature_importances_"
        elif hasattr(final_estimator, 'coef_'):
            if final_estimator.coef_.ndim > 1: importances = np.mean(np.abs(final_estimator.coef_), axis=0)
            else: importances = np.abs(final_estimator.coef_)
            importance_type = "abs(coef_)"
        elif final_estimator_name == 'CatBoostClassifier' and hasattr(model, 'get_feature_importance'):
             try: importances = model.get_feature_importance(); importance_type = "CatBoost_get_feature_importance"; print("  Using CatBoost's get_feature_importance().")
             except Exception as cat_imp_e: print(f"  Failed CatBoost importance: {cat_imp_e}"); return pd.DataFrame()
        else: print(f"  Warning: Estimator '{final_estimator_name}' lacks importance attributes."); return pd.DataFrame()
        print(f"  Successfully extracted importance ({importance_type}) from {final_estimator_name}.")
        num_importances = len(importances); feature_names_processed = feature_names
        if feature_names is None: feature_names_processed = [f"Feature_{i}" for i in range(num_importances)]; print("  Warning: No feature names. Using generic.")
        elif len(feature_names) != num_importances: feature_names_processed = [f"Importance_{i}" for i in range(num_importances)]; print(f"  Warning: Mismatch names ({len(feature_names)}) vs importances ({num_importances}). Using generic.")
        feature_importance_df = pd.DataFrame({'Feature': feature_names_processed, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        feature_importance_df = feature_importance_df[feature_importance_df['Importance'] > 1e-9]
        if feature_importance_df.empty: print("  No features with importance > 1e-9."); return pd.DataFrame()
        print(f"  Feature Importances (Top 50):"); print(feature_importance_df.head(50).to_string(index=False))
        os.makedirs(output_dir, exist_ok=True); safe_model_name = model_name_full.replace(' ', '_').replace('+', '').replace('&','').replace('/','_').replace(':','_')
        save_path = os.path.join(output_dir, f"{safe_model_name}_feature_importance.csv")
        try: feature_importance_df.to_csv(save_path, index=False); print(f"\n  Full feature importance saved to {save_path}")
        except Exception as save_e: print(f"\n Error saving importance: {save_e}")
        return feature_importance_df
    except Exception as e: print(f"  Error calculating importance: {e}"); traceback.print_exc(); return pd.DataFrame()


# --- 5 & 6. Reporting and Visualization ---
# (Keep print_formatted_results, plot_classification_results, visualize_feature_importance as before)
def print_formatted_results(results_list, output_file=None):
    if not results_list: print("No results to format."); return ""
    output_lines = []; headers = ["Model Config", "Classifier", "F1 (w)", "Recall (w)", "Prec (w)", "Specificity", "AUC", "Fit Time(s)", "WEIGHT K", "Best Params / Status"]
    output_lines.append("| " + " | ".join(headers) + " |"); output_lines.append("|" + "---|"*len(headers))
    results_list.sort(key=lambda x: x.get('F1 Weighted', -float('inf')) if pd.notna(x.get('F1 Weighted')) else -float('inf'), reverse=True)
    for result in results_list:
        f1_w = f"{result.get('F1 Weighted', np.nan):.4f}"; rec_w = f"{result.get('Recall Weighted', np.nan):.4f}"
        prec_w = f"{result.get('Precision Weighted', np.nan):.4f}"; spec = f"{result.get('Specificity', np.nan):.4f}"
        auc = f"{result.get('AUC', np.nan):.4f}"; fit_time = f"{result.get('Mean Fit Time', np.nan):.2f}"
        smote_k = str(result.get('WEIGHT K', 'N/A'))
        params = result.get('Best Params', {}); params_str = "N/A"; status = result.get('Optimization Status', 'OK')
        if status != "OK": params_str = status
        elif isinstance(params, dict) and 'error' in params: params_str = f"Error: {params['error'][:100]}"
        elif isinstance(params, dict) and params:
             cleaned = {k.split('__')[-1]: v for k, v in params.items()}; p_list = [f"{k}:{v:.4g}" if isinstance(v, float) else f"{k}:{v}" for k, v in cleaned.items()]
             params_str = ", ".join(p_list); params_str = params_str[:147] + "..." if len(params_str) > 150 else params_str
        elif isinstance(params, str): params_str = params
        params_str = params_str.replace('|', ';')
        line_values = [str(result.get('Model Config', 'N/A')), str(result.get('Classifier', 'N/A')), f1_w, rec_w, prec_w, spec, auc, fit_time, smote_k, params_str]
        output_lines.append("| " + " | ".join(line_values) + " |")
    formatted_text = "\n".join(output_lines); print("\n--- Classification Results Summary (Markdown Format) ---\n" + formatted_text + "\n--- End Summary ---")
    if output_file:
        try:
             os.makedirs(os.path.dirname(output_file), exist_ok=True)
             header_content = f"""# Classification Results
CV: Stratified {outer_cv_folds}-Fold
Tuning: BayesSearch(n={n_search_iter}, cv={search_cv_folds})
WEIGHT: Config-dependent
"""
             with open(output_file, 'w', encoding='utf-8') as f: f.write(header_content + formatted_text)
             print(f"Formatted results saved to {output_file}")
        except Exception as e: print(f"Error saving formatted results: {e}")
    return formatted_text

def plot_classification_results(results_df, save_dir):
    if results_df.empty: print("No results to plot."); return
    viz_dir = os.path.join(save_dir, 'visualizations'); os.makedirs(viz_dir, exist_ok=True); sns.set(style='whitegrid', context='talk')
    metrics = ['F1 Weighted', 'Recall Weighted', 'Precision Weighted', 'Specificity', 'AUC', 'Mean Fit Time']
    try: results_df['Full Model Name'] = results_df['Model Config'] + " | " + results_df['Classifier']; y_col = 'Full Model Name'
    except KeyError: print("Warning: Missing columns for plot labels."); y_col = results_df.index
    num_models = len(results_df); plot_height = max(8, 0.4 * num_models)
    for metric in metrics:
        if metric not in results_df.columns: print(f"Metric '{metric}' not found."); continue
        plt.figure(figsize=(16, plot_height)); asc = metric == 'Mean Fit Time'
        plot_df = results_df.dropna(subset=[metric]).copy()
        if plot_df.empty: print(f"No valid data for '{metric}'."); plt.close(); continue
        plot_df.sort_values(metric, ascending=asc, inplace=True); sns.barplot(x=metric, y=y_col, data=plot_df, palette='viridis')
        title = f'{metric} Comparison (Stratified {outer_cv_folds}-Fold CV)'; xlbl = f'{metric}' + (' (s) (Lower Better)' if asc else ' (Higher Better)')
        plt.title(title, fontsize=16); plt.xlabel(xlbl, fontsize=14); plt.ylabel('Model Config | Classifier', fontsize=14); plt.xticks(fontsize=12); plt.yticks(fontsize=10)
        if not asc and not plot_df.empty: min_v, max_v = plot_df[metric].min(), plot_df[metric].max(); plt.xlim(left=max(0, min_v-0.05), right=min(1.0, max_v+0.05))
        elif not plot_df.empty: plt.xlim(left=0)
        plt.tight_layout(); safe_name = metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/','_'); save_path = os.path.join(viz_dir, f"{safe_name}_comparison.png")
        try: plt.savefig(save_path, bbox_inches='tight'); print(f"Saved {metric} plot: {save_path}")
        except Exception as e: print(f"Error saving {metric} plot: {e}")
        plt.close()
    print(f"\nVisualizations saved to {viz_dir}")

def visualize_feature_importance(feature_importance_dict, top_n=25, save_path=None):
    valid_dict = {k: v for k, v in feature_importance_dict.items() if isinstance(v, pd.DataFrame) and not v.empty}
    if not valid_dict: print("No feature importance data to visualize."); return
    n_models = len(valid_dict); fig_height = max(8, top_n * 0.35) * n_models
    fig, axes = plt.subplots(n_models, 1, figsize=(15, fig_height), squeeze=False, sharex=True); sorted_names = sorted(valid_dict.keys())
    for idx, model_name in enumerate(sorted_names):
        df = valid_dict[model_name]; ax = axes[idx, 0]; n_plot = min(top_n, len(df)); top = df.head(n_plot)
        if n_plot == 0: ax.text(0.5, 0.5, 'No important features', ha='center', va='center', transform=ax.transAxes); ax.set_title(f'{model_name}', fontsize=16)
        else: sns.barplot(x='Importance', y='Feature', data=top, ax=ax, palette="viridis", orient='h'); ax.set_title(f'Top {n_plot} Features: {model_name}', fontsize=16); ax.set_ylabel('Feature', fontsize=14); plt.setp(ax.get_yticklabels(), fontsize=10); plt.setp(ax.get_xticklabels(), fontsize=12)
    if n_models > 0: axes[-1, 0].set_xlabel('Importance Score', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    if save_path:
        try: os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, bbox_inches='tight'); print(f"Feature importance plot saved: {save_path}")
        except Exception as e: print(f"Error saving importance plot: {e}")
    plt.close(fig)


# ==============================================================================
#                   MAIN CLASSIFICATION ANALYSIS FUNCTION
# ==============================================================================

def main_classification():
    """
    Main function for classification comparing specified feature sets across models.
    """
    global output_dir, model_configs
    output_dir = 'classification_results_weighted_compare_final'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {os.path.abspath(output_dir)}")

    # --- Load Data ---
    print("\nLoading data...")
    try:
        data_path = '/Users/kerenlint/Projects/Afeka/models/all_good_projects_with_modernbert_embeddings_enhanced_with_miniLM12.json'
        print(f"Attempting load from: {data_path}")
        if not os.path.exists(data_path): print(f"CRITICAL Error: Data file not found: {data_path}"); return
        with open(data_path, 'r') as f: data = json.load(f)
        print(f"Loaded {len(data)} projects.")
        if not isinstance(data, list) or not data: print("CRITICAL Error: Data not a list or empty."); return
    except Exception as e: print(f"CRITICAL Error loading data: {e}"); traceback.print_exc(); return

    all_results = []; feature_importance_dict = {}

    # --- Define Model Configurations ---
   # model_configs = [
     #   ("Traditional_Features_Only_Weight_FIX", None, "NO_EMBEDDINGS", False),
    #    ("PreLaunch_Features_Only_Weight", None, "PRE_LAUNCH_ONLY", False),

    #("Traditional_Features_Only_Weight", None, "NO_EMBEDDINGS", False),
    #("Features_MiniLM_Weight", 'minilm', None, False),
    #("Features_RoBERTa_Weight", 'roberta', None, False),
    #("Features_ModernBERT_Weight", 'modernbert', None, False),
    #("MiniLM_Only_Weight", 'minilm', "ONLY_EMBEDDINGS", False),
    #("RoBERTa_Only_Weight", 'roberta', "ONLY_EMBEDDINGS", False),
    #("ModernBERT_Only_Weight", 'modernbert', "ONLY_EMBEDDINGS", False),

  #  ]


    model_configs = [
       # ("PreLaunch_Features_Only_Weight", None, "PRE_LAUNCH_ONLY", False),
        #("PreLaunch_Features_MiniLM_Weight", 'minilm', "PRE_LAUNCH_ONLY", False),
       # ("PreLaunch_Features_RoBERTa_Weight", 'roberta', "PRE_LAUNCH_ONLY", False),
      #  ("PreLaunch_Features_ModernBERT_Weight", 'modernbert', "PRE_LAUNCH_ONLY", False),
      #  ("MiniLM_PreLaunch_Only_Weight", 'minilm', "ONLY_EMBEDDINGS", False),
       # ("RoBERTa_PreLaunch_Only_Weight", 'roberta', "ONLY_EMBEDDINGS", False),
       # ("ModernBERT_PreLaunch_Only_Weight", 'modernbert', "ONLY_EMBEDDINGS", False),
           ("LightGBM_Feature_Comparison", None, "NO_EMBEDDINGS", False)

    ]

    print(f"\nDefined {len(model_configs)} specific feature configurations for comparison:")
    for cfg in model_configs: print(f"  - {cfg[0]} (Embedding: {cfg[1]}, Filter: {cfg[2]}, WEIGHT: {cfg[3]})")

    # --- Define Classifiers ---
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, verbosity=0, n_jobs=-1),
       # 'CatBoost': CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False, thread_count=-1)
    }
    print(f"\nClassifiers to compare: {list(classifiers.keys())}")

    global n_search_iter, search_cv_folds, outer_cv_folds
    cv_strategy_outer = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
    print(f"BayesSearch: n_iter={n_search_iter}, cv={search_cv_folds}, n_jobs=-1")
    print(f"Outer Evaluation: Stratified {outer_cv_folds}-Fold CV, n_jobs=-1")

    # --- Loop Through Feature Configurations ---
    for config_index, (config_name, embedding_type, feature_filter_marker, apply_smote_config) in enumerate(model_configs):
        print("\n" + "="*80 + f"\nProcessing Config {config_index + 1}/{len(model_configs)}: {config_name}")
        print(f"  Embedding: {embedding_type}, Filter: {feature_filter_marker}, SMOTE: {apply_smote_config}\n" + "="*80)
        start_time_config = time.time()
        X_df_full = y_original_df = X_features = None; gc.collect(); original_feature_names = None

        try:
            # 1. Extract Features
            print("  Step 1: Extracting features...")
            X_df_full, _ = extract_features(data, embedding_type=embedding_type)
            if X_df_full is None or X_df_full.empty or 'target_class' not in X_df_full.columns or X_df_full.shape[1] <= 1:
                print(f"  Skipping '{config_name}': Extraction failed."); continue
            y_original_df = X_df_full.pop('target_class'); X_features = X_df_full
            print(f"  Features extracted (before filter): {X_features.shape}")

            # Apply Feature Filter Marker
            if feature_filter_marker == "NO_EMBEDDINGS":
                embedding_cols_to_drop = [col for col in X_features.columns if '_emb_' in col]
                if embedding_cols_to_drop:
                    print(f"  Applying 'NO_EMBEDDINGS' filter: Dropping {len(embedding_cols_to_drop)} embedding columns.")
                    X_features = X_features.drop(columns=embedding_cols_to_drop)
                    print(f"  Features shape after dropping embeddings: {X_features.shape}")
                else:
                    print("  'NO_EMBEDDINGS' filter: No embedding columns found to drop.")
            elif feature_filter_marker == "ONLY_EMBEDDINGS":
                embedding_cols_to_keep = [col for col in X_features.columns if '_emb_' in col]
                if embedding_cols_to_keep:
                    print(f"  Applying 'ONLY_EMBEDDINGS' filter: Keeping only {len(embedding_cols_to_keep)} embedding columns.")
                    X_features = X_features[embedding_cols_to_keep]
                    print(f"  Features shape after filtering: {X_features.shape}")
                else:
                    print("  Warning: No embedding columns found to keep. Skipping this config.")
                    continue
            elif feature_filter_marker == "PRE_LAUNCH_ONLY":
                non_pre_launch_features = ['commentsCount', 'updateCount'] #'projectFAQsCount', 'rewardscount', 
                features_to_drop = [col for col in X_features.columns if col in non_pre_launch_features]
                if features_to_drop:
                    print(f"  Applying 'PRE_LAUNCH_ONLY' filter: Dropping {len(features_to_drop)} non-pre-launch columns: {features_to_drop}")
                    X_features = X_features.drop(columns=features_to_drop)
                    print(f"  Features shape after filtering: {X_features.shape}")
                else:
                    print("  'PRE_LAUNCH_ONLY' filter: No non-pre-launch columns found to drop.")

            # Save Feature List
            original_feature_names = X_features.columns.tolist()
            feature_list_filename = os.path.join(output_dir, f"{config_name}_feature_list.txt")
            try:
                with open(feature_list_filename, 'w') as f_feat:
                    for feature_name in original_feature_names: f_feat.write(f"{feature_name}\n")
                print(f"  Saved final feature list ({len(original_feature_names)} features) to: {feature_list_filename}")
                
                # DummyClassifier baseline
                if not X_features.empty and original_feature_names:
                    print("Running DummyClassifier baseline...")
                    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
                    dummy_scores = cross_val_score(dummy, X_features, y_original_df, scoring='f1_weighted', cv=cv_strategy_outer, n_jobs=-1)
                    dummy_f1_mean = dummy_scores.mean()
                    dummy_f1_std = dummy_scores.std()
                    print(f"DummyClassifier F1 Weighted (mean ± std): {dummy_f1_mean:.4f} ± {dummy_f1_std:.4f}")
                    all_results.append({
                        'Model Config': f"{config_name}_Dummy",
                        'Classifier': 'DummyClassifier',
                        'F1 Weighted': dummy_f1_mean,
                        'F1 Weighted Std': dummy_f1_std,
                        'Recall Weighted': np.nan,
                        'Precision Weighted': np.nan,
                        'Specificity': np.nan,
                        'AUC': np.nan,
                        'Mean Fit Time': np.nan,
                        'WEIGHT K': 'N/A',
                        'Best Params': 'strategy=most_frequent',
                        'Optimization Status': 'Baseline'
                    })
                else:
                    print("Skipping DummyClassifier: No features available")

                # Feature correlations with target_class
                if not X_features.empty and original_feature_names:
                    print("Calculating feature correlations with target_class...")
                    correlations = X_features.join(y_original_df).corr(numeric_only=True)['target_class'].abs().sort_values(ascending=False)
                    correlation_path = os.path.join(output_dir, f"{config_name}_feature_correlations.csv")
                    correlations.to_csv(correlation_path)
                    print(f"Feature correlations saved to: {correlation_path}")
                    print("Top 20 correlations:\n", correlations.head(20))
                    high_corr = correlations[correlations > 0.8]
                    if not high_corr.empty:
                        print(f"WARNING: Found {len(high_corr)} features with high correlation (>0.8) to target_class:")
                        print(high_corr)
                else:
                    print("Skipping correlation calculation: No features available")
            except Exception as feat_save_e:
                print(f"  Warning: Could not save feature list: {feat_save_e}")

        except Exception as data_prep_e:
            print(f"  CRITICAL Error Prep: {data_prep_e}"); traceback.print_exc(); continue
        finally:
            del X_df_full; gc.collect()

        if X_features.empty or not original_feature_names:
            print(f"  Skipping '{config_name}': No features."); continue

        # 2. Base Preprocessing
        X_processed_base, y_original_array, processed_feature_names = None, None, None
        try:
            print("  Step 2: Base preprocessing...")
            X_processed_base, y_original_array, processed_feature_names = preprocess_data_base(X_features.copy(), y_original_df.copy())
            if X_processed_base is not None and X_processed_base.shape[1] > 10:
                print("Running PCA to check feature redundancy...")
                pca = PCA(n_components=0.95)
                X_pca = pca.fit_transform(X_processed_base)
                num_components = X_pca.shape[1]
                num_original_features = X_processed_base.shape[1]
                print(f"PCA: Reduced from {num_original_features} features to {num_components} components explaining 95% variance")
                if num_components < num_original_features * 0.5:
                    print("WARNING: Significant feature redundancy detected (less than 50% of original features explain 95% variance)")
                del X_pca; gc.collect()
            else:
                print("Skipping PCA: Too few features")
            if X_processed_base is None or y_original_array is None or X_processed_base.shape[1] == 0:
                print(f"  Skipping '{config_name}': Preprocessing failed."); continue
            if processed_feature_names:
                original_feature_names = processed_feature_names
            print(f"  Processed X shape: {X_processed_base.shape}, y shape: {y_original_array.shape}")
        except Exception as base_prep_e:
            print(f"  Error Preprocessing: {base_prep_e}"); traceback.print_exc(); continue
        finally:
            del X_features, y_original_df; gc.collect()

        # --- Loop Through Classifiers ---
        models_to_run_this_config = list(classifiers.keys())
        for model_key in models_to_run_this_config:
            full_model_name = f"{config_name}_{model_key}"
            print(f"\n--- Running: {model_key} | Config: {config_name} ---")
            start_time_model = time.time()
            best_model = best_params = cv_results = None
            optimization_status = "OK"; smote_k_used = "N/A"
            eval_X, eval_y = None, None; gc.collect()

            try:
                print(f"  Handling {model_key} using ImbPipeline (WEIGHT if configured)...")
                eval_X, eval_y = X_processed_base.copy(), y_original_array.copy()
                classifier_instance = clone(classifiers[model_key]); pipeline_steps = []
                if not apply_smote_config:
                    if model_key == 'RandomForest':
                        classifier_instance.set_params(class_weight='balanced')
                        print("    -> Using class_weight='balanced' for RandomForest")
                    elif model_key == 'LightGBM':
                        classifier_instance.set_params(class_weight='balanced')
                        print("    -> Using class_weight='balanced' for LightGBM")
                    elif model_key == 'XGBoost':
                        pos = np.sum(eval_y == 1)
                        neg = np.sum(eval_y == 0)
                        scale_weight = neg / pos if pos > 0 else 1.0
                        classifier_instance.set_params(scale_pos_weight=scale_weight)
                        print(f"    -> Using scale_pos_weight={scale_weight:.2f} for XGBoost")
                    elif model_key == 'CatBoost':
                        from collections import Counter
                        counter = Counter(eval_y)
                        total = sum(counter.values())
                        weights = [total / counter[0], total / counter[1]]
                        classifier_instance.set_params(class_weights=weights)
                        print(f"    -> Using class_weights={weights} for CatBoost")

                    pipeline_steps.append(('classifier', classifier_instance))
                    pipeline_for_search = ImbPipeline(pipeline_steps)
                    search_space = get_bayes_search_space(model_key)
                    if not search_space:
                        print(f"  Warning: No search space for {model_key}. Using defaults.")
                        optimization_status = "Defaults Used"
                        try:
                            best_model = pipeline_for_search.fit(eval_X, eval_y)
                            best_params = "Default Parameters"
                        except Exception as default_fit_err:
                            print(f"  ERROR: Default fit failed: {default_fit_err}")
                            optimization_status = "Search & Default Fit Failed"
                            best_model = None
                            best_params = {"error": f"Search & Default Fit Failed: {default_fit_err}"}
                    else:
                        print(f"  Step 3b: Running BayesSearchCV for {model_key} (n_iter={n_search_iter}, cv={search_cv_folds})...")
                        search_cv = BayesSearchCV(
                            estimator=pipeline_for_search,
                            search_spaces=search_space,
                            n_iter=n_search_iter,
                            cv=search_cv_folds,
                            scoring='f1_weighted',
                            n_jobs=-1,
                            verbose=1,
                            random_state=42,
                            error_score='raise'
                        )
                        try:
                            search_cv.fit(eval_X, eval_y)
                            best_model = search_cv.best_estimator_
                            best_params = search_cv.best_params_
                            print(f"  Search finished. Best F1w (inner CV): {search_cv.best_score_:.4f}")
                            print(f"  Best params: {best_params}")
                        except Exception as search_err:
                            print(f"--- ERROR BayesSearch {full_model_name}: {search_err} ---")
                            traceback.print_exc()
                            optimization_status = f"Search Failed: {type(search_err).__name__}"
                            print("  Trying default params...")
                            try:
                                best_model = pipeline_for_search.fit(eval_X, eval_y)
                                best_params = {"error": f"Search Failed, Used Defaults: {search_err}"}
                            except Exception as default_fit_err:
                                print(f"  ERROR: Default fit failed: {default_fit_err}")
                                optimization_status = "Search & Default Fit Failed"
                                best_model = None
                                best_params = {"error": f"Search & Default Fit Failed: {default_fit_err}"}
                        finally:
                            del search_cv
                            gc.collect()

                if best_model is not None:
                    print(f"\n  Step 4: Evaluating best {model_key} using {outer_cv_folds}-fold CV (n_jobs=-1)...")
                    scoring_metrics = {'f1_weighted': 'f1_weighted', 'recall_weighted': 'recall_weighted', 'precision_weighted': 'precision_weighted', 'roc_auc': 'roc_auc', 'specificity': specificity_scorer}
                    if np.isnan(eval_X).any() or np.isinf(eval_X).any():
                        print("  Cleaning NaN/Inf before CV.")
                        eval_X = np.nan_to_num(eval_X, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
                    cv_results = cross_validate(best_model, eval_X, eval_y, cv=cv_strategy_outer, scoring=scoring_metrics, return_train_score=False, n_jobs=6, error_score='raise')
                    result_entry = {
                        'Model Config': config_name,
                        'Classifier': model_key,
                        'F1 Weighted': np.mean(cv_results['test_f1_weighted']),
                        'F1 Weighted Std': np.std(cv_results['test_f1_weighted']),
                        'Recall Weighted': np.mean(cv_results['test_recall_weighted']),
                        'Precision Weighted': np.mean(cv_results['test_precision_weighted']),
                        'Specificity': np.mean(cv_results['test_specificity']),
                        'AUC': np.mean(cv_results['test_roc_auc']),
                        'Mean Fit Time': np.mean(cv_results['fit_time']),
                        'WEIGHT K': smote_k_used,
                        'Best Params': best_params,
                        'Optimization Status': optimization_status
                    }
                    all_results.append(result_entry)
                    print(f"  --- Final CV Results ({outer_cv_folds}-Fold) ---")
                    for metric in scoring_metrics:
                        print(f"    Mean test_{metric}: {np.mean(cv_results[f'test_{metric}']):.4f}")
                    print(f"    Mean Fit Time: {result_entry['Mean Fit Time']:.2f}s")
                    print(f"    WEIGHT K Used: {smote_k_used}")
                    print(f"  --- End CV Results ---")

                    # SHAP Analysis for Traditional Features Only
                    if config_name == "Traditional_Features_Only_Weight_FIX" and model_key in ['XGBoost', 'RandomForest']:
                        print(f"\n🔍 Running SHAP analysis for {config_name} + {model_key}...")
                        import shap
                        shap.initjs()
                        try:
                            classifier = best_model.named_steps['classifier']
                            explainer = shap.Explainer(classifier)
                            shap_values = explainer(eval_X)
                            summary_path = os.path.join(output_dir, f"shap_summary_traditional_{model_key.lower()}.png")
                            plt.figure()
                            shap.summary_plot(shap_values, eval_X, feature_names=original_feature_names, show=False, max_display=20)
                            plt.tight_layout()
                            plt.savefig(summary_path)
                            plt.close()
                            print(f"✅ SHAP summary plot saved to {summary_path}")
                            sample_idx = 42
                            force_path = os.path.join(output_dir, f"shap_force_plot_traditional_{model_key.lower()}.html")
                            force_plot = shap.force_plot(
                                explainer.expected_value,
                                shap_values[sample_idx].values,
                                features=eval_X[sample_idx],
                                feature_names=original_feature_names,
                                matplotlib=False
                            )
                            shap.save_html(force_path, force_plot)
                            print(f"✅ SHAP force plot saved to {force_path}")
                            shap_df = pd.DataFrame(shap_values.values, columns=original_feature_names)
                            shap_df_path = os.path.join(output_dir, f"shap_values_traditional_{model_key.lower()}.csv")
                            shap_df.to_csv(shap_df_path, index=False)
                            print(f"📄 SHAP values saved to {shap_df_path}")
                            result_entry['SHAP Summary Path'] = summary_path
                            result_entry['SHAP Force Plot Path'] = force_path
                            result_entry['SHAP Values Path'] = shap_df_path
                        except Exception as shap_e:
                            print(f"⚠️ Error running SHAP for {config_name} + {model_key}: {shap_e}")
                            traceback.print_exc()
                    if model_key == "LightGBM":
                        import shap
                        explainer = shap.Explainer(best_model.named_steps['classifier'])
                        shap_values = explainer(eval_X)
                        shap_df = pd.DataFrame(shap_values.values, columns=original_feature_names)
                        
                        # Split into positive vs. negative mean influence
                        mean_shap = shap_df.mean().sort_values(ascending=False)
                        pos_influence = mean_shap[mean_shap > 0].head(20)
                        neg_influence = mean_shap[mean_shap < 0].tail(20)

                        print("\nTop 20 Features with Positive Influence:\n", pos_influence)
                        print("\nTop 20 Features with Negative Influence:\n", neg_influence)

                        # Save optional CSV
                        pos_influence.to_csv(os.path.join(output_dir, f"{config_name}_positive_influence.csv"))
                        neg_influence.to_csv(os.path.join(output_dir, f"{config_name}_negative_influence.csv"))
                        print(f"Positive influence features saved to {os.path.join(output_dir, f'{config_name}_positive_influence.csv')}")                   
                else:
                    print(f"  Skipping final eval for {full_model_name} (model fit failed).")
                    all_results.append({
                        'Model Config': config_name,
                        'Classifier': model_key,
                        'F1 Weighted': np.nan,
                        'F1 Weighted Std': np.nan,
                        'Recall Weighted': np.nan,
                        'Precision Weighted': np.nan,
                        'Specificity': np.nan,
                        'AUC': np.nan,
                        'Mean Fit Time': np.nan,
                        'WEIGHT K': smote_k_used,
                        'Best Params': best_params,
                        'Optimization Status': optimization_status
                    })

                # 5. Feature Importance
                if best_model is not None:
                    try:
                        fit_on_base_data = X_processed_base.copy()
                        fit_on_base_target = y_original_array.copy()
                        print("    Re-fitting best pipeline on full base data for importance...")
                        final_model_for_importance = clone(best_model)
                        final_model_for_importance.fit(fit_on_base_data, fit_on_base_target)
                        if final_model_for_importance and original_feature_names:
                            importance_df = get_feature_importance(
                                final_model_for_importance,
                                original_feature_names,
                                full_model_name,
                                output_dir
                            )
                            if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
                                feature_importance_dict[full_model_name] = importance_df
                            else:
                                print("    Feature importance yielded no results.")
                        else:
                            print("    Skipping importance: model or feature names missing.")
                        del fit_on_base_data, fit_on_base_target
                    except Exception as imp_e:
                        print(f"  Error during feature importance extraction: {imp_e}")
                        traceback.print_exc()
                else:
                    print("  Skipping feature importance (model fit failed).")
            except Exception as model_e:
                print(f"--- UNHANDLED ERROR running {model_key} for config {config_name} ---")
                print(f"Error Type: {type(model_e).__name__}: {model_e}"); traceback.print_exc()
                optimization_status = f"Outer Loop Failed: {type(model_e).__name__}"
                all_results.append({
                    'Model Config': config_name,
                    'Classifier': model_key,
                    'F1 Weighted': np.nan,
                    'F1 Weighted Std': np.nan,
                    'Recall Weighted': np.nan,
                    'Precision Weighted': np.nan,
                    'Specificity': np.nan,
                    'AUC': np.nan,
                    'Mean Fit Time': np.nan,
                    'WEIGHT K': smote_k_used,
                    'Best Params': {'error': f"Outer loop: {model_e}"},
                    'Optimization Status': optimization_status
                })
            finally:
                del best_model, cv_results, best_params, eval_X, eval_y
                if 'final_model_for_importance' in locals(): del final_model_for_importance
                if 'importance_df' in locals(): del importance_df
                gc.collect()
            end_time_model = time.time()
            print(f"--- Time for {model_key} on {config_name}: {end_time_model - start_time_model:.2f} seconds ---")

        del X_processed_base, y_original_array, original_feature_names, processed_feature_names
        gc.collect()
        end_time_config = time.time()
        print(f"\nTotal time for config '{config_name}': {end_time_config - start_time_config:.2f} seconds")

    # --- Post-Processing and Final Output ---
    print("\n" + "="*80 + "\nOverall Analysis Summary & Output Generation\n" + "="*80)
    if not all_results: print("No models processed successfully."); return
    print("\nChecking model performance against DummyClassifier baseline...")
    dummy_results = [r for r in all_results if r['Classifier'] == 'DummyClassifier']
    model_results = [r for r in all_results if r['Classifier'] != 'DummyClassifier']
    for dummy in dummy_results:
        dummy_f1 = dummy['F1 Weighted']
        config = dummy['Model Config'].replace('_Dummy', '')
        for model in model_results:
            if model['Model Config'] == config and pd.notna(model['F1 Weighted']):
                model_f1 = model['F1 Weighted']
                if model_f1 - dummy_f1 < 0.1:
                    print(f"WARNING: {model['Model Config']} ({model['Classifier']}) F1 Weighted ({model_f1:.4f}) is close to DummyClassifier ({dummy_f1:.4f})")
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(output_dir, 'classification_model_results_summary.csv')
    formatted_output_path = os.path.join(output_dir, 'formatted_classification_results_summary.md')
    try: results_df.round(5).to_csv(results_csv_path, index=False, encoding='utf-8'); print(f"\nRaw results saved: {results_csv_path}")
    except Exception as e: print(f"Error saving raw results: {e}")
    # --- Save formatted results per model config ---
    try:
        results_by_config = defaultdict(list)
        for res in all_results:
            results_by_config[res['Model Config']].append(res)

        for config_name, config_results in results_by_config.items():
            safe_config_name = config_name.replace(' ', '_').replace('/', '_').replace(':', '_')
            formatted_path = os.path.join(output_dir, f"formatted_results_{safe_config_name}.md")
            try:
                print_formatted_results(config_results, output_file=formatted_path)
            except Exception as e:
                print(f"⚠️ Error writing per-model formatted output for {config_name}: {e}")

    except Exception as e: print(f"Error generating formatted results: {e}"); traceback.print_exc()
    try: plot_classification_results(results_df.copy(), output_dir)
    except Exception as e: print(f"Error generating plots: {e}"); traceback.print_exc()
    if feature_importance_dict:
        viz_importance_path = os.path.join(output_dir, 'visualizations', 'feature_importance_comparison_top25.png')
        try: visualize_feature_importance(feature_importance_dict, top_n=25, save_path=viz_importance_path)
        except Exception as e: print(f"Error generating importance plot: {e}"); traceback.print_exc()
    else: print("\nNo feature importance data generated.")
    print(f"\nClassification analysis complete! Results in: {os.path.abspath(output_dir)}")

# ==============================================================================
#                   SCRIPT EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    script_start_time = time.time()

    # --- Define Global Config Variables ---
    n_search_iter = 20      # Iterations for BayesSearchCV  # עודכן מ-RandomizedSearchCV
    search_cv_folds = 3     # Folds within RandomizedSearchCV
    outer_cv_folds = 5      # Folds for final evaluation

    # Declare globals potentially used by reporting funcs outside main
    output_dir = "classification_results_final_compare" # Default, updated in main
    model_configs = [] # Populated in main

    print("+"*30 + " Starting FINAL Classification Analysis Script " + "+"*30)
    print("Comparing Feature Sets:")
    print("  1. Traditional Features Only (WEIGHT)")
    print("  2. Features + MiniLM (WEIGHT)")
    print("  3. Features + RoBERTa (WEIGHT)")
    print("  4. Features + ModernBERT (WEIGHT)")
    print("\nComparing Classifiers: RandomForest, LightGBM, XGBoost, CatBoost")
    print(f"Predicting: Project 'state' (Successful vs. Other)")
    print(f"NOTE: Excluded leaky features (pledged, backers, ratio). Included 'goal'.")
    print(f"Using BayesSearchCV (n_iter={n_search_iter}, cv={search_cv_folds}, n_jobs=-1).")
    print(f"Using Stratified {outer_cv_folds}-Fold CV (n_jobs=-1) for final evaluation.")
    print("*** Ensure libraries installed: pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost catboost ***")

    # --- Run Main Analysis ---
    try: main_classification()
    except NameError as ne: print(f"\n--- CRITICAL NameError: {ne} ---"); print("Check library imports (xgboost, catboost?)."); traceback.print_exc()
    except Exception as main_e: print(f"\n--- CRITICAL ERROR IN MAIN: {type(main_e).__name__}: {main_e} ---"); traceback.print_exc()
    finally: print("\nPerforming final garbage collection..."); gc.collect()

    script_end_time = time.time(); total_time = script_end_time - script_start_time
    print("\n" + "+" * 30 + " Script Execution Finished " + "+" * 30)
    print(f"Total script execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Results saved in: {os.path.abspath(output_dir)}")
    print("\nEnd of script.")
    # בסוף main_classification, לפני הפקת התוצאות
    print("\nChecking model performance against DummyClassifier baseline...")
