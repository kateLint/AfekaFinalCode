import json
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import TruncatedSVD
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import traceback
import warnings
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression, Ridge


SEED = 42
np.random.seed(SEED)
warnings.filterwarnings('ignore')

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

def make_stratified_bins(y, n_bins=10, min_samples_per_bin=5):
    y_series = pd.Series(y).copy()
    if len(y_series) == 0 or y_series.nunique() <= 1:
        return None

    max_possible_bins_samples = len(y_series) // min_samples_per_bin
    max_possible_bins_unique = y_series.nunique()
    n_bins = min(n_bins, max_possible_bins_samples, max_possible_bins_unique)
    n_bins = max(2, n_bins)

    if n_bins < 2 or len(y_series) < n_bins : # Ensure enough samples for n_bins
        return None
    try:
        binned_y = pd.qcut(y_series, q=n_bins, labels=False, duplicates='drop')
        if binned_y.nunique() < 2 and n_bins >= 2:
            raise ValueError(f"qcut resulted in {binned_y.nunique()} bins.")
        if binned_y.value_counts().min() < 2 and binned_y.nunique() > 1 : # Check for very small bins
            print(f"Warning: qcut created a bin with less than 2 samples. Bin counts: {binned_y.value_counts().to_dict()}")
            raise ValueError("qcut created a bin with less than 2 samples.")
        return binned_y
    except Exception:
        try:
            binned_y_cut = pd.cut(y_series, bins=n_bins, labels=False, include_lowest=True, duplicates='drop')
            if binned_y_cut.nunique() < 2 and n_bins >= 2:
                 raise ValueError(f"pd.cut also resulted in {binned_y_cut.nunique()} bins.")
            if binned_y_cut.value_counts().min() < 2 and binned_y_cut.nunique() > 1:
                 print(f"Warning: pd.cut created a bin with less than 2 samples. Bin counts: {binned_y_cut.value_counts().to_dict()}")
                 raise ValueError("pd.cut created a bin with less than 2 samples.")
            return binned_y_cut
        except Exception:
            return None

def remove_outliers(X, y, threshold=3):
    if X.empty or y.empty: return X, y
    y_series = pd.Series(y)
    q1, q3 = y_series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    mask = (y_series >= lower_bound) & (y_series <= upper_bound)
    removed_count = len(y) - mask.sum()
    if removed_count > 0:
        print(f"Identified {removed_count} outliers from {len(y)} samples (based on y).")
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

def evaluate_model(model, X_test, y_test, model_name, output_dir):
    if X_test.empty:
        print(f"X_test is empty for {model_name}. Skipping evaluation.")
        return {}
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Since y_test and y_pred are log-transformed, inverse transform for interpretation
    y_test_orig = np.expm1(y_test)  # Inverse of log1p
    y_pred_orig = np.expm1(y_pred)
    
    # Compute regression metrics on log-transformed scale
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Compute percentage of predictions within ±20% of true pledged ratio (original scale)
    within_20_percent = np.mean((np.abs(y_pred_orig - y_test_orig) / (y_test_orig + 1e-6)) <= 0.2) * 100
    
    print(f"\n🔍 Evaluation for {model_name} on Test Set")
    print(f"📉 RMSE (log scale): {rmse:,.2f}")
    print(f"📈 MAE (log scale): {mae:,.2f}")
    print(f"📊 R² Score (log scale): {r2:.3f}")
    print(f"✅ Predictions within ±20% of true pledged ratio: {within_20_percent:.2f}%")
    
    # Plot 1: Scatter plot of actual vs. predicted (log scale)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, edgecolors='k', s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual (log scale)')
    plt.ylabel('Predicted (log scale)')
    plt.title(f'Predictions vs Actual - {model_name} (Log Scale)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f'predictions_{model_name}_log_scale.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"📈 Saved log-scale prediction scatter plot to {scatter_path}")
    
    # Plot 2: Scatter plot of actual vs. predicted (original scale)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.3, edgecolors='k', s=50)
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
    plt.xlabel('Actual Pledged-to-Goal Ratio')
    plt.ylabel('Predicted Pledged-to-Goal Ratio')
    plt.title(f'Predictions vs Actual - {model_name} (Original Scale)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    scatter_orig_path = os.path.join(output_dir, f'predictions_{model_name}_original_scale.png')
    plt.savefig(scatter_orig_path, dpi=300)
    plt.close()
    print(f"📈 Saved original-scale prediction scatter plot to {scatter_orig_path}")
    
    # Bin predictions and compute categorical metrics
    bin_metrics = bin_predictions(y_test_orig, y_pred_orig, model_name, output_dir)
    
    # Combine regression and classification metrics
    metrics = {
        'model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MSE': mse,
        'Within_20_Percent': within_20_percent,
        'Classification_Accuracy': bin_metrics.get('accuracy', 0.0)
    }
    
    return metrics

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def bin_predictions(y_true, y_pred, model_name, output_dir):
    """
    Bin actual and predicted pledged-to-goal ratios into categories and compute metrics.
    Categories: Failed (<0.1), Underfunded (0.1 to <0.8), Just Funded (0.8 to 1.2), Overfunded (>1.2)
    """
    # Define bins and labels
    bins = [-float('inf'), 0.1, 0.8, 1.2, float('inf')]
    labels = ['Failed', 'Underfunded', 'Just Funded', 'Overfunded']
    
    # Bin actual and predicted values
    y_true_binned = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)
    y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels, include_lowest=True)
    
    # Compute accuracy
    accuracy = accuracy_score(y_true_binned, y_pred_binned)
    print(f"📊 Classification Accuracy for {model_name}: {accuracy:.3f}")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_binned, y_pred_binned, labels=labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"📈 Saved confusion matrix to {cm_path}")
    
    # Save binning summary
    bin_summary = pd.DataFrame({
        'Actual': y_true_binned.value_counts().reindex(labels, fill_value=0),
        'Predicted': y_pred_binned.value_counts().reindex(labels, fill_value=0)
    })
    bin_summary_path = os.path.join(output_dir, f'bin_summary_{model_name}.csv')
    bin_summary.to_csv(bin_summary_path)
    print(f"📊 Saved binning summary to {bin_summary_path}")
    
    return {'accuracy': accuracy}


def cross_validate_model_with_skf(model, X, y, skf_splitter_obj):
    scoring = {'r2': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error'}
    y_cv_bins = make_stratified_bins(y, n_bins=skf_splitter_obj.get_n_splits())
    
    if y_cv_bins is None or y_cv_bins.nunique() < 2:
        print(f"Warning: Could not create enough bins for stratified CV for model. Using standard KFold with {skf_splitter_obj.get_n_splits()} splits.")
        cv_folds_for_cv = KFold(n_splits=skf_splitter_obj.get_n_splits(), shuffle=True, random_state=skf_splitter_obj.random_state)
    elif y_cv_bins.nunique() < skf_splitter_obj.get_n_splits():
        print(f"Warning: Number of unique bins ({y_cv_bins.nunique()}) is less than n_splits ({skf_splitter_obj.get_n_splits()}). Stratification might be suboptimal but proceeding.")
        cv_folds_for_cv = skf_splitter_obj.split(X, y_cv_bins)
    else:
        cv_folds_for_cv = skf_splitter_obj.split(X, y_cv_bins)
        
    cv_results = cross_validate(model, X, y, cv=cv_folds_for_cv, scoring=scoring, n_jobs=-1, error_score='raise')
    return {
        'CV_R2': cv_results['test_r2'].mean(), 'CV_R2_std': cv_results['test_r2'].std(),
        'CV_RMSE': -cv_results['test_rmse'].mean(), 'CV_RMSE_std': cv_results['test_rmse'].std(),
        'CV_MAE': -cv_results['test_mae'].mean(), 'CV_MAE_std': cv_results['test_mae'].std()
    }

class DataPreprocessor:
    def __init__(self, story_cols, risks_cols, output_dir='results',
                 svd_n_components=50, svd_clip_range=(-10, 10),
                 fs_max_features=50, random_state=SEED):
        self.story_cols = story_cols; self.risks_cols = risks_cols
        self.output_dir = output_dir; self.svd_n_components = svd_n_components
        self.svd_clip_range = svd_clip_range; self.fs_max_features = fs_max_features
        self.random_state = random_state
        self.imputer = None; self.scaler_non_emb = None
        self.scaler_story_emb = None; self.scaler_risks_emb = None
        self.story_svd = None; self.risks_svd = None
        self.one_hot_encoder_columns = None; self.feature_selector = None
        self.selected_features = None; self.numerical_cols_to_impute = None
        self.non_emb_cols_to_scale = None; self._last_fit_X_index = pd.Index([])


    def _preprocess_embeddings_with_svd(self, X_df, fit=False):
        X = X_df.copy()
        story_matrix = X[self.story_cols].values.astype(np.float32)
        risks_matrix = X[self.risks_cols].values.astype(np.float32)
        story_matrix = np.clip(story_matrix, *self.svd_clip_range); risks_matrix = np.clip(risks_matrix, *self.svd_clip_range)
        story_matrix = np.nan_to_num(story_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        risks_matrix = np.nan_to_num(risks_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            self.scaler_story_emb = RobustScaler(); self.scaler_risks_emb = RobustScaler()
            story_matrix_scaled = self.scaler_story_emb.fit_transform(story_matrix)
            risks_matrix_scaled = self.scaler_risks_emb.fit_transform(risks_matrix)
            actual_svd_story_comps = min(self.svd_n_components, story_matrix_scaled.shape[1]-1, story_matrix_scaled.shape[0]-1)
            actual_svd_risks_comps = min(self.svd_n_components, risks_matrix_scaled.shape[1]-1, risks_matrix_scaled.shape[0]-1)
            if actual_svd_story_comps < 1 or actual_svd_risks_comps < 1:
                print("Warning: Not enough features/samples for SVD. Skipping SVD.")
                self.story_svd, self.risks_svd = None, None # Mark SVD as not fitted
                return X # Return original X if SVD cannot be performed
            self.story_svd = TruncatedSVD(n_components=actual_svd_story_comps, random_state=self.random_state)
            self.risks_svd = TruncatedSVD(n_components=actual_svd_risks_comps, random_state=self.random_state)
            story_svd_result = self.story_svd.fit_transform(story_matrix_scaled)
            risks_svd_result = self.risks_svd.fit_transform(risks_matrix_scaled)
            self.svd_n_components_actual_story = actual_svd_story_comps # Store actual components used
            self.svd_n_components_actual_risks = actual_svd_risks_comps
            if self.story_svd:
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1); plt.plot(np.cumsum(self.story_svd.explained_variance_ratio_), marker='o'); plt.title("Story SVD")
                if self.risks_svd: plt.subplot(1, 2, 2); plt.plot(np.cumsum(self.risks_svd.explained_variance_ratio_), marker='o'); plt.title("Risks SVD")
                plt.tight_layout(); plt.savefig(os.path.join(self.output_dir, 'svd_scree.png'), dpi=300); plt.close()
        else:
            if not self.story_svd or not self.risks_svd : # Check if SVD was fitted and successful
                return X # SVD was skipped during fit, so skip transform
            story_matrix_scaled = self.scaler_story_emb.transform(story_matrix)
            risks_matrix_scaled = self.scaler_risks_emb.transform(risks_matrix)
            story_svd_result = self.story_svd.transform(story_matrix_scaled)
            risks_svd_result = self.risks_svd.transform(risks_matrix_scaled)

        X_processed = X.drop(columns=self.story_cols + self.risks_cols, errors='ignore').reset_index(drop=True)
        if self.story_svd: story_svd_df = pd.DataFrame(story_svd_result, columns=[f"StorySVD_{i}" for i in range(self.svd_n_components_actual_story)], index=X_processed.index)
        else: story_svd_df = pd.DataFrame(index=X_processed.index) # Empty if SVD skipped
        if self.risks_svd: risks_svd_df = pd.DataFrame(risks_svd_result, columns=[f"RisksSVD_{i}" for i in range(self.svd_n_components_actual_risks)], index=X_processed.index)
        else: risks_svd_df = pd.DataFrame(index=X_processed.index)
        X_processed = pd.concat([X_processed, story_svd_df, risks_svd_df], axis=1)
        return X_processed

    def fit(self, X_train_df, y_train_series):
        X_train = X_train_df.copy(); y_train = y_train_series.copy()
        self._last_fit_X_index = X_train_df.index # Store for transform save naming
        self.numerical_cols_to_impute = X_train.select_dtypes(include=np.number).columns.tolist()
        valid_numerical_cols = [col for col in self.numerical_cols_to_impute if not X_train[col].isna().all()]
        if len(valid_numerical_cols) < len(self.numerical_cols_to_impute):
            X_train = X_train.drop(columns=[c for c in self.numerical_cols_to_impute if c not in valid_numerical_cols])
        self.numerical_cols_to_impute = valid_numerical_cols
        if self.numerical_cols_to_impute:
            self.imputer = SimpleImputer(strategy='median')
            X_train[self.numerical_cols_to_impute] = self.imputer.fit_transform(X_train[self.numerical_cols_to_impute])
        if all(c in X_train.columns for c in ['goal', 'rewardscount', 'Story_Compound', 'Story_Flesch_Reading_Ease']):
            X_train['goal_per_reward'] = X_train['goal'] / (X_train['rewardscount'].replace(0, 1e-6) + 1e-6) # Avoid div by zero
            X_train['story_sentiment_readability'] = X_train['Story_Compound'] * X_train['Story_Flesch_Reading_Ease']
        self.non_emb_cols_to_scale = [c for c in X_train.select_dtypes(include=np.number).columns if not any(c.startswith(p) for p in ['Story_emb_', 'Risks_emb_', 'StorySVD_', 'RisksSVD_'])]
        if self.non_emb_cols_to_scale:
            self.scaler_non_emb = RobustScaler()
            X_train[self.non_emb_cols_to_scale] = self.scaler_non_emb.fit_transform(X_train[self.non_emb_cols_to_scale])
        if self.story_cols and self.risks_cols and all(c in X_train.columns for c in self.story_cols) and all(c in X_train.columns for c in self.risks_cols):
            X_train = self._preprocess_embeddings_with_svd(X_train, fit=True)
        X_train = pd.get_dummies(X_train, drop_first=True, dummy_na=False, columns=X_train.select_dtypes(include=['object', 'category']).columns)
        self.one_hot_encoder_columns = X_train.columns.tolist()
        
        actual_fs_max_features = min(self.fs_max_features, X_train.shape[1])
        if X_train.shape[1] > actual_fs_max_features and actual_fs_max_features > 0:
            X_fs = X_train.copy().replace([np.inf, -np.inf], np.nan)
            if X_fs.isnull().any().any():
                num_cols_fs = X_fs.select_dtypes(include=np.number).columns
                if len(num_cols_fs) > 0: X_fs[num_cols_fs] = SimpleImputer(strategy='median').fit_transform(X_fs[num_cols_fs])
            rf_fs = RandomForestRegressor(random_state=self.random_state, n_jobs=-1, n_estimators=50, max_depth=10)
            self.feature_selector = SelectFromModel(rf_fs, threshold=-np.inf, max_features=actual_fs_max_features)
            try: self.feature_selector.fit(X_fs, y_train)
            except ValueError as e: print(f"Warning: RF for feature selection failed: {e}. Skipping FS."); self.feature_selector = None
            if self.feature_selector: self.selected_features = X_train.columns[self.feature_selector.get_support()].tolist()
            else: self.selected_features = X_train.columns.tolist()
        else: self.selected_features = X_train.columns.tolist()
        return self

    def transform(self, X_df):
        X = X_df.copy(); original_index_name = X.index.name
        if not all([self.imputer, self.scaler_non_emb, self.one_hot_encoder_columns, self.selected_features]):
             raise RuntimeError("Preprocessor not fitted.")
        transform_num_cols = [col for col in self.numerical_cols_to_impute if col in X.columns]
        if transform_num_cols: X[transform_num_cols] = self.imputer.transform(X[transform_num_cols])
        if all(c in X.columns for c in ['goal', 'rewardscount', 'Story_Compound', 'Story_Flesch_Reading_Ease']):
            X['goal_per_reward'] = X['goal'] / (X['rewardscount'].replace(0, 1e-6) + 1e-6)
            X['story_sentiment_readability'] = X['Story_Compound'] * X['Story_Flesch_Reading_Ease']
        transform_non_emb_cols = [col for col in self.non_emb_cols_to_scale if col in X.columns]
        if transform_non_emb_cols: X[transform_non_emb_cols] = self.scaler_non_emb.transform(X[transform_non_emb_cols])
        index_was_reset_by_svd = False
        if self.story_cols and self.risks_cols and self.story_svd: # Check if SVD was actually fitted
             if all(c in X.columns for c in self.story_cols) and all(c in X.columns for c in self.risks_cols):
                X = self._preprocess_embeddings_with_svd(X, fit=False); index_was_reset_by_svd = True
        X = pd.get_dummies(X, drop_first=True, dummy_na=False, columns=X.select_dtypes(include=['object', 'category']).columns)
        current_cols = X.columns.tolist()
        for col in self.one_hot_encoder_columns:
            if col not in current_cols: X[col] = 0
        try: X = X[self.one_hot_encoder_columns]
        except KeyError as e:
            missing_cols = [c for c in self.one_hot_encoder_columns if c not in X.columns]
            print(f"Error aligning OHE columns. Missing in X: {missing_cols}. OHE cols: {self.one_hot_encoder_columns[:5]}. X cols: {X.columns[:5]}")
            raise e
        if self.selected_features:
            try: X = X[self.selected_features]
            except KeyError as e:
                missing_sel_cols = [c for c in self.selected_features if c not in X.columns]
                print(f"Error applying selected features. Missing in X: {missing_sel_cols}. Selected: {self.selected_features[:5]}. X cols: {X.columns[:5]}")
                raise e
        if not index_was_reset_by_svd:
            X = X.reset_index(drop=True)
            if original_index_name is not None: X.index.name = original_index_name
        dataset_type = "train_set" if X_df.index.equals(self._last_fit_X_index) else "test_set"
        X.to_csv(os.path.join(self.output_dir, f'transformed_{dataset_type}.csv'), index=False)
        return X

    def fit_transform(self, X_train_df, y_train_series):
        self.fit(X_train_df, y_train_series)
        return self.transform(X_train_df)

def extract_features(data):
    features, story_embeddings, risks_embeddings, error_count = [], [], [], 0
    for item in tqdm(data, desc="Extracting Features"):
        try:
            if item.get('state') not in ['successful', 'failed']: continue
            ratio = safe_float(item.get('pledged_to_goal_ratio'))
            if np.isnan(ratio) or ratio is None: continue
            f = {'pledged_to_goal_ratio': ratio, 'goal': safe_float(item.get('goal'), 0.0),
                 'projectFAQsCount': safe_float(item.get('projectFAQsCount'), 0.0),
                 'rewardscount': safe_float(item.get('rewardscount'), 0.0),
                 'project_length_days': safe_float(item.get('project_length_days')),
                 'preparation_days': safe_float(item.get('preparation_days')),
                 'Story_Compound': safe_float(item.get('Story_Compound')),
                 'Risks_Compound': safe_float(item.get('Risks_Compound')),
                 'Story_Flesch_Reading_Ease': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Flesch Reading Ease'))}
            for key, value in item.items():
                if key.startswith('category_') and key not in f: f[key] = 1 if value is True or str(value).lower() in ['true', '1'] else 0
            features.append(f)
            story_emb = item.get('story_miniLM', []); risks_emb = item.get('risks_miniLM', [])
            story_embeddings.append(story_emb if isinstance(story_emb, list) and len(story_emb) > 0 else [0.0] * 384)
            risks_embeddings.append(risks_emb if isinstance(risks_emb, list) and len(risks_emb) > 0 else [0.0] * 384)
        except Exception: error_count += 1
    if not features: return pd.DataFrame()
    df = pd.DataFrame(features)
    story_df = pd.DataFrame(story_embeddings, columns=[f"Story_emb_{i}" for i in range(384)], index=df.index)
    risks_df = pd.DataFrame(risks_embeddings, columns=[f"Risks_emb_{i}" for i in range(384)], index=df.index)
    return pd.concat([df, story_df, risks_df], axis=1)

def train_model(X_train, y_train, X_test, y_test, model_instance, search_space,
                cv_splitter_obj, model_name, output_dir, random_state_seed=SEED):
    try:
        if X_train.empty: print(f"X_train is empty for {model_name}. Skipping training."); return None, None, {}
        if hasattr(model_instance, 'random_state'): model_instance.set_params(random_state=random_state_seed)
        if model_name == 'XGBoost' and hasattr(model_instance, 'seed'): model_instance.set_params(seed=random_state_seed)

        y_train_bins_hpo = make_stratified_bins(y_train, n_bins=5)
        stratify_hpo = y_train_bins_hpo if y_train_bins_hpo is not None and y_train_bins_hpo.nunique() > 1 else None
        
        # Ensure X_val_hpo is not empty
        test_size_hpo = 0.15
        if len(X_train) * test_size_hpo < 2 : test_size_hpo = max(1, len(X_train) // 2) / len(X_train) if len(X_train) > 1 else 0.0
        if len(X_train) < 2 or test_size_hpo == 0.0: X_val_hpo, y_val_hpo = X_test, y_test # Fallback, not ideal
        else: _, X_val_hpo, _, y_val_hpo = train_test_split(X_train, y_train, test_size=test_size_hpo, random_state=random_state_seed, stratify=stratify_hpo)
        if X_val_hpo.empty : X_val_hpo, y_val_hpo = X_test, y_test # another fallback

        fit_kwargs = {}
        if model_name == "XGBoost": 
            if hasattr(model_instance, 'early_stopping_rounds'):
                model_instance.set_params(early_stopping_rounds=None)  # Remove for CV
            fit_kwargs = {'eval_set': [(X_val_hpo, y_val_hpo)], 'verbose': False}
        elif model_name == "LightGBM": fit_kwargs = {'callbacks': [lgb.early_stopping(stopping_rounds=10, verbose=-1)], 'eval_set': [(X_val_hpo, y_val_hpo)]}

        y_cv_bins_bayes = make_stratified_bins(y_train, n_bins=cv_splitter_obj.get_n_splits())
        if y_cv_bins_bayes is None or y_cv_bins_bayes.nunique() < 2:
            cv_iter_bayes = KFold(n_splits=cv_splitter_obj.get_n_splits(), shuffle=True, random_state=cv_splitter_obj.random_state)
        else: cv_iter_bayes = list(cv_splitter_obj.split(X_train, y_cv_bins_bayes))
        
        opt = BayesSearchCV(model_instance, search_space, n_iter=15 if model_name != 'OLS' else 3, cv=cv_iter_bayes, n_jobs=-1, scoring='neg_mean_squared_error', random_state=random_state_seed, verbose=0)
        opt.fit(X_train, y_train, **fit_kwargs)
        best_model = opt.best_estimator_
        cv_metrics = cross_validate_model_with_skf(best_model, X_train, y_train, cv_splitter_obj)
        joblib.dump(best_model, os.path.join(output_dir, f'best_{model_name}.joblib'))
        s_params = {k: (v.item() if hasattr(v, 'item') else v) for k,v in opt.best_params_.items()}
        with open(os.path.join(output_dir, f'best_{model_name}_params.json'), 'w') as f: json.dump(s_params, f, indent=2)
        metrics = evaluate_model(best_model, X_test, y_test, model_name, output_dir)
        metrics.update(cv_metrics); metrics['Best_Params'] = s_params

        if not X_test.empty:
            try:
                X_test_sample = X_test.sample(n=min(200, len(X_test)), random_state=random_state_seed) if len(X_test) > 200 else X_test
                if X_test_sample.empty: raise ValueError("X_test_sample for SHAP is empty.")
                if model_name == "OLS":
                    X_train_s_shap = X_train.sample(n=min(500, len(X_train)), random_state=random_state_seed) if len(X_train) > 500 else X_train
                    explainer = shap.LinearExplainer(best_model, X_train_s_shap); shap_values = explainer.shap_values(X_test_sample)
                    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", max_display=15, show=False)
                else:
                    masker = shap.maskers.Independent(data=X_test_sample)
                    explainer = shap.TreeExplainer(best_model, masker) if model_name in ["RandomForest", "XGBoost", "LightGBM"] else shap.Explainer(best_model, masker)
                    shap_obj = explainer(X_test_sample, check_additivity=False)
                    shap.plots.beeswarm(shap_obj, max_display=15, show=False)
                plt.title(f"SHAP - {model_name}"); plt.tight_layout(); plt.savefig(os.path.join(output_dir, f'shap_{model_name}.png'),dpi=300); plt.close()
            except Exception as e: print(f"SHAP failed for {model_name}: {e}")
        return best_model, opt.best_params_, metrics
    except Exception as e: print(f"Error training {model_name}: {e}"); traceback.print_exc(); return None, None, {}


# Add these fixes right after analyze_target_variable call

def preprocess_target_variable(y_raw, method='log_transform'):
    """
    Preprocess the target variable to handle extreme skewness
    """
    y_original = y_raw.copy()
    
    if method == 'cap_outliers':
        # Option 1: Cap extreme outliers at 99th percentile
        cap_value = y_raw.quantile(0.99)
        y_processed = y_raw.clip(upper=cap_value)
        capped_count = (y_raw > cap_value).sum()
        print(f"Capped {capped_count} values above {cap_value:.2f}")
        
    elif method == 'remove_extremes':
        # Option 2: Remove extreme outliers completely
        mask = y_raw <= y_raw.quantile(0.99)
        y_processed = y_raw[mask]
        removed_count = (~mask).sum()
        print(f"Removed {removed_count} extreme outlier samples")
        return y_processed, mask
        
    elif method == 'log_transform':
        # Option 3: Log transform (handles zeros by adding small constant)
        y_processed = np.log1p(y_raw)  # log(1 + x)
        print(f"Applied log1p transformation")
        
    elif method == 'sqrt_transform':
        # Option 4: Square root transform
        y_processed = np.sqrt(y_raw)
        print(f"Applied square root transformation")
        
    elif method == 'combined':
        # Option 5: Cap + Log transform
        cap_value = y_raw.quantile(0.95)  # More aggressive capping
        y_capped = y_raw.clip(upper=cap_value)
        y_processed = np.log1p(y_capped)
        capped_count = (y_raw > cap_value).sum()
        print(f"Capped {capped_count} values above {cap_value:.2f} then applied log1p")
    
    else:
        y_processed = y_raw
    
    # Show before/after stats
    print(f"\nBefore: Mean={y_original.mean():.3f}, Std={y_original.std():.3f}, Max={y_original.max():.3f}")
    print(f"After:  Mean={y_processed.mean():.3f}, Std={y_processed.std():.3f}, Max={y_processed.max():.3f}")
    
    return y_processed, None


def main():
    print(f"{datetime.now()} Starting Analysis")
    output_dir = 'results_regression_v4'; os.makedirs(output_dir, exist_ok=True)
    data_path = "/Users/kerenlint/Projects/Afeka/models_weight/all_good_projects_with_modernbert_embeddings_enhanced_with_miniLM12.json" #  <--- USER: VERIFY THIS PATH
    try:
        with open(data_path, 'r', encoding='utf-8') as f: data = json.load(f)
        # data = data[:800] # Dev subset
    except Exception as e: print(f"Fatal: Data loading failed: {e}"); return

    raw_df = extract_features(data)
    if raw_df.empty or 'pledged_to_goal_ratio' not in raw_df.columns: print("Error: No data after extraction."); return
    raw_df = raw_df.dropna(subset=['pledged_to_goal_ratio'])
    if raw_df.empty: print("Error: Data empty after dropping NaNs in target."); return
    X_raw = raw_df.drop(columns=['pledged_to_goal_ratio']); y_raw = raw_df['pledged_to_goal_ratio']

    y_raw = analyze_target_variable(y_raw)

    y_raw_processed, _ = preprocess_target_variable(y_raw, method='log_transform')
    y_raw = y_raw_processed


    correlations = X_raw.select_dtypes(include=[np.number]).corrwith(y_raw).abs().sort_values(ascending=False)
    print("\nTop 10 feature correlations with target (log-transformed):")
    print(correlations.head(10))


    print(f"Data shape after cleaning: {raw_df.shape}")
    print(f"Target variable stats: min={y_raw.min():.3f}, max={y_raw.max():.3f}, std={y_raw.std():.3f}")
    print(f"Feature columns: {len(X_raw.columns)}")

    y_bins_raw_split = make_stratified_bins(y_raw, n_bins=5)
    stratify_raw = y_bins_raw_split if y_bins_raw_split is not None and y_bins_raw_split.nunique() > 1 else None
    try: X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_raw, y_raw, test_size=0.2, random_state=SEED, stratify=stratify_raw)
    except ValueError: X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_raw, y_raw, test_size=0.2, random_state=SEED)
    
    X_train_no_o, y_train_no_o = remove_outliers(X_train_r, y_train_r)
    if X_train_no_o.empty: print("Error: Training data empty after outlier removal."); return

    s_cols = [c for c in X_train_no_o.columns if c.startswith("Story_emb_")]
    r_cols = [c for c in X_train_no_o.columns if c.startswith("Risks_emb_")]
    preproc = DataPreprocessor(story_cols=s_cols, risks_cols=r_cols, output_dir=output_dir, random_state=SEED)
    preproc.fit(X_train_no_o.copy(), y_train_no_o.copy())
    X_train_p = preproc.transform(X_train_no_o.copy())
    y_train_f = y_train_no_o.reset_index(drop=True) # y_train_no_o already has reset index from remove_outliers
    X_train_p = X_train_p.reset_index(drop=True)
    X_test_p = preproc.transform(X_test_r.copy())
    y_test_f = y_test_r.reset_index(drop=True)
    X_test_p = X_test_p.reset_index(drop=True)

    if len(X_train_p) != len(y_train_f) or len(X_test_p) != len(y_test_f):
        print(f"Length Mismatch! Train X:{len(X_train_p)} y:{len(y_train_f)}. Test X:{len(X_test_p)} y:{len(y_test_f)}")
        return
    if X_train_p.empty or X_test_p.empty: print("Error: Processed data is empty."); return

    models = {
        'RandomForest': (RandomForestRegressor(random_state=SEED), {'n_estimators': Integer(50,200),'max_depth':Integer(5,15),'min_samples_split':Integer(2,10),'min_samples_leaf': Integer(2,10),'max_features':Real(0.6,0.9)}),
        'XGBoost': (XGBRegressor(objective='reg:squarederror', random_state=SEED, seed=SEED, tree_method='hist'),{'n_estimators':Integer(50,300),'max_depth':Integer(3,10),'learning_rate':Real(0.01,0.2,prior='log-uniform'),'subsample':Real(0.7,1.0)}),
        'LightGBM': (lgb.LGBMRegressor(random_state=SEED,force_col_wise=True,verbosity=-1),{'n_estimators':Integer(50,300),'max_depth':Integer(3,10),'learning_rate':Real(0.01,0.2,prior='log-uniform'), 'num_leaves':Integer(10,50)}),
        'OLS': (Ridge(alpha=1e-3), {'fit_intercept': Categorical([True, False]), 'alpha': Real(1e-6, 1e-1, prior='log-uniform')})
    }
    print(f"Data shape after cleaning: {raw_df.shape}")
    print(f"Target variable stats: min={y_raw.min():.3f}, max={y_raw.max():.3f}, std={y_raw.std():.3f}")
    print(f"Feature columns: {len(X_raw.columns)}")
    results = []
    skf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, (model, space) in tqdm(models.items(), desc="Training Models"):
        _, _, metrics = train_model(X_train_p, y_train_f, X_test_p, y_test_f, model, space, skf_cv, name, output_dir, SEED)
        if metrics: results.append(metrics)
    if not results: print("No models trained."); return

    res_df = pd.DataFrame(results).sort_values('R2', ascending=False)
    res_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print(f"\n{datetime.now()} Saved comparison.\nTop Models:\n{res_df[['model', 'R2', 'CV_R2', 'RMSE']].head()}")
    if not res_df.empty:
        plt.figure(figsize=(10,7)); sns.barplot(x='R2',y='model',data=res_df,palette='viridis',orient='h')
        plt.title('R² Comparison'); plt.xlabel('R² Score'); plt.ylabel('Model'); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'r2_comparison.png'), dpi=300); plt.close()
    print(f"\n{datetime.now()} Analysis Complete")


def analyze_target_variable(y_raw):
    print("=== TARGET VARIABLE ANALYSIS ===")
    print(f"Shape: {y_raw.shape}")
    print(f"Min: {y_raw.min():.3f}")
    print(f"Max: {y_raw.max():.3f}")
    print(f"Mean: {y_raw.mean():.3f}")
    print(f"Median: {y_raw.median():.3f}")
    print(f"Std: {y_raw.std():.3f}")
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        print(f"{p}%: {y_raw.quantile(p/100):.3f}")
    
    # Count extreme values
    q99 = y_raw.quantile(0.99)
    extreme_count = (y_raw > q99).sum()
    print(f"\nValues above 99th percentile: {extreme_count} ({extreme_count/len(y_raw)*100:.1f}%)")
    
    # Distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(y_raw, bins=50, alpha=0.7)
    plt.title('Raw Distribution')
    plt.xlabel('pledged_to_goal_ratio')
    
    plt.subplot(1, 3, 2)
    plt.hist(y_raw[y_raw <= y_raw.quantile(0.95)], bins=50, alpha=0.7)
    plt.title('Distribution (95th percentile cutoff)')
    plt.xlabel('pledged_to_goal_ratio')
    
    plt.subplot(1, 3, 3)
    plt.boxplot(y_raw)
    plt.title('Box Plot')
    plt.ylabel('pledged_to_goal_ratio')
    
    plt.tight_layout()
    plt.savefig('target_analysis.png', dpi=300)
    plt.close()
    
    return y_raw



if __name__ == "__main__":
    try: main()
    except Exception as e: print(f"Fatal error: {e}"); traceback.print_exc()