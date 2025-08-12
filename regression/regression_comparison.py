import json
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import Ridge
import traceback
import warnings
from datetime import datetime
from tqdm.auto import tqdm

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
    if len(y_series) == 0 or y_series.nunique() <= 1 or len(y_series) < n_bins * min_samples_per_bin:
        return None
    n_bins = min(n_bins, len(y_series) // min_samples_per_bin, y_series.nunique())
    n_bins = max(2, n_bins)
    try:
        binned_y = pd.qcut(y_series, q=n_bins, labels=False, duplicates='drop')
        if binned_y.nunique() < 2 or binned_y.value_counts().min() < 2:
            raise ValueError("Insufficient unique bins or small bin sizes.")
        return binned_y
    except ValueError:
        try:
            binned_y_cut = pd.cut(y_series, bins=n_bins, labels=False, include_lowest=True, duplicates='drop')
            if binned_y_cut.nunique() < 2 or binned_y_cut.value_counts().min() < 2:
                raise ValueError("Insufficient unique bins or small bin sizes with pd.cut.")
            return binned_y_cut
        except ValueError:
            return None

def remove_outliers(X, y, threshold=3, remove=False):
    if X.empty or len(y) == 0:
        return X, pd.Series(y)
    y_series = pd.Series(y)
    q1, q3 = y_series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    mask = (y_series >= lower_bound) & (y_series <= upper_bound)
    removed_count = len(y) - mask.sum()
    if remove:
        if removed_count > 0:
            print(f"Removed {removed_count} outliers based on y (IQR threshold={threshold}).")
        return X[mask].reset_index(drop=True), y_series[mask].reset_index(drop=True)
    else:
        print(f"Flagged {removed_count} outliers (not removed).")
        return X, y_series, mask

def evaluate_model(model, X_test, y_test, model_name, output_dir):
    if X_test.empty or len(y_test) == 0:
        print(f"Skipping evaluation for {model_name}: Empty test data.")
        return {}
    
    y_pred = model.predict(X_test)
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    within_20_percent = np.mean(np.abs(y_pred_orig - y_test_orig) / (y_test_orig + 1e-6) <= 0.2) * 100
    
    print(f"\nEvaluation for {model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}, Within 20%={within_20_percent:.2f}%")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual (log-scale)')
    plt.ylabel('Predicted (log-scale)')
    plt.title(f'{model_name} - Log Scale Predictions')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.6, s=30)
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
    plt.xlabel('Actual Ratio')
    plt.ylabel('Predicted Ratio')
    plt.title(f'{model_name} - Original Scale Predictions')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    bin_metrics = bin_predictions(y_test_orig, y_pred_orig, model_name, output_dir)
    
    return {
        'model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MSE': mse,
        'Within_20_Percent': within_20_percent,
        'Classification_Accuracy': bin_metrics.get('accuracy', 0.0)
    }

def bin_predictions(y_true, y_pred, model_name, output_dir):
    bins = [-np.inf, 0.1, 0.8, 1.2, np.inf]
    labels = ['Failed (< 0.1)', 'Underfunded (0.1 ≤ x < 0.8)', 'Just Funded (0.8 ≤ x ≤ 1.2)', 'Overfunded (> 1.2)']
    y_true_binned = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)
    y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels, include_lowest=True)
    accuracy = accuracy_score(y_true_binned, y_pred_binned)
    cm = confusion_matrix(y_true_binned, y_pred_binned, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {accuracy:.3f}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'accuracy': accuracy}

def cross_validate_model_with_skf(model, X, y, skf_splitter_obj):
    scoring = {'r2': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error'}
    y_cv_bins = make_stratified_bins(y, n_bins=skf_splitter_obj.get_n_splits())
    if y_cv_bins is None or y_cv_bins.nunique() < 2:
        cv = KFold(n_splits=skf_splitter_obj.get_n_splits(), shuffle=True, random_state=skf_splitter_obj.random_state)
    else:
        cv = skf_splitter_obj.split(X, y_cv_bins)
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=2)
    return {
        'CV_R2': cv_results['test_r2'].mean(),
        'CV_R2_std': cv_results['test_r2'].std(),
        'CV_RMSE': -cv_results['test_rmse'].mean(),
        'CV_RMSE_std': cv_results['test_rmse'].std(),
        'CV_MAE': -cv_results['test_mae'].mean(),
        'CV_MAE_std': cv_results['test_mae'].std()
    }

def generate_enhanced_shap_analysis(model, X_train, X_test, y_train, model_name, output_dir, random_state_seed=SEED):
    try:
        embedding_patterns = ['Story_minilm_emb_', 'Risks_minilm_emb_', 'Story_roberta_emb_', 'Risks_roberta_emb_', 
                             'Story_modernbert_emb_', 'Risks_modernbert_emb_', 'StorySVD_', 'RisksSVD_']
        pure_feature_cols = [col for col in X_test.columns 
                           if not any(col.startswith(pattern) for pattern in embedding_patterns)]
        
        print(f"SHAP Analysis for {model_name}: Using {len(pure_feature_cols)} pure features out of {len(X_test.columns)} total features")
        
        X_test_pure = X_test[pure_feature_cols]
        X_train_pure = X_train[pure_feature_cols]
        
        sample_size = min(1000, len(X_test_pure)) if len(X_test_pure) > 200 else len(X_test_pure)
        X_test_sample = X_test_pure.sample(n=sample_size, random_state=random_state_seed) if len(X_test_pure) > sample_size else X_test_pure
        
        if X_test_sample.empty:
            raise ValueError("X_test_sample for SHAP is empty.")
        
        shap_values = None
        feature_importance = None
        
        if model_name == "Ridge":
            X_train_s_shap = X_train_pure.sample(n=min(500, len(X_train_pure)), random_state=random_state_seed) if len(X_train_pure) > 500 else X_train_pure
            ridge_pure = Ridge(**model.get_params())
            ridge_pure.fit(X_train_pure, y_train)
            explainer = shap.LinearExplainer(ridge_pure, X_train_s_shap)
            shap_values = explainer.shap_values(X_test_sample)
            feature_importance = pd.DataFrame({
                'feature': X_test_sample.columns,
                'mean_abs_shap': np.abs(shap_values).mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
            
        else:
            if model_name == "RandomForest":
                model_pure = RandomForestRegressor(**model.get_params())
            elif model_name == "XGBoost":
                model_pure = XGBRegressor(**model.get_params())
            elif model_name == "LightGBM":
                model_pure = lgb.LGBMRegressor(**model.get_params())
            
            model_pure.fit(X_train_pure, y_train)
            masker = shap.maskers.Independent(data=X_test_sample)
            explainer = shap.TreeExplainer(model_pure, masker)
            shap_obj = explainer(X_test_sample, check_additivity=False)
            shap_values = shap_obj.values
            feature_importance = pd.DataFrame({
                'feature': X_test_sample.columns,
                'mean_abs_shap': np.abs(shap_values).mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
        
        n_features_to_show = max(15, min(25, len(pure_feature_cols)))
        
        plt.figure(figsize=(10, 8))
        
        if model_name == "Ridge":
            top_features = feature_importance.head(n_features_to_show)
            top_indices = [X_test_sample.columns.get_loc(f) for f in top_features['feature']]
            shap_values_top = shap_values[:, top_indices]
            
            shap.summary_plot(shap_values_top, X_test_sample[top_features['feature']], 
                            plot_type="bar", max_display=n_features_to_show, show=False)
        else:
            shap_obj_df = pd.DataFrame(shap_values, columns=X_test_sample.columns)
            top_features = feature_importance.head(n_features_to_show)['feature'].tolist()
            shap_obj_filtered = shap.Explanation(
                values=shap_values[:, [X_test_sample.columns.get_loc(f) for f in top_features]],
                base_values=shap_obj.base_values,
                data=X_test_sample[top_features].values,
                feature_names=top_features
            )
            
            shap.plots.beeswarm(shap_obj_filtered, max_display=n_features_to_show, show=False)
        
        plt.title(f"SHAP Feature Importance - {model_name} (Top {n_features_to_show} Pure Features)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_{model_name}_top_{n_features_to_show}_features.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        top_20_features = feature_importance.head(20)
        plt.barh(range(len(top_20_features)), top_20_features['mean_abs_shap'])
        plt.yticks(range(len(top_20_features)), top_20_features['feature'])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Top 20 Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_{model_name}_bar_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        feature_importance.to_csv(os.path.join(output_dir, f'shap_feature_importance_{model_name}.csv'), index=False)
        
        shap_summary = {
            'top_20_features': feature_importance.head(20).to_dict('records'),
            'total_features_analyzed': len(pure_feature_cols),
            'sample_size': len(X_test_sample),
            'features_displayed': n_features_to_show
        }
        
        with open(os.path.join(output_dir, f'shap_summary_{model_name}.json'), 'w') as f:
            json.dump(shap_summary, f, indent=2)
        
        print(f"SHAP analysis completed for {model_name}. Displayed top {n_features_to_show} features.")
        
    except Exception as e:
        print(f"SHAP analysis failed for {model_name}: {e}")
        traceback.print_exc()
        return None

def create_detailed_outcome_bands_visualization(predictions_df, output_dir):
    def bin_ratio(ratio):
        if pd.isna(ratio) or ratio < 0.1:
            return "Failed (< 0.1)"
        elif ratio < 0.8:
            return "Underfunded (0.1 ≤ x < 0.8)"
        elif ratio <= 1.2:
            return "Just Funded (0.8 ≤ x ≤ 1.2)"
        else:
            return "Overfunded (> 1.2)"

    banded_results = predictions_df.apply(lambda col: col.apply(bin_ratio))
    band_counts = banded_results.apply(pd.Series.value_counts).fillna(0).astype(int).T
    logical_order = ["Failed (< 0.1)", "Underfunded (0.1 ≤ x < 0.8)", "Just Funded (0.8 ≤ x ≤ 1.2)", "Overfunded (> 1.2)"]
    band_counts = band_counts.reindex(columns=logical_order, fill_value=0)
    
    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bottom = np.zeros(len(band_counts))
    for i, outcome in enumerate(logical_order):
        values = band_counts[outcome].values
        bars = ax.bar(range(len(band_counts)), values, bottom=bottom, 
                      label=outcome, color=colors[i], edgecolor='black', linewidth=1.2)
        
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., bottom[j] + height/2.,
                       f'{int(value)}', ha='center', va='center', fontsize=11, fontweight='bold')
        
        bottom += values
    
    totals = band_counts.sum(axis=1)
    for i, (model, total) in enumerate(totals.items()):
        ax.text(i, total + 100, f'Total: {int(total)}', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, max(totals) * 1.1)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of Projects', fontsize=12)
    ax.set_title('Distribution of Predicted Project Outcomes by Model (Aligned)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(band_counts)))
    ax.set_xticklabels(band_counts.index, fontsize=11)
    
    ax.legend(title='Predicted Outcome Band', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_outcomes_bands_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    percentages = band_counts.div(band_counts.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bottom = np.zeros(len(percentages))
    for i, outcome in enumerate(logical_order):
        values = percentages[outcome].values
        bars = ax.bar(range(len(percentages)), values, bottom=bottom, 
                      label=outcome, color=colors[i], edgecolor='black', linewidth=1.2)
        
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 5:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., bottom[j] + height/2.,
                       f'{value:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
        
        bottom += values
    
    ax.set_ylim(0, 100)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Percentage of Projects (%)', fontsize=12)
    ax.set_title('Distribution of Predicted Project Outcomes by Model (Percentage)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(percentages)))
    ax.set_xticklabels(percentages.index, fontsize=11)
    
    ax.legend(title='Predicted Outcome Band', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_outcomes_bands_percentage.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    band_counts.to_csv(os.path.join(output_dir, 'predicted_outcomes_summary_detailed.csv'))
    percentages.round(2).to_csv(os.path.join(output_dir, 'predicted_outcomes_percentages.csv'))
    
    print(f"Created detailed outcome bands visualizations")
    return band_counts, percentages

class DataPreprocessor:
    def __init__(self, story_cols, risks_cols, output_dir='results',
                 svd_n_components=50, svd_clip_range=(-10, 10),
                 fs_max_features=50, random_state=SEED):
        self.story_cols = []
        self.risks_cols = []
        self.output_dir = output_dir
        self.svd_n_components = svd_n_components
        self.svd_clip_range = svd_clip_range
        self.fs_max_features = fs_max_features
        self.random_state = random_state
        self.imputer = None
        self.scaler_non_emb = None
        self.scaler_story_emb = None
        self.scaler_risks_emb = None
        self.story_svd = None
        self.risks_svd = None
        self.one_hot_encoder_columns = None
        self.feature_selector = None
        self.selected_features = None
        self.numerical_cols_to_impute = None
        self.non_emb_cols_to_scale = None
        self._last_fit_X_index = pd.Index([])

    def _preprocess_embeddings_with_svd(self, X_df, fit=False):
        return X_df

    def fit(self, X_train_df, y_train_series):
        X_train = X_train_df.copy()
        y_train = y_train_series.copy()
        self._last_fit_X_index = X_train_df.index

        self.numerical_cols_to_impute = [col for col in X_train.select_dtypes(include=np.number).columns
                                       if not (col.startswith('Story_minilm_emb_') or col.startswith('Risks_minilm_emb_') or
                                               col.startswith('Story_roberta_emb_') or col.startswith('Risks_roberta_emb_') or
                                               col.startswith('Story_modernbert_emb_') or col.startswith('Risks_modernbert_emb_') or
                                               col.startswith('StorySVD_') or col.startswith('RisksSVD_'))]
        valid_numerical_cols = [col for col in self.numerical_cols_to_impute if not X_train[col].isna().all()]
        if len(valid_numerical_cols) < len(self.numerical_cols_to_impute):
            X_train = X_train.drop(columns=[c for c in self.numerical_cols_to_impute if c not in valid_numerical_cols])
        self.numerical_cols_to_impute = valid_numerical_cols

        if self.numerical_cols_to_impute:
            self.imputer = SimpleImputer(strategy='median')
            X_train[self.numerical_cols_to_impute] = self.imputer.fit_transform(X_train[self.numerical_cols_to_impute])

        self.non_emb_cols_to_scale = [c for c in X_train.select_dtypes(include=np.number).columns
                                     if not any(c.startswith(p) for p in ['Story_minilm_emb_', 'Risks_minilm_emb_', 
                                                                        'Story_roberta_emb_', 'Risks_roberta_emb_',
                                                                        'Story_modernbert_emb_', 'Risks_modernbert_emb_',
                                                                        'StorySVD_', 'RisksSVD_'])]
        if self.non_emb_cols_to_scale:
            self.scaler_non_emb = RobustScaler()
            X_train[self.non_emb_cols_to_scale] = self.scaler_non_emb.fit_transform(X_train[self.non_emb_cols_to_scale])

        X_train = pd.get_dummies(X_train, drop_first=True, dummy_na=False, columns=X_train.select_dtypes(include=['object', 'category']).columns)
        self.one_hot_encoder_columns = X_train.columns.tolist()

        actual_fs_max_features = min(self.fs_max_features, X_train.shape[1])
        if X_train.shape[1] > actual_fs_max_features and actual_fs_max_features > 0:
            X_fs = X_train.copy().replace([np.inf, -np.inf], np.nan)
            if X_fs.isnull().any().any():
                num_cols_fs = X_fs.select_dtypes(include=np.number).columns
                if num_cols_fs.any():
                    X_fs[num_cols_fs] = SimpleImputer(strategy='median').fit_transform(X_fs[num_cols_fs])
            rf_fs = RandomForestRegressor(random_state=self.random_state, n_jobs=-1, n_estimators=50, max_depth=10)
            self.feature_selector = SelectFromModel(rf_fs, threshold=-np.inf, max_features=actual_fs_max_features)
            try:
                self.feature_selector.fit(X_fs, y_train)
            except ValueError as e:
                print(f"Warning: Feature selection failed: {e}. Using all features.")
                self.feature_selector = None
            self.selected_features = X_train.columns[self.feature_selector.get_support()].tolist() if self.feature_selector else X_train.columns.tolist()
        else:
            self.selected_features = X_train.columns.tolist()
        return self

    def transform(self, X_df):
        X = X_df.copy()
        if not all([self.imputer, self.scaler_non_emb, self.one_hot_encoder_columns, self.selected_features]):
            raise RuntimeError("Preprocessor not fitted.")
        transform_num_cols = [col for col in self.numerical_cols_to_impute if col in X.columns]
        if transform_num_cols:
            X[transform_num_cols] = self.imputer.transform(X[transform_num_cols])
        if all(c in X.columns for c in ['goal', 'rewardscount', 'Story_Compound', 'Story_Flesch_Reading_Ease']):
            X['goal_per_reward'] = X['goal'] / (X['rewardscount'].replace(0, 1e-6) + 1e-6)
            X['story_sentiment_readability'] = X['Story_Compound'] * X['Story_Flesch_Reading_Ease']
        transform_non_emb_cols = [col for col in self.non_emb_cols_to_scale if col in X.columns]
        if transform_non_emb_cols:
            X[transform_non_emb_cols] = self.scaler_non_emb.transform(X[transform_non_emb_cols])
        X = pd.get_dummies(X, drop_first=True, dummy_na=False, columns=X.select_dtypes(include=['object', 'category']).columns)
        current_cols = X.columns.tolist()
        for col in self.one_hot_encoder_columns:
            if col not in current_cols:
                X[col] = 0
        try:
            X = X[self.one_hot_encoder_columns]
        except KeyError as e:
            missing_cols = [c for c in self.one_hot_encoder_columns if c not in X.columns]
            raise KeyError(f"Missing columns in transform: {missing_cols[:5]}...") from e
        if self.selected_features:
            try:
                X = X[self.selected_features]
            except KeyError as e:
                missing_sel_cols = [c for c in self.selected_features if c not in X.columns]
                raise KeyError(f"Missing selected features: {missing_sel_cols[:5]}...") from e
        X = X.reset_index(drop=True)
        dataset_type = "train_set" if X_df.index.equals(self._last_fit_X_index) else "test_set"
        X.to_csv(os.path.join(self.output_dir, f'transformed_{dataset_type}.csv'), index=False)
        return X

    def fit_transform(self, X_train_df, y_train_series):
        self.fit(X_train_df, y_train_series)
        return self.transform(X_train_df)

def extract_features(data, embedding_type='minilm'):
    """
    Extract features from raw data with support for different embedding types.
    
    Args:
        data: Raw JSON data
        embedding_type: 'minilm' or 'roberta' for different embedding types
    
    Returns:
        DataFrame with all features and embeddings
    """
    features, story_embeddings, risks_embeddings, error_count, skipped_count = [], [], [], 0, 0
    
    # Define embedding keys for different types
    embedding_keys = {
        'minilm': ('story_minilm_embedding', 'risk_minilm_embedding'),
        'roberta': ('story_roberta_embedding', 'risk_roberta_embedding'),
        'modernbert': ('story_modernbert_embedding', 'risk_modernbert_embedding')
    }
    
    story_key, risks_key = embedding_keys.get(embedding_type.lower(), embedding_keys['minilm'])
    
    for item in tqdm(data, desc=f"Extracting Features ({embedding_type})"):
        if item.get('state') not in ['successful', 'failed']:
            skipped_count += 1
            continue
        
        ratio = safe_float(item.get('pledged_to_goal_ratio'))
        if pd.isna(ratio) or ratio <= 0:
            skipped_count += 1
            continue
        
        try:
            # Extract all features from classification script
            f = {
                'pledged_to_goal_ratio': ratio,
                'goal': safe_float(item.get('goal'), 0.0),
                'projectFAQsCount': safe_float(item.get('projectFAQsCount'), 0.0),
                'commentsCount': safe_float(item.get('commentsCount'), 0.0),
                'updateCount': safe_float(item.get('updateCount'), 0.0),
                'rewardscount': safe_float(item.get('rewardscount'), 0.0),
                'project_length_days': safe_float(item.get('project_length_days')),
                'preparation_days': safe_float(item.get('preparation_days')),
                
                # Story Analysis features
                'Story_Avg_Sentence_Length': safe_float(item.get('Story Analysis', {}).get('Average Sentence Length')),
                'Story_Flesch_Reading_Ease': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Flesch Reading Ease')),
                'Story_Flesch_Kincaid_Grade': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Flesch-Kincaid Grade Level')),
                'Story_Gunning_Fog': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Gunning Fog Index')),
                'Story_SMOG': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('SMOG Index')),
                'Story_ARI': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Automated Readability Index')),
                
                # Risks Analysis features
                'Risks_Avg_Sentence_Length': safe_float(item.get('Risks and Challenges Analysis', {}).get('Average Sentence Length')),
                'Risks_Flesch_Reading_Ease': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Flesch Reading Ease')),
                'Risks_Flesch_Kincaid_Grade': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Flesch-Kincaid Grade Level')),
                'Risks_Gunning_Fog': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Gunning Fog Index')),
                'Risks_SMOG': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('SMOG Index')),
                'Risks_ARI': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Automated Readability Index')),
                
                # Sentiment features
                'Story_Positive': safe_float(item.get('Story_Positive')),
                'Story_Neutral': safe_float(item.get('Story_Neutral')),
                'Story_Negative': safe_float(item.get('Story_Negative')),
                'Story_Compound': safe_float(item.get('Story_Compound')),
                'Risks_Positive': safe_float(item.get('Risks_Positive')),
                'Risks_Neutral': safe_float(item.get('Risks_Neutral')),
                'Risks_Negative': safe_float(item.get('Risks_Negative')),
                'Risks_Compound': safe_float(item.get('Risks_Compound')),
                
                # GCI features
                'Story_GCI': safe_float(item.get('Story GCI')),
                'Risks_GCI': safe_float(item.get('Risks and Challenges GCI'))
            }
            
            # Add category features
            f["category_Web_Combined"] = 1 if item.get("category_Web", False) or item.get("category_Web Development", False) else 0
            for key, value in item.items():
                if key.startswith('category_') and key not in ['category_Technology', 'category_Web', 'category_Web Development']:
                    f[key] = 1 if value is True or str(value).lower() in ['true', '1'] else 0
            
            features.append(f)
            
            # Extract embeddings
            story_emb = item.get(story_key, [])
            risks_emb = item.get(risks_key, [])
            
            # Handle embeddings with proper error checking
            def process_embeddings(embeddings, default_dim=384):
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    processed = []
                    for x in embeddings:
                        try:
                            val = float(x) if x is not None else 0.0
                            processed.append(0.0 if not np.isfinite(val) else val)
                        except (ValueError, TypeError):
                            processed.append(0.0)
                    
                    # Pad or truncate to consistent dimension
                    if len(processed) < default_dim:
                        processed.extend([0.0] * (default_dim - len(processed)))
                    elif len(processed) > default_dim:
                        processed = processed[:default_dim]
                    
                    return processed
                return [0.0] * default_dim
            
            story_embeddings.append(process_embeddings(story_emb))
            risks_embeddings.append(process_embeddings(risks_emb))
            
        except Exception as e:
            error_count += 1
            print(f"Error extracting item: {e}")
    
    print(f"Extracted {len(features)} features, skipped {skipped_count}, errors {error_count}")
    if not features:
        return pd.DataFrame()
    
    df = pd.DataFrame(features)
    
    # Create embedding DataFrames
    story_df = pd.DataFrame(story_embeddings, columns=[f"Story_{embedding_type}_emb_{i}" for i in range(len(story_embeddings[0]))], index=df.index)
    risks_df = pd.DataFrame(risks_embeddings, columns=[f"Risks_{embedding_type}_emb_{i}" for i in range(len(risks_embeddings[0]))], index=df.index)
    
    return pd.concat([df, story_df, risks_df], axis=1)

def train_model(X_train, y_train, X_test, y_test, model_instance, search_space,
                cv_splitter_obj, model_name, output_dir, random_state_seed=SEED):
    if len(X_train) < 10:
        print(f"Skipping {model_name}: Insufficient training data ({len(X_train)} samples).")
        return None, None, {}
    
    if hasattr(model_instance, 'random_state'):
        model_instance.set_params(random_state=random_state_seed)
    if model_name == 'XGBoost' and hasattr(model_instance, 'seed'):
        model_instance.set_params(seed=random_state_seed)

    nan_count_x = X_train.isnull().sum().sum()
    nan_count_y = y_train.isnull().sum()
    if nan_count_x > 0 or nan_count_y > 0:
        print(f"Warning: {nan_count_x} NaN values in X_train, {nan_count_y} in y_train. Imputing with median.")
        X_train = X_train.copy().replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        y_train = pd.Series(imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel(), index=y_train.index)

    y_train_bins_hpo = make_stratified_bins(y_train, n_bins=5)
    stratify_hpo = y_train_bins_hpo if y_train_bins_hpo is not None and y_train_bins_hpo.nunique() > 1 else None
    min_val_size = max(2, int(0.15 * len(X_train)))
    test_size_hpo = min_val_size / len(X_train) if len(X_train) > min_val_size else 0.0
    if len(X_train) < 2 or test_size_hpo == 0.0:
        print("Warning: Using test set as validation due to small training data.")
        X_val_hpo, y_val_hpo = X_test.copy(), y_test.copy()
    else:
        _, X_val_hpo, _, y_val_hpo = train_test_split(X_train, y_train, test_size=test_size_hpo, random_state=random_state_seed, stratify=stratify_hpo)

    fit_kwargs = {}
    if model_name == "XGBoost":
        fit_kwargs = {'eval_set': [(X_val_hpo, y_val_hpo)], 'verbose': False}
    elif model_name == "LightGBM":
        fit_kwargs = {'callbacks': [lgb.early_stopping(stopping_rounds=10, verbose=-1)], 'eval_set': [(X_val_hpo, y_val_hpo)]}

    y_cv_bins_bayes = make_stratified_bins(y_train, n_bins=cv_splitter_obj.get_n_splits())
    cv_iter_bayes = KFold(n_splits=cv_splitter_obj.get_n_splits(), shuffle=True, random_state=cv_splitter_obj.random_state) if y_cv_bins_bayes is None or y_cv_bins_bayes.nunique() < 2 else cv_splitter_obj.split(X_train, y_cv_bins_bayes)

    opt = BayesSearchCV(model_instance, search_space, n_iter=15 if model_name != 'Ridge' else 3, cv=cv_iter_bayes, n_jobs=2, scoring='neg_mean_squared_error', random_state=random_state_seed, verbose=0)
    print(f"Starting HPO for {model_name} with {len(X_train)} samples...")
    opt.fit(X_train, y_train, **fit_kwargs)
    print(f"HPO completed for {model_name}.")
    
    best_model = opt.best_estimator_
    cv_metrics = cross_validate_model_with_skf(best_model, X_train, y_train, cv_splitter_obj)
    
    joblib.dump(best_model, os.path.join(output_dir, f'best_{model_name}.joblib'))
    print(f"Saved model: best_{model_name}.joblib")
    
    best_params = {k: (v.item() if hasattr(v, 'item') else v) for k, v in opt.best_params_.items()}
    with open(os.path.join(output_dir, f'best_{model_name}_params.json'), 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved parameters: best_{model_name}_params.json")
    
    metrics = evaluate_model(best_model, X_test, y_test, model_name, output_dir)
    metrics.update(cv_metrics)
    metrics['Best_Params'] = best_params

    if not X_test.empty:
        generate_enhanced_shap_analysis(best_model, X_train, X_test, y_train, model_name, output_dir, random_state_seed)

    return best_model, best_params, metrics

def preprocess_target_variable_improved(y_raw, method='cap_and_log'):
    y = pd.Series(y_raw).copy()
    mask = None
    
    if method == 'cap_and_log':
        cap_value = y.quantile(0.95)
        y_capped = y.clip(upper=cap_value)
        y_final = np.log1p(y_capped)
        print(f"Capped {sum(y_raw > cap_value)} values above {cap_value:.2f}, then applied log1p")
        
    elif method == 'winsorize_and_log':
        from scipy.stats import mstats
        y_winsorized = pd.Series(mstats.winsorize(y, limits=[0.01, 0.01]))
        y_final = np.log1p(y_winsorized)
        print("Applied winsorization (1% each tail) then log1p")
        
    else:
        y_final = np.log1p(y)
        print("Applied log1p transformation")
    
    print(f"\nBefore: Mean={y_raw.mean():.3f}, Std={y_raw.std():.3f}, Max={y_raw.max():.3f}")
    print(f"After:  Mean={y_final.mean():.3f}, Std={y_final.std():.3f}, Max={y_final.max():.3f}")
    return y_final, mask

def analyze_target_variable(y_raw):
    y = pd.Series(y_raw)
    print("=== TARGET VARIABLE ANALYSIS ===")
    print(f"Shape: {y.shape}")
    print(f"Min: {y.min():.3f}")
    print(f"Max: {y.max():.3f}")
    print(f"Mean: {y.mean():.3f}")
    print(f"Median: {y.median():.3f}")
    print(f"Std: {y.std():.3f}")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        print(f"{p}%: {y.quantile(p/100):.3f}")
    q99 = y.quantile(0.99)
    extreme_count = (y > q99).sum()
    print(f"\nValues above 99th percentile: {extreme_count} ({extreme_count/len(y)*100:.1f}%)")
    return y

def analyze_correlations_pure_features_only(X_raw, y_log):
    embedding_patterns = ['Story_minilm_emb_', 'Risks_minilm_emb_', 'Story_roberta_emb_', 'Risks_roberta_emb_',
                         'Story_modernbert_emb_', 'Risks_modernbert_emb_', 'StorySVD_', 'RisksSVD_']
    pure_feature_cols = [col for col in X_raw.columns 
                        if not any(col.startswith(pattern) for pattern in embedding_patterns)]
    
    X_pure = X_raw[pure_feature_cols]
    correlations = X_pure.select_dtypes(include=[np.number]).corrwith(y_log).abs().sort_values(ascending=False)
    
    print("\nTop 10 PURE feature correlations with target (log-transformed):")
    print(correlations.head(10))
    return correlations

def analyze_funding_success_patterns(y_raw):
    def categorize_funding(ratio):
        if ratio < 0.01:
            return "Complete Failure (< 1%)"
        elif ratio < 0.1:
            return "Major Failure (1-10%)"
        elif ratio < 0.8:
            return "Underfunded (10-80%)"
        elif ratio <= 1.2:
            return "Successfully Funded (80-120%)"
        elif ratio <= 2.0:
            return "Overfunded (120-200%)"
        else:
            return "Major Success (> 200%)"
    
    categories = y_raw.apply(categorize_funding)
    category_counts = categories.value_counts()
    category_percentages = (category_counts / len(y_raw) * 100).round(1)
    
    print("\n=== ACTUAL FUNDING DISTRIBUTION ===")
    for category, count in category_counts.items():
        pct = category_percentages[category]
        print(f"{category}: {count:,} projects ({pct}%)")
    
    return categories, category_counts

def save_comprehensive_results(results, output_dir):
    if not results:
        print("No results to save.")
        return
    
    res_df = pd.DataFrame(results)
    
    column_order = [
        'model', 'RMSE', 'MAE', 'R2', 'MSE', 'Within_20_Percent', 
        'Classification_Accuracy', 'CV_R2', 'CV_R2_std', 'CV_RMSE', 
        'CV_RMSE_std', 'CV_MAE', 'CV_MAE_std', 'Best_Params'
    ]
    
    existing_columns = [col for col in column_order if col in res_df.columns]
    res_df = res_df[existing_columns]
    
    res_df_sorted = res_df.sort_values('R2', ascending=False)
    
    results_file = os.path.join(output_dir, 'model_comparison.csv')
    res_df_sorted.to_csv(results_file, index=False)
    print(f"Saved comprehensive model comparison to: {results_file}")
    
    summary_stats = {
        'total_models': len(res_df),
        'best_model': res_df_sorted.iloc[0]['model'],
        'best_r2': float(res_df_sorted.iloc[0]['R2']),
        'best_rmse': float(res_df_sorted.iloc[0]['RMSE']),
        'best_mae': float(res_df_sorted.iloc[0]['MAE']),
        'model_rankings': res_df_sorted[['model', 'R2', 'RMSE', 'MAE']].to_dict('records')
    }
    
    summary_file = os.path.join(output_dir, 'model_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Saved model summary to: {summary_file}")
    
    return res_df_sorted

def create_detailed_outcome_bands_visualization_real(y_raw, output_dir):
    """
    Creates a detailed bar chart visualization of actual project outcome bands.
    
    Args:
        y_raw (pd.Series): Series containing the actual pledged_to_goal_ratio values.
        output_dir (str): Directory to save the output files.
    
    Returns:
        tuple: (band_counts, percentages) DataFrames with counts and percentages.
    """
    # Input validation
    if y_raw.empty or not pd.api.types.is_numeric_dtype(y_raw):
        raise ValueError("y_raw must be a non-empty numeric Series.")
    
    y_raw = y_raw.dropna()  # Remove NaN values to avoid issues in binning
    
    def bin_ratio(ratio):
        """Categorizes a funding ratio into outcome bands."""
        if pd.isna(ratio) or ratio < 0.1:
            return "Failed (< 0.1)"
        elif ratio < 0.8:
            return "Underfunded (0.1 ≤ x < 0.8)"
        elif ratio <= 1.2:
            return "Just Funded (0.8 ≤ x ≤ 1.2)"
        else:
            return "Overfunded (> 1.2)"

    # Apply binning to real values
    banded_results = pd.Series(y_raw).apply(bin_ratio)
    band_counts = banded_results.value_counts().reindex(
        ["Failed (< 0.1)", "Underfunded (0.1 ≤ x < 0.8)", "Just Funded (0.8 ≤ x ≤ 1.2)", "Overfunded (> 1.2)"],
        fill_value=0
    ).astype(int)
    
    # Calculate percentages
    percentages = (band_counts / band_counts.sum() * 100).round(1)
    
    # Define consistent colors matching the predicted chart
    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#d62728']
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(1)  # Single bar for all data
    for i, outcome in enumerate(band_counts.index):
        values = [band_counts[outcome]]
        bars = ax.bar(0, values, bottom=bottom, label=outcome, color=colors[i], edgecolor='black', linewidth=1.2)
        
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., bottom[0] + height/2.,
                       f'{int(value)}', ha='center', va='center', fontsize=11, fontweight='bold')
        
        bottom += values
    
    ax.text(0, band_counts.sum() + 100, f'Total: {int(band_counts.sum())}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylim(0, band_counts.sum() * 1.1)
    ax.set_xlabel('Actual Outcomes', fontsize=12)
    ax.set_ylabel('Number of Projects', fontsize=12)
    ax.set_title('Distribution of Actual Project Outcomes', fontsize=14, fontweight='bold')
    ax.set_xticks([0])
    ax.set_xticklabels(['All Data'], fontsize=11)
    
    ax.legend(title='Actual Outcome Band', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_outcomes_bands_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save counts and percentages to CSV
    band_counts.to_csv(os.path.join(output_dir, 'actual_outcomes_summary_detailed.csv'))
    percentages.to_csv(os.path.join(output_dir, 'actual_outcomes_percentages.csv'))
    
    print(f"Created detailed outcome bands visualization for actual values")
    return band_counts, percentages


from datetime import datetime
import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm.auto import tqdm

def main(data_path):
    """
    Main function to analyze project funding data, train models, and visualize outcomes.
    
    Args:
        data_path (str): Path to the JSON file containing project data.
    
    Returns:
        None: Saves results to the 'results_regression_v4' directory.
    """
    print(f"{datetime.now()} Starting Analysis")
    output_dir = 'results_regression_v4'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Fatal: Data loading failed: {e}")
        return

    # Extract features from raw data with MiniLM embeddings
    print("Extracting features with MiniLM embeddings...")
    raw_df_minilm = extract_features(data, embedding_type='minilm')
    X_raw_minilm = raw_df_minilm.drop(columns=['pledged_to_goal_ratio'])
    y_raw_minilm = raw_df_minilm['pledged_to_goal_ratio']
    
    # Extract features from raw data with RoBERTa embeddings
    print("Extracting features with RoBERTa embeddings...")
    raw_df_roberta = extract_features(data, embedding_type='roberta')
    X_raw_roberta = raw_df_roberta.drop(columns=['pledged_to_goal_ratio'])
    y_raw_roberta = raw_df_roberta['pledged_to_goal_ratio']

    # Analyze and visualize actual outcomes (using MiniLM data as reference)
    band_counts_actual, percentages_actual = create_detailed_outcome_bands_visualization_real(y_raw_minilm, output_dir)

    # Analyze target variable
    y_raw = analyze_target_variable(y_raw_minilm)
    y_processed, remove_mask = preprocess_target_variable_improved(y_raw_minilm, method='cap_and_log')

    if remove_mask is not None:
        X_raw_minilm = X_raw_minilm[remove_mask]
        X_raw_roberta = X_raw_roberta[remove_mask]
        y_processed = y_processed.reset_index(drop=True)
    y_log = y_processed

    # Analyze correlations and funding patterns
    correlations_pure = analyze_correlations_pure_features_only(X_raw_minilm, y_log)
    funding_categories, category_counts = analyze_funding_success_patterns(y_raw_minilm)

    print(f"\n=== PURE FEATURE INSIGHTS ===")
    print(f"Total pure features analyzed: {len(correlations_pure)}")
    print(f"Features with correlation > 0.3: {sum(correlations_pure > 0.3)}")
    print(f"Features with correlation > 0.2: {sum(correlations_pure > 0.2)}")
    print(f"\nWeakest correlations (bottom 5):")
    print(correlations_pure.tail())

    successful_projects = sum(category_counts[cat] for cat in category_counts.index 
                            if 'Successfully Funded' in cat or 'Overfunded' in cat or 'Major Success' in cat)
    print(f"\n=== BUSINESS INSIGHTS ===")
    print(f"Total successful projects (>80% funded): {successful_projects:,} ({successful_projects/len(y_raw_minilm)*100:.1f}%)")
    print(f"Median funding ratio: {y_raw_minilm.median():.1%}")
    print(f"Projects that exceeded goal: {sum(y_raw_minilm > 1.0):,} ({sum(y_raw_minilm > 1.0)/len(y_raw_minilm)*100:.1f}%)")
    print(f"Data shape after cleaning: {raw_df_minilm.shape}")
    print(f"Target variable stats: min={y_log.min():.3f}, max={y_log.max():.3f}, std={y_log.std():.3f}")
    print(f"Feature columns (MiniLM): {len(X_raw_minilm.columns)}")
    print(f"Feature columns (RoBERTa): {len(X_raw_roberta.columns)}")

    # Split data for both embedding types
    y_bins_raw_split = make_stratified_bins(y_log, n_bins=5)
    stratify_raw = y_bins_raw_split if y_bins_raw_split is not None and y_bins_raw_split.nunique() > 1 else None
    
    try:
        X_train_minilm, X_test_minilm, y_train_minilm, y_test_minilm = train_test_split(
            X_raw_minilm, y_log, test_size=0.2, random_state=SEED, stratify=stratify_raw)
        X_train_roberta, X_test_roberta, y_train_roberta, y_test_roberta = train_test_split(
            X_raw_roberta, y_log, test_size=0.2, random_state=SEED, stratify=stratify_raw)
    except ValueError:
        X_train_minilm, X_test_minilm, y_train_minilm, y_test_minilm = train_test_split(
            X_raw_minilm, y_log, test_size=0.2, random_state=SEED)
        X_train_roberta, X_test_roberta, y_train_roberta, y_test_roberta = train_test_split(
            X_raw_roberta, y_log, test_size=0.2, random_state=SEED)

    # Outlier check for MiniLM
    X_train_minilm_no_o, y_train_minilm_no_o, outlier_mask_minilm = remove_outliers(
        X_train_minilm, y_train_minilm, threshold=3, remove=False)
    print(f"MiniLM training data after outlier check: {len(X_train_minilm_no_o)} rows, {sum(~outlier_mask_minilm)} outliers flagged.")
    
    # Outlier check for RoBERTa
    X_train_roberta_no_o, y_train_roberta_no_o, outlier_mask_roberta = remove_outliers(
        X_train_roberta, y_train_roberta, threshold=3, remove=False)
    print(f"RoBERTa training data after outlier check: {len(X_train_roberta_no_o)} rows, {sum(~outlier_mask_roberta)} outliers flagged.")
    
    if X_train_minilm_no_o.empty or X_train_roberta_no_o.empty:
        print("Error: Training data empty after outlier check.")
        return

    # Preprocess data for MiniLM
    print("Preprocessing MiniLM data...")
    s_cols = []
    r_cols = []
    preproc_minilm = DataPreprocessor(story_cols=s_cols, risks_cols=r_cols, output_dir=output_dir, random_state=SEED)
    preproc_minilm.fit(X_train_minilm_no_o.copy(), y_train_minilm_no_o.copy())
    X_train_minilm_p = preproc_minilm.transform(X_train_minilm_no_o.copy())
    y_train_minilm_f = y_train_minilm_no_o.reset_index(drop=True)
    X_train_minilm_p = X_train_minilm_p.reset_index(drop=True)
    X_test_minilm_p = preproc_minilm.transform(X_test_minilm.copy())
    y_test_minilm_f = y_test_minilm.reset_index(drop=True)
    
    # Preprocess data for RoBERTa
    print("Preprocessing RoBERTa data...")
    preproc_roberta = DataPreprocessor(story_cols=s_cols, risks_cols=r_cols, output_dir=output_dir, random_state=SEED)
    preproc_roberta.fit(X_train_roberta_no_o.copy(), y_train_roberta_no_o.copy())
    X_train_roberta_p = preproc_roberta.transform(X_train_roberta_no_o.copy())
    y_train_roberta_f = y_train_roberta_no_o.reset_index(drop=True)
    X_train_roberta_p = X_train_roberta_p.reset_index(drop=True)
    X_test_roberta_p = preproc_roberta.transform(X_test_roberta.copy())
    y_test_roberta_f = y_test_roberta.reset_index(drop=True)

    # Validation checks
    if len(X_train_minilm_p) != len(y_train_minilm_f) or len(X_test_minilm_p) != len(y_test_minilm_f):
        print(f"Length Mismatch! MiniLM Train X:{len(X_train_minilm_p)} y:{len(y_train_minilm_f)}. Test X:{len(X_test_minilm_p)} y:{len(y_test_minilm_f)}")
        return
    if len(X_train_roberta_p) != len(y_train_roberta_f) or len(X_test_roberta_p) != len(y_test_roberta_f):
        print(f"Length Mismatch! RoBERTa Train X:{len(X_train_roberta_p)} y:{len(y_train_roberta_f)}. Test X:{len(X_test_roberta_p)} y:{len(y_test_roberta_f)}")
        return
    if X_train_minilm_p.empty or X_test_minilm_p.empty or X_train_roberta_p.empty or X_test_roberta_p.empty:
        print("Error: Processed data is empty.")
        return

    # Define models and search spaces
    models = {
        'RandomForest': (RandomForestRegressor(random_state=SEED), {
            'n_estimators': Integer(50, 200), 'max_depth': Integer(5, 15),
            'min_samples_split': Integer(2, 10), 'min_samples_leaf': Integer(2, 10),
            'max_features': Real(0.6, 0.9)
        }),
        'XGBoost': (XGBRegressor(objective='reg:squarederror', random_state=SEED, seed=SEED, tree_method='hist'), {
            'n_estimators': Integer(50, 300), 'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.2, prior='log-uniform'), 'subsample': Real(0.7, 1.0)
        }),
        'LightGBM': (lgb.LGBMRegressor(random_state=SEED, force_col_wise=True, verbosity=-1), {
            'n_estimators': Integer(50, 300), 'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.2, prior='log-uniform'), 'num_leaves': Integer(10, 50)
        }),
        'Ridge': (Ridge(alpha=1e-3), {
            'alpha': Real(1e-6, 1e-1, prior='log-uniform'),
            'fit_intercept': Categorical([True, False])
        })
    }
    
    # Train models for both embedding types
    results = []
    best_models_minilm = {}
    best_models_roberta = {}
    skf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    print("Starting model training loop for MiniLM embeddings...")
    for name, (model, space) in tqdm(models.items(), desc="Training Models (MiniLM)", total=len(models)):
        print(f"Training {name} with MiniLM embeddings...")
        model_name = f"{name}_MiniLM"
        best_model, best_params, metrics = train_model(
            X_train_minilm_p, y_train_minilm_f, X_test_minilm_p, y_test_minilm_f, 
            model, space, skf_cv, model_name, output_dir, SEED)
        if metrics:
            results.append(metrics)
            best_models_minilm[name] = best_model
    
    print("Starting model training loop for RoBERTa embeddings...")
    for name, (model, space) in tqdm(models.items(), desc="Training Models (RoBERTa)", total=len(models)):
        print(f"Training {name} with RoBERTa embeddings...")
        model_name = f"{name}_RoBERTa"
        best_model, best_params, metrics = train_model(
            X_train_roberta_p, y_train_roberta_f, X_test_roberta_p, y_test_roberta_f, 
            model, space, skf_cv, model_name, output_dir, SEED)
        if metrics:
            results.append(metrics)
            best_models_roberta[name] = best_model

    if not results:
        print("No models trained.")
        return

    res_df = save_comprehensive_results(results, output_dir)
    print(f"\n{datetime.now()} Saved comprehensive results.")
    print(f"Top Models:\n{res_df[['model', 'R2', 'CV_R2', 'RMSE']].head()}")
    
    if not res_df.empty:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='R2', y='model', data=res_df, palette='viridis', orient='h')
        plt.title('R² Comparison - MiniLM vs RoBERTa Embeddings')
        plt.xlabel('R² Score')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'r2_comparison_minilm_vs_roberta.png'), dpi=300)
        plt.close()

    # Generate predictions for full dataset with both embedding types
    print("Generating predictions for full dataset...")
    
    # MiniLM predictions
    X_full_minilm_p = preproc_minilm.transform(X_raw_minilm.reset_index(drop=True))
    print(f"Full MiniLM dataset shape after preprocessing: {X_full_minilm_p.shape}")
    
    predictions_minilm = {}
    for model_name, model in best_models_minilm.items():
        pred_log = model.predict(X_full_minilm_p)
        predictions_minilm[f"{model_name}_MiniLM"] = np.expm1(pred_log)
        predictions_minilm[f"{model_name}_MiniLM"] = np.where(np.isnan(predictions_minilm[f"{model_name}_MiniLM"]), 0, predictions_minilm[f"{model_name}_MiniLM"])
    
    # RoBERTa predictions
    X_full_roberta_p = preproc_roberta.transform(X_raw_roberta.reset_index(drop=True))
    print(f"Full RoBERTa dataset shape after preprocessing: {X_full_roberta_p.shape}")
    
    predictions_roberta = {}
    for model_name, model in best_models_roberta.items():
        pred_log = model.predict(X_full_roberta_p)
        predictions_roberta[f"{model_name}_RoBERTa"] = np.expm1(pred_log)
        predictions_roberta[f"{model_name}_RoBERTa"] = np.where(np.isnan(predictions_roberta[f"{model_name}_RoBERTa"]), 0, predictions_roberta[f"{model_name}_RoBERTa"])
    
    # Combine predictions
    predictions_df = pd.DataFrame({**predictions_minilm, **predictions_roberta})
    band_counts_orig, percentages = create_detailed_outcome_bands_visualization(predictions_df, output_dir)

    # Final summary including actual outcomes
    all_models = list(best_models_minilm.keys()) + list(best_models_roberta.keys())
    final_summary = {
        'analysis_completed': str(datetime.now()),
        'total_projects_analyzed': len(X_raw_minilm),
        'embedding_types_compared': ['minilm', 'roberta'],
        'models_trained': all_models,
        'best_performing_model': res_df.iloc[0]['model'],
        'actual_outcome_distribution': {
            'counts': band_counts_actual.to_dict(),
            'percentages': percentages_actual.to_dict()
        },
        'files_generated': {
            'model_comparison': 'model_comparison.csv',
            'model_summary': 'model_summary.json',
            'r2_comparison': 'r2_comparison_minilm_vs_roberta.png',
            'joblib_files': [f'best_{name}.joblib' for name in all_models],
            'params_files': [f'best_{name}_params.json' for name in all_models],
            'shap_files': [f'shap_{name}_top_*_features.png' for name in all_models],
            'shap_bar_charts': [f'shap_{name}_bar_chart.png' for name in all_models],
            'shap_importance_files': [f'shap_feature_importance_{name}.csv' for name in all_models],
            'shap_summary_files': [f'shap_summary_{name}.json' for name in all_models],
            'outcome_visualizations': ['predicted_outcomes_bands_detailed.png', 'predicted_outcomes_bands_percentage.png'],
            'actual_outcome_files': ['actual_outcomes_summary_detailed.csv', 'actual_outcomes_percentages.csv']
        }
    }
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(final_summary, f, indent=2)

    print(f"Total rows processed: {len(predictions_df)} out of {len(X_raw_minilm)} expected.")
    print(f"\n{datetime.now()} Analysis Complete")
    print(f"\nAll files saved to: {output_dir}")
    print(f"Embedding types compared: {final_summary['embedding_types_compared']}")
    print(f"Total models trained: {len(all_models)}")
    print("Generated files:")
    for category, files in final_summary['files_generated'].items():
        if isinstance(files, list):
            print(f"  {category}: {len(files)} files")
        else:
            print(f"  {category}: {files}")

if __name__ == "__main__":
    data_path = "/Users/kerenlint/Projects/cursor/projects_with_short_stories_and_risks_with_embeddings.json"
    try:
        main(data_path)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()