import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold # Use KFold for regression
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer # Regression metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
import traceback
import time
import os
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

# Helper function for safe float conversion
def safe_float(value, default=np.nan): # Default to NaN for regression features initially
    """Safely convert value to float, handling None and non-numeric types."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        # Keep warnings minimal or use logging
        # print(f"Warning: Could not convert '{value}' to float. Using default {default}.")
        return default

# --- Modified extract_features function ---
def extract_features(data, embedding_type=None, minilm_embeddings=None):
    """
    Extract features for OLS Regression.
    Target: 'pledged_to_goal_ratio'.
    Removed: 'commentsCount', 'updateCount', 'projectFAQsCount'.
    Handles missing target and features robustly AFTER initial DataFrame creation.
    """
    features = []
    embedding_dimensions = 0 # Store max embedding dim found
    # --- FIX: Initialize error_count here ---
    error_count = 0
    # -----------------------------------------
    processed_count = 0 # Keep this if you use it later, otherwise remove

    for item_index, item in enumerate(data): # Added enumerate for better error reporting
        try:
            # Skip Technology category if needed
            if item.get("category_Technology", False):
                continue

            project_id = item.get('id', None)

            # --- Core feature dictionary ---
            feature_dict = {
                # TARGET VARIABLE - Keep it here for now, separate later
                'pledged_to_goal_ratio': item.get('pledged_to_goal_ratio', None),

                # --- REMOVED FEATURES ---
                 'projectFAQsCount': safe_float(item.get('projectFAQsCount')),
                 #'commentsCount': safe_float(item.get('commentsCount')),
                 #'updateCount': safe_float(item.get('updateCount')),
                # --- END REMOVED ---
                'goal': safe_float(item.get('goal')),
                'rewardscount': safe_float(item.get('rewardscount')),
                'project_length_days': safe_float(item.get('project_length_days')),
                'preparation_days': safe_float(item.get('preparation_days')),

                # Readability Scores
                'Story_Avg_Sentence_Length': safe_float(item.get('Story Analysis', {}).get('Average Sentence Length')),
                'Story_Flesch_Reading_Ease': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Flesch Reading Ease')),
                'Story_Flesch_Kincaid_Grade': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Flesch-Kincaid Grade Level')),
                'Story_Gunning_Fog': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Gunning Fog Index')),
                'Story_SMOG': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('SMOG Index')),
                'Story_ARI': safe_float(item.get('Story Analysis', {}).get('Readability Scores', {}).get('Automated Readability Index')),

                'Risks_Avg_Sentence_Length': safe_float(item.get('Risks and Challenges Analysis', {}).get('Average Sentence Length')),
                'Risks_Flesch_Reading_Ease': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Flesch Reading Ease')),
                'Risks_Flesch_Kincaid_Grade': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Flesch-Kincaid Grade Level')),
                'Risks_Gunning_Fog': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Gunning Fog Index')),
                'Risks_SMOG': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('SMOG Index')),
                'Risks_ARI': safe_float(item.get('Risks and Challenges Analysis', {}).get('Readability Scores', {}).get('Automated Readability Index')),

                # Sentiment Scores
                'Story_Positive': safe_float(item.get('Story_Positive')),
                'Story_Neutral': safe_float(item.get('Story_Neutral')),
                'Story_Negative': safe_float(item.get('Story_Negative')),
                'Story_Compound': safe_float(item.get('Story_Compound')),
                'Risks_Positive': safe_float(item.get('Risks_Positive')),
                'Risks_Neutral': safe_float(item.get('Risks_Neutral')),
                'Risks_Negative': safe_float(item.get('Risks_Negative')),
                'Risks_Compound': safe_float(item.get('Risks_Compound')),

                # GCI Scores
                'Story_GCI': safe_float(item.get('Story GCI')),
                'Risks_GCI': safe_float(item.get('Risks and Challenges GCI')),
            }

            # Combine Web categories
            feature_dict["category_Web_Combined"] = 1 if (item.get("category_Web", False) or item.get("category_Web Development", False)) else 0

            # Add other categories, excluding specific ones
            for key, value in item.items():
                if key.startswith("category_") and key not in ["category_Technology", "category_Web", "category_Web Development"]:
                    feature_dict[key] = 1 if value else 0

            # --------------- Handle Embeddings -----------------
            # RoBERTa Risks Only (Special case: only risks embeddings + target)
            if embedding_type == 'roberta_risks':
                risks_embeddings = item.get('risks-and-challenges_roberta')
                if isinstance(risks_embeddings, list):
                    current_emb_dim = len(risks_embeddings)
                    embedding_dimensions = max(embedding_dimensions, current_emb_dim) # Track max dim
                    roberta_risks_feature_dict = {'pledged_to_goal_ratio': feature_dict['pledged_to_goal_ratio']}
                    for i, val in enumerate(risks_embeddings):
                        roberta_risks_feature_dict[f'risks_roberta_emb_{i}'] = safe_float(val, default=0.0) # Default 0 for embeddings
                    features.append(roberta_risks_feature_dict)
                continue # Skip default append

            # Other embedding types (added to the main feature_dict)
            elif embedding_type == 'modernbert':
                story_embeddings = item.get('story_modernbert')
                risks_embeddings = item.get('risks_modernbert')
                story_dim, risks_dim = 0, 0
                if isinstance(story_embeddings, list):
                    story_dim = len(story_embeddings)
                    for i, val in enumerate(story_embeddings): feature_dict[f'story_modernbert_emb_{i}'] = safe_float(val, default=0.0)
                if isinstance(risks_embeddings, list):
                    risks_dim = len(risks_embeddings)
                    for i, val in enumerate(risks_embeddings): feature_dict[f'risks_modernbert_emb_{i}'] = safe_float(val, default=0.0)
                embedding_dimensions = max(embedding_dimensions, story_dim + risks_dim)

            elif embedding_type == 'roberta': # General RoBERTa (story + risks)
                story_embeddings = item.get('story_roberta')
                risks_embeddings = item.get('risks-and-challenges_roberta')
                story_dim, risks_dim = 0, 0
                if isinstance(story_embeddings, list):
                    story_dim = len(story_embeddings)
                    for i, val in enumerate(story_embeddings): feature_dict[f'story_roberta_emb_{i}'] = safe_float(val, default=0.0)
                if isinstance(risks_embeddings, list):
                    risks_dim = len(risks_embeddings)
                    for i, val in enumerate(risks_embeddings): feature_dict[f'risks_roberta_emb_{i}'] = safe_float(val, default=0.0) # Use correct key
                embedding_dimensions = max(embedding_dimensions, story_dim + risks_dim)

            elif embedding_type == 'miniLM' or embedding_type == 'minilm':
                story_miniLM = item.get('story_miniLM')
                risks_miniLM = item.get('risks_miniLM')
                story_dim, risks_dim = 0, 0
                if isinstance(story_miniLM, list):
                    story_dim = len(story_miniLM)
                    for i, val in enumerate(story_miniLM): feature_dict[f'story_miniLM_emb_{i}'] = safe_float(val, default=0.0)
                if isinstance(risks_miniLM, list):
                    risks_dim = len(risks_miniLM)
                    for i, val in enumerate(risks_miniLM): feature_dict[f'risks_miniLM_emb_{i}'] = safe_float(val, default=0.0)
                embedding_dimensions = max(embedding_dimensions, story_dim + risks_dim)
                # Note: Fallback to external minilm_embeddings is removed for simplicity here,
                # assuming embeddings are directly in the item if specified. Add back if needed.

            # Append the fully populated feature dict (unless it was the special roberta_risks case)
            if embedding_type != 'roberta_risks':
                 features.append(feature_dict)

            processed_count += 1 # Increment processed count here

        except Exception as e:
            error_count += 1 # Now this will work correctly
            item_id = item.get('id', 'UNKNOWN')
            print(f"Error processing item index {item_index}, ID {item_id}: {e}")
            # traceback.print_exc() # Uncomment for detailed stack trace if needed
            continue # Continue to next item

    # --- FIX: Use the initialized error_count in the final print statement ---
    print(f"Initial feature extraction summary: Processed={len(features)}, Errors={error_count}")
    # -----------------------------------------------------------------------


    # --- Post-processing after creating list of dicts ---
    if not features:
        print("Warning: No features extracted. Returning empty DataFrame.")
        return pd.DataFrame(), 0

    df = pd.DataFrame(features)
    print(f"Created DataFrame with shape: {df.shape}")

    # Ensure target variable is present and numeric, dropping invalid rows
    if 'pledged_to_goal_ratio' not in df.columns:
        print("Error: Target column 'pledged_to_goal_ratio' not found in extracted features.")
        return pd.DataFrame(), 0
    else:
        df['pledged_to_goal_ratio'] = pd.to_numeric(df['pledged_to_goal_ratio'], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=['pledged_to_goal_ratio'], inplace=True)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} rows due to missing or invalid 'pledged_to_goal_ratio'.")

    if df.empty:
        print("DataFrame is empty after dropping rows with invalid target. Cannot proceed.")
        return pd.DataFrame(), 0

    # Identify actual columns present in the DataFrame
    present_cols = df.columns.tolist()

    # Fill remaining NaNs in features (use median for robustness)
    feature_cols = [col for col in present_cols if col != 'pledged_to_goal_ratio']
    nan_counts_before_fill = df[feature_cols].isna().sum()
    total_nans_before = nan_counts_before_fill.sum()
    if total_nans_before > 0:
        print(f"Found {total_nans_before} NaN values in feature columns before final fill.")
        for col in feature_cols:
            if df[col].isna().any():
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_val = df[col].median()
                if pd.isna(median_val):
                     median_val = 0
                df[col].fillna(median_val, inplace=True)
        nan_counts_after_fill = df[feature_cols].isna().sum().sum()
        print(f"NaN values in feature columns after final fill: {nan_counts_after_fill}")

    return df, embedding_dimensions

# --- Preprocessing function with outlier removal for Regression ---
def preprocess_data(X, y, feature_selection=False, n_features_to_select=None):
    """
    Preprocessing for Regression: Outlier removal on y, scaling, normalization, opt. feature selection.
    Returns processed X (DataFrame or array), processed y (array), selected feature names (list or None).
    """
    print(f"Preprocessing started. Initial data shape X: {X.shape}, y: {y.shape}")

    # Ensure X is DataFrame for consistent processing, y is Series/Array
    if not isinstance(X, pd.DataFrame):
         print("Warning: X is not a DataFrame in preprocess_data. Feature names might be lost.")
         original_columns = None
         original_index = None
    else:
        original_columns = X.columns.tolist()
        original_index = X.index

    if not isinstance(y, (pd.Series, np.ndarray)):
        y = np.array(y)

    # 1. Outlier Removal based on Target Variable (y)
    if y is not None and len(y) > 0:
        y_series = pd.Series(y, index=original_index if original_index is not None and len(original_index) == len(y) else None)
        q1 = y_series.quantile(0.25)
        q3 = y_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        inlier_mask = (y_series >= lower_bound) & (y_series <= upper_bound) & (~y_series.isna())

        if isinstance(X, pd.DataFrame):
             if original_index is not None and X.index.equals(y_series.index): # Check alignment
                  aligned_mask = inlier_mask # Already aligned
             elif original_index is not None:
                  print("Warning: X and y indices differ. Reindexing mask.")
                  aligned_mask = inlier_mask.reindex(X.index, fill_value=False)
             else: # Cannot align if indices were lost
                  print("Warning: Cannot guarantee index alignment for outlier removal.")
                  aligned_mask = inlier_mask # Assume alignment (risky)

             # Apply mask only if shapes are compatible after potential alignment
             if len(aligned_mask) == len(X):
                  X_filtered = X[aligned_mask]
                  y_filtered = y_series[aligned_mask].values
             else:
                   print("Error: Mask shape mismatch after alignment attempt. Skipping outlier removal.")
                   X_filtered = X
                   y_filtered = y_series.values # Keep original y as array

        else: # X is numpy array
            if len(inlier_mask) == len(X):
                X_filtered = X[inlier_mask.values]
                y_filtered = y_series[inlier_mask].values
            else:
                print("Error: Shape mismatch between X array and y mask. Skipping outlier removal.")
                X_filtered = X
                y_filtered = y # Keep original y array

        rows_removed = len(y) - len(y_filtered)
        if rows_removed > 0:
             print(f"Removed {rows_removed} samples due to target variable outliers/NaNs (bounds: [{lower_bound:.4f}, {upper_bound:.4f}]).")
        X, y = X_filtered, y_filtered # Update X and y
        # Update original_columns if X is DataFrame and rows were removed
        if isinstance(X, pd.DataFrame):
            original_columns = X.columns.tolist() # Columns remain same, index changes
            original_index = X.index # Update index

    else:
        print("Skipping outlier removal (y is None or empty).")
        if isinstance(y, pd.Series): y = y.values # Ensure numpy array output


    # Check if data remains after outlier removal
    if (isinstance(X, pd.DataFrame) and X.empty) or (isinstance(X, np.ndarray) and X.shape[0] == 0):
        print("Error: No data remaining after outlier removal. Cannot preprocess further.")
        return pd.DataFrame(), np.array([]), None


    # 2. Convert to Numeric & Drop Non-Numeric (ensure this happens)
    numeric_cols = []
    if isinstance(X, pd.DataFrame):
        original_cols_current = X.columns.tolist() # Columns after potential filtering
        cols_to_drop = []
        for col in original_cols_current:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                if not pd.api.types.is_numeric_dtype(X[col]):
                    cols_to_drop.append(col)
                else:
                    numeric_cols.append(col)
            except Exception as e:
                print(f"Error converting column '{col}' during preprocessing: {e}. Dropping.")
                cols_to_drop.append(col)

        if cols_to_drop:
            print(f"Dropping non-numeric columns identified during preprocessing: {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)
            numeric_cols = X.columns.tolist() # Update numeric cols list

        if X.empty or not numeric_cols:
             print("Error: No numeric columns left after type conversion. Cannot proceed.")
             return pd.DataFrame(), y, None
        X_numeric = X # Use the remaining numeric DataFrame
        current_feature_names = numeric_cols
        current_index = X.index # Get index from remaining DataFrame
    else: # X is numpy array, assume numeric
        X_numeric = X
        current_feature_names = None # Names are lost
        current_index = None # Index is lost


    # 3. Handle infinities
    inf_count = np.isinf(X_numeric.values).sum() if isinstance(X_numeric, pd.DataFrame) else np.isinf(X_numeric).sum()
    if inf_count > 0:
        print(f"Found {inf_count} Inf values. Replacing with NaN.")
        if isinstance(X_numeric, pd.DataFrame):
             X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        else:
             X_numeric[np.isinf(X_numeric)] = np.nan


    # 4. Imputation
    nan_before_impute = np.isnan(X_numeric.values).sum() if isinstance(X_numeric, pd.DataFrame) else np.isnan(X_numeric).sum()
    print(f"NaN values before imputation: {nan_before_impute}")
    imputer = SimpleImputer(strategy='mean')
    # Apply imputer based on type
    X_imputed_array = imputer.fit_transform(X_numeric.values) if isinstance(X_numeric, pd.DataFrame) else imputer.fit_transform(X_numeric)
    print(f"NaN values after imputation: {np.isnan(X_imputed_array).sum()}")


    # 5. Scaling
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_imputed_array)

    # 6. Normalization
    normalizer = Normalizer(norm='l2')
    X_normalized_array = normalizer.fit_transform(X_scaled_array)

    # 7. Feature Selection (Optional)
    selected_feature_names_final = current_feature_names # Default if no selection
    X_final_array = X_normalized_array # Default

    if feature_selection and n_features_to_select is not None and n_features_to_select > 0 and current_feature_names is not None:
        if n_features_to_select >= X_normalized_array.shape[1]:
            print(f"Requested {n_features_to_select} features, but only {X_normalized_array.shape[1]} available. Selecting all.")
            n_features_to_select = X_normalized_array.shape[1]

        if n_features_to_select > 0 and X_normalized_array.shape[0] > 0: # Check data exists
            print(f"Performing RFE feature selection to select top {n_features_to_select} features using Lasso...")
            selector_model = Lasso(alpha=0.001, max_iter=3000, random_state=42, tol=1e-3)
            rfe = RFE(estimator=selector_model, n_features_to_select=n_features_to_select, step=0.1)
            try:
                X_final_array = rfe.fit_transform(X_normalized_array, y) # Use current y array
                selected_indices = rfe.get_support(indices=True)
                selected_feature_names_final = [current_feature_names[i] for i in selected_indices]
                print(f"Selected {len(selected_feature_names_final)} features via RFE.")
            except Exception as e:
                print(f"Error during RFE: {e}. Skipping feature selection.")
                selected_feature_names_final = current_feature_names # Revert
                X_final_array = X_normalized_array # Revert
        else:
             print("Skipping feature selection (n_features_to_select <= 0 or no data).")
             selected_feature_names_final = current_feature_names # Revert

    # Return processed data, trying to return DataFrame with correct index/columns
    if selected_feature_names_final is not None and current_index is not None:
         X_return = pd.DataFrame(X_final_array, columns=selected_feature_names_final, index=current_index)
         final_names_list = selected_feature_names_final
    else:
         # Fallback if names/index were lost
         X_return = X_final_array
         final_names_list = None

    print(f"Preprocessing finished. Final data shape X: {X_return.shape}, y: {y.shape}")
    return X_return, y, final_names_list


# --- Function to find optimal regression model and parameters ---
def optimize_ols_parameters(X, y, cv=10):
    """
    Finds the best regression model (Linear, Ridge, Lasso, ElasticNet) and its optimal
    hyperparameters using GridSearchCV.
    Returns the best parameters dictionary, the best fitted pipeline, and model name.
    """
    print("\n--- Optimizing Regression Parameters ---")
    X_values = X.values if isinstance(X, pd.DataFrame) else X # Use numpy array
    y_values = y.values if isinstance(y, pd.Series) else y

    # Final check for NaN/inf in inputs before grid search
    if np.any(np.isnan(X_values)) or np.any(np.isinf(X_values)):
        print("Error: NaN/Inf found in features before grid search. Imputing again.")
        X_values = SimpleImputer(strategy='mean').fit_transform(X_values)
    if np.any(np.isnan(y_values)) or np.any(np.isinf(y_values)):
        print("Error: NaN/Inf found in target before grid search. Cannot proceed.")
        return None, None, None

    # Define models and parameter grids
    models_and_params = [
        ('LinearRegression', Pipeline([('regressor', LinearRegression())]), {}),
        ('Ridge', Pipeline([('regressor', Ridge(random_state=42, max_iter=3000))]),
             {'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
              'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']}),
        ('Lasso', Pipeline([('regressor', Lasso(random_state=42, max_iter=5000, tol=1e-3))]),
             {'regressor__alpha': [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]}),
        ('ElasticNet', Pipeline([('regressor', ElasticNet(random_state=42, max_iter=5000, tol=1e-3))]),
             {'regressor__alpha': [0.0001, 0.001, 0.01, 0.1],
              'regressor__l1_ratio': [0.1, 0.5, 0.9]})
    ]

    # Use KFold for regression cross-validation
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)

    best_score = -np.inf # Maximize R2 or minimize neg MSE
    best_params = None
    best_estimator = None
    best_model_name = "N/A"
    scoring_metric = 'r2' # Use R-squared as the primary optimization metric

    print(f"Performing GridSearchCV with cv={cv} for models: {[m[0] for m in models_and_params]}")

    for name, pipeline, params in models_and_params:
        start_time = time.time()
        print(f"  Tuning {name}...")
        grid_search = GridSearchCV(pipeline, params, scoring=scoring_metric, cv=cv_strategy, n_jobs=-1, verbose=0)
        try:
            grid_search.fit(X_values, y_values)
            end_time = time.time()
            score = grid_search.best_score_
            print(f"    Completed in {end_time - start_time:.2f}s. Best {scoring_metric}: {score:.6f}")
            if score > best_score:
                best_score = score
                best_params = grid_search.best_params_
                best_estimator = grid_search.best_estimator_
                best_model_name = name
        except Exception as e:
            print(f"    Error during GridSearchCV for {name}: {e}")


    if best_estimator is not None:
        print(f"\nOptimization complete. Best model type: {best_model_name}")
        print(f"Best {scoring_metric} found: {best_score:.6f}")
        print(f"Best parameters: {best_params}")
    else:
        print("\nWarning: GridSearchCV failed to find an optimal model. Falling back to default LinearRegression.")
        best_estimator = Pipeline([('regressor', LinearRegression())])
        best_params = {}
        best_model_name = "LinearRegression (Fallback)"

    return best_params, best_estimator, best_model_name

# --- Feature Importance using Lasso ---
def get_feature_importance(X, y, feature_names, model_name):
    """ Calculates feature importance using a consistent Lasso model. """
    print(f"\n--- Calculating Feature Importance for: {model_name} (using Lasso) ---")

    if X is None or y is None or X.shape[0] == 0:
        print("Error: Input data for feature importance is invalid or empty.")
        return pd.DataFrame()

    # Ensure feature_names is a list
    if feature_names is None:
         if isinstance(X, pd.DataFrame):
             feature_names = X.columns.tolist()
         else:
             print("Warning: Cannot determine feature names for importance calculation.")
             # Generate generic names based on the number of features in X
             if isinstance(X, np.ndarray):
                 feature_names = [f"feature_{i}" for i in range(X.shape[1])]
             else: # Cannot determine shape
                 return pd.DataFrame()

    feature_names_list = list(feature_names) # Work with a list copy

    # Preprocess data specifically for importance (no selection)
    X_processed, y_processed, processed_feature_names = preprocess_data(X.copy(), y.copy(), feature_selection=False)

    # Check if preprocessing was successful
    if (isinstance(X_processed, pd.DataFrame) and X_processed.empty) or \
       (isinstance(X_processed, np.ndarray) and X_processed.shape[0] == 0) or \
       y_processed.shape[0] == 0:
         print("Preprocessing resulted in empty data for importance calculation. Skipping.")
         return pd.DataFrame()


    # Use the feature names returned by preprocess_data if available
    if processed_feature_names:
        current_feature_names = processed_feature_names
    else:
         # Fallback if names were lost (e.g., X wasn't DataFrame initially)
         if X_processed.shape[1] == len(feature_names_list):
              current_feature_names = feature_names_list # Use original list if length matches
              print("Warning: Using original feature names list as fallback.")
         else:
              print("Warning: Feature names inconsistent after preprocessing. Using generic names.")
              current_feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]


    # Convert to numpy if it's a DataFrame
    X_np = X_processed.values if isinstance(X_processed, pd.DataFrame) else X_processed
    y_np = y_processed # Already an array from preprocess_data

    # Filter high-dimensional data to non-embeddings *before* fitting Lasso
    non_emb_indices = []
    if X_np.shape[1] > 200: # Threshold for focusing on non-embeddings
        print("High dimensionality detected. Calculating importance only for non-embedding features.")
        non_emb_indices = [i for i, name in enumerate(current_feature_names) if 'emb_' not in str(name)]
        if non_emb_indices:
            X_np = X_np[:, non_emb_indices]
            current_feature_names = [current_feature_names[i] for i in non_emb_indices] # Update names list
            print(f"Reduced to {len(current_feature_names)} non-embedding features.")
        else:
            print("No non-embedding features found. Skipping importance.")
            return pd.DataFrame()

    if X_np.shape[1] == 0:
        print("No features left. Skipping importance.")
        return pd.DataFrame()

    # Fit Lasso model
    importance_model = Lasso(alpha=0.0001, max_iter=5000, random_state=42, tol=1e-3)
    print(f"Fitting Lasso for importance on {X_np.shape[0]} samples, {X_np.shape[1]} features...")
    try:
        importance_model.fit(X_np, y_np)
        importances = np.abs(importance_model.coef_) # Absolute coefficient values

        feature_importance_df = pd.DataFrame({
            'Feature': current_feature_names, # Use names corresponding to X_np
            'Importance': importances
        })
        feature_importance_df = feature_importance_df[feature_importance_df['Importance'] > 1e-9] # Filter near-zero
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

        # Print top N features
        top_n_print = min(30, len(feature_importance_df))
        print(f"\nTop {top_n_print} Feature Importances for {model_name} (Lasso Abs Coef):")
        print(feature_importance_df.head(top_n_print).to_string(index=False))

        # Save feature importance to CSV in results subdirectory
        safe_model_name = model_name.replace(' ', '_').replace('+', '').replace('/', '')
        importance_filename = f"{safe_model_name}_feature_importance.csv"
        output_dir = 'regression_results_no_counts/feature_importance' # Adjusted output dir
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, importance_filename)
        feature_importance_df.to_csv(save_path, index=False)
        print(f"Feature importance saved to {save_path}")

        return feature_importance_df

    except Exception as e:
        print(f"Error calculating/saving feature importance for {model_name}: {e}")
        traceback.print_exc()
        return pd.DataFrame()


# --- Visualization and Formatting Functions ---
def visualize_feature_importance(feature_importance_dict, top_n=20, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    valid_importance_dict = {k: v for k, v in feature_importance_dict.items() if v is not None and not v.empty}
    if not valid_importance_dict:
        print("No valid feature importance data to visualize.")
        return
    n_models = len(valid_importance_dict)
    fig_height = max(6, 5 * n_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, fig_height), squeeze=False)
    model_index = 0
    for model_name, importance_df in valid_importance_dict.items():
        ax = axes[model_index, 0]
        n_plot = min(top_n, len(importance_df))
        if n_plot == 0 or 'Importance' not in importance_df.columns:
             ax.text(0.5, 0.5, f'No important features found\nfor {model_name}', ha='center', va='center')
        else:
             top_features = importance_df.nlargest(n_plot, 'Importance')
             sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax, palette="viridis")
             ax.set_title(f'Top {n_plot} Features for {model_name} (Lasso Abs Coef)')
             ax.set_xlabel('Importance (Absolute Lasso Coefficient)')
             ax.set_ylabel('Feature')
             plt.setp(ax.get_yticklabels(), fontsize=10)
        model_index += 1
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance visualization saved to {save_path}")
        except Exception as e: print(f"Error saving importance plot: {e}")
    else: plt.show()
    plt.close(fig)

def plot_regression_results(results_df, save_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if results_df.empty: print("No results to plot."); return
    viz_dir = os.path.join(save_dir, 'visualizations'); os.makedirs(viz_dir, exist_ok=True)
    sns.set(style='whitegrid', context='talk')
    metrics_to_plot = {'R²': True, 'RMSE': False} # True = higher is better
    for metric, ascending in metrics_to_plot.items():
        if metric not in results_df.columns: continue
        plt.figure(figsize=(12, max(8, 0.6 * len(results_df))));
        sorted_df = results_df.sort_values(metric, ascending=not ascending)
        sns.barplot(x=metric, y='Model Name', data=sorted_df, palette='mako')
        plt.title(f'{metric} Comparison Across Optimized Models'); plt.xlabel(f'{metric} ({"Higher Better" if ascending else "Lower Better"})'); plt.ylabel('Model Configuration'); plt.tight_layout()
        try:
            plt.savefig(os.path.join(viz_dir, f'{metric.lower()}_comparison_optimized.png')); print(f"Saved {metric} comparison plot.")
        except Exception as e: print(f"Error saving {metric} plot: {e}")
        plt.close()
    print(f"Saved model visualization plots to {viz_dir}")

def print_formatted_results(results_list, output_file=None):
    if not results_list: print("No results to format."); return ""
    output_lines = []; headers = ["Model Configuration", "Best Model", "R²", "MSE", "RMSE", "MAE", "Optimal Parameters"]; output_lines.append("| " + " | ".join(headers) + " |"); output_lines.append("|" + "---|"*len(headers))
    results_list.sort(key=lambda x: x.get('R²', -float('inf')), reverse=True)
    for result in results_list:
        model_name = str(result.get('Model Name', 'N/A')); best_model_type = str(result.get('Best Model Type', 'N/A')); r2_val = f"{result.get('R²', float('nan')):.6f}"; mse_val = f"{result.get('MSE', float('nan')):.6f}"; rmse_val = f"{result.get('RMSE', float('nan')):.6f}"; mae_val = f"{result.get('MAE', float('nan')):.6f}"
        params = result.get('Best Params', {}); params_str = ", ".join([f"{k.split('__')[-1]}:{v}" for k, v in params.items()]) if isinstance(params, dict) else str(params); params_str = params_str.replace('|', ';')
        line_values = [model_name, best_model_type, r2_val, mse_val, rmse_val, mae_val, params_str]; output_lines.append("| " + " | ".join(line_values) + " |")
    formatted_text = "\n".join(output_lines); print("\n--- Optimized Regression Results Summary (Markdown Format) ---"); print(formatted_text); print("--- End Summary ---")
    if output_file:
        try:
             os.makedirs(os.path.dirname(output_file), exist_ok=True);
             with open(output_file, 'w') as f: f.write("# Optimized Regression Results\n\n" + formatted_text); print(f"Formatted results saved to {output_file}")
        except Exception as e: print(f"Error saving formatted results to {output_file}: {e}")
    return formatted_text


# ==============================================================================
#                   MAIN REGRESSION ANALYSIS FUNCTION
# ==============================================================================
def main():
    """ Main function to run regression analysis """
    print("+"*30 + " Starting Regression Analysis Script " + "+"*30)

    # --- Configuration ---
    data_path = '/Users/kerenlint/Projects/Afeka/models_weight/all_good_projects_with_modernbert_embeddings_enhanced_with_miniLM12.json'
    base_dir = os.path.dirname(data_path) if os.path.dirname(data_path) else '.'
    results_dir = os.path.join(base_dir, 'regression_results_no_counts') # Specific output dir
    os.makedirs(results_dir, exist_ok=True)

    # --- Data Loading ---
    print("Loading data...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f: data = json.load(f)
        print(f"Loaded {len(data)} records")
    except Exception as e: print(f"Fatal Error loading data: {e}"); traceback.print_exc(); return

    # --- Storage ---
    results_list = []
    importance_dict = {}

    # --- Model Configurations to Run ---
    model_configs = [
        #{'name': 'Features Only', 'type': None, 'fs': False, 'n_feat': None},
        #{'name': 'Features Only (FS)', 'type': None, 'fs': True, 'n_feat': 30}, # Select top 30 features
        #{'name': 'MiniLM Only', 'type': 'miniLM', 'only_emb': True},
        #{'name': 'Features+MiniLM', 'type': 'miniLM', 'fs': False, 'n_feat': None},
        #{'name': 'ModernBERT Only', 'type': 'modernbert', 'only_emb': True},
        #{'name': 'Features+ModernBERT', 'type': 'modernbert', 'fs': False, 'n_feat': None},
        #{'name': 'RoBERTa Only', 'type': 'roberta', 'only_emb': True},
        #{'name': 'Features+RoBERTa', 'type': 'roberta', 'fs': False, 'n_feat': None},
        #{'name': 'RoBERTa Risks Only', 'type': 'roberta_risks', 'only_emb': True}, # Special case
        {'name': 'Features+MiniLM', 'type': 'miniLM', 'fs': False, 'n_feat': None},
        {'name': 'Features+RoBERTa', 'type': 'roberta', 'fs': False, 'n_feat': None},
    ]

    # --- Main Loop ---
    for config in model_configs:
        model_run_name = config['name']
        emb_type = config.get('type')
        only_embeddings = config.get('only_emb', False)
        use_fs = config.get('fs', False)
        n_select = config.get('n_feat')

        print("\n" + "="*50 + f"\nProcessing: {model_run_name}" + "\n" + "="*50)

        try:
            # 1. Extract Features based on config
            full_df, emb_dims = extract_features(data, embedding_type=emb_type)
            if full_df.empty:
                print("Feature extraction failed or resulted in empty data. Skipping model.")
                continue

            # 2. Separate Target (y) and Features (X)
            if 'pledged_to_goal_ratio' not in full_df.columns:
                 print("Target 'pledged_to_goal_ratio' missing after extraction. Skipping.")
                 continue
            y = full_df['pledged_to_goal_ratio']
            X = full_df.drop(columns=['pledged_to_goal_ratio'])

            # Adjust X if 'only_embeddings' is True
            if only_embeddings:
                emb_cols = [col for col in X.columns if 'emb_' in col]
                if not emb_cols:
                    print(f"Warning: 'only_embeddings' is True for {model_run_name}, but no embedding columns found. Skipping.")
                    continue
                X = X[emb_cols]
                print(f"Running with Embeddings Only ({len(emb_cols)} columns).")

            if X.empty:
                 print("Feature set X is empty. Skipping model.")
                 continue

            # 3. Preprocess Data (including optional feature selection)
            X_processed, y_processed, selected_feature_names = preprocess_data(
                X, y, feature_selection=use_fs, n_features_to_select=n_select
            )
            if (isinstance(X_processed, pd.DataFrame) and X_processed.empty) or \
               (isinstance(X_processed, np.ndarray) and X_processed.shape[0] == 0):
                print("Preprocessing resulted in empty data. Skipping model.")
                continue

            # 4. Optimize Parameters (Grid Search)
            best_params, best_pipeline, best_model_type = optimize_ols_parameters(
                X_processed, y_processed, cv=5 # Use cv=5 for faster grid search
            )
            if best_pipeline is None:
                print("Parameter optimization failed. Skipping model evaluation.")
                continue

            # 5. Evaluate the Best Model using Cross-Validation
            print(f"\nEvaluating optimized {best_model_type} model for {model_run_name} with CV...")
            cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)
            scoring = {
                'r2': make_scorer(r2_score),
                'neg_mse': make_scorer(mean_squared_error, greater_is_better=False),
                'neg_rmse': make_scorer(lambda y_t, y_p: -np.sqrt(mean_squared_error(y_t, y_p)), greater_is_better=False),
                'neg_mae': make_scorer(mean_absolute_error, greater_is_better=False)
            }
            X_eval = X_processed.values if isinstance(X_processed, pd.DataFrame) else X_processed
            y_eval = y_processed

            cv_results = cross_validate(
                best_pipeline, X_eval, y_eval, cv=cv_strategy,
                scoring=scoring, return_train_score=False, n_jobs=-1, error_score='raise'
            )

            # 6. Store Results
            final_metrics = {
                'Model Name': model_run_name,
                'Best Model Type': best_model_type,
                'R²': cv_results['test_r2'].mean(),
                'MSE': -cv_results['test_neg_mse'].mean(),
                'RMSE': -cv_results['test_neg_rmse'].mean(),
                'MAE': -cv_results['test_neg_mae'].mean(),
                'Best Params': best_params,
                'Feature Count': X_processed.shape[1],
                'Embedding Count': sum(1 for col in X_processed.columns if 'emb_' in col) if isinstance(X_processed, pd.DataFrame) else 'N/A'
            }
            results_list.append(final_metrics)
            print(f"Stored results for {model_run_name}")

            # 7. Get Feature Importance (using the original X before selection)
            if not only_embeddings:
                 imp_df = get_feature_importance(
                     X, y, X.columns.tolist() if isinstance(X, pd.DataFrame) else None,
                     model_run_name
                 )
                 if imp_df is not None and not imp_df.empty:
                     importance_dict[model_run_name] = imp_df

        except Exception as e:
            print(f"\n--- Error processing configuration: {config} ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            traceback.print_exc()
            print("--- Continuing to next configuration ---")


    # --- Final Reporting ---
    print("\n" + "="*50 + "\nFinal Results Processing\n" + "="*50)

    # Print and Save Formatted Results Table
    report_path = os.path.join(results_dir, "optimized_regression_results_summary.md")
    print_formatted_results(results_list, output_file=report_path)

    # Create and Save Summary DataFrame
    if results_list:
        summary_df = pd.DataFrame([{
            'Model Name': r['Model Name'], 'Best Model Type': r['Best Model Type'],
            'Feature Count': r['Feature Count'], 'Embedding Count': r['Embedding Count'],
            'R²': r['R²'], 'MSE': r['MSE'], 'RMSE': r['RMSE'], 'MAE': r['MAE']
        } for r in results_list])
        summary_df = summary_df.sort_values('R²', ascending=False).reset_index(drop=True)
        summary_csv_path = os.path.join(results_dir, "optimized_regression_results_summary.csv")
        try:
            summary_df.to_csv(summary_csv_path, index=False); print(f"Summary results saved to {summary_csv_path}")
        except Exception as e: print(f"Error saving summary CSV: {e}")
        plot_regression_results(summary_df, results_dir)

    # Visualize Feature Importance Comparison
    if importance_dict:
        viz_path = os.path.join(results_dir, 'visualizations', 'feature_importance_comparison.png')
        visualize_feature_importance(importance_dict, top_n=25, save_path=viz_path)
    else: print("No feature importance data collected for visualization.")

    print("\nRegression analysis complete.")


# --- Script Execution ---
if __name__ == "__main__":
    script_start_time = time.time()
    main()
    script_end_time = time.time()
    total_time = script_end_time - script_start_time
    print(f"\nTotal script execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("+"*30 + " Script Finished " + "+"*30)
