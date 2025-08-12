# -*- coding: utf-8 -*-
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")  # Save to files without GUI window
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

# ==============
# Config
# ==============
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15           # From train set
EARLY_STOPPING_ROUNDS = 200
THRESHOLD_METHOD = "f1"   # "f1" or "youden"
RUN_DIR = f"runs/xgb_roberta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(RUN_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def compute_scale_pos_weight(y):
    pos = int(np.sum(y))
    neg = int((y == 0).sum())
    if pos == 0:
        warnings.warn("No positive samples in y_train; scale_pos_weight set to 1.0")
        return 1.0
    return float(neg) / float(pos)

def pick_threshold(y_true, y_prob, method="f1"):
    """
    method:
      - 'f1': maximize F1 on PR curve
      - 'youden': maximize TPR - FPR on ROC curve
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if method == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        idx = int(np.nanargmax(j))
        return float(thr[idx]) if idx < len(thr) else 0.5
    else:
        # default: F1 on PR
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        idx = int(np.nanargmax(f1s))
        return float(thr[idx]) if idx < len(thr) else 0.5

def plot_and_save_roc_pr(y_true, y_prob, run_dir):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "roc_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # PR
    ap = average_precision_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "pr_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

def safe_feature_importance_barh(features, importances, run_dir, topn=50, title="Top Feature Importances", filename="feature_importance.png"):
    s = pd.Series(importances, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 12))
    s.head(topn).plot(kind="barh")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()

def create_shap_feature_importance_plot(model, X_bg, X_test, feature_names, run_dir, topn=30, filename="shap_feature_importance.png"):
    """
    Create feature importance plot based on SHAP values showing positive and negative contributions
    """
    try:
        # Get SHAP values
        explainer = shap.Explainer(model, X_bg, feature_names=feature_names)
        shap_values = explainer(X_test)
        
        # Calculate mean absolute SHAP values for feature importance
        mean_shap_values = np.abs(shap_values.values).mean(axis=0)
        
        # Create DataFrame with feature names and their mean SHAP values
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_shap_values
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Get top features
        top_features = feature_importance_df.head(topn)
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(top_features)), top_features['mean_abs_shap'])
        
        # Color bars based on positive/negative contribution
        colors = []
        for feature in top_features['feature']:
            # Calculate mean SHAP value (can be positive or negative)
            feature_idx = feature_names.index(feature)
            mean_shap = shap_values.values[:, feature_idx].mean()
            if mean_shap > 0:
                colors.append('green')  # Positive contribution (helps success)
            else:
                colors.append('red')    # Negative contribution (hurts success)
        
        # Apply colors
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        # Customize the plot
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'SHAP Feature Importance (Top {topn})\nGreen: Helps Success, Red: Hurts Success')
        plt.gca().invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Helps Success'),
            Patch(facecolor='red', alpha=0.7, label='Hurts Success')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()
        
        # Save detailed SHAP analysis
        detailed_shap = {
            'feature': top_features['feature'].tolist(),
            'mean_abs_shap': top_features['mean_abs_shap'].tolist(),
            'mean_shap': [shap_values.values[:, feature_names.index(f)].mean() for f in top_features['feature']],
            'std_shap': [shap_values.values[:, feature_names.index(f)].std() for f in top_features['feature']]
        }
        save_json(detailed_shap, os.path.join(run_dir, filename.replace('.png', '_details.json')))
        
        print(f"✅ SHAP feature importance saved: {filename}")
        
    except Exception as e:
        warnings.warn(f"Could not create SHAP feature importance plot: {e}")
        # Fallback to regular feature importance
        safe_feature_importance_barh(feature_names, model.feature_importances_, run_dir, topn, 
                                   "Feature Importance (Fallback)", filename)

def explain_shap(model, X_bg, X_test, feature_names, run_dir, sample_force=10, max_display=30):
    """
    Uses modern SHAP API when possible, with fallback to TreeExplainer if needed.
    Saves beeswarm/waterfall as PNG and force plot as HTML.
    """
    try:
        explainer = shap.Explainer(model, X_bg, feature_names=feature_names)
        sv = explainer(X_test)
        # beeswarm
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(sv, show=False, max_display=max_display)
        plt.title("SHAP Beeswarm")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "shap_beeswarm_pure.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # waterfall for example
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(sv[0], show=False, max_display=max_display)
        plt.title("SHAP Waterfall (Sample 0)")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "shap_waterfall_pure.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # force plot to HTML
        try:
            fp = shap.force_plot(explainer.expected_value, sv[:sample_force].values, X_test.iloc[:sample_force])
            shap.save_html(os.path.join(run_dir, "shap_force_pure.html"), fp)
        except Exception as e:
            warnings.warn(f"Could not save SHAP force HTML: {e}")

        # Save SHAP values as medium-sized binary file
        try:
            np.save(os.path.join(run_dir, "shap_values_pure.npy"), sv.values)
            save_json({"expected_value": float(np.mean(explainer.expected_value))}, os.path.join(run_dir, "shap_meta_pure.json"))
        except Exception as e:
            warnings.warn(f"Could not save SHAP arrays: {e}")

    except Exception as e:
        warnings.warn(f"Falling back to TreeExplainer due to: {e}")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)

        # summary plot (legacy)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv, X_test, feature_names=feature_names, show=False, max_display=max_display)
        plt.title("SHAP Summary (fallback)")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "shap_summary_fallback_pure.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # waterfall (legacy style)
        try:
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(explainer.expected_value, sv[0], X_test.iloc[0], feature_names=feature_names, show=False, max_display=max_display)
            plt.title("SHAP Waterfall (fallback, Sample 0)")
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "shap_waterfall_fallback_pure.png"), dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as ee:
            warnings.warn(f"Fallback waterfall failed: {ee}")

# ==============
# 1) Load Data
# ==============
json_path = "/Users/kerenlint/Projects/cursor/projects_with_short_stories_and_risks_with_embeddings.json"
if not os.path.exists(json_path):
    raise FileNotFoundError(f"JSON file not found: {json_path}")
df = pd.read_json(json_path)

# ==============
# 2) Target
# ==============
df["success"] = df["state"].map({"successful": 1, "failed": 0})
df = df.dropna(subset=["success"])

# ==============
# 2.5) Unpack RoBERTa Embeddings
# ==============
assert df["story_roberta_embedding"].apply(lambda x: isinstance(x, list)).all(), "❌ Some story_roberta_embedding values are not lists"
assert df["risk_roberta_embedding"].apply(lambda x: isinstance(x, list)).all(),   "❌ Some risk_roberta_embedding values are not lists"

story_emb_df = pd.DataFrame(df["story_roberta_embedding"].tolist(), index=df.index)
story_emb_df.columns = [f"story_roberta_embedding_{i}" for i in range(story_emb_df.shape[1])]
risks_emb_df = pd.DataFrame(df["risk_roberta_embedding"].tolist(), index=df.index)
risks_emb_df.columns = [f"risk_roberta_embedding_{i}" for i in range(risks_emb_df.shape[1])]

df = df.drop(columns=["story_roberta_embedding", "risk_roberta_embedding"])
df = pd.concat([df, story_emb_df, risks_emb_df], axis=1)

# ==============
# 3) Feature extraction from nested
# ==============
def safe_get(d, path, default=np.nan):
    try:
        for key in path:
            d = d.get(key, {})
        return d if isinstance(d, (int, float)) and d is not None else default
    except Exception:
        return default

readability_keys = {
    "Flesch Reading Ease": "Flesch_Reading_Ease",
    "Flesch-Kincaid Grade Level": "Flesch_Kincaid_Grade",
    "Gunning Fog Index": "Gunning_Fog",
    "SMOG Index": "SMOG",
    "Automated Readability Index": "ARI"
}

df["Story_Avg_Sentence_Length"] = df["Story Analysis"].apply(lambda x: safe_get(x, ["Average Sentence Length"]))
df["Risks_Avg_Sentence_Length"] = df["Risks and Challenges Analysis"].apply(lambda x: safe_get(x, ["Average Sentence Length"]))

for long_key, short_key in readability_keys.items():
    df[f"Story_{short_key}"] = df["Story Analysis"].apply(lambda x: safe_get(x, ["Readability Scores", long_key]))
    df[f"Risks_{short_key}"] = df["Risks and Challenges Analysis"].apply(lambda x: safe_get(x, ["Readability Scores", long_key]))

df["Story_GCI"] = pd.to_numeric(df.get("Story GCI", np.nan), errors="coerce")
df["Risks_GCI"] = pd.to_numeric(df.get("Risks and Challenges GCI", np.nan), errors="coerce")

# ==============
# 4) Preparation days
# ==============
df["created_at"] = pd.to_datetime(df["created_at"], unit="ms", errors="coerce")
df["launched_at"] = pd.to_datetime(df["launched_at"], unit="ms", errors="coerce")
df["preparation_days"] = (df["launched_at"] - df["created_at"]).dt.days

# ==============
# 5) Combined category
# ==============
df["category_Web_Combined"] = (
    df.get("category_Web", False).fillna(False).astype(bool) |
    df.get("category_Web Development", False).fillna(False).astype(bool)
).astype(int)

# ==============
# 6) Feature list
# ==============
base_features = [
    "projectFAQsCount", "rewardscount", "project_length_days", "preparation_days",
    "Story_Avg_Sentence_Length", "Story_Flesch_Reading_Ease", "Story_Flesch_Kincaid_Grade",
    "Story_Gunning_Fog", "Story_SMOG", "Story_ARI",
    "Risks_Avg_Sentence_Length", "Risks_Flesch_Reading_Ease", "Risks_Flesch_Kincaid_Grade",
    "Risks_Gunning_Fog", "Risks_SMOG", "Risks_ARI",
    "Story_Positive", "Story_Neutral", "Story_Negative", "Story_Compound",
    "Risks_Positive", "Risks_Neutral", "Risks_Negative", "Risks_Compound",
    "Story_GCI", "Risks_GCI",
    "category_Web_Combined"
]

category_features = [c for c in df.columns if c.startswith("category_") and c != "category_Technology"]
story_roberta_features = [c for c in df.columns if c.startswith("story_roberta_embedding")]
risks_roberta_features = [c for c in df.columns if c.startswith("risk_roberta_embedding")]

all_features = list(dict.fromkeys(base_features + category_features + story_roberta_features + risks_roberta_features))

missing = [f for f in all_features if f not in df.columns]
if missing:
    print("⚠️ Missing features:", missing)
    raise KeyError("🛑 Missing features detected – cannot train the model.")

# ==============
# 7) Prepare X/y
# ==============
X = df[all_features].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["success"].astype(int)

bad_columns = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
if bad_columns:
    raise ValueError(f"🛑 Non-numeric columns detected – {bad_columns}")

# ==============
# 8) Split
# ==============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# Internal validation for ES
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=RANDOM_STATE
)

# ==============
# 9) Train XGBoost (with your found params + ES + imbalance)
# ==============
best_found = dict(
    colsample_bytree=0.9349553422213137,
    gamma=0.8833152773808622,
    learning_rate=0.020109909209436554,
    max_depth=9,
    subsample=0.6249251763376286
)

scale_pos_weight = compute_scale_pos_weight(y_tr)

xgb_params = dict(
    **best_found,
    # Increase n_estimators so ES can choose the actual number
    n_estimators=5000,
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    enable_categorical=False,
    tree_method="hist",
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)

model = xgb.XGBClassifier(**xgb_params)

model.fit(
    X_tr.values, y_tr.values,
    eval_set=[(X_tr.values, y_tr.values), (X_val.values, y_val.values)],
    verbose=False
)

best_iteration = getattr(model, "best_iteration", None)
best_ntree_limit = getattr(model, "best_ntree_limit", None)

# ==============
# 10) Evaluate + threshold tuning
# ==============
y_val_prob = model.predict_proba(X_val.values)[:, 1]
best_threshold = pick_threshold(y_val, y_val_prob, method=THRESHOLD_METHOD)

y_test_prob = model.predict_proba(X_test.values)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_test_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_test_prob)),
    "avg_precision": float(average_precision_score(y_test, y_test_prob)),
    "threshold": float(best_threshold),
    "best_iteration": int(best_iteration) if best_iteration is not None else None,
    "best_ntree_limit": int(best_ntree_limit) if best_ntree_limit is not None else None,
    "scale_pos_weight": float(scale_pos_weight),
    "class_balance": {
        "train_pos": int(np.sum(y_train)), "train_neg": int((y_train==0).sum()),
        "val_pos": int(np.sum(y_val)),     "val_neg": int((y_val==0).sum()),
        "test_pos": int(np.sum(y_test)),   "test_neg": int((y_test==0).sum()),
    },
    "report": classification_report(y_test, y_test_pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist()
}

print(f"🎯 Accuracy: {metrics['accuracy']:.4f}")
print(f"📊 AUC: {metrics['roc_auc']:.4f}")
print(f"📈 AP (AUC-PR): {metrics['avg_precision']:.4f}")
print(f"⚖️ Threshold used: {metrics['threshold']:.4f}")
print("📋 Classification Report:")
print(classification_report(y_test, y_test_pred))

save_json(metrics, os.path.join(RUN_DIR, "xgboost_eval_metrics.json"))

# ROC/PR curves
plot_and_save_roc_pr(y_test, y_test_prob, RUN_DIR)

# ==============
# 11) Save model + features
# ==============
joblib.dump(model, os.path.join(RUN_DIR, "xgboost_kickstarter_success_model.pkl"))
save_json(all_features, os.path.join(RUN_DIR, "xgboost_feature_columns.json"))

# ==============
# 12) Feature importance
# ==============
safe_feature_importance_barh(
    all_features,
    model.feature_importances_,
    RUN_DIR,
    topn=50,
    title="Top 50 Feature Importances (XGBoost)",
    filename="feature_importance_full.png"
)

# ==============
# 13) SHAP for Pure Features (without embeddings)
# ==============
print("🔍 Generating SHAP analysis for pure features only...")

# Create pure features list (without embeddings)
pure_features = base_features + category_features
print(f"📊 Pure features count: {len(pure_features)}")

# Prepare pure features data
X_pure_train = X_train[pure_features].copy()
X_pure_test = X_test[pure_features].copy()
X_pure_tr = X_tr[pure_features].copy()
X_pure_val = X_val[pure_features].copy()

# Train a new model with only pure features
print("🏋️ Training new model with pure features only...")
pure_model = xgb.XGBClassifier(**xgb_params)
pure_model.fit(
    X_pure_tr.values, y_tr.values,
    eval_set=[(X_pure_tr.values, y_tr.values), (X_pure_val.values, y_val.values)],
    verbose=False
)

# Evaluate pure features model
y_pure_val_prob = pure_model.predict_proba(X_pure_val.values)[:, 1]
pure_best_threshold = pick_threshold(y_val, y_pure_val_prob, method=THRESHOLD_METHOD)

y_pure_test_prob = pure_model.predict_proba(X_pure_test.values)[:, 1]
y_pure_test_pred = (y_pure_test_prob >= pure_best_threshold).astype(int)

pure_metrics = {
    "accuracy": float(accuracy_score(y_test, y_pure_test_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_pure_test_prob)),
    "avg_precision": float(average_precision_score(y_test, y_pure_test_prob)),
    "threshold": float(pure_best_threshold),
    "scale_pos_weight": float(scale_pos_weight),
    "pure_features_count": len(pure_features),
    "report": classification_report(y_test, y_pure_test_pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, y_pure_test_pred).tolist()
}

print(f"🎯 Pure Features - Accuracy: {pure_metrics['accuracy']:.4f}")
print(f"📊 Pure Features - AUC: {pure_metrics['roc_auc']:.4f}")
print(f"📈 Pure Features - AP (AUC-PR): {pure_metrics['avg_precision']:.4f}")
print(f"⚖️ Pure Features - Threshold used: {pure_metrics['threshold']:.4f}")

# Save pure features model and metrics
joblib.dump(pure_model, os.path.join(RUN_DIR, "xgboost_pure_features_model.pkl"))
save_json(pure_features, os.path.join(RUN_DIR, "pure_feature_columns.json"))
save_json(pure_metrics, os.path.join(RUN_DIR, "pure_features_eval_metrics.json"))

# Create background sample for SHAP analysis
bg_pure = X_pure_tr.sample(min(200, len(X_pure_tr)), random_state=RANDOM_STATE)

# SHAP-based feature importance for pure features (with positive/negative contributions)
create_shap_feature_importance_plot(
    pure_model, 
    bg_pure.values, 
    X_pure_test.values, 
    pure_features, 
    RUN_DIR, 
    topn=30, 
    filename="shap_feature_importance_pure.png"
)

# SHAP analysis for pure features
explain_shap(pure_model, bg_pure.values, X_pure_test.values, pure_features, RUN_DIR, sample_force=10, max_display=20)

print(f"✅ Done. Pure features artifacts saved to: {RUN_DIR}")
