#!/usr/bin/env python3
"""
Kickstarter: Logistic Regression + (Welch) t-tests by feature categories
------------------------------------------------------------------------
What this script does
- Loads a JSON/JSONL/CSV of Kickstarter projects with the features you listed
- Harmonizes/Flattens nested fields to match your expected feature names
- Builds tidy DataFrame with your features + target_class in {"successful","failed"}
- Defines feature categories: Project Engagement, Readability, Sentiment
- Per-feature two-sample Welch t-test between successful vs failed (+ BH-FDR)
- Fits a Statsmodels Logistic Regression (Logit) to obtain Wald z-tests, odds ratios + 95% CI
- Computes VIF (saved) but by default DOES NOT drop features for high VIF (so all features are used)
- Optional: predictive sanity-check with Scikit-Learn LogisticRegression via stratified CV
- Saves clean tables to CSV under ./outputs/

Usage (examples)
---------------
python logistic_regression_t_test.py --input /path/to/your.json --format json
python logistic_regression_t_test.py --input /path/to/your.csv --format csv

Notes
-----
- If your readability metrics are nested (e.g., inside "Story Analysis"), this script flattens them.
- The script is robust to missing columns and will compute `preparation_days` and `project_length_days` if possible.

Author: GP (for Keren)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Statsmodels for inferential logistic regression (p-values, CIs)
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Scikit-learn for preprocessing and predictive sanity check
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================
# Configuration (defaults)
# =============================
DATA_PATH = "/Users/kerenlint/Projects/cursor/projects_with_short_stories_and_risks_with_embeddings.json"
DATA_FORMAT = "json"  # json | jsonl | csv
OUTPUT_DIR = "outputs"

# -----------------------------
# Feature groups configuration
# -----------------------------
PROJECT_ENGAGEMENT = [
    "projectFAQsCount", "commentsCount", "updateCount", "rewardscount",
    "project_length_days", "preparation_days",
]

READABILITY_INDICATORS = [
    "Story_Avg_Sentence_Length", "Story_Flesch_Reading_Ease", "Story_Flesch_Kincaid_Grade",
    "Story_Gunning_Fog", "Story_SMOG", "Story_ARI",
    "Risks_Avg_Sentence_Length", "Risks_Flesch_Reading_Ease", "Risks_Flesch_Kincaid_Grade",
    "Risks_Gunning_Fog", "Risks_SMOG", "Risks_ARI",
]

SENTIMENT_CHARACTERISTICS = [
    "Story_Positive", "Story_Neutral", "Story_Negative", "Story_Compound",
    "Risks_Positive", "Risks_Neutral", "Risks_Negative", "Risks_Compound",
    "Story_GCI", "Risks_GCI",
]

ALL_FEATURES = PROJECT_ENGAGEMENT + READABILITY_INDICATORS + SENTIMENT_CHARACTERISTICS
TARGET_COL = "target_class"  # {'successful','failed'}
TARGET_ALIASES = ["state", "status"]


@dataclass
class Paths:
    input_path: str
    input_format: str  # 'json'|'jsonl'|'csv'
    out_dir: str = "outputs"


# =============================
# Utilities & IO
# =============================

def _ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataframe(paths: Paths) -> pd.DataFrame:
    """Load dataset into a tidy DataFrame with the expected schema.

    Supports:
    - JSON: list[dict] or dict with 'data' key -> list[dict]
    - JSONL: one json object per line
    - CSV: comma-separated
    """
    if not os.path.exists(paths.input_path):
        raise FileNotFoundError(f"Input file not found: {paths.input_path}")

    if paths.input_format == "json":
        with open(paths.input_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            records = obj["data"]
        elif isinstance(obj, list):
            records = obj
        else:
            raise ValueError("JSON must be a list[dict] or a dict with a 'data' list")
        df = pd.DataFrame.from_records(records)

    elif paths.input_format == "jsonl":
        rows = []
        with open(paths.input_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                rows.append(json.loads(ln))
        df = pd.DataFrame.from_records(rows)

    elif paths.input_format == "csv":
        df = pd.read_csv(paths.input_path)

    else:
        raise ValueError("--format must be one of: json, jsonl, csv")

    return df


# =============================
# Harmonization (CRITICAL)
# =============================

def _to_dt_epoch_any(x) -> pd.Timestamp | pd.NaT:
    """Epoch->UTC datetime. Supports seconds (10-digit) and milliseconds (13-digit)."""
    if pd.isna(x):
        return pd.NaT
    try:
        val = float(x)
        if val > 1e12:  # milliseconds
            val /= 1000.0
        return pd.to_datetime(val, unit="s", utc=True)
    except Exception:
        return pd.NaT


def harmonize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten nested readability metrics and align names to ALL_FEATURES.
    Also compute preparation_days and (if needed) project_length_days.
    """
    df = df.copy()
    import numpy as np

    def get_nested(d, *keys):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return np.nan
            cur = cur[k]
        return cur

    # --- Story Readability ---
    if "Story Analysis" in df.columns:
        sa = df["Story Analysis"]
        df["Story_Flesch_Reading_Ease"]   = sa.apply(lambda v: get_nested(v, "Readability Scores", "Flesch Reading Ease"))
        df["Story_Flesch_Kincaid_Grade"]  = sa.apply(lambda v: get_nested(v, "Readability Scores", "Flesch-Kincaid Grade Level"))
        df["Story_Gunning_Fog"]           = sa.apply(lambda v: get_nested(v, "Readability Scores", "Gunning Fog Index"))
        df["Story_SMOG"]                  = sa.apply(lambda v: get_nested(v, "Readability Scores", "SMOG Index"))
        df["Story_ARI"]                   = sa.apply(lambda v: get_nested(v, "Readability Scores", "Automated Readability Index"))
        # Some pipelines store ASL at top level of "Story Analysis"
        asl = sa.apply(lambda v: v.get("Average Sentence Length") if isinstance(v, dict) else np.nan)
        df["Story_Avg_Sentence_Length"] = asl

    # --- Risks Readability ---
    rc_col = None
    for cand in ["Risks and Challenges Analysis", "Risks And Challenges Analysis", "Risks_And_Challenges_Analysis"]:
        if cand in df.columns:
            rc_col = cand
            break

    if rc_col:
        ra = df[rc_col]
        df["Risks_Flesch_Reading_Ease"]   = ra.apply(lambda v: get_nested(v, "Readability Scores", "Flesch Reading Ease"))
        df["Risks_Flesch_Kincaid_Grade"]  = ra.apply(lambda v: get_nested(v, "Readability Scores", "Flesch-Kincaid Grade Level"))
        df["Risks_Gunning_Fog"]           = ra.apply(lambda v: get_nested(v, "Readability Scores", "Gunning Fog Index"))
        df["Risks_SMOG"]                  = ra.apply(lambda v: get_nested(v, "Readability Scores", "SMOG Index"))
        df["Risks_ARI"]                   = ra.apply(lambda v: get_nested(v, "Readability Scores", "Automated Readability Index"))
        asl = ra.apply(lambda v: v.get("Average Sentence Length") if isinstance(v, dict) else np.nan)
        df["Risks_Avg_Sentence_Length"] = asl

    # --- Align GCI names ---
    if "Story GCI" in df.columns and "Story_GCI" not in df.columns:
        df["Story_GCI"] = df["Story GCI"]

    for cand in ["Risks and Challenges GCI", "Risks And Challenges GCI", "Risks_GCI"]:
        if cand in df.columns:
            df["Risks_GCI"] = df[cand]
            break

    # --- preparation_days ---
    c_at = df["created_at"].apply(_to_dt_epoch_any) if "created_at" in df.columns else pd.Series(pd.NaT, index=df.index)
    l_at = df["launched_at"].apply(_to_dt_epoch_any) if "launched_at" in df.columns else pd.Series(pd.NaT, index=df.index)
    df["preparation_days"] = (l_at - c_at).dt.total_seconds() / 86400.0

    # --- project_length_days backfill (if missing or empty) ---
    if "project_length_days" not in df.columns or df["project_length_days"].isna().all():
        d_at = df["deadline"].apply(_to_dt_epoch_any) if "deadline" in df.columns else pd.Series(pd.NaT, index=df.index)
        df["project_length_days"] = (d_at - l_at).dt.total_seconds() / 86400.0
    # --- Sentiment dicts fallback ---
    # If flat columns are missing but dicts exist, extract them.
    if "Story_Sentiment" in df.columns:
        ss = df["Story_Sentiment"]
        for k_src, k_dst in [("pos", "Story_Positive"), ("neu", "Story_Neutral"),
                             ("neg", "Story_Negative"), ("compound", "Story_Compound")]:
            if k_dst not in df.columns:
                df[k_dst] = ss.apply(lambda v: v.get(k_src) if isinstance(v, dict) else np.nan)

    if "Risks_Sentiment" in df.columns:
        rs = df["Risks_Sentiment"]
        for k_src, k_dst in [("pos", "Risks_Positive"), ("neu", "Risks_Neutral"),
                             ("neg", "Risks_Negative"), ("compound", "Risks_Compound")]:
            if k_dst not in df.columns:
                df[k_dst] = rs.apply(lambda v: v.get(k_src) if isinstance(v, dict) else np.nan)

    # --- Count fallbacks from arrays ---
    if "projectFAQsCount" not in df.columns and "faqs" in df.columns:
        df["projectFAQsCount"] = df["faqs"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)
    if "rewardscount" not in df.columns and "rewards" in df.columns:
        df["rewardscount"] = df["rewards"].apply(lambda x: len(x) if isinstance(x, list) else np.nan)

    return df


# =============================
# Validation / schema
# =============================

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist; create missing ones as NaN; coerce types."""
    df = df.copy()

    # Ensure target exists; allow aliases like 'state' or 'status'
    if TARGET_COL not in df.columns:
        for alias in TARGET_ALIASES:
            if alias in df.columns:
                df[TARGET_COL] = df[alias]
                break
    if TARGET_COL not in df.columns:
        raise KeyError(f"Missing required column '{TARGET_COL}' with values in {{'successful','failed'}}")

    # Add missing feature columns as NaN
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce numeric columns to float
    for col in ALL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize target to binary 1/0
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.lower()
        .map({"successful": 1, "failed": 0})
    )
    if df[TARGET_COL].isna().any():
        bad = df.loc[df[TARGET_COL].isna(), TARGET_COL].shape[0]
        raise ValueError(f"Found {bad} rows with invalid target values. Expected 'successful' or 'failed'.")

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def audit_feature_coverage(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Report missingness and non-null share per feature; save to CSV."""
    rows = []
    for c in ALL_FEATURES:
        nn = df[c].notna().sum()
        rows.append({
            "feature": c,
            "non_null": nn,
            "non_null_rate": nn / len(df) if len(df) else 0.0
        })
    cov = pd.DataFrame(rows).sort_values("non_null_rate", ascending=False)
    cov.to_csv(os.path.join(out_dir, "feature_coverage.csv"), index=False)
    return cov


# =============================
# Inference utilities
# =============================

def build_design_matrix(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[features].copy()
    y = df[TARGET_COL].copy()
    return X, y


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factors for numeric design matrix (after imputation/scaling)."""
    vif_list = []
    for i in range(X.shape[1]):
        try:
            vif_val = variance_inflation_factor(X.values, i)
        except Exception:
            vif_val = np.nan
        vif_list.append({"feature": X.columns[i], "vif": vif_val})
    return pd.DataFrame(vif_list).sort_values("vif", ascending=False)


def logistic_regression_inference(
    df: pd.DataFrame,
    features: List[str],
    drop_high_vif: bool = False,   # IMPORTANT: default False to keep all features
    vif_threshold: float = 10.0
) -> Dict[str, Any]:
    """Fit statsmodels Logit with imputation+scaling; return coef table and (optional) VIF."""
    # Preprocess: impute median, scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_raw, y = build_design_matrix(df, features)

    # Drop features that are entirely missing to avoid shape mismatch after imputation
    non_empty_cols = [c for c in X_raw.columns if X_raw[c].notna().any()]
    dropped_empty = [c for c in X_raw.columns if c not in non_empty_cols]
    if dropped_empty:
        print(f"[warn] Dropping {len(dropped_empty)} all-missing features (populate upstream!): {dropped_empty}")
    X_raw = X_raw[non_empty_cols]

    # Impute + scale
    X_imp = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns, index=X_raw.index)

    # Drop constant columns (zero variance after imputation)
    variances = X_imp.var(axis=0)
    non_constant = variances[variances > 0].index.tolist()
    dropped_const = [c for c in X_imp.columns if c not in non_constant]
    if dropped_const:
        print(f"[warn] Dropping {len(dropped_const)} constant features (zero variance after imputation): {dropped_const}")
    X_imp = X_imp[non_constant]
    X_s = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index)

    # Class sanity
    if y.nunique() < 2:
        raise ValueError("Logit requires both classes present in target. Found only one class.")



    # VIF (for reporting)
    vif_df = compute_vif(X_s)

    # Keep all features unless explicitly asked to filter by VIF
    X_use = X_s.copy()
    if drop_high_vif and not vif_df["vif"].isna().all():
        keep = vif_df.loc[(vif_df["vif"].isna()) | (vif_df["vif"] <= vif_threshold), "feature"].tolist()
        dropped = [c for c in X_s.columns if c not in keep]
        if dropped:
            print(f"[info] Dropping {len(dropped)} high-VIF features (> {vif_threshold}): {dropped}")
        X_use = X_s[keep]
    else:
        print("[info] Keeping all available (non-empty) features; not filtering by VIF.")

    # Statsmodels Logit
    X_sm = sm.add_constant(X_use, has_constant="add")
    model = sm.Logit(y, X_sm)
    try:
        fit_res = model.fit(disp=False, maxiter=200)
    except Exception:
        # Try different optimizer if needed
        fit_res = model.fit(method="bfgs", disp=False, maxiter=400)

    coefs = fit_res.params
    conf = fit_res.conf_int(alpha=0.05)
    conf.columns = ["ci_lower", "ci_upper"]
    se = fit_res.bse
    zvals = coefs / se
    pvals = fit_res.pvalues

    # Odds ratios based on standardized features: interpret as per-1-SD change
    or_df = pd.DataFrame({
        "coef": coefs,
        "std_err": se,
        "z_value": zvals,
        "p_value": pvals,
        "odds_ratio": np.exp(coefs),
        "or_ci_lower": np.exp(conf["ci_lower"]),
        "or_ci_upper": np.exp(conf["ci_upper"]),
    })

    or_df.index.name = "term"
    or_df.reset_index(inplace=True)

    # FDR across non-constant terms
    mask = or_df["term"] != "const"
    if mask.any():
        _, qvals, _, _ = multipletests(or_df.loc[mask, "p_value"].values, method="fdr_bh")
        or_df.loc[mask, "q_value_fdr_bh"] = qvals
    else:
        or_df["q_value_fdr_bh"] = np.nan

    return {
        "coef_table": or_df.sort_values(["q_value_fdr_bh", "p_value"]).reset_index(drop=True),
        "vif_table": vif_df,
        "fitted": fit_res,
        "X_columns_used": pd.DataFrame({"feature": X_use.columns}),
    }


# =============================
# Predictive sanity check (optional)
# =============================

def predictive_cv_report(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X, y = build_design_matrix(df, features)

    # Keep only features with at least one observed value (to satisfy imputer)
    non_empty_cols = [c for c in X.columns if X[c].notna().any()]
    dropped_empty = [c for c in X.columns if c not in non_empty_cols]
    if dropped_empty:
        print(f"[warn] [cv] Dropping {len(dropped_empty)} all-missing features: {dropped_empty}")
    X = X[non_empty_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), non_empty_cols)
    ])

    clf = Pipeline([
        ("pre", pre),
        ("logreg", LogisticRegression(max_iter=200, solver="lbfgs", class_weight="balanced"))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=skf)

    # Structured report
    rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose().reset_index().rename(columns={"index": "metric"})

    y_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, y_proba)
    rep_df.loc[len(rep_df)] = {"metric": "roc_auc", "precision": np.nan, "recall": np.nan, "f1-score": auc, "support": y.shape[0]}


    return rep_df



def welch_t_tests_by_feature(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Per-feature two-sample Welch's t-test between successful vs failed.
    Returns tidy table with BH-FDR across the provided `features` only.
    Columns: feature, n_success, n_failed, t_stat, p_value, mean_success, mean_failed, std_mean_diff, q_value_fdr_bh
    - std_mean_diff: Cohen's d using Welch's pooled SD (unbiased-ish; not Hedges' exact correction).
    """
    rows = []
    # split once for speed
    mask_s = df[TARGET_COL] == 1
    mask_f = df[TARGET_COL] == 0

    for feat in features:
        if feat not in df.columns:
            rows.append({
                "feature": feat, "n_success": 0, "n_failed": 0,
                "t_stat": np.nan, "p_value": np.nan,
                "mean_success": np.nan, "mean_failed": np.nan,
                "std_mean_diff": np.nan
            })
            continue

        a = pd.to_numeric(df.loc[mask_s, feat], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[mask_f, feat], errors="coerce").dropna()

        n_s, n_f = len(a), len(b)
        if n_s < 2 or n_f < 2:
            rows.append({
                "feature": feat, "n_success": n_s, "n_failed": n_f,
                "t_stat": np.nan, "p_value": np.nan,
                "mean_success": a.mean() if n_s else np.nan,
                "mean_failed": b.mean() if n_f else np.nan,
                "std_mean_diff": np.nan
            })
            continue

        # Welch t-test
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")

        # Standardized mean diff (Cohen’s d with Welch pooled SD)
        s1, s2 = a.std(ddof=1), b.std(ddof=1)
        pooled = np.sqrt(((s1**2) + (s2**2)) / 2.0) if np.isfinite(s1) and np.isfinite(s2) else np.nan
        d = (a.mean() - b.mean()) / pooled if (pooled is not None and pooled > 0 and np.isfinite(pooled)) else np.nan

        rows.append({
            "feature": feat,
            "n_success": n_s,
            "n_failed": n_f,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
            "mean_success": a.mean(),
            "mean_failed": b.mean(),
            "std_mean_diff": d
        })

    out = pd.DataFrame(rows)

    # BH-FDR across this feature set
    if out["p_value"].notna().any():
        mask = out["p_value"].notna()
        _, q, _, _ = multipletests(out.loc[mask, "p_value"].values, method="fdr_bh")
        out.loc[mask, "q_value_fdr_bh"] = q
    else:
        out["q_value_fdr_bh"] = np.nan

    # nicer ordering: by q, then p
    out = out.sort_values(["q_value_fdr_bh", "p_value"], na_position="last").reset_index(drop=True)
    return out


# =============================
# Category analysis wrapper
# =============================

def run_category_analysis(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_out_dir(out_dir)

    # Coverage audit BEFORE inference
    coverage = audit_feature_coverage(df, out_dir)
    print("[info] Feature coverage (top 10 by non-null rate):")
    print(coverage.head(10).to_string(index=False))

    categories = [
        ("project_engagement", PROJECT_ENGAGEMENT),
        ("readability_indicators", READABILITY_INDICATORS),
        ("sentiment_characteristics", SENTIMENT_CHARACTERISTICS),
        ("all_features", ALL_FEATURES),
    ]

    # 1) t-tests per category
    for name, feats in categories:
        ttab = welch_t_tests_by_feature(df, feats)
        ttab.to_csv(os.path.join(out_dir, f"ttests_{name}.csv"), index=False)

    # 2) Logistic regression (full set, no VIF dropping)
    logit = logistic_regression_inference(df, ALL_FEATURES, drop_high_vif=False)
    logit["coef_table"].to_csv(os.path.join(out_dir, "logit_coefs.csv"), index=False)
    logit["vif_table"].to_csv(os.path.join(out_dir, "vif_table.csv"), index=False)
    logit["X_columns_used"].to_csv(os.path.join(out_dir, "columns_used.csv"), index=False)

    # 3) Predictive CV sanity check
    cvrep = predictive_cv_report(df, ALL_FEATURES)
    cvrep.to_csv(os.path.join(out_dir, "predictive_cv_report.csv"), index=False)

    # 4) Save a short README of outputs
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            """Files written by logistic_regression_t_test.py

- feature_coverage.csv: Non-null coverage per feature (check before modeling)
- ttests_project_engagement.csv: Welch t-test per feature in Project Engagement
- ttests_readability_indicators.csv: Welch t-test per feature in Readability
- ttests_sentiment_characteristics.csv: Welch t-test per feature in Sentiment
- ttests_all_features.csv: Welch t-test for all features combined (per-feature)
- logit_coefs.csv: Statsmodels Logit results with coef, p, q(FDR), odds ratios, 95% CI
- vif_table.csv: Variance Inflation Factors (computed after scaling)
- columns_used.csv: Features included in the final Logit design matrix (non-empty after harmonization)
- predictive_cv_report.csv: 5-fold CV metrics (accuracy, precision/recall/F1)
"""
        )


# =============================
# CLI
# =============================

def parse_args(argv: List[str]) -> Paths:
    p = argparse.ArgumentParser(description="Kickstarter logistic regression + t-tests pipeline")
    p.add_argument("--input", required=True, help="Path to input file (json/jsonl/csv)")
    p.add_argument("--format", required=True, choices=["json", "jsonl", "csv"], help="Input format")
    p.add_argument("--out", default="outputs", help="Output directory (default: ./outputs)")
    args = p.parse_args(argv)
    return Paths(input_path=args.input, input_format=args.format, out_dir=args.out)


def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    paths = parse_args(argv)

    try:
        df = load_dataframe(paths)
        # CRITICAL: harmonize BEFORE schema enforcement
        df = harmonize_features(df)
        df = enforce_schema(df)
    except Exception as e:
        print(f"[error] Failed to load/validate data: {e}")
        return 2

    try:
        run_category_analysis(df, paths.out_dir)
    except Exception as e:
        print(f"[error] Analysis failed: {e}")
        return 3

    print(f"[ok] Analysis complete. See outputs in: {os.path.abspath(paths.out_dir)}")
    return 0


if __name__ == "__main__":
    # Default config
    paths = Paths(input_path=DATA_PATH, input_format=DATA_FORMAT, out_dir=OUTPUT_DIR)

    try:
        df = load_dataframe(paths)
        df = harmonize_features(df)
        df = enforce_schema(df)
    except Exception as e:
        print(f"[error] Failed to load/validate data: {e}")
        raise SystemExit(2)

    try:
        run_category_analysis(df, paths.out_dir)
    except Exception as e:
        print(f"[error] Analysis failed: {e}")
        raise SystemExit(3)

    print(f"[ok] Analysis complete. See outputs in: {os.path.abspath(paths.out_dir)}")
