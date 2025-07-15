import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
import warnings

# Optional plotting
try:
    import seaborn as sns
    seaborn_installed = True
except ImportError:
    seaborn_installed = False
    print("Seaborn not installed. Skipping boxplots.")

warnings.filterwarnings("ignore")

# === Load Data ===
file_path = "/Users/kerenlint/Projects/Afeka/all_models/all_good_projects_without_embeddings.json"
df = pd.read_json(file_path)

# === Extract Readability Metrics from 'Risks and Challenges Analysis' ===
readability_keys = [
    "Flesch Reading Ease",
    "Flesch-Kincaid Grade Level",
    "Gunning Fog Index",
    "SMOG Index",
    "Automated Readability Index"
]

for key in readability_keys:
    colname = f"risks_{key.lower().replace(' ', '_').replace('-', '_')}"
    df[colname] = df["Risks and Challenges Analysis"].apply(
        lambda x: x.get("Readability Scores", {}).get(key, np.nan) if isinstance(x, dict) else np.nan
    )

df["risks_lexical_diversity"] = df["Risks and Challenges Analysis"].apply(
    lambda x: x.get("Lexical Diversity", np.nan) if isinstance(x, dict) else np.nan
)
df["risks_avg_sentence_length"] = df["Risks and Challenges Analysis"].apply(
    lambda x: x.get("Average Sentence Length", np.nan) if isinstance(x, dict) else np.nan
)

# === Prepare Labels & Features ===
df = df[df["state"].isin(["successful", "failed"])].copy()
df["success"] = (df["state"] == "successful").astype(int)

# Filter only numeric risks_ features
risk_cols = [col for col in df.columns if col.startswith("risks_") and pd.api.types.is_numeric_dtype(df[col])]
df = df[risk_cols + ["success"]].dropna()

X = df[risk_cols]
y = df["success"]

# === Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Logistic Regression with Cross-Validation ===
model = LogisticRegression(max_iter=1000)
auc_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")
model.fit(X_scaled, y)

# === Coefficients Table ===
coef_table = pd.DataFrame({
    "Feature": risk_cols,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

# === T-test Between Success and Fail ===
ttest_results = []
for col in risk_cols:
    success_vals = df[df["success"] == 1][col]
    fail_vals = df[df["success"] == 0][col]
    stat, pval = ttest_ind(success_vals, fail_vals, equal_var=False)
    ttest_results.append({"Feature": col, "p_value": pval})

ttest_df = pd.DataFrame(ttest_results).sort_values("p_value")

# === Output Results ===
print("\n==== Logistic Regression Coefficients (Risks and Challenges) ====")
print(coef_table.to_string(index=False))

print("\n==== T-Test Results (Risks: Success vs Failed) ====")
print(ttest_df.to_string(index=False))

print(f"\nMean AUC (Risks Readability): {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

# === Optional: Plot Boxplots ===
if seaborn_installed:
    for col in risk_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="success", y=col, data=df)
        plt.title(f"{col} by Project Success")
        plt.xlabel("Success (0 = Failed, 1 = Successful)")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()
