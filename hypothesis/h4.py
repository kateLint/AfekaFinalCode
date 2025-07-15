import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_json("/Users/kerenlint/Projects/Afeka/all_models/all_good_projects_without_embeddings.json")

# Filter valid states
df = df[df["state"].isin(["successful", "failed"])].copy()
df["success"] = (df["state"] == "successful").astype(int)

# All category columns except 'Technology'
category_cols = [
    col for col in df.columns
    if col.startswith("category_")
    and df[col].dtype == bool
    and col != "category_Technology"
]

# Build summary per category
results = []

for cat_col in category_cols:
    sub = df[df[cat_col] == True]
    if len(sub) < 10:
        continue  # skip small categories

    category_name = cat_col.replace("category_", "")
    avg_updates = sub["updateCount"].mean()
    success_rate = sub["success"].mean()
    total_projects = len(sub)

    results.append({
        "category": category_name,
        "avg_updates": avg_updates,
        "success_rate": success_rate,
        "num_projects": total_projects
    })

results_df = pd.DataFrame(results).sort_values("avg_updates", ascending=False)

# === Visualizations ===
plt.figure(figsize=(12, 6))
sns.barplot(x="avg_updates", y="category", data=results_df, palette="Blues_d")
plt.title("Average Updates per Project by Category")
plt.xlabel("Average # of Updates")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x="success_rate", y="category", data=results_df, palette="Greens_d")
plt.title("Success Rate by Category")
plt.xlabel("Success Rate")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# Scatter: Updates vs Success Rate
plt.figure(figsize=(8, 6))
sns.scatterplot(x="avg_updates", y="success_rate", hue="category", data=results_df, s=100)
plt.title("Avg. Updates vs Success Rate")
plt.xlabel("Average Updates")
plt.ylabel("Success Rate")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Correlation between update frequency and success rate
correlation = results_df["avg_updates"].corr(results_df["success_rate"])
print(f"\n🔍 Correlation between average updates and success rate: {correlation:.3f}")
