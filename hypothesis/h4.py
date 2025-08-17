import pandas as pd
import numpy as np
import json
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

# === Load your dataset ===
with open('/Users/kerenlint/Projects/Afeka/all_models/all_good_projects_without_embeddings.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# === Convert outcome variable ===
df['success'] = df['state'].apply(lambda x: 1 if x == 'successful' else 0)

# === Topic keyword definitions ===
topic_keywords = {
    'innovation': ["innovation", "innovative", "cutting-edge", "disruptive", "next-gen", "revolutionary", "impact"],
    'community': ["community", "join us", "support", "together", "your help", "collaborate", "our team", "feedback"],
    'transparency': ["transparency", "clear", "detailed", "honest", "risk mitigation", "specific", "backup", "plan", "tested"]
}

# === Keyword detection ===
def detect_keywords(text, keywords):
    if not isinstance(text, str): return 0
    text = text.lower()
    return int(any(re.search(rf'\b{re.escape(word)}\b', text) for word in keywords))

# === Apply detection ===
df['topic_innovation'] = df['story_clean'].apply(lambda x: detect_keywords(x, topic_keywords['innovation']))
df['topic_community'] = df['story_clean'].apply(lambda x: detect_keywords(x, topic_keywords['community']))


df['topic_framing'] = df[['topic_innovation', 'topic_community']].max(axis=1)

# === Logistic Regression ===
model = smf.logit('success ~ topic_innovation + topic_community', data=df).fit()
#smf.logit('success ~ topic_framing', data=df).fit()

# === Print Summary ===
print(model.summary())

# === Odds Ratios ===
print("\n=== Odds Ratios ===")
print(pd.Series(model.params).apply(lambda x: round(np.exp(x), 3)))
print(df['topic_transparency'].value_counts())
print(df.groupby('topic_transparency')['success'].value_counts())
# === Chi-squared test ===
ct = pd.crosstab(df['topic_framing'], df['success'])
chi2, p, _, _ = chi2_contingency(ct)
print(f"\nChi-squared test: χ² = {chi2:.4f}, p-value = {p:.4g}")
