#Projects with higher sentiment polarity in their narratives are more likely to succeed.

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, pearsonr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

# === Load Data ===
data_path = "/Users/kerenlint/Projects/Afeka/data/all_good_projects_without_embeddings.json"
df = pd.read_json(data_path)

# === Preprocess ===
df['state'] = df['state'].str.lower()
df = df[df['state'].isin(['successful', 'failed'])].copy()
df['success_binary'] = df['state'].map({'successful': 1, 'failed': 0})

# === Sentiment Extraction ===
def safe_sentiment(text):
    if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
        return {'pos': 0.0, 'compound': 0.0}
    scores = vader.polarity_scores(text)
    return {'pos': scores['pos'], 'compound': scores['compound']}

def extract_sentiment_column(colname):
    scores = df[colname].apply(safe_sentiment)
    return pd.DataFrame(scores.tolist(), index=df.index).add_prefix(colname + "_")

df = pd.concat([df,
                extract_sentiment_column("story"),
                extract_sentiment_column("risks-and-challenges"),
                df["faqs"].apply(lambda faqs: safe_sentiment(" ".join([f['question'] + " " + f['answer'] for f in faqs]) if isinstance(faqs, list) else "")).apply(pd.Series).add_prefix("faqs_")
               ], axis=1)

# === Hypothesis Testing for Story ===
success_group = df[df['success_binary'] == 1]
fail_group = df[df['success_binary'] == 0]

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1)*np.std(x, ddof=1)**2 + (ny - 1)*np.std(y, ddof=1)**2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std

def run_analysis(var, label):
    print(f"\n===== {label} Sentiment Analysis =====")
    x_success = success_group[var]
    x_fail = fail_group[var]
    t_stat, p_val = ttest_ind(x_success, x_fail, equal_var=False)
    corr, corr_p = pearsonr(df[var], df['success_binary'])
    d = cohen_d(x_success, x_fail)

    print(f"Mean {label} Compound (Success): {x_success.mean():.4f}")
    print(f"Mean {label} Compound (Fail):    {x_fail.mean():.4f}")
    print(f"T-test: t = {t_stat:.4f}, p = {p_val:.4e}")
    print(f"Correlation r = {corr:.4f}, p = {corr_p:.4e}")
    print(f"Cohen's d = {d:.4f}")

    # Visualization
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x="success_binary", y=var)
    plt.xticks([0, 1], ['Failed', 'Successful'])
    plt.title(f"Boxplot of {label} Compound Sentiment by Outcome")
    plt.ylabel(f"{label} Compound Score")
    plt.xlabel("Project Outcome")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    sns.kdeplot(data=df, x=var, hue='success_binary', fill=True)
    plt.title(f"KDE Plot of {label} Compound Sentiment")
    plt.xlabel(f"{label} Compound Score")
    plt.tight_layout()
    plt.show()

    # Logistic Regression
    X = sm.add_constant(df[[var]])
    y = df['success_binary']
    logit_model = sm.Logit(y, X).fit(disp=0)
    print(logit_model.summary())

# === Run analyses ===
run_analysis("story_compound", "Story")
run_analysis("risks-and-challenges_compound", "Risks")
run_analysis("faqs_compound", "FAQs")
