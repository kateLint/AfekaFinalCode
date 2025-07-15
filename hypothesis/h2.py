import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from bertopic import BERTopic
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.9)

# === Suppress tokenizer parallelism warning ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Step 1: Load data ===
with open("/Users/kerenlint/Projects/Afeka/all_models/all_good_projects_without_embeddings.json", "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# === Step 2: Clean ===
df = df.dropna(subset=["story_clean", "state"])
df["success"] = df["state"].map({"successful": 1, "failed": 0})

# === Step 2.1: Filter non-English texts ===
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False  # Skip if detection fails

df = df[df["story_clean"].apply(is_english)]

# === Step 3: Train BERTopic ===
print("Training BERTopic on story texts...")
story_model = BERTopic(language="english", vectorizer_model=vectorizer, calculate_probabilities=False)

story_topics, _ = story_model.fit_transform(df["story_clean"].tolist())

# === Step 4: Reduce topics (optional but recommended) ===
print("Reducing topics to 50...")
story_model.reduce_topics(df["story_clean"].tolist(), nr_topics=50)

# === Step 5: Assign new reduced topic labels ===
reduced_topics = story_model.get_topics().keys()
df["story_topic"] = story_model.get_document_info(df["story_clean"].tolist())["Topic"]
# === Optional Cleanup: Remove obviously non-English or noisy topics ===
topics_to_exclude = [48]  # Add more if needed after inspecting topic keywords
df = df[~df["story_topic"].isin(topics_to_exclude)]
df = df[df["story_topic"].notnull()]
df = df[df["story_topic"] != -1]
df["story_topic_label"] = df["story_topic"].apply(lambda t: f"StoryTopic_{t}")
df["story_topic_list"] = df["story_topic"].apply(lambda t: [f"StoryTopic_{t}"])

# === Step 6: One-hot encode topics ===
mlb = MultiLabelBinarizer()
topic_dummies = pd.DataFrame(mlb.fit_transform(df["story_topic_list"]), columns=mlb.classes_, index=df.index)
df_model = pd.concat([df, topic_dummies], axis=1)

# === Step 7: Prepare training data ===
X = df_model[mlb.classes_]
y = df_model["success"]

# Remove constant or duplicate columns
X = X.loc[:, X.nunique() > 1]
X = X.loc[:, ~X.T.duplicated()]

# === Step 8: Fit Logistic Regression ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
clf = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

# === Step 9: Extract coefficients and interpret ===
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": clf.coef_[0],
    "Odds_Ratio": np.exp(clf.coef_[0])
})

coef_df["Topic_ID"] = coef_df["Feature"].str.extract(r"StoryTopic_(\d+)").astype("Int64")

def get_topic_keywords(topic_id):
    try:
        keywords = story_model.get_topic(topic_id)
        return ", ".join([kw[0] for kw in keywords[:6]]) if keywords else "[No Keywords]"
    except:
        return "[Missing Topic]"

coef_df["Topic_Keywords"] = coef_df["Topic_ID"].apply(get_topic_keywords)

# === Step 10: Sort and display ===
top_pos = coef_df.sort_values("Odds_Ratio", ascending=False).head(10)
top_neg = coef_df.sort_values("Odds_Ratio", ascending=True).head(10)

# Save for inspection
coef_df.to_csv("topic_regression_output_reduced.csv", index=False)

# === Step 11: Print results ===
print("\n=== TOP POSITIVE TOPICS (Boost Success) ===")
print(tabulate(top_pos[["Feature", "Topic_Keywords", "Coefficient", "Odds_Ratio"]], headers="keys", tablefmt="pretty"))

print("\n=== TOP NEGATIVE TOPICS (Hurt Success) ===")
print(tabulate(top_neg[["Feature", "Topic_Keywords", "Coefficient", "Odds_Ratio"]], headers="keys", tablefmt="pretty"))

from sklearn.metrics import classification_report, roc_auc_score

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

# === Step 12: Visualize as bar plots ===
def plot_top_topics(df, title, top_n=10, positive=True):
    df = df.sort_values("Odds_Ratio", ascending=not positive).head(top_n)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df["Topic_Keywords"], df["Odds_Ratio"], color="green" if positive else "red")
    plt.xlabel("Odds Ratio")
    plt.title(title)
    plt.gca().invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center', ha='left', fontsize=8)
    plt.tight_layout()
    plt.show()

plot_top_topics(top_pos, "Top POSITIVE Topics (Boost Success)", positive=True)
plot_top_topics(top_neg, "Top NEGATIVE Topics (Hurt Success)", positive=False)

# === Optional: Interactive topic visualization (requires notebook or browser) ===
# story_model.visualize_topics().show()
# === Step 13: Automatically assign topics to H6 themes and test H6 hypothesis ===
from statsmodels.api import Logit, add_constant
from scipy.stats import chi2_contingency

# --- Define theme keyword sets ---
theme_keywords = {
    "Innovation": {"tech", "robot", "arduino", "raspberry", "pi", "ai", "software", "hardware", "device", "engineering", "iot", "gadget"},
    "Community": {"pet", "dog", "cat", "shelter", "children", "family", "education", "school", "volunteer", "support", "care"},
    "Transparency": {"clean", "filter", "health", "medical", "water", "air", "toothbrush", "purifier", "privacy", "hygiene", "trust"}
}

# --- Extract keywords per topic ---
def get_topic_keywords_map(model, top_n=10):
    topic_keywords = {}
    for topic_id in model.get_topics():
        words = model.get_topic(topic_id)
        if words:
            topic_keywords[topic_id] = {kw[0] for kw in words[:top_n]}
        else:
            topic_keywords[topic_id] = set()
    return topic_keywords

topic_keywords_map = get_topic_keywords_map(story_model, top_n=10)

# --- Match topics to themes based on overlap ---
def match_themes(keywords, theme_dict):
    matched = set()
    for theme, theme_words in theme_dict.items():
        if keywords & theme_words:
            matched.add(theme)
    return list(matched)

topic_theme_map = {
    tid: match_themes(keywords, theme_keywords)
    for tid, keywords in topic_keywords_map.items()
}

# --- Assign to each project based on topic ---
def assign_themes_to_project(topic_id):
    return topic_theme_map.get(topic_id, [])

df["themes"] = df["story_topic"].apply(assign_themes_to_project)

# --- Binary indicators for each theme ---
for theme in theme_keywords:
    df[theme] = df["themes"].apply(lambda x: int(theme in x))

df["HasAnyFraming"] = df[list(theme_keywords.keys())].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
# === Step: Theme appearance rate in successful vs. all projects ===
print("\n=== Theme Appearance Rates by Project Success ===")
for theme in theme_keywords:
    success_rate = df[df[theme] == 1]["success"].mean()
    overall_rate = df[theme].mean()
    print(f"{theme:<12} — Theme Success Rate: {success_rate:.2%} | Theme Presence: {overall_rate:.2%}")

# === Logistic regression test of H6 ===
print("\n=== Logistic Regression Results for H6 ===")
X_h6 = add_constant(df[list(theme_keywords.keys())])
y_h6 = df["success"]
model_h6 = Logit(y_h6, X_h6).fit()
print(model_h6.summary())

# === Chi-square tests ===
print("\n=== Chi-Square Tests for H6 ===")
for col in list(theme_keywords.keys()) + ["HasAnyFraming"]:
    table = pd.crosstab(df[col], df["success"])
    chi2, p, _, _ = chi2_contingency(table)
    print(f"{col:<15} → p-value: {p:.4f}")
