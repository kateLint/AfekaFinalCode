import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import json

# === 1. טעינת הנתונים ===
json_path = "/Users/kerenlint/Projects/Afeka/all_models/all_good_projects_with_modernbert_embeddings_enhanced_with_miniLM12.json"
df = pd.read_json(json_path)

# === 2. יצירת עמודת יעד בינארית (1=successful, 0=failed) ===
df["success"] = df["state"].map({"successful": 1, "failed": 0})
df = df.dropna(subset=["success"])

# === 2.5. פירוק MiniLM Embeddings ===
assert df["story_miniLM"].apply(lambda x: isinstance(x, list)).all(), "❌ יש story_miniLM שלא רשימה"
assert df["risks_miniLM"].apply(lambda x: isinstance(x, list)).all(), "❌ יש risks_miniLM שלא רשימה"

story_emb_df = pd.DataFrame(df["story_miniLM"].tolist(), index=df.index)
story_emb_df.columns = [f"story_miniLM_{i}" for i in range(story_emb_df.shape[1])]
risks_emb_df = pd.DataFrame(df["risks_miniLM"].tolist(), index=df.index)
risks_emb_df.columns = [f"risks_miniLM_{i}" for i in range(risks_emb_df.shape[1])]

df = df.drop(columns=["story_miniLM", "risks_miniLM"])
df = pd.concat([df, story_emb_df, risks_emb_df], axis=1)

# === 3. חילוץ פיצ'רים ממבנים מקוננים ===
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

# === 4. חישוב preparation_days ===
df["created_at"] = pd.to_datetime(df["created_at"], unit="ms", errors="coerce")
df["launched_at"] = pd.to_datetime(df["launched_at"], unit="ms", errors="coerce")
df["preparation_days"] = (df["launched_at"] - df["created_at"]).dt.days

# === 5. יצירת category_Web_Combined ===
df["category_Web_Combined"] = (
    df.get("category_Web", False).fillna(False).astype(bool) |
    df.get("category_Web Development", False).fillna(False).astype(bool)
).astype(int)

# === 6. הגדרת רשימת פיצ'רים ===
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

category_features = [
    col for col in df.columns if col.startswith("category_") and col != "category_Technology"
]

story_minilm_features = [col for col in df.columns if col.startswith("story_miniLM")]
risks_minilm_features = [col for col in df.columns if col.startswith("risks_miniLM")]

all_features = base_features + category_features + story_minilm_features + risks_minilm_features
all_features = list(dict.fromkeys(all_features))  # הסרת כפילויות

# בדיקת עמודות חסרות
missing_features = [f for f in all_features if f not in df.columns]
if missing_features:
    print("⚠️ הפיצ'רים הבאים חסרים בדאטה:", missing_features)
    raise KeyError("🛑 קיימים פיצ'רים חסרים – אי אפשר לאמן ככה את המודל.")

# === 7. הכנת הנתונים לאימון ===
X = df[all_features].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["success"]

bad_columns = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
if bad_columns:
    raise ValueError(f"🛑 קיימות עמודות לא מספריות – {bad_columns}")

# === 8. פיצול ל־Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 9. אימון מודל LightGBM ===
model = lgb.LGBMClassifier(
    learning_rate=0.02742,
    max_depth=-1,  # אפשר להשאיר ללא הגבלה אם לא ציינת max_depth
    n_estimators=363,
    num_leaves=41,
    subsample=0.9615,
    colsample_bytree=0.91198,
    random_state=42
)
model.fit(X_train, y_train)

# === 10. הערכת ביצועים ===
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

print("🎯 דיוק:", accuracy_score(y_test, y_pred))
print("📊 AUC:", roc_auc_score(y_test, y_pred_prob))
print("📋 דו״ח:")
print(classification_report(y_test, y_pred))

# === 11. שמירת המודל ורשימת הפיצ'רים ===
joblib.dump(model, "lightgbm_kickstarter_success_model.pkl")
with open("lightgbm_feature_columns.json", "w") as f:
    json.dump(all_features, f)

print("✅ המודל והפיצ'רים נשמרו בהצלחה.")
