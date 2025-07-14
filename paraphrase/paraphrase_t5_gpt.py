import pandas as pd
import numpy as np
import joblib
import json
import torch
import random
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from typing import Tuple, List
import nltk
import threading
import textwrap

# Ensure punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Load Models ===
paraphraser_model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
paraphraser_tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
clf_model = joblib.load("/Users/kerenlint/Projects/Afeka/all_models/lightgbm_kickstarter_success_model.pkl")
with open("/Users/kerenlint/Projects/Afeka/all_models/lightgbm_feature_columns.json") as f:
    clf_features = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paraphraser_model = paraphraser_model.to(device)


from keybert import KeyBERT

kw_model = KeyBERT("sentence-transformers/all-MiniLM-L12-v2")

import warnings

def extract_keyphrases(text, top_n=3) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=top_n
            )
        return " / ".join([kw for kw, _ in keywords])
    except Exception as e:
        print(f"⚠️ Keyword extraction failed: {e}")
        return "N/A"



def render_score_bar(probability: float, width: int = 20) -> str:
    filled = int(probability * width)
    empty = width - filled
    return "🟩" * filled + "⬜" * empty


# === Helper Functions ===
def clean_paraphrase(text: str) -> str:
    return text.strip().replace(" .", ".").replace("..", ".").replace("\n", " ")

def generate_paraphrases(text: str, params: dict, num_return: int = 4) -> List[str]:
    input_text = f"paraphrase: {text} </s>"
    inputs = paraphraser_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    try:
        outputs = paraphraser_model.generate(
            inputs,
            num_return_sequences=num_return,
            do_sample=True,
            num_beams=num_return,  # Ensures beams >= num_return_sequences
            **params
        )
        return [clean_paraphrase(paraphraser_tokenizer.decode(o, skip_special_tokens=True)) for o in outputs]
    except Exception as e:
        print(f"Error generating paraphrases: {e}")
        return []

def multi_sentence_coherence(orig: str, para: str, threshold: float = 0.6) -> bool:
    orig_sents = sent_tokenize(orig)
    para_sents = sent_tokenize(para)
    if not orig_sents or not para_sents:
        return False
    scores = []
    for s1, s2 in zip(orig_sents, para_sents):
        v1 = minilm.encode(s1)
        v2 = minilm.encode(s2)
        scores.append(cosine_similarity([v1], [v2])[0][0])
    return np.mean(scores) >= threshold

def evaluate_text_success(text: str, df_template: pd.DataFrame, text_type: str = 'story') -> float:
    embed = minilm.encode(text)
    prefix = 'story_miniLM_' if text_type == 'story' else 'risks_miniLM_'
    for i, val in enumerate(embed):
        if f"{prefix}{i}" in df_template.columns:
            df_template[f"{prefix}{i}"] = val
    return clf_model.predict_proba(df_template[clf_features])[0, 1]

import optuna

from typing import List, Dict

def explain_params(params: Dict[str, float]) -> List[str]:
    explanations = []

    # Top-k explanation
    if params["top_k"] <= 40:
        explanations.append("A low top-k value increases coherence but may limit linguistic diversity.")
    elif params["top_k"] >= 100:
        explanations.append("A high top-k value encourages creativity but can lead to inconsistency.")
    else:
        explanations.append("A medium top-k value balances diversity and coherence.")

    # Top-p explanation
    if params["top_p"] >= 0.95:
        explanations.append("A high top-p value allows flexible word selection, promoting creativity.")
    elif params["top_p"] <= 0.88:
        explanations.append("A low top-p value restricts choices, supporting a more conservative structure.")
    else:
        explanations.append("A medium top-p value offers a good balance between stability and diversity.")

    # Temperature explanation
    if params["temperature"] < 0.9:
        explanations.append("A low temperature ensures stability but reduces novelty.")
    elif params["temperature"] > 1.3:
        explanations.append("A high temperature generates more surprising, less predictable phrasing.")
    else:
        explanations.append("A medium temperature balances consistency and diversity.")

    return explanations

def optimize_paraphrase_optuna(original_text: str,
                                df_template: pd.DataFrame,
                                text_type: str = 'story',
                                coherence_threshold: float = 0.6,
                                n_trials: int = 25) -> Tuple[str, float, dict, List[str]]:
    """
    Bayesian optimization of paraphrasing parameters using Optuna.
    Returns best paraphrase, probability, parameters, and explanation list.
    """
    def objective(trial):
        params = {
            'top_k': trial.suggest_int("top_k", 20, 150),
            'top_p': trial.suggest_float("top_p", 0.85, 0.98),
            'temperature': trial.suggest_float("temperature", 0.8, 1.5),
            'max_length': 512
        }
        paraphrases = generate_paraphrases(original_text, params, num_return=2)
        for para in paraphrases:
            if not multi_sentence_coherence(original_text, para, coherence_threshold):
                continue
            prob = evaluate_text_success(para, df_template.copy(), text_type)
            trial.set_user_attr("paraphrase", para)
            return prob
        return 0.0  # if no good paraphrase passed coherence

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_para = best_trial.user_attrs["paraphrase"]
    best_score = best_trial.value
    best_params = best_trial.params
    best_explanations = explain_params(best_params)

    return best_para, best_score, best_params, best_explanations

# === Input Project ===
project_input = {
    "goal": 131421,
    "rewardscount": 6,
    "projectFAQsCount": 8,
    "project_length_days": 30,
    "preparation_days": 5,
    "category_Web_Development": 1,
    "story": "Innovative Device is an ambitious project aimed at revolutionizing the Gadgets industry. While the concept behind Innovative Device was met with enthusiasm, we faced significant challenges in securing the necessary funding and resources. The journey has been filled with obstacles, ranging from supply chain issues to unexpected technical difficulties. Despite our best efforts, these setbacks delayed our timeline and affected our ability to bring Innovative Device to market at the scale we envisioned.Our hope was to introduce Innovative Device to the Gadgets market and make a meaningful impact. Although this campaign did not reach its goal, the feedback and support from our community have been invaluable. We will continue exploring alternative funding options and look forward to relaunching Innovative Device in the future with a stronger foundation.",
    "risks": "Launching Innovative Device in the field of Gadgets comes with its own set of challenges. One of the biggest concerns is ensuring that Innovative Device integrates seamlessly with existing Gadgets solutions. Compatibility issues may arise, requiring extensive testing and refinements before mass production. Additionally, sourcing high-quality components for Gadgets-specific hardware can be time-consuming and costly.Security is another major factor. Innovative Device will need to maintain strict data protection standards to ensure privacy and prevent cyber threats. The regulatory landscape for Gadgets is evolving, and ensuring compliance with industry standards is crucial for Innovative Device to be legally distributed in multiple markets.Our team is prepared to address these risks by implementing a robust quality control process, working closely with industry experts, and securing partnerships with reliable manufacturers to ensure a smooth launch of Innovative Device in the Gadgets market.",
}

story_embed = minilm.encode(project_input["story"])
risks_embed = minilm.encode(project_input["risks"])
for i, val in enumerate(story_embed):
    project_input[f"story_miniLM_{i}"] = val
for i, val in enumerate(risks_embed):
    project_input[f"risks_miniLM_{i}"] = val

df_base = pd.DataFrame([project_input])
missing_feats = [f for f in clf_features if f not in df_base.columns]
df_missing = pd.DataFrame([[0]*len(missing_feats)], columns=missing_feats)
df_base = pd.concat([df_base, df_missing], axis=1)
df_base = df_base[clf_features]


def get_quick_suggestions(
    text: str,
    df_template: pd.DataFrame,
    text_type: str = 'story',
    num_return: int = 1,
    coherence_threshold: float = 0.6
) -> List[Tuple[str, float, dict, float, bool]]:
    """
    Returns list of (paraphrase, success_prob, gen_params, coherence_score, is_strong)
    """
    presets = [
        {"top_k": 40, "top_p": 0.92, "temperature": 1.0, "max_length": 512},
        {"top_k": 80, "top_p": 0.95, "temperature": 1.2, "max_length": 512},
        {"top_k": 60, "top_p": 0.88, "temperature": 0.9, "max_length": 512},
    ]
    results = []
    for preset in presets:
        paras = generate_paraphrases(text, preset, num_return=num_return)
        for p in paras:
            score = evaluate_text_success(p, df_template.copy(), text_type)
            coherence = multi_sentence_coherence_score(text, p)
            is_strong = coherence >= coherence_threshold
            results.append((p, score, preset, coherence, is_strong))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results


def multi_sentence_coherence_score(orig: str, para: str) -> float:
    orig_sents = sent_tokenize(orig)
    para_sents = sent_tokenize(para)
    if not orig_sents or not para_sents:
        return 0.0
    scores = []
    for s1, s2 in zip(orig_sents, para_sents):
        v1 = minilm.encode(s1)
        v2 = minilm.encode(s2)
        scores.append(cosine_similarity([v1], [v2])[0][0])
    return np.mean(scores)


def run_optuna_async(original_text, df_template, text_type='story'):
    def task():
        print(f"\n🔍 Optuna מחפש ניסוח משופר ל-{text_type}... זה ייקח כדקה.")
        best_para, score, params, explain = optimize_paraphrase_optuna(original_text, df_template.copy(), text_type)
        print(f"\n✨ Optuna מצא ניסוח משופר ל-{text_type}!")
        print(f"✅ Success Probability: {score:.2%}")
        print(f"📈 Visual Score: {render_score_bar(score)} ({score:.2%})")
        print("📜 Paraphrased Text:\n", textwrap.fill(best_para, width=100))
        print("🧠 Explanations:")
        for e in explain:
            print("•", e)
    threading.Thread(target=task).start()


if __name__ == "__main__":
    story = project_input["story"]
    risks = project_input["risks"]

    # ===== שלב 1: הערכת טקסטים מקוריים =====
    orig_story_prob = evaluate_text_success(story, df_base.copy(), 'story')
    orig_risks_prob = evaluate_text_success(risks, df_base.copy(), 'risks')
    orig_combined_prob = clf_model.predict_proba(df_base[clf_features])[0, 1]

    print("\n🎯 Original STORY:")
    print(textwrap.fill(story, width=100))
    print(f"✅ Success Probability: {orig_story_prob:.2%}")
    print(f"📈 Visual Score: {render_score_bar(orig_story_prob)} ({orig_story_prob:.2%})")

    print("\n🎯 Original RISKS:")
    print(textwrap.fill(risks, width=100))
    print(f"✅ Success Probability: {orig_risks_prob:.2%}")
    print(f"🎯 Combined: {orig_combined_prob:.2%}")
    print(f"📈 Visual Score: {render_score_bar(orig_combined_prob)} ({orig_combined_prob:.2%})")

    # ===== שלב 2: הצעות מהירות ל-STORY =====
    print("\n⚡ Fast Paraphrase Suggestions for STORY:")
    story_fast = get_quick_suggestions(story, df_base.copy(), 'story')
    for i, (txt, prob, p, coherence, is_strong) in enumerate(story_fast):
        theme = extract_keyphrases(txt)
        print(f"\n🔹 Suggestion #{i+1}")
        print(f"🧠 Theme: {theme}")
        print(textwrap.fill(txt, width=100))
        print(f"✅ Success Probability: {prob:.2%}")
        print(f"📈 Visual Score: {render_score_bar(prob)} ({prob:.2%})")

        print(f"🧪 Params: top_k={p['top_k']}, top_p={p['top_p']}, temperature={p['temperature']}")
        print(f"🧠 Coherence Score: {coherence:.2f} {'✅ Strong' if is_strong else '⚠️ Weak'}")

    # ===== שלב 3: הצעות מהירות ל-RISKS =====
    print("\n⚡ Fast Paraphrase Suggestions for RISKS:")
    risks_fast = get_quick_suggestions(risks, df_base.copy(), 'risks')
    for i, (txt, prob, p, coherence, is_strong) in enumerate(story_fast):
        theme = extract_keyphrases(txt)
        print(f"\n🔹 Suggestion #{i+1}")
        print(f"🧠 Theme: {theme}")
        print(textwrap.fill(txt, width=100))
        print(f"✅ Success Probability: {prob:.2%}")
        print(f"📈 Visual Score: {render_score_bar(prob)} ({prob:.2%})")
        print(f"🧪 Params: top_k={p['top_k']}, top_p={p['top_p']}, temperature={p['temperature']}")
        print(f"🧠 Coherence Score: {coherence:.2f} {'✅ Strong' if is_strong else '⚠️ Weak'}")

    # ===== שלב 4: שאל אם לרוץ על Optuna =====
    decision = input("\n🔧 להריץ אופטימיזציה מלאה עם Optuna? (y/n): ").strip().lower()
    if decision.startswith('y'):
        run_optuna_async(story, df_base.copy(), 'story')
        run_optuna_async(risks, df_base.copy(), 'risks')
    else:
        print("\n✅ סיימנו עם הצעות מהירות בלבד. בהצלחה!")
