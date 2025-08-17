import pandas as pd
import spacy
from tqdm import tqdm
import os

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

file_path = "/Users/kerenlint/Projects/Afeka/all_models/all_good_projects_without_embeddings.json"
output_path = "/Users/kerenlint/Projects/Afeka/all_models/all_projects_with_passive_analysis.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at: {file_path}")

df = pd.read_json(file_path, convert_dates=False)

def is_passive(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "nsubjpass" or (token.tag_ == "VBN" and token.head.dep_ == "auxpass"):
            return True
    return False

def analyze_passive(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0, 0
    doc = nlp(text)
    total = 0
    passive = 0
    for sent in doc.sents:
        total += 1
        if is_passive(sent.text):
            passive += 1
    return passive, total

def apply_passive_analysis(df, column_name):
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found in dataset")
        return df
    
    passive_counts = []
    total_counts = []
    for text in tqdm(df[column_name], desc=f"Analyzing {column_name} for passive voice"):
        passive, total = analyze_passive(text)
        passive_counts.append(passive)
        total_counts.append(total)
    
    df[f"{column_name}_passive_sentences"] = passive_counts
    df[f"{column_name}_total_sentences"] = total_counts
    df[f"{column_name}_passive_ratio"] = df[f"{column_name}_passive_sentences"] / df[f"{column_name}_total_sentences"].replace(0, 1)
    return df

df = apply_passive_analysis(df, "story")
df = apply_passive_analysis(df, "risks-and-challenges")

df.to_csv(output_path, index=False)
print(f"\n✅ Passive voice analysis completed and saved to:\n{output_path}")