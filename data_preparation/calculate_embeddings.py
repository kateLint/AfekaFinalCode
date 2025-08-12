import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

# --- Input/output paths ---
input_path = '/Users/kerenlint/Projects/cursor/projects_with_short_stories_and_risks_cleaned_512_tokens.json'
output_path = "/Users/kerenlint/Projects/cursor/projects_with_short_stories_and_risks_with_embeddings.json"

# --- Load and check data ---
with open(input_path, 'r', encoding='utf-8') as f:
    projects = json.load(f)

num_projects = len(projects)
missing_story = [p for p in projects if 'story_clean' not in p]
missing_risks = [p for p in projects if 'risks-and-challenges_clean' not in p]

print(f"[CHECK] Total projects in input: {num_projects}")
print(f"[CHECK] Projects missing 'story_clean': {len(missing_story)}")
print(f"[CHECK] Projects missing 'risks-and-challenges_clean': {len(missing_risks)}")
if missing_story:
    print(f"[CHECK] Example missing 'story_clean': {missing_story[0]}")
if missing_risks:
    print(f"[CHECK] Example missing 'risks-and-challenges_clean': {missing_risks[0]}")

# --- Model names ---
MODELS = {
    "modernbert": "answerdotai/ModernBERT-base",  # CLS pooling
    "roberta": "roberta-base",                    # CLS pooling
    "minilm": "sentence-transformers/all-MiniLM-L12-v2"  # mean pooling
}

# --- Load models and tokenizers ---
model_tokenizer = {}
for key, model_name in MODELS.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model_tokenizer[key] = (tokenizer, model)

# --- Embedding function with CLS or mean pooling ---
def get_embedding(text, tokenizer, model, use_cls=False):
    if not text or not isinstance(text, str) or not text.strip():
        return None
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        if use_cls:
            return outputs.last_hidden_state[:, 0, :].squeeze().tolist()  # [CLS]
        else:
            return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # mean pooling

# --- Validate data ---
df = pd.read_json(input_path)
has_story_and_risks = df['risks-and-challenges_clean'].apply(lambda x: isinstance(x, str) and x.strip() != "") & \
                      df['story_clean'].apply(lambda x: isinstance(x, str) and x.strip() != "")
count = has_story_and_risks.sum()
total = len(df)
print(f"Total projects: {total}")
print(f"Projects with both non-empty 'story' and 'risks': {count}")

# --- Generate embeddings ---
for project in tqdm(projects, desc="Generating embeddings"):
    risks_text = project.get("risks-and-challenges_clean", "")
    story_text = project.get("story_clean", "")

    # RoBERTa — use CLS
    tokenizer, model = model_tokenizer["roberta"]
    project["risk_roberta_embedding"] = get_embedding(risks_text, tokenizer, model, use_cls=True)
    project["story_roberta_embedding"] = get_embedding(story_text, tokenizer, model, use_cls=True)

    # ModernBERT — use CLS
    tokenizer, model = model_tokenizer["modernbert"]
    project["risk_modernbert_embedding"] = get_embedding(risks_text, tokenizer, model, use_cls=True)
    project["story_modernbert_embedding"] = get_embedding(story_text, tokenizer, model, use_cls=True)

    # MiniLM — use mean pooling (optimized for it)
    tokenizer, model = model_tokenizer["minilm"]
    project["risk_minilm_embedding"] = get_embedding(risks_text, tokenizer, model, use_cls=False)
    project["story_minilm_embedding"] = get_embedding(story_text, tokenizer, model, use_cls=False)

# --- Filter and save output ---
projects = [p for p in projects if p.get("story_clean") and p.get("risks-and-challenges_clean")]

with open(output_path, "w", encoding="utf-8") as outfile:
    json.dump(projects, outfile, ensure_ascii=False, indent=2)

num_total = len(projects)
num_success = sum(1 for p in projects if p.get('state') == 'successful')
num_failed = sum(1 for p in projects if p.get('state') == 'failed')

print(f"Embeddings added and saved to {output_path}")
print(f"Total projects: {num_total}")
print(f"Successful projects: {num_success}")
print(f"Failed projects: {num_failed}")
