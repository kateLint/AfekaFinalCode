import json
from transformers import AutoTokenizer

# Paths
data_path = '/Users/kerenlint/Projects/Afeka/models/all_good_projects_with_modernbert_embeddings_enhanced_with_miniLM12.json'
output_path = '/Users/kerenlint/Projects/Afeka/models/projects_with_short_stories_and_risks.json'

# Model names/paths
MODEL_NAMES = {
    "modernbert": "answerdotai/ModernBERT-base",  # ModernBERT official checkpoint
    "roberta": "roberta-base",
    "minilm": "sentence-transformers/all-MiniLM-L12-v2"
}

# Load tokenizers
TOKENIZERS = {name: AutoTokenizer.from_pretrained(model) for name, model in MODEL_NAMES.items()}

# Load projects
def load_projects(path):
    with open(path, 'r') as f:
        return json.load(f)

def is_within_token_limit(text, tokenizer, max_tokens=512):
    tokens = tokenizer(text, truncation=False, return_tensors="pt")
    return tokens['input_ids'].shape[1] <= max_tokens

def project_within_limits(project):
    story = project.get("story_clean", "")
    risks = project.get("risks-and-challenges_clean", "")
    for name, tokenizer in TOKENIZERS.items():
        if not is_within_token_limit(story, tokenizer):
            return False
        if not is_within_token_limit(risks, tokenizer):
            return False
    return True

def main():
    projects = load_projects(data_path)
    filtered = []
    failed = []
    for p in projects:
        if project_within_limits(p):
            filtered.append(p)
        else:
            failed.append(p)
    print(f"Total projects: {len(projects)}")
    print(f"Successful (within token limits): {len(filtered)}")
    print(f"Failed (exceeded token limits): {len(failed)}")
    with open(output_path, 'w') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"Filtered projects saved to: {output_path}")

if __name__ == "__main__":
    main()
