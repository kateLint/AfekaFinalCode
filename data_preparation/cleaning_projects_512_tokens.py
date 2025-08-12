import json
from transformers import AutoTokenizer

# Paths
data_path = '/Users/kerenlint/Projects/cursor/projects_with_short_stories_and_risks_cleaned.json'
output_path = '/Users/kerenlint/Projects/cursor/projects_with_short_stories_and_risks_cleaned_512_tokens.json'

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
    # Determine the correct timestamp for Jan 1, 2025
    # If your data is in seconds, use 1735689600
    # If your data is in milliseconds, use 1735689600000
    CUTOFF_TIMESTAMP = 1735689600000  # Jan 1, 2025 in seconds
    for p in projects:
        # Remove projects launched in 2025 or later
        launched_at = p.get('launched_at')
        if launched_at is not None and launched_at >= CUTOFF_TIMESTAMP:
            continue
        if project_within_limits(p):
            filtered.append(p)
        else:
            failed.append(p)
    print(f"Total projects: {len(projects)}")
    print(f"Successful (within token limits): {len(filtered)}")
    print(f"Failed (exceeded token limits): {len(failed)}")
    # Count states in filtered projects
    num_successful = sum(1 for p in filtered if p.get("state") == "successful")
    num_failed = sum(1 for p in filtered if p.get("state") == "failed")
    print(f"Of the {len(filtered)} projects within token limits:")
    print(f"  Successful: {num_successful}")
    print(f"  Failed: {num_failed}")
    with open(output_path, 'w') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"Filtered projects saved to: {output_path}")

if __name__ == "__main__":
    main()
