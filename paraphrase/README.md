# 📘 Kickstarter Paraphrasing Optimizer (RoBERTa Version)

A research-grade tool that uses **paraphrasing, RoBERTa embeddings, and machine learning** to improve Kickstarter campaign success through better narrative phrasing.

---

## 🚀 Features

- ⚙️ **Paraphrase Generation** with [`humarin/chatgpt_paraphraser_on_T5_base`](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base)
- 📊 **Success Evaluation** using a pretrained **XGBoost classifier**
- 📎 **Keyphrase Extraction** via [KeyBERT](https://github.com/MaartenGr/KeyBERT)
- 🧠 **Coherence Scoring** using **RoBERTa embeddings**
- 🔍 **Bayesian Hyperparameter Optimization** with [Optuna](https://optuna.org/)
- ⚡ **Quick Suggestions** using preset decoding strategies
- 📉 **Visual Success Bars** for user-friendly feedback

---

## 🖥️ Usage

### 1. Clone and install requirements:
```bash
git clone <your-repo-url>
cd kickstarter-paraphraser
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### 2. Run the main script:
```bash
python paraphrasa_roberta.py
```

---

## 📁 File Structure

```text
.
├── paraphrasa_roberta.py                # Main script
├── xgboost_kickstarter_success_model.pkl # Trained classifier model
├── xgboost_feature_columns.json          # Feature list
├── requirements.txt
├── README.md
└── (optional) new_projects.json
```

---

## 📥 Inputs

A dictionary-style project input with at least:
- `story`: the main narrative
- `risks`: the risk section
- Structured features:
  - `goal`
  - `rewardscount`
  - `projectFAQsCount`
  - `project_length_days`
  - `preparation_days`
  - category flags (e.g., `category_Web_Development`)

---

## 📤 Outputs

- Console visualization of:
  - Success probabilities
  - Score bars
  - Top paraphrases
  - Optimization reports

---

## 📚 Models Used

| Model                                                 | Purpose                       |
|------------------------------------------------------|-------------------------------|
| `humarin/chatgpt_paraphraser_on_T5_base`             | Paraphrase generation         |
| `sentence-transformers/roberta-base-nli-mean-tokens` | Embeddings + Coherence Check |
| XGBoost classifier                                   | Success prediction           |
| KeyBERT                                              | Keyphrase extraction         |

---

## ⚙️ Optimization

Uses Optuna to find best decoder parameters:
- `top_k`: token diversity (20–150)
- `top_p`: nucleus sampling (0.85–0.98)
- `temperature`: randomness (0.8–1.5)

All trials filtered by coherence threshold (default: `0.60`).

---

## 🔢 Example Input and Output

### 📝 Input (Python dictionary format)

```python
project_input = {
    "goal": 131421,
    "rewardscount": 6,
    "projectFAQsCount": 8,
    "project_length_days": 30,
    "preparation_days": 5,
    "category_Web_Development": 1,
    "story": "Innovative Device is an ambitious project aimed at revolutionizing the Gadgets industry...",
    "risks": "Launching Innovative Device in the field of Gadgets comes with its own set of challenges..."
}
```

### ✅ Output (Console Example)

```text
🎯 ORIGINAL STORY:
✅ Success Probability (combined): 42.35%
📈 Visual Score: 🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (42.35%)

⚡ Fast Paraphrase Suggestions:
🔹 Suggestion #1
🧠 Theme: gadgets / innovation / challenges
✅ Success Probability: 56.10%
🧠 Coherence Score: 0.74 ✅ Strong
📜 Paraphrased Text: This groundbreaking device is designed to redefine how we interact with modern gadgets...
```

---

## 📜 License

MIT License. For academic or research use, please cite appropriately.

---

## ✨ Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Optuna](https://optuna.org/)
- [KeyBERT](https://github.com/MaartenGr/KeyBERT)
- Kickstarter for the dataset
