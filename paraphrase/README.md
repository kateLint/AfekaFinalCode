# 📘 Kickstarter Paraphrasing Optimizer

A research-grade tool that uses **paraphrasing, embeddings, and machine learning** to improve Kickstarter campaign success through better narrative phrasing.

---

## 🚀 Features

- ⚙️ **Paraphrase Generation** with `humarin/chatgpt_paraphraser_on_T5_base`
- 📊 **Success Evaluation** using a pretrained LightGBM classifier
- 📎 **Keyphrase Extraction** via KeyBERT
- 🧠 **Coherence Scoring** using MiniLM embeddings
- 🔍 **Bayesian Hyperparameter Optimization** with Optuna
- ⚡ **Quick Suggestions** using preset decoding strategies
- 📉 **Visual Success Bars** for user-friendly feedback

---

## 🖥️ Usage

### 1. Clone and install requirements:
```bash
git clone <your-repo-url>
cd kickstarter-paraphraser
pip install -r requirements.txt
```

### 2. Run the main script:
```bash
python paraphrasing_optimizer.py
```

---

## 📁 File Structure

```text
.
├── paraphrasing_optimizer.py     # Main script
├── lightgbm_kickstarter_success_model.pkl
├── lightgbm_feature_columns.json
├── requirements.txt
├── README.md
└── (optional) new_projects.json
```

---

## 📥 Inputs

A dictionary-style project input with at least:
- `story`: the main narrative
- `risks`: the risk section
- Structured features: `goal`, `rewardscount`, `category_Web_Development`, etc.

---

## 📤 Outputs

- Console visualization of:
  - Success probabilities
  - Score bars
  - Top paraphrases
  - Optimization reports
- Optional: Save results to CSV (planned)

---

## 📚 Models Used

| Model                                 | Purpose                       |
|--------------------------------------|-------------------------------|
| `humarin/chatgpt_paraphraser_on_T5_base` | Paraphrase generation         |
| `sentence-transformers/all-MiniLM-L12-v2` | Embeddings + Coherence Check |
| LightGBM classifier                  | Success prediction           |
| KeyBERT                              | Keyphrase extraction         |

---

## ⚙️ Optimization

Uses Optuna to find best decoder parameters:
- `top_k`: token diversity
- `top_p`: nucleus sampling
- `temperature`: randomness

All trials filtered by coherence threshold (default: 0.6).

---

## 🛠 Future Improvements

- [ ] Save best paraphrases to CSV
- [ ] Add Gradio interface
- [ ] Extend to pledged ratio prediction
- [ ] Add support for Hebrew translation/analysis


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
🎯 Original STORY:
✅ Success Probability: 42.35%
📈 Visual Score: 🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜

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

- Hugging Face Transformers
- Sentence-Transformers
- Optuna team
- Kickstarter for the dataset
