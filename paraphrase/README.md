# Kickstarter AI Success Predictor & Story Optimizer

This project is a **Kickstarter campaign success prediction and optimization tool** that:
- Uses **RoBERTa embeddings**, **XGBoost classification**, and **Optuna hyperparameter tuning**  
- Generates **paraphrased project stories** to increase predicted success probability  
- Evaluates **coherence, key phrases, and thematic relevance**  
- Provides **quick suggestions** or performs **full optimization**  

---

## Features
- **Prediction Pipeline**: Estimates Kickstarter campaign success probability using a trained model.
- **Text Embedding**: Supports long text chunking & pooling for RoBERTa embeddings (768-dim).
- **Paraphrasing**: Uses a fine-tuned T5 model to generate multiple paraphrase variations.
- **Keyphrase Extraction**: Identifies the main themes of your campaign story.
- **Coherence Scoring**: Measures similarity between original and paraphrased sentences.
- **Optuna Optimization**: Finds optimal generation parameters for maximum predicted success.
- **Logging & Health Checks**: Tracks processes and validates system components.

---

## Requirements
install dependences from requirements

---

## Environment Variables

Optional environment variables to override default paths and settings:

| Variable           | Description                                     | Default                                              |
| ------------------ | ----------------------------------------------- | ---------------------------------------------------- |
| `KS_MODEL_PATH`    | Path to trained XGBoost model                   | `.../xgboost_kickstarter_success_model.pkl`          |
| `KS_FEATURES_PATH` | Path to JSON list of feature columns            | `.../xgboost_feature_columns.json`                   |

---

## How It Works

1. **Load Models**

   * Paraphraser: `humarin/chatgpt_paraphraser_on_T5_base`
   * RoBERTa Embedder: configurable via `KS_ROBERTA_NAME`
   * XGBoost Classifier: trained on Kickstarter campaign data
   * KeyBERT: for key phrase extraction

2. **Prepare Features**

   * Extract numerical and categorical features from `project_input`
   * Generate RoBERTa embeddings for `story` and `risks` text

3. **Predict Probability**

   * Use classifier to predict success probability based on features + embeddings

4. **Generate Suggestions**

   * Quick suggestions: multiple paraphrased variations ranked by probability
   * Optuna optimization: search for the best generation parameters

5. **Output Results**

   * Probability scores, coherence scores, key phrases, and parameter explanations

---

## Usage

Run the script:

```bash
python main.py
```

You will see:

* Original story and predicted success probability
* Quick paraphrase suggestions with probabilities
* Option to run a full Optuna optimization search

Example output snippet:

```
🎯 ORIGINAL STORY:
Innovative Device is an ambitious project aimed at revolutionizing the Gadgets industry...

✅ Success Probability (combined): 42.13%
📈 Visual Score: 🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (42.13%)

⚡ Fast Paraphrase Suggestions for STORY:
🔹 Suggestion #1
🧠 Theme: innovative device / gadgets / ambitious project
Innovative Device seeks to transform the gadgets industry with...
✅ Success Probability: 48.56%
📈 Visual Score: 🟩🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (48.56%)
🧠 Coherence Score: 0.78 ✅ Strong
```

---

## Example `project_input` Structure

```python
project_input = {
    "goal": 131421,
    "rewardscount": 6,
    "projectFAQsCount": 8,
    "project_length_days": 30,
    "preparation_days": 5,
    "category_Web_Development": 1,
    "story": "Your Kickstarter story text...",
    "risks": "Your project risks text..."
}
```
