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
install dependences from requirements.txt

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
{
    "goal": 131421,
    "rewardscount": 6, #optional
    "projectFAQsCount": 8, #optional
    "project_length_days": 30, #optional
    "preparation_days": 5, #optional
    "category_Web_Development": 1, #optional
    "story": (
        "Innovative Device is an ambitious project aimed at revolutionizing the Gadgets industry. While the concept behind Innovative Device was met with enthusiasm, we faced significant challenges in securing the necessary funding and resources. The journey has been filled with obstacles, ranging from supply chain issues to unexpected technical difficulties. Despite our best efforts, these setbacks delayed our timeline and affected our ability to bring Innovative Device to market at the scale we envisioned. Our hope was to introduce Innovative Device to the Gadgets market and make a meaningful impact. Although this campaign did not reach its goal, the feedback and support from our community have been invaluable. We will continue exploring alternative funding options and look forward to relaunching Innovative Device in the future with a stronger foundation."
    ),
    "risks": (
        "Launching Innovative Device in the field of Gadgets comes with its own set of challenges. One of the biggest concerns is ensuring that Innovative Device integrates seamlessly with existing Gadgets solutions. Compatibility issues may arise, requiring extensive testing and refinements before mass production. Additionally, sourcing high-quality components for Gadgets-specific hardware can be time-consuming and costly. Security is another major factor. Innovative Device will need to maintain strict data protection standards to ensure privacy and prevent cyber threats. The regulatory landscape for Gadgets is evolving, and ensuring compliance with industry standards is crucial for Innovative Device to be legally distributed in multiple markets. Our team is prepared to address these risks by implementing a robust quality control process, working closely with industry experts, and securing partnerships with reliable manufacturers to ensure a smooth launch."
    ),
}
```

The output:

🎯 ORIGINAL STORY:
Innovative Device is an ambitious project aimed at revolutionizing the Gadgets industry. While the
concept behind Innovative Device was met with enthusiasm, we faced significant challenges in
securing the necessary funding and resources. The journey has been filled with obstacles, ranging
from supply chain issues to unexpected technical difficulties. Despite our best efforts, these
setbacks delayed our timeline and affected our ability to bring Innovative Device to market at the
scale we envisioned. Our hope was to introduce Innovative Device to the Gadgets market and make a
meaningful impact. Although this campaign did not reach its goal, the feedback and support from our
community have been invaluable. We will continue exploring alternative funding options and look
forward to relaunching Innovative Device in the future with a stronger foundation.
📖 Story-only Probability: 9.43%

⚠️ ORIGINAL RISKS:
Launching Innovative Device in the field of Gadgets comes with its own set of challenges. One of the
biggest concerns is ensuring that Innovative Device integrates seamlessly with existing Gadgets
solutions. Compatibility issues may arise, requiring extensive testing and refinements before mass
production. Additionally, sourcing high-quality components for Gadgets-specific hardware can be
time-consuming and costly. Security is another major factor. Innovative Device will need to maintain
strict data protection standards to ensure privacy and prevent cyber threats. The regulatory
landscape for Gadgets is evolving, and ensuring compliance with industry standards is crucial for
Innovative Device to be legally distributed in multiple markets. Our team is prepared to address
these risks by implementing a robust quality control process, working closely with industry experts,
and securing partnerships with reliable manufacturers to ensure a smooth launch.
⚠️ Risks-only Probability: 17.21%

🎯 Combined Probability: 12.16%
📈 Visual Score: 🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (12.16%)


⚡ Paraphrase Suggestions — STORY first, then RISKS:
🔹 Suggestion #1 [STORY]
🧠 Theme: revolutionize gadgets industry / device revolutionize gadgets / required funding infrastructure
The goal of Innovative Device is to revolutionize the Gadgets industry through innovation. Despite
this initial enthusiasm, we encountered significant obstacles in acquiring the required funding and
infrastructure. These obstacles, such as supply chain problems and technical problems, caused us to
delay the implementation of Innovative Device to the market. Despite this, we aim to expand our
range of products in a significant way by focusing solely on the Gadgets industry. The overwhelming
response from CSQA conferences and workshops, as well as individuals, indicates our commitment to
providing appropriate funding.
✅ Success Probability: 14.18%
📈 Visual Score: 🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (14.18%)
🧪 Params: top_k=80, top_p=0.95, temperature=1.2
🧠 Coherence: 0.72 ✅ Strong
🔹 Suggestion #2 [STORY]
🧠 Theme: ambitious project seeks / difficulties hindered efforts / challenges obtaining necessary
Innovative Device is an ambitious project that seeks to revolutionize the Gadgets industry, but it
has been a struggle. We encountered numerous challenges in obtaining the necessary funding and
resources, such as supply chain problems and technical difficulties. These difficulties hindered our
efforts to achieve the scale of our vision, which we hope will be achieved in the Gadgets market.
Despite this disappointment, we remain hopeful and will continue to seek more funding from community
members.
✅ Success Probability: 13.03%
📈 Visual Score: 🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (13.03%)
🧪 Params: top_k=60, top_p=0.88, temperature=0.9
🧠 Coherence: 0.71 ✅ Strong
🔹 Suggestion #3 [STORY]
🧠 Theme: desire revolutionize gadgets / revolutionize gadgets industry / strong desire revolutionize
Our initiative, Innovative Device, was born out of a strong desire to revolutionize the Gadgets
industry. Despite a challenging launch, we were met with numerous obstacles, including poor supply
chain issues and unexpected technical issues. We resorted to pitching Innovative Device to other
companies, but these difficulties only added fuel to our fire. Despite this disappointment, we
believe that we must not hold back from pursuing funding or restarting our project in a similar way.
The feedback we have received has since been invaluable in promoting the idea further sharing,
despite ongoing, in response to an overall
✅ Success Probability: 6.63%
📈 Visual Score: 🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (6.63%)
🧪 Params: top_k=40, top_p=0.92, temperature=1.0
🧠 Coherence: 0.72 ✅ Strong
🔹 Suggestion #1 [RISKS]
🧠 Theme: expensive testing security / consuming expensive testing / expensive testing
We understand the challenges that come with launching Innovative Device in the market of Gadgets,
including the need to integrate the product into existing Gadgets solutions through years of
compatability and time-consuming and expensive testing; security is also a top priority, with even
the latest hardware having to be made in a time-consuming and low-quality process. Lastly, we aim to
secure facilities in a strong and reliable manner, and to ensure that innovative Device is ready for
market success through rigorous quality control of its existing components, including the ideally,
dependable services and dependable tools
✅ Success Probability: 12.07%
📈 Visual Score: 🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (12.07%)
🧪 Params: top_k=80, top_p=0.95, temperature=1.2
🧠 Coherence: 0.77 ✅ Strong
🔹 Suggestion #2 [RISKS]
🧠 Theme: solutions require time / require time effort / hardware costly time
When launching Innovative Device in the Gadgets market, we face multiple challenges. Specifically,
we focus on ensuring that our devices can integrate seamlessly with existing Gadgets solutions,
which may require time and effort to refine. Additionally, we recognize that sourcing top-notch
components for Gadgets-specific hardware can be a costly and time-consuming undertaking, and will
require significant stumbling blocks due to the need to uphold strict data protection standards.
Ultimately, we aim to ensure that our devices adhere to industry standards that we expect reliable
quality.
✅ Success Probability: 11.00%
📈 Visual Score: 🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (11.00%)
🧪 Params: top_k=40, top_p=0.92, temperature=1.0
🧠 Coherence: 0.74 ✅ Strong
🔹 Suggestion #3 [RISKS]
🧠 Theme: hardware time consuming / expensive endeavor security / time consuming expensive
Our team is well-equipped to handle the challenges and pitfalls that come with launching Innovative
Device within the open world of Gadgets. One of the primary concerns is that it may not be
compatible with existing Gadgets solutions, leading to time and expense on both sides. Furthermore,
sourcing high-quality components for Gadgets-specific hardware can be a time-consuming and expensive
endeavor. Moreover, security needs to be taken into account, and a quality product must comply with
industry standards in order to be sold in multiple markets.
✅ Success Probability: 10.88%
📈 Visual Score: 🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (10.88%)
🧪 Params: top_k=60, top_p=0.88, temperature=0.9
🧠 Coherence: 0.61 ✅ Strong

🔷 Best combo is story2 + risk1 = 13.92%
📈 Visual Score: 🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (13.92%)

🔧 Run optimization with Optuna for STORY & RISKS? (y/n): y
[I 2025-08-12 15:45:21,493] A new study created in memory with name: no-name-2c57d340-967c-4174-958c-b1af20a963ab
🔄 STORY Trial 1/10 …[I 2025-08-12 15:45:25,364] Trial 0 finished with value: 0.14697660505771637 and parameters: {'top_k': 54, 'top_p': 0.8706289163800806, 'temperature': 0.9946885636460518}. Best is trial 0 with value: 0.14697660505771637.
🔄 STORY Trial 2/10 …[I 2025-08-12 15:45:30,029] Trial 1 finished with value: 0.12732796370983124 and parameters: {'top_k': 80, 'top_p': 0.8917300702676217, 'temperature': 1.162874974418276}. Best is trial 0 with value: 0.14697660505771637.
...
...
...
✅ Optimization completed for risks

🔹 OPTUNA RESULT — STORY:
🧠 Theme: revolutionize gadgets industry / seeks revolutionize gadgets / bring market larger
Innovative Device is a project that seeks to revolutionize the Gadgets industry. Despite our
enthusiasm for the idea, we encountered significant obstacles in obtaining the necessary funding and
resources. These challenges, along with supply chain issues, have significantly impacted our
timeline as we strive to bring it to market on a larger scale. Despite this setback, we remain
committed to building on our previous success and gratefully accepting the feedback and support we
have received from our community.
✅ Success Probability: 18.78%
🧪 Params: {'top_k': 112, 'top_p': 0.9293992888111543, 'temperature': 0.8206255834472103}
🧠 Notes:
• Very high top-k => creative but unstable.
• Medium top-p is balanced.
• Low temperature => predictable phrasing; high => surprising phrasing.

🔹 OPTUNA RESULT — RISKS:
🧠 Theme: challenges include time / extensive testing refinement / involve extensive testing
The process of launching Innovative Device in the Gadgets industry presents several challenges, one
of which is integrating seamlessly with existing Gadgets solutions. This may involve extensive
testing and refinement, while other challenges include time and money required to source high-
quality components for Gadgets-specific hardware. Security is also a top priority, and it will
require adherence to strict data protection standards to prevent cyber threats. To ensure a
successful launch, we will work with reliable manufacturers to ensure seamless integration across
all devices.
✅ Success Probability: 14.58%
🧪 Params: {'top_k': 150, 'top_p': 0.8705642120335156, 'temperature': 0.8056614629169244}
🧠 Notes:
• Very high top-k => creative but unstable.
• Low top-p narrows choice distribution (stable).
• Low temperature => predictable phrasing; high => surprising phrasing.

🔷 COMBINED (apply STORY+RISKS best):
✅ Success Probability: 21.16%
📈 Visual Score: 🟩🟩🟩🟩⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ (21.16%)




```


