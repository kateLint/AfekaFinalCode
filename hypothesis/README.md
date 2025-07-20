# 📘 Kickstarter Research Toolkit

This repository contains a suite of Python scripts for hypothesis-driven analysis of Kickstarter project success. The toolkit includes sentiment analysis, topic modeling, readability metrics, and category-wise statistical insights — designed for research or machine learning pipelines around crowdfunding prediction.

---

## 📁 File Overview

| File | Purpose |
|------|---------|
| `h1.py` | **Sentiment Hypothesis Testing** – Runs sentiment analysis using VADER on `story`, `risks`, and `faqs`, with statistical testing and logistic regression. |
| `h2.py` | **Topic Modeling (BERTopic)** – Trains BERTopic on cleaned `story` texts, reduces topics, assigns labels, performs logistic regression, and tests H6 hypothesis using manually defined themes. |
| `h3_story.py` | **Readability Analysis (Story)** – Extracts and evaluates readability and lexical metrics from the "Story Analysis" section. |
| `h3_risks.py` | **Readability Analysis (Risks)** – Similar to above but for "Risks and Challenges", with logistic regression and statistical testing. |
| `h4.py` | **Update Behavior by Category** – Explores how update count varies across categories (excluding Technology) and its relationship to success rate. |

---

## 🧩 Data Requirements

Most scripts expect the following files to exist locally:

- `all_good_projects_without_embeddings.json`

---

## 🚀 How to Use

Each script is **independent**, but assumes relevant JSON input files. Here's how to run and interpret them:

---

### 🧪 `h1.py` – Sentiment-Based Hypotheses

- 📥 **Input**: `all_good_projects_without_embeddings.json`
- 🔍 **What it does**:
  - Extracts VADER sentiment for: `story`, `risks-and-challenges`, and `faqs`.
  - Performs t-tests, correlation, and logistic regression.
  - Visualizes with boxplots and KDEs.

**Usage**:
```bash
python h1.py
```

---

### 🧠 `h2.py` – Topic Modeling with BERTopic

- 📥 **Input**: `all_good_projects_without_embeddings.json`
- 🔍 **What it does**:
  - Trains BERTopic on English-only stories.
  - Reduces to 50 topics, extracts coefficients.
  - Maps topics to themes (e.g., Innovation, Community).
  - Tests logistic regression (H6) and chi-square.

**Usage**:
```bash
python h2.py
```

🛠️ Dependencies: `bertopic`, `tabulate`, `scikit-learn`, `langdetect`

---

### 📖 `h3_story.py` – Readability (Story)

- 📥 **Input**: `all_good_projects_without_embeddings.json`
- 🔍 **What it does**:
  - Extracts readability scores and lexical diversity from `Story Analysis`.
  - Runs logistic regression and t-tests.
  - Plots boxplots per readability metric.

**Usage**:
```bash
python h3_story.py
```

---

### ⚠️ `h3_risks.py` – Readability (Risks)

- 📥 **Input**: `all_good_projects_without_embeddings.json`
- 🔍 **What it does**:
  - Same flow as `h3_story.py`, but for `Risks and Challenges Analysis`.

**Usage**:
```bash
python h3_risks.py
```

---

### 📊 `h4.py` – Category-Wise Analysis

- 📥 **Input**: `all_good_projects_without_embeddings.json`
- 🔍 **What it does**:
  - Calculates average update count and success rate per category (excluding Technology).
  - Visualizes: Barplots, scatter plot, correlation.

**Usage**:
```bash
python h4.py
```

---

## 🛠️ Installation Tips

Make sure to install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels nltk bertopic tabulate langdetect
```

Also, download VADER lexicon for `nltk` (needed by `h1.py`):

```python
import nltk
nltk.download('vader_lexicon')
```

