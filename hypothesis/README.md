# Kickstarter Project Analysis Toolkit

This repository contains a suite of Python scripts developed to analyze Kickstarter campaign data, focusing on textual and behavioral predictors of project success. Each script corresponds to a hypothesis or set of hypotheses in a broader research context (e.g., Master's thesis).

## 📁 File Overview

| File        | Purpose                                                                 |
|-------------|-------------------------------------------------------------------------|
| `h1-h3.py`  | Logistic regression + Welch t-tests to evaluate the impact of project engagement, readability, and sentiment on success. |
| `h4.py`     | Topic modeling and keyword detection (H4), assessing thematic framing (e.g., innovation, community, transparency). |
| `h5.py`     | Exploratory data analysis of project category impact, update frequency, and success correlation (H5). |
| `h6.py`     | Passive voice detection and sentence structure analysis of project text (H6). |

---

## 📊 Hypotheses & Corresponding Scripts

| Hypothesis | Description | Script |
|------------|-------------|--------|
| **H1**     | Higher project engagement (e.g., updates, comments) correlates with success. | `h1-h3.py` |
| **H2**     | Higher readability in project text correlates with success. | `h1-h3.py` |
| **H3**     | More positive or clearer sentiment correlates with success. | `h1-h3.py` |
| **H4**     | Thematic framing (innovation, community, transparency) affects success probability. | `h4.py` |
| **H5**     | Categories with more updates show higher success rates. | `h5.py` |
| **H6**     | Lower passive voice usage and shorter sentences improve success. | `h6.py` |

---

## 🧪 Script Details

### `h1-h3.py`: Logistic Regression + t-tests
- Loads Kickstarter JSON/CSV/JSONL data.
- Harmonizes nested readability, sentiment, and engagement features.
- Computes:
  - Welch t-tests by feature group.
  - Logistic regression with standardized coefficients, odds ratios, Wald z-tests.
  - VIF for multicollinearity diagnostics.
  - 5-fold stratified CV report for predictive sanity check.
- Output: `.csv` files and a `README.txt` summary under `./outputs/`.

### `h4.py`: Topic Framing via Keywords
- Defines keyword sets for:
  - **Innovation** (e.g., "cutting-edge", "revolutionary")
  - **Community** (e.g., "support", "together")
  - **Transparency** (e.g., "plan", "tested", "honest")
- Applies keyword search to cleaned `story` texts.
- Computes logistic regression and chi-squared tests.
- Prints summary stats and odds ratios.

### `h5.py`: Category-Based Update Analysis
- Loads data and filters by project state.
- Computes average number of updates and success rate per category (excluding Technology).
- Visualizes:
  - Barplots of average updates and success rates by category.
  - Scatter plot of updates vs. success rate.
  - Pearson correlation between update frequency and success.

### `h6.py`: Passive Voice Analysis
- Uses spaCy to detect passive constructions in:
  - `story`
  - `risks-and-challenges`
- Extracts:
  - Passive count, ratio, and average sentence length.
  - Examples of passive constructions.
- Outputs:
  - CSV with added metrics.
  - Visualizations (histograms, boxplots, scatter plots).
  - Text file with example sentences.
  - Summary report with statistics and correlations.

---
