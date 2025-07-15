# KickstarterStructure

[![Python 3.7+](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/) [![LightGBM](https://img.shields.io/badge/LightGBM-1.6-orange)](https://lightgbm.readthedocs.io/en/stable/)

A machine learning pipeline to predict kickstarter projects success.

---

The raw data file all_good_projects_without_embeddings.json were produced from files downloaded from: https://webrobots.io/kickstarter-datasets/ from 2015-2014.
Then projects from scrapped_projects.json were combined with those files, and clean_data.py created the final all_good_projects_without_embeddings.json

## Prerequisites

- Python 3.7 or higher
- scikit-learn  
- numpy  
- pandas  
- joblib

# Install dependencies
pip install scikit-learn numpy pandas joblib


## Project Structure

```bash
NMR2structure/
├── main.py                       # Training and evaluation entry point
├── model.py                      # Gradient boosting model definitions
├── database.py                   # Data loading and substructure mapping
├── nmrshiftdb2withsignals.sd     # Raw NMRShiftDB2 dataset
└── README.md                     # Project documentation
```

## Evaluation

* General model (166 motifs): AUC = 0.77, Precision = 71%, Recall = 55%
* Carbohydrate model: AUC = 0.93, Precision = 93%, Recall = 88%
