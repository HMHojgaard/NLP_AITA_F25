# NLP_AITA_F25
Code and results for moral judgment classification on AITA using NLP



# Moral Judgment Classification in AITA

## Overview
This repository contains the code and outputs for an NLP exam project analyzing **crowd-sourced moral judgment** in narrative text from the *Am I the Asshole?* (AITA) subreddit.

The project investigates:
- whether NLP models can predict moral verdicts (YTA, NTA, ES),
- how classical and transformer-based approaches compare, and
- which linguistic cues contribute to model decisions.

The analysis combines:
- a TF-IDF + Logistic Regression baseline,
- a fine-tuned DistilBERT transformer model,
- class-weighted training to address label imbalance, and
- token-level interpretability using gradient-based attribution.

All models and analysis are implemented and documented in a single Jupyter notebook (AITA_NLP_nb), with results exported as tables, figures, and CSV files.

---

## Repository Structure

NLP_AITA_F25/
│
├── AITA_NLP_nb.ipynb
│ # Main analysis notebook (preprocessing, models, evaluation, interpretability)
│
├── data/
│ └── data.zip
│ # Compressed dataset (train/validation + external test CSVs)
│
├── classification_reports/
│ ├── tfidf/
│ │ ├── tfidf_classification_report.csv
│ │ └── tfidf_external_test_classification_report.csv
│ │
│ └── bert/
│ ├── bert_epoch1_classification_report.csv
│ ├── bert_epoch2_classification_report.csv
│ ├── bert_w_epoch2_classification_report.csv
│ └── bert_external_test_classification_report.csv
│
├── plots_results/
│ ├── tfidf/
│ │ ├── TF-IDF+logisticregression_cm.png
│ │ └── TF-IDF+logisticregression_confusionmatrix.png
│ │
│ └── bert/
│ ├── bert_1epoch_cm.png
│ └── bert_2epochs_cm.png
│
├── tables_results/
│ ├── results_table_compare.csv
│ └── results_table_ext_compare.csv
│
├── bert_token/
│ ├── bert_token_importance_yta_vs_nta.csv
│ ├── bert_token_importance_yta_vs_nta_top50.csv
│ ├── bert_token_contrastive_aggregated.csv
│ └── bert_token_contrastive_top_tokens.png
│
├── LICENSE
└── README.md
---

## Requirements
- **Python**: 3.10 or later  
- **Hardware**: GPU recommended for training

### Main dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- torch
- transformers
- datasets
- evaluate
- accelerate

All required packages are installed directly in the notebook via `pip`.

---

## Note
- The folder **data* is compressed. Unzip before running notebook. 






