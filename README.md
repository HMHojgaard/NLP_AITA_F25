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

- **AITA_NLP_nb.ipynb**  
  Main analysis notebook containing all preprocessing, modeling, evaluation, and interpretability analyses.

- **data/**  
  Input datasets (compressed for GitHub size limits).
  - `data.zip`  
    Contains the training/validation dataset and the external test dataset (CSV files).

- **classification_reports/**  
  Model evaluation results saved as CSV files.
  - **tfidf/**
    - `tfidf_classification_report.csv`
    - `tfidf_external_test_classification_report.csv`
  - **bert/**
    - `bert_epoch1_classification_report.csv`
    - `bert_epoch2_classification_report.csv`
    - `bert_w_epoch2_classification_report.csv`
    - `bert_external_test_classification_report.csv`

- **plots_results/**  
  Confusion matrices and evaluation plots.
  - **tfidf/**
    - `TF-IDF+logisticregression_cm.png`
    - `TF-IDF+logisticregression_confusionmatrix.png`
  - **bert/**
    - `bert_1epoch_cm.png`
    - `bert_2epochs_cm.png`

- **tables_results/**  
  Aggregated result tables used in the paper.
  - `results_table_compare.csv`
  - `results_table_ext_compare.csv`

- **bert_token/**  
  Token-level attribution and interpretability outputs.
  - `bert_token_importance_yta_vs_nta.csv`
  - `bert_token_importance_yta_vs_nta_top50.csv`
  - `bert_token_contrastive_aggregated.csv`
  - `bert_token_contrastive_top_tokens.png`

- **README.md**  
  What you're reading right now; Project description and reproducibility instructions.

- **LICENSE**  
  Repository license.

---
## Installation (UCloud)

This project was developed using Python 3.10+.

To install dependencies on UCloud:

1. Open a terminal in your UCloud environment
2. Navigate to the project root directory
3. Install dependencies using:

bash
pip install -r requirements.txt

All required packages can also be installed directly in the notebook via `pip` (See chunk 0: Setup).

---

## Note
- The folder **data* is compressed. Unzip before running notebook. 






