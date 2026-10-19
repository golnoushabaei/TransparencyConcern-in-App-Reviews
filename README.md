# Transparency Concern Detection in App Reviews

This repository contains the code and resources for a three-stage pipeline to detect transparency concerns in mobile app reviews using LLMs, deep learning, traditional machine learning, and topic modelling.

The code accompanies the experiments described in the associated academic paper.

---

## Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data](#data)
- [Stage 1 – Data filtering](#stage-1--data-filtering)
- [Stage 2 – Transparency detection models](#stage-2--transparency-detection-models)
- [Stage 3 – Topic clustering](#stage-3--topic-clustering)
- [Reproducibility notes](#reproducibility-notes)

---

## Overview

The pipeline consists of three main stages:

1. **Stage 1 – Data filtering**
   Filter raw app reviews by score/length and apply two keyword-filtering sets. The final labelled dataset is provided under `data/ground_truth/`.

2. **Stage 2 – Transparency detection**  
   Train and evaluate:
   - LLM-based classifiers (multiple prompts, multiple backends),
   - Deep learning models (BERT, DistilBERT, RoBERTa, ensemble),
   - Classical ML models (TF–IDF, USE, Word2Vec, and combinations).

3. **Stage 3 – Topic clustering**  
   Apply the best-performing RoBERTa model to the full dataset, run BERTopic on positive reviews, and perform LLM-based topic reduction.

---

## Project Structure

```text
TransparencyConcern-in-App-Reviews/
├── README.md
├── LICENSE
├── requirements.txt
├── config.example.py        # copy to config.py and edit for your environment
│
├── data/
│   ├── apps/                # app names used as clustering stopwords
│   ├── ground_truth/        # labelled data and annotations
│   └── keywords/            # keyword sets used to filter data
│
├── prompts/
│   ├── llm-experiment-prompt1.txt      # LLM experiments (ground truth, prompt 1)
│   ├── llm-experiment-prompt2.txt
│   ├── llm-experiment-prompt3.txt
│   ├── transparency-definition.txt
│   └── bertopic-reduction-prompt.txt   # LLM-based topic reduction (Stage 3)
│
├── src/
│   ├── stage1-data_curation/
│   │   ├── __init__.py
│   │   ├── config_stage1.py
│   │   ├── filter_by_score_length.py      # filter on rating & length
│   │   ├── filter_keywords_set1.py        # keyword set 1
│   │   └── filter_keywords_set2.py        # keyword set 2
│   │
│   ├── stage2_transparency_detection/
│   │   ├── __init__.py
│   │   ├── cleaning_reviews_ground_truth.py
│   │   ├── prompts/
│   │   │   ├── __init__.py
│   │   │   └── prompt_loader.py           # Python-side prompt loader
│   │   ├── backends/                      # LLM backends
│   │   │   ├── __init__.py
│   │   │   ├── azure_backend.py
│   │   │   ├── bedrock_backend.py
│   │   │   └── openai_backend.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── deep-learning/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── bert.py
│   │   │   │   ├── distilbert.py
│   │   │   │   ├── ensemble.py
│   │   │   │   └── roberta.py
│   │   │   ├── llms/
│   │   │   │   ├── __init__.py
│   │   │   │   └── llm_experiment.py      # unified LLM experiment script
│   │   │   ├── machine-learning/
│   │   │   │   ├── __init__.py
│   │   │   │   └── ml-experiments.py      # classical ML experiments
│   │
│   ├── stage3-topic_clustering/
│   │   ├── __init__.py
│   │   ├── roberta-predictions.py         # apply RoBERTa checkpoint to full dataset
│   │   └── clustering.py                  # BERTopic + LLM-based topic reduction
│
└── .gitignore
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/golnoushabaei/TransparencyConcern-in-App-Reviews.git
cd TransparencyConcern-in-App-Reviews
```

### Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

The ML code assumes `PyTorch`, `PyTorch Lightning`, `transformers`, `BERTopic`, `sentence-transformers`, `tensorflow-hub`, and `gensim` are installed via `requirements.txt`.

---

## Configuration

### Global config (Stage 2 and Stage 3 LLM usage)

Copy the example configuration:

```bash
cp config.example.py config.py
```

Edit `config.py` and set:

- Data and output paths in `FileConfig`.
- Azure / Bedrock / OpenAI credentials in `AzureConfig`, `BedrockConfig`, `OpenAIConfig`.
- Backend and runtime parameters in `RunConfig`:

```python
run = RunConfig(
    backend="azure",          # or "bedrock", "openai"
    sleep_seconds=0.05,
    max_completion_tokens=80,
)
```

For different experiments (prompts, outputs, backends), update:

- `files.input_path`
- `files.output_path`
- `prompt.template_name` (e.g. `llm-experiment-prompt1`, `llm-experiment-prompt2`, `llm-experiment-prompt3`)

### Stage 1 config

`src/stage1-data_curation/config_stage1.py` controls Stage 1 paths:

```python
raw_data: Path = Path("data/raw/RawData.csv")
filtered_score_len: Path = Path("data/intermediate/DataFilteredByScore.csv")
keywords_set1: Path = Path("data/keywords/keywords_set1.csv")
keywords_set2: Path = Path("data/keywords/keywords_set2.csv")
...
```

Adjust paths if your data lives elsewhere.

---

## Data

### Raw data

The raw app-review dataset is not included in this repository. Place it at `data/raw/RawData.csv` before running Stage 1. The repository's `data/README.md` currently contains no acquisition instructions.

Required columns:

- `score` (numeric rating, e.g. 1–5)
- `content` (review text)

### Keywords

- `data/keywords/keywords_set1.csv` – initial 195 keywords (one per row).
- `data/keywords/keywords_set2.csv` – refined 43 keywords (with a `Keyword` column).

### Ground truth and supporting data

- `data/ground_truth/ground-truth-final.csv` – labelled dataset used by the LLM experiments and topic clustering.
- `data/ground_truth/Cleaned-Ground-Truth-DL.xlsx` – cleaned dataset used by the deep-learning and machine-learning experiments.
- Other annotation and topic-analysis spreadsheets in `data/ground_truth/` are supporting research materials.
- `data/apps/Apps.xlsx` – app names used to build custom stopwords for topic clustering.

---

## Stage 1 – Data filtering

### Step 1: Filter by score and length

```bash
python -m src.stage1-data_curation.filter_by_score_length
```

**Input:** `data/raw/RawData.csv`  
**Output:** `data/intermediate/DataFilteredByScore.csv`

(Reviews with `score <= 3` and at least 5 tokens in `content`.)

### Step 2: Keyword filtering – set 1 (195 keywords)

```bash
python -m src.stage1-data_curation.filter_keywords_set1
```

Uses `data/keywords/keywords_set1.csv`.

Outputs:

- `data/intermediate/FilteredData_keywords_set1.csv`
- `results/stage1/Keyword_summary_set1.csv`

### Step 3: Keyword filtering – set 2 (43 keywords)

```bash
python -m src.stage1-data_curation.filter_keywords_set2
```

Uses `data/keywords/keywords_set2.csv`.

Outputs:

- `data/intermediate/FilteredData_keywords_set2.csv`
- `results/stage1/Keyword_summary_set2.csv`

---

## Stage 2 – Transparency detection models

### 2.1 Cleaning ground truth for DL/ML

```bash
python -m src.stage2_transparency_detection.cleaning_reviews_ground_truth
```

**Input:** `data/ground_truth/ground-truth-final.csv`
**Output:** `results/ground_truth/Cleaned-Ground-Truth-DL.xlsx`

Columns:

- `clean_review`
- `Label`

### 2.2 LLM experiments (LLM-based classifiers)

```bash
python -m src.stage2_transparency_detection.models.llms.llm_experiment
```

This unified script:

- Loads prompt from `prompts/<template_name>.txt`
- Uses backend from `config.py` (`azure`, `bedrock`, `openai`)
- Reads from `files.input_path` (default: `data/ground_truth/ground-truth-final.csv`)
- Writes predictions to `files.output_path` (e.g. `results/llm_exp1_azure.csv`)
- Computes accuracy, precision, recall, and F1 if ground truth is present.

To reproduce different LLM experiments:

- Change `prompt.template_name` (e.g. `llm-experiment-prompt1`, `...2`, `...3`)
- Change `run.backend` to `"azure"`, `"bedrock"`, or `"openai"`
- Adjust `files.output_path` for each run.

### 2.3 Deep learning models

Each DL script reads `data/ground_truth/Cleaned-Ground-Truth-DL.xlsx`
and performs K-fold cross-validation, saving best checkpoints via PyTorch Lightning.

Run from project root:

```bash
python -m src.stage2_transparency_detection.models.deep-learning.bert
python -m src.stage2_transparency_detection.models.deep-learning.distilbert
python -m src.stage2_transparency_detection.models.deep-learning.roberta
python -m src.stage2_transparency_detection.models.deep-learning.ensemble
```

The cleaning script writes its generated workbook under `results/`, while the model scripts read the committed workbook under `data/ground_truth/`. If you regenerate the workbook, copy it to the model input path or update the relevant paths in the scripts.

**Inputs:** `clean_review`, `Label`

Outputs:

- Fold-wise checkpoints (e.g. `bert_new_gt_fold_*.ckpt`, `roberta_new_gt_fold_*.ckpt`)
- Metrics printed to stdout (precision, recall, F1, accuracy)

### 2.4 Machine learning models

```bash
python -m src.stage2_transparency_detection.models.machine-learning.ml-experiments
```

This script runs K-fold experiments for:

#### Features

- TFIDF (char)
- USE
- Word2Vec
- TFIDF + USE
- TFIDF + Word2Vec
- USE + Word2Vec
- TFIDF + USE + Word2Vec

#### Models

- Logistic Regression
- SVM (linear)
- Random Forest

The script:

- Reads `data/ground_truth/Cleaned-Ground-Truth-DL.xlsx` (`clean_review`, `Label`)
- Prints average precision, recall, F1, and accuracy
- Saves best models as `Model-<name>.pkl` in the working directory.

---

## Stage 3 – Topic clustering

### 3.1 RoBERTa predictions on full dataset

```bash
python -m src.stage3-topic_clustering.roberta-predictions
```

Inputs:

- Fine-tuned RoBERTa checkpoint: set `CKPT_PATH` in `src/stage3-topic_clustering/roberta-predictions.py`
- Ground truth: `data/ground_truth/Cleaned-Ground-Truth-DL.xlsx`
- Cleaned full dataset parts: `data/intermediate/Final-Cleaned_Dataset_part*.csv`

Behaviour:

- Removes rows already in ground truth (by `clean_review`)
- Applies RoBERTa to all remaining reviews in chunks
- Saves chunk-level predictions in `results/stage3/roberta_chunks/predictions_chunk_*.csv`

Combines them into `results/stage3/final_predictions.csv` with columns:

- `row_id`
- `Review`
- `clean_review`
- `pred_label`
- `pred_prob`

### 3.2 Construct positive set for clustering

Within `clustering.py`, `load_positives()`:

Takes:

- Ground truth label 1 reviews (`Label == 1`)
- RoBERTa positives (`pred_label == 1` from `final_predictions.csv`)

Combines them into a single `positives_df` for topic modelling.

### 3.3 BERTopic clustering

```bash
python -m src.stage3-topic_clustering.clustering
```

This script:

- Loads positives (as above)
- Removes app names from text via custom stopwords (`data/apps/Apps.xlsx`)
- Embeds texts with `sentence-transformers/all-MiniLM-L6-v2`
- Runs BERTopic to discover topics:
  - Base model + reduction experiments (various `nr_topics` with silhouette scores)
  - UMAP + HDBSCAN tuned variant, reduced to 45 topics
- Stores assignments in `positives_df["reduced_topics_2_45"]`

### 3.4 LLM-based topic reduction (two passes)

`run_llm_reduction()` performs two LLM reductions using `bertopic-reduction-prompt.txt`:

#### First reduction – keywords only

- Extracts top keywords per topic
- Builds a prompt listing topics and keywords
- LLM returns merged themes and topic groupings (JSON)
- Used for initial analysis/inspection

#### Second reduction – keywords + examples

- Extracts keywords and example texts per topic
- Builds a richer prompt listing topics, keywords, and example snippets
- LLM returns a refined set of themes and topic groupings (JSON)

This second pass is treated as the final theme mapping.

Final output:

`results/stage3/topics_experiments.csv` with columns including:

- `clean_review`
- `reduced_topics_2_45`
- `llm_topic_id_2`
- `llm_topic_label_2` (final LLM-assigned theme labels)

plus any other columns carried from `positives_df`.

---

## Reproducibility notes

### LLM backends

The LLM experiments and LLM-based topic reduction share the same backend interface:

- Azure OpenAI (`azure_backend.py`)
- AWS Bedrock (`bedrock_backend.py`)
- OpenAI API (`openai_backend.py`)

Select the backend in `config.py` (`run.backend`).

### Checkpoints and large models

Fine-tuned RoBERTa checkpoints are not committed.

Users should either:

- Train their own using the provided DL scripts, or
- Download the authors’ checkpoint from an external model repository (if provided in the paper/supplementary materials) and set `CKPT_PATH` accordingly.

### Large data files

Raw/intermediate data and generated results are not included in the repository.

Supply the raw and intermediate datasets required by the scripts at the paths listed above to run the complete pipeline.

For any questions about running specific experiments or mapping code to paper tables/figures, see the paper’s supplementary material or open an issue in this repository.
