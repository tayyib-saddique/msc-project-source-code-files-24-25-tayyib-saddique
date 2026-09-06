# MSc research project: 2024 US election discourse

This repository contains the data-processing and modelling work for analysing
party-aligned discourse and sentiment in tweets from the 2024 US election.

## Repository layout

- `x_processing/run.py` — canonical TF-IDF training pipeline.
- `x_processing/predictions.py` — batch inference over unlabelled tweets.
- `x_processing/temporal.py` — event-impact analysis and rolling-sentiment plot.
- `x_processing/experiment.py` — alternative model and embedding experiments.
- `x_processing/experiment_mlp.py` — sentence-embedding MLP experiment.
- `x_processing/finetune.py` — hyperparameter search for shortlisted models.
- `x_processing/preprocessing.py` — shared preprocessing and weak labelling.
- `x_processing/config.py` — paths and project-wide constants.
- `x_processing/*.ipynb` — exploratory and report-oriented analyses.
- `x_processing/models`, `outputs`, and `figures` — committed research results.
- `x-24-us-election` — raw-data Git submodule.

The raw dataset is large and maintained separately. Initialise it after cloning:

```bash
git submodule update --init --recursive
```

## Setup

Python 3.10 or 3.11 is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m nltk.downloader stopwords wordnet
```

## Running the pipeline

Run modules from the repository root so imports and output paths are stable:

```bash
python3 -m x_processing.run
python3 -m x_processing.predictions
python3 -m x_processing.temporal
```

Generated Parquet files and Joblib models are intentionally ignored. Text
reports, figures, and aggregate CSV outputs are retained as reproducible
research artefacts.

## Data source

The project uses the USC X 24 US Election dataset. Follow its licensing and
citation requirements when using or redistributing derived results:
https://github.com/sinking8/usc-x-24-us-election. 

Data from the USC X 24 US Election dataset (CC BY-NC-SA 4.0). 
Cite: Balasubramanian et al., arXiv:2411.00376.
