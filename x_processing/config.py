"""Paths and constants shared by the processing pipelines."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "x_processing"
RAW_DATA_DIR = PROJECT_ROOT / "x-24-us-election"
MODEL_DIR = DATA_DIR / "models"
EXPERIMENT_MODEL_DIR = MODEL_DIR / "experiments"
OUTPUT_DIR = DATA_DIR / "outputs"
FIGURE_DIR = DATA_DIR / "figures"

LABELLED_PARQUET = DATA_DIR / "train_labelled.parquet"
UNLABELLED_PARQUET = DATA_DIR / "train_unlabelled.parquet"
PREDICTIONS_PARQUET = DATA_DIR / "predictions_with_timestamps.parquet"

CAMPAIGN_START_DATE = pd.Timestamp("2024-05-01")
STRONG_SENTIMENT_THRESHOLD = 0.8

CANDIDATE_KEYWORDS = {
    "democrat": [
        "#bidenharris2024",
        "#kamalaharris2024",
        "@joebiden",
        "@kamalaharris",
        "democrats",
    ],
    "republican": [
        "#maga",
        "republican",
        "#trump2024",
        "@realdonaldtrump",
    ],
}
