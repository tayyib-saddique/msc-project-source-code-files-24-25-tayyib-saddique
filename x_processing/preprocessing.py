"""Shared text preprocessing and weak-labelling utilities."""

from functools import lru_cache
import logging
from pathlib import Path
import re

import emoji
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from x_processing.config import (
    CAMPAIGN_START_DATE,
    CANDIDATE_KEYWORDS,
    STRONG_SENTIMENT_THRESHOLD,
)


LOGGER = logging.getLogger(__name__)
RAW_COLUMNS = ["id", "rawContent", "lang", "date", "epoch"]
LABELLED_COLUMNS = [
    "clean_text",
    "date",
    "timestamp",
    "party",
    "sentiment",
    "sentiment_score",
]
UNLABELLED_COLUMNS = ["id", "clean_text", "text", "date", "timestamp"]


def download_nltk_resources() -> None:
    """Download the corpora used by preprocessing, if they are not present."""
    for resource in ("stopwords", "wordnet"):
        nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def _text_tools():
    """Load NLP helpers lazily so importing this module has no network side effects."""
    try:
        words = set(stopwords.words("english"))
    except LookupError as exc:
        raise RuntimeError(
            "Missing NLTK data. Run `python -m nltk.downloader stopwords wordnet`."
        ) from exc
    return WordNetLemmatizer(), words, SentimentIntensityAnalyzer()


def convert_to_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Add a timestamp parsed from ``date``, falling back to Unix ``epoch``."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    if "epoch" in df.columns:
        missing = df["timestamp"].isna() & df["epoch"].notna()
        df.loc[missing, "timestamp"] = pd.to_datetime(
            df.loc[missing, "epoch"], unit="s", errors="coerce"
        )
    return df


def filter_campaign_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Return tweets on or after the configured campaign start date."""
    with_timestamps = convert_to_timestamp(df)
    return with_timestamps.loc[
        with_timestamps["timestamp"] >= CAMPAIGN_START_DATE
    ].copy()


def preprocess(text: str) -> str:
    """Normalize a tweet for text-feature extraction."""
    lemmatizer, stop_words, _ = _text_tools()
    normalized = emoji.demojize((text or "").lower(), delimiters=(" ", " "))
    normalized = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s#']", " ", normalized)
    # preserve_line avoids requiring Punkt merely to tokenize a single tweet.
    tokens = word_tokenize(normalized, preserve_line=True)
    return " ".join(
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    )


def detect_party(text: str):
    """Weakly label the party mentioned in a tweet, or return ``None``."""
    normalized = (text or "").lower()
    for party, keywords in CANDIDATE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return party
    return None


def sentiment_score(text: str) -> float:
    """Return VADER's compound score for a tweet."""
    _, _, analyzer = _text_tools()
    return analyzer.polarity_scores(text or "")["compound"]


def load_preprocess_weak_label(file_path):
    """Read one compressed export and split it into labelled/unlabelled rows."""
    try:
        df = pd.read_csv(
            file_path,
            compression="gzip",
            usecols=RAW_COLUMNS,
            dtype={
                "id": str,
                "rawContent": str,
                "lang": str,
                "date": str,
                "epoch": object,
            },
        )
        df = filter_campaign_tweets(df.loc[df["lang"] == "en"])
        if df.empty:
            return None, None

        df = df.rename(columns={"rawContent": "text"})
        df["clean_text"] = df["text"].apply(preprocess)
        df["party"] = df["text"].apply(detect_party)
        df["sentiment_score"] = df["text"].apply(sentiment_score)
        df["sentiment"] = df["sentiment_score"].map(
            lambda score: (
                "positive"
                if score >= STRONG_SENTIMENT_THRESHOLD
                else "negative"
                if score <= -STRONG_SENTIMENT_THRESHOLD
                else None
            )
        )

        labelled = df.dropna(subset=["party", "sentiment"])
        unlabelled = df[df["party"].isna() | df["sentiment"].isna()]
        return labelled[LABELLED_COLUMNS], unlabelled[UNLABELLED_COLUMNS]
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        LOGGER.error("Could not process %s: %s", file_path, exc)
        return None, None


def find_input_files(directory, extension: str = ".csv.gz") -> list[Path]:
    """Return matching input files recursively in deterministic order."""
    return sorted(Path(directory).rglob(f"*{extension}"))
