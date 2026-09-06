import time

import joblib
import pandas as pd

from x_processing.config import (
    CAMPAIGN_START_DATE,
    MODEL_DIR,
    PREDICTIONS_PARQUET,
    UNLABELLED_PARQUET,
)
from x_processing.preprocessing import convert_to_timestamp


OUTPUT_PARQUET = PREDICTIONS_PARQUET

CHUNK_SIZE = 500_000


def filter_campaign_period(df):
    """Add timestamps when needed and retain the campaign period."""
    if "timestamp" not in df.columns:
        df = convert_to_timestamp(df)
    return df.loc[df["timestamp"] >= CAMPAIGN_START_DATE].copy()


def load_model(classifier, party=None):
    """Load a trained classifier by role."""
    if classifier == "party":
        filename = "LightGBM_party_classifier.joblib"
    elif classifier == "sentiment":
        if not party:
            raise ValueError("Party must be specified for sentiment classifier")
        filename = f"LinearSVC_{party.lower()}_sentiment_classifier.joblib"
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    model_path = MODEL_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loaded: {filename}")
    return joblib.load(model_path)


def chunked_predict(model, texts, chunk_size=CHUNK_SIZE, label=""):
    """Predict in chunks with progress tracking."""
    n = len(texts)
    results = []
    start = time.time()

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        results.extend(model.predict(texts[i:end]))

        chunk_num = (i // chunk_size) + 1
        total_chunks = (n + chunk_size - 1) // chunk_size
        elapsed = time.time() - start

        print(
            f"[{label}] {chunk_num}/{total_chunks} chunks "
            f"({end:,}/{n:,} rows, {elapsed:.1f}s elapsed)"
        )

    return results


def print_summary(df):
    """Print a sentiment summary and save it to a text file."""
    summary_file = MODEL_DIR.parent / "sentiment_summary.txt"
    summary_lines = []
    party_column = "party" if "party" in df.columns else "discourse_alignment"

    for party in ["democrat", "republican"]:
        mask = df[party_column] == party
        total = mask.sum()
        positive = (df.loc[mask, "sentiment"] == "positive").sum()
        negative = (df.loc[mask, "sentiment"] == "negative").sum()
        pos_ratio = positive / total if total > 0 else 0
        neg_ratio = negative / total if total > 0 else 0

        # Print to console
        print(f"\n{party.capitalize()}:")
        print(f"  Total: {total:,}")
        print(f"  Positive: {positive:,} ({pos_ratio:.2%})")
        print(f"  Negative: {negative:,} ({neg_ratio:.2%})")
        if unclassified := total - positive - negative:
            print(f"  Unclassified: {unclassified:,} ({unclassified / total:.2%})")

        # Prepare lines to write to file
        summary_lines.append(f"{party.capitalize()}:\n")
        summary_lines.append(f"  Positive sentiment: {positive}/{total} ({pos_ratio:.2%})\n")
        summary_lines.append(f"  Negative sentiment: {negative}/{total} ({neg_ratio:.2%})\n")
        if unclassified:
            summary_lines.append(
                f"  Unclassified: {unclassified}/{total} "
                f"({unclassified / total:.2%})\n"
            )
        summary_lines.append("\n")

    # Save to file
    summary_file.write_text("".join(summary_lines), encoding="utf-8")

    print(f"\nSentiment summary saved to {summary_file}")


def generate_predictions(df):
    """Classify party alignment and party-specific sentiment for a dataframe."""
    party_model = load_model("party")
    sentiment_models = {
        "democrat": load_model("sentiment", party="democrat"),
        "republican": load_model("sentiment", party="republican"),
    }
    df = filter_campaign_period(df)
    df["party"] = chunked_predict(
        party_model, df["clean_text"].tolist(), label="Party"
    )

    for party, model in sentiment_models.items():
        mask = df["party"] == party
        if mask.any():
            df.loc[mask, "sentiment"] = chunked_predict(
                model,
                df.loc[mask, "clean_text"].tolist(),
                label=party.capitalize(),
            )
    return df


def main():
    start_total = time.time()

    # Check if predictions already exist
    if OUTPUT_PARQUET.exists():
        print(f"\nLoading existing predictions from {OUTPUT_PARQUET}")
        df = pd.read_parquet(OUTPUT_PARQUET)
        print(f"Loaded {len(df):,} tweets")
        print_summary(df)
        print(f"\nTotal workflow completed in {time.time() - start_total:.2f} seconds")
    else:
        # Load and filter data
        print(f"\nLoading data from {UNLABELLED_PARQUET}")
        df = pd.read_parquet(UNLABELLED_PARQUET)
        print(f"  Loaded {len(df):,} tweets")
        df = generate_predictions(df)

        print("\nParty distribution:")
        for party, count in df["party"].value_counts().items():
            print(f"{party.capitalize()}: {count:,} ({count/len(df):.1%})")

        # Save results
        print(f"\nSaving to {OUTPUT_PARQUET}")
        df.to_parquet(OUTPUT_PARQUET, index=False)
        print("Saved")

        # Print summary
        print_summary(df)

    print(f"\nTotal workflow completed in {time.time() - start_total:.2f} seconds")


if __name__ == "__main__":
    main()
