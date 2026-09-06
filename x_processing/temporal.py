from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from x_processing.config import (
    FIGURE_DIR,
    OUTPUT_DIR,
    PREDICTIONS_PARQUET,
    UNLABELLED_PARQUET,
)
from x_processing.predictions import generate_predictions


INPUT_DATA = UNLABELLED_PARQUET

# Temporal Scope
ANALYSIS_START = datetime(2024, 5, 1)
ANALYSIS_END = datetime(2024, 11, 30, 23, 59, 59)

# Key Events
EVENTS = {
    "Trump Assassination Attempt": datetime(2024, 7, 13),
    "Biden Withdrawal": datetime(2024, 7, 21),
    "Harris Nomination": datetime(2024, 8, 6),
}


def generate_discourse_predictions(df):
    """Classify discourse alignment and tone using the shared inference path."""
    print(f"Generating predictions for {len(df):,} tweets...")
    df = generate_predictions(df).rename(
        columns={"party": "discourse_alignment"}
    )
    df.to_parquet(PREDICTIONS_PARQUET, index=False)
    return df


def analyze_event_impact(df, window_days=7):
    """Statistically evaluates if an event shifted the tone of party-aligned discourse."""
    results = []
    for event, date in EVENTS.items():
        for align in ["democrat", "republican"]:
            pre = df[
                (df["discourse_alignment"] == align)
                & (df["timestamp"] >= date - timedelta(days=window_days))
                & (df["timestamp"] < date)
            ]
            post = df[
                (df["discourse_alignment"] == align)
                & (df["timestamp"] >= date)
                & (df["timestamp"] <= date + timedelta(days=window_days))
            ]

            if len(pre) < 15 or len(post) < 15:
                continue

            # Chi-Square Test for Sentiment Distribution Shift
            contingency = pd.crosstab(
                ["pre"] * len(pre) + ["post"] * len(post),
                list(pre["sentiment"]) + list(post["sentiment"]),
            )
            _, p_val, _, _ = stats.chi2_contingency(contingency)

            results.append(
                {
                    "event": event,
                    "alignment": align,
                    "shift": (post["sentiment"] == "positive").mean()
                    - (pre["sentiment"] == "positive").mean(),
                    "significant": p_val < 0.05,
                }
            )
    return pd.DataFrame(results, columns=["event", "alignment", "shift", "significant"])


def plot_rolling_discourse(df):
    """Visualizes the 7-day rolling sentiment of party-aligned discourse."""
    daily = (
        df.assign(date=df["timestamp"].dt.date)
        .groupby(["date", "discourse_alignment"])
        .agg(pos_ratio=("sentiment", lambda values: (values == "positive").mean()))
        .reset_index()
    )

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"democrat": "#1f77b4", "republican": "#d62728"}

    for align in ["democrat", "republican"]:
        data = daily[daily["discourse_alignment"] == align].sort_values("date")
        rolling = data["pos_ratio"].rolling(7).mean()
        ax.plot(
            data["date"],
            rolling,
            label=f"{align.capitalize()}-Aligned Discourse",
            color=colors[align],
            lw=2.5,
        )

    for event, dt in EVENTS.items():
        ax.axvline(x=dt.date(), color="gray", ls="--", alpha=0.6)
        ax.text(dt.date(), ax.get_ylim()[1] * 0.95, f" {event}", rotation=90, size=9)

    ax.set_title("2024 Election: Sentiment of Party-Aligned Discourse", fontsize=15)
    ax.set_ylabel("Positive Sentiment Ratio (7-Day MA)")
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "discourse_sentiment.png", dpi=300)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if PREDICTIONS_PARQUET.exists():
        df = pd.read_parquet(PREDICTIONS_PARQUET)
        # predictions.py uses the older name for the same model output.
        if "discourse_alignment" not in df.columns and "party" in df.columns:
            df = df.rename(columns={"party": "discourse_alignment"})
    else:
        df = generate_discourse_predictions(pd.read_parquet(INPUT_DATA))

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[
        (df["timestamp"] >= ANALYSIS_START) & (df["timestamp"] <= ANALYSIS_END)
    ]

    # Run Analyses
    event_impact = analyze_event_impact(df)
    event_impact.to_csv(OUTPUT_DIR / "event_discourse_impact.csv", index=False)

    plot_rolling_discourse(df)
    print("Pipeline Updated: Terminology now reflects 'Aligned Discourse'.")


if __name__ == "__main__":
    main()
