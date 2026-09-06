"""Train the canonical TF-IDF election classifiers."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from x_processing.config import LABELLED_PARQUET, MODEL_DIR, RAW_DATA_DIR, UNLABELLED_PARQUET
from x_processing.preprocessing import (
    download_nltk_resources,
    find_input_files,
    load_preprocess_weak_label,
)

def train_and_save(X, y, model, name, task, max_features, n_splits=5):
    """Cross-validate, evaluate, refit, and save one classifier."""
    class_counts = pd.Series(y).value_counts()
    if class_counts.size < 2:
        raise ValueError(f"{task} requires at least two classes")
    if class_counts.min() < n_splits:
        raise ValueError(f"{task} needs at least {n_splits} examples in every class for CV")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
        ("clf", model),
    ])
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=folds,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        n_jobs=-1,
    )
    cv_summary = (
        f"{name} [{task}] Cross-Validation Results ({n_splits}-fold):\n"
        f"Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}\n"
        f"Precision: {cv_results['test_precision_macro'].mean():.4f} ± "
        f"{cv_results['test_precision_macro'].std():.4f}\n"
        f"Recall:    {cv_results['test_recall_macro'].mean():.4f} ± {cv_results['test_recall_macro'].std():.4f}\n"
        f"F1 Score:  {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}\n"
    )
    print(f"\n{cv_summary}")
    (MODEL_DIR / f"{name}_{task}_cv_results.txt").write_text(cv_summary, encoding="utf-8")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)
    evaluation = (
        f"Model: {name} [{task}]\nAccuracy: {accuracy:.4f}\n\n"
        f"Classification Report:\n{report}\nConfusion Matrix:\n{matrix}\n"
    )
    (MODEL_DIR / f"{name}_{task}_evaluation.txt").write_text(evaluation, encoding="utf-8")

    # The holdout fit above is for evaluation only; persist a model fitted to all data.
    pipeline.fit(X, y)
    model_path = MODEL_DIR / f"{name}_{task}_classifier.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model: {model_path}")
    return pipeline, cv_results


def _build_training_data():
    files = find_input_files(RAW_DATA_DIR)
    if not files:
        raise FileNotFoundError(f"No .csv.gz files found beneath {RAW_DATA_DIR}")

    labelled_parts = []
    unlabelled_parts = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(load_preprocess_weak_label, path) for path in files]
        for future in as_completed(futures):
            labelled, unlabelled = future.result()
            if labelled is not None and not labelled.empty:
                labelled_parts.append(labelled)
            if unlabelled is not None and not unlabelled.empty:
                unlabelled_parts.append(unlabelled)

    if not labelled_parts:
        raise RuntimeError("Preprocessing produced no labelled tweets")

    labelled = pd.concat(labelled_parts, ignore_index=True)
    unlabelled = pd.concat(unlabelled_parts, ignore_index=True) if unlabelled_parts else pd.DataFrame()
    labelled.to_parquet(LABELLED_PARQUET, index=False)
    unlabelled.to_parquet(UNLABELLED_PARQUET, index=False)
    return labelled


def main():
    start = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    download_nltk_resources()
    labelled = pd.read_parquet(LABELLED_PARQUET) if LABELLED_PARQUET.exists() else _build_training_data()

    train_and_save(
        labelled["clean_text"],
        labelled["party"],
        lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=30,
            num_leaves=63,
            random_state=42,
            force_row_wise=True,
        ),
        "LightGBM",
        "party",
        20_000,
    )

    for party in ("democrat", "republican"):
        subset = labelled[labelled["party"] == party]
        task = f"{party}_sentiment"
        train_and_save(
            subset["clean_text"],
            subset["sentiment"],
            LogisticRegression(max_iter=300, n_jobs=-1, random_state=42),
            "LogisticRegression",
            task,
            10_000,
        )
        train_and_save(
            subset["clean_text"],
            subset["sentiment"],
            LinearSVC(max_iter=3000, random_state=42),
            "LinearSVC",
            task,
            10_000,
        )

    print(f"\nTotal runtime: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
