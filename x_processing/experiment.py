import os
import time
import pandas as pd
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
import joblib
import torch

from x_processing.config import (
    EXPERIMENT_MODEL_DIR,
    LABELLED_PARQUET,
    RAW_DATA_DIR,
    UNLABELLED_PARQUET,
)
from x_processing.preprocessing import (
    download_nltk_resources,
    find_input_files,
    load_preprocess_weak_label,
)


MODEL_DIR = str(EXPERIMENT_MODEL_DIR)
labelled_parquet = str(LABELLED_PARQUET)
unlabelled_parquet = str(UNLABELLED_PARQUET)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#  Evaluation 
def evaluate_model(pipeline, X_test, y_test, model_name, output_path=None):
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = (
        f"\n{model_name}\n"
        f"Accuracy: {acc:.4f}\n"
        f"Classification Report:\n{classification_report(y_test, y_pred)}\n"
        f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n"
    )
    print(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return acc

#  Transformers 
class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=256, device=DEVICE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values.ravel().tolist()
        else:
            X = list(X)

        with torch.no_grad():
            embeddings = self.model.encode(
                X,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
        return embeddings

# ---------------- Model Candidates ----------------
def get_model_candidates(for_embeddings=False):
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=500, solver="saga", n_jobs=-1, random_state=42
        ),
        "LinearSVC": LinearSVC(
            max_iter=3000, random_state=42
        ),
        "SGDClassifier": SGDClassifier(
            max_iter=1000, tol=1e-3, loss='hinge', penalty='l2', n_jobs=-1, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_features='sqrt', max_depth=12, n_jobs=-1, random_state=42
        ),
        "MultinomialNB": MultinomialNB(alpha=1.0),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=12, num_leaves=31, learning_rate=0.05,
            n_jobs=-1, random_state=42, force_col_wise=True
        )
    }

    if for_embeddings:
        candidates.pop("LinearSVC", None)
        candidates.pop("MultinomialNB", None)
    return candidates

#  Pipeline Builder 
def build_pipeline(clf, feature_mode="tfidf", max_features=20000, batch_size=256, pca_components=256):
    if feature_mode == "tfidf":
        features = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    elif feature_mode == "embeddings":
        features = Pipeline([
            ('embed', EmbeddingTransformer(batch_size=batch_size)),
            ('scale', StandardScaler(with_mean=False)),
            ('pca', PCA(n_components=pca_components))
        ])
    else:
        raise ValueError("Invalid feature_mode. Choose from ['tfidf', 'embeddings'].")

    pipeline = Pipeline([
        ('features', features),
        ('clf', clf)
    ])
    return pipeline

#  Training 
def train_and_save_top_models(X, y, model_candidates, task_name, feature_mode='tfidf',
                              tfidf_max_features=20000, test_size=0.2, output_file=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"X_train: {len(X_train)}, y_train: {len(y_train)}")
    print(f"X_test: {len(X_test)}, y_test: {len(y_test)}")

    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Distribution report for {task_name}\n\n"
                    f"X_train: {len(X_train)}, y_train: {len(y_train)}\n"
                    f"X_test: {len(X_test)}, y_test: {len(y_test)}\n\n")

    def check_distribution(name, labels):
        counts = Counter(labels)
        total = sum(counts.values())
        dist = {cls: f"{count} ({count / total:.2%})" for cls, count in counts.items()}
        report_line = f"{task_name} - {name} distribution: {dist}"
        print(report_line)
        if output_file:
            with open(output_file, 'a') as f:
                f.write(report_line + "\n")

    check_distribution("Train", y_train)
    check_distribution("Test", y_test)
    check_distribution("Full", y)

    results = []
    print(f"\nTraining models for {task_name} using feature mode: {feature_mode}")

    for name, clf in model_candidates.items():
        print(f"\n{name} for {task_name} ({feature_mode})")
        start_time = time.time()
        pipeline = build_pipeline(clf, feature_mode=feature_mode, max_features=tfidf_max_features)
        pipeline.fit(X_train, y_train)

        report_path = os.path.join(MODEL_DIR, f"{task_name}_{name}_{feature_mode}_report.txt")
        acc = evaluate_model(pipeline, X_test, y_test, f"{task_name} ({name}, {feature_mode})", output_path=report_path)
        print(f"Training + evaluation time: {time.time() - start_time:.2f} seconds")

        results.append((name, pipeline, acc))

    results.sort(key=lambda x: x[2], reverse=True)

    for rank, (name, model, acc) in enumerate(results[:2], start=1):
        path = os.path.join(MODEL_DIR, f"top{rank}_{name}_{task_name}_{feature_mode}_classifier.joblib")
        joblib.dump(model, path)
        print(f"Saved top{rank} model: {path} (Accuracy: {acc:.4f})")

    return results

#  Main 
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    download_nltk_resources()
    total_start = time.time()

    # Load or preprocess data
    start = time.time()
    if os.path.exists(labelled_parquet):
        print(f"Loading labelled data from {labelled_parquet}")
        labelled_df = pd.read_parquet(labelled_parquet)
        print(f"Total rows in labelled data is {len(labelled_df)}")
    else:
        print("Preprocessing raw data...")
        all_files = find_input_files(RAW_DATA_DIR)
        print(f"Found {len(all_files)} files")

        labelled_dfs, unlabelled_dfs = [], []
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            futures = {executor.submit(load_preprocess_weak_label, file): file for file in all_files}
            for i, future in enumerate(as_completed(futures)):
                file = futures[future]
                try:
                    labelled, unlabelled = future.result()
                    if labelled is not None:
                        labelled_dfs.append(labelled)
                    if unlabelled is not None and not unlabelled.empty:
                        unlabelled_dfs.append(unlabelled)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

                if i % 50 == 0:
                    print(f"Processed {i}/{len(all_files)} files")

        if not labelled_dfs:
            print("No labelled data found, exiting.")
            return

        labelled_df = pd.concat(labelled_dfs, ignore_index=True)
        labelled_df.to_parquet(labelled_parquet, index=False)
        print(f"Labelled data saved to {labelled_parquet}")
        print(f"Total rows in labelled data is {len(labelled_df)}")

        if unlabelled_dfs:
            unlabelled_df = pd.concat(unlabelled_dfs, ignore_index=True)
            unlabelled_df.to_parquet(unlabelled_parquet, index=False)
            print(f"Unlabelled data saved to {unlabelled_parquet}")
            print(f"Total rows in unlabelled data is {len(unlabelled_df)}")

        print(f"Data loading / preprocessing took {time.time() - start:.2f} seconds")

    # Prepare datasets
    labelled_df.dropna(subset=['party', 'sentiment'], inplace=True)
    labelled_df.reset_index(drop=True, inplace=True)

    X_party = labelled_df['clean_text']
    y_party = labelled_df['party']

    for feature_mode in ['tfidf', 'embeddings']:
        print(f"\nTraining party classifiers with feature mode: {feature_mode}")
        model_candidates = get_model_candidates(for_embeddings=(feature_mode == "embeddings"))
        train_and_save_top_models(
            X_party, y_party, model_candidates,
            task_name="party", feature_mode=feature_mode,
            output_file=os.path.join(MODEL_DIR, f"party_distribution.txt")
        )

    # Sentiment Classification per Party
    for party in ['democrat', 'republican']:
        party_data = labelled_df[labelled_df['party'] == party]
        X_sent = party_data['clean_text']
        y_sent = party_data['sentiment']

        for feature_mode in ['tfidf', 'embeddings']:
            print(f"\nTraining sentiment classifiers for {party} with feature mode: {feature_mode}")
            model_candidates = get_model_candidates(for_embeddings=(feature_mode == "embeddings"))
            train_and_save_top_models(
                X_sent, y_sent, model_candidates,
                task_name=f"{party}_sentiment", feature_mode=feature_mode,
                output_file=os.path.join(MODEL_DIR, f"{party}_sentiment_distribution.txt")
            )

    print(f"\nTotal pipeline execution took {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
