import os
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from skorch import NeuralNetClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

import joblib

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

# --- Globals / Configuration (adapt to environment) ---
MODEL_DIR = EXPERIMENT_MODEL_DIR
INPUT_DIR = RAW_DATA_DIR
N_JOBS = os.cpu_count() or 4

# Embedding / model defaults
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 256
PCA_COMPONENTS = 256           # set None to disable PCA
MLP_HIDDEN_DIM = 256
MLP_MAX_EPOCHS = 8             # adjust up for final training
MLP_LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Embeddings & PyTorch MLP (skorch wrapper)
# ------------------------
class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Uses SentenceTransformer to convert text list to float32 numpy embeddings.
    Implements batch_size-driven encoding to avoid OOMs.
    """
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, batch_size=EMBED_BATCH_SIZE, device=DEVICE):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.model = None

    def fit(self, X, y=None):
        # Initialize model lazily to avoid heavy memory usage in process creation
        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        # Accept Series, list, numpy array; produce float32 numpy embeddings
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values.ravel().tolist()
        else:
            X = list(X)
        # SentenceTransformer.encode handles batching internally, but we still pass batch_size
        with torch.no_grad():
            embeddings = self.model.encode(
                X,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
        return embeddings.astype(np.float32)


class MultiClassMLP(nn.Module):
    """
    Simple 2-layer MLP returning logits for CrossEntropyLoss.
    """
    def __init__(self, input_dim, hidden_dim=MLP_HIDDEN_DIM, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        return self.out(X)  # logits (no softmax)


def build_embedding_mlp_pipeline(input_dim=384, n_classes=2, pca_components=PCA_COMPONENTS,
                                 batch_size=EMBED_BATCH_SIZE, device=DEVICE,
                                 max_epochs=MLP_MAX_EPOCHS, lr=MLP_LR):
    """
    Build a sklearn Pipeline:
      - EmbeddingTransformer -> (Optional) StandardScaler(with_mean=False) -> PCA -> Skorch NeuralNetClassifier
    """
    steps = [("embed", EmbeddingTransformer(batch_size=batch_size, device=device))]
    if pca_components is not None:
        steps += [("scale", StandardScaler(with_mean=False)), ("pca", PCA(n_components=pca_components))]
        model_input_dim = pca_components
    else:
        model_input_dim = input_dim

    net = NeuralNetClassifier(
        module=MultiClassMLP,
        module__input_dim=model_input_dim,
        module__hidden_dim=MLP_HIDDEN_DIM,
        module__output_dim=n_classes,
        criterion=nn.CrossEntropyLoss,
        max_epochs=max_epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        iterator_train__shuffle=True,
        # keep deterministic-ish behaviour
        verbose=0
    )
    steps.append(("mlp", net))
    from sklearn.pipeline import Pipeline
    return Pipeline(steps)


# ------------------------
# Training / Evaluation helpers
# ------------------------
def safe_train_test_split(X, y, test_size=0.2, stratify=None, random_state=42):
    """
    A wrapper to handle small datasets for stratified split.
    """
    try:
        return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)
    except ValueError:
        # fallback without stratify (small classes / single-class in split)
        logging.warning("Stratified split failed, using non-stratified split.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_and_report(pipeline, X_test, y_test, label_encoder: LabelEncoder, report_path: Path):
    """
    Predict, print + save classification report + confusion matrix + accuracy.
    """
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    target_names = list(label_encoder.classes_)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    full_report = (
        f"Accuracy: {acc:.4f}\n\n"
        f"Classification Report:\n{report}\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    logging.info("Evaluation result:\n%s", full_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(full_report)
    return acc


def train_and_save_model(X, y, task_name: str, model_dir: Path = MODEL_DIR,
                         input_dim=384, pca_components=PCA_COMPONENTS,
                         batch_size=EMBED_BATCH_SIZE, device=DEVICE,
                         max_epochs=MLP_MAX_EPOCHS, lr=MLP_LR):
    """
    Train an embedding->MLP classifier for provided texts X and labels y (1D list-like).
    Saves: model.joblib, label_encoder.joblib and a report text file.
    Returns: (pipeline, label_encoder, acc)
    """
    logging.info("Training task '%s' samples=%d", task_name, len(X))
    if len(X) < 10:
        logging.warning("Not enough samples (%d) for task %s — skipping", len(X), task_name)
        return None, None, None

    # Encode labels into integer classes
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # train/test
    stratify = y_enc if len(np.unique(y_enc)) > 1 else None
    X_train, X_test, y_train, y_test = safe_train_test_split(
        X, y_enc, stratify=stratify
    )

    pipeline = build_embedding_mlp_pipeline(input_dim=input_dim, n_classes=len(le.classes_),
                                            pca_components=pca_components, batch_size=batch_size,
                                            device=device, max_epochs=max_epochs, lr=lr)

    start = time.time()
    logging.info("Fitting pipeline for %s (epochs=%s) ...", task_name, max_epochs)
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    logging.info("Finished training '%s' in %.2f s", task_name, elapsed)

    # Evaluate + save report
    report_path = model_dir / f"{task_name}_report.txt"
    acc = evaluate_and_report(pipeline, X_test, y_test, le, report_path)

    # Save pipeline & encoder
    model_path = model_dir / f"{task_name}_embedding_mlp.joblib"
    enc_path = model_dir / f"{task_name}_label_encoder.joblib"
    joblib.dump(pipeline, model_path)
    joblib.dump(le, enc_path)
    logging.info("Saved model '%s' -> %s and encoder -> %s", task_name, model_path, enc_path)

    return pipeline, le, acc


# ------------------------
# Top-level main
# ------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    total_start = time.time()
    logging.info("Starting pipeline (device=%s)", DEVICE)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    download_nltk_resources()

    # 1) Load or preprocess & create labelled parquet
    if LABELLED_PARQUET.exists():
        logging.info("Loading labelled data from %s", LABELLED_PARQUET)
        labelled_df = pd.read_parquet(LABELLED_PARQUET)
        logging.info("Loaded labelled rows: %d", len(labelled_df))
    else:
        # find files and process in parallel
        logging.info("Labelled parquet not found, searching INPUT_DIR=%s for csv.gz files", INPUT_DIR)
        files = find_input_files(INPUT_DIR, extension=".csv.gz")
        logging.info("Found %d compressed files to process", len(files))
        labelled_dfs = []
        unlabelled_dfs = []

        with ProcessPoolExecutor(max_workers=min(N_JOBS, 8)) as executor:
            futures = {executor.submit(load_preprocess_weak_label, f): f for f in files}
            for i, fut in enumerate(as_completed(futures)):
                fpath = futures[fut]
                try:
                    labelled, unlabelled = fut.result()
                    if labelled is not None and not labelled.empty:
                        labelled_dfs.append(labelled)
                    if unlabelled is not None and not unlabelled.empty:
                        unlabelled_dfs.append(unlabelled)
                except Exception as e:
                    logging.exception("Error processing %s: %s", fpath, e)
                if (i + 1) % 50 == 0:
                    logging.info("Processed %d/%d files", i + 1, len(files))

        if not labelled_dfs:
            logging.error("No labelled data produced during preprocessing. Exiting.")
            return

        labelled_df = pd.concat(labelled_dfs, ignore_index=True)
        labelled_df.to_parquet(LABELLED_PARQUET, index=False)
        logging.info("Saved labelled parquet (%d rows) -> %s", len(labelled_df), LABELLED_PARQUET)

        if unlabelled_dfs:
            unlabelled_df = pd.concat(unlabelled_dfs, ignore_index=True)
            unlabelled_df.to_parquet(UNLABELLED_PARQUET, index=False)
            logging.info("Saved unlabelled parquet (%d rows) -> %s", len(unlabelled_df), UNLABELLED_PARQUET)

    # 2) Clean & prepare datasets (only rows with both party & sentiment are in the labelled set)
    labelled_df = labelled_df.dropna(subset=["party", "sentiment"]).reset_index(drop=True)
    logging.info("After dropna, labelled rows: %d", len(labelled_df))
    if labelled_df.empty:
        logging.error("No labelled rows after dropna. Exiting.")
        return

    # Task A: Entity classification (party)
    X_entity = labelled_df["clean_text"].tolist()
    y_entity = labelled_df["party"].tolist()   # e.g., 'democrat' / 'republican'

    logging.info("Training entity classifier (party) with %d samples", len(X_entity))
    entity_model, entity_le, entity_acc = train_and_save_model(
        X_entity, y_entity, task_name="entity_party",
        model_dir=MODEL_DIR, input_dim=384, pca_components=PCA_COMPONENTS,
        batch_size=EMBED_BATCH_SIZE, device=DEVICE, max_epochs=MLP_MAX_EPOCHS, lr=MLP_LR
    )

    # Task B: Democrat sentiment - only use rows labelled 'democrat'
    dem_rows = labelled_df[labelled_df["party"] == "democrat"]
    if not dem_rows.empty:
        X_dem = dem_rows["clean_text"].tolist()
        y_dem = dem_rows["sentiment"].tolist()  # 'positive' / 'negative' only (others filtered by dropna)
        logging.info("Training democrat sentiment classifier with %d samples", len(X_dem))
        dem_model, dem_le, dem_acc = train_and_save_model(
            X_dem, y_dem, task_name="dem_sentiment",
            model_dir=MODEL_DIR, input_dim=384, pca_components=PCA_COMPONENTS,
            batch_size=EMBED_BATCH_SIZE, device=DEVICE, max_epochs=MLP_MAX_EPOCHS, lr=MLP_LR
        )
    else:
        logging.warning("No democrat-labelled rows found; skipping dem_sentiment training.")
        dem_model = dem_le = dem_acc = None

    # Task C: Republican sentiment - only use rows labelled 'republican'
    rep_rows = labelled_df[labelled_df["party"] == "republican"]
    if not rep_rows.empty:
        X_rep = rep_rows["clean_text"].tolist()
        y_rep = rep_rows["sentiment"].tolist()
        logging.info("Training republican sentiment classifier with %d samples", len(X_rep))
        rep_model, rep_le, rep_acc = train_and_save_model(
            X_rep, y_rep, task_name="rep_sentiment",
            model_dir=MODEL_DIR, input_dim=384, pca_components=PCA_COMPONENTS,
            batch_size=EMBED_BATCH_SIZE, device=DEVICE, max_epochs=MLP_MAX_EPOCHS, lr=MLP_LR
        )
    else:
        logging.warning("No republican-labelled rows found; skipping rep_sentiment training.")
        rep_model = rep_le = rep_acc = None

    logging.info("Total pipeline execution time: %.2f seconds", time.time() - total_start)


if __name__ == "__main__":
    main()
