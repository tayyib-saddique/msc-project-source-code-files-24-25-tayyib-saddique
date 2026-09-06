import time
import joblib
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import lightgbm as lgb

from x_processing.config import EXPERIMENT_MODEL_DIR, LABELLED_PARQUET

# CONFIG
MODEL_DIR = EXPERIMENT_MODEL_DIR
FINE_TUNED_DIR = MODEL_DIR / "fine_tuned"
ALL_REPORTS_DIR = FINE_TUNED_DIR / "reports"

MODEL_N_ITER = {
    'LogisticRegression': 30,
    'LinearSVC': 30,
    'LightGBM': 20
}

labelled_parquet = LABELLED_PARQUET

# FUNCTIONS
def get_model_candidates(party=False):
    if party:
        return ['LightGBM', 'LinearSVC']
    return ['LinearSVC', 'LogisticRegression']

def hyperparameter_tune_pipeline(X_tfidf, y_train, model_type, n_iter=None, cv_folds=2, n_jobs=8):
    if n_iter is None:
        n_iter = MODEL_N_ITER.get(model_type, 10)

    # Classifier + parameter distributions
    if model_type == 'LogisticRegression':
        clf = LogisticRegression(max_iter=500, random_state=42)
        param_distributions = {'C': scipy.stats.loguniform(0.01, 100)}

    elif model_type == "LinearSVC":
        clf = LinearSVC(max_iter=3000, random_state=42)
        param_distributions = {'C': scipy.stats.loguniform(0.01, 100),
                               'loss': ['hinge', 'squared_hinge']}

    elif model_type == "LightGBM":
        clf = lgb.LGBMClassifier(n_jobs=8, force_row_wise=True, random_state=42)
        param_distributions = {'num_leaves': [31, 63],
                               'learning_rate': [0.01, 0.05, 0.1],
                               'n_estimators': [100, 200, 500],
                               'max_depth': [-1, 30]}
    else:
        raise ValueError("Unsupported model_type")

    random_search = RandomizedSearchCV(
        clf,
        param_distributions,
        n_iter=n_iter,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=n_jobs,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_tfidf, y_train)

    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_

def train_and_save_top_models(X, y, model_candidates, task_name, output_file, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_acc = 0
    best_model_info = None

    for model_type in model_candidates:
        print(f"\nTraining {model_type} for task {task_name} with {n_splits}-fold CV")
        fold_accs = []
        fold_estimators = []
        fold_params = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # Fit text features inside each fold to prevent validation leakage.
            vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
            X_train_text, X_val_text = X.iloc[train_idx], X.iloc[val_idx]
            X_train = vectorizer.fit_transform(X_train_text)
            X_val = vectorizer.transform(X_val_text)
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Sample a subset for hyperparameter tuning
            sample_frac = 0.2
            class_count = y_train.nunique()
            n_sample = max(class_count * 2, int(X_train.shape[0] * sample_frac))
            n_sample = min(n_sample, X_train.shape[0])
            if n_sample < X_train.shape[0]:
                sampler = StratifiedShuffleSplit(
                    n_splits=1, train_size=n_sample, random_state=42 + fold_idx
                )
                sample_idx, _ = next(sampler.split(X_train, y_train))
            else:
                sample_idx = np.arange(X_train.shape[0])
            X_sample = X_train[sample_idx]
            y_sample = y_train.iloc[sample_idx]

            best_estimator, best_params = hyperparameter_tune_pipeline(
                X_sample, y_sample, model_type
            )

            # Retrain on fold training data
            best_estimator.fit(X_train, y_train)
            y_pred = best_estimator.predict(X_val)
            fold_acc = accuracy_score(y_val, y_pred)
            fold_accs.append(fold_acc)
            fold_estimators.append(best_estimator)
            fold_params.append(best_params)
            print(f"Fold {fold_idx+1} accuracy: {fold_acc:.4f}")

        avg_acc = np.mean(fold_accs)
        print(f"{model_type} average CV accuracy: {avg_acc:.4f}")

        # Save report
        model_report_path = ALL_REPORTS_DIR / f"{task_name}_{model_type}_report.txt"
        with open(model_report_path, 'w') as f:
            f.write(f"Task: {task_name}\nModel: {model_type}\nAverage CV Accuracy: {avg_acc:.4f}\n\n")
            f.write("Best Parameters:\n")
            selected_fold = int(np.argmax(fold_accs))
            selected_params = fold_params[selected_fold]
            for k, v in selected_params.items():
                f.write(f"{k}: {v}\n")
        print(f"Saved report for {model_type} at {model_report_path}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_model_info = {
                'model': Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))),
                    ("clf", fold_estimators[selected_fold]),
                ]),
                'model_type': model_type,
                'accuracy': avg_acc,
                'best_params': selected_params
            }

    # Retrain best model on full data
    if best_model_info:
        best_model_info['model'].fit(X, y)
        save_path = FINE_TUNED_DIR / f"{best_model_info['model_type']}_{task_name}_best_model.joblib"
        joblib.dump(best_model_info['model'], save_path)
        print(
            f"\nSaved best model ({best_model_info['model_type']}) with accuracy "
            f"{best_model_info['accuracy']:.4f} to {save_path}"
        )

        with open(output_file, 'w') as f:
            f.write(f"{best_model_info['model_type']}: {best_model_info['accuracy']:.4f} -> {save_path}\n")

# MAIN
def main():
    total_start = time.time()
    ALL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading labelled data from {labelled_parquet}")
    labelled_df = pd.read_parquet(labelled_parquet)
    print(f"Total rows in labelled data: {len(labelled_df)}")

    # Party Classification
    X_party = labelled_df['clean_text']
    y_party = labelled_df['party']
    print("\nTraining party classifiers")
    train_and_save_top_models(
        X_party, y_party, get_model_candidates(party=True),
        task_name="party",
        output_file=MODEL_DIR / "party_distribution.txt",
        n_splits=3
    )

    # Sentiment Classification per Party
    for party in ['democrat', 'republican']:
        party_data = labelled_df[labelled_df['party'] == party]
        X_sent = party_data['clean_text']
        y_sent = party_data['sentiment']
        print(f"\nTraining sentiment classifiers for {party}")
        train_and_save_top_models(
            X_sent, y_sent, get_model_candidates(party=False),
            task_name=f"{party}_sentiment",
            output_file=MODEL_DIR / f"{party}_sentiment_distribution.txt",
            n_splits=3
        )

    print(f"\nTotal pipeline execution took {time.time() - total_start:.2f} seconds")

if __name__ == "__main__":
    main()
