"""
train_model.py — Health Setu Symptom Classifier Training Script
---------------------------------------------------------------
Trains a Bernoulli Naive Bayes classifier (pure numpy, no scipy/scikit-learn)
on the educational dataset at data/symptoms_dataset.csv.

The trained model is serialised to JSON at model/symptom_model.json.
The feature column list is saved to model/features.json.

IMPORTANT: This is a DEMONSTRATION / EDUCATIONAL model only.
It must NOT be used as a substitute for professional medical advice.
"""

import os
import json
import numpy as np
import pandas as pd

from classifier import BernoulliNaiveBayes

DATA_PATH    = os.path.join("data", "symptoms_dataset.csv")
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "symptom_model.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")


def evaluate(clf, X_test, y_test):
    """Print a simple per-class accuracy summary."""
    y_pred = clf.predict(X_test)
    classes = sorted(set(y_test))
    correct = sum(p == t for p, t in zip(y_pred, y_test))
    overall = correct / len(y_test) if len(y_test) > 0 else 0

    print(f"\nOverall accuracy on test split: {correct}/{len(y_test)} = {overall:.1%}\n")
    print(f"{'Condition':<35}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}")
    print("-" * 60)
    for cls in classes:
        idxs = [i for i, y in enumerate(y_test) if y == cls]
        correct_cls = sum(1 for i in idxs if y_pred[i] == cls)
        total_cls   = len(idxs)
        acc_cls     = correct_cls / total_cls if total_cls else 0
        print(f"{cls:<35}  {correct_cls:>7}  {total_cls:>5}  {acc_cls:>6.1%}")
    print()


def train():
    print(f"Loading dataset from {DATA_PATH} …")
    df = pd.read_csv(DATA_PATH)

    feature_cols = [c for c in df.columns if c != "condition"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["condition"].to_numpy()

    # Simple train/test split (last 20% as test)
    n_test = max(1, int(len(X) * 0.2))
    # Shuffle deterministically
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    train_idx, test_idx = idx[n_test:], idx[:n_test]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"Training on {len(X_train)} samples, evaluating on {len(X_test)} samples …")

    clf = BernoulliNaiveBayes(alpha=1.0)
    clf.fit(X_train, y_train)

    evaluate(clf, X_test, y_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    clf.save(MODEL_PATH)
    with open(FEATURES_PATH, "w", encoding="utf-8") as fh:
        json.dump(feature_cols, fh)

    print(f"Model saved    : {MODEL_PATH}")
    print(f"Features saved : {FEATURES_PATH}")


if __name__ == "__main__":
    train()
