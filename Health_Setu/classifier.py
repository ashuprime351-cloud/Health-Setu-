"""
classifier.py — Pure-numpy Naive Bayes classifier for Health Setu
------------------------------------------------------------------
Implements a Bernoulli Naive Bayes classifier from scratch using only
numpy and the Python standard library — no scipy, no scikit-learn.

Bernoulli Naive Bayes is well-suited to binary (0/1) symptom features.

For educational / demonstration use only.
"""

import json
import math
import numpy as np


class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes for binary feature vectors.

    Parameters
    ----------
    alpha : float
        Laplace smoothing parameter (default 1.0).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_ = None          # list of class labels
        self.log_priors_ = None       # log P(class)  — shape (n_classes,)
        self.log_probs_ = None        # log P(feat=1|class) — (n_classes, n_features)
        self.log_neg_probs_ = None    # log P(feat=0|class) — (n_classes, n_features)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNaiveBayes":
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = list(np.unique(y))
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        n_samples = len(y)

        log_priors = np.zeros(n_classes)
        log_probs  = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes_):
            mask = y == cls
            n_cls = mask.sum()
            log_priors[i] = math.log(n_cls / n_samples)

            # Laplace-smoothed feature probabilities
            count_feat = X[mask].sum(axis=0)
            prob_feat = (count_feat + self.alpha) / (n_cls + 2 * self.alpha)
            log_probs[i] = np.log(prob_feat)

        self.log_priors_ = log_priors
        self.log_probs_     = log_probs
        self.log_neg_probs_ = np.log(1.0 - np.exp(log_probs))
        return self

    # ------------------------------------------------------------------
    def _log_joint(self, X: np.ndarray) -> np.ndarray:
        """Return log-joint scores for each sample and class."""
        X = np.array(X, dtype=float)
        # log P(x|c) = x·log p + (1-x)·log(1-p)
        scores = (
            X @ self.log_probs_.T
            + (1 - X) @ self.log_neg_probs_.T
            + self.log_priors_
        )
        return scores  # (n_samples, n_classes)

    def predict(self, X: np.ndarray) -> list:
        scores = self._log_joint(X)
        indices = np.argmax(scores, axis=1)
        return [self.classes_[i] for i in indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalised class probabilities using log-sum-exp trick."""
        scores = self._log_joint(X)
        # Log-sum-exp normalisation
        max_scores = scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return proba

    # ------------------------------------------------------------------
    # Serialisation  (pure JSON — no joblib/pickle needed)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "alpha":          self.alpha,
            "classes":        self.classes_,
            "log_priors":     self.log_priors_.tolist(),
            "log_probs":      self.log_probs_.tolist(),
            "log_neg_probs":  self.log_neg_probs_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BernoulliNaiveBayes":
        obj = cls(alpha=d["alpha"])
        obj.classes_       = d["classes"]
        obj.log_priors_    = np.array(d["log_priors"])
        obj.log_probs_     = np.array(d["log_probs"])
        obj.log_neg_probs_ = np.array(d["log_neg_probs"])
        return obj

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: str) -> "BernoulliNaiveBayes":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
