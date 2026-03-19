"""
Model training, evaluation, and serialization for March Madness predictions.

Provides an ensemble model (Logistic Regression + XGBoost), leave-one-
tournament-out cross-validation, and final model training with serialization.
"""

import os
import sys
import json

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# Try XGBoost first, fall back to sklearn's GradientBoosting
try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except (ImportError, OSError):
    _HAS_XGBOOST = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    XGBOOST_PARAMS, LOGISTIC_PARAMS, RANDOM_SEED,
    TRAINED_MODEL_FILE, MODEL_METRICS_FILE, ALL_MODEL_FEATURES,
)


class BracketModel:
    """
    Bracket prediction model: logistic regression, XGBoost, or an
    ensemble that averages predicted probabilities from both.
    """

    VALID_TYPES = ("logistic", "xgboost", "ensemble")

    def __init__(self, model_type: str = "ensemble"):
        if model_type not in self.VALID_TYPES:
            raise ValueError(f"model_type must be one of {self.VALID_TYPES}")
        self.model_type = model_type
        self._logistic = None
        self._xgboost = None

        if model_type in ("logistic", "ensemble"):
            self._logistic = LogisticRegression(**LOGISTIC_PARAMS)
        if model_type in ("xgboost", "ensemble"):
            if _HAS_XGBOOST:
                self._xgboost = XGBClassifier(**XGBOOST_PARAMS)
            else:
                # Fallback: sklearn GradientBoosting (no libomp needed)
                self._xgboost = GradientBoostingClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=RANDOM_SEED,
                )

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> "BracketModel":
        """Train the underlying model(s)."""
        if self._logistic is not None:
            self._logistic.fit(X, y)
        if self._xgboost is not None:
            self._xgboost.fit(X, y)
        return self

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(team_a wins) for each row."""
        if self.model_type == "logistic":
            return self._logistic.predict_proba(X)[:, 1]
        if self.model_type == "xgboost":
            return self._xgboost.predict_proba(X)[:, 1]
        # ensemble
        p_lr = self._logistic.predict_proba(X)[:, 1]
        p_xgb = self._xgboost.predict_proba(X)[:, 1]
        return (p_lr + p_xgb) / 2.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary predictions (threshold 0.5)."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    # ── Serialization ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BracketModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected BracketModel, got {type(model)}")
        return model

    def __repr__(self):
        return f"BracketModel(model_type='{self.model_type}')"


# ── Leave-One-Tournament-Out CV ──────────────────────────────────────────


def evaluate_loto_cv(
    matchup_data: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label",
    year_col: str = "year",
) -> dict:
    """
    Leave-One-Tournament-Out cross-validation.

    Trains on all years except the held-out year, evaluates on
    the held-out year.  Prints a summary table and returns metrics.
    """
    years = sorted(matchup_data[year_col].unique())
    per_year = []

    print()
    print("====== Leave-One-Tournament-Out Cross-Validation ======")
    print(f"{'Year':<8}| {'Accuracy':>8} | {'Log Loss':>8} | {'Brier':>8} | {'Games':>5}")
    print(f"{'-'*8}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*6}")

    all_y_true, all_y_prob = [], []

    for year in years:
        test_mask = matchup_data[year_col] == year
        X_train = matchup_data.loc[~test_mask, feature_cols]
        y_train = matchup_data.loc[~test_mask, label_col]
        X_test = matchup_data.loc[test_mask, feature_cols]
        y_test = matchup_data.loc[test_mask, label_col]

        model = BracketModel("ensemble")
        model.train(X_train, y_train)
        y_prob = model.predict_proba(X_test)
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        bs = brier_score_loss(y_test, y_prob)

        per_year.append(dict(year=int(year), accuracy=acc, log_loss=ll,
                             brier_score=bs, games=len(y_test)))
        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

        print(f"{int(year):<8}| {acc*100:>7.1f}% | {ll:>8.3f} | {bs:>8.3f} | {len(y_test):>5}")

    # Overall
    yt = np.array(all_y_true)
    yp = np.array(all_y_prob)
    print(f"{'-'*8}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*6}")
    print(f"{'Overall':<8}| {accuracy_score(yt,(yp>=.5).astype(int))*100:>7.1f}% "
          f"| {log_loss(yt,yp):>8.3f} | {brier_score_loss(yt,yp):>8.3f} | {len(yt):>5}")
    print()

    return {"per_year": per_year, "overall": {
        "accuracy": float(accuracy_score(yt, (yp >= .5).astype(int))),
        "log_loss": float(log_loss(yt, yp)),
        "brier_score": float(brier_score_loss(yt, yp)),
        "total_games": len(yt),
    }}


# ── Final Model Training ─────────────────────────────────────────────────


def train_final_model(
    matchup_data: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label",
) -> BracketModel:
    """Train on ALL historical data, save model + metrics, return model."""
    X = matchup_data[feature_cols]
    y = matchup_data[label_col]

    model = BracketModel("ensemble")
    model.train(X, y)
    model.save(TRAINED_MODEL_FILE)
    print(f"Model saved → {TRAINED_MODEL_FILE}")

    y_prob = model.predict_proba(X)
    metrics = {
        "model_type": model.model_type,
        "n_samples": len(y),
        "n_features": len(feature_cols),
        "in_sample_accuracy": float(accuracy_score(y, model.predict(X))),
        "in_sample_log_loss": float(log_loss(y, y_prob)),
    }
    os.makedirs(os.path.dirname(MODEL_METRICS_FILE), exist_ok=True)
    with open(MODEL_METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {MODEL_METRICS_FILE}")

    return model
