import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml", "training"))

import pandas as pd
import pytest

from train_churn_model import load_features, train_and_evaluate, FEATURE_COLUMNS, TARGET_COLUMN


def make_synthetic_features(n=200, seed=0):
    """A small synthetic customer_features-shaped DataFrame for fast, offline model tests."""
    import numpy as np
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "customer_id": [f"CUST{i:04d}" for i in range(n)],
        "total_transactions": rng.integers(1, 20, n),
        "lifetime_value": rng.uniform(50, 10000, n),
        "avg_transaction_value": rng.uniform(10, 500, n),
        "distinct_categories_purchased": rng.integers(1, 8, n),
        "days_since_last_purchase": rng.integers(0, 200, n),
        "customer_tenure_days": rng.integers(0, 700, n),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "age": rng.integers(18, 75, n).astype(float),
    })
    # Label loosely tied to tenure + transactions (not identical to a feature,
    # unlike the real days_since_last_purchase-based label -- this avoids the
    # leakage trap in the test fixture itself).
    risk_score = (
        -0.01 * df["customer_tenure_days"]
        - 0.3 * df["total_transactions"]
        + rng.normal(0, 5, n)
    )
    df["churned"] = (risk_score > risk_score.median()).astype(int)
    return df


def test_load_features_encodes_region_and_fills_age(tmp_path):
    df = make_synthetic_features(n=50)
    df.loc[0, "age"] = None
    path = tmp_path / "features.parquet"
    df.to_parquet(path)

    loaded = load_features(str(path))
    assert "region_encoded" in loaded.columns
    assert loaded["age"].isnull().sum() == 0


def test_train_and_evaluate_returns_valid_metrics():
    df = make_synthetic_features(n=300)
    df["region_encoded"] = df["region"].astype("category").cat.codes
    model, params, metrics, importances, (X_test, y_test) = train_and_evaluate(df)

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert len(importances) == len(FEATURE_COLUMNS)
    assert abs(sum(importances.values()) - 1.0) < 1e-6  # importances sum to ~1


def test_model_beats_random_baseline():
    """Sanity check: the model should do meaningfully better than a coin flip
    on this synthetic (non-trivial, non-leaky) dataset."""
    df = make_synthetic_features(n=400, seed=1)
    df["region_encoded"] = df["region"].astype("category").cat.codes
    _, _, metrics, _, _ = train_and_evaluate(df)

    assert metrics["roc_auc"] > 0.6, (
        f"ROC-AUC {metrics['roc_auc']:.3f} is too close to random (0.5) -- "
        "model may not be learning a real signal"
    )


def test_no_leaky_feature_in_training_columns():
    """days_since_last_purchase must never be a training feature, since the
    churned label in gold_aggregates.py is directly derived from it."""
    assert "days_since_last_purchase" not in FEATURE_COLUMNS
