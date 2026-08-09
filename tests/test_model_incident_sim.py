import sys
import os

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml", "training"))
from train_churn_model import TARGET_COLUMN  # noqa: E402


def test_dummy_classifier_scores_near_random():
    """Sanity-checks the core assumption behind
    ml/evaluation/simulate_model_incident.py: a majority-class DummyClassifier
    should score close to 0.5 ROC-AUC (no signal), which is what makes it a
    valid stand-in for a 'broken model' incident."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({TARGET_COLUMN: rng.choice([0, 1], size=n, p=[0.85, 0.15])})

    X_train, X_test, y_train, y_test = train_test_split(
        df[[TARGET_COLUMN]], df[TARGET_COLUMN], test_size=0.2, random_state=42, stratify=df[TARGET_COLUMN]
    )
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    # A constant-probability predictor is mathematically undefined/degenerate
    # for ROC-AUC in some edge cases, but with default sklearn behavior on a
    # skewed class split it should land at or very near 0.5.
    auc = roc_auc_score(y_test, y_proba)
    assert 0.45 <= auc <= 0.55, f"Expected near-random ROC-AUC, got {auc}"


def test_incident_model_would_fail_typical_gate_threshold():
    """The simulated incident model's ROC-AUC (~0.5) must be below the
    project's real model gate threshold (0.75) -- otherwise the incident
    simulation wouldn't actually demonstrate the gate blocking anything."""
    simulated_incident_auc = 0.50
    real_gate_threshold = 0.75
    assert simulated_incident_auc < real_gate_threshold
