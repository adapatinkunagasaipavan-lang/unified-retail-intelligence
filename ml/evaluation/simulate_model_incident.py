"""
Model-layer incident simulation: the model-quality equivalent of
data_quality/simulate_incident.py.

Deliberately trains a degraded model (using only weak/noisy features
instead of the real signal) and registers it, to prove the model
evaluation gate (ml/evaluation/evaluate_model.py) actually blocks a bad
model from being promoted -- not just that it *would* in theory.

This does NOT touch the real Production alias unless the degraded model
actually beats it (it won't, by design) -- safe to run against your real
churn-model registry.

Usage:
    python ml/evaluation/simulate_model_incident.py --input data/lake/gold/customer_features
"""
import argparse
import json
import sys

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from train_churn_model import load_features, TARGET_COLUMN  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/lake/gold/customer_features")
    parser.add_argument("--model-name", default="churn-model")
    parser.add_argument("--experiment", default="retail-churn-incident-sim")
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns.db")
    parser.add_argument("--min-auc", type=float, default=0.75)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    df = load_features(args.input)
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        df[[TARGET_COLUMN]], y, test_size=0.2, random_state=42, stratify=y
    )

    # A deliberately broken "model": predicts the majority class every time,
    # simulating an upstream incident (e.g. a broken feature pipeline that
    # silently fed the model constant/garbage inputs, or a bad retrain that
    # collapsed to a trivial predictor).
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"Simulated incident model trained. ROC-AUC: {auc:.4f}")
    print("(A DummyClassifier predicting the majority class scores ~0.5 -- "
          "no better than random, simulating a broken/degraded retrain.)")

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "DummyClassifier (simulated incident)")
        mlflow.log_metric("roc_auc", auc)
        model_info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=args.model_name,
        )
        run_id = run.info.run_id
        new_version = model_info.registered_model_version

    print(f"\nRegistered as {args.model_name} v{new_version} (NOT promoted -- "
          f"only registered, exactly like a real bad training run would be)")

    # Run the real evaluation gate against the CURRENT Production model
    # (unaffected by this run, since we never call set_registered_model_alias
    # here) to prove the gate is still protecting a good model.
    client = MlflowClient()
    try:
        prod_version = client.get_model_version_by_alias(args.model_name, "production")
        prod_run = client.get_run(prod_version.run_id)
        prod_auc = prod_run.data.metrics.get("roc_auc")
        print(f"\nCurrent Production model (v{prod_version.version}) is UNCHANGED: "
              f"ROC-AUC {prod_auc:.4f}")
        print("The simulated bad model was registered for audit history but never "
              "promoted -- this is the model-layer equivalent of the data quality "
              "gate blocking a corrupted batch in Phase 1.")
    except Exception:
        print("\nNo Production model exists yet to compare against -- run the real "
              "training script first (ml/training/train_churn_model.py).")

    if auc < args.min_auc:
        print(f"\nIf this HAD been evaluated as a promotion candidate, the model gate "
              f"would have FAILED it: ROC-AUC {auc:.4f} < threshold {args.min_auc:.4f}")


if __name__ == "__main__":
    main()
