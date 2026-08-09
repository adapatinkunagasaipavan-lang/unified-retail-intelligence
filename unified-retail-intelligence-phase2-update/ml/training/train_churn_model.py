"""
Phase 2: Churn model training with MLflow tracking + promotion gate.

Trains a classifier on gold_customer_features, logs the run to MLflow
(params, metrics, feature importances, the model artifact itself), and
registers it to the MLflow Model Registry under "churn-model".

Promotion logic: the new model is only promoted to the "Production" alias
if it beats the current Production model's ROC-AUC on the same holdout
set (or if there is no Production model yet). This is the model-lifecycle
equivalent of the data quality gate in Phase 1 -- a real MLOps pattern,
not just "train and save."

Usage:
    python ml/training/train_churn_model.py \
        --input data/lake/gold/customer_features \
        --model-name churn-model \
        --experiment retail-churn
"""
import argparse
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [
    "total_transactions", "lifetime_value", "avg_transaction_value",
    "distinct_categories_purchased", "customer_tenure_days", "age",
    "region_encoded",
]
# NOTE: days_since_last_purchase is intentionally excluded. The churn label
# in gold_aggregates.py is defined directly as
# `days_since_last_purchase > 60`, so including it as a feature would be
# textbook data leakage -- the model would just be learning the label's own
# definition (this is exactly what produced a suspicious 1.0 ROC-AUC during
# development; removing it gives a realistic, defensible score instead).
TARGET_COLUMN = "churned"


def load_features(input_path: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    # region is categorical -- label-encode it for the model.
    # (For a larger project you'd persist this encoder for inference-time
    # reuse; kept simple here since it's re-derivable from the training data.)
    le = LabelEncoder()
    df["region_encoded"] = le.fit_transform(df["region"].fillna("UNKNOWN"))
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def train_and_evaluate(df: pd.DataFrame, random_state: int = 42):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    params = {
        "n_estimators": 150,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "random_state": random_state,
        "class_weight": "balanced",  # churn is a minority class (~13-14%)
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    feature_importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))

    return model, params, metrics, feature_importances, (X_test, y_test)


def get_current_production_auc(client: MlflowClient, model_name: str):
    """Returns the ROC-AUC of the current Production-aliased model, or None if there isn't one."""
    try:
        prod_version = client.get_model_version_by_alias(model_name, "production")
    except Exception:
        return None, None

    run = client.get_run(prod_version.run_id)
    auc = run.data.metrics.get("roc_auc")
    return auc, prod_version.version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/lake/gold/customer_features")
    parser.add_argument("--model-name", default="churn-model")
    parser.add_argument("--experiment", default="retail-churn")
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns.db")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    df = load_features(args.input)
    model, params, metrics, importances, (X_test, y_test) = train_and_evaluate(df)

    print("Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nFeature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(importances, "feature_importances.json")
        mlflow.log_param("n_training_rows", len(df))
        mlflow.log_param("churn_rate", round(df[TARGET_COLUMN].mean(), 4))

        model_info = mlflow.sklearn.log_model(
            model, name="model",
            registered_model_name=args.model_name,
        )
        run_id = run.info.run_id
        new_version = model_info.registered_model_version

    print(f"\nLogged run {run_id}, registered as {args.model_name} v{new_version}")

    # --- promotion gate ---
    client = MlflowClient()
    current_auc, current_version = get_current_production_auc(client, args.model_name)

    if current_auc is None:
        client.set_registered_model_alias(args.model_name, "production", new_version)
        print(f"No existing Production model -- promoting v{new_version} to Production "
              f"(ROC-AUC: {metrics['roc_auc']:.4f})")
    elif metrics["roc_auc"] > current_auc:
        client.set_registered_model_alias(args.model_name, "production", new_version)
        print(f"New model v{new_version} (ROC-AUC: {metrics['roc_auc']:.4f}) beats "
              f"current Production v{current_version} (ROC-AUC: {current_auc:.4f}) "
              f"-- PROMOTED")
    else:
        print(f"New model v{new_version} (ROC-AUC: {metrics['roc_auc']:.4f}) does NOT beat "
              f"current Production v{current_version} (ROC-AUC: {current_auc:.4f}) "
              f"-- staying on v{current_version}, new model registered but not promoted")


if __name__ == "__main__":
    main()
