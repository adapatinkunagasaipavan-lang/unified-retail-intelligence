"""
Inference: load the current Production churn model and score customers.

Also supports --explain, which prints per-feature contribution for a single
customer using the model's feature importances weighted by that customer's
feature values -- a lightweight, dependency-free stand-in for full SHAP,
answering "why is this customer flagged as high-risk?" (the GenAI agent in
Phase 3 will wrap this to answer that question in natural language).

Usage:
    python ml/inference/score_customers.py --input data/lake/gold/customer_features
    python ml/inference/score_customers.py --input data/lake/gold/customer_features --explain CUST000184
"""
import argparse

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from train_churn_model import FEATURE_COLUMNS, load_features  # noqa: E402


def load_production_model(model_name: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    prod_version = client.get_model_version_by_alias(model_name, "production")
    model_uri = f"models:/{model_name}/{prod_version.version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model, prod_version.version


def explain_customer(model, df: pd.DataFrame, customer_id: str):
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        print(f"Customer {customer_id} not found.")
        return

    x = row[FEATURE_COLUMNS]
    proba = model.predict_proba(x)[0][1]
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))

    print(f"\nCustomer {customer_id} -- churn risk score: {proba:.1%}")
    print("Top contributing factors (feature importance x this customer's value):")

    # Simple weighted-contribution approximation: importance * normalized feature value.
    # A real SHAP integration (Phase 3 stretch) would replace this with exact
    # per-prediction attributions -- this gives directionally similar, explainable output today.
    contributions = []
    for feat in FEATURE_COLUMNS:
        val = x[feat].values[0]
        norm_val = (val - df[feat].mean()) / (df[feat].std() + 1e-9)
        contributions.append((feat, importances[feat] * norm_val, val))

    contributions.sort(key=lambda c: -abs(c[1]))
    for feat, contrib, raw_val in contributions[:4]:
        direction = "increases" if contrib > 0 else "decreases"
        print(f"  - {feat} = {raw_val:.1f}  ({direction} risk, weight {contrib:+.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/lake/gold/customer_features")
    parser.add_argument("--model-name", default="churn-model")
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns.db")
    parser.add_argument("--explain", type=str, default=None,
                         help="customer_id to print an explanation for, instead of scoring everyone")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    model, version = load_production_model(args.model_name, args.tracking_uri)
    df = load_features(args.input)

    if args.explain:
        explain_customer(model, df, args.explain)
        return

    proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    df["churn_risk_score"] = proba

    print(f"Scored {len(df)} customers using {args.model_name} v{version}\n")
    top = df.sort_values("churn_risk_score", ascending=False).head(args.top_n)
    print(f"Top {args.top_n} highest-risk customers:")
    print(top[["customer_id", "region", "lifetime_value", "days_since_last_purchase",
               "churn_risk_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
