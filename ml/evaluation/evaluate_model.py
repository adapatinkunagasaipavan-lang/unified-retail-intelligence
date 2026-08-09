"""
Model evaluation gate: checks the current Production-aliased churn model
meets a minimum ROC-AUC bar. This is the model-quality equivalent of
data_quality/dq_checks.py -- CI calls this after training and BLOCKS
deployment if the metric is below threshold.

Usage:
    python ml/evaluation/evaluate_model.py --model-name churn-model --min-auc 0.75
"""
import argparse
import json
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="churn-model")
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns.db")
    parser.add_argument("--min-auc", type=float, default=0.75)
    parser.add_argument("--report-out", type=str, default=None,
                         help="Optional path to write the JSON report to, for monitoring/run_pipeline.py to pick up.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    try:
        prod_version = client.get_model_version_by_alias(args.model_name, "production")
    except Exception:
        print(f"MODEL GATE FAILED: no Production-aliased version of '{args.model_name}' exists yet.",
              file=sys.stderr)
        sys.exit(1)

    run = client.get_run(prod_version.run_id)
    auc = run.data.metrics.get("roc_auc")

    print(f"Production model: {args.model_name} v{prod_version.version}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Threshold: {args.min_auc:.4f}")

    report = {
        "model_name": args.model_name,
        "model_version": prod_version.version,
        "roc_auc": auc,
        "threshold": args.min_auc,
        "passed": bool(auc is not None and auc >= args.min_auc),
    }
    if args.report_out:
        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump(report, f, indent=2)

    if auc is None:
        print("MODEL GATE FAILED: roc_auc metric not found on the run.", file=sys.stderr)
        sys.exit(1)

    if auc < args.min_auc:
        print(f"\nMODEL GATE FAILED: ROC-AUC {auc:.4f} is below threshold {args.min_auc:.4f}",
              file=sys.stderr)
        sys.exit(1)

    print(f"\nModel gate passed: ROC-AUC {auc:.4f} >= {args.min_auc:.4f}")


if __name__ == "__main__":
    main()
