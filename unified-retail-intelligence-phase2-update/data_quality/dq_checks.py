"""
Data Quality scoring for the Bronze -> Silver boundary.

Computes a single interpretable score (% of rows passing all checks) plus
a per-rule breakdown, and exits non-zero if the score drops below a
threshold -- this is what the CI/CD pipeline (Phase 4) hooks into to
BLOCK a deployment when data quality regresses.

This is also the script used for the "deliberate incident" demo:
run it against a clean batch (~98% score) and then against a batch with
injected bad rows (~72% score) to show the before/after story.

Usage:
    python data_quality/dq_checks.py --input data/lake/bronze/transactions --threshold 0.95
"""
import argparse
import json
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


RULES = {
    "not_null_customer_id": lambda df: F.col("customer_id").isNotNull(),
    "not_null_product_category": lambda df: F.col("product_category").isNotNull(),
    "non_negative_amount": lambda df: F.col("transaction_amount") >= 0,
    "valid_quantity": lambda df: F.col("quantity").rlike(r"^\d+$"),
}


def get_spark(app_name: str = "dq-checks") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def run_checks(df):
    total = df.count()
    if total == 0:
        return {"total_rows": 0, "overall_score": 0.0, "rules": {}}

    rule_results = {}
    for name, rule_fn in RULES.items():
        passing = df.filter(rule_fn(df)).count()
        rule_results[name] = {
            "passing_rows": passing,
            "failing_rows": total - passing,
            "pass_rate": round(passing / total, 4),
        }

    # A row passes overall only if it passes every rule.
    all_conditions = None
    for rule_fn in RULES.values():
        cond = rule_fn(df)
        all_conditions = cond if all_conditions is None else (all_conditions & cond)

    fully_passing = df.filter(all_conditions).count()
    overall_score = round(fully_passing / total, 4)

    return {
        "total_rows": total,
        "fully_passing_rows": fully_passing,
        "overall_score": overall_score,
        "rules": rule_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--format", default="parquet", choices=["delta", "parquet"])
    parser.add_argument("--threshold", type=float, default=0.95,
                         help="Minimum overall_score to pass. Exits 1 if below this.")
    args = parser.parse_args()

    spark = get_spark()
    df = spark.read.format(args.format).load(args.input)

    report = run_checks(df)
    print(json.dumps(report, indent=2))

    score = report["overall_score"]
    if score < args.threshold:
        print(f"\nDATA QUALITY GATE FAILED: score {score:.1%} is below threshold {args.threshold:.1%}",
              file=sys.stderr)
        spark.stop()
        sys.exit(1)
    else:
        print(f"\nData quality gate passed: {score:.1%} >= {args.threshold:.1%}")

    spark.stop()


if __name__ == "__main__":
    main()
