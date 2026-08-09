"""
Deliberate incident simulation for the "tell me about a production incident"
interview story.

Takes the clean Bronze batch, injects a much higher rate of bad rows
(simulating an upstream source system sending a broken batch), re-runs the
DQ gate, and shows it fail and block. This is meant to be run once to
generate docs/incident_postmortem.md content -- see that file for the
narrative writeup.

Usage:
    python data_quality/simulate_incident.py \
        --input data/lake/bronze/transactions \
        --output data/lake/bronze/_incident/transactions \
        --bad-fraction 0.30
"""
import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from dq_checks import run_checks


def get_spark(app_name: str = "simulate-incident") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def inject_incident(df, bad_fraction: float):
    """
    Simulate an upstream incident: a fraction of rows get corrupted the way
    a real broken source feed might corrupt them -- nulled category, negative
    amount, garbage quantity -- applied at a MUCH higher rate than normal
    background noise (5%) to represent an actual incident (e.g. 30%).
    """
    return df.withColumn(
        "_incident_roll", F.rand(seed=1)
    ).withColumn(
        "product_category",
        F.when(F.col("_incident_roll") < bad_fraction, F.lit(None)).otherwise(F.col("product_category"))
    ).withColumn(
        "transaction_amount",
        F.when(F.col("_incident_roll") < bad_fraction, F.col("transaction_amount") * -1).otherwise(F.col("transaction_amount"))
    ).drop("_incident_roll")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="parquet", choices=["delta", "parquet"])
    parser.add_argument("--bad-fraction", type=float, default=0.30)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    spark = get_spark()
    clean_df = spark.read.format(args.format).load(args.input)

    print("=== BEFORE INCIDENT (normal batch) ===")
    before_report = run_checks(clean_df)
    print(json.dumps({"overall_score": before_report["overall_score"]}, indent=2))

    incident_df = inject_incident(clean_df, args.bad_fraction)
    incident_df.write.mode("overwrite").format(args.format).save(args.output)

    print("\n=== AFTER INCIDENT (corrupted batch simulating upstream failure) ===")
    after_report = run_checks(incident_df)
    print(json.dumps({"overall_score": after_report["overall_score"]}, indent=2))

    print(f"\nData Quality Score")
    print(f"  Before: {before_report['overall_score']:.1%}")
    print(f"  After:  {after_report['overall_score']:.1%}")

    if after_report["overall_score"] < args.threshold:
        print(f"\nPipeline: FAILS  (score {after_report['overall_score']:.1%} < threshold {args.threshold:.1%})")
        print("Deployment: BLOCKED")
        print("Alert: TRIGGERED (would page/notify in production via Slack/PagerDuty webhook)")
    else:
        print("\nPipeline: PASSES -- try a higher --bad-fraction to trigger the gate")

    spark.stop()


if __name__ == "__main__":
    main()
