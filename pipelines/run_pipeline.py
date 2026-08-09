"""
End-to-end pipeline orchestrator: Bronze -> DQ gate -> Silver -> Gold.

In production this would be a Databricks Workflow / Airflow DAG with each
step as a separate task (so failures/retries are per-step and visible in
the orchestrator UI). This script is the local equivalent -- the exact
same underlying job modules, just called in sequence -- so you can run
the whole thing with one command while developing, and translate it
1:1 into DAG tasks later (see docs/DATABRICKS.md).

Usage:
    python pipelines/run_pipeline.py --raw-input data/raw/transactions.csv
"""
import argparse
import subprocess
import sys


def run_step(description: str, cmd: list[str]):
    print(f"\n{'='*70}\nSTEP: {description}\n{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nPIPELINE HALTED at step: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", default="data/raw/transactions.csv")
    parser.add_argument("--customers", default="data/raw/customers.csv")
    parser.add_argument("--format", default="parquet", choices=["delta", "parquet"])
    parser.add_argument("--dq-threshold", type=float, default=0.90)
    args = parser.parse_args()

    fmt = args.format

    run_step("Bronze ingest", [
        sys.executable, "transformations/bronze/bronze_ingest.py",
        "--input", args.raw_input,
        "--output", "data/lake/bronze/transactions",
        "--format", fmt,
    ])

    run_step("Data quality gate (Bronze)", [
        sys.executable, "data_quality/dq_checks.py",
        "--input", "data/lake/bronze/transactions",
        "--format", fmt,
        "--threshold", str(args.dq_threshold),
    ])

    run_step("Silver transform", [
        sys.executable, "transformations/silver/silver_transactions.py",
        "--input", "data/lake/bronze/transactions",
        "--output", "data/lake/silver/transactions",
        "--quarantine", "data/lake/silver/_quarantine/transactions",
        "--format", fmt,
    ])

    run_step("Gold aggregates", [
        sys.executable, "transformations/gold/gold_aggregates.py",
        "--silver-transactions", "data/lake/silver/transactions",
        "--customers", args.customers,
        "--output-sales", "data/lake/gold/category_sales",
        "--output-features", "data/lake/gold/customer_features",
        "--format", fmt,
    ])

    print(f"\n{'='*70}\nPIPELINE COMPLETE\n{'='*70}")


if __name__ == "__main__":
    main()
