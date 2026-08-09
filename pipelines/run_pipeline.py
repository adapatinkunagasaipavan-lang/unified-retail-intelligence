"""
End-to-end pipeline orchestrator: Bronze -> DQ gate -> Silver -> Gold ->
train model -> model gate, with monitoring metrics logged after each run
(including failed runs, so the dashboard shows incidents, not just successes).

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
import tempfile
import time


def run_step(description: str, cmd: list[str], halt_on_failure: bool = True) -> int:
    print(f"\n{'='*70}\nSTEP: {description}\n{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0 and halt_on_failure:
        print(f"\nPIPELINE HALTED at step: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", default="data/raw/transactions.csv")
    parser.add_argument("--customers", default="data/raw/customers.csv")
    parser.add_argument("--format", default="parquet", choices=["delta", "parquet"])
    parser.add_argument("--dq-threshold", type=float, default=0.90)
    parser.add_argument("--model-min-auc", type=float, default=0.75)
    parser.add_argument("--metrics-history", default="monitoring/metrics_history.jsonl")
    parser.add_argument("--no-monitoring", action="store_true",
                         help="Skip writing to the monitoring metrics history.")
    args = parser.parse_args()

    fmt = args.format
    start_time = time.time()
    tmp_dir = tempfile.mkdtemp(prefix="pipeline_reports_")
    dq_report_path = f"{tmp_dir}/dq_report.json"
    model_report_path = f"{tmp_dir}/model_report.json"

    def log_metrics_and_maybe_exit(exit_code: int = None):
        if not args.no_monitoring:
            log_cmd = [
                sys.executable, "monitoring/log_run.py",
                "--dq-report", dq_report_path,
                "--model-report", model_report_path,
                "--history-file", args.metrics_history,
                "--duration-seconds", str(time.time() - start_time),
            ]
            subprocess.run(log_cmd)
        if exit_code is not None:
            sys.exit(exit_code)

    run_step("Bronze ingest", [
        sys.executable, "transformations/bronze/bronze_ingest.py",
        "--input", args.raw_input,
        "--output", "data/lake/bronze/transactions",
        "--format", fmt,
    ])

    dq_exit = run_step("Data quality gate (Bronze)", [
        sys.executable, "data_quality/dq_checks.py",
        "--input", "data/lake/bronze/transactions",
        "--format", fmt,
        "--threshold", str(args.dq_threshold),
        "--report-out", dq_report_path,
    ], halt_on_failure=False)

    if dq_exit != 0:
        log_metrics_and_maybe_exit(exit_code=dq_exit)

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

    run_step("Train churn model (MLflow tracked)", [
        sys.executable, "ml/training/train_churn_model.py",
        "--input", "data/lake/gold/customer_features",
    ])

    model_exit = run_step("Model evaluation gate", [
        sys.executable, "ml/evaluation/evaluate_model.py",
        "--model-name", "churn-model",
        "--min-auc", str(args.model_min_auc),
        "--report-out", model_report_path,
    ], halt_on_failure=False)

    log_metrics_and_maybe_exit(exit_code=model_exit if model_exit != 0 else None)

    print(f"\n{'='*70}\nPIPELINE COMPLETE\n{'='*70}")


if __name__ == "__main__":
    main()
