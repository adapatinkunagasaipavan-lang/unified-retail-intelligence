"""
Appends a single record per pipeline run to monitoring/metrics_history.jsonl,
combining the data quality report and model evaluation report from that run.
This is what the monitoring dashboard (monitoring/dashboard.py) reads to
show trends over time -- the same pattern a real Grafana/Databricks
Lakehouse Monitoring setup would use, just file-based for a portfolio
project with no external monitoring stack to stand up.

Usage:
    python monitoring/log_run.py \
        --dq-report /tmp/dq_report.json \
        --model-report /tmp/model_report.json \
        --history-file monitoring/metrics_history.jsonl
"""
import argparse
import json
import os
from datetime import datetime, timezone


def append_run_record(dq_report_path: str, model_report_path: str, history_file: str,
                       pipeline_duration_seconds: float = None):
    record = {"timestamp": datetime.now(timezone.utc).isoformat()}

    if dq_report_path and os.path.exists(dq_report_path):
        with open(dq_report_path) as f:
            dq = json.load(f)
        record["dq_overall_score"] = dq.get("overall_score")
        record["dq_total_rows"] = dq.get("total_rows")
        record["dq_fully_passing_rows"] = dq.get("fully_passing_rows")

    if model_report_path and os.path.exists(model_report_path):
        with open(model_report_path) as f:
            model = json.load(f)
        record["model_roc_auc"] = model.get("roc_auc")
        record["model_version"] = model.get("model_version")
        record["model_gate_passed"] = model.get("passed")

    if pipeline_duration_seconds is not None:
        record["pipeline_duration_seconds"] = round(pipeline_duration_seconds, 2)

    os.makedirs(os.path.dirname(history_file) or ".", exist_ok=True)
    with open(history_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dq-report", default=None)
    parser.add_argument("--model-report", default=None)
    parser.add_argument("--history-file", default="monitoring/metrics_history.jsonl")
    parser.add_argument("--duration-seconds", type=float, default=None)
    args = parser.parse_args()

    record = append_run_record(args.dq_report, args.model_report, args.history_file, args.duration_seconds)
    print(f"Logged run to {args.history_file}:")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
