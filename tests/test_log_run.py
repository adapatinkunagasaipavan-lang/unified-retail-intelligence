import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "monitoring"))

from log_run import append_run_record


def test_append_run_record_combines_both_reports(tmp_path):
    dq_report = tmp_path / "dq.json"
    dq_report.write_text(json.dumps({
        "overall_score": 0.95, "total_rows": 1000, "fully_passing_rows": 950
    }))
    model_report = tmp_path / "model.json"
    model_report.write_text(json.dumps({
        "roc_auc": 0.88, "model_version": 3, "passed": True
    }))
    history_file = tmp_path / "history.jsonl"

    record = append_run_record(str(dq_report), str(model_report), str(history_file), pipeline_duration_seconds=12.345)

    assert record["dq_overall_score"] == 0.95
    assert record["model_roc_auc"] == 0.88
    assert record["model_version"] == 3
    assert record["model_gate_passed"] is True
    assert record["pipeline_duration_seconds"] == 12.35  # rounded
    assert "timestamp" in record

    # file actually has one line matching the record
    lines = history_file.read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["dq_overall_score"] == 0.95


def test_append_run_record_handles_missing_model_report(tmp_path):
    """Simulates a pipeline run that halted at the DQ gate before training --
    should still log what it has (DQ metrics) without crashing."""
    dq_report = tmp_path / "dq.json"
    dq_report.write_text(json.dumps({"overall_score": 0.60, "total_rows": 500, "fully_passing_rows": 300}))
    history_file = tmp_path / "history.jsonl"

    record = append_run_record(str(dq_report), None, str(history_file))

    assert record["dq_overall_score"] == 0.60
    assert "model_roc_auc" not in record


def test_append_run_record_appends_not_overwrites(tmp_path):
    dq_report = tmp_path / "dq.json"
    dq_report.write_text(json.dumps({"overall_score": 0.9}))
    history_file = tmp_path / "history.jsonl"

    append_run_record(str(dq_report), None, str(history_file))
    append_run_record(str(dq_report), None, str(history_file))

    lines = history_file.read_text().strip().split("\n")
    assert len(lines) == 2
