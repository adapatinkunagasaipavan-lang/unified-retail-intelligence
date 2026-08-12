"""
Monitoring dashboard: pipeline health, data quality trend, model
performance trend, and GenAI query activity -- the observability layer
the plan calls for ("cost-per-query trend, latency p95, eval score over
time" translated into what this project actually produces: DQ score,
model ROC-AUC, and query volume, since this project doesn't call a
metered LLM API).

Run with:
    streamlit run monitoring/dashboard.py --server.port 8502
"""
import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Pipeline Monitoring", page_icon="📈", layout="wide")

st.title("📈 Pipeline & Model Monitoring")
st.caption(
    "Every pipeline run (including failed ones) and every GenAI query is logged here. "
    "This is the observability layer -- the same thing Grafana/Databricks Lakehouse "
    "Monitoring would show in a real deployment, file-based for this portfolio project."
)

METRICS_FILE = st.sidebar.text_input("Metrics history file", value="monitoring/metrics_history.jsonl")
QUERY_LOG_FILE = st.sidebar.text_input("GenAI query log file", value="monitoring/query_log.jsonl")


def load_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


metrics_df = load_jsonl(METRICS_FILE)
query_df = load_jsonl(QUERY_LOG_FILE)

if metrics_df.empty:
    st.warning(
        "No pipeline runs logged yet. Run `python pipelines/run_pipeline.py` "
        "at least once -- it logs to this file automatically."
    )
else:
    latest = metrics_df.iloc[-1]
    total_runs = len(metrics_df)
    dq_failures = int((metrics_df.get("dq_overall_score", pd.Series(dtype=float)) < 0.90).sum()) \
        if "dq_overall_score" in metrics_df else 0
    model_failures = int((metrics_df.get("model_gate_passed", pd.Series(dtype=bool)) == False).sum()) \
        if "model_gate_passed" in metrics_df else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total pipeline runs", total_runs)
    col2.metric("Latest DQ score",
                f"{latest.get('dq_overall_score', float('nan')):.1%}" if pd.notna(latest.get("dq_overall_score")) else "N/A")
    col3.metric("Latest model ROC-AUC",
                f"{latest.get('model_roc_auc', float('nan')):.3f}" if pd.notna(latest.get("model_roc_auc")) else "N/A")
    col4.metric("DQ gate failures (history)", dq_failures,
                delta=None if dq_failures == 0 else "needs attention", delta_color="inverse")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Data Quality Score Over Time")
        if "dq_overall_score" in metrics_df.columns:
            fig = px.line(metrics_df, x="timestamp", y="dq_overall_score", markers=True)
            fig.add_hline(y=0.90, line_dash="dash", line_color="red",
                          annotation_text="typical gate threshold (90%)")
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No DQ score data yet.")

    with right:
        st.subheader("Model ROC-AUC Over Time")
        if "model_roc_auc" in metrics_df.columns and metrics_df["model_roc_auc"].notna().any():
            fig = px.line(metrics_df, x="timestamp", y="model_roc_auc", markers=True)
            fig.add_hline(y=0.75, line_dash="dash", line_color="red",
                          annotation_text="typical gate threshold (0.75)")
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model runs logged yet (pipeline may have halted before training).")

    st.subheader("Pipeline Run Duration")
    if "pipeline_duration_seconds" in metrics_df.columns:
        fig = px.bar(metrics_df, x="timestamp", y="pipeline_duration_seconds")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Run History (raw)")
    st.dataframe(metrics_df.sort_values("timestamp", ascending=False), use_container_width=True)

st.divider()
st.subheader("GenAI Query Activity")
if query_df.empty:
    st.info(
        "No GenAI queries logged yet. Ask the assistant something "
        "(`python genai/agents/router.py`) -- queries are logged automatically."
    )
else:
    c1, c2 = st.columns(2)
    c1.metric("Total queries", len(query_df))
    if "matched_intent" in query_df.columns:
        c2.metric("Unmatched (fallback) rate",
                  f"{(query_df['matched_intent'] == 'unmatched').mean():.1%}")

    if "matched_intent" in query_df.columns:
        intent_counts = query_df["matched_intent"].value_counts().reset_index()
        intent_counts.columns = ["intent", "count"]
        fig = px.bar(intent_counts, x="intent", y="count", title="Query intent distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(query_df.sort_values("timestamp", ascending=False), use_container_width=True)
