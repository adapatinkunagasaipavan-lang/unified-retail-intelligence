# Running this on Databricks

All the transformation code (`transformations/`, `data_quality/`, `pipelines/`)
is plain PySpark with no local-only dependencies, so it runs unchanged on
Databricks. Two things differ:

## 1. Delta instead of Parquet

Locally, `get_spark()` falls back to Parquet because pulling Delta's JARs via
`configure_spark_with_delta_pip` needs Maven Central access. On Databricks,
Delta is available natively — no JAR download needed. Just pass `--format delta`
to every script; Databricks runtimes already have the Delta catalog configured,
so `get_spark(use_delta=True)` will work immediately without any extra setup.

## 2. Orchestration

`pipelines/run_pipeline.py` runs the four steps as local subprocesses for
simplicity. On Databricks, translate this 1:1 into a **Databricks Workflow**
with one task per step:

```
[Bronze ingest] --> [DQ gate] --> [Silver transform] --> [Gold aggregates]
```

Each task points at the same script with the same arguments, just run as a
Databricks Job task instead of a subprocess. This gives you per-step retry,
task-level logs, and a visual DAG in the Databricks UI — much stronger
interview material than "I ran a Python script."

## 3. Unity Catalog (optional upgrade)

Once comfortable with the file-path version, register the Gold tables in
Unity Catalog (`spark.sql("CREATE TABLE gold.category_sales USING DELTA LOCATION ...")`)
so the GenAI text-to-SQL agent (Phase 3) can query them by name instead of
by path — this is also what a real BI/analytics team would expect.

## 4. Feature Store

`gold_customer_features` as built here is a plain Delta table. Databricks
Feature Store adds versioning, online serving, and point-in-time lookups on
top of exactly this table — worth adding in Phase 2 once the churn model
needs to serve features at inference time, not just training time.
