# Unified Retail Intelligence Platform

An end-to-end cloud data and AI platform built on Databricks, PySpark, and
Delta Lake — implementing Bronze/Silver/Gold pipelines, automated data-quality
gating, MLflow-based model lifecycle management, CI/CD evaluation gates, and
a GenAI RAG/text-to-SQL assistant for natural-language analytics.

*Independent portfolio project — not built for or on behalf of any employer.*

## Architecture

```
Raw Data (synthetic retail transactions)
   |
   v
Bronze  (raw ingest, schema enforcement, lineage metadata)
   |
   v
Data Quality Gate  <-- blocks the pipeline if quality regresses
   |
   v
Silver  (cleaning, dedup, business rules, quarantine table)
   |
   v
Gold    (category sales for BI, customer features for ML)
   |
   +--> ML: churn model + MLflow tracking          [Phase 2 -- in progress]
   |
   +--> GenAI: text-to-SQL / RAG assistant           [Phase 3 -- in progress]
   |
   +--> CI/CD: GitHub Actions test + DQ gate         [done -- see .github/workflows/ci.yml]
```

## Status

- [x] **Phase 1 — Data Engineering Foundation.** Bronze -> Silver -> Gold on
      PySpark, Delta-ready (Parquet fallback locally), automated data quality
      scoring, quarantine table for failed rows, full pytest suite, CI/CD
      wired to run tests + the DQ gate on every push.
- [ ] **Phase 2 — ML Model + MLflow.** Churn model trained on
      `gold_customer_features`, tracked in MLflow with a promotion gate.
- [ ] **Phase 3 — GenAI Layer.** Text-to-SQL over Gold tables + a
      SHAP-explained "why is this customer high-risk" agent.
- [ ] **Phase 4 — Full MLOps.** CI/CD evaluation gates for the model,
      Streamlit UI, monitoring dashboard, Docker + deployment.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic raw data (5% realistic dirtiness injected on purpose)
python ingestion/generate_synthetic_data.py --rows 5000 --customers 800 --out data/raw/transactions.csv

# 2. Run the full pipeline: Bronze -> DQ gate -> Silver -> Gold
python pipelines/run_pipeline.py --format parquet --dq-threshold 0.90

# 3. Run the test suite
pytest tests/ -v

# 4. See the incident story in action
python data_quality/simulate_incident.py \
  --input data/lake/bronze/transactions \
  --output data/lake/bronze/_incident/transactions \
  --bad-fraction 0.30
```

## What each layer proves

| Layer | What it demonstrates |
|---|---|
| **Bronze** | Schema-enforced ingestion, lineage metadata, incremental-load pattern |
| **Data Quality Gate** | A single interpretable score, per-rule breakdown, hard threshold that blocks bad data from propagating -- see `docs/incident_postmortem.md` for a real before/after run |
| **Silver** | Real cleaning logic (not just `.dropna()`) -- multi-rule tagging, a quarantine table instead of silent data loss, full unit test coverage |
| **Gold** | Business-ready aggregates for both BI (`category_sales`) and ML (`customer_features`), with a documented, reproducible churn label |
| **CI/CD** | Every push runs the real pipeline against fresh synthetic data and fails the build if the DQ gate fails -- not just "tests pass" |

## Why Databricks-native, tested locally

The transform code is plain PySpark with no local-only dependencies -- see
`docs/DATABRICKS.md` for exactly what changes (Delta vs Parquet, Workflow
orchestration, Unity Catalog) when this is deployed to a real Databricks
workspace instead of run locally.

## Repo structure

```
unified-retail-intelligence/
├── ingestion/              # synthetic data generator (stand-in for a real source connector)
├── transformations/
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── data_quality/           # DQ scoring + the incident simulation
├── feature_engineering/    # Phase 2
├── ml/                     # Phase 2: training / evaluation / inference
├── genai/                  # Phase 3: rag / text_to_sql / agents
├── pipelines/               # orchestrator
├── tests/
├── infrastructure/          # Phase 4
├── .github/workflows/       # CI/CD
├── docker/                  # Phase 4
├── monitoring/               # Phase 4
└── docs/
```

## Resume bullets this project supports

> Built an end-to-end cloud data and AI platform using Databricks, PySpark,
> and Delta Lake, implementing Bronze/Silver/Gold pipelines, automated
> data-quality validation with a hard deployment gate, and CI/CD testing
> via GitHub Actions.

> Simulated and documented a production data-quality incident (score drop
> from 95.9% to 66.8%), demonstrating automated detection, pipeline
> blocking, and recovery -- see `docs/incident_postmortem.md`.
