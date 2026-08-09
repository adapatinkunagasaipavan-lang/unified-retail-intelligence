# Unified Retail Intelligence Platform

An end-to-end cloud data and AI platform built on Databricks, PySpark, and
Delta Lake — implementing Bronze/Silver/Gold pipelines, automated data-quality
gating, MLflow-based model lifecycle management, CI/CD evaluation gates, and
a GenAI RAG/text-to-SQL assistant for natural-language analytics.

*Independent portfolio project — not built for or on behalf of any employer.*

## Architecture

See `docs/architecture.md` for the full diagram (data flow, ML lifecycle,
GenAI layer, and the Docker deployment view). Quick summary:

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
   +--> ML: churn model + MLflow tracking + promotion gate  [done]
   |
   +--> GenAI: text-to-SQL / churn explainer, no API key    [done]
   |
   +--> Monitoring: pipeline + query metrics, live dashboard [done]
   |
   +--> CI/CD: GitHub Actions -- tests, gates, Docker builds  [done]
```

For a live 2-3 minute walkthrough (including the two bugs caught during
development), see `docs/demo_script.md`.

## Status

- [x] **Phase 1 — Data Engineering Foundation.** Bronze -> Silver -> Gold on
      PySpark, Delta-ready (Parquet fallback locally), automated data quality
      scoring, quarantine table for failed rows, full pytest suite, CI/CD
      wired to run tests + the DQ gate on every push.
- [x] **Phase 2 — ML Model + MLflow.** Churn model (Random Forest) trained on
      `gold_customer_features`, tracked in MLflow (params/metrics/artifacts),
      registered to the MLflow Model Registry with a promotion gate (a new
      model is only promoted to Production if it beats the current
      Production ROC-AUC), plus a CI evaluation gate and inference/explanation
      script. See `docs/data_leakage_note.md` for a real bug caught and fixed
      during this phase.
- [x] **Phase 3 — GenAI Layer.** A two-mode natural-language agent, no API
      key required: (1) text-to-SQL over the Gold tables via DuckDB for
      data questions ("top selling categories", "churn rate"), and (2) a
      churn-risk explainer that answers "why is customer X high risk?"
      using the Production model + a reference-profile comparison. Wrapped
      in both a CLI chat (`genai/agents/router.py`) and a Streamlit UI
      (`genai/app.py`). See `docs/explanation_direction_bug_note.md` for a
      second real bug caught and fixed during this phase.
- [x] **Phase 4 — Full MLOps polish.** Monitoring logging wired into every
      pipeline run (including failed ones, so incidents are visible, not
      hidden), a Streamlit monitoring dashboard (data quality trend, model
      ROC-AUC trend, GenAI query activity), a model-layer incident
      simulation mirroring the Phase 1 data incident, and three
      Docker images (pipeline / genai / monitoring) wired together with
      docker-compose, actually built (not just written) on every CI run.
      See `docs/model_incident_postmortem.md` and `docs/deployment.md`.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic raw data (5% realistic dirtiness injected on purpose)
python ingestion/generate_synthetic_data.py --rows 5000 --customers 800 --out data/raw/transactions.csv

# 2. Run the full pipeline: Bronze -> DQ gate -> Silver -> Gold -> train model -> model gate
python pipelines/run_pipeline.py --format parquet --dq-threshold 0.90 --model-min-auc 0.75

# 3. Run the test suite (data pipeline + model tests)
pytest tests/ -v

# 4. See the data-quality incident story in action
python data_quality/simulate_incident.py \
  --input data/lake/bronze/transactions \
  --output data/lake/bronze/_incident/transactions \
  --bad-fraction 0.30

# 5. Score customers with the Production model, or explain one customer's risk
python ml/inference/score_customers.py --input data/lake/gold/customer_features --top-n 10
python ml/inference/score_customers.py --input data/lake/gold/customer_features --explain CUST000279

# 6. Inspect experiments in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db

# 7. Chat with the GenAI assistant (CLI, interactive)
python genai/agents/router.py --gold-dir data/lake/gold

# 8. Or launch the Streamlit chat UI
streamlit run genai/app.py

# 9. Launch the monitoring dashboard (data quality + model + GenAI activity trends)
streamlit run monitoring/dashboard.py --server.port 8502

# 10. See the model-layer incident story in action (mirrors step 4, for the model)
python ml/evaluation/simulate_model_incident.py --input data/lake/gold/customer_features --min-auc 0.75

# 11. Or run everything in Docker (see docs/deployment.md for first-time setup)
docker compose build
docker compose run --rm pipeline
docker compose up genai monitoring
```

## What each layer proves

| Layer | What it demonstrates |
|---|---|
| **Bronze** | Schema-enforced ingestion, lineage metadata, incremental-load pattern |
| **Data Quality Gate** | A single interpretable score, per-rule breakdown, hard threshold that blocks bad data from propagating -- see `docs/incident_postmortem.md` for a real before/after run |
| **Silver** | Real cleaning logic (not just `.dropna()`) -- multi-rule tagging, a quarantine table instead of silent data loss, full unit test coverage |
| **Gold** | Business-ready aggregates for both BI (`category_sales`) and ML (`customer_features`), with a documented, reproducible churn label |
| **ML Training + MLflow** | Full experiment tracking, model registry, and a promotion gate that only ships a model if it's actually better -- see `docs/data_leakage_note.md` for a real bug caught in development |
| **Model Evaluation Gate** | CI-blocking check on Production model quality, same pattern as the data quality gate |
| **Inference + Explanation** | Batch scoring plus a per-customer "why is this customer high-risk" explanation using a reference-profile comparison -- see `docs/explanation_direction_bug_note.md` for a real bug caught in the first version |
| **GenAI Agent (text-to-SQL + explainer)** | A two-mode natural-language interface with grounded answers by construction -- data responses are generated from executed SQL against the actual Gold tables via DuckDB, while risk explanations are produced by the real Production model, not freeform generation. No API key, fully offline. |
| **Monitoring** | Every pipeline run (including failed ones) and every GenAI query logged to a time series, visualized in a live dashboard -- data quality trend, model ROC-AUC trend, query intent distribution |
| **Model Incident Simulation** | A deliberately broken model (0.50 ROC-AUC) proven to get registered-but-not-promoted, while the real Production model stays untouched -- see `docs/model_incident_postmortem.md` |
| **Deployment (Docker)** | Three purpose-split images (pipeline / genai / monitoring), wired together with docker-compose, actually built on every CI run -- see `docs/deployment.md` |
| **CI/CD** | Every push runs the real pipeline (data + model), a GenAI agent smoke test, the model incident simulation, and builds all three Docker images -- fails the build if any gate fails or any image fails to build |

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

> Trained and productionized a customer churn model with MLflow experiment
> tracking, model registry, and an automated promotion gate that only
> deploys a model when it beats the current Production version's ROC-AUC.

> Caught and fixed a data leakage bug during model development (ROC-AUC
> dropped from an unrealistic 1.0 to a defensible 0.91 after removing a
> label-derived feature), and added a regression test to prevent recurrence.

> Built a two-mode GenAI assistant (text-to-SQL over Gold tables via DuckDB,
> plus a model-backed churn-risk explainer) with grounded, source-traceable
> answers by construction, wrapped in a Streamlit chat UI, requiring no
> external API key.

> Containerized the full platform into three purpose-split Docker images
> (batch pipeline, GenAI inference service, monitoring dashboard),
> orchestrated with docker-compose, with CI validating every image actually
> builds on every push.

> Built a monitoring dashboard tracking data quality score, model ROC-AUC,
> and GenAI query activity over time, and simulated both a data-layer and a
> model-layer production incident to prove the quality gates actually block
> bad deployments, not just claim to.

> Simulated and documented a production data-quality incident (score drop
> from 95.9% to 66.8%), demonstrating automated detection, pipeline
> blocking, and recovery -- see `docs/incident_postmortem.md`.
