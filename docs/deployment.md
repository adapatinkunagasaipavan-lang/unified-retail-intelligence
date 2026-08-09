# Deployment: Docker + Docker Compose

Three services, three images -- deliberately split so the heavy one
(Spark/Java) doesn't bloat the two that don't need it:

| Service | Image | What it does | Port |
|---|---|---|---|
| `pipeline` | `Dockerfile.pipeline` | Runs Bronze -> Silver -> Gold -> train model -> gates, then exits | n/a (batch job) |
| `genai` | `Dockerfile.genai` | Chat UI (text-to-SQL + churn explainer) | 8501 |
| `monitoring` | `Dockerfile.monitoring` | Metrics/query dashboard | 8502 |

## First-time setup

SQLite (used for MLflow tracking) needs its file to **already exist on
the host** before Docker mounts it as a volume -- otherwise Docker
creates an empty *directory* named `mlruns.db` instead of a file, which
breaks MLflow. Do this once before your first `docker compose up`:

```bash
# macOS/Linux
touch mlruns.db

# Windows (Command Prompt)
type nul > mlruns.db
```

Also make sure `data/raw/transactions.csv` and `data/raw/customers.csv`
exist (generate them with `python ingestion/generate_synthetic_data.py`
if you haven't already) -- the `pipeline` service reads from `data/raw`,
which is bind-mounted, not baked into the image.

## Running it

```bash
# Build all three images
docker compose build

# Run the pipeline once (Bronze -> Silver -> Gold -> train -> gates)
docker compose run --rm pipeline

# Start the chat UI and monitoring dashboard
docker compose up genai monitoring
```

Then open:
- `http://localhost:8501` -- the GenAI chat assistant
- `http://localhost:8502` -- the monitoring dashboard

## Re-running the pipeline

```bash
docker compose run --rm pipeline
```

Since `data/`, `monitoring/`, and the MLflow files are bind-mounted (not
copied into the image), every run updates the same files your host
machine sees -- refresh the dashboard/chat UI in your browser and
they'll reflect the new run immediately, no rebuild needed.

## Why not a single image?

A single "do everything" image would need the JVM (for PySpark) even
for the lightweight chat/dashboard services that never touch Spark --
unnecessary image bloat and slower cold starts for services that are
meant to stay running. Splitting by actual runtime dependency is the
same reasoning a real platform team would apply.

## What a full production deployment would add (out of scope here)

- Push images to a registry (ECR/GCR/ACR/Docker Hub) instead of building
  locally
- Replace SQLite with a real Postgres-backed MLflow tracking server
- Run `pipeline` on a schedule (Databricks Workflow / Airflow / a
  Kubernetes CronJob) instead of manually via `docker compose run`
- Add health checks and restart policies to `genai`/`monitoring` in
  `docker-compose.yml`
- Front the services with a reverse proxy / ingress and real
  authentication instead of raw `localhost` ports
