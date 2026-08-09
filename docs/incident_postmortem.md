# Incident Postmortem: Simulated Upstream Data Corruption

*(Simulated for portfolio purposes via `data_quality/simulate_incident.py` — run it
yourself to reproduce these exact numbers.)*

## Summary

A simulated batch representing an upstream source-system failure introduced
a much higher rate of invalid rows (nulled `product_category`, negative
`transaction_amount`) than normal background noise. The data quality gate
caught it before it reached the Silver/Gold layers and blocked the pipeline.

## Timeline

1. **Normal batch processed.** Data Quality Score: **95.9%** (background
   noise rate, consistent with typical runs). Pipeline proceeds normally.
2. **Corrupted batch simulated.** ~30% of rows in the incoming batch had
   `product_category` nulled and `transaction_amount` flipped negative —
   representing an upstream system sending malformed records (e.g. a schema
   change or a broken ETL job on the source side).
3. **Data Quality Score dropped to 66.8%** — well below the 90-95% threshold
   configured in `pipelines/run_pipeline.py`.
4. **Pipeline halted.** `dq_checks.py` exits with a non-zero status code;
   `run_pipeline.py`'s `run_step()` catches this and stops before Silver/Gold
   ever run on the corrupted batch.
5. **Deployment blocked.** In the CI/CD workflow (`.github/workflows/ci.yml`),
   this same non-zero exit fails the CI job — so a "deploy" step gated on
   CI passing would never fire.
6. **Alert (documented, not wired to a real endpoint in this portfolio
   version):** in production this would push to a Slack webhook or
   PagerDuty via the DQ script's exit hook — noted here as the next
   real integration to add.

## Root cause (simulated)

Upstream source system sent a batch with a higher-than-normal rate of
missing category data and sign-flipped amounts — consistent with a
schema/mapping bug on the producer side.

## What caught it

The `not_null_product_category` and `non_negative_amount` rules in
`data_quality/dq_checks.py`, aggregated into a single `overall_score`
that the pipeline treats as a hard gate.

## What would happen next in production

- Silver/Gold tables are **not** touched — they retain the last known-good
  state, so downstream BI dashboards and the churn model don't silently
  serve on corrupted data.
- The quarantined rows (from `silver_transactions.py`'s quarantine table)
  give the on-call engineer the exact rows and failure reasons to hand back
  to the upstream team.
- Once the upstream fix ships, re-running the pipeline on a corrected batch
  restores the score above threshold and processing resumes automatically —
  no manual intervention needed beyond re-triggering the run.

## Why this matters for the role

This is the exact "tell me about a production incident" story interviewers
ask for — a concrete score, a concrete threshold, a concrete blocked
deployment, and a concrete recovery path, not just "I used MLflow."
