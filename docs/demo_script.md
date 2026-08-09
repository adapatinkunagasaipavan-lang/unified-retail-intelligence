# 2-3 Minute Demo Script

A tight, spoken walkthrough for interviews or a recorded demo. Assumes
you're screen-sharing a terminal with the repo already cloned and
dependencies installed. Total run time: about 2.5 minutes of talking
over ~3-4 minutes of actual command output (some of it can be sped up /
pre-run before the call if time is tight -- see the note at the end).

---

### 0:00 -- Open with the one-sentence pitch (10s)

> "This is an end-to-end data and AI platform I built: a PySpark pipeline
> with data quality gates, a churn model tracked in MLflow with an
> automated promotion gate, a GenAI assistant that answers questions
> about the data with zero freeform generation, and monitoring across
> all of it. I'll show you the pipeline, then a caught bug, then the
> assistant."

### 0:10 -- Run the pipeline (30s of talking over ~2 min of output)

```bash
python pipelines/run_pipeline.py --format parquet --dq-threshold 0.90 --model-min-auc 0.75
```

> "This runs Bronze, a data quality gate, Silver, Gold, trains the churn
> model, and gates the model on ROC-AUC -- all in one command, same
> structure as a Databricks Workflow. Notice the DQ gate output here --
> [point at the JSON] -- a single interpretable score plus a per-rule
> breakdown, not just a pass/fail."

*(Let it run in the background while you talk over the next section, or
pre-run it before the call and just show the tail of the output.)*

### 1:40 -- The bug story (40s) -- your strongest material

> "The interesting part isn't really the pipeline running -- it's this."

Open `docs/data_leakage_note.md` (or just tell it):

> "My first version of the churn model scored a perfect 1.0 ROC-AUC.
> That's not a win, it's a red flag -- it meant the label was leaking
> into the features. The churn label is literally defined as
> `days_since_last_purchase > 60`, and I'd left that column in training.
> I removed it, added a regression test so it can't come back, and got a
> realistic 0.91. I did the same thing with a second bug in the GenAI
> explainer -- an explanation that said a 95%-risk customer's top factors
> were all 'decreasing their risk,' which is incoherent. Fixed that too,
> documented both."

### 2:20 -- The GenAI assistant (30s)

```bash
python genai/agents/router.py --gold-dir data/lake/gold
```

Ask it live:
```
What were the top 5 selling categories?
Why is customer CUST000279 high risk?
```

> "Two modes: data questions generate and execute real SQL against the
> Gold tables via DuckDB -- no hallucination possible by construction,
> since the answer IS the query result. Risk questions use the actual
> Production model plus a reference-profile comparison, not a made-up
> explanation. No API key needed anywhere."

### 2:50 -- Close (10s)

> "Everything here is tested -- 20 passing tests -- containerized in
> three Docker images, and CI builds and validates all of it on every
> push. Repo's on GitHub if you want to look at the code."

---

## If you only have 90 seconds

Skip the live pipeline run (too slow to watch). Instead:
1. Show the green GitHub Actions history (10s) -- "four phases, all
   green, plus one real CI failure I caught and fixed within minutes"
2. Tell the data leakage bug story (40s) -- this is the single highest-
   value 40 seconds of the whole demo
3. Ask the GenAI assistant one question live (30s)
4. Close with the repo link (10s)

## If they ask "why didn't you use a real LLM for the chat part?"

Have this answer ready:

> "Deliberate choice -- I wanted the whole thing runnable offline with
> no API key or cost to manage, and it forces the answers to be
> literally grounded in real queries rather than trusting a model not to
> hallucinate. The architecture -- router, intent matching, SQL
> generation, evidence display -- is the same shape you'd use with a
> real LLM in the loop; swapping the intent-matching layer for an LLM
> call is a contained change, not a rewrite."
