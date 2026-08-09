# Incident Postmortem: Simulated Model-Layer Regression

*(Simulated via `ml/evaluation/simulate_model_incident.py` — run it
yourself to reproduce these exact numbers.)*

## Summary

A simulated bad retrain (a model that predicts the majority class for
every customer, representing a broken feature pipeline or a collapsed
training run) was registered to the MLflow Model Registry. The promotion
gate correctly kept it out of Production.

## Timeline

1. **Real Production model in place.** `churn-model` v1, ROC-AUC 0.9168,
   trained on real features via `ml/training/train_churn_model.py`.
2. **Incident simulated.** A `DummyClassifier` (predicts the majority
   class regardless of input — the model-layer equivalent of a pipeline
   silently feeding constant or garbage features) is trained and logged.
3. **ROC-AUC: 0.5000** — no better than a coin flip.
4. **Registered but NOT promoted.** The simulation script registers the
   bad model as `churn-model` v2 for audit history (a real bad training
   run in production would still show up in the registry — that's the
   point of having one), but never touches the `production` alias.
5. **Production remained on v1 throughout**, ROC-AUC still 0.9168,
   confirmed by re-running `ml/evaluation/evaluate_model.py` after the
   incident — the gate reports the same healthy model, completely
   unaffected.
6. **If this had gone through the real promotion path** (`train_churn_model.py`'s
   promotion logic), it would have been compared against Production's
   0.9168 and rejected automatically, since 0.50 < 0.9168.

## What caught it

Two independent layers, matching the plan's design:
- **The promotion gate** in `train_churn_model.py` only flips the
  `production` alias if the new model's ROC-AUC beats the current
  Production model's.
- **The CI evaluation gate** (`ml/evaluation/evaluate_model.py`) would
  independently fail the build if the *current* Production model ever
  fell below the 0.75 threshold — a second check, not just trusting the
  promotion logic never has a bug.

## Why simulate this instead of just describing it

Same reasoning as the Phase 1 data incident: a claim ("the gate would
block a bad model") is weaker than a reproducible run that shows the
exact before/after numbers. Anyone can re-run
`ml/evaluation/simulate_model_incident.py` and see Production stay
healthy while a genuinely bad model gets safely registered-but-rejected.

## Why this matters for the role

This is the model-lifecycle version of "tell me about a production
incident" — a concrete ROC-AUC, a concrete comparison, a concrete
rejection, and a concrete confirmation that the real model was never at
risk. Pairs with `docs/incident_postmortem.md` (the data-layer version)
to cover both halves of an MLOps engineer's actual job: protecting data
quality *and* protecting model quality.
