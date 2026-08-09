# Note: A Misleading Explanation Bug Caught During Development

A second real bug caught during this build, alongside the data leakage
issue in Phase 2 -- worth documenting for the same reason.

## What happened

The first version of `genai/agents/churn_explainer.py` computed each
feature's "contribution" to a customer's risk score as:

```python
norm_val = (val - df[feat].mean()) / df[feat].std()
contribution = importances[feat] * norm_val
```

...then labeled positive contributions as "increasing risk" and negative
contributions as "decreasing risk."

Running it on a customer with a **95% churn risk score** (genuinely high
risk) produced this explanation:

> "The biggest factors are: how long they've been a customer (**decreasing**
> their risk), their total lifetime spend (**decreasing** their risk),
> how many purchases they've made (**decreasing** their risk)."

A high-risk customer whose top explanatory factors are all described as
*decreasing* their risk is incoherent -- and would be an embarrassing
thing to demo.

## Why it happened

`RandomForestClassifier.feature_importances_` measures how much a
feature reduces impurity **on average across the whole forest** -- it
carries no information about *direction* for any individual prediction.
Multiplying it by a z-score assumes a linear, monotonic relationship
between the feature and risk, which trees don't have. The sign of the
resulting number was essentially noise dressed up as a direction.

## The fix

Replaced the z-score heuristic with a **reference-profile comparison**:
for each feature, compute the average value among churned vs. retained
customers, then check which profile the customer's actual value sits
closer to. This produces a directionally honest statement -- "typical of
customers who churn" or "typical of customers who stay" -- without
claiming a false precision the model doesn't support.

## Result after the fix

Same 95%-risk customer, re-explained:

> "The biggest factors are: how long they've been a customer (**typical
> of customers who churn**), their total lifetime spend (**typical of
> customers who churn**), how many purchases they've made (**typical of
> customers who churn**)."

Coherent, and verified against a low-risk customer too (10% risk score),
whose top factor came back "typical of customers who stay" -- the
labels now track the actual prediction direction.

## The broader lesson

Both bugs in this project (the Phase 2 data leakage and this one) share
a root cause: **a metric or heuristic that looked plausible in isolation
but didn't hold up against a simple sanity check** (a perfect ROC-AUC;
an explanation that contradicts its own risk score). The fix in both
cases was the same discipline -- run the output against an obviously
wrong case (a leak-free retrain; a genuinely high-risk customer) and
trust the contradiction over the number.
