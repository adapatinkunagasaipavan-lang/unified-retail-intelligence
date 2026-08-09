# Note: A Data Leakage Bug Caught During Development

Worth documenting because it's exactly the kind of thing interviewers ask
about ("tell me about a bug you caught in your own model").

## What happened

The first version of the churn model included `days_since_last_purchase`
as a training feature. Training it produced a **ROC-AUC of 1.0000** --
a perfect score.

## Why that's a red flag, not a win

A perfect score on a real-world churn prediction task is almost never
real signal -- it's almost always **data leakage**: the model has
access to information that (directly or indirectly) encodes the answer.

In this case, the leak was direct: `gold_aggregates.py` defines the
label as:

```python
features = features.withColumn(
    "churned",
    F.when(F.col("days_since_last_purchase") > 60, 1).otherwise(0)
)
```

`days_since_last_purchase` **is** the label's definition. Including it as
a feature meant the model was just learning "if this number is over 60,
predict 1" -- not predicting churn from customer behavior at all. It
would have been useless on any real new customer scored before their
churn status was already obvious.

## The fix

Removed `days_since_last_purchase` from `FEATURE_COLUMNS` in
`ml/training/train_churn_model.py`, with a comment explaining why, and
added a regression test
(`tests/test_train_churn_model.py::test_no_leaky_feature_in_training_columns`)
that fails the build if it's ever re-added.

## Result after the fix

| Metric | Before (leaky) | After (fixed) |
|---|---|---|
| ROC-AUC | 1.0000 | 0.9106 |
| Precision | 0.9545 | 0.4222 |
| Recall | 1.0000 | 0.9048 |

0.91 ROC-AUC with realistic precision/recall trade-offs is a defensible,
believable result for a churn model on this dataset. 1.0 was not.

## The broader lesson

Any feature that is definitionally close to the label -- not just
identical to it -- deserves scrutiny. `customer_tenure_days` remained in
the feature set here because it's *correlated* with churn behavior but
isn't the label's direct definition; that's a legitimate feature, not a
leak. The test above only catches the one confirmed leaky column, so
this is also a reminder that automated tests catch *known* leakage, not
all possible leakage.
