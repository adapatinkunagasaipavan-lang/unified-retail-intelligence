"""
Churn risk explainer: answers "why is customer X considered high risk?"

Wraps ml/inference/score_customers.py so the same underlying model +
explanation logic is used, just packaged as a natural-language answer with
retrieved evidence -- the "ML questions" half of the plan's GenAI agent:
"retrieves prediction -> retrieves relevant features/SHAP explanation ->
generates explanation -> shows evidence."

Usage (library):
    from churn_explainer import ChurnExplainerAgent
    agent = ChurnExplainerAgent()
    result = agent.explain("CUST000279")
    print(result.answer)
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ml", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ml", "inference"))
from train_churn_model import FEATURE_COLUMNS, load_features  # noqa: E402
from score_customers import load_production_model  # noqa: E402


@dataclass
class ExplanationResult:
    answer: str
    risk_score: Optional[float] = None
    evidence: Optional[pd.DataFrame] = None
    model_version: Optional[str] = None


class ChurnExplainerAgent:
    def __init__(self, gold_dir: str = "data/lake/gold",
                 model_name: str = "churn-model",
                 tracking_uri: str = "sqlite:///mlruns.db"):
        self.gold_dir = gold_dir
        self.model, self.model_version = load_production_model(model_name, tracking_uri)
        self.df = load_features(f"{gold_dir}/customer_features")
        # Reference profiles: average feature values for churned vs retained
        # customers. Used to describe whether a customer's values look more
        # like the "churned" population or the "retained" population --
        # this is more honest for a non-linear model (Random Forest) than a
        # z-score * importance heuristic, which has no reliable direction
        # for tree ensembles and produced misleading "decreases risk"
        # labels on genuinely high-risk customers during development.
        self._churned_profile = self.df[self.df["churned"] == 1][FEATURE_COLUMNS].mean()
        self._retained_profile = self.df[self.df["churned"] == 0][FEATURE_COLUMNS].mean()

    def explain(self, customer_id: str) -> ExplanationResult:
        row = self.df[self.df["customer_id"] == customer_id]
        if row.empty:
            return ExplanationResult(
                f"I don't have a record for customer {customer_id}. "
                f"Check the ID -- customer IDs look like CUST000279."
            )

        x = row[FEATURE_COLUMNS]
        proba = self.model.predict_proba(x)[0][1]
        importances = dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))

        # For each feature, measure which reference profile (churned vs
        # retained) this customer's value sits closer to, weighted by how
        # important that feature is to the model overall.
        contributions = []
        for feat in FEATURE_COLUMNS:
            val = x[feat].values[0]
            churned_avg = self._churned_profile[feat]
            retained_avg = self._retained_profile[feat]
            dist_to_churned = abs(val - churned_avg)
            dist_to_retained = abs(val - retained_avg)
            spread = abs(churned_avg - retained_avg) + 1e-9
            # positive score = looks more like a churned customer on this feature
            leans_churned_score = (dist_to_retained - dist_to_churned) / spread
            weighted_score = importances[feat] * leans_churned_score
            contributions.append((feat, weighted_score, val))
        contributions.sort(key=lambda c: -abs(c[1]))

        risk_level = "high" if proba >= 0.6 else ("moderate" if proba >= 0.3 else "low")
        top_factors = contributions[:3]

        factor_sentences = []
        FEATURE_LABELS = {
            "customer_tenure_days": "how long they've been a customer",
            "lifetime_value": "their total lifetime spend",
            "total_transactions": "how many purchases they've made",
            "distinct_categories_purchased": "how many different categories they buy from",
            "avg_transaction_value": "their average order size",
            "age": "their age",
            "region_encoded": "their region",
        }
        for feat, score, raw_val in top_factors:
            leans = "typical of customers who churn" if score > 0 else "typical of customers who stay"
            label = FEATURE_LABELS.get(feat, feat)
            factor_sentences.append(f"{label} ({leans})")

        answer = (
            f"Customer {customer_id} has a {proba:.0%} churn risk score, which is {risk_level}. "
            f"The biggest factors are: {', '.join(factor_sentences)}."
        )

        evidence = pd.DataFrame([{
            "customer_id": customer_id,
            "churn_risk_score": round(float(proba), 4),
            **{feat: row[feat].values[0] for feat in FEATURE_COLUMNS},
        }])

        return ExplanationResult(answer, float(proba), evidence, self.model_version)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", default="data/lake/gold")
    parser.add_argument("--customer-id", required=True)
    args = parser.parse_args()

    agent = ChurnExplainerAgent(gold_dir=args.gold_dir)
    result = agent.explain(args.customer_id)

    print(f"Q: Why is customer {args.customer_id} considered high risk?\n")
    print(f"A: {result.answer}\n")
    if result.evidence is not None:
        print("Evidence (source: churn-model + gold_customer_features):")
        print(result.evidence.to_string(index=False))


if __name__ == "__main__":
    main()
