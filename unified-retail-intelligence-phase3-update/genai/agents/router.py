"""
Router: dispatches a natural-language question to either the text-to-SQL
engine (data questions) or the churn explainer (ML/customer-risk
questions), matching the plan's two-mode agent:

  1. Data questions -> generates SQL -> executes against Gold tables -> answer + source
  2. ML questions -> retrieves prediction -> retrieves features -> explanation + evidence

Usage (library):
    from router import Agent
    agent = Agent(gold_dir="data/lake/gold")
    result = agent.ask("Why is customer CUST000279 high risk?")
    print(result.answer)

Usage (CLI chat loop):
    python genai/agents/router.py --gold-dir data/lake/gold
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "text_to_sql"))
from nl_query_engine import NLQueryEngine  # noqa: E402
from churn_explainer import ChurnExplainerAgent  # noqa: E402

CUSTOMER_ID_PATTERN = re.compile(r"(CUST\d{6})", re.IGNORECASE)


class Agent:
    def __init__(self, gold_dir: str = "data/lake/gold",
                 model_name: str = "churn-model",
                 tracking_uri: str = "sqlite:///mlruns.db"):
        self.sql_engine = NLQueryEngine(gold_dir=gold_dir)
        self._explainer = None  # lazy-loaded: only needed if an ML question is asked
        self._gold_dir = gold_dir
        self._model_name = model_name
        self._tracking_uri = tracking_uri

    @property
    def explainer(self):
        if self._explainer is None:
            self._explainer = ChurnExplainerAgent(
                gold_dir=self._gold_dir,
                model_name=self._model_name,
                tracking_uri=self._tracking_uri,
            )
        return self._explainer

    def ask(self, question: str):
        customer_match = CUSTOMER_ID_PATTERN.search(question)
        is_risk_question = any(
            kw in question.lower() for kw in ["risk", "churn risk", "high risk", "why is customer", "likely to churn"]
        )

        if customer_match and (is_risk_question or "why" in question.lower()):
            customer_id = customer_match.group(1).upper()
            return self.explainer.explain(customer_id)

        return self.sql_engine.answer(question)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", default="data/lake/gold")
    parser.add_argument("--question", default=None,
                         help="Ask a single question and exit. If omitted, starts an interactive chat loop.")
    args = parser.parse_args()

    agent = Agent(gold_dir=args.gold_dir)

    if args.question:
        result = agent.ask(args.question)
        print(result.answer)
        return

    print("Retail Intelligence Assistant (Phase 3, no API key needed)")
    print("Ask about sales/categories/churn, or 'why is customer CUST000279 high risk'.")
    print("Type 'exit' to quit.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        result = agent.ask(q)
        print(result.answer + "\n")


if __name__ == "__main__":
    main()
