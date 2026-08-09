"""
Router: dispatches a natural-language question to either the text-to-SQL
engine (data questions) or the churn explainer (ML/customer-risk
questions), matching the plan's two-mode agent:

  1. Data questions -> generates SQL -> executes against Gold tables -> answer + source
  2. ML questions -> retrieves prediction -> retrieves features -> explanation + evidence

Every question asked is logged to monitoring/query_log.jsonl (timestamp,
question, which agent handled it, and whether it matched a known
intent) -- this is what monitoring/dashboard.py's "GenAI Query Activity"
section reads.

Usage (library):
    from router import Agent
    agent = Agent(gold_dir="data/lake/gold")
    result = agent.ask("Why is customer CUST000279 high risk?")
    print(result.answer)

Usage (CLI chat loop):
    python genai/agents/router.py --gold-dir data/lake/gold
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "text_to_sql"))
from nl_query_engine import NLQueryEngine  # noqa: E402
from churn_explainer import ChurnExplainerAgent  # noqa: E402

CUSTOMER_ID_PATTERN = re.compile(r"(CUST\d{6})", re.IGNORECASE)
DEFAULT_QUERY_LOG = os.path.join(os.path.dirname(__file__), "..", "..", "monitoring", "query_log.jsonl")


def _log_query(question: str, agent_used: str, matched_intent: str, answer: str,
               query_log_path: str = DEFAULT_QUERY_LOG):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "agent_used": agent_used,
        "matched_intent": matched_intent,
        "answer_preview": (answer[:120] + "...") if len(answer) > 120 else answer,
    }
    try:
        os.makedirs(os.path.dirname(query_log_path) or ".", exist_ok=True)
        with open(query_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # logging should never break the actual answer


class Agent:
    def __init__(self, gold_dir: str = "data/lake/gold",
                 model_name: str = "churn-model",
                 tracking_uri: str = "sqlite:///mlruns.db",
                 query_log_path: str = DEFAULT_QUERY_LOG):
        self.sql_engine = NLQueryEngine(gold_dir=gold_dir)
        self._explainer = None  # lazy-loaded: only needed if an ML question is asked
        self._gold_dir = gold_dir
        self._model_name = model_name
        self._tracking_uri = tracking_uri
        self.query_log_path = query_log_path

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
            result = self.explainer.explain(customer_id)
            _log_query(question, "churn_explainer", "customer_risk_explanation", result.answer, self.query_log_path)
            return result

        result = self.sql_engine.answer(question)
        _log_query(question, "text_to_sql", result.matched_intent or "unmatched", result.answer, self.query_log_path)
        return result


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
