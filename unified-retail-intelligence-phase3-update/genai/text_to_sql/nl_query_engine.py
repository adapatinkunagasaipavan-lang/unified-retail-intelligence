"""
Text-to-SQL over the Gold tables.

Deliberately built WITHOUT a hosted LLM API, so the whole project runs
offline with no API key to manage. Intent is matched with a small set of
robust regex patterns (a real production version would swap this layer
for an LLM call -- the rest of the pipeline: SQL generation, execution
against real Gold tables via DuckDB, and grounded/cited answers, doesn't
change either way). This still demonstrates the real pattern the plan
calls for: "generates SQL -> executes against Gold tables -> returns
answer -> shows source table/query."

Usage (library):
    from nl_query_engine import NLQueryEngine
    engine = NLQueryEngine(gold_dir="data/lake/gold")
    result = engine.answer("What were the top 5 selling categories?")
    print(result.answer)
    print(result.sql)

Usage (CLI, one-off question):
    python genai/text_to_sql/nl_query_engine.py \
        --gold-dir data/lake/gold \
        --question "What were the top 5 selling categories?"
"""
import argparse
import re
from dataclasses import dataclass, field
from typing import Optional

import duckdb
import pandas as pd


@dataclass
class QueryResult:
    answer: str
    sql: Optional[str] = None
    data: Optional[pd.DataFrame] = None
    source_table: Optional[str] = None
    matched_intent: Optional[str] = None


class NLQueryEngine:
    """Grounded natural-language query engine over the category_sales and
    customer_features Gold tables. No hallucination is possible by
    construction: every answer is derived from a real SQL query executed
    against the actual Parquet data, never generated freeform by a model.
    """

    def __init__(self, gold_dir: str = "data/lake/gold"):
        self.con = duckdb.connect(database=":memory:")
        self.con.execute(
            f"CREATE VIEW category_sales AS SELECT * FROM read_parquet('{gold_dir}/category_sales/*.parquet')"
        )
        self.con.execute(
            f"CREATE VIEW customer_features AS SELECT * FROM read_parquet('{gold_dir}/customer_features/*.parquet')"
        )

    def answer(self, question: str) -> QueryResult:
        q = question.strip().lower()

        # --- Intent: top N selling categories ---
        m = re.search(r"top\s+(\d+)?.*(selling|sales).*categor", q) or re.search(
            r"(best|top).*categor", q
        )
        if m:
            n = 5
            n_match = re.search(r"top\s+(\d+)", q)
            if n_match:
                n = int(n_match.group(1))
            sql = f"""
                SELECT product_category,
                       SUM(total_sales) AS total_sales,
                       SUM(total_units) AS total_units
                FROM category_sales
                GROUP BY product_category
                ORDER BY total_sales DESC
                LIMIT {n}
            """.strip()
            df = self.con.execute(sql).fetchdf()
            lines = [f"{i+1}. {r.product_category} — ${r.total_sales:,.2f} ({int(r.total_units)} units)"
                     for i, r in df.iterrows()]
            answer = f"Top {n} categories by total sales:\n" + "\n".join(lines)
            return QueryResult(answer, sql, df, "category_sales", "top_categories")

        # --- Intent: total sales for a specific category ---
        m = re.search(r"total sales (?:for|in|of)\s+([a-z &]+)", q) or re.search(
            r"how much.*sell.*(?:in|for)\s+([a-z &]+)", q
        )
        if m:
            category_guess = m.group(1).strip().title()
            sql = f"""
                SELECT product_category, SUM(total_sales) AS total_sales, SUM(total_units) AS total_units
                FROM category_sales
                WHERE lower(product_category) LIKE lower('%{category_guess}%')
                GROUP BY product_category
            """.strip()
            df = self.con.execute(sql).fetchdf()
            if df.empty:
                return QueryResult(
                    f"I couldn't find a category matching '{category_guess}'.", sql, df, "category_sales"
                )
            row = df.iloc[0]
            answer = (f"{row.product_category}: ${row.total_sales:,.2f} in total sales "
                      f"({int(row.total_units)} units).")
            return QueryResult(answer, sql, df, "category_sales", "category_total")

        # --- Intent: overall churn rate ---
        if "churn rate" in q or ("how many" in q and "churn" in q):
            sql = """
                SELECT COUNT(*) AS total_customers,
                       SUM(churned) AS churned_customers,
                       ROUND(AVG(churned) * 100, 1) AS churn_rate_pct
                FROM customer_features
            """.strip()
            df = self.con.execute(sql).fetchdf()
            row = df.iloc[0]
            answer = (f"{int(row.churned_customers)} of {int(row.total_customers)} customers "
                      f"are flagged as churned ({row.churn_rate_pct}%).")
            return QueryResult(answer, sql, df, "customer_features", "churn_rate")

        # --- Intent: overall revenue / total sales ---
        if "total revenue" in q or "total sales" in q or "how much did we sell" in q:
            sql = "SELECT SUM(total_sales) AS total_revenue FROM category_sales"
            df = self.con.execute(sql).fetchdf()
            row = df.iloc[0]
            answer = f"Total revenue across all categories: ${row.total_revenue:,.2f}."
            return QueryResult(answer, sql, df, "category_sales", "total_revenue")

        # --- Intent: distinct customers who bought a given category ---
        m = re.search(r"how many (?:distinct )?customers.*(?:bought|purchased)\s+([a-z &]+)", q)
        if m:
            category_guess = m.group(1).strip().title()
            sql = f"""
                SELECT product_category, SUM(distinct_customers) AS approx_distinct_customers
                FROM category_sales
                WHERE lower(product_category) LIKE lower('%{category_guess}%')
                GROUP BY product_category
            """.strip()
            df = self.con.execute(sql).fetchdf()
            if df.empty:
                return QueryResult(f"I couldn't find a category matching '{category_guess}'.", sql, df, "category_sales")
            row = df.iloc[0]
            answer = (f"Roughly {int(row.approx_distinct_customers)} distinct customers "
                      f"(summed across days -- see note) bought {row.product_category}.")
            return QueryResult(answer, sql, df, "category_sales", "category_customers")

        return QueryResult(
            "I don't have a query pattern for that yet. I can answer questions about: "
            "top selling categories, total sales/revenue, sales for a specific category, "
            "and overall churn rate. For customer-specific risk questions, ask "
            "\"why is customer <ID> high risk\" instead.",
            matched_intent="unmatched",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-dir", default="data/lake/gold")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    engine = NLQueryEngine(gold_dir=args.gold_dir)
    result = engine.answer(args.question)

    print(f"Q: {args.question}\n")
    print(f"A: {result.answer}\n")
    if result.sql:
        print(f"Generated SQL:\n{result.sql}\n")
    if result.source_table:
        print(f"Source table: {result.source_table}")


if __name__ == "__main__":
    main()
