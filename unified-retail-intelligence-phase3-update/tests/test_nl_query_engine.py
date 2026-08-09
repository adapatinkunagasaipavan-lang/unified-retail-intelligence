import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "genai", "text_to_sql"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "genai", "agents"))

import pandas as pd
import pytest

from nl_query_engine import NLQueryEngine


@pytest.fixture(scope="module")
def gold_dir(tmp_path_factory):
    """Builds a tiny, self-contained set of Gold-shaped parquet tables so
    these tests don't depend on the full pipeline having been run first."""
    base = tmp_path_factory.mktemp("gold")

    sales_dir = base / "category_sales"
    sales_dir.mkdir()
    sales = pd.DataFrame({
        "transaction_date": ["2026-01-01", "2026-01-02", "2026-01-01"],
        "product_category": ["Electronics", "Electronics", "Books"],
        "total_sales": [1000.0, 500.0, 200.0],
        "total_units": [10, 5, 4],
        "distinct_customers": [8, 4, 3],
        "transaction_count": [10, 5, 4],
    })
    sales.to_parquet(sales_dir / "part-0.parquet")

    features_dir = base / "customer_features"
    features_dir.mkdir()
    features = pd.DataFrame({
        "customer_id": ["CUST000001", "CUST000002", "CUST000003", "CUST000004"],
        "churned": [1, 0, 0, 1],
    })
    features.to_parquet(features_dir / "part-0.parquet")

    return str(base)


def test_top_categories_query(gold_dir):
    engine = NLQueryEngine(gold_dir=gold_dir)
    result = engine.answer("What were the top 2 selling categories?")
    assert "Electronics" in result.answer
    assert result.sql is not None
    assert "category_sales" in result.sql.lower()


def test_category_total_query(gold_dir):
    engine = NLQueryEngine(gold_dir=gold_dir)
    result = engine.answer("What is the total sales for Electronics?")
    assert "1,500" in result.answer or "1500" in result.answer


def test_churn_rate_query(gold_dir):
    engine = NLQueryEngine(gold_dir=gold_dir)
    result = engine.answer("What is the churn rate?")
    assert "2 of 4" in result.answer
    assert "50.0%" in result.answer


def test_unmatched_question_does_not_crash(gold_dir):
    engine = NLQueryEngine(gold_dir=gold_dir)
    result = engine.answer("What's the weather like today?")
    assert "don't have a query pattern" in result.answer


def test_unknown_category_returns_graceful_message(gold_dir):
    engine = NLQueryEngine(gold_dir=gold_dir)
    result = engine.answer("What is the total sales for Zzzznotarealcategory?")
    assert "couldn't find" in result.answer.lower()
