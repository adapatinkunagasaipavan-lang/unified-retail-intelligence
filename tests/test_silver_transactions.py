import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "transformations", "silver"))

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

from silver_transactions import clean_transactions

BRONZE_TEST_SCHEMA = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("product_category", StringType(), True),
    StructField("quantity", StringType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("transaction_amount", DoubleType(), True),
    StructField("payment_method", StringType(), True),
    StructField("transaction_ts", StringType(), True),
    StructField("_ingested_at", StringType(), True),
    StructField("_source_file", StringType(), True),
])


@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder.appName("test").master("local[1]").getOrCreate()
    yield spark
    spark.stop()


def make_bronze_row(spark, **overrides):
    # explicit schema so an all-None column (e.g. customer_id=None) doesn't
    # break Spark's schema inference, which can't determine a type from
    # a single-row all-null column.
    base = {
        "transaction_id": "tx-1",
        "customer_id": "CUST001",
        "product_category": "Electronics",
        "quantity": "2",
        "unit_price": 100.0,
        "transaction_amount": 200.0,
        "payment_method": "credit_card",
        "transaction_ts": "2026-08-01T10:00:00",
        "_ingested_at": "2026-08-01T10:05:00",
        "_source_file": "test.csv",
    }
    base.update(overrides)
    row = [base[f.name] for f in BRONZE_TEST_SCHEMA.fields]
    return spark.createDataFrame([row], schema=BRONZE_TEST_SCHEMA)


def test_clean_row_passes(spark):
    df = make_bronze_row(spark)
    clean_df, quarantined_df = clean_transactions(df)
    assert clean_df.count() == 1
    assert quarantined_df.count() == 0


def test_null_customer_id_is_quarantined(spark):
    df = make_bronze_row(spark, customer_id=None)
    clean_df, quarantined_df = clean_transactions(df)
    assert clean_df.count() == 0
    assert quarantined_df.count() == 1
    failures = quarantined_df.first()["_dq_failures"]
    assert "null_customer_id" in failures


def test_negative_amount_is_quarantined(spark):
    df = make_bronze_row(spark, transaction_amount=-50.0)
    clean_df, quarantined_df = clean_transactions(df)
    assert clean_df.count() == 0
    failures = quarantined_df.first()["_dq_failures"]
    assert "negative_amount" in failures


def test_invalid_quantity_is_quarantined(spark):
    df = make_bronze_row(spark, quantity="N/A")
    clean_df, quarantined_df = clean_transactions(df)
    assert clean_df.count() == 0
    failures = quarantined_df.first()["_dq_failures"]
    assert "invalid_quantity" in failures


def test_duplicate_transaction_id_deduplicated(spark):
    single = make_bronze_row(spark)
    df = single.union(single)  # exact duplicate row
    clean_df, quarantined_df = clean_transactions(df)
    assert clean_df.count() == 1


def test_row_can_fail_multiple_rules(spark):
    df = make_bronze_row(spark, customer_id=None, transaction_amount=-10.0)
    clean_df, quarantined_df = clean_transactions(df)
    failures = quarantined_df.first()["_dq_failures"]
    assert "null_customer_id" in failures
    assert "negative_amount" in failures
    assert len(failures) == 2
