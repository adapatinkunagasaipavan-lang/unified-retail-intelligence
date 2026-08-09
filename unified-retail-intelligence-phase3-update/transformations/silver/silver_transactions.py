"""
Silver layer: clean, deduplicate, and enforce business rules on Bronze data.

Rules applied here:
  - drop rows with null customer_id or product_category (can't be used downstream)
  - cast quantity to integer, dropping rows where it can't be cast (e.g. "N/A")
  - drop negative transaction_amount (data entry errors)
  - deduplicate on transaction_id (Bronze may contain replayed/duplicate rows)
  - parse transaction_ts into a real timestamp column
  - every dropped row is captured in a quarantine table for audit, not silently discarded

Usage:
    python transformations/silver/silver_transactions.py \
        --input data/lake/bronze/transactions \
        --output data/lake/silver/transactions \
        --quarantine data/lake/silver/_quarantine/transactions \
        --format parquet
"""
import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


def get_spark(app_name: str = "silver-transactions") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def clean_transactions(df):
    """Returns (clean_df, quarantined_df) -- quarantined rows are tagged with why they failed."""

    df = df.dropDuplicates(["transaction_id"])

    df = df.withColumn(
        "quantity_int",
        F.when(F.col("quantity").rlike(r"^\d+$"), F.col("quantity").cast(IntegerType())).otherwise(None)
    )

    df = df.withColumn("transaction_ts_parsed", F.to_timestamp("transaction_ts"))

    # Tag every failure reason (a row can fail more than one rule).
    # Built with F.filter(..., isNotNull) rather than array_remove(arr, None) --
    # the latter collapses to a NULL array under ANSI mode when any element is null.
    raw_failures = F.array(
        F.when(F.col("customer_id").isNull(), F.lit("null_customer_id")),
        F.when(F.col("product_category").isNull(), F.lit("null_product_category")),
        F.when(F.col("quantity_int").isNull(), F.lit("invalid_quantity")),
        F.when(F.col("transaction_amount") < 0, F.lit("negative_amount")),
        F.when(F.col("transaction_ts_parsed").isNull(), F.lit("unparseable_timestamp")),
    )
    df = df.withColumn(
        "_dq_failures",
        F.filter(raw_failures, lambda x: x.isNotNull())
    )

    clean_df = (
        df.filter(F.size(F.col("_dq_failures")) == 0)
        .select(
            "transaction_id", "customer_id", "product_category",
            F.col("quantity_int").alias("quantity"),
            "unit_price", "transaction_amount", "payment_method",
            F.col("transaction_ts_parsed").alias("transaction_ts"),
            "_ingested_at", "_source_file",
        )
    )

    quarantined_df = (
        df.filter(F.size(F.col("_dq_failures")) > 0)
        .select("transaction_id", "customer_id", "product_category", "quantity",
                "transaction_amount", "_dq_failures", "_ingested_at")
    )

    return clean_df, quarantined_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--format", default="parquet", choices=["delta", "parquet"])
    args = parser.parse_args()

    spark = get_spark()
    bronze_df = spark.read.format(args.format).load(args.input)

    clean_df, quarantined_df = clean_transactions(bronze_df)

    clean_df.write.mode("overwrite").format(args.format).save(args.output)
    quarantined_df.write.mode("overwrite").format(args.format).save(args.quarantine)

    total = bronze_df.count()
    clean_count = clean_df.count()
    quarantined_count = quarantined_df.count()

    print(f"Silver transform complete:")
    print(f"  input rows:       {total}")
    print(f"  clean rows:       {clean_count} ({clean_count/total:.1%})")
    print(f"  quarantined rows: {quarantined_count} ({quarantined_count/total:.1%})")
    print(f"  -> clean:      {args.output}")
    print(f"  -> quarantine: {args.quarantine}")

    spark.stop()


if __name__ == "__main__":
    main()
