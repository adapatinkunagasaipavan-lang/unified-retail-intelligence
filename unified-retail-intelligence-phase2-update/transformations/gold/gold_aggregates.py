"""
Gold layer: business-level aggregated tables, built on Silver.

Produces two Gold tables:
  1. gold_category_sales   -- daily sales by product category (for BI/dashboards
                               and for the GenAI text-to-SQL agent to query)
  2. gold_customer_features -- per-customer aggregates used as ML features
                               in Phase 2 (churn model)

Usage:
    python transformations/gold/gold_aggregates.py \
        --silver-transactions data/lake/silver/transactions \
        --customers data/raw/customers.csv \
        --output-sales data/lake/gold/category_sales \
        --output-features data/lake/gold/customer_features \
        --format parquet
"""
import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def get_spark(app_name: str = "gold-aggregates") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def build_category_sales(silver_df):
    return (
        silver_df
        .withColumn("transaction_date", F.to_date("transaction_ts"))
        .groupBy("transaction_date", "product_category")
        .agg(
            F.sum("transaction_amount").alias("total_sales"),
            F.sum("quantity").alias("total_units"),
            F.countDistinct("customer_id").alias("distinct_customers"),
            F.count("transaction_id").alias("transaction_count"),
        )
        .orderBy("transaction_date", "product_category")
    )


def build_customer_features(silver_df, customers_df, as_of_date=None):
    """
    Per-customer features for the churn model in Phase 2.
    as_of_date defaults to max(transaction_ts) in the data -- this makes the
    "days_since_last_purchase" feature reproducible for backtesting.
    """
    if as_of_date is None:
        as_of_date = silver_df.agg(F.max("transaction_ts")).first()[0]

    agg = (
        silver_df
        .groupBy("customer_id")
        .agg(
            F.count("transaction_id").alias("total_transactions"),
            F.sum("transaction_amount").alias("lifetime_value"),
            F.avg("transaction_amount").alias("avg_transaction_value"),
            F.countDistinct("product_category").alias("distinct_categories_purchased"),
            F.max("transaction_ts").alias("last_purchase_ts"),
            F.min("transaction_ts").alias("first_purchase_ts"),
        )
        .withColumn(
            "days_since_last_purchase",
            F.datediff(F.lit(as_of_date), F.col("last_purchase_ts"))
        )
        .withColumn(
            "customer_tenure_days",
            F.datediff(F.col("last_purchase_ts"), F.col("first_purchase_ts"))
        )
    )

    features = agg.join(
        customers_df.select("customer_id", "region", "age", "signup_date"),
        on="customer_id",
        how="left",
    )

    # Simple churn label for Phase 2: no purchase in 60+ days = churned.
    # (This is a heuristic label for a portfolio project -- in production
    # you'd validate this threshold against actual repurchase-cycle data.)
    features = features.withColumn(
        "churned",
        F.when(F.col("days_since_last_purchase") > 60, 1).otherwise(0)
    )

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver-transactions", required=True)
    parser.add_argument("--customers", required=True)
    parser.add_argument("--output-sales", required=True)
    parser.add_argument("--output-features", required=True)
    parser.add_argument("--format", default="parquet", choices=["delta", "parquet"])
    args = parser.parse_args()

    spark = get_spark()
    silver_df = spark.read.format(args.format).load(args.silver_transactions)
    customers_df = spark.read.option("header", True).option("inferSchema", True).csv(args.customers)

    sales_df = build_category_sales(silver_df)
    features_df = build_customer_features(silver_df, customers_df)

    sales_df.write.mode("overwrite").format(args.format).save(args.output_sales)
    features_df.write.mode("overwrite").format(args.format).save(args.output_features)

    print(f"Gold category_sales:     {sales_df.count()} rows -> {args.output_sales}")
    print(f"Gold customer_features:  {features_df.count()} rows -> {args.output_features}")
    churn_rate = features_df.filter(F.col("churned") == 1).count() / features_df.count()
    print(f"  churn rate in feature set: {churn_rate:.1%}")

    spark.stop()


if __name__ == "__main__":
    main()
