"""
Bronze layer: ingest raw CSV drops as-is into Delta format.

Philosophy of Bronze: minimal transformation. We enforce a schema so the
job doesn't silently corrupt data, we add ingestion metadata (source file,
ingestion timestamp) for lineage, but we do NOT clean, dedupe, or validate
business rules here -- that's Silver's job. Bronze is the immutable
"as received" record.

Usage:
    python transformations/bronze/bronze_ingest.py \
        --input data/raw/transactions.csv \
        --output data/lake/bronze/transactions \
        --format delta   # or "parquet" if Delta isn't available locally
"""
import argparse
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType
)

TRANSACTIONS_SCHEMA = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("product_category", StringType(), True),
    StructField("quantity", StringType(), True),       # kept as string in Bronze -- may contain "N/A"
    StructField("unit_price", DoubleType(), True),
    StructField("transaction_amount", DoubleType(), True),
    StructField("payment_method", StringType(), True),
    StructField("transaction_ts", StringType(), True),  # kept as string in Bronze -- parsed properly in Silver
])


def get_spark(app_name: str = "bronze-ingest", use_delta: bool = False) -> SparkSession:
    """
    use_delta=True requires network access to Maven Central to fetch the Delta
    JARs (configure_spark_with_delta_pip). On Databricks, Delta is available
    natively -- just use spark.sql.extensions / spark_catalog config directly
    and skip configure_spark_with_delta_pip entirely (see docs/DATABRICKS.md).
    Locally/offline, we fall back to Parquet, which is a drop-in swap: same
    DataFrame API, same partitioning, just without Delta's transaction log,
    time travel, and MERGE support.
    """
    builder = SparkSession.builder.appName(app_name)
    if use_delta:
        from delta import configure_spark_with_delta_pip
        builder = (
            builder
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        return configure_spark_with_delta_pip(builder).getOrCreate()
    return builder.getOrCreate()


def ingest(spark: SparkSession, input_path: str, output_path: str, output_format: str):
    df = (
        spark.read
        .option("header", True)
        .schema(TRANSACTIONS_SCHEMA)
        .csv(input_path)
    )

    bronze_df = (
        df
        .withColumn("_ingested_at", F.lit(datetime.now(timezone.utc).isoformat()))
        .withColumn("_source_file", F.input_file_name())
    )

    writer = bronze_df.write.mode("append")
    if output_format == "delta":
        writer.format("delta").save(output_path)
    else:
        writer.format("parquet").save(output_path)

    print(f"Bronze ingest complete: {bronze_df.count()} rows -> {output_path} ({output_format})")
    return bronze_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", default="parquet", choices=["delta", "parquet"])
    args = parser.parse_args()

    spark = get_spark(use_delta=(args.format == "delta"))
    ingest(spark, args.input, args.output, args.format)
    spark.stop()


if __name__ == "__main__":
    main()
