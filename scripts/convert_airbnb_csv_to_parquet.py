#!/usr/bin/env python
"""Convert the validated Airbnb Rio total CSV to a Spark-friendly Parquet file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType


RAW_RELATIVE_PATH = Path("data/raw/airbnb_rio/total_data.csv")
OUTPUT_RELATIVE_PATH = Path("data/processed/airbnb_rio/listings_clean.parquet")

TARGET_ROOM_TYPES = [
    "Entire home/apt",
    "Private room",
    "Shared room",
    "Hotel room",
]

CATEGORICAL_COLUMNS = [
    "property_type",
    "neighbourhood_cleansed",
    "cancellation_policy",
    "bed_type",
]

NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "guests_included",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "calculated_host_listings_count",
    "reviews_per_month",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "number_of_reviews_ltm",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
]

CURRENCY_COLUMNS = [
    "price",
    "weekly_price",
    "monthly_price",
    "security_deposit",
    "cleaning_fee",
    "extra_people",
]

BOOLEAN_COLUMNS = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "is_location_exact",
    "has_availability",
    "instant_bookable",
    "is_business_travel_ready",
    "require_guest_profile_picture",
    "require_guest_phone_verification",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(RAW_RELATIVE_PATH))
    parser.add_argument("--output", default=str(OUTPUT_RELATIVE_PATH))
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--master", default="local[4]")
    parser.add_argument("--driver-memory", default="12g")
    parser.add_argument("--executor-memory", default="12g")
    return parser.parse_args()


def build_spark(args: argparse.Namespace) -> SparkSession:
    return (
        SparkSession.builder.appName("ac2_airbnb_csv_to_parquet")
        .master(args.master)
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.executor.memory", args.executor_memory)
        .config("spark.sql.shuffle.partitions", str(args.partitions))
        .config("spark.sql.files.maxPartitionBytes", "64m")
        .getOrCreate()
    )


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd().resolve() / path


def input_csv_file(input_path: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"CSV file not found: {input_path}")
    if input_path.name != "total_data.csv":
        raise ValueError(
            "This converter is intentionally scoped to total_data.csv. "
            f"Received: {input_path}"
        )
    return input_path


def normalized_header(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        header = next(csv.reader(file))

    names: list[str] = []
    used: dict[str, int] = {}
    for index, raw_name in enumerate(header):
        name = raw_name.strip()
        count = used.get(name, 0)
        used[name] = count + 1
        if count:
            name = f"{name}_{count + 1}"
        names.append(name)
    return names


def column_exists(df, column_name: str) -> bool:
    return column_name in df.columns


def string_or_null(df, column_name: str):
    if not column_exists(df, column_name):
        return F.lit(None).cast("string")
    value = F.trim(F.col(column_name))
    return F.when(value == "", None).otherwise(value)


def category_or_unknown(df, column_name: str):
    if not column_exists(df, column_name):
        return F.lit("unknown")
    value = F.trim(F.col(column_name))
    return F.when(value == "", F.lit("unknown")).otherwise(value)


def currency_to_double(df, column_name: str):
    if not column_exists(df, column_name):
        return F.lit(None).cast("double")
    cleaned = F.regexp_replace(F.col(column_name), r"[$,]", "")
    return F.when(F.trim(cleaned) == "", None).otherwise(cleaned.cast("double"))


def boolean_to_int(df, column_name: str):
    if not column_exists(df, column_name):
        return F.lit(None).cast("double")
    value = F.lower(F.trim(F.col(column_name)))
    return (
        F.when(value == "t", F.lit(1.0))
        .when(value == "f", F.lit(0.0))
        .otherwise(None)
    )


def numeric_to_double(df, column_name: str):
    if not column_exists(df, column_name):
        return F.lit(None).cast("double")
    value = F.trim(F.col(column_name))
    return F.when(value == "", None).otherwise(value.cast("double"))


def read_total_csv(spark: SparkSession, csv_path: Path):
    schema = StructType([
        StructField(column_name, StringType(), True)
        for column_name in normalized_header(csv_path)
    ])

    return (
        spark.read.schema(schema)
        .option("header", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("mode", "PERMISSIVE")
        .csv(str(csv_path))
    )


def clean_total_dataframe(raw_df, csv_path: Path):
    last_scraped = F.to_date(string_or_null(raw_df, "last_scraped"))

    return raw_df.select(
        F.lit(csv_path.name).alias("source_file"),
        F.year(last_scraped).cast("double").alias("snapshot_year"),
        F.coalesce(
            numeric_to_double(raw_df, "month"),
            F.month(last_scraped).cast("double"),
        ).alias("snapshot_month"),
        string_or_null(raw_df, "id").alias("listing_id"),
        string_or_null(raw_df, "host_id").alias("host_id"),
        last_scraped.alias("last_scraped"),
        F.to_date(string_or_null(raw_df, "host_since")).alias("host_since"),
        string_or_null(raw_df, "room_type").alias("room_type"),
        *[
            category_or_unknown(raw_df, column_name).alias(column_name)
            for column_name in CATEGORICAL_COLUMNS
        ],
        *[
            numeric_to_double(raw_df, column_name).alias(column_name)
            for column_name in NUMERIC_COLUMNS
        ],
        *[
            currency_to_double(raw_df, column_name).alias(column_name)
            for column_name in CURRENCY_COLUMNS
        ],
        *[
            boolean_to_int(raw_df, column_name).alias(column_name)
            for column_name in BOOLEAN_COLUMNS
        ],
    ).filter(F.col("room_type").isin(TARGET_ROOM_TYPES))


def main() -> None:
    args = parse_args()
    input_path = input_csv_file(resolve_project_path(args.input))
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spark = build_spark(args)
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading total Airbnb CSV: {input_path}")
    cleaned_df = clean_total_dataframe(read_total_csv(spark, input_path), input_path)

    print("Cleaned schema:")
    cleaned_df.printSchema()

    print("Rows by target before write:")
    cleaned_df.groupBy("room_type").count().orderBy(F.desc("count")).show(truncate=False)

    print(f"Writing Parquet to {output_path}")
    (
        cleaned_df.repartition(args.partitions)
        .write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(str(output_path))
    )

    parquet_df = spark.read.parquet(str(output_path))
    print("Parquet verification:")
    print(f"Rows: {parquet_df.count()}")
    parquet_df.groupBy("room_type").count().orderBy(F.desc("count")).show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
