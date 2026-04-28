"""
spark_processor.py
------------------
Main PySpark script. Reads a large CSV, applies transformations,
and writes a clean output CSV that Streamlit will consume.

Usage:
    python src/spark_processor.py --input data/your_file.csv
"""

import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
from transformations import (
    clean_dataframe,
    compute_summary,
    compute_time_series,
    compute_top_n,
)


# ─────────────────────────────────────────────
# Spark session
# ─────────────────────────────────────────────

def create_spark_session(app_name: str = "CSV Analyzer") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")   # tune for local mode
        .getOrCreate()
    )


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(input_path: str, output_dir: str = "data") -> None:
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n📂 Loading: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    print(f"✅ Loaded {df.count():,} rows × {len(df.columns)} columns")
    df.printSchema()

    # ── 1. Clean ──────────────────────────────
    df = clean_dataframe(df)
    print(f"🧹 After cleaning: {df.count():,} rows")

    # ── 2. Summary stats per category ─────────
    summary_df = compute_summary(df)
    _save(summary_df, os.path.join(output_dir, "summary.csv"))

    # ── 3. Time series (if date column exists) ─
    if "date" in [c.lower() for c in df.columns]:
        ts_df = compute_time_series(df)
        _save(ts_df, os.path.join(output_dir, "time_series.csv"))

    # ── 4. Top-N rows by value ─────────────────
    top_df = compute_top_n(df, n=20)
    _save(top_df, os.path.join(output_dir, "top_n.csv"))

    # ── 5. Full cleaned dataset (sample) ───────
    sample = df.limit(5000)          # cap for UI performance
    _save(sample, os.path.join(output_dir, "cleaned_sample.csv"))

    spark.stop()
    print("\n🎉 Pipeline complete. Outputs written to:", output_dir)


def _save(spark_df, path: str) -> None:
    """Convert to Pandas and save as a single CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    spark_df.toPandas().to_csv(path, index=False)
    print(f"  💾 Saved → {path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySpark CSV Analyzer")
    parser.add_argument("--input",  required=True, help="Path to input CSV")
    parser.add_argument("--output", default="data", help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.input, args.output)
