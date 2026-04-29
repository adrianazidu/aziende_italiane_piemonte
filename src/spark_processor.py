"""
spark_processor.py
------------------
Downloads the NYC Yellow Taxi dataset from Kaggle and runs
6 Spark transformations on it, each producing a separate CSV.

Outputs written to data/taxi/:
  1. outliers.csv       — impossible / suspicious trips
  2. ranked.csv         — drivers ranked by revenue (window function)
  3. running_totals.csv — cumulative revenue over time (window function)
  4. sessions.csv       — trips grouped into rush-hour sessions
  5. pivot.csv          — revenue pivot: payment type × hour of day
  6. time_series.csv    — daily trip count and revenue trends
  7. joined.csv         — trips joined with vendor + payment lookup tables

Usage:
    python src/spark_processor.py

The script downloads the dataset automatically if not already present.
"""

import os, sys, glob

# Must be set BEFORE pyspark imports so the JVM picks them up at startup
# Fixes: UnsupportedOperationException: getSubject is not supported (Java 17+)
    # Java 17+ locked down internal APIs for security
    # --add-opens opens locked package for everyone
    # must be here not in spark session config, then spark has already started
os.environ['JAVA_TOOL_OPTIONS'] = ' '.join([
    '--add-opens=java.base/javax.security.auth=ALL-UNNAMED',
    '--add-opens=java.base/java.lang=ALL-UNNAMED',
    '--add-opens=java.base/java.lang.invoke=ALL-UNNAMED',
    '--add-opens=java.base/java.lang.reflect=ALL-UNNAMED',
    '--add-opens=java.base/java.io=ALL-UNNAMED',
    '--add-opens=java.base/java.net=ALL-UNNAMED',
    '--add-opens=java.base/java.nio=ALL-UNNAMED',
    '--add-opens=java.base/java.util=ALL-UNNAMED',
    '--add-opens=java.base/java.util.concurrent=ALL-UNNAMED',
    '--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED',
    '--add-opens=java.base/sun.nio.ch=ALL-UNNAMED',
    '--add-opens=java.base/sun.nio.cs=ALL-UNNAMED',
    '--add-opens=java.base/sun.util.calendar=ALL-UNNAMED',
])

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

KAGGLE_DATASET = "elemento/nyc-yellow-taxi-trip-data"   # ~200MB, ~1.5M rows
RAW_DIR        = "data/taxi/raw"
OUT_DIR        = "data/taxi"


# ─────────────────────────────────────────────
# Step 0 — Download
# ─────────────────────────────────────────────

def download_dataset() -> str:
    """Download the Kaggle dataset and return path to the first CSV found."""
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if csv_files:
        print(f"✅ Dataset already present: {csv_files[0]}")
        return csv_files[0]

    print("⬇️  Downloading NYC Yellow Taxi dataset from Kaggle…")
    try:
        import kaggle
        kaggle.api.authenticate()
    except Exception as e:
        print(f"❌ Kaggle auth failed: {e}")
        print("   Make sure ~/.kaggle/kaggle.json exists.")
        sys.exit(1)

    os.makedirs(RAW_DIR, exist_ok=True)
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=RAW_DIR, unzip=True)

    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not csv_files:
        print("❌ No CSV found after download.")
        sys.exit(1)

    print(f"✅ Downloaded → {csv_files[0]}")
    return csv_files[0]


# ─────────────────────────────────────────────
# Spark session
# ─────────────────────────────────────────────

def get_spark() -> SparkSession:
    # Make sure PySpark can find Java on Streamlit Cloud
    import os
    if not os.environ.get("JAVA_HOME"):
        for path in ["/usr/lib/jvm/java-11-openjdk-amd64",
                     "/usr/lib/jvm/java-11-openjdk",
                     "/usr/lib/jvm/default-java"]:
            if os.path.exists(path):
                os.environ["JAVA_HOME"] = path
                break

    # Fix for Java 17+ compatibility on Windows
    java_opts = " ".join([
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
    ])

    return (
        SparkSession.builder
        .appName("NYC Taxi Analyzer")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.extraJavaOptions", java_opts)
        .config("spark.executor.extraJavaOptions", java_opts)
        .getOrCreate()
    )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def save(df, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.toPandas().to_csv(path, index=False)
    count = df.count()
    print(f"  💾 {name:30s} ({count:,} rows)")


# ─────────────────────────────────────────────
# Load & normalise
# ─────────────────────────────────────────────

def load_and_clean(spark: SparkSession, path: str):
    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"  📦 Raw: {df.count():,} rows × {len(df.columns)} columns")

    # Normalise column names
    for c in df.columns:
        clean = c.strip().lower().replace(" ", "_")
        if clean != c:
            df = df.withColumnRenamed(c, clean)

    # Pick up the datetime column (varies by dataset version)
    dt_col = next((c for c in df.columns if "pickup" in c and ("date" in c or "time" in c)), None)
    if dt_col:
        df = df.withColumn("pickup_dt", F.to_timestamp(F.col(dt_col)))
        df = df.withColumn("hour",  F.hour("pickup_dt"))
        df = df.withColumn("date",  F.to_date("pickup_dt"))
        df = df.withColumn("month", F.date_format("pickup_dt", "yyyy-MM"))

    # Keep only sensible rows
    fare_col = next((c for c in df.columns if "fare" in c or "total" in c), None)
    dist_col = next((c for c in df.columns if "dist" in c), None)

    if fare_col:
        df = df.filter(F.col(fare_col).between(0.01, 500))
    if dist_col:
        df = df.filter(F.col(dist_col).between(0.01, 100))
    if "passenger_count" in df.columns:
        df = df.filter(F.col("passenger_count").between(1, 6))

    print(f"  🧹 After cleaning: {df.count():,} rows")
    df.cache()
    return df, fare_col, dist_col


# ─────────────────────────────────────────────
# 1. Outlier detection
# ─────────────────────────────────────────────

def outliers(df, fare_col):
    """Flag statistically suspicious trips using mean ± 3 std."""
    print("\n🔍 Function 1 — Outlier Detection")
    if not fare_col:
        print("  ⚠️  No fare column found, skipping."); return

    stats = df.select(
        F.mean(fare_col).alias("mean"),
        F.stddev(fare_col).alias("std")
    ).first()

    threshold = stats["mean"] + 3 * stats["std"]

    result = df.withColumn(
        "outlier_reason",
        F.when(F.col(fare_col) > threshold, "fare > mean+3σ")
         .when(F.col(fare_col) < 1.0,       "fare < $1")
         .when(F.col("passenger_count") > 5, "passenger_count > 5")
         .otherwise(None)
    ).filter(F.col("outlier_reason").isNotNull()) \
     .select("outlier_reason", fare_col,
             *[c for c in ["passenger_count","trip_distance","hour","date"]
               if c in df.columns])

    save(result, "outliers.csv")


# ─────────────────────────────────────────────
# 2 & 3. Window functions — ranking + running totals
# ─────────────────────────────────────────────

def window_functions(df, fare_col):
    """Rank payment types by daily revenue and compute running totals."""
    print("\n🪟  Function 2 & 3 — Window Functions")
    if not fare_col or "date" not in df.columns:
        print("  ⚠️  Missing columns, skipping."); return

    pay_col = next((c for c in df.columns if "payment" in c), None)
    group_col = pay_col if pay_col else (
        next((c for c in df.columns if "vendor" in c), None)
    )
    if not group_col:
        print("  ⚠️  No grouping column found, skipping."); return

    # Daily revenue per group
    daily = (
        df.groupBy("date", group_col)
          .agg(F.sum(fare_col).cast(DoubleType()).alias("daily_revenue"),
               F.count("*").alias("trip_count"))
    )

    # Rank within each date
    rank_win = Window.partitionBy("date").orderBy(F.desc("daily_revenue"))
    ranked = daily.withColumn("rank", F.rank().over(rank_win))
    save(ranked, "ranked.csv")

    # Running total over time
    run_win = Window.partitionBy(group_col).orderBy("date") \
                    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    running = daily.withColumn(
        "cumulative_revenue", F.sum("daily_revenue").over(run_win)
    ).withColumn(
        "cumulative_trips", F.sum("trip_count").over(run_win)
    )
    save(running, "running_totals.csv")


# ─────────────────────────────────────────────
# 4. Sessionization
# ─────────────────────────────────────────────

def sessionize(df, fare_col):
    """Label trips by time-of-day session: Morning rush, Daytime, Evening rush, Night."""
    print("\n⏰  Function 4 — Sessionization")
    if "hour" not in df.columns:
        print("  ⚠️  No hour column, skipping."); return

    cols = ["hour", "date", "month"] + \
           ([fare_col] if fare_col else []) + \
           [c for c in ["trip_distance","passenger_count"] if c in df.columns]

    result = df.select(*set(cols)).withColumn(
        "session",
        F.when((F.col("hour") >= 7)  & (F.col("hour") < 10),  "Morning Rush")
         .when((F.col("hour") >= 10) & (F.col("hour") < 16),  "Daytime")
         .when((F.col("hour") >= 16) & (F.col("hour") < 20),  "Evening Rush")
         .when((F.col("hour") >= 20) | (F.col("hour") < 5),   "Night")
         .otherwise("Early Morning")
    )

    # Aggregate by session
    agg_cols = []
    if fare_col:
        agg_cols += [F.sum(fare_col).cast(DoubleType()).alias("total_revenue"),
                     F.avg(fare_col).alias("avg_fare")]
    agg_cols += [F.count("*").alias("trip_count")]
    if "trip_distance" in df.columns:
        agg_cols += [F.avg("trip_distance").alias("avg_distance")]

    summary = result.groupBy("session").agg(*agg_cols) \
                    .orderBy(F.desc("trip_count"))
    save(summary, "sessions.csv")


# ─────────────────────────────────────────────
# 5. Pivot table
# ─────────────────────────────────────────────

def pivot_table(df, fare_col):
    """Revenue pivot: hour of day × session."""
    print("\n📊  Function 5 — Pivot Table")
    if not fare_col or "hour" not in df.columns:
        print("  ⚠️  Missing columns, skipping."); return

    # Hour × session pivot
    with_session = df.withColumn(
        "session",
        F.when((F.col("hour") >= 7)  & (F.col("hour") < 10),  "Morning_Rush")
         .when((F.col("hour") >= 10) & (F.col("hour") < 16),  "Daytime")
         .when((F.col("hour") >= 16) & (F.col("hour") < 20),  "Evening_Rush")
         .when((F.col("hour") >= 20) | (F.col("hour") < 5),   "Night")
         .otherwise("Early_Morning")
    )

    result = (
        with_session
        .groupBy("hour")
        .pivot("session", ["Morning_Rush","Daytime","Evening_Rush","Night","Early_Morning"])
        .agg(F.round(F.sum(fare_col).cast(DoubleType()), 2))
        .orderBy("hour")
        .fillna(0)
    )
    save(result, "pivot.csv")


# ─────────────────────────────────────────────
# 6. Time series
# ─────────────────────────────────────────────

def time_series(df, fare_col, dist_col):
    """Daily and monthly aggregations for trend analysis."""
    print("\n📈  Function 6 — Time Series")
    if "date" not in df.columns:
        print("  ⚠️  No date column, skipping."); return

    agg_exprs = [F.count("*").alias("trip_count")]
    if fare_col:
        agg_exprs += [
            F.sum(fare_col).cast(DoubleType()).alias("total_revenue"),
            F.avg(fare_col).alias("avg_fare"),
        ]
    if dist_col:
        agg_exprs += [F.avg(dist_col).alias("avg_distance")]
    if "passenger_count" in df.columns:
        agg_exprs += [F.sum("passenger_count").alias("total_passengers")]

    result = df.groupBy("date").agg(*agg_exprs).orderBy("date")
    save(result, "time_series.csv")


# ─────────────────────────────────────────────
# 7. Join with lookup tables
# ─────────────────────────────────────────────

def join_lookups(spark: SparkSession, df, fare_col):
    """Join trips with vendor and payment type lookup tables."""
    print("\n🔗  Function 7 — Join with Lookup Tables")

    import pandas as pd

    # Vendor lookup
    vendor_pd = pd.DataFrame({
        "vendor_id_key": [1, 2],
        "vendor_name":   ["Creative Mobile Technologies", "VeriFone Inc"]
    })

    # Payment type lookup
    payment_pd = pd.DataFrame({
        "payment_type_key": [1, 2, 3, 4, 5, 6],
        "payment_name":     ["Credit Card","Cash","No Charge",
                             "Dispute","Unknown","Voided Trip"]
    })

    vendor_spark  = spark.createDataFrame(vendor_pd)
    payment_spark = spark.createDataFrame(payment_pd)

    result = df
    vendor_col  = next((c for c in df.columns if "vendor" in c), None)
    payment_col = next((c for c in df.columns if "payment" in c), None)

    if vendor_col:
        result = result.join(
            vendor_spark,
            result[vendor_col].cast("int") == vendor_spark["vendor_id_key"],
            how="left"
        ).drop("vendor_id_key")

    if payment_col:
        result = result.join(
            payment_spark,
            result[payment_col].cast("int") == payment_spark["payment_type_key"],
            how="left"
        ).drop("payment_type_key")

    # Select a clean subset for output
    keep = [c for c in ["date","hour","month","vendor_name","payment_name",
                        fare_col,"trip_distance","passenger_count"]
            if c and c in result.columns]
    save(result.select(*keep).limit(50000), "joined.csv")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  NYC Yellow Taxi — PySpark Analysis Pipeline")
    print("=" * 55)

    csv_path = download_dataset()

    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("\n📂 Loading and cleaning data…")
    df, fare_col, dist_col = load_and_clean(spark, csv_path)

    outliers(df, fare_col)
    window_functions(df, fare_col)
    sessionize(df, fare_col)
    pivot_table(df, fare_col)
    time_series(df, fare_col, dist_col)
    join_lookups(spark, df, fare_col)

    spark.stop()

    print("\n" + "=" * 55)
    print("  ✅  All done! Outputs written to data/taxi/")
    print("=" * 55)


if __name__ == "__main__":
    run()