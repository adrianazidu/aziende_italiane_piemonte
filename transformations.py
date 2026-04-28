"""
transformations.py
------------------
Reusable PySpark transformation functions.
Each function takes a Spark DataFrame and returns a Spark DataFrame.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


# ─────────────────────────────────────────────
# 1. Cleaning
# ─────────────────────────────────────────────

def clean_dataframe(df: DataFrame) -> DataFrame:
    """
    Generic cleaning steps:
    - Strip whitespace from string columns
    - Normalize column names (lowercase, no spaces)
    - Drop fully duplicate rows
    - Drop rows where ALL values are null
    """
    # Normalize column names
    renamed = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    for old, new in renamed.items():
        if old != new:
            df = df.withColumnRenamed(old, new)

    # Strip whitespace from string columns
    string_cols = [f.name for f in df.schema.fields if str(f.dataType) == "StringType()"]
    for col in string_cols:
        df = df.withColumn(col, F.trim(F.col(col)))

    # Drop fully duplicate rows
    df = df.dropDuplicates()

    # Drop rows where ALL columns are null
    df = df.dropna(how="all")

    return df


# ─────────────────────────────────────────────
# 2. Summary statistics
# ─────────────────────────────────────────────

def compute_summary(df: DataFrame, category_col: str = None) -> DataFrame:
    """
    Group by the first string column (or category_col if given)
    and compute count, sum, avg, min, max for all numeric columns.
    """
    # Auto-detect category column
    if category_col is None:
        str_cols = [f.name for f in df.schema.fields if str(f.dataType) == "StringType()"]
        category_col = str_cols[0] if str_cols else None

    # Auto-detect numeric columns
    numeric_cols = [
        f.name for f in df.schema.fields
        if str(f.dataType) in ("IntegerType()", "LongType()", "DoubleType()", "FloatType()")
    ]

    if not numeric_cols:
        print("⚠️  No numeric columns found for summary.")
        return df

    agg_exprs = []
    for col in numeric_cols:
        agg_exprs += [
            F.count(col).alias(f"{col}_count"),
            F.sum(col).cast(DoubleType()).alias(f"{col}_sum"),
            F.avg(col).alias(f"{col}_avg"),
            F.min(col).alias(f"{col}_min"),
            F.max(col).alias(f"{col}_max"),
        ]

    if category_col:
        return df.groupBy(category_col).agg(*agg_exprs).orderBy(F.desc(f"{numeric_cols[0]}_count"))
    else:
        return df.agg(*agg_exprs)


# ─────────────────────────────────────────────
# 3. Time series
# ─────────────────────────────────────────────

def compute_time_series(df: DataFrame, date_col: str = "date") -> DataFrame:
    """
    Aggregate numeric columns by month.
    Assumes date_col is castable to DateType.
    """
    numeric_cols = [
        f.name for f in df.schema.fields
        if str(f.dataType) in ("IntegerType()", "LongType()", "DoubleType()", "FloatType()")
        and f.name != date_col
    ]

    df = df.withColumn("month", F.date_format(F.to_date(F.col(date_col)), "yyyy-MM"))

    agg_exprs = [F.sum(c).cast(DoubleType()).alias(f"{c}_total") for c in numeric_cols]
    agg_exprs += [F.count("*").alias("row_count")]

    return df.groupBy("month").agg(*agg_exprs).orderBy("month")


# ─────────────────────────────────────────────
# 4. Top-N
# ─────────────────────────────────────────────

def compute_top_n(df: DataFrame, n: int = 20, sort_col: str = None) -> DataFrame:
    """
    Return the top-N rows sorted by the first numeric column (descending).
    """
    if sort_col is None:
        numeric_cols = [
            f.name for f in df.schema.fields
            if str(f.dataType) in ("IntegerType()", "LongType()", "DoubleType()", "FloatType()")
        ]
        sort_col = numeric_cols[0] if numeric_cols else df.columns[0]

    return df.orderBy(F.desc(sort_col)).limit(n)


# ─────────────────────────────────────────────
# 5. Custom filter
# ─────────────────────────────────────────────

def filter_by_value(df: DataFrame, col: str, min_val=None, max_val=None) -> DataFrame:
    """Filter a numeric column between min_val and max_val."""
    if min_val is not None:
        df = df.filter(F.col(col) >= min_val)
    if max_val is not None:
        df = df.filter(F.col(col) <= max_val)
    return df


def filter_by_category(df: DataFrame, col: str, values: list) -> DataFrame:
    """Keep only rows where col is in the given list."""
    return df.filter(F.col(col).isin(values))
