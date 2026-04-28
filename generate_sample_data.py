"""
generate_sample_data.py
-----------------------
Generates a realistic sample CSV for testing the pipeline.
Run once before spark_processor.py.

Usage:
    python generate_sample_data.py --rows 100000
"""

import argparse
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

CATEGORIES   = ["Electronics", "Clothing", "Food", "Sports", "Books", "Home", "Toys"]
REGIONS      = ["North", "South", "East", "West", "Central"]
SUB_CATS     = {
    "Electronics": ["Phones", "Laptops", "Audio", "Cameras"],
    "Clothing":    ["Men", "Women", "Kids", "Accessories"],
    "Food":        ["Bakery", "Dairy", "Produce", "Meat"],
    "Sports":      ["Outdoor", "Gym", "Team Sports", "Water Sports"],
    "Books":       ["Fiction", "Science", "History", "Children"],
    "Home":        ["Furniture", "Decor", "Kitchen", "Garden"],
    "Toys":        ["Board Games", "Outdoor", "Educational", "Action Figures"],
}


def generate(n_rows: int, output_path: str) -> None:
    random.seed(42)
    np.random.seed(42)

    start_date = datetime(2022, 1, 1)

    categories   = [random.choice(CATEGORIES) for _ in range(n_rows)]
    sub_cats     = [random.choice(SUB_CATS[c]) for c in categories]
    regions      = [random.choice(REGIONS) for _ in range(n_rows)]
    dates        = [start_date + timedelta(days=random.randint(0, 730)) for _ in range(n_rows)]
    quantities   = np.random.randint(1, 50, n_rows)
    unit_prices  = np.round(np.random.uniform(5, 500, n_rows), 2)
    revenues     = np.round(quantities * unit_prices, 2)
    discounts    = np.round(np.random.uniform(0, 0.3, n_rows), 2)
    ratings      = np.round(np.random.uniform(1, 5, n_rows), 1)
    customer_ids = [f"CUST-{random.randint(1000, 9999)}" for _ in range(n_rows)]

    df = pd.DataFrame({
        "order_id":    range(1, n_rows + 1),
        "customer_id": customer_ids,
        "date":        [d.strftime("%Y-%m-%d") for d in dates],
        "category":    categories,
        "sub_category": sub_cats,
        "region":      regions,
        "quantity":    quantities,
        "unit_price":  unit_prices,
        "revenue":     revenues,
        "discount":    discounts,
        "rating":      ratings,
    })

    # Inject ~2% nulls for realism
    for col in ["rating", "discount", "region"]:
        mask = np.random.rand(n_rows) < 0.02
        df.loc[mask, col] = None

    df.to_csv(output_path, index=False)
    print(f"✅ Generated {n_rows:,} rows → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows",   type=int, default=100_000, help="Number of rows")
    parser.add_argument("--output", default="data/sample_data.csv")
    args = parser.parse_args()
    generate(args.rows, args.output)
