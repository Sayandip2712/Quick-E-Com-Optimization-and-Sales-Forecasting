"""
02_synthesize_with_sdv_normal.py
-------------------------------------------------------------
CTGAN/SDV synthesis for the ML-ready, no-event orders table.

IMPORTANT DESIGN CHOICE, worth understanding before you run this:
single-table CTGAN can only preserve relationships between columns it
actually sees. The whole point of 01's redesign was that order_value
depends on the customer's avg_order_value_band/tier and on order
timing -- but band/tier live in customers.csv, not orders.csv. If you
ran CTGAN on the bare `orders` table alone, it would learn only the
UNCONDITIONAL distribution of order_value, throw away the
band -> order_value structure entirely, and hand you back a synthetic
dataset with much LESS learnable signal than the real one (defeating
the purpose of this whole redesign).

Fix: denormalize first -- merge customer attributes (and store city)
into a flat orders_ml.csv, then run CTGAN on THAT. This is simpler and
more robust across SDV versions than multi-table HMASynthesizer, and
directly solves the actual problem.

Requires:  pip install sdv --break-system-packages   (or in a venv)
NOT executed in the authoring sandbox (no internet there to install
sdv). Written against SDV's current documented API; check
https://docs.sdv.dev if your installed version's API has moved.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.cag import Inequality

orders = pd.read_csv(f"{DATA_DIR}/orders.csv", parse_dates=["order_timestamp", "delivery_timestamp"])
customers = pd.read_csv(f"{DATA_DIR}/customers.csv")
stores = pd.read_csv(f"{DATA_DIR}/dark_stores.csv")

# ------------------------------------------------------------------
# 1. Denormalize: pull in exactly the columns that actually drive
#    order_value in 01's generator, so CTGAN can see the relationship.
# ------------------------------------------------------------------
orders_ml = (
    orders
    .merge(customers[["customer_id", "tier", "avg_order_value_band"]], on="customer_id", how="left")
    .merge(stores[["store_id", "city"]].rename(columns={"city": "store_city"}), on="store_id", how="left")
)
orders_ml["hour_of_day"] = orders_ml["order_timestamp"].dt.hour
orders_ml["day_of_week"] = orders_ml["order_timestamp"].dt.dayofweek
orders_ml["is_weekend"] = (orders_ml["day_of_week"] >= 5).astype(int)

print(f"Denormalized table: {orders_ml.shape}, columns: {list(orders_ml.columns)}")

# ------------------------------------------------------------------
# 2. Fit CTGAN on a representative sample of the denormalized table
# ------------------------------------------------------------------
orders_sample = orders_ml.sample(n=60000, random_state=42).reset_index(drop=True)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(orders_sample)
metadata.update_column(column_name="order_id", sdtype="id")
metadata.update_column(column_name="customer_id", sdtype="id")
metadata.update_column(column_name="store_id", sdtype="categorical")
metadata.update_column(column_name="tier", sdtype="categorical")
metadata.update_column(column_name="avg_order_value_band", sdtype="categorical")
metadata.update_column(column_name="store_city", sdtype="categorical")
metadata.set_primary_key("order_id")

synthesizer = CTGANSynthesizer(
    metadata,
    epochs=300,           # a bit higher than before -- more columns to learn jointly now
    batch_size=500,
    verbose=True,
)
synthesizer.add_constraints(constraints=[
    Inequality(low_column_name="order_timestamp", high_column_name="delivery_timestamp")
])

print(f"\nFitting CTGAN on {len(orders_sample)} rows, {orders_sample.shape[1]} columns...")
synthesizer.fit(orders_sample)

synthetic_ml = synthesizer.sample(num_rows=len(orders_ml))
synthetic_ml.to_csv(f"{DATA_DIR}/synthetic_orders_ml.csv", index=False)
print("Saved synthetic_orders_ml.csv:", synthetic_ml.shape)

# Also produce a version matching the ORIGINAL orders.csv schema
# (drop the denormalized columns) for anyone who wants that instead.
synthetic_orders_only = synthetic_ml[
    ["order_id", "customer_id", "store_id", "order_timestamp", "delivery_timestamp", "order_value"]
]
synthetic_orders_only.to_csv(f"{DATA_DIR}/synthetic_orders_ctgan_normal.csv", index=False)
print("Saved synthetic_orders_ctgan_normal.csv (original orders.csv schema):", synthetic_orders_only.shape)

# ------------------------------------------------------------------
# 3. THE CHECK THAT ACTUALLY MATTERS: did CTGAN preserve the
#    band -> order_value gradient, or average it away?
# ------------------------------------------------------------------
print("\nReal mean order_value by band (from 01's output: low~204, mid~484, high~1200):")
print(orders_sample.groupby("avg_order_value_band")["order_value"].mean().reindex(["low", "mid", "high"]))
print("\nSynthetic mean order_value by band:")
print(synthetic_ml.groupby("avg_order_value_band")["order_value"].mean().reindex(["low", "mid", "high"]))
print("""
If these two don't look reasonably similar (same ordering, roughly
comparable spread), CTGAN under-fit the relationship -- try more
epochs, a larger training sample, or fall back to the multi-table
HMASynthesizer approach from the earlier version of this script,
which models customers and orders as properly linked tables instead
of asking a single-table model to capture the joint distribution.
""")
