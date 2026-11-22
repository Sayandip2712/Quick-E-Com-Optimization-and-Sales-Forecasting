"""
01_generate_base_data_normal.py
-------------------------------------------------------------
Quick-commerce base data generator -- ML-ready version.

Changes from the previous normal-demand version:
  - `is_event_window` and `match_phase` are GONE. Not constant, not
    present at all. There's no event in this dataset, so there's no
    reason to carry event columns.
  - order_value is no longer just "sum of randomly picked items" --
    it now has REAL, learnable structure so a model trained on it can
    achieve good (not trivial, not random) predictive accuracy:
      * customers with avg_order_value_band="high" genuinely buy more
        items AND skew toward pricier items; "low" band customers buy
        fewer, cheaper items
      * metro customers buy slightly more per order than non-metro
      * baskets are larger at dinner time (19:00-22:00) and on weekends
    All of this is real, deterministic-but-noisy structure (Poisson/
    multinomial randomness on top of a genuine mean effect) -- exactly
    the kind of pattern gradient boosting is good at recovering, and
    exactly the kind of pattern pure noise would NOT let it recover.
      This is a deliberate modeling choice: it is what makes
    order_value predictable enough for XGBoost/CatBoost to "give good
    results" (i.e., decent R²/MAE) rather than fitting noise (R² near 0).

Outputs (into ./data/):
  - dark_stores.csv, customers.csv, products.csv : unchanged structure
  - orders.csv       : order_id, customer_id, store_id, order_timestamp,
                        delivery_timestamp, order_value  -- NO event columns
  - order_items.csv  : unchanged structure
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

# ------------------------------------------------------------------
# 1. DARK STORES
# ------------------------------------------------------------------
CITIES = ["Mumbai", "Bengaluru", "Delhi NCR", "Hyderabad", "Pune"]
N_STORES = 15

dark_stores = pd.DataFrame({
    "store_id": [f"DS{str(i).zfill(3)}" for i in range(1, N_STORES + 1)],
    "city": RNG.choice(CITIES, N_STORES),
})
dark_stores["zone"] = dark_stores["city"].str[:3].str.upper() + "-" + \
    (RNG.integers(1, 5, N_STORES)).astype(str)
dark_stores["capacity_orders_per_hour"] = RNG.integers(40, 120, N_STORES)
dark_stores["launch_date"] = pd.to_datetime("2024-01-01") + \
    pd.to_timedelta(RNG.integers(0, 500, N_STORES), unit="D")

# ------------------------------------------------------------------
# 2. CUSTOMERS
# ------------------------------------------------------------------
N_CUSTOMERS = 20_000
tiers = RNG.choice(["metro", "non_metro"], N_CUSTOMERS, p=[0.75, 0.25])
customers = pd.DataFrame({
    "customer_id": [f"C{str(i).zfill(6)}" for i in range(1, N_CUSTOMERS + 1)],
    "city": RNG.choice(CITIES, N_CUSTOMERS),
    "tier": tiers,
    "signup_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(RNG.integers(0, 900, N_CUSTOMERS), unit="D"),
    "avg_order_value_band": RNG.choice(["low", "mid", "high"], N_CUSTOMERS, p=[0.4, 0.45, 0.15]),
})

# Fast per-customer lookup for the order-generation loop below
cust_band_map = dict(zip(customers.customer_id, customers.avg_order_value_band))
cust_tier_map = dict(zip(customers.customer_id, customers.tier))

# ------------------------------------------------------------------
# 3. PRODUCT CATALOG (unchanged -- fixed catalog prices, no per-order
#    price inflation; basket SIZE and COMPOSITION vary instead, which
#    is the realistic way real quick-commerce order values differ)
# ------------------------------------------------------------------
PRODUCTS = [
    ("Potato Chips 90g", "snacks", 20, True),
    ("Nachos 150g", "snacks", 60, True),
    ("Mixed Namkeen 200g", "snacks", 55, True),
    ("Frozen Momos 12pc", "instant_food", 130, True),
    ("Instant Noodles 4-pack", "instant_food", 60, True),
    ("Cola 750ml", "beverages", 40, True),
    ("Beer (non-alc) 330ml", "beverages", 60, True),
    ("Energy Drink 250ml", "beverages", 45, True),
    ("Ice Cream Tub 500ml", "frozen_desserts", 150, True),
    ("Chicken Wings Frozen 500g", "frozen_meat", 220, True),
    ("Toned Milk 500ml", "dairy", 28, False),
    ("Bread 400g", "bakery", 45, False),
    ("Eggs (6pc)", "dairy", 42, False),
    ("Onions 1kg", "produce", 35, False),
    ("Tomatoes 1kg", "produce", 30, False),
    ("Basmati Rice 1kg", "staples", 90, False),
    ("Toor Dal 1kg", "staples", 140, False),
    ("Dish Wash Liquid 500ml", "household", 99, False),
    ("Detergent Powder 1kg", "household", 120, False),
    ("Shampoo 180ml", "personal_care", 180, False),
]
products = pd.DataFrame(PRODUCTS, columns=["product_name", "category", "price", "event_sensitive"])
products.insert(0, "product_id", [f"P{str(i).zfill(4)}" for i in range(1, len(products) + 1)])

print(f"dark_stores: {dark_stores.shape}, customers: {customers.shape}, products: {products.shape}")

# ------------------------------------------------------------------
# 4. HOURLY DEMAND INTENSITY -- plain seasonality, no event term
# ------------------------------------------------------------------
SIM_START = pd.to_datetime("2026-01-01")
SIM_END = pd.to_datetime("2026-07-30 23:00:00")
hours = pd.date_range(SIM_START, SIM_END, freq="h")

def daily_baseline(hour_of_day: np.ndarray) -> np.ndarray:
    bumps = [(8, 1.3, 1.2), (13, 1.5, 1.3), (18, 1.8, 1.5), (21, 2.2, 1.6)]
    val = np.full_like(hour_of_day, 0.3, dtype=float)
    for center, width, height in bumps:
        val += height * np.exp(-0.5 * ((hour_of_day - center) / width) ** 2)
    return val

hour_of_day_arr = hours.hour.values
dow_arr = hours.dayofweek.values
weekend_mult = np.where(dow_arr >= 5, 1.25, 1.0)
trend = 1.0 + 0.0015 * np.arange(len(hours))
baseline_intensity = daily_baseline(hour_of_day_arr) * weekend_mult * trend

hourly_df = pd.DataFrame({
    "timestamp": hours,
    "hour_of_day": hour_of_day_arr,
    "is_weekend": dow_arr >= 5,
    "demand_intensity": baseline_intensity,
})

# ------------------------------------------------------------------
# 5. ITEM-SELECTION WEIGHTS BY CUSTOMER BAND (this is the core of the
#    "learnable structure" -- high-band customers see prices weighted
#    UP, low-band customers see prices weighted DOWN, mid-band is flat)
# ------------------------------------------------------------------
event_sensitive_products = products[products.event_sensitive].product_id.tolist()
regular_products = products[~products.event_sensitive].product_id.tolist()
items_all = event_sensitive_products + regular_products
price_map = dict(zip(products.product_id, products.price))
prices_arr = np.array([price_map[p] for p in items_all], dtype=float)

def price_weighted_probs(exponent: float) -> np.ndarray:
    """exponent>0 skews toward pricier items, <0 toward cheaper, 0=uniform."""
    w = prices_arr ** exponent
    return w / w.sum()

ITEM_PROBS_BY_BAND = {
    "low":  price_weighted_probs(-0.9),
    "mid":  price_weighted_probs(0.0),
    "high": price_weighted_probs(0.9),
}

# Mean basket size (Poisson lambda) by band -- high band genuinely buys
# more items per order, not just pricier ones. Separation widened (and
# qty range narrowed below) vs. an earlier draft after checking with an
# actual HistGradientBoostingRegressor fit: this combination gets a
# realistic R^2 ~ 0.55-0.6 on held-out data, vs ~0.26 before -- a much
# more convincing "good result" while still leaving genuine noise (not
# a suspicious R^2 ~1.0).
BASE_ITEM_LAMBDA = {"low": 1.0, "mid": 2.5, "high": 5.5}
DINNER_HOURS = {19, 20, 21, 22}

# ------------------------------------------------------------------
# 6. SAMPLE ORDER-LEVEL ROWS
# ------------------------------------------------------------------
BASE_ORDERS_PER_STORE_PER_HOUR = 1
order_rows, item_rows = [], []
order_counter = item_counter = 1

for _, hrow in hourly_df.iterrows():
    lam = hrow["demand_intensity"] * BASE_ORDERS_PER_STORE_PER_HOUR
    is_dinner = hrow["hour_of_day"] in DINNER_HOURS
    for store_id in dark_stores.store_id:
        n_orders = RNG.poisson(lam)
        if n_orders == 0:
            continue
        cust_ids = RNG.choice(customers.customer_id.values, n_orders)
        for cust_id in cust_ids:
            band = cust_band_map[cust_id]
            tier = cust_tier_map[cust_id]

            order_id = f"O{order_counter:08d}"
            order_counter += 1
            offset_min = RNG.integers(0, 60)
            order_ts = hrow["timestamp"] + pd.Timedelta(minutes=int(offset_min))
            prep_min = RNG.integers(4, 12)
            delivery_ts = order_ts + pd.Timedelta(minutes=int(prep_min))

            # --- genuine, learnable basket-size effect ---
            item_lambda = BASE_ITEM_LAMBDA[band]
            if hrow["is_weekend"]:
                item_lambda += 0.4
            if is_dinner:
                item_lambda += 0.4
            if tier == "metro":
                item_lambda += 0.2
            n_items = RNG.poisson(item_lambda) + 1  # at least 1 item

            item_pool = RNG.choice(items_all, n_items, p=ITEM_PROBS_BY_BAND[band])

            order_value = 0.0
            for pid in item_pool:
                qty = int(RNG.integers(1, 3))  # 1-2 units per line item (was 1-3)
                price = price_map[pid]
                item_rows.append({
                    "order_item_id": f"OI{item_counter:09d}",
                    "order_id": order_id,
                    "product_id": pid,
                    "qty": qty,
                    "unit_price": price,
                })
                item_counter += 1
                order_value += qty * price

            order_rows.append({
                "order_id": order_id,
                "customer_id": cust_id,
                "store_id": store_id,
                "order_timestamp": order_ts,
                "delivery_timestamp": delivery_ts,
                "order_value": round(order_value, 2),
            })

print(f"Generated {len(order_rows):,} orders and {len(item_rows):,} order items")

orders = pd.DataFrame(order_rows)
order_items = pd.DataFrame(item_rows)

# ------------------------------------------------------------------
# 7. SAVE
# ------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)
dark_stores.to_csv(f"{OUT_DIR}/dark_stores.csv", index=False)
customers.to_csv(f"{OUT_DIR}/customers.csv", index=False)
products.to_csv(f"{OUT_DIR}/products.csv", index=False)
orders.to_csv(f"{OUT_DIR}/orders.csv", index=False)
order_items.to_csv(f"{OUT_DIR}/order_items.csv", index=False)

print("\nSaved to:", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    print(" -", f)
print("\norders.csv columns:", list(orders.columns))

# ------------------------------------------------------------------
# 8. VERIFY the learnable structure actually exists (not just claimed)
# ------------------------------------------------------------------
check = orders.merge(customers[["customer_id", "avg_order_value_band", "tier"]], on="customer_id")
print("\nMean order_value by customer band (should clearly increase low -> high):")
print(check.groupby("avg_order_value_band")["order_value"].mean().reindex(["low", "mid", "high"]))
print("\nMean order_value by tier:")
print(check.groupby("tier")["order_value"].mean())
check["hour"] = check.order_timestamp.dt.hour
check["is_dinner"] = check.hour.isin(DINNER_HOURS)
print("\nMean order_value: dinner-hour orders vs. other hours:")
print(check.groupby("is_dinner")["order_value"].mean())
