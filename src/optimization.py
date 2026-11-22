"""
src/optimization.py
-------------------------------------------------------------
Operations Research & Dark Store Optimization Module
Provides algorithms for:
  1. Rider & In-Store Picker Staffing under strict SLA (10-minute delivery).
  2. Dark Store Capacity & Bottleneck Risk Evaluation.
  3. Inventory Safety Stock & Reorder Point (ROP) Policy.
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_hourly_staffing(hourly_order_volume, avg_pick_minutes=3.0, avg_delivery_minutes=7.0):
    """
    Computes required warehouse floor pickers and delivery fleet riders per hour
    to guarantee SLA compliance (orders dispatched and delivered within target threshold).

    Parameters:
        hourly_order_volume (int, float, or pd.Series): Number of incoming orders in the hour.
        avg_pick_minutes (float): Average time for picking, packing & bagging an order (default: 3 min).
        avg_delivery_minutes (float): Average transit/doorstep time per rider trip (default: 7 min).

    Returns:
        dict or pd.DataFrame: Required staffing counts and capacity utilization.
    """
    is_series = isinstance(hourly_order_volume, (pd.Series, np.ndarray, list))
    orders = np.array(hourly_order_volume, dtype=float)

    # 1 worker-hour = 60 working minutes
    # Pickers required = ceil((Orders * Pick_Time) / 60)
    # Riders required = ceil((Orders * Delivery_Time) / 60)
    pickers_needed = np.ceil((orders * avg_pick_minutes) / 60.0).astype(int)
    riders_needed = np.ceil((orders * avg_delivery_minutes) / 60.0).astype(int)

    # Avoid zero staffing during operating hours if there's any demand
    pickers_needed = np.where((orders > 0) & (pickers_needed < 1), 1, pickers_needed)
    riders_needed = np.where((orders > 0) & (riders_needed < 1), 1, riders_needed)
    total_staff = pickers_needed + riders_needed

    if is_series:
        return pd.DataFrame({
            "order_volume": orders,
            "pickers_needed": pickers_needed,
            "riders_needed": riders_needed,
            "total_staff_needed": total_staff
        })
    else:
        return {
            "order_volume": float(orders),
            "pickers_needed": int(pickers_needed),
            "riders_needed": int(riders_needed),
            "total_staff_needed": int(total_staff)
        }


def evaluate_dark_store_capacity(store_id, store_hourly_df, dark_stores_df):
    """
    Evaluates whether a dark store's order volume breaches its hourly operational capacity.

    Parameters:
        store_id (str): Store identifier (e.g., 'DS001') or 'ALL'.
        store_hourly_df (pd.DataFrame): Dataframe with 'hour_of_day' and 'order_count'.
        dark_stores_df (pd.DataFrame): Dataframe containing 'store_id' and 'capacity_orders_per_hour'.

    Returns:
        dict: Detailed capacity metrics, utilization rate, bottleneck hours, and risk level.
    """
    if store_id == "ALL":
        # Aggregate capacity across all active dark stores
        total_capacity = int(dark_stores_df["capacity_orders_per_hour"].sum())
        city = "All Networks (15 Stores)"
        zone = "Multi-Zone"
    else:
        store_row = dark_stores_df[dark_stores_df["store_id"] == store_id]
        if store_row.empty:
            total_capacity = 90  # Default fallback if store not found
            city, zone = "Unknown", "Unknown"
        else:
            total_capacity = int(store_row["capacity_orders_per_hour"].values[0])
            city = str(store_row["city"].values[0])
            zone = str(store_row["zone"].values[0])

    if store_hourly_df.empty:
        return {
            "store_id": store_id,
            "city": city,
            "zone": zone,
            "capacity_per_hour": total_capacity,
            "peak_hourly_orders": 0,
            "avg_hourly_orders": 0,
            "peak_utilization_pct": 0.0,
            "risk_level": "Optimal",
            "bottleneck_hours": []
        }

    peak_hourly_orders = int(store_hourly_df["order_count"].max())
    avg_hourly_orders = float(store_hourly_df["order_count"].mean())
    peak_utilization_pct = round((peak_hourly_orders / total_capacity) * 100, 1)

    # Identify hours operating above 85% capacity threshold
    bottleneck_rows = store_hourly_df[store_hourly_df["order_count"] >= (total_capacity * 0.85)]
    bottleneck_hours = sorted(bottleneck_rows["hour_of_day"].tolist()) if not bottleneck_rows.empty else []

    if peak_utilization_pct > 95.0:
        risk_level = "Critical SLA Breach Risk"
        risk_color = "danger"
    elif peak_utilization_pct >= 80.0:
        risk_level = "High Load Warning"
        risk_color = "warning"
    else:
        risk_level = "Healthy / Optimal"
        risk_color = "success"

    return {
        "store_id": store_id,
        "city": city,
        "zone": zone,
        "capacity_per_hour": total_capacity,
        "peak_hourly_orders": peak_hourly_orders,
        "avg_hourly_orders": round(avg_hourly_orders, 1),
        "peak_utilization_pct": peak_utilization_pct,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "bottleneck_hours": bottleneck_hours
    }


def calculate_inventory_policy(daily_demand_series, lead_time_days=2, target_service_level=0.95):
    """
    Computes Safety Stock and Reorder Point (ROP) under continuous review policy.

    Formula:
        Safety Stock (SS) = Z * sigma_d * sqrt(Lead_Time)
        Reorder Point (ROP) = (mu_d * Lead_Time) + SS

    Parameters:
        daily_demand_series (pd.Series or list): Historical daily order counts.
        lead_time_days (int): Supplier replenishment turnaround in days.
        target_service_level (float): Desired probability of not stocking out (default 0.95).

    Returns:
        dict: Policy parameters (mean demand, safety stock, ROP, stockout probability).
    """
    demand = np.array(daily_demand_series, dtype=float)
    if len(demand) < 2:
        return {
            "mean_daily_demand": 0.0,
            "std_daily_demand": 0.0,
            "safety_stock": 0,
            "reorder_point": 0,
            "target_service_level": target_service_level
        }

    mu_d = float(np.mean(demand))
    sigma_d = float(np.std(demand, ddof=1))
    z_score = float(stats.norm.ppf(target_service_level))

    safety_stock = int(np.ceil(z_score * sigma_d * np.sqrt(lead_time_days)))
    lead_time_demand = int(np.ceil(mu_d * lead_time_days))
    reorder_point = lead_time_demand + safety_stock

    return {
        "mean_daily_demand": round(mu_d, 1),
        "std_daily_demand": round(sigma_d, 1),
        "lead_time_days": lead_time_days,
        "target_service_level_pct": round(target_service_level * 100, 1),
        "safety_stock_units": safety_stock,
        "lead_time_demand_units": lead_time_demand,
        "reorder_point_units": reorder_point
    }
