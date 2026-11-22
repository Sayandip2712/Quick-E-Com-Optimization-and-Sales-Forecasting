"""
scripts/train_forecasting_models.py
-------------------------------------------------------------
Trains and serializes the baseline and tuned XGBoost demand forecasting models.
Saves model weights (.json) and evaluation metrics into the models/ directory
to enable instantaneous dashboard rendering without on-the-fly re-training.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "eda_output.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_and_save():
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cannot find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate daily order volume
    daily = df.set_index("date").resample("D").size().reset_index(name="order_count")
    daily = daily.sort_values("date")

    # Time-series feature engineering
    daily["lag_1"] = daily["order_count"].shift(1)
    daily["lag_7"] = daily["order_count"].shift(7)
    daily["rolling_7"] = daily["order_count"].shift(1).rolling(7).mean()
    daily["rolling_14"] = daily["order_count"].shift(1).rolling(14).mean()
    daily["weekday"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["weekday"].isin([5, 6]).astype(int)
    daily["weekday_sin"] = np.sin(2 * np.pi * daily["weekday"] / 7)
    daily["weekday_cos"] = np.cos(2 * np.pi * daily["weekday"] / 7)

    daily = daily.dropna().reset_index(drop=True)
    features = ["weekday", "is_weekend", "weekday_sin", "weekday_cos", "lag_1", "lag_7", "rolling_7", "rolling_14"]

    # Chronological 85% Train / 15% Test split
    split_idx = int(len(daily) * 0.85)
    train_df = daily.iloc[:split_idx]
    test_df = daily.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df["order_count"]
    X_test, y_test = test_df[features], test_df["order_count"]

    print(f"Training set: {len(X_train)} days | Test set: {len(X_test)} days")

    # 1. Baseline Model
    print("Fitting Baseline XGBoost...")
    base_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )
    base_model.fit(X_train, y_train)
    base_preds = base_model.predict(X_test)

    # 2. Tuned Model
    print("Fitting Tuned XGBoost...")
    tuned_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        objective="reg:squarederror",
        random_state=42
    )
    tuned_model.fit(X_train, y_train)
    tuned_preds = tuned_model.predict(X_test)

    # Calculate Evaluation Metrics
    def calc_metrics(y_true, y_pred):
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100)
        accuracy = max(0.0, 100.0 - wape)
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "wape": round(wape, 2),
            "accuracy": round(accuracy, 2),
            "mape": round(mape, 2)
        }

    base_metrics = calc_metrics(y_test, base_preds)
    tuned_metrics = calc_metrics(y_test, tuned_preds)

    print("\n--- Model Evaluation Summary ---")
    print(f"Baseline XGBoost -> Accuracy: {base_metrics['accuracy']}%, WAPE: {base_metrics['wape']}%, MAE: {base_metrics['mae']}, RMSE: {base_metrics['rmse']}, R2: {base_metrics['r2']}")
    print(f"Tuned XGBoost    -> Accuracy: {tuned_metrics['accuracy']}%, WAPE: {tuned_metrics['wape']}%, MAE: {tuned_metrics['mae']}, RMSE: {tuned_metrics['rmse']}, R2: {tuned_metrics['r2']}")

    # Serialize Models
    base_path = os.path.join(MODELS_DIR, "xgb_baseline.json")
    tuned_path = os.path.join(MODELS_DIR, "xgb_tuned.json")
    base_model.save_model(base_path)
    tuned_model.save_model(tuned_path)
    print(f"Saved models to {base_path} and {tuned_path}")

    # Serialize Metrics
    metrics_payload = {
        "features": features,
        "train_days": len(X_train),
        "test_days": len(X_test),
        "baseline": base_metrics,
        "tuned": tuned_metrics,
        "improvement": {
            "mae_reduction_pct": round(((base_metrics["mae"] - tuned_metrics["mae"]) / base_metrics["mae"]) * 100, 2),
            "rmse_reduction_pct": round(((base_metrics["rmse"] - tuned_metrics["rmse"]) / base_metrics["rmse"]) * 100, 2),
            "r2_gain": round(tuned_metrics["r2"] - base_metrics["r2"], 4)
        }
    }
    metrics_path = os.path.join(MODELS_DIR, "forecast_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"Saved metrics summary to {metrics_path}")


if __name__ == "__main__":
    train_and_save()
