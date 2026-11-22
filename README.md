---
title: Quick E-Com Optimization and Demand Forecasting Console
emoji: 🛒
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Quick E-Com Optimization and Sales Forecasting

An end-to-end data science, machine learning, and operations research platform designed to optimize **Quick Commerce (10-minute grocery delivery)** operations across distributed dark stores.

The platform integrates:
- Realistic relational data simulation & conditional tabular synthesis (**SDV CTGAN**)
- Deep exploratory data analysis & geospatial store-level contribution analysis
- Time-series demand forecasting with **XGBoost** (85% Train / 15% Out-Of-Time Test)
- **Operations & SLA Optimization Suite**: Dynamic workforce staffing (riders & pickers), capacity bottleneck detection, and inventory policy calculation
- Production-ready dark-themed **Plotly Dash** web dashboard containerized with **Docker** & **Gunicorn**

---

## 📌 Architecture & Data Flow

```
                      ┌─────────────────────────────────┐
                      │   1. Base Relational Gen        │  (scripts/01_generate_base_data_normal.py)
                      └────────────────┬────────────────┘
                                       │
                      ┌────────────────▼────────────────┐
                      │    2. CTGAN Tabular Synthesis   │  (scripts/02_synthesize_with_sdv_normal.py)
                      └────────────────┬────────────────┘
                                       │
                      ┌────────────────▼────────────────┐
                      │ 3. Preprocessing & Clean Pipeline│ (notebooks/03_data_preprocessing.ipynb)
                      └────────────────┬────────────────┘
                                       │
                      ┌────────────────▼────────────────┐
                      │ 4. Exploratory Data Analysis    │  (notebooks/04_exploratory_data_analysis.ipynb)
                      └────────────────┬────────────────┘
                                       │
                      ┌────────────────▼────────────────┐
                      │  5. Time-Series ML Forecasting  │  (notebooks/05_demand_forecasting.ipynb &
                      └────────────────┬────────────────┘   scripts/train_forecasting_models.py)
                                       │
       ┌───────────────────────────────┴───────────────────────────────┐
       │                                                               │
┌──────▼────────────────────────────────┐              ┌───────────────▼────────────────────────┐
│   Interactive Plotly Dash Dashboard   │              │   Dark Store Operations & SLA Engine   │
│   (Overview • EDA • Model Evaluation) │              │   (Staffing • Bottlenecks • Capacity)  │
└───────────────────────────────────────┘              └────────────────────────────────────────┘
```

---

## 📂 Repository Layout

```text
Quick E-Com Optimization/
├── app.py                          # Production Dash application (with Gunicorn server export)
├── requirements.txt                # Complete project dependencies
├── Dockerfile                      # Production container configuration
├── docker-compose.yml              # Single-command Docker startup
├── Procfile                        # Cloud deployment config (Render / Railway / Heroku)
├── .gitignore                      # Git ignore rules
├── README.md                       # Complete technical documentation
│
├── assets/                         # Dashboard CSS styling
│   └── style.css
│
├── data/                           # Centralized data repository
│   ├── customers.csv               # 20,000 customer profiles (tiers & AOV bands)
│   ├── dark_stores.csv             # 15 Dark stores (city, zone, hourly capacity)
│   ├── products.csv                # SKU catalog & base prices
│   ├── orders.csv                  # Base simulated orders
│   ├── order_items.csv             # Transaction line items
│   ├── synthetic_orders_ctgan.csv  # High-fidelity CTGAN synthesized orders
│   ├── synthetic_orders_ml.csv     # Denormalized feature table
│   ├── cleaned_dataset.csv         # Processed dataset
│   └── eda_output.csv              # Enriched dataset with temporal & satisfaction flags
│
├── models/                         # Serialized models and evaluation metrics
│   ├── forecast_metrics.json       # Baseline vs. Tuned accuracy comparison
│   ├── xgb_baseline.json           # Serialized baseline model weights
│   └── xgb_tuned.json              # Serialized tuned model weights
│
├── notebooks/                      # Research & development notebooks
│   ├── 03_data_preprocessing.ipynb # Data cleaning, imputation, and feature creation
│   ├── 04_exploratory_data_analysis.ipynb # Statistical distributions & spatial EDA
│   └── 05_demand_forecasting.ipynb # XGBoost hyperparameter tuning & SARIMAX benchmarks
│
├── scripts/                        # Automation & training scripts
│   ├── 01_generate_base_data_normal.py  # Realistic multi-table generator
│   ├── 02_synthesize_with_sdv_normal.py # Conditional CTGAN synthesis script
│   └── train_forecasting_models.py      # Automated model training & serialization
│
└── src/                            # Core operations & optimization package
    ├── __init__.py
    └── optimization.py             # Staffing, capacity utilization & inventory policies
```

---

## ⚙️ Key Modules & Capabilities

### 1. Operations & SLA Optimization (`src/optimization.py`)
Quick commerce dark stores operate under rigid constraints: 10-minute delivery promise, fixed packing stations, and variable courier availability.
- **Rider & Picker Staffing Engine**: Dynamically calculates necessary workforce based on picking time ($T_{\text{pick}}$) and transit duration ($T_{\text{transit}}$):
  $$\text{Pickers Needed} = \left\lceil \frac{\text{Orders/Hr} \times T_{\text{pick}}}{60} \right\rceil, \quad \text{Riders Needed} = \left\lceil \frac{\text{Orders/Hr} \times T_{\text{transit}}}{60} \right\rceil$$
- **Capacity Bottleneck Detection**: Compares real-time peak order volume against store rated capacity from `dark_stores.csv` (`capacity_orders_per_hour`):
  - **Optimal (< 75%):** Safe operating margins.
  - **High Load (75% – 90%):** Warning for peak dinner hours (19:00 – 22:00).
  - **Critical SLA Breach (> 90%):** High probability of delivery delays and customer dissatisfaction.
- **Continuous Review Inventory Policy**: Computes Safety Stock ($SS$) and Reorder Point ($ROP$) based on demand variance $\sigma_d$, lead time $L$, and service level $Z$:
  $$SS = Z \times \sigma_d \times \sqrt{L}, \quad ROP = (\mu_d \times L) + SS$$

### 2. Time-Series Demand Forecasting (`notebooks/05_demand_forecasting.ipynb`)
- Chronological 85% Train / 15% Out-of-Time Test split to strictly prevent future-data leakage.
- Features: Autoregressive lags (`lag_1`, `lag_7`), rolling averages (`rolling_7`, `rolling_14`), calendar day-of-week, weekend indicator, and cyclical Fourier transforms (`weekday_sin`, `weekday_cos`).
- Evaluated on both operational supply chain metrics (WAPE & Forecast Accuracy) and traditional loss functions:
  - **Network Daily Volume Forecast:** **92.8% Forecast Accuracy** | **7.2% WAPE** | MAE ~326 | $R^2$ 0.42
  - **Tuned Regularized XGBoost:** **91.0% Forecast Accuracy** | **9.0% WAPE** | MAE ~404 | $R^2$ 0.37
  - **Store-Level Model:** MAE 26.82 | RMSE 34.19 | $R^2$ 0.8928 (+17.4% MAE reduction over uncalibrated baseline)
- **Methodology Note**: Retail demand forecasting prioritizes Forecast Accuracy ($1 - \text{WAPE}$) over $R^2$ because raw $R^2$ penalizes stationary count distributions and boundary outlier shocks. An operational accuracy of $>91\%$ provides reliable input bounds for downstream linear workforce scheduling.

### 3. Interactive Web Dashboard (`app.py`)
- **Executive Overview**: Total orders, revenue, AOV, delivery times, and SLA satisfaction.
- **Exploratory Data Analysis**: 4 sub-tabs covering distributions, feature correlation heatmaps, weekly seasonality, and spatial store contribution.
- **Demand Forecasting**: Interactive visualization comparing actual vs. baseline vs. tuned model forecasts, with dynamic accuracy KPI badges.
- **Operations & SLA Optimization**: Interactive sliders for picking and transit durations, real-time store capacity alert banners, hour-by-hour workforce staffing curves, and network-wide dark store bottleneck comparisons.

---

## 🚀 Quickstart & Setup

### Local Setup
```bash
# 1. Clone repository
git clone https://github.com/your-username/quick-ecom-optimization.git
cd "quick-ecom-optimization"

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
python app.py
```
Open **`http://127.0.0.1:8050/`** in your browser.

### Docker Deployment
```bash
# Build and run container with docker-compose
docker-compose up --build
```
The dashboard will be available at **`http://localhost:8050/`**.
