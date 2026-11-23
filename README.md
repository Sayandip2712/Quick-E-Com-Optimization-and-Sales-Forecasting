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
- Evaluated on operational supply chain metrics (WAPE & Forecast Accuracy) and traditional loss metrics:
  - **Tuned XGBoost (4-Fold TimeSeriesSplit CV):** **~94.0% Forecast Accuracy** | **~6.0% WAPE** | MAE ~247 | $R^2$ ~0.83
  - **Baseline XGBoost (Default Hyperparameters):** **94.1% Forecast Accuracy** | **5.9% WAPE** | MAE ~242 | $R^2$ 0.84
  - **Seasonal-Naive Benchmark (`lag_7`):** **79.6% Forecast Accuracy** | **20.4% WAPE** | MAE ~830
- **Methodology Note**: Retail demand forecasting prioritizes Forecast Accuracy ($1 - \text{WAPE}$) over $R^2$ because raw $R^2$ penalizes stationary count distributions. Evaluation uses an out-of-time next-day ($t+1$) test set excluding boundary shocks. Both XGBoost models achieve $\sim 94\%$ operational accuracy, substantially beating the seasonal-naive persistence baseline ($20.4\%$ WAPE).

### 3. Interactive Web Dashboard (`app.py`)
- **Executive Overview**: Total orders, revenue, AOV, delivery times, and SLA turnaround compliance.
- **Exploratory Data Analysis**: Sub-tabs covering distributions, feature correlation heatmaps (genuine operational metrics), delivery duration box plots, weekly seasonality, and spatial store contribution.
- **Demand Forecasting**: Interactive visualization comparing actual vs. baseline vs. cross-validated tuned XGBoost forecasts alongside the seasonal-naive benchmark, with dynamic accuracy KPI badges and parameter reporting.
- **Operations & SLA Optimization**: Interactive sliders for picking and transit durations, real-time store capacity alert banners (Critical/Watch/Optimal), hour-by-hour workforce staffing curves, and network-wide dark store bottleneck comparisons.

---

## 🚀 Quickstart & Setup

### Local Setup
```bash
# 1. Clone repository
git clone https://github.com/Sayandip2712/Quick-E-Com-Optimization-and-Sales-Forecasting.git
cd Quick-E-Com-Optimization-and-Sales-Forecasting

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

### PythonAnywhere Cloud Deployment
1. Create a free **Beginner account** on [PythonAnywhere](https://www.pythonanywhere.com/).
2. Open a **Bash console** and clone the repo:
   ```bash
   git clone --depth 1 https://github.com/Sayandip2712/Quick-E-Com-Optimization-and-Sales-Forecasting.git
   cd Quick-E-Com-Optimization-and-Sales-Forecasting
   rm data/cleaned_dataset.csv data/order_items.csv data/orders.csv data/synthetic_orders_*.csv
   python3.10 -m venv venv && source venv/bin/activate
   pip install --no-cache-dir dash dash-bootstrap-components plotly pandas numpy scikit-learn xgboost statsmodels
   ```
3. In the **Web tab**, create a Manual configuration app (Python 3.10), point the virtualenv and working directory to the project path, and set the WSGI configuration file:
   ```python
   from app import server as application
   ```
4. Click **Reload** to view your live app.
