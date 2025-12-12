# Retail Demand Forecasting & Inventory Optimization

**A complete end-to-end retail analytics project.**

This project implements a robust retail analytics pipeline that forecasts daily sales using a **LightGBM** time-series model and converts those predictions into actionable inventory decisions. It features a **Streamlit** dashboard for interactive planning, allowing users to simulate various supply chain scenarios and export optimized inventory plans.

---

## 🚀 Features

### 1. Demand Forecasting
* **Target:** Rossmann daily store-level sales.
* **Validation:** Time-based split (validating on the last 6 weeks of data).
* **Feature Engineering:**
    * **Calendar:** DayOfWeek, Week, Month, Weekend flags.
    * **Events:** Public holidays, School holidays, Promotions.
    * **Competition:** Distance metrics and "Promo2" timings.
    * **Lag & Rolling:** Lag features (t-1, t-7, t-14) and Rolling statistics (Mean/Std for 7, 14, 28 days).
* **Model:** LightGBM Regressor trained on `log1p(Sales)`.

### 2. Inventory Optimization
* **Uncertainty Modeling:** Per-store error ($\sigma$) estimated from validation residuals.
* **Policy Logic:** Calculates critical supply chain metrics:
    * **Safety Stock:** Dynamic buffer based on lead time and review period.
    * **Reorder Point (ROP):** Trigger level for new orders.
    * **Order-Up-To Level (S):** Target inventory ceiling.
    * **Recommended Order Quantity:** The specific amount to buy *now*.
* **Service Levels:** Supports dynamic targets (90%, 95%, 97%, 99%).

### 3. Interactive Dashboard (Streamlit)
* **Visuals:** Actual vs. Forecast plots with "Low Forecast" handling (closed days).
* **Controls:** Store selector and "What-If" parameters (Lead Time, Review Period, Current Stock).
* **Exports:**
    * Store-specific forecast CSV.
    * System-wide "All-Stores" inventory plan.

---

## 🛠 Tech Stack

* **Core:** Python, Pandas, NumPy
* **ML:** LightGBM, Scikit-learn
* **Math:** SciPy (Probability functions/z-scores)
* **App:** Streamlit
* **Data:** Parquet (PyArrow), Joblib (Model persistence)

---

## 📂 Project Structure

```text
retail_forecast_inventory/
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── artifacts/                  # Trained models and metadata
│   └── rossmann_v1/
│       ├── lgbm_model.joblib   # Serialized LightGBM model
│       ├── meta.json           # Model metadata/params
│       ├── sigma.json          # Pre-calculated error metrics per store
│       └── submission.csv      # Kaggle-style submission file
├── data/                       # Data storage
│   └── processed/
│       └── rossmann/
│           ├── train_merged_clean.parquet
│           └── test_merged.parquet
└── src/
    └── inventory.py            # Inventory logic and math functions