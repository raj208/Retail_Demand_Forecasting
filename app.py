# import json
# import numpy as np
# import pandas as pd
# import streamlit as st

# from src.inventory import inventory_decision

# ART = "artifacts/rossmann_v1"
# DATA_DIR = "data/processed/rossmann"

# st.set_page_config(page_title="Retail Forecast + Inventory", layout="wide")

# @st.cache_data
# def load_artifacts():
#     meta = json.load(open(f"{ART}/meta.json", "r"))
#     sigma = json.load(open(f"{ART}/sigma.json", "r"))
#     sub = pd.read_csv(f"{ART}/submission.csv")
#     return meta, sigma, sub

# @st.cache_data
# def load_data():
#     train = pd.read_parquet(f"{DATA_DIR}/train_merged_clean.parquet")
#     test = pd.read_parquet(f"{DATA_DIR}/test_merged.parquet")
#     train["Date"] = pd.to_datetime(train["Date"])
#     test["Date"] = pd.to_datetime(test["Date"])
#     return train, test

# def get_sigma(sigma_payload, store_id: int) -> float:
#     # sigma_store keys are strings
#     s = sigma_payload["sigma_store"].get(str(store_id), None)
#     return float(s) if s is not None else float(sigma_payload["sigma_global"])

# meta, sigma_payload, sub = load_artifacts()
# train, test = load_data()

# # Merge forecasts into test
# test_pred = test.merge(sub, on="Id", how="left")
# test_pred = test_pred.rename(columns={"Sales": "ForecastSales"})
# test_pred["ForecastSales"] = test_pred["ForecastSales"].fillna(0.0)

# stores = sorted(test_pred["Store"].unique().tolist())

# st.title("Retail Demand Forecasting & Inventory Optimization")
# st.caption("Forecasts are precomputed from the trained LightGBM model and merged with the official future dates (test set).")

# with st.sidebar:
#     st.header("Controls")
#     store_id = st.selectbox("Select Store", stores, index=0)

#     st.subheader("Inventory Inputs")
#     lead_time = st.number_input("Lead time (days)", min_value=1, max_value=60, value=7, step=1)
#     review_days = st.number_input("Review period (days)", min_value=0, max_value=60, value=7, step=1)
#     service = st.selectbox("Service level", [0.90, 0.95, 0.97, 0.99], index=1)

#     on_hand = st.number_input("On-hand stock (units)", min_value=0.0, value=0.0, step=10.0)
#     on_order = st.number_input("On-order stock (units)", min_value=0.0, value=0.0, step=10.0)

#     horizon = st.slider("Show forecast horizon (days)", min_value=7, max_value=42, value=14, step=1)

# store_sigma = get_sigma(sigma_payload, store_id)

# store_test = test_pred[test_pred["Store"] == store_id].sort_values("Date").reset_index(drop=True)
# store_train = train[train["Store"] == store_id].sort_values("Date").reset_index(drop=True)

# # Select forecast window
# store_future = store_test.head(int(horizon)).copy()
# forecast_series = store_future["ForecastSales"].to_numpy(dtype=float)

# decision = inventory_decision(
#     forecast_series=forecast_series,
#     sigma_daily=store_sigma,
#     lead_time_days=int(lead_time),
#     review_days=int(review_days),
#     service_level=float(service),
#     on_hand=float(on_hand),
#     on_order=float(on_order)
# )

# c1, c2, c3, c4 = st.columns(4)
# c1.metric("σ (daily uncertainty)", f"{store_sigma:,.2f}")
# c2.metric("Reorder Point (ROP)", f"{decision.reorder_point:,.2f}")
# c3.metric("Safety Stock (PP)", f"{decision.safety_stock:,.2f}")
# c4.metric("Recommended Order Qty", f"{decision.order_qty:,.2f}")

# st.divider()

# left, right = st.columns([1.2, 1])

# with left:
#     st.subheader("Forecast (future days)")
#     show_df = store_future[["Date"]].copy()
#     show_df["ForecastSales"] = store_future["ForecastSales"].round(2)
#     if "Open" in store_future.columns:
#         show_df["Open"] = store_future["Open"].astype(int)
#     if "Promo" in store_future.columns:
#         show_df["Promo"] = store_future["Promo"].astype(int)

#     st.dataframe(show_df, use_container_width=True)

#     csv_bytes = show_df.to_csv(index=False).encode("utf-8")
#     st.download_button("Download forecast CSV", data=csv_bytes, file_name=f"store_{store_id}_forecast.csv", mime="text/csv")

# with right:
#     st.subheader("Actual vs Forecast (visual)")
#     # last 90 actual days + available future forecast days
#     actual_tail = store_train.tail(90)[["Date", "Sales"]].rename(columns={"Sales": "ActualSales"})
#     forecast_part = store_test.head(42)[["Date", "ForecastSales"]]

#     plot_df = pd.merge(actual_tail, forecast_part, on="Date", how="outer").sort_values("Date")
#     plot_df = plot_df.set_index("Date")

#     # Streamlit line chart expects numeric columns
#     st.line_chart(plot_df[["ActualSales", "ForecastSales"]])

# st.subheader("Inventory Calculation Details")
# details = pd.DataFrame([{
#     "LeadTimeDays": lead_time,
#     "ReviewDays": review_days,
#     "ProtectionPeriodDays": lead_time + review_days,
#     "ServiceLevel": service,
#     "z_value": decision.z_value,
#     "Demand_LeadTime": decision.demand_lead_time,
#     "Demand_ProtectionPeriod": decision.demand_protection_period,
#     "SafetyStock(PP)": decision.safety_stock,
#     "ROP": decision.reorder_point,
#     "OrderUpTo": decision.order_up_to,
#     "OnHand": on_hand,
#     "OnOrder": on_order,
#     "RecommendedOrderQty": decision.order_qty
# }])
# st.dataframe(details, use_container_width=True)
import json
import numpy as np
import pandas as pd
import streamlit as st

from src.inventory import inventory_decision

ART = "artifacts/rossmann_v1"
DATA_DIR = "data/processed/rossmann"

st.set_page_config(page_title="Retail Forecast + Inventory", layout="wide")


@st.cache_data
def load_artifacts():
    meta = json.load(open(f"{ART}/meta.json", "r"))
    sigma = json.load(open(f"{ART}/sigma.json", "r"))
    sub = pd.read_csv(f"{ART}/submission.csv")
    return meta, sigma, sub


@st.cache_data
def load_data():
    train = pd.read_parquet(f"{DATA_DIR}/train_merged_clean.parquet")
    test = pd.read_parquet(f"{DATA_DIR}/test_merged.parquet")
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])
    return train, test


@st.cache_data
def build_test_pred(test_df: pd.DataFrame, sub_df: pd.DataFrame) -> pd.DataFrame:
    df = test_df.merge(sub_df, on="Id", how="left")
    df = df.rename(columns={"Sales": "ForecastSales"})
    df["ForecastSales"] = df["ForecastSales"].fillna(0.0)
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def get_sigma(sigma_payload, store_id: int) -> float:
    s = sigma_payload["sigma_store"].get(str(store_id), None)
    return float(s) if s is not None else float(sigma_payload["sigma_global"])


def store_summary_row(df_store: pd.DataFrame) -> dict:
    """Pick stable store attributes from the first row in test data."""
    r = df_store.iloc[0]
    keys = [
        "StoreType", "Assortment", "CompetitionDistance",
        "Promo2", "PromoInterval",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2SinceWeek", "Promo2SinceYear"
    ]
    out = {}
    for k in keys:
        if k in df_store.columns:
            out[k] = r[k]
    return out


@st.cache_data
def build_inventory_plan_csv(
    test_pred_df: pd.DataFrame,
    sigma_payload: dict,
    lead_time: int,
    review_days: int,
    service: float,
    default_on_hand: float,
    default_on_order: float
) -> bytes:
    rows = []
    PP = int(lead_time) + int(review_days)

    # For each store, take first PP future days (test period is already "future")
    for store_id, g in test_pred_df.groupby("Store"):
        g = g.sort_values("Date").reset_index(drop=True)
        forecast = g["ForecastSales"].to_numpy(dtype=float)

        sigma_daily = get_sigma(sigma_payload, int(store_id))

        dec = inventory_decision(
            forecast_series=forecast[:PP],
            sigma_daily=sigma_daily,
            lead_time_days=int(lead_time),
            review_days=int(review_days),
            service_level=float(service),
            on_hand=float(default_on_hand),
            on_order=float(default_on_order),
        )

        # For reporting
        forecast_pp = forecast[:PP]
        forecast_L = forecast[: int(lead_time)]
        rows.append({
            "Store": int(store_id),
            "sigma_daily": sigma_daily,
            "LeadTimeDays": int(lead_time),
            "ReviewDays": int(review_days),
            "ProtectionPeriodDays": int(PP),
            "ServiceLevel": float(service),
            "z_value": dec.z_value,
            "ForecastSum_LeadTime": float(np.sum(forecast_L)),
            "ForecastSum_PP": float(np.sum(forecast_pp)),
            "SafetyStock_PP": float(dec.safety_stock),
            "ROP": float(dec.reorder_point),
            "OrderUpTo": float(dec.order_up_to),
            "DefaultOnHand": float(default_on_hand),
            "DefaultOnOrder": float(default_on_order),
            "RecommendedOrderQty": float(dec.order_qty),
        })

    plan = pd.DataFrame(rows).sort_values("RecommendedOrderQty", ascending=False)
    return plan.to_csv(index=False).encode("utf-8")


# ---------- App ----------
meta, sigma_payload, sub = load_artifacts()
train, test = load_data()
test_pred = build_test_pred(test, sub)

stores = sorted(test_pred["Store"].unique().tolist())

st.title("Retail Demand Forecasting & Inventory Optimization")
st.caption("Forecasts are precomputed from LightGBM; inventory decisions use service level + forecast uncertainty (σ).")

with st.sidebar:
    st.header("Controls")

    store_id = st.selectbox("Select Store", stores, index=0)

    st.subheader("Forecast Display")
    horizon = st.slider("Show forecast horizon (days)", min_value=7, max_value=42, value=14, step=1)
    hide_closed = st.checkbox("Hide closed days (table only)", value=True)

    st.subheader("Inventory Inputs")
    lead_time = st.number_input("Lead time (days)", min_value=1, max_value=60, value=7, step=1)
    review_days = st.number_input("Review period (days)", min_value=0, max_value=60, value=7, step=1)
    service = st.selectbox("Service level", [0.90, 0.95, 0.97, 0.99], index=1)

    on_hand = st.number_input("On-hand stock (units)", min_value=0.0, value=0.0, step=10.0)
    on_order = st.number_input("On-order stock (units)", min_value=0.0, value=0.0, step=10.0)

    st.divider()
    st.subheader("Export Inventory Plan (All Stores)")
    st.caption("Uses the same inputs above as defaults for every store.")
    export_pp = int(lead_time) + int(review_days)
    st.write(f"Protection period = **{export_pp} days**")

    csv_bytes = build_inventory_plan_csv(
        test_pred_df=test_pred,
        sigma_payload=sigma_payload,
        lead_time=int(lead_time),
        review_days=int(review_days),
        service=float(service),
        default_on_hand=float(on_hand),
        default_on_order=float(on_order),
    )
    st.download_button(
        "Download inventory plan CSV",
        data=csv_bytes,
        file_name="inventory_plan_all_stores.csv",
        mime="text/csv"
    )

# Store slices
store_sigma = get_sigma(sigma_payload, int(store_id))

store_test = test_pred[test_pred["Store"] == store_id].sort_values("Date").reset_index(drop=True)
store_train = train[train["Store"] == store_id].sort_values("Date").reset_index(drop=True)

# Store summary card (10A)
summary = store_summary_row(store_test)
summary_df = pd.DataFrame([summary])

# Forecast window + KPIs (10A)
store_future = store_test.head(int(horizon)).copy()
forecast_series = store_future["ForecastSales"].to_numpy(dtype=float)

sum7 = float(np.sum(forecast_series[:7]))
sum14 = float(np.sum(forecast_series[:14]))
avg_day = float(np.mean(forecast_series)) if len(forecast_series) else 0.0
max_day = float(np.max(forecast_series)) if len(forecast_series) else 0.0

# Inventory decision
decision = inventory_decision(
    forecast_series=forecast_series,              # includes zeros for Open=0 (correct)
    sigma_daily=store_sigma,
    lead_time_days=int(lead_time),
    review_days=int(review_days),
    service_level=float(service),
    on_hand=float(on_hand),
    on_order=float(on_order)
)

# Top metrics row
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("σ (daily uncertainty)", f"{store_sigma:,.2f}")
c2.metric("Forecast sum (7d)", f"{sum7:,.0f}")
c3.metric("Forecast sum (14d)", f"{sum14:,.0f}")
c4.metric("Avg / day", f"{avg_day:,.0f}")
c5.metric("Max / day", f"{max_day:,.0f}")
c6.metric("Recommended Order Qty", f"{decision.order_qty:,.0f}")

# Store summary + inventory metrics
left_top, right_top = st.columns([1.2, 1])
with left_top:
    st.subheader("Store Summary")
    st.dataframe(summary_df, use_container_width=True)

with right_top:
    st.subheader("Inventory Outputs")
    o1, o2, o3 = st.columns(3)
    o1.metric("Reorder Point (ROP)", f"{decision.reorder_point:,.0f}")
    o2.metric("Safety Stock (PP)", f"{decision.safety_stock:,.0f}")
    o3.metric("Order-up-to Level (S)", f"{decision.order_up_to:,.0f}")

st.divider()

# Table + chart
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Forecast (future days)")
    show_df = store_future[["Date"]].copy()
    show_df["ForecastSales"] = store_future["ForecastSales"].round(2)

    if "Open" in store_future.columns:
        show_df["Open"] = store_future["Open"].astype(int)
    if "Promo" in store_future.columns:
        show_df["Promo"] = store_future["Promo"].astype(int)

    if hide_closed and "Open" in show_df.columns:
        show_df = show_df[show_df["Open"] == 1].copy()

    st.dataframe(show_df, use_container_width=True)

    csv_store = show_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download store forecast CSV",
        data=csv_store,
        file_name=f"store_{store_id}_forecast.csv",
        mime="text/csv"
    )

with right:
    st.subheader("Actual vs Forecast (visual)")
    actual_tail = store_train.tail(90)[["Date", "Sales"]].rename(columns={"Sales": "ActualSales"})
    forecast_part = store_test.head(42)[["Date", "ForecastSales"]]


    #GAP FOR WEEKENDS
    # forecast_part = store_test.head(42)[["Date", "ForecastSales", "Open"]].copy()
    # forecast_part.loc[forecast_part["Open"] == 0, "ForecastSales"] = np.nan
    # forecast_part = forecast_part[["Date", "ForecastSales"]]
    plot_df = pd.merge(actual_tail, forecast_part, on="Date", how="outer").sort_values("Date").set_index("Date")
    st.line_chart(plot_df[["ActualSales", "ForecastSales"]])

st.subheader("Inventory Calculation Details")
details = pd.DataFrame([{
    "LeadTimeDays": int(lead_time),
    "ReviewDays": int(review_days),
    "ProtectionPeriodDays": int(lead_time) + int(review_days),
    "ServiceLevel": float(service),
    "z_value": decision.z_value,
    "Demand_LeadTime": decision.demand_lead_time,
    "Demand_ProtectionPeriod": decision.demand_protection_period,
    "SafetyStock(PP)": decision.safety_stock,
    "ROP": decision.reorder_point,
    "OrderUpTo": decision.order_up_to,
    "OnHand": float(on_hand),
    "OnOrder": float(on_order),
    "RecommendedOrderQty": decision.order_qty
}])
st.dataframe(details, use_container_width=True)
