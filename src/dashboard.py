import streamlit as st
import duckdb
import pandas as pd
import pickle
import numpy as np

st.set_page_config(
    page_title="NYC Taxi Analytics",
    page_icon="🚕",
    layout="wide"
)

# Load data
@st.cache_resource
def load_db():
    return duckdb.connect("data/taxi.duckdb", read_only=True)

@st.cache_resource
def load_model():
    with open("src/ml/artifacts/best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def get_daily_summary():
    con = load_db()
    return con.execute("""
        SELECT * FROM mart_daily_summary ORDER BY trip_date
    """).df()

@st.cache_data
def get_hourly_patterns():
    con = load_db()
    return con.execute("""
        SELECT * FROM mart_hourly_patterns ORDER BY pickup_hour
    """).df()

con = load_db()
artifact = load_model()
model = artifact["model"]
le = artifact["label_encoder"]

# Header
st.title("NYC Taxi Analytics Pipeline")
st.caption("End-to-end data engineering project — 8.7M trips, Q1 2023")

# Top metrics
daily = get_daily_summary()
hourly = get_hourly_patterns()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trips", f"{daily['total_trips'].sum():,.0f}")
col2.metric("Total Revenue", f"${daily['total_revenue'].sum():,.0f}")
col3.metric("Avg Fare", f"${daily['avg_fare'].mean():.2f}")
col4.metric("Avg Tip", f"{daily['avg_tip_pct'].mean():.1f}%")

st.divider()

# Charts
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Daily Trips")
    st.line_chart(daily.set_index("trip_date")["total_trips"])

with col_right:
    st.subheader("Daily Revenue ($)")
    st.line_chart(daily.set_index("trip_date")["total_revenue"])

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("Trips by Hour of Day")
    st.bar_chart(hourly.set_index("pickup_hour")["total_trips"])

with col_right2:
    st.subheader("Avg Fare by Hour ($)")
    st.bar_chart(hourly.set_index("pickup_hour")["avg_fare"])

st.divider()

# Fare Predictor
st.subheader("Fare Predictor")
st.caption("GradientBoosting model — MAE $0.94, R2 = 0.969")

p1, p2, p3 = st.columns(3)

with p1:
    trip_distance = st.slider("Trip Distance (miles)", 0.1, 30.0, 2.5)
    passenger_count = st.selectbox("Passengers", [1, 2, 3, 4, 5, 6])
    payment_type = st.selectbox("Payment", [1, 2], format_func=lambda x: "Card" if x == 1 else "Cash")

with p2:
    pickup_hour = st.slider("Pickup Hour", 0, 23, 8)
    pickup_dow = st.selectbox("Day of Week", list(range(7)),
                               format_func=lambda x: ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][x])
    pickup_month = st.selectbox("Month", [1, 2, 3], format_func=lambda x: ["Jan","Feb","Mar"][x-1])

with p3:
    trip_duration_min = st.slider("Trip Duration (min)", 1.0, 120.0, 15.0)
    avg_speed_mph = st.slider("Avg Speed (mph)", 1.0, 60.0, 12.0)
    time_of_day = st.selectbox("Time of Day", ["morning_rush", "evening_rush", "off_peak"])

if st.button("Predict Fare", type="primary"):
    time_enc = le.transform([time_of_day])[0]
    input_data = np.array([[
        trip_distance, passenger_count, pickup_hour, pickup_dow,
        pickup_month, 161, 237, payment_type,
        trip_duration_min, avg_speed_mph, time_enc
    ]])
    prediction = round(float(model.predict(input_data)[0]), 2)
    st.success(f"Predicted Fare: **${prediction}**")

st.divider()

# Pipeline summary
st.subheader("Pipeline Architecture")
st.code("""
Raw Parquet (NYC TLC)
    -> Python ingestion (download_data.py)
    -> DuckDB cleaning + feature engineering (clean_data.py)
    -> dbt transforms: stg_trips, mart_daily_summary, mart_hourly_patterns
    -> GradientBoosting ML model (MLflow tracked)
    -> FastAPI prediction endpoint (/predict)
    -> Streamlit dashboard (this page)
""", language="text")

st.caption("Built by Brayan Hawald Ngowi | github.com/masterbry | master-bry.vercel.app")