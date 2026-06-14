"""
NYC Taxi Analytics Dashboard
=============================
Interactive Streamlit dashboard for the NYC Taxi pipeline.
Tabs: Overview, Time Analysis, Fare Predictor, About.
Uses Plotly for all visualizations.
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="NYC Taxi Analytics — Q1 2023",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
        .stTabs [data-baseweb="tab"] { padding: 0.5rem 1.25rem; }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def get_daily_summary():
    df = pd.read_csv("exports/mart_daily_summary.csv", parse_dates=["trip_date"])
    df["month"] = df["trip_date"].dt.month_name()
    df["dow"] = df["trip_date"].dt.day_name()
    df["is_weekend"] = df["trip_date"].dt.dayofweek.isin([5, 6])
    return df

@st.cache_data
def get_hourly_patterns():
    return pd.read_csv("exports/mart_hourly_patterns.csv")

@st.cache_data
def get_monthly_agg(daily):
    return daily.groupby("month").agg(
        total_trips=("total_trips", "sum"),
        total_revenue=("total_revenue", "sum"),
        avg_fare=("avg_fare", "mean"),
        avg_tip_pct=("avg_tip_pct", "mean"),
    ).reset_index()

@st.cache_resource
def load_model():
    with open("src/ml/artifacts/best_model.pkl", "rb") as f:
        return pickle.load(f)

daily = get_daily_summary()
hourly = get_hourly_patterns()
monthly = get_monthly_agg(daily)
artifact = load_model()
model = artifact["model"]
le = artifact["label_encoder"]

# ──────────────────────────────────────────────
# SIDEBAR — GLOBAL FILTERS
# ──────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/NYC_Taxi_%28SvG%29.jpg/640px-NYC_Taxi_%28SvG%29.jpg",
    width=280,
)

st.sidebar.header("Filters")

months = daily["month"].unique().tolist()
selected_months = st.sidebar.multiselect(
    "Month", months, default=months, key="month_filter"
)

day_types = ["Weekday", "Weekend"]
selected_day_types = st.sidebar.multiselect(
    "Day Type", day_types, default=day_types, key="day_filter"
)

filtered = daily[
    daily["month"].isin(selected_months)
    & daily["is_weekend"].isin(
        [dt == "Weekend" for dt in selected_day_types]
    )
]

st.sidebar.divider()
st.sidebar.markdown("**Pipeline Metadata**")
st.sidebar.markdown(f"**Trips:** {daily['total_trips'].sum():,.0f}")
st.sidebar.markdown(f"**Period:** {daily['trip_date'].min().date()} → {daily['trip_date'].max().date()}")
st.sidebar.markdown(f"**Model:** GradientBoosting (R²=0.969)")
st.sidebar.markdown("---")
st.sidebar.caption("Built by Brayan Hawald Ngowi · [GitHub](https://github.com/masterbry)")

# ──────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────
st.title("🚕 NYC Taxi Trip Analytics")
st.caption(
    "End-to-end data pipeline ingesting, cleaning, transforming, and modeling "
    f"**{daily['total_trips'].sum():,.0f}** yellow taxi trips from Q1 2023. "
    "Charts powered by Plotly."
)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab_overview, tab_time, tab_predictor, tab_about = st.tabs([
    "📊 Overview",
    "⏰ Time Analysis",
    "💰 Fare Predictor",
    "🔧 About",
])

# ═══════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════
with tab_overview:
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    total_trips = filtered["total_trips"].sum()
    total_revenue = filtered["total_revenue"].sum()
    avg_fare = filtered["avg_fare"].mean()
    avg_tip = filtered["avg_tip_pct"].mean()
    trip_days = filtered["trip_date"].nunique()

    prev_total = daily[~daily["trip_date"].isin(filtered["trip_date"])]["total_trips"].sum()
    trips_delta = f"{((total_trips - prev_total) / prev_total * 100):+.1f}% vs other" if prev_total > 0 else "—"

    kpi1.metric("Total Trips", f"{total_trips:,.0f}", trips_delta)
    kpi2.metric("Total Revenue", f"${total_revenue:,.0f}")
    kpi3.metric("Avg Fare", f"${avg_fare:.2f}")
    kpi4.metric("Avg Tip Rate", f"{avg_tip:.1f}%")
    kpi5.metric("Trip Days", f"{trip_days}")

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Daily Trip Volume & Revenue")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=filtered["trip_date"],
                y=filtered["total_trips"],
                name="Trips",
                marker_color="#1f77b4",
                opacity=0.7,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=filtered["trip_date"],
                y=filtered["total_revenue"],
                name="Revenue ($)",
                marker_color="#ff7f0e",
                line=dict(width=2.5),
            ),
            secondary_y=True,
        )

        window = min(7, len(filtered))
        if window > 1:
            ma = filtered["total_trips"].rolling(window=window, center=True).mean()
            fig.add_trace(
                go.Scatter(
                    x=filtered["trip_date"],
                    y=ma,
                    name=f"{window}-day MA",
                    line=dict(color="red", width=2, dash="dash"),
                ),
                secondary_y=False,
            )

        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=10, b=0),
            height=400,
        )
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="Trips", secondary_y=False)
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Monthly Summary")
        month_filtered = monthly[monthly["month"].isin(selected_months)]
        fig2 = px.bar(
            month_filtered,
            x="month",
            y="total_trips",
            color="month",
            text_auto=",.0f",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=400,
        )
        fig2.update_xaxes(title_text="")
        fig2.update_yaxes(title_text="Trips")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Data Preview")
    with st.expander("Click to expand daily summary table", expanded=False):
        st.dataframe(
            filtered.sort_values("trip_date")[
                ["trip_date", "total_trips", "avg_fare", "total_revenue", "avg_tip_pct", "morning_rush_trips", "evening_rush_trips"]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )

# ═══════════════════════════════════════════════
# TAB 2: TIME ANALYSIS
# ═══════════════════════════════════════════════
with tab_time:
    st.subheader("Hourly Demand Patterns")

    col_a, col_b = st.columns(2)

    with col_a:
        fig3 = px.bar(
            hourly,
            x="pickup_hour",
            y="total_trips",
            color="time_of_day",
            title="Trips by Hour of Day",
            labels={"pickup_hour": "Hour", "total_trips": "Trips", "time_of_day": "Time of Day"},
            color_discrete_map={
                "morning_rush": "#ff7f0e",
                "evening_rush": "#d62728",
                "off_peak": "#1f77b4",
            },
            text_auto=",.0f",
        )
        fig3.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        fig3.update_xaxes(dtick=2)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        fig4 = px.bar(
            hourly,
            x="pickup_hour",
            y="avg_fare",
            color="time_of_day",
            title="Average Fare by Hour of Day",
            labels={"pickup_hour": "Hour", "avg_fare": "Avg Fare ($)", "time_of_day": "Time of Day"},
            color_discrete_map={
                "morning_rush": "#ff7f0e",
                "evening_rush": "#d62728",
                "off_peak": "#1f77b4",
            },
            text_auto=".2f",
        )
        fig4.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        fig4.update_xaxes(dtick=2)
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Trips by Day of Week")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_data = filtered.groupby("dow").agg(
            total_trips=("total_trips", "sum"),
            avg_fare=("avg_fare", "mean"),
        ).reindex(dow_order).reset_index()

        fig5 = px.bar(
            dow_data,
            x="dow",
            y="total_trips",
            color="dow",
            color_discrete_sequence=px.colors.qualitative.Pastel1,
            text_auto=",.0f",
        )
        fig5.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=350,
        )
        fig5.update_xaxes(title_text="")
        fig5.update_yaxes(title_text="Trips")
        st.plotly_chart(fig5, use_container_width=True)

    with col_d:
        st.subheader("Avg Fare by Day of Week")
        fig6 = px.bar(
            dow_data,
            x="dow",
            y="avg_fare",
            color="dow",
            color_discrete_sequence=px.colors.qualitative.Pastel2,
            text_auto=".2f",
        )
        fig6.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=350,
        )
        fig6.update_xaxes(title_text="")
        fig6.update_yaxes(title_text="Avg Fare ($)")
        st.plotly_chart(fig6, use_container_width=True)

    st.divider()

    st.subheader("Daily Fare & Tip Trend")
    col_e, col_f = st.columns(2)

    with col_e:
        fig7 = px.line(
            filtered,
            x="trip_date",
            y="avg_fare",
            markers=True,
            labels={"trip_date": "", "avg_fare": "Avg Fare ($)"},
        )
        fig7.update_traces(line=dict(color="#2ca02c", width=2))
        fig7.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300, hovermode="x unified")
        st.plotly_chart(fig7, use_container_width=True)

    with col_f:
        fig8 = px.line(
            filtered,
            x="trip_date",
            y="avg_tip_pct",
            markers=True,
            labels={"trip_date": "", "avg_tip_pct": "Avg Tip (%)"},
        )
        fig8.update_traces(line=dict(color="#9467bd", width=2))
        fig8.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300, hovermode="x unified")
        st.plotly_chart(fig8, use_container_width=True)

    st.divider()

    st.subheader("Rush Hour Analysis")
    rush = filtered[["trip_date", "morning_rush_trips", "evening_rush_trips"]].melt(
        id_vars=["trip_date"], var_name="period", value_name="trips"
    )
    rush["period"] = rush["period"].replace({
        "morning_rush_trips": "Morning Rush (6-10)",
        "evening_rush_trips": "Evening Rush (16-20)",
    })
    fig9 = px.area(
        rush,
        x="trip_date",
        y="trips",
        color="period",
        labels={"trip_date": "", "trips": "Trips", "period": ""},
        color_discrete_map={
            "Morning Rush (6-10)": "#ff7f0e",
            "Evening Rush (16-20)": "#d62728",
        },
    )
    fig9.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, hovermode="x unified")
    st.plotly_chart(fig9, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 3: FARE PREDICTOR
# ═══════════════════════════════════════════════
with tab_predictor:
    st.subheader("Predict Taxi Fare")
    st.caption(
        "GradientBoosting regressor — MAE = **$0.94**, RMSE = **$3.26**, R² = **0.969**"
    )
    st.markdown("Set the trip parameters below and click **Predict Fare**.")

    p1, p2, p3 = st.columns(3)

    with p1:
        trip_distance = st.slider("Trip Distance (miles)", 0.1, 30.0, 2.5, 0.1)
        passenger_count = st.selectbox("Passengers", [1, 2, 3, 4, 5, 6])
        payment_type = st.selectbox(
            "Payment", [1, 2],
            format_func=lambda x: "💳 Card" if x == 1 else "💵 Cash",
        )

    with p2:
        pickup_hour = st.slider("Pickup Hour", 0, 23, 8)
        pickup_dow = st.selectbox(
            "Day of Week", list(range(7)),
            format_func=lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x],
        )
        pickup_month = st.selectbox(
            "Month", [1, 2, 3],
            format_func=lambda x: ["Jan", "Feb", "Mar"][x - 1],
        )

    with p3:
        trip_duration_min = st.slider("Trip Duration (min)", 1.0, 120.0, 15.0, 0.5)
        avg_speed_mph = st.slider("Avg Speed (mph)", 1.0, 60.0, 12.0, 0.5)
        time_of_day = st.selectbox(
            "Time of Day",
            ["morning_rush", "evening_rush", "off_peak"],
            format_func=lambda x: {
                "morning_rush": "🌅 Morning Rush (6-10)",
                "evening_rush": "🌆 Evening Rush (16-20)",
                "off_peak": "🌙 Off-Peak",
            }[x],
        )

    predict_col, info_col = st.columns([1, 2])

    with predict_col:
        predict_btn = st.button("🚕 Predict Fare", type="primary", use_container_width=True)

        if predict_btn:
            time_enc = le.transform([time_of_day])[0]
            input_data = np.array([[
                trip_distance, passenger_count, pickup_hour, pickup_dow,
                pickup_month, 161, 237, payment_type,
                trip_duration_min, avg_speed_mph, time_enc,
            ]])
            prediction = round(float(model.predict(input_data)[0]), 2)
            st.success(f"### ${prediction:.2f}")
            st.caption(f"Base components: fare + tips + surcharges")
        else:
            st.info("Set parameters and press Predict.")

    with info_col:
        st.markdown("#### Feature Summary")
        data = {
            "Feature": [
                "Trip Distance", "Passengers", "Pickup Hour", "Day of Week",
                "Month", "Pickup Location", "Dropoff Location", "Payment Type",
                "Duration (min)", "Avg Speed (mph)", "Time of Day",
            ],
            "Value": [
                f"{trip_distance} mi",
                passenger_count,
                f"{pickup_hour}:00",
                ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][pickup_dow],
                ["Jan", "Feb", "Mar"][pickup_month - 1],
                "161 (Manhattan)",
                "237 (Manhattan)",
                "Card" if payment_type == 1 else "Cash",
                f"{trip_duration_min} min",
                f"{avg_speed_mph} mph",
                time_of_day.replace("_", " ").title(),
            ],
        }
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "Model trained on 400K trips with 11 features. "
        "Best model: GradientBoostingRegressor (n_estimators=100, max_depth=5). "
        "Pickup/dropoff locations default to Manhattan zone IDs 161 and 237."
    )

# ═══════════════════════════════════════════════
# TAB 4: ABOUT
# ═══════════════════════════════════════════════
with tab_about:
    st.subheader("About This Project")

    col_about_left, col_about_right = st.columns([3, 2])

    with col_about_left:
        st.markdown("""
        This dashboard is the frontend of an **end-to-end data engineering and machine learning pipeline**
        that processes **8.7 million NYC yellow taxi trips** from Q1 2023 (January–March).

        **Pipeline Steps:**

        1. **Ingestion** — Downloads raw Parquet files from the NYC TLC public S3 bucket
        2. **Quality Check** — Validates data completeness, validity, and consistency
        3. **Cleaning** — Removes anomalous records (negative fares, zero distances, etc.)
           and engineers features (time components, speed, tip percentage)
        4. **Loading** — Loads cleaned data into a persistent DuckDB database
        5. **Transformation** — dbt models stage raw data and produce aggregated marts
        6. **ML Training** — Trains and compares 3 regression models tracked with MLflow
        7. **Serving** — FastAPI endpoint serves the best model for real-time predictions
        8. **Dashboard** — This interactive Streamlit application
        """)

    with col_about_right:
        st.markdown("#### Tech Stack")
        techs = [
            ("🐍 Python 3.13", "Core language"),
            ("🦆 DuckDB", "Embedded OLAP database"),
            ("🏗️ dbt", "Data transformation"),
            ("🤖 scikit-learn", "ML models"),
            ("📊 MLflow", "Experiment tracking"),
            ("⚡ FastAPI", "Prediction API"),
            ("📈 Streamlit", "Dashboard"),
            ("📉 Plotly", "Interactive charts"),
        ]
        for name, desc in techs:
            st.markdown(f"**{name}** — {desc}")

        st.divider()

        st.markdown("#### Model Performance")
        perf = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "R²"],
            "Value": ["$0.94", "$3.26", "0.969"],
        })
        st.dataframe(perf, hide_index=True, use_container_width=True)

    st.divider()

    st.subheader("Pipeline Architecture")
    st.code("""
Raw Parquet (NYC TLC)
    │
    ├── download_data.py   ──►  data/raw/*.parquet
    ├── quality_check.py   ──►  Data quality report
    ├── clean_data.py      ──►  data/processed/trips_cleaned.parquet
    ├── load_to_duckdb.py  ──►  data/taxi.duckdb (raw_trips)
    ├── dbt transforms     ──►  stg_trips → mart_daily_summary, mart_hourly_patterns
    ├── train_model.py     ──►  MLflow tracking → best_model.pkl
    ├── api.py             ──►  FastAPI /predict endpoint
    └── dashboard.py       ──►  Streamlit (this page)
    """, language="text")

    st.markdown("---")
    st.markdown(
        "Built by **Brayan Hawald Ngowi** · "
        "[GitHub](https://github.com/masterbry) · "
        "[Website](https://master-bry.vercel.app)"
    )
