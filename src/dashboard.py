"""
NYC Taxi Analytics Dashboard — Professional Portfolio Edition
==============================================================
Interactive Streamlit dashboard for the NYC Taxi pipeline.
Features Plotly with dark theme, glassmorphism UI, animated charts,
3D visualizations, heatmaps, Sankey diagrams, and more.
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import duckdb
import os

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — Dark Glassmorphism Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #0f0f1a 100%);
    }

    .block-container {
        padding-top: 1.5rem !important;
        max-width: 100% !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #e2e8f0 !important;
        letter-spacing: -0.01em;
    }

    h1 { font-weight: 700; letter-spacing: -0.02em; }
    h2 { font-weight: 600; letter-spacing: -0.015em; }
    h3 { font-weight: 600; letter-spacing: -0.01em; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 500;
        font-size: 0.9rem;
        color: #94a3b8 !important;
        transition: all 0.3s ease;
        letter-spacing: 0.01em;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2)) !important;
        color: #fff !important;
        box-shadow: 0 0 20px rgba(0,212,255,0.15);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem 1rem;
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(0,212,255,0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #fff !important;
        line-height: 1.3;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    div[data-testid="column"] {
        min-width: 0;
        overflow: hidden;
    }

    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #fff !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 24px rgba(0,212,255,0.25) !important;
        letter-spacing: 0.01em;
    }

    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 40px rgba(0,212,255,0.4) !important;
    }

    .stSelectbox, .stSlider {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid rgba(255,255,255,0.06);
    }

    div[data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
        line-height: 1.7;
    }

    .stCaption {
        color: #64748b !important;
    }

    hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 1.5rem 0 !important;
    }

    .stDataFrame {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .insight-box {
        background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,58,237,0.08));
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
    }

    .insight-box p { margin: 0; color: #e2e8f0 !important; font-size: 0.9rem; }

    .stSidebar {
        background: rgba(15,15,26,0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }

    .stSidebar .sidebar-content {
        background: transparent !important;
    }

    .st-emotion-cache-1wrcu25 { display: none; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    /* Plotly chart container */
    .js-plotly-plot { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PLOTLY DARK THEME DEFAULTS
# ──────────────────────────────────────────────

THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#cbd5e1", "family": "Inter, sans-serif"},
    "xaxis": {
        "gridcolor": "rgba(255,255,255,0.04)",
        "zerolinecolor": "rgba(255,255,255,0.06)",
        "title_font": {"size": 12, "color": "#94a3b8"},
    },
    "yaxis": {
        "gridcolor": "rgba(255,255,255,0.04)",
        "zerolinecolor": "rgba(255,255,255,0.06)",
        "title_font": {"size": 12, "color": "#94a3b8"},
    },
    "legend": {
        "font": {"color": "#94a3b8"},
        "bgcolor": "rgba(0,0,0,0)",
    },
    "hoverlabel": {
        "bgcolor": "#1e293b",
        "font_color": "#f1f5f9",
        "bordercolor": "rgba(255,255,255,0.1)",
    },
    "margin": {"l": 0, "r": 0, "t": 65, "b": 10},
    "title": {"automargin": True, "font": {"size": 13}, "y": 0.98},
}

def apply_theme(fig):
    fig.update_layout(**THEME)
    return fig

COLORS = {
    "cyan": "#00d4ff",
    "purple": "#7c3aed",
    "amber": "#f59e0b",
    "pink": "#ec4899",
    "green": "#10b981",
    "red": "#ef4444",
    "blue": "#3b82f6",
    "teal": "#14b8a6",
}

# ──────────────────────────────────────────────
# DATA SOURCE DETECTION
# ──────────────────────────────────────────────

@st.cache_data
def check_data_available():
    db_ok = os.path.exists("data/taxi.duckdb")
    parquet_ok = os.path.exists("data/processed/trips_cleaned.parquet")
    csv_daily_ok = os.path.exists("exports/mart_daily_summary.csv")
    csv_hourly_ok = os.path.exists("exports/mart_hourly_patterns.csv")
    model_ok = os.path.exists("src/ml/artifacts/best_model.pkl")
    return {
        "duckdb": db_ok,
        "parquet": parquet_ok,
        "csv_daily": csv_daily_ok,
        "csv_hourly": csv_hourly_ok,
        "model": model_ok,
        "has_raw": db_ok or parquet_ok,
    }

data_src = check_data_available()

# ──────────────────────────────────────────────
# DUCKDB CONNECTION
# ──────────────────────────────────────────────

@st.cache_resource
def get_duckdb():
    if data_src["duckdb"]:
        return duckdb.connect("data/taxi.duckdb", read_only=True)
    if data_src["parquet"]:
        return duckdb.connect()
    return None

def query(q):
    con = get_duckdb()
    if con is None:
        return None
    try:
        return con.execute(q).df()
    except Exception:
        return None

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def get_daily_summary():
    if not data_src["csv_daily"]:
        return pd.DataFrame()
    df = pd.read_csv("exports/mart_daily_summary.csv", parse_dates=["trip_date"])
    df["month"] = df["trip_date"].dt.month_name()
    df["dow_name"] = df["trip_date"].dt.day_name()
    df["dow_num"] = df["trip_date"].dt.dayofweek
    df["is_weekend"] = df["dow_num"].isin([5, 6])
    return df

@st.cache_data
def get_hourly_patterns():
    if not data_src["csv_hourly"]:
        return pd.DataFrame()
    return pd.read_csv("exports/mart_hourly_patterns.csv")

@st.cache_data
def get_heatmap_data():
    return query("""
        SELECT pickup_dow, pickup_hour, COUNT(*) AS trips
        FROM read_parquet('data/processed/trips_cleaned.parquet')
        GROUP BY pickup_dow, pickup_hour
        ORDER BY pickup_dow, pickup_hour
    """)

@st.cache_data
def get_scatter_sample():
    return query("""
        SELECT fare_amount, trip_distance, tip_amount, time_of_day,
               pickup_hour, pickup_dow, pickup_month
        FROM read_parquet('data/processed/trips_cleaned.parquet')
        WHERE fare_amount BETWEEN 3 AND 100
          AND trip_distance BETWEEN 0.1 AND 40
        LIMIT 80000
    """)

@st.cache_data
def get_location_data():
    return query("""
        SELECT PULocationID, DOLocationID, COUNT(*) AS trips,
               ROUND(AVG(fare_amount), 2) AS avg_fare,
               ROUND(AVG(trip_distance), 2) AS avg_distance
        FROM read_parquet('data/processed/trips_cleaned.parquet')
        GROUP BY PULocationID, DOLocationID
        ORDER BY trips DESC
    """)

@st.cache_data
def get_payment_dist():
    return query("""
        SELECT payment_type,
               CASE payment_type
                   WHEN 1 THEN 'Credit Card'
                   WHEN 2 THEN 'Cash'
                   WHEN 3 THEN 'No Charge'
                   WHEN 4 THEN 'Dispute'
                   WHEN 5 THEN 'Unknown'
                   WHEN 6 THEN 'Voided Trip'
                   ELSE 'Other'
               END AS payment_desc,
               COUNT(*) AS trips,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct,
               ROUND(AVG(fare_amount), 2) AS avg_fare,
               ROUND(AVG(tip_amount), 2) AS avg_tip
        FROM read_parquet('data/processed/trips_cleaned.parquet')
        WHERE payment_type IN (1, 2)
        GROUP BY payment_type
        ORDER BY payment_type
    """)

@st.cache_data
def get_metrics_summary():
    df = query("""
        SELECT
            COUNT(*) AS total_trips,
            ROUND(SUM(total_amount), 2) AS total_revenue,
            ROUND(AVG(fare_amount), 2) AS avg_fare,
            ROUND(AVG(tip_amount), 2) AS avg_tip,
            ROUND(AVG(trip_distance), 2) AS avg_distance,
            ROUND(AVG(trip_duration_min), 2) AS avg_duration,
            ROUND(AVG(avg_speed_mph), 2) AS avg_speed,
            ROUND(AVG(tip_pct), 2) AS avg_tip_pct,
            ROUND(SUM(congestion_surcharge), 2) AS total_congestion,
            ROUND(SUM(tolls_amount), 2) AS total_tolls
        FROM read_parquet('data/processed/trips_cleaned.parquet')
    """)
    if df is not None and len(df) > 0:
        return df.iloc[0].to_dict()
    return None

@st.cache_data
def get_rush_hour_breakdown():
    if data_src["has_raw"]:
        return query("""
            SELECT
                time_of_day,
                COUNT(*) AS trips,
                ROUND(AVG(fare_amount), 2) AS avg_fare,
                ROUND(AVG(trip_distance), 2) AS avg_distance,
                ROUND(AVG(tip_amount), 2) AS avg_tip
            FROM read_parquet('data/processed/trips_cleaned.parquet')
            GROUP BY time_of_day
            ORDER BY trips DESC
        """)
    if data_src["csv_hourly"]:
        df = pd.read_csv("exports/mart_hourly_patterns.csv")
        return df.groupby("time_of_day").agg(
            trips=("total_trips", "sum"),
            avg_fare=("avg_fare", "mean"),
            avg_tip=("avg_tip_pct", "mean"),
        ).reset_index().sort_values("trips", ascending=False)
    return None

@st.cache_data
def get_dow_breakdown():
    if data_src["has_raw"]:
        df = query("""
            SELECT
                pickup_dow,
                COUNT(*) AS trips,
                ROUND(AVG(fare_amount), 2) AS avg_fare,
                ROUND(AVG(tip_amount), 2) AS avg_tip,
                ROUND(AVG(trip_distance), 2) AS avg_distance,
                ROUND(AVG(trip_duration_min), 2) AS avg_duration
            FROM read_parquet('data/processed/trips_cleaned.parquet')
            GROUP BY pickup_dow
            ORDER BY pickup_dow
        """)
        if df is not None:
            df["day"] = df["pickup_dow"].map({
                0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"
            })
        return df
    if data_src["csv_daily"]:
        df = pd.read_csv("exports/mart_daily_summary.csv", parse_dates=["trip_date"])
        df["dow_num"] = df["trip_date"].dt.dayofweek
        df["day"] = df["trip_date"].dt.day_name().str[:3]
        dow = df.groupby(["dow_num", "day"]).agg(
            trips=("total_trips", "sum"),
            avg_fare=("avg_fare", "mean"),
            avg_tip=("avg_tip_pct", "mean"),
            avg_distance=("avg_distance", "mean"),
        ).reset_index().sort_values("dow_num")
        dow["avg_duration"] = dow["avg_distance"] / 15 * 60
        return dow
    return None

@st.cache_data
def get_monthly_breakdown():
    df = query("""
        SELECT
            pickup_month,
            ROUND(AVG(fare_amount), 2) AS avg_fare,
            ROUND(AVG(trip_distance), 2) AS avg_distance,
            ROUND(AVG(tip_amount), 2) AS avg_tip,
            ROUND(AVG(trip_duration_min), 2) AS avg_duration
        FROM read_parquet('data/processed/trips_cleaned.parquet')
        GROUP BY pickup_month
        ORDER BY pickup_month
    """)
    if df is not None:
        df["month_name"] = df["pickup_month"].map({1: "Jan", 2: "Feb", 3: "Mar"})
    return df

@st.cache_resource
def load_model():
    if not data_src["model"]:
        return None
    with open("src/ml/artifacts/best_model.pkl", "rb") as f:
        return pickle.load(f)

# ──────────────────────────────────────────────
# LOAD ALL DATA
# ──────────────────────────────────────────────

with st.spinner("Loading pipeline data..."):
    daily = get_daily_summary()
    hourly = get_hourly_patterns()
    heatmap_data = get_heatmap_data()
    scatter_sample = get_scatter_sample()
    location_data = get_location_data()
    payment_dist = get_payment_dist()
    metrics = get_metrics_summary()
    rush_hour = get_rush_hour_breakdown()
    dow_data = get_dow_breakdown()
    monthly_data = get_monthly_breakdown()
    artifact = load_model()
    model = artifact["model"] if artifact else None
    le = artifact["label_encoder"] if artifact else None

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding:1rem 0;">
            <h3 style="margin:0; font-weight:700; font-size:1.5rem; background: linear-gradient(135deg, #00d4ff, #7c3aed);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                NYC TAXI
            </h3>
            <p style="color:#64748b; font-size:0.85rem; margin:0.25rem 0 0 0; letter-spacing:0.02em;">Analytics Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    months = daily["month"].unique().tolist() if not daily.empty else []
    selected_months = st.multiselect("Month", months, default=months)

    day_types = ["Weekday", "Weekend"]
    selected_day_types = st.multiselect("Day Type", day_types, default=day_types)

    filtered = daily[
        daily["month"].isin(selected_months)
        & daily["is_weekend"].isin([dt == "Weekend" for dt in selected_day_types])
    ] if not daily.empty else daily

    st.divider()

    st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border-radius:12px; padding:1rem;
                    border:1px solid rgba(255,255,255,0.06);">
            <p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;
                      margin:0 0 0.75rem 0;">Pipeline Metadata</p>
    """, unsafe_allow_html=True)

    if metrics:
        st.markdown(f"**Total Trips:** {metrics['total_trips']:,.0f}")
        st.markdown(f"**Avg Fare:** ${metrics['avg_fare']:.2f}")
    else:
        st.markdown(f"**Total Trips:** {daily['total_trips'].sum():,.0f}" if not daily.empty else "**Total Trips:** N/A")
    st.markdown(f"**Period:** Jan 1 – Mar 31, 2023")
    st.markdown(f"**Model:** GradientBoosting · R² = 0.969")
    st.markdown(f"**MAE:** $0.94")

    st.markdown("""
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
        <p style="color:#475569; font-size:0.8rem; text-align:center;">
            Built by Brayan Hawald Ngowi<br>
            <a href="https://github.com/masterbry" style="color:#00d4ff; text-decoration:none;">GitHub</a>
            ·
            <a href="https://master-bry.vercel.app" style="color:#00d4ff; text-decoration:none;">Portfolio</a>
        </p>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TITLE SECTION
# ──────────────────────────────────────────────

st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:2.5rem; font-weight:800; margin:0;
                   background: linear-gradient(135deg, #f1f5f9, #94a3b8);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            NYC Yellow Taxi Analytics
        </h1>
        <p style="color:#64748b; font-size:1rem; margin:0.25rem 0 0 0;">
            Q1 2023 · 8.8 million trips · End-to-end data pipeline
        </p>
    </div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# KPI ROW — Styled Metric Cards
# ──────────────────────────────────────────────
total_trips_f = filtered["total_trips"].sum() if not filtered.empty else 0
total_revenue_f = filtered["total_revenue"].sum() if not filtered.empty else 0
avg_fare_f = filtered["avg_fare"].mean() if not filtered.empty else 0
avg_tip_f = filtered["avg_tip_pct"].mean() if not filtered.empty else 0


total_metric = f"{metrics['total_trips']:,.0f} total" if metrics else f"{total_trips_f:,.0f}"
revenue_metric = f"${metrics['total_revenue']:,.0f} total" if metrics else f"${total_revenue_f:,.0f}"
fare_metric = f"${metrics['avg_fare']:.2f} overall" if metrics else f"${avg_fare_f:.2f}"
tip_metric = f"{metrics['avg_tip_pct']:.1f}% overall" if metrics else f"{avg_tip_f:.1f}%"

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Total Trips", f"{total_trips_f:,.0f}", total_metric)

with k2:
    st.metric("Revenue", f"${total_revenue_f:,.0f}", revenue_metric)

with k3:
    st.metric("Avg Fare", f"${avg_fare_f:.2f}", fare_metric)

with k4:
    st.metric("Tip Rate", f"{avg_tip_f:.1f}%", tip_metric)

st.divider()

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────

t1, t2, t3, t4, t5 = st.tabs([
    " Overview",
    " Time & Patterns",
    " Locations",
    " Fare Predictor",
    " About",
])

# ═══════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════
 
with t1:
    row1 = st.columns([3, 2])

    with row1[0]:
        st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Daily Trip Volume</h3>", unsafe_allow_html=True)

        window = min(7, len(filtered))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=filtered["trip_date"], y=filtered["total_trips"],
                mode="lines", name="Daily Trips",
                line=dict(color=COLORS["cyan"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.08)",
                hovertemplate="%{x|%b %d}<br>Trips: %{y:,.0f}<extra></extra>",
            ),
            secondary_y=False,
        )
        if len(filtered) > 1:
            ma = filtered["total_trips"].rolling(window=window, center=True).mean()
            fig.add_trace(
                go.Scatter(
                    x=filtered["trip_date"], y=ma,
                    mode="lines", name=f"{window}-Day MA",
                    line=dict(color=COLORS["amber"], width=2, dash="dot"),
                    hovertemplate="%{x|%b %d}<br>MA: %{y:,.0f}<extra></extra>",
                ),
                secondary_y=False,
                
            )
        fig.add_trace(
            go.Scatter(
                x=filtered["trip_date"], y=filtered["total_revenue"],
                mode="lines+markers", name="Revenue ($)",
                line=dict(color=COLORS["purple"], width=2),
                marker=dict(size=4, color=COLORS["purple"]),
                yaxis="y2",
                hovertemplate="%{x|%b %d}<br>Revenue: $%{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=0),
            height=380,
        )
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Trips", secondary_y=False)
        fig.update_yaxes(title="Revenue ($)", secondary_y=True)
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with row1[1]:
        st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Revenue by Month</h3>", unsafe_allow_html=True)
        month_rev = filtered.groupby("month").agg(
            total_revenue=("total_revenue", "sum"),
            total_trips=("total_trips", "sum"),
        ).reset_index()

        fig = px.pie(
            month_rev, values="total_revenue", names="month",
            color="month",
            color_discrete_map={
                "January": COLORS["cyan"],
                "February": COLORS["purple"],
                "March": COLORS["amber"],
            },
            hole=0.55,
        )
        fig.update_traces(
            textposition="outside",
            textinfo="label+percent",
            hovertemplate="%{label}<br>Revenue: $%{value:,.0f}<br>%{percent}<extra></extra>",
        )
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    st.divider()

    row2 = st.columns(4)

    rush_colors = {"morning_rush": COLORS["amber"], "evening_rush": COLORS["purple"], "off_peak": COLORS["cyan"]}
    rush_labels = {"morning_rush": "Morning Rush", "evening_rush": "Evening Rush", "off_peak": "Off-Peak"}

    has_rush = rush_hour is not None and not rush_hour.empty
    has_payment = payment_dist is not None and not payment_dist.empty

    if has_rush:
        rush_hour["label"] = rush_hour["time_of_day"].map(rush_labels)
        max_trips = rush_hour["trips"].max()

    with row2[0]:
        if has_rush:
            fig = px.bar(
                rush_hour, x="label", y="trips", color="time_of_day",
                color_discrete_map=rush_colors,
                text_auto=",.0f",
                title="Trips by Time of Day",
            )
            fig.update_traces(showlegend=False)
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Weekday", "Weekend"], y=[
                filtered[~filtered["is_weekend"]]["total_trips"].sum() if not filtered.empty else 0,
                filtered[filtered["is_weekend"]]["total_trips"].sum() if not filtered.empty else 0
            ], marker_color=[COLORS["cyan"], COLORS["purple"]], text=[f"{filtered[~filtered['is_weekend']]['total_trips'].sum():,.0f}" if not filtered.empty else "0", f"{filtered[filtered['is_weekend']]['total_trips'].sum():,.0f}" if not filtered.empty else "0"], textposition="outside"))
            fig.update_layout(title="Weekday vs Weekend")
        fig.update_layout(height=300, xaxis_title="", yaxis_title="")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with row2[1]:
        fig = px.bar(
            hourly, x="pickup_hour", y="avg_fare",
            color="time_of_day", color_discrete_map=rush_colors,
            title="Fare by Hour",
            labels={"pickup_hour": "Hour", "avg_fare": "Fare ($)", "time_of_day": ""},
            text_auto=".1f",
        ) if not hourly.empty else go.Figure()
        if not hourly.empty:
            fig.update_traces(showlegend=False)
            fig.update_xaxes(dtick=4)
        fig.update_layout(height=300, xaxis_title="", yaxis_title="")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with row2[2]:
        if has_rush:
            fig = px.bar(
                rush_hour, x="label", y="avg_tip", color="time_of_day",
                color_discrete_map=rush_colors,
                text_auto=".1f",
                title="Avg Tip %",
            )
            fig.update_traces(showlegend=False)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered["trip_date"] if not filtered.empty else [], y=filtered["avg_tip_pct"] if not filtered.empty else [], mode="lines+markers", name="Tip %", line=dict(color=COLORS["pink"], width=2)))
            fig.update_layout(title="Tip Rate Trend")
        fig.update_layout(height=300, xaxis_title="", yaxis_title="Avg Tip (%)")
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    with row2[3]:
        if has_payment:
            fig = px.pie(
                payment_dist, values="trips", names="payment_desc",
                title="Payment Type Split",
                color="payment_desc",
                color_discrete_map={
                    "Credit Card": COLORS["cyan"],
                    "Cash": COLORS["amber"],
                },
                hole=0.5,
            )
            fig.update_traces(textposition="outside", textinfo="percent")
        elif not filtered.empty:
            ww = filtered.groupby("is_weekend").agg(trips=("total_trips", "sum")).reset_index()
            ww["label"] = ww["is_weekend"].map({False: "Weekday", True: "Weekend"})
            fig = px.pie(
                ww, values="trips", names="label",
                title="Weekday vs Weekend",
                color="label",
                color_discrete_map={"Weekday": COLORS["cyan"], "Weekend": COLORS["purple"]},
                hole=0.5,
            )
            fig.update_traces(textposition="outside", textinfo="percent")
        else:
            fig = go.Figure()
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(apply_theme(fig), use_container_width=True)

    st.divider()

    if has_rush:
        evening_trips = rush_hour[rush_hour["time_of_day"] == "evening_rush"]["trips"].values[0]
        morning_trips = rush_hour[rush_hour["time_of_day"] == "morning_rush"]["trips"].values[0]
        rush_ratio = f"{evening_trips / morning_trips:.1f}x" if morning_trips > 0 else "N/A"
    else:
        evening_trips = filtered["evening_rush_trips"].sum() if not filtered.empty else 0
        morning_trips = filtered["morning_rush_trips"].sum() if not filtered.empty else 0
        rush_ratio = f"{evening_trips / morning_trips:.1f}x" if morning_trips > 0 else "N/A"

    insight_trips = f"{metrics['total_trips']:,.0f}" if metrics else f"{daily['total_trips'].sum():,.0f}"
    insight_fare = f"${metrics['avg_fare']}" if metrics else f"${avg_fare_f:.2f}"
    insight_card_pct = f"{payment_dist[payment_dist['payment_type'] == 1]['pct'].values[0]:.1f}" if has_payment else "majority"

    st.markdown("""
        <div class="insight-box">
            <p><strong style="color:#00d4ff;">Key Insight:</strong>
            The pipeline processes <strong>{}</strong> trips with an average fare of
            <strong>{}</strong>. Evening rush hour sees <strong>{}</strong> more
            trips than morning rush. Credit card payments dominate at
            <strong>{}%</strong>.</p>
        </div>
    """.format(insight_trips, insight_fare, rush_ratio, insight_card_pct), unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 2: TIME & PATTERNS
# ═══════════════════════════════════════════════

with t2:
    has_heatmap = heatmap_data is not None and not heatmap_data.empty
    has_scatter = scatter_sample is not None and not scatter_sample.empty
    has_dow = dow_data is not None and not dow_data.empty
    has_monthly = monthly_data is not None and not monthly_data.empty
    has_raw = data_src["has_raw"]

    if has_heatmap:
        st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Hour × Day of Week Heatmap</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; margin-top:-0.25rem;'>Trip density: darker = higher volume</p>", unsafe_allow_html=True)

        dow_labels = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
        pivot = heatmap_data.pivot_table(
            values="trips", index="pickup_hour", columns="pickup_dow", aggfunc="sum"
        )
        pivot.columns = [dow_labels[c] for c in pivot.columns]
        pivot = pivot[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0, "#0a0a1a"],
                [0.25, "#1a1a3e"],
                [0.5, "#00d4ff"],
                [0.75, "#7c3aed"],
                [1, "#f59e0b"],
            ],
            hovertemplate="Day: %{x}<br>Hour: %{y}:00<br>Trips: %{z:,.0f}<extra></extra>",
        ))
        fig.update_layout(height=450, xaxis_title="", yaxis_title="Hour of Day")
        fig.update_yaxes(dtick=2, autorange="reversed")
        st.plotly_chart(apply_theme(fig), use_container_width=True)
    else:
        st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Hourly Trip Distribution</h3>", unsafe_allow_html=True)
        if not hourly.empty:
            fig = px.bar(
                hourly, x="pickup_hour", y="total_trips", color="time_of_day",
                color_discrete_map=rush_colors,
                title="Trips by Hour of Day",
                labels={"pickup_hour": "Hour", "total_trips": "Trips"},
                text_auto=",.0f",
            )
            fig.update_layout(height=450, showlegend=False)
            fig.update_xaxes(dtick=2)
            st.plotly_chart(apply_theme(fig), use_container_width=True)

    st.divider()

    row_t1 = st.columns(2)

    with row_t1[0]:
        if has_scatter:
            st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>3D Trip Analysis</h3>", unsafe_allow_html=True)
            fig = px.scatter_3d(
                scatter_sample.sample(min(30000, len(scatter_sample)), random_state=42),
                x="trip_distance", y="fare_amount", z="tip_amount",
                color="time_of_day",
                color_discrete_map=rush_colors,
                opacity=0.4,
                size_max=2,
                labels={
                    "trip_distance": "Distance (mi)",
                    "fare_amount": "Fare ($)",
                    "tip_amount": "Tip ($)",
                    "time_of_day": "Time of Day",
                },
            )
            fig.update_traces(marker=dict(size=1.5))
            fig.update_layout(
                height=450,
                scene=dict(
                    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", backgroundcolor="rgba(0,0,0,0)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", backgroundcolor="rgba(0,0,0,0)"),
                    zaxis=dict(gridcolor="rgba(255,255,255,0.06)", backgroundcolor="rgba(0,0,0,0)"),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                ),
                legend=dict(orientation="h", y=1.02, x=0.3),
            )
            st.plotly_chart(apply_theme(fig), use_container_width=True)
        else:
            st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Daily Trip & Revenue Trend</h3>", unsafe_allow_html=True)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            if not filtered.empty:
                fig.add_trace(go.Scatter(x=filtered["trip_date"], y=filtered["total_trips"], mode="lines", name="Trips", line=dict(color=COLORS["cyan"], width=2.5), fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"), secondary_y=False)
                fig.add_trace(go.Scatter(x=filtered["trip_date"], y=filtered["total_revenue"], mode="lines", name="Revenue", line=dict(color=COLORS["purple"], width=2)), secondary_y=True)
            fig.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0.25))
            fig.update_yaxes(title="Trips", secondary_y=False)
            fig.update_yaxes(title="Revenue ($)", secondary_y=True)
            st.plotly_chart(apply_theme(fig), use_container_width=True)

    with row_t1[1]:
        if has_dow and len(dow_data) >= 5:
            st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Day of Week — Multi-Metric Radar</h3>", unsafe_allow_html=True)
            dow_norm = dow_data.copy()
            for col in ["trips", "avg_fare", "avg_tip", "avg_distance", "avg_duration"]:
                mx = dow_norm[col].max()
                dow_norm[col + "_norm"] = dow_norm[col] / mx if mx > 0 else 0

            fig = go.Figure()
            radar_metrics = [
                ("trips_norm", "Trips", COLORS["cyan"]),
                ("avg_fare_norm", "Fare", COLORS["amber"]),
                ("avg_tip_norm", "Tip", COLORS["green"]),
                ("avg_distance_norm", "Distance", COLORS["purple"]),
                ("avg_duration_norm", "Duration", COLORS["pink"]),
            ]
            for col, name, color in radar_metrics:
                fig.add_trace(go.Scatterpolar(
                    r=dow_norm[col].tolist() + [dow_norm[col].iloc[0]],
                    theta=dow_norm["day"].tolist() + [dow_norm["day"].iloc[0]],
                    name=name,
                    line=dict(color=color, width=2),
                    fill="toself",
                    opacity=0.3,
                ))
            fig.update_layout(
                height=450,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.06)", showticklabels=False),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="#94a3b8"),
                ),
                legend=dict(orientation="h", y=1.08, x=0.25),
            )
            st.plotly_chart(apply_theme(fig), use_container_width=True)
        else:
            st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Revenue & Tip Trend</h3>", unsafe_allow_html=True)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            if not filtered.empty:
                fig.add_trace(go.Scatter(x=filtered["trip_date"], y=filtered["total_revenue"], mode="lines", name="Revenue", line=dict(color=COLORS["green"], width=2.5), fill="tozeroy", fillcolor="rgba(16,185,129,0.08)"), secondary_y=False)
                fig.add_trace(go.Scatter(x=filtered["trip_date"], y=filtered["avg_tip_pct"], mode="lines", name="Tip %", line=dict(color=COLORS["pink"], width=2)), secondary_y=True)
            fig.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0.25))
            fig.update_yaxes(title="Revenue ($)", secondary_y=False)
            fig.update_yaxes(title="Tip (%)", secondary_y=True)
            st.plotly_chart(apply_theme(fig), use_container_width=True)

    st.divider()

    row_t2 = st.columns(2)

    with row_t2[0]:
        if has_raw:
            st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Hourly Trip Profile by Month</h3>", unsafe_allow_html=True)
            monthly_hourly = query("""
                SELECT pickup_month, pickup_hour, COUNT(*) AS trips
                FROM read_parquet('data/processed/trips_cleaned.parquet')
                GROUP BY pickup_month, pickup_hour
                ORDER BY pickup_month, pickup_hour
            """)
            if monthly_hourly is not None and not monthly_hourly.empty:
                fig = px.line(
                    monthly_hourly, x="pickup_hour", y="trips", color="pickup_month",
                    color_discrete_map={1: COLORS["cyan"], 2: COLORS["purple"], 3: COLORS["amber"]},
                    labels={"pickup_hour": "Hour", "trips": "Trips", "pickup_month": "Month"},
                    line_shape="spline",
                )
                fig.update_traces(line=dict(width=3))
                fig.update_layout(height=350, hovermode="x unified")
                fig.update_xaxes(dtick=2)
                st.plotly_chart(apply_theme(fig), use_container_width=True)
        else:
            st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Hourly Trip Profile</h3>", unsafe_allow_html=True)
            if not hourly.empty:
                fig = px.bar(
                    hourly, x="pickup_hour", y="total_trips",
                    color="time_of_day", color_discrete_map=rush_colors,
                    labels={"pickup_hour": "Hour", "total_trips": "Trips"},
                )
                fig.update_layout(height=350, showlegend=False)
                fig.update_xaxes(dtick=2)
                st.plotly_chart(apply_theme(fig), use_container_width=True)

    with row_t2[1]:
        if has_monthly:
            st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Monthly Metric Comparison</h3>", unsafe_allow_html=True)
            fig = go.Figure()
            month_metrics = ["avg_fare", "avg_distance", "avg_tip", "avg_duration"]
            colors_m = [COLORS["cyan"], COLORS["amber"], COLORS["green"], COLORS["pink"]]
            names_m = ["Avg Fare ($)", "Avg Distance (mi)", "Avg Tip ($)", "Avg Duration (min)"]

            for i, (m, c, n) in enumerate(zip(month_metrics, colors_m, names_m)):
                norm = monthly_data[m]
                norm_norm = norm / norm.max() if norm.max() > 0 else norm
                fig.add_trace(go.Bar(
                    name=n,
                    x=monthly_data["month_name"],
                    y=norm_norm,
                    marker_color=c,
                    opacity=0.8,
                    hovertemplate="%{x}<br>%{data.name}: %{customdata}<extra></extra>",
                    customdata=[[round(v, 2)] for v in monthly_data[m]],
                ))
            fig.update_layout(
                barmode="group",
                height=350,
                legend=dict(orientation="h", y=1.08, x=0.15),
                yaxis_title="Normalized Value (0–1)",
            )
            st.plotly_chart(apply_theme(fig), use_container_width=True)
        else:
            st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Revenue & Tips Over Time</h3>", unsafe_allow_html=True)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            if not filtered.empty:
                fig.add_trace(go.Scatter(x=filtered["trip_date"], y=filtered["total_revenue"], mode="lines", name="Revenue", line=dict(color=COLORS["green"], width=2)), secondary_y=False)
                fig.add_trace(go.Scatter(x=filtered["trip_date"], y=filtered["avg_tip_pct"], mode="lines", name="Tip %", line=dict(color=COLORS["pink"], width=2)), secondary_y=True)
            fig.update_layout(height=350, hovermode="x unified")
            st.plotly_chart(apply_theme(fig), use_container_width=True)

    st.divider()

    st.markdown("""
        <div class="insight-box">
            <p><strong style="color:#00d4ff;">Key Insights:</strong>
            Trip volume peaks at <strong>6 PM</strong> on weekdays. Weekends show a
            flatter distribution with higher late-night activity.
            <strong>Wednesday</strong> is the busiest day; <strong>Sunday</strong>
            is the quietest. January has slightly lower average fares than March.</p>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 3: LOCATIONS
# ═══════════════════════════════════════════════
with t3:
    has_locations = location_data is not None and not location_data.empty

    if has_locations:
        st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Top Pickup & Dropoff Locations</h3>", unsafe_allow_html=True)

        top_pu = location_data.groupby("PULocationID").agg(
            trips=("trips", "sum"), avg_fare=("avg_fare", "mean")
        ).reset_index().sort_values("trips", ascending=False).head(12)
        top_pu.columns = ["LocationID", "trips", "avg_fare"]

        top_do = location_data.groupby("DOLocationID").agg(
            trips=("trips", "sum"), avg_fare=("avg_fare", "mean")
        ).reset_index().sort_values("trips", ascending=False).head(12)
        top_do.columns = ["LocationID", "trips", "avg_fare"]

        row_l1 = st.columns(2)

        with row_l1[0]:
            fig = px.bar(
                top_pu.sort_values("trips"),
                y="LocationID", x="trips",
                orientation="h",
                color="trips",
                color_continuous_scale=[[0, "#1a1a3e"], [0.5, "#00d4ff"], [1, "#7c3aed"]],
                labels={"LocationID": "Zone ID", "trips": "Pickup Trips"},
                text_auto=",.0f",
            )
            fig.update_traces(showlegend=False)
            fig.update_layout(height=400, xaxis_title="Trip Count", yaxis_title="")
            st.plotly_chart(apply_theme(fig), use_container_width=True)

        with row_l1[1]:
            fig = px.bar(
                top_do.sort_values("trips"),
                y="LocationID", x="trips",
                orientation="h",
                color="trips",
                color_continuous_scale=[[0, "#1a1a3e"], [0.5, "#7c3aed"], [1, "#f59e0b"]],
                labels={"LocationID": "Zone ID", "trips": "Dropoff Trips"},
                text_auto=",.0f",
            )
            fig.update_traces(showlegend=False)
            fig.update_layout(height=400, xaxis_title="Trip Count", yaxis_title="")
            st.plotly_chart(apply_theme(fig), use_container_width=True)

        st.divider()

        st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Top Routes Flow (Sankey)</h3>", unsafe_allow_html=True)
        top_routes = location_data.head(20)
        all_nodes = list(set(top_routes["PULocationID"].tolist() + top_routes["DOLocationID"].tolist()))
        node_map = {n: i for i, n in enumerate(all_nodes)}

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                label=[f"PU {n}" for n in all_nodes],
                color=[COLORS["cyan"] for n in all_nodes],
            ),
            link=dict(
                source=[node_map[r["PULocationID"]] for _, r in top_routes.iterrows()],
                target=[node_map[r["DOLocationID"]] for _, r in top_routes.iterrows()],
                value=top_routes["trips"].tolist(),
                color=[COLORS["purple"] for _ in range(len(top_routes))],
                hovertemplate="%{source.label} → %{target.label}<br>Trips: %{value:,.0f}<extra></extra>",
            ),
        )])
        fig.update_layout(height=500)
        st.plotly_chart(apply_theme(fig), use_container_width=True)
    else:
        st.markdown("<h3 style='font-weight:600; margin-bottom:0.5rem;'>Trip Activity Overview</h3>", unsafe_allow_html=True)
        col_l1, col_l2 = st.columns(2)

        with col_l1:
            st.markdown("<h4 style='color:#94a3b8; font-weight:500;'>Daily Rush Hour Split</h4>", unsafe_allow_html=True)
            if not filtered.empty:
                rush_pivot = filtered[["trip_date", "morning_rush_trips", "evening_rush_trips"]].melt(
                    id_vars=["trip_date"], var_name="period", value_name="trips"
                )
                rush_pivot["period"] = rush_pivot["period"].map({
                    "morning_rush_trips": "Morning (6-10)",
                    "evening_rush_trips": "Evening (16-20)",
                })
                fig = px.area(
                    rush_pivot, x="trip_date", y="trips", color="period",
                    color_discrete_map={"Morning (6-10)": COLORS["amber"], "Evening (16-20)": COLORS["purple"]},
                )
                fig.update_layout(height=350, legend=dict(orientation="h", y=1.08, x=0.2))
                st.plotly_chart(apply_theme(fig), use_container_width=True)

        with col_l2:
            st.markdown("<h4 style='color:#94a3b8; font-weight:500;'>Daily Metrics Overview</h4>", unsafe_allow_html=True)
            if not filtered.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered["trip_date"], y=filtered["avg_distance"], mode="lines+markers", name="Avg Distance (mi)", line=dict(color=COLORS["green"], width=2)))
                fig.add_trace(go.Scatter(x=filtered["trip_date"], y=filtered["avg_duration_min"], mode="lines+markers", name="Avg Duration (min)", line=dict(color=COLORS["pink"], width=2)))
                fig.update_layout(height=350, legend=dict(orientation="h", y=1.08, x=0.2), hovermode="x unified")
                st.plotly_chart(apply_theme(fig), use_container_width=True)

        st.markdown("""
            <div style="background:rgba(255,255,255,0.03); border-radius:12px; padding:1rem; margin-top:1rem;
                        border:1px solid rgba(255,255,255,0.06);">
                <p style="color:#64748b; margin:0; font-size:0.9rem;">
                    <strong>Location-specific analysis</strong> (top pickup/dropoff zones, route Sankey diagram)
                    requires the full Parquet dataset. Deploy the complete pipeline to enable spatial analytics.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
        <div class="insight-box">
            <p><strong style="color:#00d4ff;">Key Insight:</strong>
            The busiest routes are concentrated in Manhattan. Taxi zones
            <strong>161</strong> (JFK Airport) and <strong>237</strong>
            (Upper East Side) are among the top pickup/dropoff locations,
            reflecting the mix of airport and intra-city trips.</p>
        </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 4: FARE PREDICTOR
# ═══════════════════════════════════════════════

with t4:
    st.markdown("""
        <div style="margin-bottom:0.5rem;">
            <h3 style="font-weight:600; margin:0;">Fare Predictor</h3>
            <p style="color:#64748b; margin:0;">
                GradientBoosting Regressor · MAE = <strong style="color:#00d4ff;">$0.94</strong>
                · RMSE = <strong style="color:#7c3aed;">$3.26</strong>
                · R² = <strong style="color:#f59e0b;">0.969</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

    model_ready = model is not None and le is not None

    p_col1, p_col2, p_col3 = st.columns(3)

    with p_col1:
        st.markdown("<p style='color:#94a3b8; font-size:0.85rem; font-weight:600;'>TRIP PARAMETERS</p>", unsafe_allow_html=True)
        trip_distance = st.slider("Trip Distance (miles)", 0.1, 30.0, 2.5, 0.1)
        passenger_count = st.selectbox("Passengers", [1, 2, 3, 4, 5, 6])
        payment_type = st.selectbox(
            "Payment", [1, 2],
            format_func=lambda x: "Credit Card" if x == 1 else "Cash",
        )
        time_of_day = st.selectbox(
            "Time of Day",
            ["morning_rush", "evening_rush", "off_peak"],
            format_func=lambda x: {
                "morning_rush": "Morning Rush (6–10)",
                "evening_rush": "Evening Rush (16–20)",
                "off_peak": "Off-Peak",
            }[x],
        )

    with p_col2:
        st.markdown("<p style='color:#94a3b8; font-size:0.85rem; font-weight:600;'>TIME CONTEXT</p>", unsafe_allow_html=True)
        pickup_hour = st.slider("Pickup Hour", 0, 23, 8)
        pickup_dow = st.selectbox(
            "Day of Week", list(range(7)),
            format_func=lambda x: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][x],
        )
        pickup_month = st.selectbox(
            "Month", [1, 2, 3],
            format_func=lambda x: ["January", "February", "March"][x - 1],
        )

    with p_col3:
        st.markdown("<p style='color:#94a3b8; font-size:0.85rem; font-weight:600;'>DURATION & SPEED</p>", unsafe_allow_html=True)
        trip_duration_min = st.slider("Trip Duration (min)", 1.0, 120.0, 15.0, 0.5)
        avg_speed_mph = st.slider("Avg Speed (mph)", 1.0, 60.0, 12.0, 0.5)

    st.divider()

    pred_col, features_col = st.columns([1, 1])

    with pred_col:
        if model_ready:
            predict_btn = st.button("Predict Fare", type="primary", use_container_width=True)

            if predict_btn:
                time_enc = le.transform([time_of_day])[0]
                input_data = np.array([[
                    trip_distance, passenger_count, pickup_hour, pickup_dow,
                    pickup_month, 161, 237, payment_type,
                    trip_duration_min, avg_speed_mph, time_enc,
                ]])
                prediction = round(float(model.predict(input_data)[0]), 2)

                st.markdown(f"""
                    <div style="background:linear-gradient(135deg, rgba(0,212,255,0.1), rgba(124,58,237,0.1));
                              border:1px solid rgba(0,212,255,0.2); border-radius:16px;
                              padding:1.5rem; text-align:center;">
                        <p style="color:#94a3b8; font-size:0.85rem; margin:0;">Predicted Fare</p>
                        <p style="font-size:3.5rem; font-weight:800; margin:0;
                                  background:linear-gradient(135deg, #00d4ff, #7c3aed);
                                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            ${prediction:.2f}
                        </p>
                        <p style="color:#64748b; font-size:0.85rem; margin:0.5rem 0 0 0;">
                            Based on 11 features · GradientBoosting model
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                              border-radius:16px; padding:2rem; text-align:center;">
                        <p style="color:#64748b; font-size:1.5rem; font-weight:300; margin:0;">—</p>
                        <p style="color:#64748b; margin:0;">Adjust parameters and click Predict</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("""
                <div style="margin-top:1rem;">
                    <p style="color:#475569; font-size:0.8rem;">
                        Model trained on 400K trips with 11 features.
                        Pickup/dropoff default to Manhattan zones 161 and 237.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                          border-radius:16px; padding:2rem; text-align:center;">
                    <p style="color:#64748b; font-size:3rem; font-weight:200; margin:0;">—</p>
                    <p style="color:#94a3b8; font-size:1.1rem; margin:0.5rem 0 0 0;">Model not loaded</p>
                    <p style="color:#64748b; font-size:0.85rem; margin-top:0.5rem;">
                        The ML model file (best_model.pkl) is not available in this environment.
                    </p>
                    <p style="color:#475569; font-size:0.8rem;">
                        Run the full pipeline locally to generate the model artifact.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    with features_col:
        st.markdown("<p style='color:#94a3b8; font-size:0.85rem; font-weight:600;'>CURRENT INPUT VALUES</p>", unsafe_allow_html=True)
        feature_df = pd.DataFrame({
            "Feature": [
                "Trip Distance", "Passengers", "Pickup Hour", "Day of Week",
                "Month", "Pickup Location", "Dropoff Location", "Payment Type",
                "Duration (min)", "Avg Speed (mph)", "Time of Day",
            ],
            "Value": [
                f"{trip_distance} mi",
                str(passenger_count),
                f"{pickup_hour}:00",
                ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][pickup_dow],
                ["Jan", "Feb", "Mar"][pickup_month - 1],
                "Zone 161 (Manhattan)",
                "Zone 237 (Manhattan)",
                "Credit Card" if payment_type == 1 else "Cash",
                f"{trip_duration_min} min",
                f"{avg_speed_mph} mph",
                time_of_day.replace("_", " ").title(),
            ],
        })
        st.dataframe(feature_df, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 5: ABOUT
# ═══════════════════════════════════════════════

with t5:
    st.markdown("""
        <div style="margin-bottom:1rem;">
            <h3 style="font-weight:600; margin:0;">About This Project</h3>
            <p style="color:#64748b; margin:0;">
                End-to-end data engineering + ML pipeline processing 8.8M NYC taxi trips.
            </p>
        </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns([3, 2])

    with a1:
        st.markdown("""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                        border-radius:16px; padding:1.5rem;">
        """, unsafe_allow_html=True)

        pipeline_steps = [
            ("01", "Ingestion", "Download raw Parquet files from NYC TLC S3 bucket"),
            ("02", "Quality Check", "Validate completeness, validity, and accuracy"),
            ("03", "Cleaning", "Remove anomalies, engineer time/speed/tip features"),
            ("04", "DuckDB Load", "Load 8.8M cleaned trips into OLAP database"),
            ("05", "dbt Transforms", "Stage data → daily summary & hourly pattern marts"),
            ("06", "ML Training", "Train & compare 3 models, track with MLflow"),
            ("07", "FastAPI Serving", "Deploy best model as /predict REST endpoint"),
            ("08", "Dashboard", "This interactive Streamlit + Plotly dashboard"),
        ]

        for num, title, desc in pipeline_steps:
            st.markdown(f"""
                <div style="display:flex; gap:0.75rem; padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                    <div style="font-size:0.85rem; font-weight:700; width:2rem; color:#00d4ff;">{num}</div>
                    <div>
                        <p style="font-weight:600; margin:0; color:#e2e8f0;">{title}</p>
                        <p style="color:#64748b; font-size:0.85rem; margin:0;">{desc}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with a2:
        st.markdown("""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                        border-radius:16px; padding:1.5rem; margin-bottom:1rem;">
                <p style="color:#94a3b8; font-size:0.85rem; font-weight:600; margin:0 0 1rem 0;">TECH STACK</p>
        """, unsafe_allow_html=True)

        techs = [
            ("Python 3.13", "Core language"),
            ("DuckDB", "Embedded OLAP engine"),
            ("dbt", "Data transformation"),
            ("scikit-learn", "ML models"),
            ("MLflow", "Experiment tracking"),
            ("FastAPI + Uvicorn", "REST API"),
            ("Streamlit", "Dashboard framework"),
            ("Plotly", "Interactive charts"),
        ]
        for name, desc in techs:
            st.markdown(f"""
                <div style="display:flex; gap:0.5rem; padding:0.35rem 0;">
                    <div>
                        <strong style="font-size:0.9rem; color:#e2e8f0;">{name}</strong>
                        <span style="color:#64748b; font-size:0.85rem;"> — {desc}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                        border-radius:16px; padding:1.5rem;">
                <p style="color:#94a3b8; font-size:0.85rem; font-weight:600; margin:0 0 0.75rem 0;">MODEL PERFORMANCE</p>
        """, unsafe_allow_html=True)

        perf = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "R²"],
            "Value": ["$0.94", "$3.26", "0.969"],
            "Interpretation": ["Excellent accuracy", "Low variance", "Strong fit"],
        })
        st.dataframe(perf, hide_index=True, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                    border-radius:16px; padding:1.5rem;">
            <p style="color:#94a3b8; font-size:0.85rem; font-weight:600; margin:0 0 0.75rem 0;">PIPELINE ARCHITECTURE</p>
            <pre style="background:rgba(0,0,0,0.3); border-radius:8px; padding:1rem; overflow-x:auto; color:#94a3b8; font-size:0.85rem; line-height:1.6;">
Raw Parquet (NYC TLC)
    │
    ├── download_data.py      ──►  data/raw/*.parquet
    ├── quality_check.py       ──►  Data quality report
    ├── clean_data.py         ──►  data/processed/trips_cleaned.parquet
    ├── load_to_duckdb.py     ──►  data/taxi.duckdb (raw_trips)
    ├── dbt transforms        ──►  stg_trips → mart_daily_summary, mart_hourly_patterns
    ├── train_model.py        ──►  MLflow → best_model.pkl
    ├── api.py                ──►  FastAPI /predict endpoint
    └── dashboard.py          ──►  Streamlit (this page)
            </pre>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
        <div style="text-align:center; padding:1rem;">
            <p style="color:#64748b;">
                Built by <strong style="color:#e2e8f0;">Brayan Hawald Ngowi</strong>
                ·
                <a href="https://github.com/masterbry" style="color:#00d4ff; text-decoration:none;">GitHub</a>
                ·
                <a href="https://master-bry.vercel.app" style="color:#7c3aed; text-decoration:none;">Portfolio</a>
            </p>
            <p style="color:#475569; font-size:0.8rem;">
                © 2023 · NYC Taxi Data Pipeline · All rights reserved
            </p>
        </div>
    """, unsafe_allow_html=True)
