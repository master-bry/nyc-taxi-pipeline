# NYC Taxi Analytics Pipeline

End-to-end data engineering and machine learning project built on 8.7 million NYC taxi trips from Q1 2023. Demonstrates a production-style pipeline from raw data ingestion through transformation, ML modeling, API serving, and live dashboard.

## Live Demo

- Dashboard: [Deploy on Streamlit Cloud — see setup below]
- API Docs: `http://localhost:8000/docs` (FastAPI Swagger UI)

---

## Architecture

```
NYC TLC Parquet Files (raw data source)
             |
             v
  Python Ingestion Layer
  download_data.py — fetches monthly parquet files from NYC TLC
             |
             v
  Data Quality Check
  quality_check.py — audits nulls, negatives, date anomalies
             |
             v
  Cleaning + Feature Engineering
  clean_data.py — DuckDB SQL, removes 6.3% invalid rows
  adds pickup_hour, trip_duration_min, avg_speed_mph, time_of_day
             |
             v
  dbt Transformation Layer
  stg_trips (view) — renamed columns, type casting
  mart_daily_summary (table) — daily revenue and trip aggregates
  mart_hourly_patterns (table) — demand patterns by hour
             |
             v
  ML Training — MLflow tracked
  3 models compared: LinearRegression, RandomForest, GradientBoosting
  Best: GradientBoosting — MAE $0.94, RMSE $3.26, R2 0.969
             |
             v
  FastAPI Prediction Endpoint
  POST /predict — real-time fare prediction from trip features
  GET /stats — model metadata and performance metrics
             |
             v
  Streamlit Dashboard
  Live analytics: daily trends, hourly patterns, interactive fare predictor
```

---

## Results

| Metric | Value |
|--------|-------|
| Raw trips ingested | 9,384,487 |
| Trips after cleaning | 8,797,764 |
| Data removed | 6.3% (invalid fares, dates, distances) |
| Best model | GradientBoosting |
| MAE | $0.94 |
| RMSE | $3.26 |
| R2 Score | 0.969 |
| Training rows | 400,000 |
| Test rows | 100,000 |

---

## Data Quality Issues Found and Fixed

| Issue | Count | Action |
|-------|-------|--------|
| Null passenger count | 236,179 | Removed |
| Negative fares | 82,873 | Removed |
| Zero distance trips | 135,488 | Removed |
| Trips dated before 2023 (oldest: 2001) | multiple | Filtered to Q1 2023 only |
| Fares above $500 | small number | Removed as outliers |

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Language | Python 3.13 | Core pipeline |
| Storage | DuckDB | Columnar processing, handles GB-scale parquet without Spark overhead |
| Transformation | dbt-duckdb | SQL models, data tests, auto-generated documentation |
| ML Training | Scikit-learn | Model training and comparison |
| Experiment Tracking | MLflow | Logs params, metrics, and model artifacts |
| API | FastAPI | Serves fare predictions via REST |
| Dashboard | Streamlit | Interactive analytics and live predictor |
| Version Control | Git + GitHub | Source control |

---

## Project Structure

```
nyc-taxi-pipeline/
├── data/
│   ├── raw/                         # Downloaded parquet files from NYC TLC
│   └── processed/                   # Cleaned parquet after pipeline
├── src/
│   ├── ingestion/
│   │   ├── download_data.py         # Downloads 3 months of yellow taxi data
│   │   ├── quality_check.py         # Audits nulls, negatives, date range
│   │   ├── clean_data.py            # Cleaning rules and feature engineering
│   │   └── load_to_duckdb.py        # Loads cleaned parquet into taxi.duckdb
│   ├── ml/
│   │   ├── train_model.py           # Trains and compares 3 models via MLflow
│   │   ├── api.py                   # FastAPI app with /predict and /stats
│   │   └── artifacts/
│   │       └── best_model.pkl       # Serialized GradientBoosting model
│   └── dashboard.py                 # Streamlit analytics dashboard
├── taxi_transforms/                 # dbt project
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_trips.sql        # Renamed and cast columns
│   │   │   └── schema.yml           # Column descriptions and data tests
│   │   └── marts/
│   │       ├── mart_daily_summary.sql    # Daily trip and revenue aggregates
│   │       ├── mart_hourly_patterns.sql  # Hourly demand and speed patterns
│   │       └── schema.yml               # Uniqueness and not-null tests
│   └── dbt_project.yml
├── notebooks/
│   ├── 01_explore.py                # Schema and row count exploration
│   └── 02_verify_marts.py           # Verify dbt mart outputs
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Setup and Reproduction

### Requirements

- Python 3.11 or higher
- Git
- 4GB free disk space for data files

### Installation

```bash
git clone https://github.com/masterbry/nyc-taxi-pipeline.git
cd nyc-taxi-pipeline

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### dbt Profile Setup

Create the file `~/.dbt/profiles.yml`:

```yaml
taxi_transforms:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: /absolute/path/to/nyc-taxi-pipeline/data/taxi.duckdb
      threads: 2
```

### Run the Full Pipeline

```bash
# Step 1: Download raw data (approx 1.5GB)
python src/ingestion/download_data.py

# Step 2: Check data quality
python src/ingestion/quality_check.py

# Step 3: Clean and engineer features
python src/ingestion/clean_data.py

# Step 4: Load into DuckDB
python src/ingestion/load_to_duckdb.py

# Step 5: Run dbt transforms
cd taxi_transforms
dbt run
dbt test
cd ..

# Step 6: Train ML models
python src/ml/train_model.py

# Step 7: Start API server
uvicorn src.ml.api:app --port 8000

# Step 8: Start dashboard
streamlit run src/dashboard.py --server.port 8501
```

---

## API Reference

### POST /predict

Predicts fare amount from trip features.

Request body:

```json
{
  "trip_distance": 2.5,
  "passenger_count": 1,
  "pickup_hour": 8,
  "pickup_dow": 1,
  "pickup_month": 1,
  "pickup_location_id": 161,
  "dropoff_location_id": 237,
  "payment_type": 1,
  "trip_duration_min": 15.0,
  "avg_speed_mph": 12.0,
  "time_of_day": "morning_rush"
}
```

Response:

```json
{
  "predicted_fare_usd": 14.72,
  "model_name": "GradientBoostingRegressor",
  "confidence_note": "MAE=$0.94 on 100,000 test trips (R2=0.969)"
}
```

### GET /stats

Returns model metadata and performance metrics.

### GET /health

Returns API health status.

---

## Key Insights from the Data

- Evening rush (16:00-19:00) has the highest trip volume at 2.35 million trips across Q1
- Early morning hours (04:00-05:00) have the highest average fares ($22-$27) likely due to airport trips
- Average NYC taxi speed drops to 10 mph during midday (10:00-15:00) versus 18 mph at 04:00
- New Year weekend (Jan 1-2) shows lower trip counts with higher average fares
- Tip percentage averages 20.9% across all payment types

---

## Model Comparison

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| LinearRegression | $1.49 | $4.33 | 0.945 |
| RandomForest | $0.94 | $3.28 | 0.968 |
| GradientBoosting | $0.94 | $3.26 | 0.969 |

GradientBoosting selected as best model. Top predictive features are trip duration, trip distance, and pickup location.

---

## Data Source

NYC Taxi and Limousine Commission Trip Record Data.
Available at: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Data is released monthly in Parquet format. This project uses Yellow Taxi data for January, February, and March 2023.

---

## Author

**Brayan Hawald Ngowi**
Final-year Software Engineering student, University of Dodoma (UDOM), Tanzania.
Backend and Infrastructure experience including UNICEF/UDOM STEM Project and eMazingira environmental platform.

- GitHub: [github.com/masterbry](https://github.com/masterbry)
- Portfolio: [master-bry.vercel.app](https://master-bry.vercel.app)

live: https://nyc-taxi2026.streamlit.app/
