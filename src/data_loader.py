import duckdb
import pandas as pd
import os

def get_connection():
    # Works locally with DuckDB file, falls back to parquet on cloud
    db_path = "data/taxi.duckdb"
    if os.path.exists(db_path):
        return duckdb.connect(db_path, read_only=True)
    return duckdb.connect()

def get_daily_summary():
    con = get_connection()
    parquet_path = "data/processed/trips_cleaned.parquet"
    if not con.execute("SHOW TABLES").df()["name"].str.contains("mart_daily_summary").any():
        return con.execute(f"""
            SELECT
                DATE_TRUNC('day', tpep_pickup_datetime) AS trip_date,
                COUNT(*) AS total_trips,
                ROUND(AVG(fare_amount), 2) AS avg_fare,
                ROUND(SUM(total_amount), 2) AS total_revenue,
                ROUND(AVG(tip_pct), 2) AS avg_tip_pct
            FROM read_parquet('{parquet_path}')
            GROUP BY 1 ORDER BY 1
        """).df()
    return con.execute("SELECT * FROM mart_daily_summary ORDER BY trip_date").df()

def get_hourly_patterns():
    con = get_connection()
    parquet_path = "data/processed/trips_cleaned.parquet"
    if not con.execute("SHOW TABLES").df()["name"].str.contains("mart_hourly_patterns").any():
        return con.execute(f"""
            SELECT
                pickup_hour,
                time_of_day,
                COUNT(*) AS total_trips,
                ROUND(AVG(fare_amount), 2) AS avg_fare,
                ROUND(AVG(avg_speed_mph), 2) AS avg_speed_mph
            FROM read_parquet('{parquet_path}')
            GROUP BY 1, 2 ORDER BY 1
        """).df()
    return con.execute("SELECT * FROM mart_hourly_patterns ORDER BY pickup_hour").df()