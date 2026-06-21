"""
Data loading and query utilities for NYC Taxi Pipeline.

Provides functions to connect to DuckDB and retrieve analytics data.
"""

from typing import Optional
import duckdb
import pandas as pd
import os


def get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """
    Get a DuckDB connection.

    Works locally with DuckDB file, falls back to in-memory database on cloud.

    Args:
        read_only: Whether to open database in read-only mode (default: True)

    Returns:
        duckdb.DuckDBPyConnection: Active database connection

    Raises:
        FileNotFoundError: If DuckDB database cannot be accessed
    """
    db_path = "data/taxi.duckdb"
    if os.path.exists(db_path):
        return duckdb.connect(db_path, read_only=read_only)
    return duckdb.connect()


def get_daily_summary() -> pd.DataFrame:
    """
    Get daily trip summary statistics.

    Queries mart_daily_summary table if available, otherwise aggregates from parquet.

    Returns:
        pd.DataFrame: Daily trip statistics with columns:
            - trip_date: Date of trips
            - total_trips: Number of trips
            - avg_fare: Average fare amount
            - total_revenue: Total revenue
            - avg_tip_pct: Average tip percentage

    Raises:
        Exception: If query execution fails
    """
    con = get_connection()
    parquet_path = "data/processed/trips_cleaned.parquet"
    
    tables = con.execute("SHOW TABLES").df()["name"].tolist()
    
    if "mart_daily_summary" not in tables:
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


def get_hourly_patterns() -> pd.DataFrame:
    """
    Get hourly demand and performance patterns.

    Queries mart_hourly_patterns table if available, otherwise aggregates from parquet.

    Returns:
        pd.DataFrame: Hourly statistics with columns:
            - pickup_hour: Hour of day (0-23)
            - time_of_day: Time period category
            - total_trips: Number of trips in hour
            - avg_fare: Average fare amount
            - avg_speed_mph: Average speed in miles per hour

    Raises:
        Exception: If query execution fails
    """
    con = get_connection()
    parquet_path = "data/processed/trips_cleaned.parquet"
    
    tables = con.execute("SHOW TABLES").df()["name"].tolist()
    
    if "mart_hourly_patterns" not in tables:
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