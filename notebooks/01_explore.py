import duckdb
import pandas as pd

# DuckDB inasoma Parquet moja kwa moja — haraka sana
con = duckdb.connect()

# Ona schema
print("=== SCHEMA ===")
print(con.execute("""
    DESCRIBE SELECT * FROM 'data/raw/*.parquet' LIMIT 1
""").df())

# Ona saizi
print("\n=== ROW COUNT ===")
print(con.execute("""
    SELECT COUNT(*) as total_rows
    FROM 'data/raw/*.parquet'
""").fetchone())

# Sample rows 5
print("\n=== SAMPLE DATA ===")
print(con.execute("""
    SELECT
        tpep_pickup_datetime,
        tpep_dropoff_datetime,
        passenger_count,
        trip_distance,
        fare_amount,
        tip_amount,
        total_amount,
        PULocationID,
        DOLocationID
    FROM 'data/raw/*.parquet'
    LIMIT 5
""").df().to_string())