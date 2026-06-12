import duckdb
import os

con = duckdb.connect()
os.makedirs("data/processed", exist_ok=True)

print(" Starting cleaning pipeline...")

# ============================================================
# HATUA 1: Angalia total kabla ya cleaning
# ============================================================
total_before = con.execute("""
    SELECT COUNT(*) FROM 'data/raw/*.parquet'
""").fetchone()[0]
print(f"\n Rows BEFORE cleaning: {total_before:,}")

# ============================================================
# HATUA 2: Apply cleaning rules + feature engineering
# ============================================================
print("  Applying cleaning rules...")

con.execute("""
CREATE OR REPLACE TABLE cleaned_trips AS

SELECT
    -- Original columns (zilizokuwepo)
    VendorID,
    tpep_pickup_datetime,
    tpep_dropoff_datetime,
    passenger_count,
    trip_distance,
    RatecodeID,
    PULocationID,
    DOLocationID,
    payment_type,
    fare_amount,
    tip_amount,
    tolls_amount,
    total_amount,
    congestion_surcharge,

    --  FEATURE ENGINEERING (columns mpya)
    EXTRACT(hour   FROM tpep_pickup_datetime) AS pickup_hour,
    EXTRACT(dow    FROM tpep_pickup_datetime) AS pickup_dow,   -- 0=Sunday
    EXTRACT(month  FROM tpep_pickup_datetime) AS pickup_month,

    -- Trip duration kwa dakika
    ROUND(
        DATEDIFF('minute', tpep_pickup_datetime, tpep_dropoff_datetime)
    , 2) AS trip_duration_min,

    -- Speed (mph) kwa kugundua outliers baadaye
    CASE
        WHEN DATEDIFF('minute', tpep_pickup_datetime, tpep_dropoff_datetime) > 0
        THEN ROUND(trip_distance /
            (DATEDIFF('minute', tpep_pickup_datetime, tpep_dropoff_datetime) / 60.0)
        , 2)
        ELSE NULL
    END AS avg_speed_mph,

    -- Tip percentage
    CASE
        WHEN fare_amount > 0
        THEN ROUND((tip_amount / fare_amount) * 100, 2)
        ELSE 0
    END AS tip_pct,

    -- Rush hour flag
    CASE
        WHEN EXTRACT(hour FROM tpep_pickup_datetime) BETWEEN 7 AND 9   THEN 'morning_rush'
        WHEN EXTRACT(hour FROM tpep_pickup_datetime) BETWEEN 16 AND 19 THEN 'evening_rush'
        ELSE 'off_peak'
    END AS time_of_day

FROM 'data/raw/*.parquet'

WHERE
    --  CLEANING RULES
    fare_amount        > 0                          -- Hakuna negative/zero fare
    AND fare_amount    < 500                        -- Ondoa extreme outliers
    AND trip_distance  > 0                          -- Trip lazima iwe na umbali
    AND trip_distance  < 200                        -- Max realistic distance
    AND passenger_count >= 1                        -- Lazima kuwe na abiria
    AND passenger_count <= 6                        -- Max capacity ya taxi
    AND total_amount   > 0
    AND tpep_pickup_datetime >= '2023-01-01'        -- Ondoa data ya 2001
    AND tpep_pickup_datetime <  '2023-04-01'        -- Ndani ya Q1 2023 tu
    AND tpep_dropoff_datetime > tpep_pickup_datetime -- Dropoff lazima iwe baada ya pickup
    AND DATEDIFF('minute',
        tpep_pickup_datetime,
        tpep_dropoff_datetime) <= 180               -- Max 3 hours kwa trip
""")

# ============================================================
# HATUA 3: Angalia total baada ya cleaning
# ============================================================
total_after = con.execute("SELECT COUNT(*) FROM cleaned_trips").fetchone()[0]
removed = total_before - total_after
pct_removed = (removed / total_before) * 100

print(f" Rows AFTER cleaning:  {total_after:,}")
print(f"  Rows removed:         {removed:,} ({pct_removed:.1f}%)")

# ============================================================
# HATUA 4: Quick stats za cleaned data
# ============================================================
print("\n CLEANED DATA STATS:")
print(con.execute("""
    SELECT
        ROUND(AVG(fare_amount), 2)        AS avg_fare,
        ROUND(AVG(trip_distance), 2)      AS avg_distance_miles,
        ROUND(AVG(trip_duration_min), 2)  AS avg_duration_min,
        ROUND(AVG(tip_pct), 2)            AS avg_tip_pct,
        ROUND(AVG(avg_speed_mph), 2)      AS avg_speed_mph
    FROM cleaned_trips
""").df().to_string())

print("\n TRIPS BY TIME OF DAY:")
print(con.execute("""
    SELECT time_of_day, COUNT(*) as trips,
           ROUND(AVG(fare_amount), 2) as avg_fare
    FROM cleaned_trips
    GROUP BY time_of_day
    ORDER BY trips DESC
""").df().to_string())

# ============================================================
# HATUA 5: Save kwa Parquet — processed data
# ============================================================
print("\n Saving cleaned data to data/processed/...")
con.execute("""
    COPY cleaned_trips
    TO 'data/processed/trips_cleaned.parquet'
    (FORMAT PARQUET)
""")

print(" Pipeline complete! File: data/processed/trips_cleaned.parquet")