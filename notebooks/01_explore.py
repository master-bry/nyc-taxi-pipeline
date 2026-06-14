"""
NYC Taxi Trip Data — Exploratory Data Analysis
================================================
Performs a comprehensive exploration of ~8.7M yellow taxi trips (Q1 2023).
Covers schema inspection, statistical summaries, quality checks, distribution
analysis, temporal patterns, spatial insights, and pairwise correlations.
"""

import duckdb
import pandas as pd

con = duckdb.connect()

# ──────────────────────────────────────────────
# 1. SCHEMA INSPECTION
# ──────────────────────────────────────────────
print("=" * 72)
print("SECTION 1: SCHEMA INSPECTION")
print("=" * 72)

schema = con.execute("""
    DESCRIBE SELECT * FROM 'data/raw/*.parquet' LIMIT 1
""").df()
print(schema.to_string(index=False))

# ──────────────────────────────────────────────
# 2. DATA VOLUME
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 2: DATA VOLUME")
print("=" * 72)

row_count = con.execute("""
    SELECT
        COUNT(*)                                                         AS total_rows,
        COUNT(DISTINCT filename)                                         AS file_count,
        MIN(tpep_pickup_datetime)                                        AS earliest_trip,
        MAX(tpep_pickup_datetime)                                        AS latest_trip,
        DATEDIFF('day', MIN(tpep_pickup_datetime), MAX(tpep_pickup_datetime)) AS date_span_days
    FROM read_csv('data/raw/*.parquet', filename=true)
""").df()
print(row_count.to_string(index=False))

# ──────────────────────────────────────────────
# 3. COLUMN-LEVEL QUALITY ASSESSMENT
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 3: COLUMN QUALITY ASSESSMENT")
print("=" * 72)

quality = con.execute("""
    SELECT
        column_name,
        total_rows,
        non_null,
        total_rows - non_null                                           AS null_count,
        ROUND(100.0 * (total_rows - non_null) / total_rows, 2)          AS null_pct,
        ROUND(100.0 * non_null / total_rows, 2)                         AS completeness_pct
    FROM (
        SELECT
            'VendorID'              AS column_name, COUNT(*) AS total_rows,
            COUNT(VendorID)         AS non_null FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'tpep_pickup_datetime', COUNT(*), COUNT(tpep_pickup_datetime) FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'tpep_dropoff_datetime', COUNT(*), COUNT(tpep_dropoff_datetime) FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'passenger_count',     COUNT(*), COUNT(passenger_count)     FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'trip_distance',       COUNT(*), COUNT(trip_distance)       FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'RatecodeID',          COUNT(*), COUNT(RatecodeID)          FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'PULocationID',        COUNT(*), COUNT(PULocationID)        FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'DOLocationID',        COUNT(*), COUNT(DOLocationID)        FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'payment_type',        COUNT(*), COUNT(payment_type)        FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'fare_amount',         COUNT(*), COUNT(fare_amount)         FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'extra',               COUNT(*), COUNT(extra)               FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'mta_tax',             COUNT(*), COUNT(mta_tax)             FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'tip_amount',          COUNT(*), COUNT(tip_amount)          FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'tolls_amount',        COUNT(*), COUNT(tolls_amount)        FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'total_amount',        COUNT(*), COUNT(total_amount)        FROM 'data/raw/*.parquet'
        UNION ALL
        SELECT 'congestion_surcharge', COUNT(*), COUNT(congestion_surcharge) FROM 'data/raw/*.parquet'
    )
    ORDER BY column_name
""").df()
print(quality.to_string(index=False))

# ──────────────────────────────────────────────
# 4. NUMERIC SUMMARY STATISTICS
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 4: NUMERIC SUMMARY STATISTICS")
print("=" * 72)

stats = con.execute("""
    SELECT
        'fare_amount'                                                   AS metric,
        COUNT(*)                                                        AS count,
        ROUND(MIN(fare_amount), 2)                                      AS min_val,
        ROUND(PERCENTILE_CONT(0.01)  WITHIN GROUP (ORDER BY fare_amount), 2) AS p01,
        ROUND(PERCENTILE_CONT(0.25)  WITHIN GROUP (ORDER BY fare_amount), 2) AS p25,
        ROUND(AVG(fare_amount), 2)                                      AS mean_val,
        ROUND(PERCENTILE_CONT(0.50)  WITHIN GROUP (ORDER BY fare_amount), 2) AS median,
        ROUND(PERCENTILE_CONT(0.75)  WITHIN GROUP (ORDER BY fare_amount), 2) AS p75,
        ROUND(PERCENTILE_CONT(0.99)  WITHIN GROUP (ORDER BY fare_amount), 2) AS p99,
        ROUND(MAX(fare_amount), 2)                                      AS max_val,
        ROUND(STDDEV(fare_amount), 2)                                   AS std_val
    FROM 'data/raw/*.parquet'
    WHERE fare_amount > 0
    UNION ALL
    SELECT
        'trip_distance',
        COUNT(*),
        ROUND(MIN(trip_distance), 2),
        ROUND(PERCENTILE_CONT(0.01)  WITHIN GROUP (ORDER BY trip_distance), 2),
        ROUND(PERCENTILE_CONT(0.25)  WITHIN GROUP (ORDER BY trip_distance), 2),
        ROUND(AVG(trip_distance), 2),
        ROUND(PERCENTILE_CONT(0.50)  WITHIN GROUP (ORDER BY trip_distance), 2),
        ROUND(PERCENTILE_CONT(0.75)  WITHIN GROUP (ORDER BY trip_distance), 2),
        ROUND(PERCENTILE_CONT(0.99)  WITHIN GROUP (ORDER BY trip_distance), 2),
        ROUND(MAX(trip_distance), 2),
        ROUND(STDDEV(trip_distance), 2)
    FROM 'data/raw/*.parquet'
    WHERE trip_distance > 0
    UNION ALL
    SELECT
        'tip_amount',
        COUNT(*),
        ROUND(MIN(tip_amount), 2),
        ROUND(PERCENTILE_CONT(0.01)  WITHIN GROUP (ORDER BY tip_amount), 2),
        ROUND(PERCENTILE_CONT(0.25)  WITHIN GROUP (ORDER BY tip_amount), 2),
        ROUND(AVG(tip_amount), 2),
        ROUND(PERCENTILE_CONT(0.50)  WITHIN GROUP (ORDER BY tip_amount), 2),
        ROUND(PERCENTILE_CONT(0.75)  WITHIN GROUP (ORDER BY tip_amount), 2),
        ROUND(PERCENTILE_CONT(0.99)  WITHIN GROUP (ORDER BY tip_amount), 2),
        ROUND(MAX(tip_amount), 2),
        ROUND(STDDEV(tip_amount), 2)
    FROM 'data/raw/*.parquet'
    WHERE tip_amount > 0
    UNION ALL
    SELECT
        'passenger_count',
        COUNT(*),
        ROUND(MIN(passenger_count), 2),
        ROUND(PERCENTILE_CONT(0.01)  WITHIN GROUP (ORDER BY passenger_count), 2),
        ROUND(PERCENTILE_CONT(0.25)  WITHIN GROUP (ORDER BY passenger_count), 2),
        ROUND(AVG(passenger_count), 2),
        ROUND(PERCENTILE_CONT(0.50)  WITHIN GROUP (ORDER BY passenger_count), 2),
        ROUND(PERCENTILE_CONT(0.75)  WITHIN GROUP (ORDER BY passenger_count), 2),
        ROUND(PERCENTILE_CONT(0.99)  WITHIN GROUP (ORDER BY passenger_count), 2),
        ROUND(MAX(passenger_count), 2),
        ROUND(STDDEV(passenger_count), 2)
    FROM 'data/raw/*.parquet'
    WHERE passenger_count > 0
    UNION ALL
    SELECT
        'trip_duration_min',
        COUNT(*),
        ROUND(MIN(EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0), 2),
        ROUND(PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0), 2),
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0), 2),
        ROUND(AVG(EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0), 2),
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0), 2),
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0), 2),
        ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0), 2),
        ROUND(MAX(EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0), 2),
        ROUND(STDDEV(EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0), 2)
    FROM 'data/raw/*.parquet'
    WHERE tpep_dropoff_datetime > tpep_pickup_datetime
""").df()
print(stats.to_string(index=False))

# ──────────────────────────────────────────────
# 5. DISTRIBUTIONS — CATEGORICAL & DISCRETE
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 5: CATEGORICAL DISTRIBUTIONS")
print("=" * 72)

# 5a. Passenger count
print("\n--- Passenger Count Distribution ---")
print(con.execute("""
    SELECT
        passenger_count,
        COUNT(*)                                                           AS trip_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct,
        ROUND(SUM(fare_amount))                                           AS total_fare
    FROM 'data/raw/*.parquet'
    WHERE passenger_count BETWEEN 1 AND 9
    GROUP BY passenger_count
    ORDER BY passenger_count
""").df().to_string(index=False))

# 5b. Payment type
print("\n--- Payment Type Distribution ---")
print(con.execute("""
    SELECT
        payment_type,
        CASE payment_type
            WHEN 1 THEN 'Credit Card'
            WHEN 2 THEN 'Cash'
            WHEN 3 THEN 'No Charge'
            WHEN 4 THEN 'Dispute'
            WHEN 5 THEN 'Unknown'
            WHEN 6 THEN 'Voided Trip'
            ELSE 'Other'
        END                                                               AS payment_desc,
        COUNT(*)                                                           AS trip_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct,
        ROUND(AVG(fare_amount), 2)                                        AS avg_fare,
        ROUND(AVG(tip_amount), 2)                                         AS avg_tip,
        ROUND(100.0 * SUM(tip_amount) / NULLIF(SUM(fare_amount), 0), 2)   AS tip_rate_pct
    FROM 'data/raw/*.parquet'
    WHERE payment_type IN (1, 2, 3, 4, 5, 6)
    GROUP BY payment_type
    ORDER BY payment_type
""").df().to_string(index=False))

# 5c. Ratecode
print("\n--- Ratecode Distribution ---")
print(con.execute("""
    SELECT
        RatecodeID,
        CASE RatecodeID
            WHEN 1 THEN 'Standard'
            WHEN 2 THEN 'JFK'
            WHEN 3 THEN 'Newark'
            WHEN 4 THEN 'Nassau / Westchester'
            WHEN 5 THEN 'Negotiated'
            WHEN 6 THEN 'Group Ride'
            WHEN 99 THEN 'Unknown'
            ELSE 'Other'
        END                                                               AS ratecode_desc,
        COUNT(*)                                                           AS trip_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct,
        ROUND(AVG(fare_amount), 2)                                        AS avg_fare,
        ROUND(AVG(trip_distance), 2)                                      AS avg_distance
    FROM 'data/raw/*.parquet'
    GROUP BY RatecodeID
    ORDER BY trip_count DESC
""").df().to_string(index=False))

# 5d. Vendor distribution
print("\n--- Vendor Distribution ---")
print(con.execute("""
    SELECT
        VendorID,
        CASE VendorID
            WHEN 1 THEN 'Creative Mobile Technologies'
            WHEN 2 THEN 'VeriFone Inc.'
            ELSE 'Other'
        END                                                               AS vendor_desc,
        COUNT(*)                                                           AS trip_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct,
        ROUND(AVG(fare_amount), 2)                                        AS avg_fare
    FROM 'data/raw/*.parquet'
    GROUP BY VendorID
    ORDER BY VendorID
""").df().to_string(index=False))

# ──────────────────────────────────────────────
# 6. TEMPORAL PATTERNS
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 6: TEMPORAL PATTERNS")
print("=" * 72)

# 6a. Monthly aggregates
print("\n--- Monthly Aggregates ---")
print(con.execute("""
    SELECT
        EXTRACT('year'  FROM tpep_pickup_datetime)                        AS yr,
        EXTRACT('month' FROM tpep_pickup_datetime)                        AS mo,
        COUNT(*)                                                           AS trip_count,
        ROUND(AVG(fare_amount), 2)                                        AS avg_fare,
        ROUND(AVG(trip_distance), 2)                                      AS avg_distance,
        ROUND(AVG(tip_amount), 2)                                         AS avg_tip,
        ROUND(SUM(total_amount))                                          AS total_revenue
    FROM 'data/raw/*.parquet'
    GROUP BY yr, mo
    ORDER BY yr, mo
""").df().to_string(index=False))

# 6b. Hourly demand profile
print("\n--- Hourly Trip Profile ---")
print(con.execute("""
    SELECT
        EXTRACT('hour' FROM tpep_pickup_datetime)                        AS pickup_hour,
        COUNT(*)                                                           AS trip_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct_of_day,
        ROUND(AVG(fare_amount), 2)                                        AS avg_fare,
        ROUND(AVG(trip_distance), 2)                                      AS avg_distance
    FROM 'data/raw/*.parquet'
    GROUP BY pickup_hour
    ORDER BY pickup_hour
""").df().to_string(index=False))

# 6c. Day-of-week profile
print("\n--- Day-of-Week Profile ---")
print(con.execute("""
    SELECT
        EXTRACT('dow' FROM tpep_pickup_datetime)                         AS dow_number,
        CASE EXTRACT('dow' FROM tpep_pickup_datetime)
            WHEN 0 THEN 'Sunday'
            WHEN 1 THEN 'Monday'
            WHEN 2 THEN 'Tuesday'
            WHEN 3 THEN 'Wednesday'
            WHEN 4 THEN 'Thursday'
            WHEN 5 THEN 'Friday'
            WHEN 6 THEN 'Saturday'
        END                                                               AS day_name,
        COUNT(*)                                                           AS trip_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct_of_week,
        ROUND(AVG(fare_amount), 2)                                        AS avg_fare,
        ROUND(AVG(trip_distance), 2)                                      AS avg_distance,
        ROUND(AVG(tip_amount), 2)                                         AS avg_tip
    FROM 'data/raw/*.parquet'
    GROUP BY dow_number
    ORDER BY dow_number
""").df().to_string(index=False))

# 6d. Weekday vs weekend
print("\n--- Weekday vs Weekend ---")
print(con.execute("""
    SELECT
        CASE WHEN EXTRACT('dow' FROM tpep_pickup_datetime) IN (0, 6)
             THEN 'Weekend' ELSE 'Weekday' END                           AS day_type,
        COUNT(*)                                                           AS trip_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct,
        ROUND(AVG(fare_amount), 2)                                        AS avg_fare,
        ROUND(AVG(trip_distance), 2)                                      AS avg_distance,
        ROUND(AVG(tip_amount), 2)                                         AS avg_tip,
        ROUND(AVG(EXTRACT('hour' FROM tpep_pickup_datetime)), 1)          AS avg_pickup_hour
    FROM 'data/raw/*.parquet'
    GROUP BY day_type
""").df().to_string(index=False))

# ──────────────────────────────────────────────
# 7. SPATIAL INSIGHTS
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 7: SPATIAL INSIGHTS")
print("=" * 72)

# 7a. Top pickup locations
print("\n--- Top 10 Pickup Locations ---")
print(con.execute("""
    SELECT
        PULocationID,
        COUNT(*)                                                           AS trip_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct,
        ROUND(AVG(fare_amount), 2)                                        AS avg_fare,
        ROUND(AVG(trip_distance), 2)                                      AS avg_distance
    FROM 'data/raw/*.parquet'
    GROUP BY PULocationID
    ORDER BY trip_count DESC
    LIMIT 10
""").df().to_string(index=False))

# 7b. Top dropoff locations
print("\n--- Top 10 Dropoff Locations ---")
print(con.execute("""
    SELECT
        DOLocationID,
        COUNT(*)                                                           AS trip_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct,
        ROUND(AVG(fare_amount), 2)                                        AS avg_fare
    FROM 'data/raw/*.parquet'
    GROUP BY DOLocationID
    ORDER BY trip_count DESC
    LIMIT 10
""").df().to_string(index=False))

# 7c. Most popular route
print("\n--- Top 10 Routes (Pickup → Dropoff) ---")
print(con.execute("""
    SELECT
        PULocationID,
        DOLocationID,
        COUNT(*)                                                           AS trip_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct,
        ROUND(AVG(fare_amount), 2)                                        AS avg_fare,
        ROUND(AVG(trip_distance), 2)                                      AS avg_distance,
        ROUND(AVG(tip_amount), 2)                                         AS avg_tip
    FROM 'data/raw/*.parquet'
    GROUP BY PULocationID, DOLocationID
    ORDER BY trip_count DESC
    LIMIT 10
""").df().to_string(index=False))

# ──────────────────────────────────────────────
# 8. ANOMALY & OUTLIER SCAN
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 8: ANOMALY SCAN")
print("=" * 72)

anomalies = con.execute("""
    SELECT
        'Fare <= 0'                                                       AS anomaly_type,
        COUNT(*)                                                           AS record_count,
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4) AS pct_of_total
    FROM 'data/raw/*.parquet' WHERE fare_amount <= 0
    UNION ALL
    SELECT
        'Fare > $500',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4)
    FROM 'data/raw/*.parquet' WHERE fare_amount > 500
    UNION ALL
    SELECT
        'Trip distance <= 0',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4)
    FROM 'data/raw/*.parquet' WHERE trip_distance <= 0
    UNION ALL
    SELECT
        'Trip distance > 200 miles',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4)
    FROM 'data/raw/*.parquet' WHERE trip_distance > 200
    UNION ALL
    SELECT
        'Passenger count <= 0',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4)
    FROM 'data/raw/*.parquet' WHERE passenger_count <= 0
    UNION ALL
    SELECT
        'Passenger count > 6',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4)
    FROM 'data/raw/*.parquet' WHERE passenger_count > 6
    UNION ALL
    SELECT
        'Dropoff <= Pickup time',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4)
    FROM 'data/raw/*.parquet' WHERE tpep_dropoff_datetime <= tpep_pickup_datetime
    UNION ALL
    SELECT
        'Trip duration > 3 hours',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4)
    FROM 'data/raw/*.parquet'
    WHERE EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0 > 180
""").df()
print(anomalies.to_string(index=False))

# ──────────────────────────────────────────────
# 9. CORRELATION BETWEEN KEY METRICS
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 9: PAIRWISE CORRELATIONS (sample ~200K rows)")
print("=" * 72)

corr = con.execute("""
    WITH sample AS (
        SELECT
            fare_amount,
            tip_amount,
            trip_distance,
            passenger_count,
            EXTRACT('hour' FROM tpep_pickup_datetime)                    AS pickup_hour,
            EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0   AS duration_min
        FROM 'data/raw/*.parquet'
        USING SAMPLE 200000
        WHERE fare_amount > 0
          AND trip_distance > 0
          AND tpep_dropoff_datetime > tpep_pickup_datetime
    )
    SELECT
        ROUND(CORR(fare_amount,   tip_amount),       4)      AS fare_vs_tip,
        ROUND(CORR(fare_amount,   trip_distance),    4)      AS fare_vs_distance,
        ROUND(CORR(fare_amount,   passenger_count),  4)      AS fare_vs_passengers,
        ROUND(CORR(fare_amount,   pickup_hour),      4)      AS fare_vs_hour,
        ROUND(CORR(fare_amount,   duration_min),     4)      AS fare_vs_duration,
        ROUND(CORR(tip_amount,    trip_distance),    4)      AS tip_vs_distance,
        ROUND(CORR(tip_amount,    duration_min),     4)      AS tip_vs_duration,
        ROUND(CORR(trip_distance, duration_min),     4)      AS distance_vs_duration
    FROM sample
""").df()
print(corr.to_string(index=False))

con.close()
print("\n" + "=" * 72)
print("EXPLORATORY ANALYSIS COMPLETE")
print("=" * 72)
