import duckdb

con = duckdb.connect()

print("=" * 50)
print("DATA QUALITY REPORT")
print("=" * 50)

# 1. Null counts
print("\n NULL VALUES PER COLUMN:")
print(con.execute("""
    SELECT
        COUNT(*) - COUNT(passenger_count)    AS null_passenger,
        COUNT(*) - COUNT(trip_distance)      AS null_distance,
        COUNT(*) - COUNT(fare_amount)        AS null_fare,
        COUNT(*) - COUNT(tpep_pickup_datetime) AS null_pickup,
        COUNT(*) - COUNT(PULocationID)       AS null_pu_location
    FROM 'data/raw/*.parquet'
""").df().to_string())

# 2. Negative / zero values (dirty data)
print("\n SUSPICIOUS VALUES:")
print(con.execute("""
    SELECT
        COUNT(*) FILTER (WHERE fare_amount <= 0)     AS negative_fare,
        COUNT(*) FILTER (WHERE trip_distance <= 0)   AS zero_distance,
        COUNT(*) FILTER (WHERE passenger_count <= 0) AS zero_passengers,
        COUNT(*) FILTER (WHERE total_amount <= 0)    AS negative_total
    FROM 'data/raw/*.parquet'
""").df().to_string())

# 3. Date range — kuhakikisha data iko ndani ya 2023 Q1
print("\n DATE RANGE:")
print(con.execute("""
    SELECT
        MIN(tpep_pickup_datetime) AS earliest_trip,
        MAX(tpep_pickup_datetime) AS latest_trip
    FROM 'data/raw/*.parquet'
""").df().to_string())

# 4. Outliers za fare
print("\n FARE AMOUNT STATS:")
print(con.execute("""
    SELECT
        MIN(fare_amount)                    AS min_fare,
        MAX(fare_amount)                    AS max_fare,
        ROUND(AVG(fare_amount), 2)          AS avg_fare,
        ROUND(PERCENTILE_CONT(0.95)
            WITHIN GROUP (ORDER BY fare_amount), 2) AS p95_fare
    FROM 'data/raw/*.parquet'
    WHERE fare_amount > 0
""").df().to_string())

print("\n Quality check complete.")