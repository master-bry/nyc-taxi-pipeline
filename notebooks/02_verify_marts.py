"""
dbt Mart Verification
======================
Validates the output of dbt transformations by inspecting the mart tables
and running data integrity assertions. Covers completeness, consistency,
referential integrity, trend anomaly detection, and time-series gaps.
"""

import duckdb
import pandas as pd

con = duckdb.connect("data/taxi.duckdb")

# ──────────────────────────────────────────────
# 1. TABLE AVAILABILITY
# ──────────────────────────────────────────────
print("=" * 72)
print("SECTION 1: TABLE AVAILABILITY")
print("=" * 72)

tables = con.execute("SHOW TABLES").df()
print(tables.to_string(index=False))

# ──────────────────────────────────────────────
# 2. MART_DAILY_SUMMARY — FULL INSPECTION
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 2: MART_DAILY_SUMMARY — FULL INSPECTION")
print("=" * 72)

print("\n--- Schema ---")
print(con.execute("DESCRIBE mart_daily_summary").df().to_string(index=False))

row_count_daily = con.execute("SELECT COUNT(*) AS row_count FROM mart_daily_summary").fetchone()[0]
print(f"\nRow count: {row_count_daily}")

print("\n--- First 14 Days ---")
print(con.execute("""
    SELECT
        trip_date,
        total_trips,
        avg_fare,
        total_revenue,
        avg_tip_pct,
        morning_rush_trips,
        evening_rush_trips
    FROM mart_daily_summary
    ORDER BY trip_date
    LIMIT 14
""").df().to_string(index=False))

print("\n--- Last 14 Days ---")
print(con.execute("""
    SELECT
        trip_date,
        total_trips,
        avg_fare,
        total_revenue,
        avg_tip_pct,
        morning_rush_trips,
        evening_rush_trips
    FROM mart_daily_summary
    ORDER BY trip_date DESC
    LIMIT 14
""").df().to_string(index=False))

# ──────────────────────────────────────────────
# 3. MART_HOURLY_PATTERNS — FULL INSPECTION
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 3: MART_HOURLY_PATTERNS — FULL INSPECTION")
print("=" * 72)

print("\n--- Schema ---")
print(con.execute("DESCRIBE mart_hourly_patterns").df().to_string(index=False))

row_count_hourly = con.execute("SELECT COUNT(*) AS row_count FROM mart_hourly_patterns").fetchone()[0]
print(f"\nRow count: {row_count_hourly}")

print("\n--- All Hourly Records ---")
print(con.execute("""
    SELECT
        pickup_hour,
        time_of_day,
        total_trips,
        avg_fare,
        avg_speed_mph
    FROM mart_hourly_patterns
    ORDER BY pickup_hour
""").df().to_string(index=False))

# ──────────────────────────────────────────────
# 4. DATA INTEGRITY ASSERTIONS
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 4: DATA INTEGRITY ASSERTIONS")
print("=" * 72)

assertions = con.execute("""
    SELECT
        'daily: trip_date is unique'                                      AS assertion,
        CASE WHEN COUNT(*) = COUNT(DISTINCT trip_date) THEN 'PASS' ELSE 'FAIL' END AS status
    FROM mart_daily_summary
    UNION ALL
    SELECT
        'daily: no null trip_date',
        CASE WHEN COUNT(*) = COUNT(trip_date) THEN 'PASS' ELSE 'FAIL' END
    FROM mart_daily_summary
    UNION ALL
    SELECT
        'daily: no null total_trips',
        CASE WHEN COUNT(*) = COUNT(total_trips) THEN 'PASS' ELSE 'FAIL' END
    FROM mart_daily_summary
    UNION ALL
    SELECT
        'daily: total_trips > 0',
        CASE WHEN MIN(total_trips) > 0 THEN 'PASS' ELSE 'FAIL' END
    FROM mart_daily_summary
    UNION ALL
    SELECT
        'daily: avg_fare in plausible range (0, 200)',
        CASE WHEN MIN(avg_fare) > 0 AND MAX(avg_fare) < 200 THEN 'PASS' ELSE 'FAIL' END
    FROM mart_daily_summary
    UNION ALL
    SELECT
        'hourly: pickup_hour is unique',
        CASE WHEN COUNT(*) = COUNT(DISTINCT pickup_hour) THEN 'PASS' ELSE 'FAIL' END
    FROM mart_hourly_patterns
    UNION ALL
    SELECT
        'hourly: no null pickup_hour',
        CASE WHEN COUNT(*) = COUNT(pickup_hour) THEN 'PASS' ELSE 'FAIL' END
    FROM mart_hourly_patterns
    UNION ALL
    SELECT
        'hourly: covers all 24 hours',
        CASE WHEN COUNT(*) = 24 THEN 'PASS' ELSE 'FAIL' END
    FROM mart_hourly_patterns
    UNION ALL
    SELECT
        'hourly: total_trips > 0 for all hours',
        CASE WHEN MIN(total_trips) > 0 THEN 'PASS' ELSE 'FAIL' END
    FROM mart_hourly_patterns
    UNION ALL
    SELECT
        'hourly: avg_fare in plausible range (0, 200)',
        CASE WHEN MIN(avg_fare) > 0 AND MAX(avg_fare) < 200 THEN 'PASS' ELSE 'FAIL' END
    FROM mart_hourly_patterns
    UNION ALL
    SELECT
        'hourly: avg_speed_mph in plausible range (0, 100)',
        CASE WHEN MIN(avg_speed_mph) > 0 AND MAX(avg_speed_mph) < 100 THEN 'PASS' ELSE 'FAIL' END
    FROM mart_hourly_patterns
""").df()
print(assertions.to_string(index=False))

# ──────────────────────────────────────────────
# 5. CONSISTENCY CHECKS
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 5: CROSS-TABLE CONSISTENCY")
print("=" * 72)

total_from_daily = con.execute("""
    SELECT SUM(total_trips) AS total_trips, SUM(total_revenue) AS total_revenue
    FROM mart_daily_summary
""").fetchone()

total_from_staging = con.execute("""
    SELECT COUNT(*) AS total_trips, ROUND(SUM(total_amount), 2) AS total_revenue
    FROM stg_trips
""").fetchone()

print(f"""
  Staging total trips:   {total_from_staging[0]:>10,}
  Daily mart total trips:{total_from_daily[0]:>10,}
  Match: {'PASS' if abs(total_from_staging[0] - total_from_daily[0]) < 1 else 'FAIL'}

  Staging total revenue:   ${total_from_staging[1]:>12,.2f}
  Daily mart total revenue:${total_from_daily[1]:>12,.2f}
  Match: {'PASS' if abs(total_from_staging[1] - total_from_daily[1]) < 1_000 else 'FAIL'}
""")

# ──────────────────────────────────────────────
# 6. TREND ANALYSIS
# ──────────────────────────────────────────────
print("=" * 72)
print("SECTION 6: TREND ANALYSIS")
print("=" * 72)

print("\n--- Best & Worst Days ---")
print(con.execute("""
    (SELECT 'Highest Trips' AS metric, trip_date::VARCHAR AS date,
            total_trips::VARCHAR AS value
     FROM mart_daily_summary ORDER BY total_trips DESC LIMIT 1)
    UNION ALL
    (SELECT 'Lowest Trips', trip_date::VARCHAR,
            total_trips::VARCHAR
     FROM mart_daily_summary ORDER BY total_trips ASC LIMIT 1)
    UNION ALL
    (SELECT 'Highest Revenue', trip_date::VARCHAR,
            '$' || total_revenue::VARCHAR
     FROM mart_daily_summary ORDER BY total_revenue DESC LIMIT 1)
    UNION ALL
    (SELECT 'Highest Avg Fare', trip_date::VARCHAR,
            '$' || avg_fare::VARCHAR
     FROM mart_daily_summary ORDER BY avg_fare DESC LIMIT 1)
    UNION ALL
    (SELECT 'Lowest Avg Fare', trip_date::VARCHAR,
            '$' || avg_fare::VARCHAR
     FROM mart_daily_summary ORDER BY avg_fare ASC LIMIT 1)
""").df().to_string(index=False))

# ──────────────────────────────────────────────
# 7. ANOMALY DETECTION — OUTLIER DAYS
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 7: ANOMALY DETECTION (DAILY)")
print("=" * 72)

daily_stats = con.execute("""
    SELECT
        ROUND(AVG(total_trips))                                      AS avg_daily_trips,
        ROUND(STDDEV(total_trips))                                   AS std_daily_trips,
        MIN(total_trips)                                             AS min_daily_trips,
        MAX(total_trips)                                             AS max_daily_trips,
        ROUND(AVG(total_revenue))                                    AS avg_daily_revenue,
        ROUND(STDDEV(total_revenue))                                 AS std_daily_revenue
    FROM mart_daily_summary
""").df()
print(daily_stats.to_string(index=False))

print("\n--- Days with Trips > 2σ from Mean ---")
print(con.execute("""
    WITH stats AS (
        SELECT
            AVG(total_trips) AS m,
            STDDEV(total_trips) AS s
        FROM mart_daily_summary
    )
    SELECT
        trip_date,
        total_trips,
        avg_fare,
        total_revenue,
        ROUND((total_trips - stats.m) / stats.s, 2) AS z_score
    FROM mart_daily_summary, stats
    WHERE ABS(total_trips - stats.m) > 2 * stats.s
    ORDER BY total_trips DESC
""").df().to_string(index=False))

# ──────────────────────────────────────────────
# 8. TIME-SERIES GAP CHECK
# ──────────────────────────────────────────────
print("\n" + "=" * 72)
print("SECTION 8: TIME-SERIES CONTINUITY")
print("=" * 72)

gaps = con.execute("""
    WITH daily AS (
        SELECT trip_date, LAG(trip_date) OVER (ORDER BY trip_date) AS prev_date
        FROM mart_daily_summary
    )
    SELECT
        prev_date || ' → ' || trip_date                               AS gap_range,
        DATEDIFF('day', prev_date, trip_date) - 1                      AS missing_days
    FROM daily
    WHERE DATEDIFF('day', prev_date, trip_date) > 1
    ORDER BY trip_date
""").df()
if len(gaps) > 0:
    print(gaps.to_string(index=False))
else:
    print("No gaps detected — daily time series is continuous.")

con.close()
print("\n" + "=" * 72)
print("MART VERIFICATION COMPLETE")
print("=" * 72)
