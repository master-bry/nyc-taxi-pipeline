"""
Data Quality Report
====================
Validates raw NYC TLC trip data (Q1 2023) across completeness, validity,
consistency, and accuracy dimensions. Produces a structured summary with
actionable thresholds and PASS/FAIL flags for each check.
"""

import duckdb
from datetime import datetime

con = duckdb.connect()

YEAR = 2023
QUARTER = 1
REPORT_DATE = datetime.now().strftime("%Y-%m-%d %H:%M")

HEADER = f"""
{'=' * 72}
DATA QUALITY REPORT
NYC Yellow Taxi — Q{YEAR} Q{QUARTER}
Generated: {REPORT_DATE}
{'=' * 72}
"""
print(HEADER)

# ──────────────────────────────────────────────
# 1. COMPLETENESS — NULL ANALYSIS
# ──────────────────────────────────────────────
print("─" * 72)
print("DIMENSION: COMPLETENESS")
print("─" * 72)

nulls = con.execute("""
    SELECT
        COUNT(*)                                                          AS total_rows,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct,
        COUNT(*) - COUNT(VendorID)                                        AS null_vendor,
        COUNT(*) - COUNT(tpep_pickup_datetime)                            AS null_pickup,
        COUNT(*) - COUNT(tpep_dropoff_datetime)                           AS null_dropoff,
        COUNT(*) - COUNT(passenger_count)                                 AS null_passengers,
        COUNT(*) - COUNT(trip_distance)                                   AS null_distance,
        COUNT(*) - COUNT(RatecodeID)                                      AS null_ratecode,
        COUNT(*) - COUNT(PULocationID)                                    AS null_pu_location,
        COUNT(*) - COUNT(DOLocationID)                                    AS null_do_location,
        COUNT(*) - COUNT(payment_type)                                    AS null_payment,
        COUNT(*) - COUNT(fare_amount)                                     AS null_fare,
        COUNT(*) - COUNT(extra)                                           AS null_extra,
        COUNT(*) - COUNT(mta_tax)                                         AS null_mta_tax,
        COUNT(*) - COUNT(tip_amount)                                      AS null_tip,
        COUNT(*) - COUNT(tolls_amount)                                    AS null_tolls,
        COUNT(*) - COUNT(total_amount)                                    AS null_total,
        COUNT(*) - COUNT(congestion_surcharge)                            AS null_congestion
    FROM 'data/raw/*.parquet'
""").df().T
nulls.columns = ["value"]
print(nulls.to_string(header=False))

# ──────────────────────────────────────────────
# 2. VALIDITY — Domain & Range Checks
# ──────────────────────────────────────────────
print("\n" + "─" * 72)
print("DIMENSION: VALIDITY")
print("─" * 72)

validity = con.execute("""
    SELECT
        'Negative or zero fare'                                           AS check_name,
        COUNT(*)                                                           AS records,
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4) AS pct,
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END                AS status
    FROM 'data/raw/*.parquet' WHERE fare_amount <= 0
    UNION ALL
    SELECT
        'Fare exceeds $500',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM 'data/raw/*.parquet' WHERE fare_amount > 500
    UNION ALL
    SELECT
        'Zero or negative distance',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM 'data/raw/*.parquet' WHERE trip_distance <= 0
    UNION ALL
    SELECT
        'Distance exceeds 200 miles',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM 'data/raw/*.parquet' WHERE trip_distance > 200
    UNION ALL
    SELECT
        'Invalid passenger count (<=0)',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM 'data/raw/*.parquet' WHERE passenger_count <= 0
    UNION ALL
    SELECT
        'Invalid passenger count (>6)',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM 'data/raw/*.parquet' WHERE passenger_count > 6
    UNION ALL
    SELECT
        'Dropoff time <= Pickup time',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM 'data/raw/*.parquet' WHERE tpep_dropoff_datetime <= tpep_pickup_datetime
    UNION ALL
    SELECT
        'Trip duration exceeds 180 minutes',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM 'data/raw/*.parquet'
    WHERE EPOCH(tpep_dropoff_datetime - tpep_pickup_datetime) / 60.0 > 180
    UNION ALL
    SELECT
        'Negative tip amount',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM 'data/raw/*.parquet' WHERE tip_amount < 0
    UNION ALL
    SELECT
        'Negative total amount',
        COUNT(*),
        ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM 'data/raw/*.parquet'), 4),
        CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
    FROM 'data/raw/*.parquet' WHERE total_amount <= 0
""").df()
print(validity.to_string(index=False))

# ──────────────────────────────────────────────
# 3. ACCURACY — Statistical Profile
# ──────────────────────────────────────────────
print("\n" + "─" * 72)
print("DIMENSION: ACCURACY (Statistical Profile)")
print("─" * 72)

print("\n--- Fare Amount Distribution ---")
print(con.execute("""
    SELECT
        COUNT(*)                                                           AS total_fares,
        ROUND(MIN(fare_amount), 2)                                        AS min_fare,
        ROUND(PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY fare_amount), 2)   AS p01,
        ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY fare_amount), 2)   AS p05,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY fare_amount), 2)   AS p25,
        ROUND(AVG(fare_amount), 2)                                          AS avg_fare,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY fare_amount), 2)   AS median,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY fare_amount), 2)   AS p75,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY fare_amount), 2)   AS p95,
        ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY fare_amount), 2)   AS p99,
        ROUND(MAX(fare_amount), 2)                                          AS max_fare,
        ROUND(STDDEV(fare_amount), 2)                                       AS std_fare
    FROM 'data/raw/*.parquet'
    WHERE fare_amount > 0
""").df().to_string(index=False))

print("\n--- Trip Distance Distribution ---")
print(con.execute("""
    SELECT
        COUNT(*)                                                           AS total,
        ROUND(MIN(trip_distance), 2)                                       AS min_dist,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY trip_distance), 2) AS p25,
        ROUND(AVG(trip_distance), 2)                                       AS avg_dist,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY trip_distance), 2) AS median,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY trip_distance), 2) AS p75,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY trip_distance), 2) AS p95,
        ROUND(MAX(trip_distance), 2)                                       AS max_dist
    FROM 'data/raw/*.parquet'
    WHERE trip_distance > 0
""").df().to_string(index=False))

print("\n--- Tip Amount Distribution (non-zero) ---")
print(con.execute("""
    SELECT
        COUNT(*)                                                           AS total_tips,
        ROUND(MIN(tip_amount), 2)                                          AS min_tip,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY tip_amount), 2) AS p25,
        ROUND(AVG(tip_amount), 2)                                          AS avg_tip,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY tip_amount), 2) AS median_tip,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY tip_amount), 2) AS p75,
        ROUND(MAX(tip_amount), 2)                                          AS max_tip
    FROM 'data/raw/*.parquet'
    WHERE tip_amount > 0
""").df().to_string(index=False))

# ──────────────────────────────────────────────
# 4. CONSISTENCY — Date Range & Temporal
# ──────────────────────────────────────────────
print("\n" + "─" * 72)
print("DIMENSION: CONSISTENCY (Date Range & Temporal)")
print("─" * 72)

date_check = con.execute("""
    SELECT
        MIN(tpep_pickup_datetime)                                         AS earliest_trip,
        MAX(tpep_pickup_datetime)                                         AS latest_trip,
        COUNT(DISTINCT EXTRACT('month' FROM tpep_pickup_datetime))        AS distinct_months,
        COUNT(DISTINCT EXTRACT('day' FROM tpep_pickup_datetime))          AS distinct_days,
        CASE
            WHEN MIN(tpep_pickup_datetime) >= '2023-01-01'
             AND MAX(tpep_pickup_datetime) <= '2023-04-01'
            THEN 'PASS' ELSE 'FAIL'
        END                                                               AS q1_range_check,
        CASE
            WHEN COUNT(DISTINCT EXTRACT('month' FROM tpep_pickup_datetime)) = 3
            THEN 'PASS' ELSE 'FAIL'
        END                                                               AS month_coverage
    FROM 'data/raw/*.parquet'
""").df()
print(date_check.to_string(index=False))

# ──────────────────────────────────────────────
# 5. SUMMARY — Record Counts by Month
# ──────────────────────────────────────────────
print("\n" + "─" * 72)
print("RECORD COUNTS BY MONTH")
print("─" * 72)

monthly = con.execute("""
    SELECT
        EXTRACT('month' FROM tpep_pickup_datetime)                       AS month,
        COUNT(*)                                                           AS row_count,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                AS pct
    FROM 'data/raw/*.parquet'
    GROUP BY month
    ORDER BY month
""").df()
print(monthly.to_string(index=False))

# ──────────────────────────────────────────────
# 6. OVERALL RESULTS
# ──────────────────────────────────────────────
print("\n" + "─" * 72)
print("OVERALL QUALITY SCORE")
print("─" * 72)

failed = validity[validity["status"] == "FAIL"]
pass_count = len(validity[validity["status"] == "PASS"])
fail_count = len(failed)
total_checks = len(validity)
score = round(100.0 * pass_count / total_checks, 1) if total_checks else 0

print(f"  Checks run:        {total_checks}")
print(f"  Passed:            {pass_count}")
print(f"  Failed:            {fail_count}")
print(f"  Quality Score:     {score}%")

if fail_count > 0:
    print(f"\n  Action required on {fail_count} check(s):")
    for _, row in failed.iterrows():
        print(f"    - {row['check_name']}: {int(row['records']):,} records ({row['pct']}%)")

print(f"\n  Recommended action: run clean_data.py to remediate failures.")

con.close()
print("\n" + "=" * 72)
print("QUALITY CHECK COMPLETE")
print("=" * 72)
