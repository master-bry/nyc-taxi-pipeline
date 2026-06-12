-- Business-level aggregation: daily trip summary.
-- Used for dashboard and trend analysis.

SELECT
    DATE_TRUNC('day', pickup_at)        AS trip_date,
    COUNT(*)                            AS total_trips,
    ROUND(AVG(fare_amount), 2)          AS avg_fare,
    ROUND(AVG(trip_distance), 2)        AS avg_distance,
    ROUND(AVG(trip_duration_min), 2)    AS avg_duration_min,
    ROUND(SUM(total_amount), 2)         AS total_revenue,
    ROUND(AVG(tip_pct), 2)              AS avg_tip_pct,
    COUNT(*) FILTER (
        WHERE time_of_day = 'morning_rush'
    )                                   AS morning_rush_trips,
    COUNT(*) FILTER (
        WHERE time_of_day = 'evening_rush'
    )                                   AS evening_rush_trips
FROM {{ ref('stg_trips') }}
GROUP BY DATE_TRUNC('day', pickup_at)
ORDER BY trip_date