-- Hourly aggregation: reveals demand patterns by hour of day.
-- Used for ML feature analysis and dashboard charts.

SELECT
    pickup_hour,
    time_of_day,
    COUNT(*)                            AS total_trips,
    ROUND(AVG(fare_amount), 2)          AS avg_fare,
    ROUND(AVG(trip_distance), 2)        AS avg_distance,
    ROUND(AVG(tip_pct), 2)             AS avg_tip_pct,
    ROUND(AVG(avg_speed_mph), 2)        AS avg_speed_mph
FROM {{ ref('stg_trips') }}
GROUP BY pickup_hour, time_of_day
ORDER BY pickup_hour