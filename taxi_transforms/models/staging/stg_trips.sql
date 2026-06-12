-- Staging model: light renaming and type casting only.
-- No business logic here.

SELECT
    VendorID                                    AS vendor_id,
    tpep_pickup_datetime                        AS pickup_at,
    tpep_dropoff_datetime                       AS dropoff_at,
    CAST(passenger_count AS INTEGER)            AS passenger_count,
    trip_distance,
    CAST(PULocationID AS INTEGER)               AS pickup_location_id,
    CAST(DOLocationID AS INTEGER)               AS dropoff_location_id,
    CAST(payment_type AS INTEGER)               AS payment_type,
    fare_amount,
    tip_amount,
    tolls_amount,
    total_amount,
    congestion_surcharge,
    pickup_hour,
    pickup_dow,
    pickup_month,
    trip_duration_min,
    avg_speed_mph,
    tip_pct,
    time_of_day
FROM {{ source('main', 'raw_trips') }}