import duckdb

con = duckdb.connect("data/taxi.duckdb")

print("=== DAILY SUMMARY (first 7 days) ===")
print(con.execute("""
    SELECT trip_date, total_trips, avg_fare, total_revenue
    FROM mart_daily_summary
    ORDER BY trip_date
    LIMIT 7
""").df().to_string())

print("\n=== HOURLY PATTERNS ===")
print(con.execute("""
    SELECT pickup_hour, time_of_day, total_trips, avg_fare, avg_speed_mph
    FROM mart_hourly_patterns
    ORDER BY pickup_hour
""").df().to_string())

con.close()