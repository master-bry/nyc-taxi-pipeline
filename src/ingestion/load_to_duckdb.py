import duckdb

# Connect to persistent DuckDB database file
con = duckdb.connect("data/taxi.duckdb")

print("Loading cleaned parquet into DuckDB...")

con.execute("""
    CREATE OR REPLACE TABLE raw_trips AS
    SELECT * FROM read_parquet('data/processed/trips_cleaned.parquet')
""")

count = con.execute("SELECT COUNT(*) FROM raw_trips").fetchone()[0]
print(f"Loaded {count:,} rows into raw_trips table")

con.close()
print("Done. Database saved at data/taxi.duckdb")