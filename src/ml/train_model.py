import duckdb
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# MLflow experiment setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc_taxi_fare_prediction")

# Load data from DuckDB
print("Loading data from DuckDB...")
con = duckdb.connect("data/taxi.duckdb")

df = con.execute("""
    SELECT
        trip_distance,
        passenger_count,
        pickup_hour,
        pickup_dow,
        pickup_month,
        pickup_location_id,
        dropoff_location_id,
        payment_type,
        trip_duration_min,
        avg_speed_mph,
        time_of_day,
        fare_amount
    FROM stg_trips
    WHERE avg_speed_mph IS NOT NULL
      AND avg_speed_mph < 100
    LIMIT 500000
""").df()

con.close()
print(f"Loaded {len(df):,} rows for training")

# Encode categorical column
le = LabelEncoder()
df["time_of_day_enc"] = le.fit_transform(df["time_of_day"])

# Define features and target
FEATURES = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_dow",
    "pickup_month",
    "pickup_location_id",
    "dropoff_location_id",
    "payment_type",
    "trip_duration_min",
    "avg_speed_mph",
    "time_of_day_enc",
]

X = df[FEATURES]
y = df["fare_amount"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

# Models to compare
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
}

best_model = None
best_mae = float("inf")
best_run_id = None

print("\nTraining models...")

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        # Log to MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("features", FEATURES)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        print(f"\n{name}:")
        print(f"  MAE  = ${mae:.2f}")
        print(f"  RMSE = ${rmse:.2f}")
        print(f"  R2   = {r2:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_run_id = mlflow.active_run().info.run_id
            best_name = name

print(f"\nBest model: {best_name} (MAE=${best_mae:.2f})")

# Save best model
os.makedirs("src/ml/artifacts", exist_ok=True)
import pickle
with open("src/ml/artifacts/best_model.pkl", "wb") as f:
    pickle.dump({"model": best_model, "features": FEATURES, "label_encoder": le}, f)

print("Best model saved to src/ml/artifacts/best_model.pkl")
print(f"MLflow run ID: {best_run_id}")
