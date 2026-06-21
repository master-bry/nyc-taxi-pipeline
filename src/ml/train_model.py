"""
Machine Learning model training for NYC Taxi fare prediction.

Trains and compares multiple regression models using MLflow for experiment tracking.
"""

from typing import Dict, Tuple, Any
import os
import pickle

import duckdb
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator


# Feature list for consistency
FEATURES: list[str] = [
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


def load_training_data(
    db_path: str = "data/taxi.duckdb",
    limit: int = 500000
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Load and prepare training data from DuckDB.

    Args:
        db_path: Path to DuckDB database
        limit: Maximum number of rows to load

    Returns:
        Tuple of (processed dataframe, fitted label encoder)

    Raises:
        FileNotFoundError: If database does not exist
        Exception: If query fails
    """
    print(f"Loading data from {db_path}...")
    con = duckdb.connect(db_path, read_only=True)

    df = con.execute(f"""
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
        LIMIT {limit}
    """).df()

    con.close()
    print(f"Loaded {len(df):,} rows for training")

    # Encode categorical column
    le = LabelEncoder()
    df["time_of_day_enc"] = le.fit_transform(df["time_of_day"])

    return df, le


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Args:
        df: Input dataframe
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df[FEATURES]
    y = df["fare_amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    experiment_name: str = "nyc_taxi_fare_prediction"
) -> Tuple[BaseEstimator, float, str]:
    """
    Train and evaluate multiple regression models.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        experiment_name: MLflow experiment name

    Returns:
        Tuple of (best_model, best_mae, best_run_id)
    """
    # MLflow experiment setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    # Models to compare
    models: Dict[str, BaseEstimator] = {
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

    best_model: BaseEstimator | None = None
    best_mae: float = float("inf")
    best_run_id: str = ""
    best_name: str = ""

    print("\nTraining models...")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Train
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            # Log to MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_param("features", str(FEATURES))
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

    return best_model, best_mae, best_run_id


def save_model(
    model: BaseEstimator,
    features: list[str],
    label_encoder: LabelEncoder,
    artifact_dir: str = "src/ml/artifacts",
    filename: str = "best_model.pkl"
) -> str:
    """
    Save trained model and metadata to disk.

    Args:
        model: Trained model
        features: List of feature names
        label_encoder: Fitted label encoder
        artifact_dir: Directory to save artifacts
        filename: Name of pickle file

    Returns:
        Full path to saved model file
    """
    os.makedirs(artifact_dir, exist_ok=True)

    artifact_path = os.path.join(artifact_dir, filename)
    with open(artifact_path, "wb") as f:
        pickle.dump(
            {"model": model, "features": features, "label_encoder": label_encoder},
            f
        )

    print(f"Model saved to {artifact_path}")
    return artifact_path


def main() -> None:
    """Main training pipeline."""
    # Load and prepare data
    df, le = load_training_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train models
    best_model, best_mae, best_run_id = train_models(X_train, X_test, y_train, y_test)

    # Save model
    save_model(best_model, FEATURES, le)

    print(f"\nMLflow run ID: {best_run_id}")


if __name__ == "__main__":
    main()
