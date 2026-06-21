"""
FastAPI server for NYC Taxi fare prediction.

Provides REST endpoints for real-time fare predictions using a trained model.
"""

from typing import Any, Dict
import os
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder


# Initialize FastAPI app
app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="Predicts taxi fare based on trip features. Built on 8.7M NYC trips.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Load model once at startup
MODEL_PATH: str = "src/ml/artifacts/best_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    artifact: Dict[str, Any] = pickle.load(f)

model: BaseEstimator = artifact["model"]
features: list[str] = artifact["features"]
le: LabelEncoder = artifact["label_encoder"]


class TripInput(BaseModel):
    """Input schema for fare prediction request."""

    trip_distance: float = Field(
        ..., 
        example=2.5,
        description="Trip distance in miles",
        gt=0
    )
    passenger_count: int = Field(
        ..., 
        example=1,
        ge=1, 
        le=6,
        description="Number of passengers"
    )
    pickup_hour: int = Field(
        ...,
        example=8,
        ge=0,
        le=23,
        description="Hour of pickup (0-23, UTC)"
    )
    pickup_dow: int = Field(
        ...,
        example=1,
        ge=0,
        le=6,
        description="Day of week (0=Sunday, 6=Saturday)"
    )
    pickup_month: int = Field(
        ...,
        example=1,
        ge=1,
        le=12,
        description="Month of pickup"
    )
    pickup_location_id: int = Field(
        ...,
        example=161,
        description="NYC taxi zone location ID"
    )
    dropoff_location_id: int = Field(
        ...,
        example=237,
        description="NYC taxi zone location ID"
    )
    payment_type: int = Field(
        ...,
        example=1,
        ge=1,
        le=2,
        description="1=Credit Card, 2=Cash"
    )
    trip_duration_min: float = Field(
        ...,
        example=15.0,
        gt=0,
        description="Trip duration in minutes"
    )
    avg_speed_mph: float = Field(
        ...,
        example=12.0,
        gt=0,
        description="Average speed in miles per hour"
    )
    time_of_day: str = Field(
        ...,
        example="morning_rush",
        description="Time period: 'morning_rush', 'evening_rush', or 'off_peak'"
    )


class PredictionOutput(BaseModel):
    """Output schema for fare prediction response."""

    predicted_fare_usd: float = Field(
        ...,
        description="Predicted fare amount in USD"
    )
    model_name: str = Field(
        ...,
        description="Name of the regression model used"
    )
    confidence_note: str = Field(
        ...,
        description="Model performance metrics note"
    )


class ModelStats(BaseModel):
    """Model performance statistics."""

    training_rows: int
    test_rows: int
    best_model: str
    metrics: Dict[str, float]
    features_used: list[str]


@app.get("/", tags=["Info"])
def root() -> Dict[str, Any]:
    """
    Get API information.

    Returns:
        Dict with API status, model info, and documentation link
    """
    return {
        "status": "running",
        "model": type(model).__name__,
        "features": features,
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Dict indicating service health status
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
def predict(trip: TripInput) -> PredictionOutput:
    """
    Predict taxi fare for given trip features.

    Args:
        trip: Trip details for prediction

    Returns:
        PredictionOutput with predicted fare and model info

    Raises:
        HTTPException: If time_of_day is invalid
    """
    try:
        time_enc = le.transform([trip.time_of_day])[0]
    except ValueError as e:
        valid_times = list(le.classes_)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid time_of_day. Must be one of: {valid_times}"
        ) from e

    # Prepare input array in correct feature order
    input_data = np.array([[
        trip.trip_distance,
        trip.passenger_count,
        trip.pickup_hour,
        trip.pickup_dow,
        trip.pickup_month,
        trip.pickup_location_id,
        trip.dropoff_location_id,
        trip.payment_type,
        trip.trip_duration_min,
        trip.avg_speed_mph,
        time_enc
    ]])

    # Make prediction
    prediction: np.ndarray = model.predict(input_data)
    predicted_fare: float = round(float(prediction[0]), 2)

    return PredictionOutput(
        predicted_fare_usd=predicted_fare,
        model_name=type(model).__name__,
        confidence_note="MAE=$0.94 on 100,000 test trips (R2=0.969)"
    )


@app.get("/stats", response_model=ModelStats, tags=["Model Info"])
def stats() -> Dict[str, Any]:
    """
    Get model performance statistics.

    Returns:
        Dict with model metrics, training info, and features used
    """
    return {
        "training_rows": 400000,
        "test_rows": 100000,
        "best_model": type(model).__name__,
        "metrics": {
            "mae_usd": 0.94,
            "rmse_usd": 3.26,
            "r2_score": 0.9688
        },
        "features_used": features
    }