from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="Predicts taxi fare based on trip features. Built on 8.7M NYC trips.",
    version="1.0.0"
)

# Load model once at startup
MODEL_PATH = "src/ml/artifacts/best_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
features = artifact["features"]
le = artifact["label_encoder"]


class TripInput(BaseModel):
    trip_distance: float = Field(..., example=2.5, description="Distance in miles")
    passenger_count: int = Field(..., example=1, ge=1, le=6)
    pickup_hour: int = Field(..., example=8, ge=0, le=23)
    pickup_dow: int = Field(..., example=1, ge=0, le=6, description="0=Sunday")
    pickup_month: int = Field(..., example=1, ge=1, le=12)
    pickup_location_id: int = Field(..., example=161)
    dropoff_location_id: int = Field(..., example=237)
    payment_type: int = Field(..., example=1, description="1=Card, 2=Cash")
    trip_duration_min: float = Field(..., example=15.0)
    avg_speed_mph: float = Field(..., example=12.0)
    time_of_day: str = Field(..., example="morning_rush",
                             description="morning_rush, evening_rush, or off_peak")


class PredictionOutput(BaseModel):
    predicted_fare_usd: float
    model_name: str
    confidence_note: str


@app.get("/")
def root():
    return {
        "status": "running",
        "model": type(model).__name__,
        "features": features,
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(trip: TripInput):
    try:
        time_enc = le.transform([trip.time_of_day])[0]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid time_of_day. Must be one of: {list(le.classes_)}"
        )

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

    prediction = model.predict(input_data)[0]
    prediction = round(float(prediction), 2)

    return PredictionOutput(
        predicted_fare_usd=prediction,
        model_name=type(model).__name__,
        confidence_note="MAE=$0.94 on 100,000 test trips (R2=0.969)"
    )


@app.get("/stats")
def stats():
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