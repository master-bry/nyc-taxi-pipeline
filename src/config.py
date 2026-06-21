"""
Configuration management for NYC Taxi Pipeline.

Supports environment-based settings (dev/prod) with Pydantic validation.
Load settings via: from src.config import get_settings
"""

import os
from typing import Literal
from pydantic import Field, BaseSettings


class DatabaseSettings(BaseSettings):
    """Database connection configuration."""

    duckdb_path: str = Field(default="data/taxi.duckdb", description="Path to DuckDB database")
    read_only: bool = Field(default=True, description="Open database in read-only mode")

    class Config:
        env_prefix = "DB_"


class MLSettings(BaseSettings):
    """Machine Learning model configuration."""

    model_path: str = Field(
        default="src/ml/artifacts/best_model.pkl",
        description="Path to trained model artifact"
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    mlflow_experiment: str = Field(default="nyc_taxi", description="MLflow experiment name")

    class Config:
        env_prefix = "ML_"


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    workers: int = Field(default=4, description="Number of worker processes")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )

    class Config:
        env_prefix = "API_"


class DashboardSettings(BaseSettings):
    """Streamlit dashboard configuration."""

    page_title: str = Field(default="NYC Taxi Analytics", description="Dashboard page title")
    theme: Literal["light", "dark"] = Field(default="dark", description="Dashboard theme")
    layout: Literal["centered", "wide"] = Field(default="wide", description="Page layout")
    cache_ttl: int = Field(default=3600, description="Cache time-to-live in seconds")

    class Config:
        env_prefix = "DASHBOARD_"


class DataSettings(BaseSettings):
    """Data paths and sources configuration."""

    raw_data_path: str = Field(default="data/raw", description="Raw data directory")
    processed_data_path: str = Field(default="data/processed", description="Processed data directory")
    exports_path: str = Field(default="exports", description="Exports directory")
    logs_path: str = Field(default="logs", description="Logs directory")

    class Config:
        env_prefix = "DATA_"


class Settings(BaseSettings):
    """Main application settings."""

    environment: Literal["development", "production"] = Field(
        default="development", description="Deployment environment"
    )
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    api: APISettings = Field(default_factory=APISettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    data: DataSettings = Field(default_factory=DataSettings)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings() -> Settings:
    """
    Get application settings.

    Returns:
        Settings: Validated application configuration
    """
    return Settings()
