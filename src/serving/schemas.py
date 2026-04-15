"""
schemas.py
----------
Pydantic v2 request and response models for the inference API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime


class AlertLevel(str, Enum):
    CRITICAL = "CRITICAL"    # Failure probability > 0.75, RUL < 48h
    HIGH = "HIGH"            # Failure probability > 0.55, RUL < 96h
    MEDIUM = "MEDIUM"        # Failure probability > 0.35
    LOW = "LOW"              # Normal operation


class SensorReading(BaseModel):
    timestamp: Optional[str] = Field(
        default=None, description="ISO 8601 timestamp"
    )
    op_setting_1: float = Field(description="Operating setting 1")
    op_setting_2: float = Field(description="Operating setting 2")
    op_setting_3: float = Field(default=100.0)
    sensors: Dict[str, float] = Field(
        description="Sensor readings keyed by sensor name (sensor_1 ... sensor_21)"
    )

    @field_validator("sensors")
    @classmethod
    def validate_sensors(cls, v):
        valid_sensors = {f"sensor_{i}" for i in range(1, 22)}
        invalid = set(v.keys()) - valid_sensors
        if invalid:
            raise ValueError(f"Invalid sensor names: {invalid}")
        return v


class PredictRequest(BaseModel):
    asset_id: str = Field(
        description="Equipment identifier (e.g. PUMP-07, COMP-03)"
    )
    readings: List[SensorReading] = Field(
        min_length=1,
        description="Time-ordered list of sensor readings (most recent last). "
                    "Minimum 1, recommended 30 for best accuracy.",
    )
    subset: Optional[str] = Field(
        default="FD001",
        description="Model variant to use. FD001 (default) or FD003.",
    )


class FeatureContribution(BaseModel):
    sensor: str
    shap_value: float
    direction: str   # "increasing" or "decreasing"


class PredictResponse(BaseModel):
    asset_id: str
    failure_probability_96h: float = Field(
        description="Probability of failure within 96 operating hours (0–1)"
    )
    predicted_rul_cycles: float = Field(
        description="Predicted Remaining Useful Life in cycles (≈ hours)"
    )
    alert_level: AlertLevel
    top_contributing_sensors: List[FeatureContribution] = Field(
        description="Top sensors driving this prediction (SHAP-based)"
    )
    recommendation: str
    model_version: str
    inference_time_ms: float
    timestamp: str


class BatchPredictRequest(BaseModel):
    assets: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    n_assets: int
    n_critical: int
    n_high: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class AssetHistoryPoint(BaseModel):
    cycle: int
    anomaly_score: float
    failure_probability: float
    predicted_rul: float


class AssetHistoryResponse(BaseModel):
    asset_id: str
    history: List[AssetHistoryPoint]
    total_cycles_monitored: int
