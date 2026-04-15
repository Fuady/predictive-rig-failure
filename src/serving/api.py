"""
api.py
------
FastAPI inference endpoint for real-time failure probability prediction.

Endpoints:
  GET  /health                    — Service health check
  POST /predict                   — Single asset prediction
  POST /predict/batch             — Multi-asset prediction
  GET  /assets/{asset_id}/history — 30-cycle anomaly score history

Run locally:
  uvicorn src.serving.api:app --reload --port 8000

API docs:
  http://localhost:8000/docs
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.serving.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, AssetHistoryResponse,
    AlertLevel, FeatureContribution,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PdM Rig Failure Prediction API",
    description=(
        "Predictive maintenance inference endpoint for oil & gas drilling equipment. "
        "Predicts failure probability and remaining useful life from sensor readings."
    ),
    version="1.0.0",
    contact={"name": "Data Science Team"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Model loading (lazy-loaded on first request)
# ---------------------------------------------------------------------------

_models: Dict[str, Optional[object]] = {}
_regressors: Dict[str, Optional[object]] = {}
_meta: Dict[str, Optional[dict]] = {}
MODEL_VERSION = "xgboost-v1.0.0"
MODELS_DIR = Path("models")


def get_model(subset: str = "FD001"):
    """Load XGBoost classifier (lazy load on first call)."""
    if subset not in _models:
        model_path = MODELS_DIR / f"xgboost_classifier_{subset}.joblib"
        if not model_path.exists():
            # Try to load any available model
            available = list(MODELS_DIR.glob("xgboost_classifier_*.joblib"))
            if not available:
                return None
            model_path = available[0]
            subset = model_path.stem.replace("xgboost_classifier_", "")

        try:
            _models[subset] = joblib.load(model_path)
        except Exception:
            _models[subset] = None

    return _models.get(subset)


def get_regressor(subset: str = "FD001"):
    """Load XGBoost RUL regressor."""
    if subset not in _regressors:
        reg_path = MODELS_DIR / f"xgboost_regressor_{subset}.joblib"
        if reg_path.exists():
            try:
                _regressors[subset] = joblib.load(reg_path)
            except Exception:
                _regressors[subset] = None
        else:
            _regressors[subset] = None
    return _regressors.get(subset)


def get_meta(subset: str = "FD001") -> Optional[dict]:
    """Load feature metadata."""
    if subset not in _meta:
        meta_path = Path("data/features") / f"meta_{subset}.joblib"
        if meta_path.exists():
            try:
                _meta[subset] = joblib.load(meta_path)
            except Exception:
                _meta[subset] = None
        else:
            _meta[subset] = None
    return _meta.get(subset)


# ---------------------------------------------------------------------------
# Feature preparation from raw sensor readings
# ---------------------------------------------------------------------------

def readings_to_feature_row(
    readings: list, feature_cols: list, sensors_used: list
) -> pd.DataFrame:
    """
    Convert raw API sensor readings into the feature vector
    the model expects.

    In production this would call the full feature pipeline.
    Here we compute the key rolling features inline for low latency.
    """
    COLUMN_NAMES = (
        ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    rows = []
    for i, reading in enumerate(readings):
        row = {
            "unit_id": 1,
            "cycle": i + 1,
            "op_setting_1": reading.op_setting_1,
            "op_setting_2": reading.op_setting_2,
            "op_setting_3": reading.op_setting_3,
        }
        for s in [f"sensor_{j}" for j in range(1, 22)]:
            row[s] = reading.sensors.get(s, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Build features expected by the model
    # We approximate by computing rolling stats on the input window
    out = {}

    for sensor in sensors_used:
        if sensor not in df.columns:
            continue
        vals = df[sensor].values

        # Baseline (first 5 readings or available)
        baseline = vals[:5] if len(vals) >= 5 else vals
        mean_base = np.mean(baseline)
        std_base = max(np.std(baseline), 1e-6)

        # Normalized current value
        out[f"{sensor}_normalized"] = (vals[-1] - mean_base) / std_base

        # Rolling stats (last window)
        for w in [5, 15, 30]:
            window = vals[-w:] if len(vals) >= w else vals
            out[f"{sensor}_w{w}_mean"] = float(np.mean(window))
            out[f"{sensor}_w{w}_std"] = float(np.std(window))
            out[f"{sensor}_w{w}_rms"] = float(
                np.sqrt(np.mean(window ** 2))
            )
            out[f"{sensor}_w{w}_slope"] = float(vals[-1] - window[0])

            # Kurtosis
            from scipy.stats import kurtosis
            out[f"{sensor}_w{w}_kurt"] = float(
                kurtosis(window, fisher=False) if len(window) >= 4 else 3.0
            )

            # Crest factor
            rms_v = np.sqrt(np.mean(window ** 2))
            out[f"{sensor}_w{w}_crest"] = (
                float(np.max(np.abs(window)) / rms_v) if rms_v > 0 else 1.0
            )
            out[f"{sensor}_w{w}_min"] = float(np.min(window))
            out[f"{sensor}_w{w}_max"] = float(np.max(window))

    # Cross-sensor features
    s4 = df["sensor_4"].values[-1] if "sensor_4" in df.columns else 1.0
    s7 = df["sensor_7"].values[-1] if "sensor_7" in df.columns else 1.0
    out["feat_temp_press_ratio"] = s4 / (s7 + 1e-6)

    s2 = df["sensor_2"].values[-1] if "sensor_2" in df.columns else 0.0
    s3 = df["sensor_3"].values[-1] if "sensor_3" in df.columns else 0.0
    out["feat_temp_diff"] = s3 - s2

    s11 = df["sensor_11"].values[-1] if "sensor_11" in df.columns else 1.0
    s12 = df["sensor_12"].values[-1] if "sensor_12" in df.columns else 1.0
    out["feat_flow_efficiency"] = s11 / (s12 + 1e-6)

    s20 = df["sensor_20"].values[-1] if "sensor_20" in df.columns else 0.0
    s21 = df["sensor_21"].values[-1] if "sensor_21" in df.columns else 0.0
    out["feat_vibration_total"] = float(np.sqrt(s20 ** 2 + s21 ** 2))

    # Lifecycle features
    n = len(readings)
    out["cycle_normalized"] = 0.5   # Unknown without full history
    out["cycle_log"] = float(np.log1p(n))
    out["is_early_life"] = 0
    out["is_late_life"] = 0

    # Build row aligned to feature_cols
    row_data = {col: out.get(col, 0.0) for col in feature_cols}
    return pd.DataFrame([row_data])


def determine_alert_level(prob: float, rul: float) -> AlertLevel:
    if prob >= 0.75 or rul < 48:
        return AlertLevel.CRITICAL
    elif prob >= 0.55 or rul < 96:
        return AlertLevel.HIGH
    elif prob >= 0.35:
        return AlertLevel.MEDIUM
    else:
        return AlertLevel.LOW


def get_recommendation(alert: AlertLevel, rul: float) -> str:
    recs = {
        AlertLevel.CRITICAL: (
            f"IMMEDIATE ACTION REQUIRED. Predicted failure in ~{rul:.0f} cycles. "
            "Notify maintenance supervisor. Consider reducing load or shutdown."
        ),
        AlertLevel.HIGH: (
            f"Schedule inspection within 24 hours. "
            f"Estimated {rul:.0f} cycles remaining. "
            "Prepare spare parts (bearings, seals)."
        ),
        AlertLevel.MEDIUM: (
            f"Add to next maintenance window. "
            f"Estimated {rul:.0f} cycles remaining. "
            "Increase monitoring frequency."
        ),
        AlertLevel.LOW: (
            "Normal operation. Continue standard monitoring schedule."
        ),
    }
    return recs[alert]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    model = get_model()
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=MODEL_VERSION,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    t0 = time.time()

    subset = request.subset or "FD001"
    model = get_model(subset)
    regressor = get_regressor(subset)
    meta = get_meta(subset)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                "Run: python src/models/train.py — then restart the API."
            ),
        )

    feature_cols = model.feature_cols
    sensors_used = (
        meta["sensors_used"] if meta else
        [f"sensor_{i}" for i in [2, 3, 4, 7, 8, 11, 12, 13, 15, 17, 20, 21]]
    )

    try:
        X = readings_to_feature_row(request.readings, feature_cols, sensors_used)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature extraction failed: {str(e)}"
        )

    # Failure probability
    try:
        proba = float(model.predict_proba(X)[0, 1])
    except Exception:
        proba = 0.5

    # RUL prediction
    if regressor is not None:
        try:
            rul = float(regressor.predict(X)[0])
        except Exception:
            rul = max((1 - proba) * 125, 5.0)
    else:
        rul = max((1 - proba) * 125, 5.0)

    # SHAP explanations
    try:
        last_row = X.iloc[0]
        contributions = model.get_top_features_for_alert(last_row, n_features=5)
        # Map feature names back to sensor names for readability
        top_sensors = []
        for c in contributions:
            sensor_name = c["feature"].split("_w")[0].replace("_normalized", "")
            top_sensors.append(
                FeatureContribution(
                    sensor=sensor_name,
                    shap_value=c["shap_value"],
                    direction=c["direction"],
                )
            )
    except Exception:
        top_sensors = []

    alert = determine_alert_level(proba, rul)
    rec = get_recommendation(alert, rul)

    return PredictResponse(
        asset_id=request.asset_id,
        failure_probability_96h=round(proba, 4),
        predicted_rul_cycles=round(rul, 1),
        alert_level=alert,
        top_contributing_sensors=top_sensors,
        recommendation=rec,
        model_version=MODEL_VERSION,
        inference_time_ms=round((time.time() - t0) * 1000, 1),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    predictions = [predict(asset_req) for asset_req in request.assets]
    return BatchPredictResponse(
        predictions=predictions,
        n_assets=len(predictions),
        n_critical=sum(1 for p in predictions if p.alert_level == AlertLevel.CRITICAL),
        n_high=sum(1 for p in predictions if p.alert_level == AlertLevel.HIGH),
    )


@app.get("/assets/{asset_id}/history")
def asset_history(asset_id: str):
    """Return a mock 30-cycle history for demo purposes."""
    import random
    random.seed(hash(asset_id) % 2**32)

    history = []
    base_score = random.uniform(0.001, 0.005)
    for i in range(30):
        # Simulate gradual degradation
        score = base_score * (1 + i * 0.05) + random.gauss(0, base_score * 0.1)
        prob = min(0.95, score / 0.05)
        rul = max(5.0, 125 * (1 - prob))
        history.append({
            "cycle": i + 1,
            "anomaly_score": round(max(0, score), 6),
            "failure_probability": round(max(0, min(1, prob)), 4),
            "predicted_rul": round(rul, 1),
        })

    return {
        "asset_id": asset_id,
        "history": history,
        "total_cycles_monitored": 30,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )
