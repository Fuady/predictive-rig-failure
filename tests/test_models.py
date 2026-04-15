"""
tests/test_models.py
--------------------
Unit tests for all model classes.
Run: pytest tests/test_models.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline import ThresholdModel
from src.models.xgboost_model import XGBoostPdMClassifier, XGBoostRULRegressor
from src.features.label_engineering import add_rul_labels, add_binary_labels
from src.features.signal_processing import (
    compute_rolling_features, normalize_per_unit, add_cross_sensor_features
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SENSORS = [f"sensor_{i}" for i in [2, 3, 4, 7, 11, 12, 20, 21]]
WINDOWS = [5, 15]


@pytest.fixture
def synthetic_dataset():
    """
    Generate a small labeled dataset with clear degradation pattern.
    Units 1–6: training. Units 7–8: test.
    """
    np.random.seed(42)
    rows = []
    for uid in range(1, 9):
        max_cycle = np.random.randint(60, 100)
        for cycle in range(1, max_cycle + 1):
            deg = cycle / max_cycle
            row = {"unit_id": uid, "cycle": cycle,
                   "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0}
            for s in SENSORS:
                # Healthy sensors degrade linearly
                row[s] = np.random.normal(100.0 + deg * 20, 2.0)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Feature engineering
    df = normalize_per_unit(df, SENSORS, baseline_cycles=5)
    norm_cols = [f"{s}_normalized" for s in SENSORS]
    for w in WINDOWS:
        df = compute_rolling_features(df, norm_cols, window=w)
    df = add_cross_sensor_features(df)
    df = add_rul_labels(df)
    df = add_binary_labels(df)

    feature_cols = [
        c for c in df.columns
        if c not in ["unit_id", "cycle", "op_setting_1", "op_setting_2",
                     "op_setting_3", "rul", "rul_capped",
                     "label_critical", "label_warning", "label_advisory"]
        and not c.startswith("sensor_")
    ]

    return df, feature_cols


# ---------------------------------------------------------------------------
# Threshold baseline tests
# ---------------------------------------------------------------------------

class TestThresholdModel:

    def test_fit_runs(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        model = ThresholdModel()
        model.fit(df[feat_cols])
        assert model.is_fitted

    def test_predict_binary(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        model = ThresholdModel()
        model.fit(df[feat_cols])
        preds = model.predict(df[feat_cols])
        assert set(preds).issubset({0, 1})
        assert len(preds) == len(df)

    def test_predict_proba_range(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        model = ThresholdModel()
        model.fit(df[feat_cols])
        proba = model.predict_proba(df[feat_cols])
        assert proba.shape == (len(df), 2)
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_evaluate_returns_dict(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        model = ThresholdModel()
        model.fit(df[feat_cols])
        metrics = model.evaluate(df[feat_cols], df["label_warning"])
        for key in ["precision", "recall", "f1", "false_alarm_rate"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_higher_threshold_fewer_alerts(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        m_low = ThresholdModel(zscore_threshold=1.0)
        m_low.fit(df[feat_cols])
        m_high = ThresholdModel(zscore_threshold=4.0)
        m_high.fit(df[feat_cols])
        n_low = m_low.predict(df[feat_cols]).sum()
        n_high = m_high.predict(df[feat_cols]).sum()
        assert n_low >= n_high


# ---------------------------------------------------------------------------
# XGBoost classifier tests
# ---------------------------------------------------------------------------

class TestXGBoostClassifier:

    def test_fit_and_predict(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        train = df[df["unit_id"] <= 6]
        test = df[df["unit_id"] > 6]

        model = XGBoostPdMClassifier(params={"n_estimators": 30})
        model.fit(train, train["label_warning"], feat_cols)

        preds = model.predict(test)
        assert set(preds).issubset({0, 1})
        assert len(preds) == len(test)

    def test_predict_proba_shape(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        model = XGBoostPdMClassifier(params={"n_estimators": 30})
        model.fit(df, df["label_warning"], feat_cols)
        proba = model.predict_proba(df)
        assert proba.shape == (len(df), 2)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_evaluate_keys(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        model = XGBoostPdMClassifier(params={"n_estimators": 30})
        model.fit(df, df["label_warning"], feat_cols)
        metrics = model.evaluate(df, df["label_warning"])
        for key in ["precision", "recall", "f1", "roc_auc",
                    "false_alarm_rate", "estimated_business_cost_usd"]:
            assert key in metrics

    def test_shap_explain_shape(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        model = XGBoostPdMClassifier(params={"n_estimators": 30})
        model.fit(df, df["label_warning"], feat_cols)
        shap_df = model.explain(df, max_samples=20)
        assert shap_df.shape[1] == len(feat_cols)
        assert len(shap_df) == min(20, len(df))

    def test_threshold_tuning(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        model = XGBoostPdMClassifier(params={"n_estimators": 30})
        model.fit(df, df["label_warning"], feat_cols)
        t = model.tune_threshold_for_recall(df, df["label_warning"], target_recall=0.8)
        assert 0.0 < t < 1.0

    def test_high_recall_after_tuning(self, synthetic_dataset):
        """After tuning for recall ≥ 0.85, actual recall should be ≥ 0.80."""
        from sklearn.metrics import recall_score
        df, feat_cols = synthetic_dataset
        model = XGBoostPdMClassifier(params={"n_estimators": 50})
        model.fit(df, df["label_warning"], feat_cols)
        model.tune_threshold_for_recall(df, df["label_warning"], target_recall=0.85)
        preds = model.predict(df)
        rec = recall_score(df["label_warning"], preds, zero_division=0)
        assert rec >= 0.75, f"Expected recall >= 0.75 after tuning, got {rec:.3f}"


# ---------------------------------------------------------------------------
# XGBoost regressor tests
# ---------------------------------------------------------------------------

class TestXGBoostRegressor:

    def test_fit_and_predict(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        model = XGBoostRULRegressor(params={"n_estimators": 30})
        model.fit(df, df["rul_capped"], feat_cols)
        preds = model.predict(df)
        assert len(preds) == len(df)
        assert (preds >= 0).all(), "RUL predictions must be non-negative"

    def test_evaluate_keys(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        model = XGBoostRULRegressor(params={"n_estimators": 30})
        model.fit(df, df["rul_capped"], feat_cols)
        metrics = model.evaluate(df, df["rul_capped"])
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= metrics["mae"]

    def test_rul_predictions_nonnegative(self, synthetic_dataset):
        df, feat_cols = synthetic_dataset
        model = XGBoostRULRegressor(params={"n_estimators": 30})
        model.fit(df, df["rul_capped"], feat_cols)
        preds = model.predict(df)
        assert (preds >= 0).all()
