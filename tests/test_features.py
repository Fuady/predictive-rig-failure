"""
tests/test_features.py
----------------------
Unit tests for feature engineering pipeline.
Run: pytest tests/test_features.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.signal_processing import (
    compute_rms,
    compute_crest_factor,
    compute_kurtosis,
    compute_rolling_features,
    add_cross_sensor_features,
    normalize_per_unit,
)
from src.features.label_engineering import (
    add_rul_labels,
    add_binary_labels,
    add_lifecycle_features,
    ALERT_HORIZONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Create a minimal synthetic sensor DataFrame for testing."""
    np.random.seed(42)
    rows = []
    for unit_id in range(1, 4):
        for cycle in range(1, 51):
            row = {
                "unit_id": unit_id,
                "cycle": cycle,
                "op_setting_1": 0.0023,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
            }
            for s in range(1, 22):
                row[f"sensor_{s}"] = np.random.normal(100.0, 5.0)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Signal processing tests
# ---------------------------------------------------------------------------

class TestSignalProcessing:

    def test_rms_positive_signal(self):
        signal = np.array([1.0, -1.0, 1.0, -1.0])
        assert compute_rms(signal) == pytest.approx(1.0, rel=1e-6)

    def test_rms_zeros(self):
        signal = np.zeros(10)
        assert compute_rms(signal) == 0.0

    def test_rms_constant(self):
        signal = np.full(10, 3.0)
        assert compute_rms(signal) == pytest.approx(3.0, rel=1e-6)

    def test_crest_factor_sinusoid(self):
        t = np.linspace(0, 2 * np.pi, 1000)
        signal = np.sin(t)
        cf = compute_crest_factor(signal)
        # For a sinusoid, crest factor = sqrt(2) ≈ 1.414
        assert cf == pytest.approx(np.sqrt(2), rel=0.01)

    def test_crest_factor_zero_signal(self):
        assert compute_crest_factor(np.zeros(10)) == 0.0

    def test_crest_factor_spike(self):
        # Impulsive signal (bearing fault analog) → high crest factor
        signal = np.zeros(100)
        signal[50] = 10.0
        cf = compute_crest_factor(signal)
        assert cf > 5.0, "Impulsive signal should have high crest factor"

    def test_kurtosis_normal_distribution(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 10000)
        k = compute_kurtosis(signal)
        # Normal distribution kurtosis (Fisher=False) ≈ 3.0
        assert k == pytest.approx(3.0, abs=0.2)

    def test_kurtosis_impulsive(self):
        # Signal with spikes has kurtosis >> 3
        signal = np.zeros(1000)
        signal[[100, 300, 700]] = 20.0
        k = compute_kurtosis(signal)
        assert k > 10.0, "Impulsive signal should have high kurtosis"

    def test_rolling_features_shape(self, sample_df):
        sensors = ["sensor_2", "sensor_3"]
        result = compute_rolling_features(sample_df, sensors, window=5)
        # Should add multiple new columns
        assert len(result.columns) > len(sample_df.columns)
        assert len(result) == len(sample_df)

    def test_rolling_features_no_leakage(self, sample_df):
        """Rolling features should not cross unit boundaries."""
        sensors = ["sensor_2"]
        result = compute_rolling_features(sample_df, sensors, window=5)
        # First cycle of each unit should have no leakage from previous unit
        for uid in result["unit_id"].unique():
            unit_rows = result[result["unit_id"] == uid]
            first_mean = unit_rows.iloc[0]["sensor_2_w5_mean"]
            # Should equal just the first reading (window of 1)
            first_raw = unit_rows.iloc[0]["sensor_2"]
            assert first_mean == pytest.approx(first_raw, rel=1e-5)

    def test_cross_sensor_features_added(self, sample_df):
        # Add required sensor columns
        result = add_cross_sensor_features(sample_df)
        assert "feat_temp_press_ratio" in result.columns
        assert "feat_temp_diff" in result.columns
        assert "feat_vibration_total" in result.columns

    def test_normalize_per_unit_zero_mean(self, sample_df):
        sensors = ["sensor_2"]
        result = normalize_per_unit(sample_df, sensors, baseline_cycles=5)
        assert "sensor_2_normalized" in result.columns
        # Early cycles (baseline) should have mean ~ 0 after normalization
        for uid in result["unit_id"].unique():
            early = result[
                (result["unit_id"] == uid) & (result["cycle"] <= 5)
            ]["sensor_2_normalized"]
            assert early.mean() == pytest.approx(0.0, abs=0.5)


# ---------------------------------------------------------------------------
# Label engineering tests
# ---------------------------------------------------------------------------

class TestLabelEngineering:

    def test_rul_labels_range(self, sample_df):
        result = add_rul_labels(sample_df)
        assert "rul" in result.columns
        assert (result["rul"] >= 0).all()

    def test_rul_zero_at_last_cycle(self, sample_df):
        result = add_rul_labels(sample_df)
        for uid in result["unit_id"].unique():
            unit = result[result["unit_id"] == uid]
            last_rul = unit.sort_values("cycle").iloc[-1]["rul"]
            assert last_rul == 0

    def test_rul_monotonically_decreasing_per_unit(self, sample_df):
        result = add_rul_labels(sample_df)
        for uid in result["unit_id"].unique():
            unit = result[result["unit_id"] == uid].sort_values("cycle")
            rul_vals = unit["rul"].values
            # RUL should decrease by 1 per cycle
            diffs = np.diff(rul_vals)
            assert (diffs == -1).all()

    def test_rul_capped(self, sample_df):
        result = add_rul_labels(sample_df, max_rul=30)
        assert result["rul_capped"].max() <= 30
        assert result["rul_capped"].min() >= 0

    def test_binary_labels_require_rul(self, sample_df):
        with pytest.raises(ValueError, match="rul"):
            add_binary_labels(sample_df)

    def test_binary_labels_values(self, sample_df):
        df = add_rul_labels(sample_df)
        df = add_binary_labels(df)
        for col in [f"label_{name}" for name in ALERT_HORIZONS]:
            assert col in df.columns
            assert set(df[col].unique()).issubset({0, 1})

    def test_binary_label_consistency(self, sample_df):
        """Critical label (short horizon) must be subset of warning label."""
        df = add_rul_labels(sample_df)
        df = add_binary_labels(df)
        # Every critical=1 row must also be warning=1
        critical_rows = df[df["label_critical"] == 1]
        assert (critical_rows["label_warning"] == 1).all()

    def test_lifecycle_features_range(self, sample_df):
        result = add_lifecycle_features(sample_df)
        assert "cycle_normalized" in result.columns
        assert result["cycle_normalized"].between(0, 1).all()
        assert (result["is_early_life"].isin([0, 1])).all()
        assert (result["is_late_life"].isin([0, 1])).all()
