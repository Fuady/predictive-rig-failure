"""
signal_processing.py
--------------------
Compute time-domain and frequency-domain features from sensor time series.

Features mirror what would be computed on real rig sensor data:
  - Rolling statistics (mean, std, RMS)
  - Kurtosis (sensitive to impulsive spikes from bearing defects)
  - Crest factor (peak / RMS — early fault indicator)
  - FFT-based spectral energy
  - Rate of change (first derivative)
  - Cross-sensor interaction features
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew


INFORMATIVE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_8", "sensor_11", "sensor_12", "sensor_13",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

WINDOW_SIZES = [5, 15, 30]   # rolling window in cycles


# ---------------------------------------------------------------------------
# Time-domain features
# ---------------------------------------------------------------------------

def compute_rms(series: np.ndarray) -> float:
    """Root Mean Square — sensitive to overall vibration energy."""
    return float(np.sqrt(np.mean(series ** 2)))


def compute_crest_factor(series: np.ndarray) -> float:
    """
    Crest Factor = peak / RMS.
    Rises sharply when early bearing pitting creates impulsive spikes.
    Healthy equipment: CF ~ 1.4 (sinusoidal)
    Early fault:      CF ~ 3–5
    Advanced fault:   CF > 6
    """
    rms = compute_rms(series)
    if rms == 0:
        return 0.0
    return float(np.max(np.abs(series)) / rms)


def compute_kurtosis(series: np.ndarray) -> float:
    """
    Statistical kurtosis of sensor readings.
    Healthy: kurtosis ~ 3 (normal distribution)
    Fault:   kurtosis > 6 (impulsive content)
    """
    if len(series) < 4:
        return 0.0
    return float(kurtosis(series, fisher=False))


def compute_rolling_features(
    df: pd.DataFrame,
    sensors: List[str],
    window: int,
    group_col: str = "unit_id",
) -> pd.DataFrame:
    """
    Compute rolling statistics for each sensor over a given window.

    For each sensor, computes:
      - mean, std, min, max (within window)
      - RMS
      - kurtosis
      - crest factor
      - rate of change (slope over window)

    Parameters
    ----------
    df       : DataFrame sorted by unit_id, cycle
    sensors  : list of sensor column names
    window   : rolling window size in cycles
    group_col: column used to group by equipment unit

    Returns
    -------
    DataFrame with new feature columns added (original columns preserved)
    """
    result = df.copy()
    result = result.sort_values([group_col, "cycle"]).reset_index(drop=True)

    for sensor in sensors:
        col = df[sensor]
        prefix = f"{sensor}_w{window}"

        grouped = result.groupby(group_col)[sensor]

        result[f"{prefix}_mean"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        result[f"{prefix}_std"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
        result[f"{prefix}_min"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        result[f"{prefix}_max"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        result[f"{prefix}_rms"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).apply(compute_rms, raw=True)
        )
        result[f"{prefix}_kurt"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda v: kurtosis(v, fisher=False) if len(v) >= 4 else 3.0,
                raw=True,
            )
        )
        result[f"{prefix}_crest"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                compute_crest_factor, raw=True
            )
        )
        # Rate of change: difference between current and window-start value
        result[f"{prefix}_slope"] = grouped.transform(
            lambda x: x.diff(window).fillna(0)
        )

    return result


# ---------------------------------------------------------------------------
# Frequency-domain features
# ---------------------------------------------------------------------------

def compute_fft_features(
    series: np.ndarray,
    sampling_rate: float = 1.0,
    n_bands: int = 5,
) -> dict:
    """
    Compute FFT-based spectral features from a sensor window.

    In real O&G applications:
    - Vibration sensors sample at 5–20 kHz
    - Bearing defect frequencies (BPFI, BPFO) are computed from
      shaft speed and bearing geometry
    - Here we use normalized frequency bands as an approximation

    Parameters
    ----------
    series        : sensor reading array
    sampling_rate : Hz (1.0 for cycle-based data)
    n_bands       : number of equal frequency bands

    Returns
    -------
    dict with spectral band energies and dominant frequency
    """
    if len(series) < 4:
        return {f"fft_band_{i}": 0.0 for i in range(n_bands)} | {"fft_dom_freq": 0.0}

    # Remove mean (detrend)
    detrended = series - np.mean(series)

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(len(detrended))
    windowed = detrended * window

    # FFT
    fft_vals = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sampling_rate)

    features = {}

    # Energy in each frequency band
    band_size = len(fft_vals) // n_bands
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size
        band_energy = float(np.sum(fft_vals[start:end] ** 2))
        features[f"fft_band_{i}"] = band_energy

    # Dominant frequency
    if len(fft_vals) > 0:
        dom_idx = np.argmax(fft_vals)
        features["fft_dom_freq"] = float(freqs[dom_idx]) if len(freqs) > dom_idx else 0.0
    else:
        features["fft_dom_freq"] = 0.0

    # Spectral entropy (measure of signal complexity)
    psd = fft_vals ** 2
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spectral_entropy = -float(np.sum(psd_norm * np.log(psd_norm + 1e-10)))
    features["spectral_entropy"] = spectral_entropy

    return features


def add_fft_features(
    df: pd.DataFrame,
    sensors: List[str],
    window: int = 30,
    group_col: str = "unit_id",
) -> pd.DataFrame:
    """
    Apply FFT feature extraction over rolling windows for each sensor.
    """
    result = df.copy()
    result = result.sort_values([group_col, "cycle"]).reset_index(drop=True)

    for sensor in sensors[:4]:   # FFT on top-4 sensors (performance balance)
        fft_rows = []

        for _, group in result.groupby(group_col):
            values = group[sensor].values
            rows = []
            for i in range(len(values)):
                start = max(0, i - window + 1)
                window_vals = values[start : i + 1]
                feats = compute_fft_features(window_vals)
                rows.append(feats)
            fft_rows.extend(rows)

        fft_df = pd.DataFrame(fft_rows)
        fft_df.columns = [f"{sensor}_{col}" for col in fft_df.columns]

        for col in fft_df.columns:
            result[col] = fft_df[col].values

    return result


# ---------------------------------------------------------------------------
# Cross-sensor features (domain-informed)
# ---------------------------------------------------------------------------

def add_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between sensors that have physical meaning
    in rotating equipment diagnostics.

    In real rig context:
    - Temp rise WITH pressure drop → seal degradation
    - Vibration rise WITH flow drop → impeller wear
    - Current rise WITH speed stable → mechanical binding
    """
    result = df.copy()

    if "sensor_4" in df.columns and "sensor_7" in df.columns:
        # Temperature rise + pressure drop ratio (indicator of degradation)
        result["feat_temp_press_ratio"] = df["sensor_4"] / (df["sensor_7"] + 1e-6)

    if "sensor_2" in df.columns and "sensor_3" in df.columns:
        # Temperature differential across stages
        result["feat_temp_diff"] = df["sensor_3"] - df["sensor_2"]

    if "sensor_11" in df.columns and "sensor_12" in df.columns:
        # Flow efficiency proxy
        result["feat_flow_efficiency"] = df["sensor_11"] / (df["sensor_12"] + 1e-6)

    if "sensor_20" in df.columns and "sensor_21" in df.columns:
        # Combined vibration magnitude (vector sum)
        result["feat_vibration_total"] = np.sqrt(
            df["sensor_20"] ** 2 + df["sensor_21"] ** 2
        )

    return result


# ---------------------------------------------------------------------------
# Normalization per unit (z-score relative to unit's own baseline)
# ---------------------------------------------------------------------------

def normalize_per_unit(
    df: pd.DataFrame,
    sensors: List[str],
    group_col: str = "unit_id",
    baseline_cycles: int = 20,
) -> pd.DataFrame:
    """
    Normalize each sensor relative to each unit's own baseline.
    This removes unit-to-unit manufacturing variance, leaving only
    the degradation trend — which is what we want to model.

    baseline_cycles: number of early cycles to compute baseline stats from.
    """
    result = df.copy()

    for sensor in sensors:
        baseline_stats = (
            df[df["cycle"] <= baseline_cycles]
            .groupby(group_col)[sensor]
            .agg(["mean", "std"])
            .rename(columns={"mean": f"{sensor}_baseline_mean",
                              "std": f"{sensor}_baseline_std"})
        )
        result = result.join(baseline_stats, on=group_col)

        std_col = f"{sensor}_baseline_std"
        mean_col = f"{sensor}_baseline_mean"

        result[f"{sensor}_normalized"] = (
            (result[sensor] - result[mean_col]) /
            (result[std_col].clip(lower=1e-6))
        )
        result.drop(columns=[mean_col, std_col], inplace=True)

    return result
