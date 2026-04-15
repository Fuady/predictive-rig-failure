"""
data_loader.py
--------------
Load, validate, and provide summary statistics for raw C-MAPSS data.
Includes a mapping from C-MAPSS sensor names to real O&G equipment analogs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


RAW_DATA_DIR = Path("data/raw")

COLUMN_NAMES = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors that carry degradation signal (others are near-constant)
INFORMATIVE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_8", "sensor_11", "sensor_12", "sensor_13",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

# Mapping to real O&G equipment for portfolio context
OG_SENSOR_MAP: Dict[str, str] = {
    "sensor_2":  "Mud pump discharge temperature (°C)",
    "sensor_3":  "Compressor outlet temperature (°C)",
    "sensor_4":  "Top drive motor winding temperature (°C)",
    "sensor_7":  "Mud pump discharge pressure (psi)",
    "sensor_8":  "Centrifugal pump flow velocity (m/s)",
    "sensor_11": "Pump bypass flow ratio (%)",
    "sensor_12": "Fuel / torque ratio (dimensionless)",
    "sensor_13": "Gear train vibration proxy (m/s²)",
    "sensor_15": "Bearing housing temperature (°C)",
    "sensor_17": "Shaft speed (RPM)",
    "sensor_20": "Vibration RMS — radial axis (mm/s)",
    "sensor_21": "Vibration RMS — axial axis (mm/s)",
}


def load_dataset(
    subset: str = "FD001",
    split: str = "train",
) -> pd.DataFrame:
    """
    Load a C-MAPSS subset.

    Parameters
    ----------
    subset : str
        One of FD001, FD002, FD003, FD004.
    split : str
        'train' or 'test'.

    Returns
    -------
    pd.DataFrame with named columns and dtypes validated.
    """
    path = RAW_DATA_DIR / f"{split}_{subset}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run: python src/ingestion/download_data.py"
        )

    df = pd.read_csv(path)

    # Handle legacy txt format (space-separated, no header)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=r"\s+", header=None)
        df.dropna(axis=1, how="all", inplace=True)
        df.columns = COLUMN_NAMES[: df.shape[1]]

    # Enforce column names
    if "unit_id" not in df.columns:
        df.columns = COLUMN_NAMES[: df.shape[1]]

    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    return df


def load_rul_labels(subset: str = "FD001") -> pd.Series:
    """Load true RUL values for test set evaluation."""
    path = RAW_DATA_DIR / f"RUL_{subset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"RUL file not found: {path}")
    df = pd.read_csv(path)
    return df.iloc[:, 0]


def load_train_test(
    subset: str = "FD001",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Convenience loader: returns train, test, and true RUL.

    Returns
    -------
    train_df, test_df, rul_series
    """
    train = load_dataset(subset, "train")
    test = load_dataset(subset, "test")
    rul = load_rul_labels(subset)
    return train, test, rul


def validate_data(df: pd.DataFrame) -> Dict:
    """
    Run data quality checks. Returns a validation report dict.
    """
    report = {
        "n_units": df["unit_id"].nunique(),
        "n_rows": len(df),
        "n_sensors": len([c for c in df.columns if c.startswith("sensor_")]),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "sensor_stats": {},
        "issues": [],
    }

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    for col in sensor_cols:
        s = df[col]
        report["sensor_stats"][col] = {
            "mean": round(s.mean(), 4),
            "std": round(s.std(), 4),
            "min": round(s.min(), 4),
            "max": round(s.max(), 4),
            "pct_constant": round((s == s.mode().iloc[0]).mean() * 100, 2),
        }

        # Flag near-constant sensors (std < 0.001)
        if s.std() < 0.001:
            report["issues"].append(f"CONSTANT sensor: {col} (std={s.std():.6f})")

        # Flag sensors with >5% missing
        miss_pct = s.isnull().mean() * 100
        if miss_pct > 5:
            report["issues"].append(
                f"HIGH MISSING: {col} ({miss_pct:.1f}% null)"
            )

    return report


def print_summary(df: pd.DataFrame, subset: str = "FD001") -> None:
    """Print a human-readable data summary."""
    report = validate_data(df)
    print(f"\n{'='*50}")
    print(f"  Dataset: {subset}")
    print(f"{'='*50}")
    print(f"  Units (equipment):  {report['n_units']}")
    print(f"  Total rows:         {report['n_rows']:,}")
    print(f"  Sensors:            {report['n_sensors']}")

    total_missing = sum(report["missing_values"].values())
    print(f"  Missing values:     {total_missing}")

    cycles = df.groupby("unit_id")["cycle"].max()
    print(f"  Lifecycle (cycles): min={cycles.min()}, "
          f"mean={cycles.mean():.0f}, max={cycles.max()}")

    if report["issues"]:
        print(f"\n  Data issues found:")
        for issue in report["issues"]:
            print(f"    ⚠ {issue}")
    else:
        print("\n  No data quality issues found.")

    print(f"\n  O&G sensor mapping (informative sensors):")
    for sensor, description in OG_SENSOR_MAP.items():
        stat = report["sensor_stats"].get(sensor, {})
        mean = stat.get("mean", "N/A")
        std = stat.get("std", "N/A")
        print(f"    {sensor:<12} → {description}")
        print(f"               mean={mean}, std={std}")
