"""
label_engineering.py
--------------------
Create Remaining Useful Life (RUL) labels and binary failure labels.

In real O&G operations, labels come from CMMS (SAP PM / IBM Maximo):
- Each failure event creates a work order with timestamp and failure code
- We compute RUL by working backwards from each failure event
- Binary label: "will this asset fail within N hours?"

For NASA C-MAPSS:
- Training data runs to failure (RUL = 0 at last cycle)
- Test data is truncated (true RUL provided separately)
"""

import pandas as pd
import numpy as np
from typing import Optional


# Maximum RUL cap — beyond this, equipment is "healthy" and
# the exact distance to failure is not useful for scheduling.
# 168 cycles = 1 week of operation (1 cycle ≈ 1 operating hour)
MAX_RUL = 125   # Tuned for C-MAPSS; in production use 168 (1 week)

# Alert horizons (in cycles / hours)
ALERT_HORIZONS = {
    "critical": 48,    # Must act now
    "warning": 96,     # Schedule within 4 days
    "advisory": 168,   # Plan for next maintenance window
}


def add_rul_labels(
    df: pd.DataFrame,
    group_col: str = "unit_id",
    max_rul: int = MAX_RUL,
) -> pd.DataFrame:
    """
    Add Remaining Useful Life labels to training data.

    For each unit:
      1. Find max cycle (= failure cycle, since training runs to failure)
      2. RUL at each cycle = max_cycle - current_cycle
      3. Cap RUL at max_rul (equipment is 'healthy' beyond this)

    Parameters
    ----------
    df      : DataFrame with unit_id and cycle columns
    max_rul : cap value for RUL (cycles)

    Returns
    -------
    DataFrame with added columns: rul, rul_capped
    """
    result = df.copy()

    # Maximum cycle per unit = failure cycle
    max_cycles = df.groupby(group_col)["cycle"].max().rename("max_cycle")
    result = result.join(max_cycles, on=group_col)

    result["rul"] = result["max_cycle"] - result["cycle"]
    result["rul_capped"] = result["rul"].clip(upper=max_rul)
    result.drop(columns=["max_cycle"], inplace=True)

    return result


def add_binary_labels(
    df: pd.DataFrame,
    horizons: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Add binary classification labels for each alert horizon.

    Creates columns:
      - label_critical  : 1 if RUL <= 48
      - label_warning   : 1 if RUL <= 96
      - label_advisory  : 1 if RUL <= 168

    These are used as targets for the XGBoost classifier.
    The primary target is label_warning (72-hour prediction horizon
    is the operational standard in O&G PdM).
    """
    if horizons is None:
        horizons = ALERT_HORIZONS

    result = df.copy()
    if "rul" not in result.columns:
        raise ValueError("Run add_rul_labels() before add_binary_labels()")

    for name, horizon in horizons.items():
        result[f"label_{name}"] = (result["rul"] <= horizon).astype(int)

    return result


def add_lifecycle_features(
    df: pd.DataFrame,
    group_col: str = "unit_id",
) -> pd.DataFrame:
    """
    Add lifecycle context features that help models distinguish
    early-life from end-of-life sensor patterns.

    Features added:
      - cycle_normalized : cycle / max_cycle (relative age 0–1)
      - cycle_log        : log(cycle) (captures early rapid change)
      - is_early_life    : first 10% of lifecycle
      - is_late_life     : last 20% of lifecycle
    """
    result = df.copy()
    max_cycles = df.groupby(group_col)["cycle"].max().rename("max_cycle")
    result = result.join(max_cycles, on=group_col)

    result["cycle_normalized"] = result["cycle"] / result["max_cycle"]
    result["cycle_log"] = np.log1p(result["cycle"])
    result["is_early_life"] = (result["cycle_normalized"] <= 0.10).astype(int)
    result["is_late_life"] = (result["cycle_normalized"] >= 0.80).astype(int)

    result.drop(columns=["max_cycle"], inplace=True)
    return result


def get_test_rul_labels(
    test_df: pd.DataFrame,
    true_rul_series: pd.Series,
    group_col: str = "unit_id",
    max_rul: int = MAX_RUL,
) -> pd.DataFrame:
    """
    For test data (truncated sequences), assign true RUL labels
    using the provided ground truth series.

    Parameters
    ----------
    test_df         : test DataFrame (last cycle of each unit is target point)
    true_rul_series : Series of true RUL values (one per unit)
    max_rul         : cap value

    Returns
    -------
    test_df with rul, rul_capped, and binary label columns for the
    LAST observation of each unit (the prediction point).
    """
    # Get last observation per unit
    last_obs = (
        test_df.sort_values([group_col, "cycle"])
        .groupby(group_col)
        .last()
        .reset_index()
    )

    rul_map = dict(
        zip(range(1, len(true_rul_series) + 1), true_rul_series.values)
    )
    last_obs["rul"] = last_obs[group_col].map(rul_map)
    last_obs["rul_capped"] = last_obs["rul"].clip(upper=max_rul)

    for name, horizon in ALERT_HORIZONS.items():
        last_obs[f"label_{name}"] = (last_obs["rul"] <= horizon).astype(int)

    return last_obs


def print_label_stats(df: pd.DataFrame) -> None:
    """Print class balance statistics for each label column."""
    label_cols = [c for c in df.columns if c.startswith("label_")]

    print("\nLabel statistics:")
    print(f"  Total samples: {len(df):,}")
    if "rul" in df.columns:
        print(f"  RUL range: {df['rul'].min():.0f} – {df['rul'].max():.0f} cycles")
        print(f"  Mean RUL: {df['rul'].mean():.1f} cycles")
    print()

    for col in label_cols:
        n_pos = df[col].sum()
        pct = 100 * n_pos / len(df)
        print(f"  {col:<20} positive={n_pos:>6,} ({pct:5.1f}%)  "
              f"negative={len(df)-n_pos:>6,} ({100-pct:5.1f}%)")
