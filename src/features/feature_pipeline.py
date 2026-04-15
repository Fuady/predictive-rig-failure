"""
feature_pipeline.py
-------------------
Orchestrates the full feature engineering pipeline:
  raw data → signal features → labels → feature matrix → saved to disk

Run this after download_data.py:
    python src/features/feature_pipeline.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.data_loader import (
    load_train_test,
    INFORMATIVE_SENSORS,
    validate_data,
)
from src.features.signal_processing import (
    compute_rolling_features,
    add_cross_sensor_features,
    normalize_per_unit,
    WINDOW_SIZES,
)
from src.features.label_engineering import (
    add_rul_labels,
    add_binary_labels,
    add_lifecycle_features,
    get_test_rul_labels,
    print_label_stats,
)

FEATURES_DIR = Path("data/features")
PROCESSED_DIR = Path("data/processed")


def build_feature_matrix(
    df: pd.DataFrame,
    sensors: list,
    is_train: bool = True,
) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline to a DataFrame.
    """
    print("  1. Normalizing sensors per unit...")
    df = normalize_per_unit(df, sensors)

    print("  2. Computing rolling features...")
    normalized_sensors = [f"{s}_normalized" for s in sensors]
    for window in WINDOW_SIZES:
        df = compute_rolling_features(df, normalized_sensors, window=window)

    print("  3. Adding cross-sensor interaction features...")
    df = add_cross_sensor_features(df)

    if is_train:
        print("  4. Adding RUL and binary labels...")
        df = add_rul_labels(df)
        df = add_binary_labels(df)

    print("  5. Adding lifecycle features...")
    df = add_lifecycle_features(df)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return only the engineered feature column names (not raw sensors or labels)."""
    exclude_prefixes = ("sensor_", "label_", "unit_id", "cycle",
                        "op_setting", "rul")
    feature_cols = [
        c for c in df.columns
        if not any(c.startswith(p) for p in exclude_prefixes)
        and c not in ("unit_id", "cycle", "rul", "rul_capped")
    ]
    return feature_cols


def run_pipeline(subset: str = "FD001") -> dict:
    """
    Run the full feature engineering pipeline for one C-MAPSS subset.

    Returns dict with train_df, test_df, feature_cols, label_cols.
    """
    print(f"\n{'='*55}")
    print(f"  Feature Engineering Pipeline — {subset}")
    print(f"{'='*55}")

    # Load raw data
    print("\nLoading raw data...")
    train_df, test_df, true_rul = load_train_test(subset)
    print(f"  Train: {len(train_df):,} rows, {train_df['unit_id'].nunique()} units")
    print(f"  Test:  {len(test_df):,} rows,  {test_df['unit_id'].nunique()} units")

    # Validate
    print("\nValidating data quality...")
    report = validate_data(train_df)
    if report["issues"]:
        for issue in report["issues"]:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ No data quality issues")

    # Determine which sensors to use
    available_sensors = [
        s for s in INFORMATIVE_SENSORS if s in train_df.columns
    ]
    print(f"\nUsing {len(available_sensors)} informative sensors: "
          f"{available_sensors[:4]}...")

    # Build training feature matrix
    print("\nBuilding TRAIN feature matrix...")
    train_feat = build_feature_matrix(train_df, available_sensors, is_train=True)

    # Build test feature matrix
    print("\nBuilding TEST feature matrix...")
    test_feat = build_feature_matrix(test_df, available_sensors, is_train=False)

    # Get feature column names
    feature_cols = get_feature_columns(train_feat)
    label_cols = [c for c in train_feat.columns if c.startswith("label_")]

    print(f"\n✓ Feature matrix built:")
    print(f"  Features:  {len(feature_cols)}")
    print(f"  Labels:    {label_cols}")

    # Fill any remaining NaNs (can arise from rolling windows at sequence start)
    train_feat[feature_cols] = train_feat[feature_cols].fillna(0)
    test_feat[feature_cols] = test_feat[feature_cols].fillna(0)

    # Print label balance
    print_label_stats(train_feat)

    # Save to disk
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_out = FEATURES_DIR / f"train_{subset}_features.parquet"
    test_out = FEATURES_DIR / f"test_{subset}_features.parquet"
    meta_out = FEATURES_DIR / f"meta_{subset}.joblib"

    train_feat.to_parquet(train_out, index=False)
    test_feat.to_parquet(test_out, index=False)

    meta = {
        "subset": subset,
        "feature_cols": feature_cols,
        "label_cols": label_cols,
        "sensors_used": available_sensors,
        "true_rul_test": true_rul.tolist(),
        "n_train": len(train_feat),
        "n_test": len(test_feat),
    }
    joblib.dump(meta, meta_out)

    print(f"\n✓ Saved features:")
    print(f"  {train_out}")
    print(f"  {test_out}")
    print(f"  {meta_out}")

    return {
        "train_df": train_feat,
        "test_df": test_feat,
        "feature_cols": feature_cols,
        "label_cols": label_cols,
        "meta": meta,
        "true_rul": true_rul,
    }


def load_features(subset: str = "FD001") -> dict:
    """Load pre-computed feature matrices from disk."""
    train_path = FEATURES_DIR / f"train_{subset}_features.parquet"
    test_path = FEATURES_DIR / f"test_{subset}_features.parquet"
    meta_path = FEATURES_DIR / f"meta_{subset}.joblib"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Features not found for {subset}. "
            "Run: python src/features/feature_pipeline.py"
        )

    meta = joblib.load(meta_path)
    return {
        "train_df": pd.read_parquet(train_path),
        "test_df": pd.read_parquet(test_path),
        "feature_cols": meta["feature_cols"],
        "label_cols": meta["label_cols"],
        "true_rul": pd.Series(meta["true_rul_test"]),
        "meta": meta,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument("--subset", default="FD001",
                        choices=["FD001", "FD002", "FD003", "FD004"],
                        help="C-MAPSS subset to process")
    parser.add_argument("--all", action="store_true",
                        help="Process all available subsets")
    args = parser.parse_args()

    subsets = ["FD001", "FD003"] if args.all else [args.subset]
    for subset in subsets:
        # Only run if raw data exists
        from pathlib import Path as P
        raw_check = P("data/raw") / f"train_{subset}.csv"
        if not raw_check.exists():
            print(f"\nSkipping {subset}: raw data not found.")
            print("Run: python src/ingestion/download_data.py")
            continue
        run_pipeline(subset)

    print("\n✓ Feature pipeline complete.")
    print("  Next: python src/models/train.py")
