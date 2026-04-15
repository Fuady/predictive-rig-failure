"""
drift_detector.py
-----------------
Monitor for data and model drift in production using Evidently AI.

In production O&G deployments:
- Sensors get replaced (data drift: distribution shifts)
- Operating conditions change seasonally (ambient temperature in Gulf)
- Models degrade silently over months without drift detection

This module:
  1. Computes reference dataset statistics from training data
  2. Monitors incoming inference data for distribution shifts
  3. Generates weekly HTML drift reports
  4. Logs drift metrics to MLflow

Run weekly:
    python src/monitoring/drift_detector.py

Or schedule with cron:
    0 6 * * 1 cd /path/to/project && python src/monitoring/drift_detector.py
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MONITORING_DIR = Path("monitoring/reports")
MONITORING_DIR.mkdir(parents=True, exist_ok=True)


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI) — measures distribution shift.

    PSI < 0.10  : No significant change
    PSI 0.10–0.25: Moderate change, investigate
    PSI > 0.25  : Major shift, retrain model

    Used in O&G when sensors are replaced or calibrated.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]

    exp_pcts = (exp_counts / len(expected)).clip(min=1e-6)
    act_pcts = (act_counts / max(len(actual), 1)).clip(min=1e-6)

    psi = np.sum((act_pcts - exp_pcts) * np.log(act_pcts / exp_pcts))
    return float(psi)


def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list,
    report_name: str = "drift_report",
) -> dict:
    """
    Compute drift metrics between reference (training) and current (production) data.

    Returns a dictionary with:
    - PSI per feature
    - Features with significant drift (PSI > 0.1)
    - Overall drift score
    - Recommended action
    """
    results = {
        "report_name": report_name,
        "timestamp": datetime.now().isoformat(),
        "n_reference": len(reference_df),
        "n_current": len(current_df),
        "feature_drift": {},
        "drifted_features": [],
        "critical_features": [],
    }

    for feat in feature_cols:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue

        ref_vals = reference_df[feat].dropna().values
        cur_vals = current_df[feat].dropna().values

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        psi = compute_psi(ref_vals, cur_vals)
        ref_mean = float(np.mean(ref_vals))
        cur_mean = float(np.mean(cur_vals))
        mean_shift_pct = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-6) * 100

        drift_level = (
            "critical" if psi > 0.25 else
            "moderate" if psi > 0.10 else
            "stable"
        )

        results["feature_drift"][feat] = {
            "psi": round(psi, 4),
            "drift_level": drift_level,
            "ref_mean": round(ref_mean, 4),
            "cur_mean": round(cur_mean, 4),
            "mean_shift_pct": round(mean_shift_pct, 2),
        }

        if psi > 0.10:
            results["drifted_features"].append(feat)
        if psi > 0.25:
            results["critical_features"].append(feat)

    # Overall drift score
    all_psi = [v["psi"] for v in results["feature_drift"].values()]
    results["overall_psi_mean"] = round(np.mean(all_psi) if all_psi else 0.0, 4)
    results["n_drifted"] = len(results["drifted_features"])
    results["n_critical"] = len(results["critical_features"])

    # Recommendation
    if results["n_critical"] > 0:
        results["recommendation"] = (
            f"RETRAIN REQUIRED: {results['n_critical']} features show critical drift. "
            f"Affected: {results['critical_features'][:3]}"
        )
    elif results["n_drifted"] > 3:
        results["recommendation"] = (
            "INVESTIGATE: Multiple features drifting. "
            "Check sensor calibration records. Consider retraining."
        )
    else:
        results["recommendation"] = "No action needed. Model distribution is stable."

    return results


def generate_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list,
    output_path: Path,
) -> bool:
    """
    Generate an Evidently HTML drift report.
    Falls back to a JSON report if Evidently is not installed.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])

        # Sample columns to those available in both datasets
        common_cols = [c for c in feature_cols
                       if c in reference_df.columns and c in current_df.columns][:15]

        report.run(
            reference_data=reference_df[common_cols].head(500),
            current_data=current_df[common_cols].head(500),
        )
        report.save_html(str(output_path))
        print(f"  Evidently HTML report saved: {output_path}")
        return True

    except ImportError:
        print("  Evidently not installed. Generating JSON report instead.")
        return False
    except Exception as e:
        print(f"  Evidently report failed: {e}. Using JSON fallback.")
        return False


def run_monitoring(
    subset: str = "FD001",
    use_synthetic_current: bool = True,
) -> dict:
    """
    Run the full monitoring pipeline.

    In production:
    - reference_df = training feature matrix
    - current_df = last 7 days of inference requests (logged by the API)

    Here we simulate "current" data as a slightly drifted version of training.
    """
    from src.features.feature_pipeline import load_features

    print(f"\n{'='*55}")
    print(f"  Drift Monitoring — {subset}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    data = load_features(subset)
    reference_df = data["train_df"]
    feature_cols = data["feature_cols"][:20]   # Monitor top 20 features

    if use_synthetic_current:
        # Simulate mild sensor drift (e.g., sensor recalibration)
        current_df = reference_df.sample(500, random_state=99).copy()
        rng = np.random.default_rng(99)

        # Introduce drift in 3 sensors (as would happen in real O&G ops)
        for col in feature_cols[:3]:
            if col in current_df.columns:
                current_df[col] = (
                    current_df[col]
                    + rng.normal(0.5, 0.2, len(current_df))  # shift mean
                )
        print("  Using synthetic drifted data for demonstration.")
    else:
        # In production: load from inference logs
        inference_log = Path("monitoring/inference_log.parquet")
        if inference_log.exists():
            current_df = pd.read_parquet(inference_log)
        else:
            raise FileNotFoundError(
                "No inference log found. "
                "Run with use_synthetic_current=True for demo."
            )

    print(f"  Reference samples: {len(reference_df):,}")
    print(f"  Current samples:   {len(current_df):,}")
    print(f"  Features monitored:{len(feature_cols)}")

    # Run drift analysis
    results = run_drift_analysis(
        reference_df, current_df, feature_cols,
        report_name=f"{subset}_{datetime.now().strftime('%Y%m%d')}"
    )

    # Print summary
    print(f"\n  Overall PSI:      {results['overall_psi_mean']:.4f}")
    print(f"  Drifted features: {results['n_drifted']}")
    print(f"  Critical:         {results['n_critical']}")
    print(f"\n  Recommendation: {results['recommendation']}")

    if results["drifted_features"]:
        print(f"\n  Drifted features:")
        for feat in results["drifted_features"][:5]:
            d = results["feature_drift"][feat]
            print(
                f"    {feat:<35} PSI={d['psi']:.4f}  "
                f"mean: {d['ref_mean']:.3f} → {d['cur_mean']:.3f}  "
                f"({d['drift_level'].upper()})"
            )

    # Save JSON report
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = MONITORING_DIR / f"drift_{subset}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON report: {json_path}")

    # Try Evidently HTML report
    html_path = MONITORING_DIR / f"drift_{subset}_{ts}.html"
    generate_evidently_report(
        reference_df, current_df, feature_cols, html_path
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--live", action="store_true",
                        help="Use real inference logs (not synthetic)")
    args = parser.parse_args()

    results = run_monitoring(
        subset=args.subset,
        use_synthetic_current=not args.live,
    )
    print("\n✓ Monitoring complete.")
