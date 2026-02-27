"""
Generate and validate submission CSV files.

Usage:
    python src/submission.py --validate outputs/submissions/submission_v1.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "datasets"
SUBMISSION_DIR = ROOT_DIR / "outputs" / "submissions"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = [
    "Latitude",
    "Longitude",
    "Sample Date",
    "Total Alkalinity",
    "Electrical Conductance",
    "Dissolved Reactive Phosphorus",
]


def create_submission(val_df, predictions, version="v1"):
    """
    Create a submission CSV from validation data and predictions.

    Args:
        val_df: DataFrame with Latitude, Longitude, Sample Date
        predictions: dict mapping target names to prediction arrays
        version: version string for the filename
    """
    template = pd.read_csv(DATA_DIR / "submission_template.csv")

    submission = pd.DataFrame({
        "Latitude": template["Latitude"],
        "Longitude": template["Longitude"],
        "Sample Date": template["Sample Date"],
        "Total Alkalinity": predictions.get("Total Alkalinity", np.nan),
        "Electrical Conductance": predictions.get("Electrical Conductance", np.nan),
        "Dissolved Reactive Phosphorus": predictions.get("Dissolved Reactive Phosphorus", np.nan),
    })

    # Validate before saving
    issues = validate_submission(submission)
    if issues:
        print("WARNING: Submission has issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Submission validation passed.")

    # Save
    output_path = SUBMISSION_DIR / f"submission_{version}.csv"
    submission.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path}")

    # Print summary stats
    print("\nPrediction summary:")
    for col in ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]:
        vals = submission[col]
        print(f"  {col}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
              f"min={vals.min():.2f}, max={vals.max():.2f}")

    return submission


def validate_submission(df):
    """
    Check submission for common errors before uploading.
    Returns a list of issues (empty list = all good).
    """
    issues = []

    # Check row count
    if len(df) != 200:
        issues.append(f"Expected 200 rows, got {len(df)}")

    # Check required columns
    for col in REQUIRED_COLS:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")

    # Check for NaN in predictions
    for col in ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                issues.append(f"{col} has {nan_count} NaN values")

    # Check for negative predictions
    for col in ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(f"{col} has {neg_count} negative values")

    # Check for extreme outliers (sanity check based on training data ranges)
    ranges = {
        "Total Alkalinity": (0, 600),
        "Electrical Conductance": (0, 5000),
        "Dissolved Reactive Phosphorus": (0, 500),
    }
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
            if out_of_range > 0:
                issues.append(f"{col} has {out_of_range} values outside typical range [{lo}, {hi}]")

    # Check that lat/lon/date match template
    template = pd.read_csv(DATA_DIR / "submission_template.csv")
    if not np.allclose(df["Latitude"].values, template["Latitude"].values, atol=1e-4):
        issues.append("Latitude values don't match template")
    if not np.allclose(df["Longitude"].values, template["Longitude"].values, atol=1e-4):
        issues.append("Longitude values don't match template")

    return issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a submission file")
    parser.add_argument("--validate", type=str, help="Path to submission CSV to validate")
    args = parser.parse_args()

    if args.validate:
        df = pd.read_csv(args.validate)
        issues = validate_submission(df)
        if issues:
            print("ISSUES FOUND:")
            for i in issues:
                print(f"  - {i}")
        else:
            print("Submission is valid!")
            print(f"\nShape: {df.shape}")
            for col in ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]:
                print(f"  {col}: mean={df[col].mean():.2f}, range=[{df[col].min():.2f}, {df[col].max():.2f}]")
