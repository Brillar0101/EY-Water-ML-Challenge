"""
Load and merge all datasets for the water quality prediction pipeline.
Handles training data, validation data, and pre-extracted feature CSVs.
"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"


def load_training_data():
    """Load the water quality training dataset."""
    df = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
    df["Sample Date"] = pd.to_datetime(df["Sample Date"], dayfirst=True)
    return df


def load_submission_template():
    """Load the validation/submission template (targets are NaN)."""
    df = pd.read_csv(DATA_DIR / "submission_template.csv")
    df["Sample Date"] = pd.to_datetime(df["Sample Date"], dayfirst=True)
    return df


def load_landsat_features(split="train"):
    """Load pre-extracted Landsat features."""
    fname = "train_landsat_features.csv" if split == "train" else "val_landsat_features.csv"
    df = pd.read_csv(DATA_DIR / fname)
    df["Sample Date"] = pd.to_datetime(df["Sample Date"], dayfirst=True)
    return df


def load_terraclimate_features(split="train"):
    """Load pre-extracted TerraClimate features (PET only from benchmark)."""
    fname = "train_terraclimate_features.csv" if split == "train" else "val_terraclimate_features.csv"
    df = pd.read_csv(DATA_DIR / fname)
    df["Sample Date"] = pd.to_datetime(df["Sample Date"], dayfirst=True)
    return df


def _parse_dates(series):
    """Parse dates flexibly — handles both DD-MM-YYYY and YYYY-MM-DD formats."""
    try:
        return pd.to_datetime(series, dayfirst=True)
    except (ValueError, TypeError):
        return pd.to_datetime(series, format="mixed", dayfirst=True)


def load_extended_terraclimate(split="train"):
    """Load extended TerraClimate features (all 14 variables) if available."""
    fname = f"{split}_terraclimate_extended.csv"
    fpath = DATA_DIR / "processed" / fname
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    df["Sample Date"] = _parse_dates(df["Sample Date"])
    return df


def load_terrain_features():
    """Load elevation/terrain features if available."""
    fpath = DATA_DIR / "processed" / "terrain_features.csv"
    if not fpath.exists():
        return None
    return pd.read_csv(fpath)


def load_landcover_features():
    """Load land cover features if available."""
    fpath = DATA_DIR / "processed" / "landcover_features.csv"
    if not fpath.exists():
        return None
    return pd.read_csv(fpath)


def load_soil_features():
    """Load soil features if available."""
    fpath = DATA_DIR / "processed" / "soil_features.csv"
    if not fpath.exists():
        return None
    return pd.read_csv(fpath)


def get_unique_stations(df):
    """Extract unique station locations from a dataset."""
    stations = df.groupby(["Latitude", "Longitude"]).size().reset_index(name="sample_count")
    return stations


def merge_features(target_df, feature_dfs, on_cols=None):
    """
    Merge multiple feature DataFrames onto a target DataFrame.
    Joins on Latitude + Longitude + Sample Date by default.
    For static features (terrain, soil, landcover), joins on Latitude + Longitude only.
    """
    if on_cols is None:
        on_cols = ["Latitude", "Longitude", "Sample Date"]

    result = target_df.copy()
    for fdf in feature_dfs:
        if fdf is None:
            continue
        # Determine join columns based on what's available
        join_cols = [c for c in on_cols if c in fdf.columns]
        # Drop duplicate columns before merge
        new_cols = [c for c in fdf.columns if c not in result.columns or c in join_cols]
        result = result.merge(fdf[new_cols], on=join_cols, how="left")

    return result


def build_full_dataset(split="train"):
    """
    Build the complete feature dataset by merging all available sources.
    Returns a DataFrame with all features joined onto the target data.
    """
    # Load base data
    if split == "train":
        base_df = load_training_data()
    else:
        base_df = load_submission_template()

    # Load feature sources
    landsat = load_landsat_features(split)
    tc_basic = load_terraclimate_features(split)
    tc_extended = load_extended_terraclimate(split)
    terrain = load_terrain_features()
    landcover = load_landcover_features()
    soil = load_soil_features()

    # Temporal features on Lat/Lon/Date
    temporal_dfs = [landsat]
    if tc_extended is not None:
        temporal_dfs.append(tc_extended)
    else:
        temporal_dfs.append(tc_basic)

    # Merge temporal features (joined on lat/lon/date)
    result = merge_features(base_df, temporal_dfs)

    # Merge static features (joined on lat/lon only)
    static_dfs = [terrain, landcover, soil]
    result = merge_features(result, static_dfs, on_cols=["Latitude", "Longitude"])

    return result


TARGET_COLS = [
    "Total Alkalinity",
    "Electrical Conductance",
    "Dissolved Reactive Phosphorus",
]

ID_COLS = ["Latitude", "Longitude", "Sample Date"]


if __name__ == "__main__":
    print("Loading training data...")
    train = build_full_dataset("train")
    print(f"Training shape: {train.shape}")
    print(f"Columns: {list(train.columns)}")
    print(f"\nMissing values:\n{train.isnull().sum()}")

    print("\nLoading validation data...")
    val = build_full_dataset("val")
    print(f"Validation shape: {val.shape}")

    print("\nUnique training stations:")
    stations = get_unique_stations(train)
    print(f"  {len(stations)} stations")
    print(f"  Lat range: {stations.Latitude.min():.2f} to {stations.Latitude.max():.2f}")
    print(f"  Lon range: {stations.Longitude.min():.2f} to {stations.Longitude.max():.2f}")
