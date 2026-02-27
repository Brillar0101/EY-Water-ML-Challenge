"""
Feature engineering for the water quality prediction pipeline.
Derives temporal, spectral, and interaction features from raw data.
"""

import pandas as pd
import numpy as np


def add_temporal_features(df):
    """
    Add time-based features derived from Sample Date.
    No external data needed — purely computed from the date column.
    """
    dt = pd.to_datetime(df["Sample Date"])

    df = df.copy()
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["year"] = dt.dt.year

    # Cyclic encoding for month (captures Jan-Dec continuity)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Cyclic encoding for day of year
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # South Africa seasons (southern hemisphere)
    # DJF = Summer (wet), MAM = Autumn, JJA = Winter (dry), SON = Spring
    season_map = {
        12: 0, 1: 0, 2: 0,   # Summer (wet season)
        3: 1, 4: 1, 5: 1,    # Autumn
        6: 2, 7: 2, 8: 2,    # Winter (dry season)
        9: 3, 10: 3, 11: 3,  # Spring
    }
    df["season"] = df["month"].map(season_map)

    # Quarter
    df["quarter"] = dt.dt.quarter

    return df


def add_landsat_indices(df):
    """
    Compute additional spectral indices from Landsat bands.
    Assumes columns: nir, green, swir16, swir22 exist.
    NDMI and MNDWI may already exist from pre-extracted data.
    """
    df = df.copy()
    eps = 1e-10

    # Recompute NDMI and MNDWI if bands are present (handles NaN better)
    if "nir" in df.columns and "swir16" in df.columns:
        df["NDMI"] = (df["nir"] - df["swir16"]) / (df["nir"] + df["swir16"] + eps)

    if "green" in df.columns and "swir22" in df.columns:
        df["MNDWI"] = (df["green"] - df["swir22"]) / (df["green"] + df["swir22"] + eps)

    # NDWI (water detection) - Green vs NIR
    if "green" in df.columns and "nir" in df.columns:
        df["NDWI"] = (df["green"] - df["nir"]) / (df["green"] + df["nir"] + eps)

    # Moisture Stress Index
    if "nir" in df.columns and "swir22" in df.columns:
        df["MSI"] = df["swir22"] / (df["nir"] + eps)

    # Band ratios that may capture water chemistry signals
    if "swir16" in df.columns and "swir22" in df.columns:
        df["SWIR_ratio"] = df["swir16"] / (df["swir22"] + eps)

    if "green" in df.columns and "swir16" in df.columns:
        df["green_swir16_ratio"] = df["green"] / (df["swir16"] + eps)

    return df


def add_missing_indicators(df, columns=None):
    """
    Add binary indicator columns for missing values.
    This lets the model learn that 'missing' itself is informative
    (e.g., Landsat NaN means cloud cover, which correlates with rain).
    """
    if columns is None:
        columns = ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]

    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    return df


def impute_missing_values(df, strategy="knn", exclude_cols=None):
    """
    Handle missing values in feature columns.
    Options: 'median', 'knn', 'group_median'
    """
    if exclude_cols is None:
        exclude_cols = [
            "Latitude", "Longitude", "Sample Date",
            "Total Alkalinity", "Electrical Conductance",
            "Dissolved Reactive Phosphorus",
        ]

    df = df.copy()
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ["float64", "float32", "int64"]]

    if strategy == "median":
        for col in feature_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    elif strategy == "group_median":
        # Group by station (lat/lon) and fill with station median first,
        # then global median for remaining NaN
        for col in feature_cols:
            if df[col].isna().any():
                station_medians = df.groupby(["Latitude", "Longitude"])[col].transform("median")
                df[col] = df[col].fillna(station_medians)
                df[col] = df[col].fillna(df[col].median())

    elif strategy == "knn":
        # KNN imputation using scikit-learn
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df[feature_cols] = imputer.fit_transform(df[feature_cols])

    return df


def add_station_features(train_df, target_df=None):
    """
    Compute per-station aggregate statistics from training data.
    These capture the baseline characteristics of each location.
    For validation data, we use the nearest training station's stats.
    """
    station_stats = train_df.groupby(["Latitude", "Longitude"]).agg(
        sample_count=("Total Alkalinity", "count"),
        ta_mean=("Total Alkalinity", "mean"),
        ta_std=("Total Alkalinity", "std"),
        ec_mean=("Electrical Conductance", "mean"),
        ec_std=("Electrical Conductance", "std"),
        drp_mean=("Dissolved Reactive Phosphorus", "mean"),
        drp_std=("Dissolved Reactive Phosphorus", "std"),
    ).reset_index()

    station_stats["ta_std"] = station_stats["ta_std"].fillna(0)
    station_stats["ec_std"] = station_stats["ec_std"].fillna(0)
    station_stats["drp_std"] = station_stats["drp_std"].fillna(0)

    if target_df is None:
        return station_stats

    # For validation: find nearest training station and use its stats
    from scipy.spatial import cKDTree

    train_coords = np.radians(station_stats[["Latitude", "Longitude"]].values)
    val_coords = np.radians(target_df[["Latitude", "Longitude"]].values)

    tree = cKDTree(train_coords)
    dist, idx = tree.query(val_coords, k=1)

    nearest_stats = station_stats.iloc[idx].reset_index(drop=True)
    nearest_stats["nearest_station_dist_deg"] = np.degrees(dist)

    # Only keep the derived columns (not lat/lon which would conflict)
    stat_cols = [c for c in nearest_stats.columns if c not in ["Latitude", "Longitude"]]
    for col in stat_cols:
        target_df[col] = nearest_stats[col].values

    return target_df


def build_features(df, is_training=True, train_df=None):
    """
    Full feature engineering pipeline.
    Applies all feature transformations in sequence.
    """
    df = add_temporal_features(df)
    df = add_landsat_indices(df)
    df = add_missing_indicators(df)
    df = impute_missing_values(df, strategy="median")

    return df


def get_feature_columns(df, exclude_targets=True):
    """Return list of feature columns (excluding IDs and targets)."""
    exclude = {"Latitude", "Longitude", "Sample Date"}
    if exclude_targets:
        exclude |= {
            "Total Alkalinity",
            "Electrical Conductance",
            "Dissolved Reactive Phosphorus",
        }

    return [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "float32", "int64", "int32"]]


if __name__ == "__main__":
    from data_loader import load_training_data, load_landsat_features, load_terraclimate_features, merge_features

    print("Loading data...")
    train = load_training_data()
    landsat = load_landsat_features("train")
    tc = load_terraclimate_features("train")

    merged = merge_features(train, [landsat, tc])
    print(f"Merged shape: {merged.shape}")

    featured = build_features(merged)
    print(f"After features: {featured.shape}")

    feat_cols = get_feature_columns(featured)
    print(f"\nFeature columns ({len(feat_cols)}):")
    for c in feat_cols:
        print(f"  {c}: {featured[c].isna().sum()} missing")
