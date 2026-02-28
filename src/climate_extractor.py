"""
Extract ALL TerraClimate variables (14 total) for training and validation locations.
Adapts the provided extraction notebook to pull all variables instead of just PET.

Usage:
    python src/climate_extractor.py --split train
    python src/climate_extractor.py --split val
    python src/climate_extractor.py --split both
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm import tqdm

import pystac_client
import planetary_computer as pc


DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
OUTPUT_DIR = DATA_DIR / "processed"

# All TerraClimate variables (skip swe — snow is negligible in South Africa)
TC_VARIABLES = [
    "tmax",  # Max temperature (C * 10)
    "tmin",  # Min temperature (C * 10)
    "ppt",   # Precipitation accumulation (mm)
    "vap",   # Vapor pressure (kPa * 100)
    "srad",  # Downward shortwave radiation (W/m2)
    "ws",    # Wind speed (m/s * 100)
    "aet",   # Actual evapotranspiration (mm)
    "pet",   # Reference evapotranspiration (mm)
    "q",     # Runoff (mm)
    "def",   # Climate water deficit (mm)
    "soil",  # Soil moisture (mm)
    "pdsi",  # Palmer Drought Severity Index
    "vpd",   # Vapor pressure deficit (kPa * 100)
]


def load_terraclimate_dataset():
    """Open TerraClimate Zarr dataset from Microsoft Planetary Computer."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]

    if "xarray:storage_options" in asset.extra_fields:
        ds = xr.open_zarr(
            asset.href,
            storage_options=asset.extra_fields["xarray:storage_options"],
            consolidated=True,
        )
    else:
        ds = xr.open_dataset(
            asset.href,
            **asset.extra_fields["xarray:open_kwargs"],
        )

    return ds


def filter_south_africa(ds, var_name):
    """
    Filter TerraClimate data for South Africa region and 2011-2015 time window.
    Returns a DataFrame with columns: Latitude, Longitude, Sample Date, <var_name>
    """
    ds_filtered = ds[var_name].sel(time=slice("2011-01-01", "2015-12-31"))

    frames = []
    for i in tqdm(range(len(ds_filtered.time)), desc=f"Filtering {var_name}"):
        df_step = ds_filtered.isel(time=i).to_dataframe().reset_index()
        df_step = df_step[
            (df_step["lat"] > -35.18) & (df_step["lat"] < -21.72) &
            (df_step["lon"] > 14.97) & (df_step["lon"] < 32.79)
        ]
        frames.append(df_step)

    df_final = pd.concat(frames, ignore_index=True)
    df_final["time"] = df_final["time"].astype(str)
    df_final = df_final.rename(columns={"lat": "Latitude", "lon": "Longitude", "time": "Sample Date"})

    return df_final


def assign_nearest_climate(sa_df, climate_df, var_name):
    """
    Map nearest TerraClimate grid point values to each sampling location.
    Uses KD-tree for spatial matching and finds the closest time match.
    """
    sa_coords = np.radians(sa_df[["Latitude", "Longitude"]].values)
    climate_coords = np.radians(climate_df[["Latitude", "Longitude"]].values)

    tree = cKDTree(climate_coords)
    dist, idx = tree.query(sa_coords, k=1)

    nearest_points = climate_df.iloc[idx].reset_index(drop=True)
    sa_df = sa_df.reset_index(drop=True)
    sa_df[["nearest_lat", "nearest_lon"]] = nearest_points[["Latitude", "Longitude"]]

    sa_dates = pd.to_datetime(sa_df["Sample Date"], dayfirst=True, errors="coerce")
    climate_dates = pd.to_datetime(climate_df["Sample Date"], errors="coerce")

    climate_values = []
    for i in tqdm(range(len(sa_df)), desc=f"Mapping {var_name}"):
        sample_date = sa_dates.iloc[i]
        nearest_lat = sa_df.loc[i, "nearest_lat"]
        nearest_lon = sa_df.loc[i, "nearest_lon"]

        subset = climate_df[
            (climate_df["Latitude"] == nearest_lat) &
            (climate_df["Longitude"] == nearest_lon)
        ]

        if subset.empty:
            climate_values.append(np.nan)
            continue

        nearest_idx = (climate_dates.loc[subset.index] - sample_date).abs().idxmin()
        climate_values.append(subset.loc[nearest_idx, var_name])

    return climate_values


def extract_all_variables(split="train"):
    """
    Extract all TerraClimate variables for the specified split.
    Saves result to datasets/processed/{split}_terraclimate_extended.csv
    """
    # Load target locations
    if split == "train":
        target_df = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
    else:
        target_df = pd.read_csv(DATA_DIR / "submission_template.csv")

    target_df["Sample Date"] = pd.to_datetime(target_df["Sample Date"], dayfirst=True)

    print(f"\nExtracting TerraClimate for {split} ({len(target_df)} rows)")
    print(f"Variables: {TC_VARIABLES}")

    # Output dataframe starts with location/date identifiers
    result_df = target_df[["Latitude", "Longitude", "Sample Date"]].copy()

    # Extract each variable
    for var_name in TC_VARIABLES:
        print(f"\n{'='*60}")
        print(f"Extracting: {var_name}")
        print(f"{'='*60}")

        # Check if we already have this variable (resume capability)
        output_path = OUTPUT_DIR / f"{split}_terraclimate_extended.csv"
        if output_path.exists():
            existing = pd.read_csv(output_path)
            if var_name in existing.columns:
                print(f"  Already extracted, skipping...")
                result_df[var_name] = existing[var_name].values
                continue

        # Reload dataset each variable to get a fresh auth token
        print("  Connecting to TerraClimate dataset (fresh token)...")
        ds = load_terraclimate_dataset()

        # Filter for South Africa region
        tc_filtered = filter_south_africa(ds, var_name)

        # Map to sample locations
        values = assign_nearest_climate(target_df, tc_filtered, var_name)
        result_df[var_name] = values

        # Save incrementally (so we can resume if interrupted)
        result_df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    # Add lag features
    print("\nComputing lag features...")
    result_df = add_climate_lag_features(result_df)

    # Final save
    output_path = OUTPUT_DIR / f"{split}_terraclimate_extended.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\nFinal output: {output_path}")
    print(f"Shape: {result_df.shape}")

    return result_df


def add_climate_lag_features(df):
    """
    Add lagged climate features for key variables.
    For each station, compute previous month and 3-month rolling average.
    """
    df = df.copy()
    df["Sample Date"] = pd.to_datetime(df["Sample Date"])

    # Sort by location then date
    df = df.sort_values(["Latitude", "Longitude", "Sample Date"]).reset_index(drop=True)

    lag_vars = ["ppt", "tmax", "soil", "q"]

    for var in lag_vars:
        if var not in df.columns:
            continue

        # Within each station, compute lags
        grouped = df.groupby(["Latitude", "Longitude"])

        # 1-month lag (previous observation at same station)
        df[f"{var}_lag1"] = grouped[var].shift(1)

        # Rolling 3-observation mean at same station
        df[f"{var}_roll3"] = grouped[var].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract TerraClimate features")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.split in ("train", "both"):
        extract_all_variables("train")

    if args.split in ("val", "both"):
        extract_all_variables("val")

    print("\nDone!")
