"""
Extract elevation and terrain features from SRTM DEM via Microsoft Planetary Computer.
These are STATIC features — same for all dates at a given station.

Usage:
    python src/terrain_extractor.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import pystac_client
import planetary_computer as pc
from odc.stac import stac_load


DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_unique_stations():
    """Get all unique station locations from both training and validation."""
    train = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
    val = pd.read_csv(DATA_DIR / "submission_template.csv")

    all_locs = pd.concat([
        train[["Latitude", "Longitude"]],
        val[["Latitude", "Longitude"]],
    ]).drop_duplicates().reset_index(drop=True)

    return all_locs


def extract_elevation(lat, lon, buffer_deg=0.005):
    """
    Extract elevation from Copernicus DEM via Planetary Computer.
    Uses a small buffer around the point and takes the median.
    buffer_deg=0.005 is roughly 500m.
    """
    bbox = [
        lon - buffer_deg,
        lat - buffer_deg,
        lon + buffer_deg,
        lat + buffer_deg,
    ]

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox,
    )

    items = list(search.items())

    if not items:
        return {"elevation": np.nan, "slope": np.nan, "aspect": np.nan, "roughness": np.nan}

    try:
        data = stac_load(
            items,
            bands=["data"],
            bbox=bbox,
            crs="EPSG:4326",
            resolution=30 / 111320,
        )

        elev = data["data"].isel(time=0).values.astype(float)
        elev[elev < -500] = np.nan  # Remove no-data values

        if np.all(np.isnan(elev)):
            return {"elevation": np.nan, "slope": np.nan, "aspect": np.nan, "roughness": np.nan}

        # Compute terrain derivatives
        median_elev = float(np.nanmedian(elev))

        # Slope (gradient magnitude in degrees)
        dy, dx = np.gradient(elev, 30)  # 30m pixel spacing
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        median_slope = float(np.nanmedian(np.degrees(slope_rad)))

        # Aspect (direction of steepest descent)
        aspect = np.degrees(np.arctan2(-dy, dx))
        aspect = (aspect + 360) % 360
        median_aspect = float(np.nanmedian(aspect))

        # Roughness (standard deviation of elevation)
        roughness = float(np.nanstd(elev))

        # Topographic Wetness Index approximation
        # TWI = ln(contributing_area / tan(slope))
        # Simplified: use inverse slope as proxy
        slope_safe = np.maximum(slope_rad, 0.01)
        twi = np.log(1.0 / np.tan(slope_safe))
        median_twi = float(np.nanmedian(twi))

        return {
            "elevation": median_elev,
            "slope": median_slope,
            "aspect": median_aspect,
            "roughness": roughness,
            "twi": median_twi,
        }

    except Exception as e:
        print(f"  Error at ({lat}, {lon}): {e}")
        return {"elevation": np.nan, "slope": np.nan, "aspect": np.nan,
                "roughness": np.nan, "twi": np.nan}


def extract_all_terrain():
    """
    Extract terrain features for all unique station locations.
    Saves to datasets/processed/terrain_features.csv
    """
    output_path = OUTPUT_DIR / "terrain_features.csv"

    stations = get_unique_stations()
    print(f"Extracting terrain for {len(stations)} unique locations")

    # Resume from existing if available
    if output_path.exists():
        existing = pd.read_csv(output_path)
        print(f"Resuming from {len(existing)} already extracted")
        done_coords = set(zip(existing["Latitude"], existing["Longitude"]))
        stations = stations[
            ~stations.apply(lambda r: (r["Latitude"], r["Longitude"]) in done_coords, axis=1)
        ]
        print(f"Remaining: {len(stations)}")
        results = existing.to_dict("records")
    else:
        results = []

    for _, row in tqdm(stations.iterrows(), total=len(stations), desc="Terrain"):
        terrain = extract_elevation(row["Latitude"], row["Longitude"])
        terrain["Latitude"] = row["Latitude"]
        terrain["Longitude"] = row["Longitude"]
        results.append(terrain)

        # Save periodically
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Shape: {result_df.shape}")
    print(result_df.describe())

    return result_df


if __name__ == "__main__":
    extract_all_terrain()
