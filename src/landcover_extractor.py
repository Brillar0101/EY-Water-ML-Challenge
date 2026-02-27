"""
Extract land cover composition from ESA WorldCover 10m via Microsoft Planetary Computer.
These are STATIC features — same for all dates at a given station.

For each station, computes the percentage of each land cover class within
multiple buffer radii (1km, 5km).

ESA WorldCover Classes:
    10 = Tree cover
    20 = Shrubland
    30 = Grassland
    40 = Cropland
    50 = Built-up
    60 = Bare / sparse vegetation
    70 = Snow and ice
    80 = Permanent water bodies
    90 = Herbaceous wetland
    95 = Mangroves
   100 = Moss and lichen

Usage:
    python src/landcover_extractor.py
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

# ESA WorldCover class labels
LC_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water_bodies",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}


def get_unique_stations():
    """Get all unique station locations from both training and validation."""
    train = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
    val = pd.read_csv(DATA_DIR / "submission_template.csv")

    all_locs = pd.concat([
        train[["Latitude", "Longitude"]],
        val[["Latitude", "Longitude"]],
    ]).drop_duplicates().reset_index(drop=True)

    return all_locs


def extract_landcover(lat, lon, buffer_deg=0.045):
    """
    Extract land cover composition from ESA WorldCover 2021.

    Args:
        lat, lon: Station coordinates.
        buffer_deg: Buffer in degrees around the point.
                    0.009 ~ 1km, 0.045 ~ 5km at South African latitudes.

    Returns:
        dict with percentage of each land cover class.
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
        collections=["esa-worldcover"],
        bbox=bbox,
        query={"esa_worldcover:product_version": {"eq": "V200"}},
    )

    items = list(search.items())

    if not items:
        # Try V100 (2020) as fallback
        search = catalog.search(
            collections=["esa-worldcover"],
            bbox=bbox,
        )
        items = list(search.items())

    if not items:
        return {f"lc_{name}_pct": np.nan for name in LC_CLASSES.values()}

    try:
        data = stac_load(
            items,
            bands=["map"],
            bbox=bbox,
            crs="EPSG:4326",
            resolution=10 / 111320,  # ~10m pixels
        )

        lc = data["map"].isel(time=0).values.flatten()
        lc = lc[~np.isnan(lc)]

        if len(lc) == 0:
            return {f"lc_{name}_pct": np.nan for name in LC_CLASSES.values()}

        total_pixels = len(lc)
        result = {}

        for class_val, class_name in LC_CLASSES.items():
            count = np.sum(lc == class_val)
            result[f"lc_{class_name}_pct"] = float(count / total_pixels * 100)

        # Derived features
        result["lc_vegetation_pct"] = (
            result["lc_tree_cover_pct"] +
            result["lc_shrubland_pct"] +
            result["lc_grassland_pct"]
        )
        result["lc_human_pct"] = (
            result["lc_cropland_pct"] +
            result["lc_built_up_pct"]
        )
        result["lc_water_wetland_pct"] = (
            result["lc_water_bodies_pct"] +
            result["lc_herbaceous_wetland_pct"] +
            result["lc_mangroves_pct"]
        )

        # Landscape diversity (Shannon index)
        proportions = np.array([
            result[f"lc_{name}_pct"] / 100
            for name in LC_CLASSES.values()
        ])
        proportions = proportions[proportions > 0]
        result["lc_shannon_diversity"] = float(-np.sum(proportions * np.log(proportions)))

        return result

    except Exception as e:
        print(f"  Error at ({lat}, {lon}): {e}")
        return {f"lc_{name}_pct": np.nan for name in LC_CLASSES.values()}


def extract_multiscale_landcover(lat, lon):
    """
    Extract land cover at two buffer scales and combine.
    1km (~0.009 deg) captures immediate surroundings.
    5km (~0.045 deg) captures regional land use patterns.
    """
    # 1km buffer
    lc_1km = extract_landcover(lat, lon, buffer_deg=0.009)
    lc_1km = {k.replace("lc_", "lc_1km_"): v for k, v in lc_1km.items()}

    # 5km buffer
    lc_5km = extract_landcover(lat, lon, buffer_deg=0.045)
    lc_5km = {k.replace("lc_", "lc_5km_"): v for k, v in lc_5km.items()}

    return {**lc_1km, **lc_5km}


def extract_all_landcover():
    """
    Extract land cover features for all unique station locations.
    Saves to datasets/processed/landcover_features.csv
    """
    output_path = OUTPUT_DIR / "landcover_features.csv"

    stations = get_unique_stations()
    print(f"Extracting land cover for {len(stations)} unique locations")

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

    for _, row in tqdm(stations.iterrows(), total=len(stations), desc="Land Cover"):
        lc = extract_multiscale_landcover(row["Latitude"], row["Longitude"])
        lc["Latitude"] = row["Latitude"]
        lc["Longitude"] = row["Longitude"]
        results.append(lc)

        # Save periodically
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Shape: {result_df.shape}")
    print(result_df.describe())

    return result_df


if __name__ == "__main__":
    extract_all_landcover()
