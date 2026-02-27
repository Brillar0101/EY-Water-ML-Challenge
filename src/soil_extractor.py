"""
Extract soil properties from SoilGrids 250m via REST API.
These are STATIC features — same for all dates at a given station.

SoilGrids provides global soil data at 250m resolution.
We extract properties at two depths (0-5cm surface, 15-30cm subsurface).

Variables extracted:
    - pH (H2O)                → affects alkalinity directly
    - Organic carbon          → affects phosphorus retention
    - Clay fraction           → filtration capacity
    - Sand fraction           → permeability
    - Silt fraction           → derived from clay + sand
    - Cation exchange capacity → mineral content indicator
    - Bulk density            → soil compaction
    - Nitrogen                → nutrient loading proxy

Usage:
    python src/soil_extractor.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import time


DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOILGRIDS_API = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Properties and their depth intervals
SOIL_PROPERTIES = {
    "phh2o": "ph",           # pH in H2O (x10)
    "soc": "organic_carbon", # Soil organic carbon (dg/kg)
    "clay": "clay_pct",      # Clay fraction (g/kg)
    "sand": "sand_pct",      # Sand fraction (g/kg)
    "silt": "silt_pct",      # Silt fraction (g/kg)
    "cec": "cec",            # Cation exchange capacity (mmol(c)/kg)
    "bdod": "bulk_density",  # Bulk density (cg/cm3)
    "nitrogen": "nitrogen",  # Total nitrogen (cg/kg)
}

DEPTH_LABELS = ["0-5cm", "15-30cm"]


def extract_soil_point(lat, lon, retries=3):
    """
    Query SoilGrids REST API for soil properties at a single point.

    Returns a dict with soil property values at two depth intervals.
    Uses the 'mean' aggregation from SoilGrids.
    """
    params = {
        "lon": lon,
        "lat": lat,
        "property": list(SOIL_PROPERTIES.keys()),
        "depth": DEPTH_LABELS,
        "value": "mean",
    }

    for attempt in range(retries):
        try:
            response = requests.get(SOILGRIDS_API, params=params, timeout=30)

            if response.status_code == 429:
                # Rate limited — wait and retry
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                print(f"  API error {response.status_code} at ({lat}, {lon})")
                return _empty_soil_result()

            data = response.json()
            return _parse_soil_response(data)

        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            print(f"  Timeout at ({lat}, {lon})")
            return _empty_soil_result()

        except Exception as e:
            print(f"  Error at ({lat}, {lon}): {e}")
            return _empty_soil_result()

    return _empty_soil_result()


def _parse_soil_response(data):
    """Parse the JSON response from SoilGrids into a flat dict."""
    result = {}

    try:
        layers = data.get("properties", {}).get("layers", [])

        for layer in layers:
            prop_name = layer["name"]
            col_prefix = SOIL_PROPERTIES.get(prop_name, prop_name)

            for depth_entry in layer.get("depths", []):
                depth_label = depth_entry["label"]
                depth_tag = depth_label.replace("-", "_").replace("cm", "")

                value = depth_entry.get("values", {}).get("mean", None)

                col_name = f"soil_{col_prefix}_{depth_tag}"
                result[col_name] = float(value) if value is not None else np.nan

        # Apply unit conversions
        # pH is stored as x10
        for key in list(result.keys()):
            if "ph_" in key and not np.isnan(result.get(key, np.nan)):
                result[key] = result[key] / 10.0

        # Clay, sand, silt stored as g/kg — convert to percentage
        for key in list(result.keys()):
            if any(f"{frac}_pct" in key for frac in ["clay", "sand", "silt"]):
                if not np.isnan(result.get(key, np.nan)):
                    result[key] = result[key] / 10.0

        # Add derived features from surface (0-5cm) values
        ph_0_5 = result.get("soil_ph_0_5", np.nan)
        oc_0_5 = result.get("soil_organic_carbon_0_5", np.nan)
        clay_0_5 = result.get("soil_clay_pct_0_5", np.nan)
        sand_0_5 = result.get("soil_sand_pct_0_5", np.nan)
        cec_0_5 = result.get("soil_cec_0_5", np.nan)

        # Soil texture class proxy (sand/clay ratio)
        if not np.isnan(clay_0_5) and clay_0_5 > 0:
            result["soil_sand_clay_ratio"] = sand_0_5 / clay_0_5
        else:
            result["soil_sand_clay_ratio"] = np.nan

        # Nutrient capacity proxy (CEC * organic carbon)
        if not np.isnan(cec_0_5) and not np.isnan(oc_0_5):
            result["soil_nutrient_capacity"] = cec_0_5 * oc_0_5
        else:
            result["soil_nutrient_capacity"] = np.nan

    except Exception as e:
        print(f"  Parse error: {e}")
        return _empty_soil_result()

    return result


def _empty_soil_result():
    """Return a dict of NaN for all expected soil columns."""
    result = {}
    for prop_name, col_prefix in SOIL_PROPERTIES.items():
        for depth_label in DEPTH_LABELS:
            depth_tag = depth_label.replace("-", "_").replace("cm", "")
            result[f"soil_{col_prefix}_{depth_tag}"] = np.nan

    result["soil_sand_clay_ratio"] = np.nan
    result["soil_nutrient_capacity"] = np.nan

    return result


def get_unique_stations():
    """Get all unique station locations from both training and validation."""
    train = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
    val = pd.read_csv(DATA_DIR / "submission_template.csv")

    all_locs = pd.concat([
        train[["Latitude", "Longitude"]],
        val[["Latitude", "Longitude"]],
    ]).drop_duplicates().reset_index(drop=True)

    return all_locs


def extract_all_soil():
    """
    Extract soil features for all unique station locations.
    Saves to datasets/processed/soil_features.csv
    """
    output_path = OUTPUT_DIR / "soil_features.csv"

    stations = get_unique_stations()
    print(f"Extracting soil properties for {len(stations)} unique locations")

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

    for _, row in tqdm(stations.iterrows(), total=len(stations), desc="Soil"):
        soil = extract_soil_point(row["Latitude"], row["Longitude"])
        soil["Latitude"] = row["Latitude"]
        soil["Longitude"] = row["Longitude"]
        results.append(soil)

        # Respect API rate limits
        time.sleep(0.5)

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
    extract_all_soil()
