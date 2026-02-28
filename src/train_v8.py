"""
Training pipeline v8 — NWU data as additional IDW stations.

v28 (calibrated NWU) scored -0.101 because the learned calibration
distorted predictions. v21 (hybrid IDW + ML blend) scored 0.258.

NEW STRATEGY: Don't replace the IDW model — ENRICH it.
Add NWU stations (converted to competition units) as extra data points
in the IDW pool. Since NWU stations are 0-17km from validation points
(vs 100-300km for original training stations), IDW will naturally
upweight them.

Unit conversions (empirical median ratios from 127 overlapping stations):
    EC:  NWU_mSm * 10.0 = competition EC (ratio 10.0, very tight)
    TAL: NWU_mgL * 1.0  = competition TAL (ratio 1.08, essentially same)
    DRP: NWU_PO4_mgL * 1000 * correction_factor

Approaches:
    v30: Expanded IDW + ML residual (v21 with NWU-enriched station pool)
    v31: Sweep DRP conversion factor (most uncertain conversion)
    v32: NWU for TAL only + v21 for EC/DRP (per-target best source)
    v33: Expanded training rows (NWU samples as training data)

Usage:
    python src/train_v8.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from submission import create_submission

TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
DATA_DIR = ROOT_DIR / "datasets"


def load_data():
    """Load competition training + validation data with temporal features."""
    train = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
    train["Sample Date"] = pd.to_datetime(train["Sample Date"], dayfirst=True)
    val = pd.read_csv(DATA_DIR / "submission_template.csv")
    val["Sample Date"] = pd.to_datetime(val["Sample Date"], dayfirst=True)

    landsat_tr = pd.read_csv(DATA_DIR / "train_landsat_features.csv")
    landsat_tr["Sample Date"] = pd.to_datetime(landsat_tr["Sample Date"], dayfirst=True)
    landsat_va = pd.read_csv(DATA_DIR / "val_landsat_features.csv")
    landsat_va["Sample Date"] = pd.to_datetime(landsat_va["Sample Date"], dayfirst=True)

    tc_tr = pd.read_csv(DATA_DIR / "train_terraclimate_features.csv")
    tc_tr["Sample Date"] = pd.to_datetime(tc_tr["Sample Date"], dayfirst=True)
    tc_va = pd.read_csv(DATA_DIR / "val_terraclimate_features.csv")
    tc_va["Sample Date"] = pd.to_datetime(tc_va["Sample Date"], dayfirst=True)

    ext_path = DATA_DIR / "processed"
    tc_ext_tr = tc_ext_va = None
    if (ext_path / "train_terraclimate_extended.csv").exists():
        tc_ext_tr = pd.read_csv(ext_path / "train_terraclimate_extended.csv")
        tc_ext_tr["Sample Date"] = pd.to_datetime(tc_ext_tr["Sample Date"], format="mixed", dayfirst=True)
        tc_ext_va = pd.read_csv(ext_path / "val_terraclimate_extended.csv")
        tc_ext_va["Sample Date"] = pd.to_datetime(tc_ext_va["Sample Date"], format="mixed", dayfirst=True)

    join = ["Latitude", "Longitude", "Sample Date"]
    train = train.merge(landsat_tr, on=join, how="left")
    train = train.merge(tc_tr, on=join, how="left")
    val = val.merge(landsat_va, on=join, how="left")
    val = val.merge(tc_va, on=join, how="left")

    if tc_ext_tr is not None:
        ext_cols = [c for c in tc_ext_tr.columns if c not in ["pet"] or c in join]
        train = train.merge(tc_ext_tr[ext_cols], on=join, how="left")
        val = val.merge(tc_ext_va[ext_cols], on=join, how="left")

    eps = 1e-10
    for df in [train, val]:
        df["month"] = df["Sample Date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy"] = df["Sample Date"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
        df["is_wet"] = ((df["month"] >= 10) | (df["month"] <= 3)).astype(int)
        if "green" in df.columns and "nir" in df.columns:
            df["NDWI"] = (df["green"] - df["nir"]) / (df["green"] + df["nir"] + eps)
        if "nir" in df.columns and "swir16" in df.columns:
            df["MSI"] = df["swir16"] / (df["nir"] + eps)
        if "swir16" in df.columns and "swir22" in df.columns:
            df["SWIR_ratio"] = df["swir16"] / (df["swir22"] + eps)
        for col in ["nir", "green", "swir16", "swir22"]:
            if col in df.columns:
                df[col + "_miss"] = df[col].isna().astype(int)

    return train, val


def get_temporal_features(df):
    exclude = set(TARGETS + ["Latitude", "Longitude", "Sample Date", "month", "doy"])
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]


def load_nwu_stations(ec_factor=10.0, tal_factor=1.0, drp_factor=1000.0):
    """
    Load NWU station means in competition units.
    Returns DataFrame with Latitude, Longitude, and target columns.
    """
    nwu = pd.read_csv(DATA_DIR / "external" / "nwu_nearby_water_quality.csv")
    mapping = pd.read_csv(DATA_DIR / "external" / "nearby_stations_mapping.csv")

    # Clean sentinel values
    for col in ["ec_msm", "po4_mgl", "tal_mgl"]:
        nwu.loc[nwu[col] < -9000, col] = np.nan
        nwu.loc[nwu[col] < 0, col] = np.nan

    # Compute station means in NWU units
    station_means = nwu.groupby("station_id").agg({
        "ec_msm": "mean",
        "tal_mgl": "mean",
        "po4_mgl": "mean",
    }).reset_index()

    # Merge with coordinates
    station_means = station_means.merge(
        mapping[["SAMPLE STATION ID", "lat", "lon"]].rename(
            columns={"SAMPLE STATION ID": "station_id"}),
        on="station_id", how="left"
    )

    # Convert to competition units
    result = pd.DataFrame({
        "Latitude": station_means["lat"],
        "Longitude": station_means["lon"],
        "Total Alkalinity": station_means["tal_mgl"] * tal_factor,
        "Electrical Conductance": station_means["ec_msm"] * ec_factor,
        "Dissolved Reactive Phosphorus": station_means["po4_mgl"] * drp_factor,
    }).dropna()

    # Filter extreme outliers (beyond 3x competition training range)
    result.loc[result["Electrical Conductance"] > 5000, "Electrical Conductance"] = np.nan
    result.loc[result["Total Alkalinity"] > 1000, "Total Alkalinity"] = np.nan
    result.loc[result["Dissolved Reactive Phosphorus"] > 600, "Dissolved Reactive Phosphorus"] = np.nan

    return result


def load_nwu_monthly(ec_factor=10.0, tal_factor=1.0, drp_factor=1000.0):
    """
    Load NWU monthly station means in competition units.
    Returns DataFrame with Latitude, Longitude, month, and target columns.
    """
    nwu = pd.read_csv(DATA_DIR / "external" / "nwu_nearby_water_quality.csv")
    mapping = pd.read_csv(DATA_DIR / "external" / "nearby_stations_mapping.csv")

    for col in ["ec_msm", "po4_mgl", "tal_mgl"]:
        nwu.loc[nwu[col] < -9000, col] = np.nan
        nwu.loc[nwu[col] < 0, col] = np.nan

    nwu["date"] = pd.to_datetime(nwu["date"])
    nwu["month"] = nwu["date"].dt.month

    # Monthly station means
    monthly = nwu.groupby(["station_id", "month"]).agg({
        "ec_msm": "mean",
        "tal_mgl": "mean",
        "po4_mgl": "mean",
    }).reset_index()

    monthly = monthly.merge(
        mapping[["SAMPLE STATION ID", "lat", "lon"]].rename(
            columns={"SAMPLE STATION ID": "station_id"}),
        on="station_id", how="left"
    )

    result = pd.DataFrame({
        "Latitude": monthly["lat"],
        "Longitude": monthly["lon"],
        "month": monthly["month"],
        "Total Alkalinity": monthly["tal_mgl"] * tal_factor,
        "Electrical Conductance": monthly["ec_msm"] * ec_factor,
        "Dissolved Reactive Phosphorus": monthly["po4_mgl"] * drp_factor,
    }).dropna(subset=["Latitude", "Longitude"])

    return result


def idw_from_stations(station_df, val_coords, target, k=10, power=2):
    """IDW prediction using arbitrary station dataframe."""
    valid = station_df.dropna(subset=[target])
    if len(valid) == 0:
        return np.full(len(val_coords), np.nan)

    coords = np.radians(valid[["Latitude", "Longitude"]].values)
    vals = valid[target].values
    val_rad = np.radians(val_coords)

    tree = cKDTree(coords)
    actual_k = min(k, len(valid))
    dists, idxs = tree.query(val_rad, k=actual_k)
    dists_km = dists * 6371

    weights = 1.0 / (dists_km ** power + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(weights * vals[idxs], axis=1)


def idw_leave_one_out(station_df, target, k=10, power=2):
    """Leave-one-station-out IDW. station_df must have Latitude, Longitude, target."""
    stations = station_df.dropna(subset=[target])
    coords = stations[["Latitude", "Longitude"]].values
    vals = stations[target].values
    n = len(stations)

    baselines = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        other_coords = np.radians(coords[mask])
        other_vals = vals[mask]

        this_coord = np.radians(coords[i:i + 1])
        tree = cKDTree(other_coords)
        actual_k = min(k, len(other_coords))
        dists, idxs = tree.query(this_coord, k=actual_k)
        dists_km = dists * 6371

        weights = 1.0 / (dists_km ** power + 1e-6)
        weights = weights / weights.sum()
        baselines[i] = np.sum(weights * other_vals[idxs])

    return baselines


def expanded_station_means(train_df, nwu_stations):
    """
    Combine original training station means with NWU station means.
    If a station exists in both, average the means.
    """
    # Original training station means
    orig = train_df.groupby(["Latitude", "Longitude"])[TARGETS].mean().reset_index()

    # Check for overlap (same location in both datasets)
    combined = pd.concat([orig, nwu_stations[["Latitude", "Longitude"] + TARGETS]],
                         ignore_index=True)

    # Group by location to merge any overlapping stations
    combined = combined.groupby(
        [combined["Latitude"].round(3), combined["Longitude"].round(3)]
    )[TARGETS].mean().reset_index()

    return combined


# ============================================================
# v30: Expanded IDW + ML residual (v21 structure with NWU stations)
# ============================================================
def approach_expanded_hybrid(train_df, val_df, nwu_stations):
    """
    Same structure as v21 (50/50 blend of optimized hybrid + temporal ML),
    but with NWU stations added to the IDW pool.
    """
    print("\n" + "=" * 80)
    print("v30: EXPANDED IDW HYBRID (NWU enriched)")
    print("=" * 80)

    temporal_feats = get_temporal_features(train_df)
    temporal_feats = [f for f in temporal_feats if f in val_df.columns]
    val_coords = val_df[["Latitude", "Longitude"]].values

    # Build expanded station pool
    expanded = expanded_station_means(train_df, nwu_stations)
    n_orig = train_df.groupby(["Latitude", "Longitude"]).ngroups
    print("  Original stations: %d, NWU stations: %d, Combined: %d" % (
        n_orig, len(nwu_stations), len(expanded)))

    predictions = {}

    for target in TARGETS:
        print("\n--- %s ---" % target)

        # Sweep k and power on expanded station pool
        best_score = -999
        best_k, best_power = 10, 2

        expanded_clean = expanded.dropna(subset=[target])

        for k in [3, 5, 10, 20, 50]:
            for power in [1, 2, 3]:
                # LOO IDW on expanded stations
                loo_pred = idw_leave_one_out(expanded_clean, target, k=k, power=power)
                score = r2_score(expanded_clean[target], loo_pred)
                if score > best_score:
                    best_score = score
                    best_k, best_power = k, power

        print("  Best IDW: k=%d, power=%d (LOO R2=%.4f)" % (best_k, best_power, best_score))

        # IDW baseline for validation (from expanded pool)
        val_baseline = idw_from_stations(expanded_clean, val_coords, target,
                                         k=best_k, power=best_power)

        # IDW baseline for training (leave-one-station-out from ORIGINAL only)
        orig_stations = train_df.groupby(["Latitude", "Longitude"])[target].mean().reset_index()
        train_loo = idw_leave_one_out(orig_stations, target, k=min(best_k, len(orig_stations)-1),
                                      power=best_power)

        # Map back to training rows
        train_baseline = np.zeros(len(train_df))
        for i, (_, st) in enumerate(orig_stations.iterrows()):
            row_mask = ((train_df["Latitude"] == st["Latitude"]) &
                        (train_df["Longitude"] == st["Longitude"]))
            train_baseline[row_mask.values] = train_loo[i]

        residuals = train_df[target] - train_baseline

        # Train residual model
        X_tr = train_df[temporal_feats].fillna(train_df[temporal_feats].median())
        X_va = val_df[temporal_feats].fillna(train_df[temporal_feats].median())

        # CV residual model
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for tr_idx, va_idx in kf.split(X_tr):
            rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                                       min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf.fit(X_tr.iloc[tr_idx], residuals.iloc[tr_idx])
            hybrid = train_baseline[va_idx] + rf.predict(X_tr.iloc[va_idx])
            cv_scores.append(r2_score(train_df[target].iloc[va_idx], hybrid))
        print("  CV R2 (expanded hybrid): %.4f" % np.mean(cv_scores))

        # Train final residual model
        rf = RandomForestRegressor(n_estimators=500, max_depth=10,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_tr, residuals)
        res_pred = rf.predict(X_va)

        hybrid_pred = np.maximum(val_baseline + res_pred, 0)

        # Temporal ML prediction (same as v21)
        try:
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            xgb = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
                               reg_alpha=0.5, reg_lambda=3.0, tree_method="gpu_hist",
                               device="cuda", random_state=42)
            lgbm = LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
                                 reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1)
            xgb.fit(X_tr, train_df[target])
            lgbm.fit(X_tr, train_df[target])
            rf2 = RandomForestRegressor(n_estimators=500, max_depth=15,
                                        min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf2.fit(X_tr, train_df[target])
            ml_pred = (rf2.predict(X_va) + xgb.predict(X_va) + lgbm.predict(X_va)) / 3
        except ImportError:
            print("  XGBoost/LightGBM not available, using RF only for ML")
            rf2 = RandomForestRegressor(n_estimators=500, max_depth=15,
                                        min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf2.fit(X_tr, train_df[target])
            ml_pred = rf2.predict(X_va)

        # 50/50 blend (same as v21)
        blended = 0.5 * hybrid_pred + 0.5 * ml_pred
        predictions[target] = np.maximum(blended, 0)

        print("  hybrid_mean=%.1f, ml_mean=%.1f, blend_mean=%.1f" % (
            hybrid_pred.mean(), ml_pred.mean(), blended.mean()))

    create_submission(val_df, predictions, version="v30_expanded_hybrid")
    return predictions


# ============================================================
# v31: Sweep DRP conversion factor
# ============================================================
def approach_drp_sweep(train_df, val_df):
    """
    The DRP conversion from NWU PO4 is the most uncertain.
    Try multiple conversion factors and generate separate submissions.
    Also try EC factor variations.
    """
    print("\n" + "=" * 80)
    print("v31: DRP CONVERSION FACTOR SWEEP")
    print("=" * 80)

    temporal_feats = get_temporal_features(train_df)
    temporal_feats = [f for f in temporal_feats if f in val_df.columns]
    val_coords = val_df[["Latitude", "Longitude"]].values

    # Load NWU overlap data for calibration
    nwu_overlap = pd.read_csv(DATA_DIR / "external" / "nwu_train_overlap_wq.csv")
    for col in ["ec_msm", "tal_mgl", "po4_mgl"]:
        nwu_overlap.loc[nwu_overlap[col] < -9000, col] = np.nan
        nwu_overlap.loc[nwu_overlap[col] < 0, col] = np.nan

    train_nwu_map = pd.read_csv(DATA_DIR / "external" / "train_nwu_station_mapping.csv")

    # Find optimal conversion factors by minimizing error on overlapping stations
    print("\n--- Finding optimal conversion factors ---")

    for target in TARGETS:
        if target == "Electrical Conductance":
            nwu_col = "ec_msm"
            test_factors = [8, 9, 10, 11, 12]
        elif target == "Total Alkalinity":
            nwu_col = "tal_mgl"
            test_factors = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
        else:
            nwu_col = "po4_mgl"
            test_factors = [200, 300, 400, 500, 600, 800, 1000]

        best_factor = None
        best_error = 999

        for factor in test_factors:
            errors = []
            for _, m in train_nwu_map.iterrows():
                sid = m["nwu_sid"]
                stn_data = nwu_overlap[nwu_overlap["station_id"] == sid]
                if len(stn_data) == 0:
                    continue
                nwu_val = stn_data[nwu_col].dropna().mean()
                if np.isnan(nwu_val):
                    continue

                row_mask = (
                    (abs(train_df["Latitude"] - m["train_lat"]) < 0.001) &
                    (abs(train_df["Longitude"] - m["train_lon"]) < 0.001)
                )
                train_val = train_df.loc[row_mask, target].mean()
                if np.isnan(train_val):
                    continue

                errors.append((nwu_val * factor - train_val) ** 2)

            rmse = np.sqrt(np.mean(errors))
            if rmse < best_error:
                best_error = rmse
                best_factor = factor

        print("  %s: best_factor=%.1f (RMSE=%.2f)" % (target, best_factor, best_error))

    # Use optimized factors
    # For now, use the sweep results to generate submissions with different DRP factors
    for drp_factor in [300, 500, 700, 1000]:
        nwu_stations = load_nwu_stations(
            ec_factor=10.0, tal_factor=1.0, drp_factor=drp_factor)

        expanded = expanded_station_means(train_df, nwu_stations)

        preds = {}
        for target in TARGETS:
            preds[target] = idw_from_stations(
                expanded.dropna(subset=[target]),
                val_coords, target, k=5, power=2
            )
            preds[target] = np.maximum(preds[target], 0)

        version = "v31_drp%d" % drp_factor
        print("  DRP_factor=%d: TAL=%.1f, EC=%.1f, DRP=%.1f" % (
            drp_factor, preds["Total Alkalinity"].mean(),
            preds["Electrical Conductance"].mean(),
            preds["Dissolved Reactive Phosphorus"].mean()))
        create_submission(val_df, preds, version=version)


# ============================================================
# v32: Per-target best source
# ============================================================
def approach_per_target(train_df, val_df, nwu_stations):
    """
    Use the best approach per target:
    - TAL: NWU-enriched IDW (strong signal, R2=0.90 calibration)
    - EC: Original IDW only (NWU EC conversion is noisy)
    - DRP: Original IDW only (NWU DRP conversion very uncertain)

    Blend with temporal ML for all three.
    """
    print("\n" + "=" * 80)
    print("v32: PER-TARGET BEST SOURCE")
    print("=" * 80)

    temporal_feats = get_temporal_features(train_df)
    temporal_feats = [f for f in temporal_feats if f in val_df.columns]
    val_coords = val_df[["Latitude", "Longitude"]].values

    # Original station means
    orig_stations = train_df.groupby(["Latitude", "Longitude"])[TARGETS].mean().reset_index()

    # Expanded station means (for TAL only)
    expanded = expanded_station_means(train_df, nwu_stations)

    predictions = {}

    for target in TARGETS:
        print("\n--- %s ---" % target)

        # Choose station pool based on target
        if target == "Total Alkalinity":
            # Use expanded pool (NWU TAL is well-calibrated)
            pool = expanded.dropna(subset=[target])
            print("  Using EXPANDED station pool (%d stations)" % len(pool))
        else:
            # Use original pool only (NWU EC/DRP conversion too noisy)
            pool = orig_stations.dropna(subset=[target])
            print("  Using ORIGINAL station pool (%d stations)" % len(pool))

        # IDW from chosen pool
        val_baseline = idw_from_stations(pool, val_coords, target, k=10, power=2)

        # Train IDW baseline (always from original, for residual training)
        train_loo = idw_leave_one_out(orig_stations, target, k=10, power=2)
        train_baseline = np.zeros(len(train_df))
        for i, (_, st) in enumerate(orig_stations.iterrows()):
            row_mask = ((train_df["Latitude"] == st["Latitude"]) &
                        (train_df["Longitude"] == st["Longitude"]))
            train_baseline[row_mask.values] = train_loo[i]

        residuals = train_df[target] - train_baseline

        X_tr = train_df[temporal_feats].fillna(train_df[temporal_feats].median())
        X_va = val_df[temporal_feats].fillna(train_df[temporal_feats].median())

        # Train residual model
        rf = RandomForestRegressor(n_estimators=500, max_depth=10,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_tr, residuals)
        res_pred = rf.predict(X_va)

        hybrid_pred = np.maximum(val_baseline + res_pred, 0)

        # Temporal ML
        try:
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            xgb = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
                               reg_alpha=0.5, reg_lambda=3.0, tree_method="gpu_hist",
                               device="cuda", random_state=42)
            lgbm = LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
                                 reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1)
            xgb.fit(X_tr, train_df[target])
            lgbm.fit(X_tr, train_df[target])
            rf2 = RandomForestRegressor(n_estimators=500, max_depth=15,
                                        min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf2.fit(X_tr, train_df[target])
            ml_pred = (rf2.predict(X_va) + xgb.predict(X_va) + lgbm.predict(X_va)) / 3
        except ImportError:
            rf2 = RandomForestRegressor(n_estimators=500, max_depth=15,
                                        min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf2.fit(X_tr, train_df[target])
            ml_pred = rf2.predict(X_va)

        blended = 0.5 * hybrid_pred + 0.5 * ml_pred
        predictions[target] = np.maximum(blended, 0)
        print("  hybrid_mean=%.1f, ml_mean=%.1f, blend_mean=%.1f" % (
            hybrid_pred.mean(), ml_pred.mean(), blended.mean()))

    create_submission(val_df, predictions, version="v32_per_target_source")
    return predictions


# ============================================================
# v33: Multiple blend ratios
# ============================================================
def approach_blend_sweep(train_df, val_df, nwu_stations):
    """
    Try different blend ratios between expanded-IDW hybrid and temporal ML.
    v21 used 50/50 — maybe NWU-enriched IDW deserves higher weight.
    """
    print("\n" + "=" * 80)
    print("v33: BLEND RATIO SWEEP")
    print("=" * 80)

    temporal_feats = get_temporal_features(train_df)
    temporal_feats = [f for f in temporal_feats if f in val_df.columns]
    val_coords = val_df[["Latitude", "Longitude"]].values

    expanded = expanded_station_means(train_df, nwu_stations)
    orig_stations = train_df.groupby(["Latitude", "Longitude"])[TARGETS].mean().reset_index()

    X_tr = train_df[temporal_feats].fillna(train_df[temporal_feats].median())
    X_va = val_df[temporal_feats].fillna(train_df[temporal_feats].median())

    # Pre-compute ML predictions for all targets
    ml_preds = {}
    hybrid_preds = {}

    for target in TARGETS:
        print("\n--- %s ---" % target)

        # Expanded IDW hybrid
        val_baseline = idw_from_stations(expanded.dropna(subset=[target]),
                                         val_coords, target, k=10, power=2)

        train_loo = idw_leave_one_out(orig_stations, target, k=10, power=2)
        train_baseline = np.zeros(len(train_df))
        for i, (_, st) in enumerate(orig_stations.iterrows()):
            row_mask = ((train_df["Latitude"] == st["Latitude"]) &
                        (train_df["Longitude"] == st["Longitude"]))
            train_baseline[row_mask.values] = train_loo[i]

        residuals = train_df[target] - train_baseline
        rf = RandomForestRegressor(n_estimators=500, max_depth=10,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_tr, residuals)
        hybrid_preds[target] = np.maximum(val_baseline + rf.predict(X_va), 0)

        # Temporal ML
        try:
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor
            xgb = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
                               reg_alpha=0.5, reg_lambda=3.0, tree_method="gpu_hist",
                               device="cuda", random_state=42)
            lgbm = LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
                                 reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1)
            xgb.fit(X_tr, train_df[target])
            lgbm.fit(X_tr, train_df[target])
            rf2 = RandomForestRegressor(n_estimators=500, max_depth=15,
                                        min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf2.fit(X_tr, train_df[target])
            ml_preds[target] = (rf2.predict(X_va) + xgb.predict(X_va) + lgbm.predict(X_va)) / 3
        except ImportError:
            rf2 = RandomForestRegressor(n_estimators=500, max_depth=15,
                                        min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf2.fit(X_tr, train_df[target])
            ml_preds[target] = rf2.predict(X_va)

        print("  hybrid_mean=%.1f, ml_mean=%.1f" % (
            hybrid_preds[target].mean(), ml_preds[target].mean()))

    # Generate submissions at different blend ratios
    for hybrid_weight in [0.3, 0.5, 0.7]:
        preds = {}
        for target in TARGETS:
            blended = hybrid_weight * hybrid_preds[target] + (1 - hybrid_weight) * ml_preds[target]
            preds[target] = np.maximum(blended, 0)

        version = "v33_blend_%d%d" % (int(hybrid_weight * 100), int((1 - hybrid_weight) * 100))
        create_submission(val_df, preds, version=version)
        print("  %s: TAL=%.1f, EC=%.1f, DRP=%.1f" % (
            version, preds["Total Alkalinity"].mean(),
            preds["Electrical Conductance"].mean(),
            preds["Dissolved Reactive Phosphorus"].mean()))


if __name__ == "__main__":
    print("=" * 80)
    print("TRAIN v8 — NWU AS ADDITIONAL IDW STATIONS")
    print("=" * 80)

    train_df, val_df = load_data()
    print("Training: %d rows, %d features" % (len(train_df), len(train_df.columns)))
    print("Validation: %d rows" % len(val_df))

    # Load NWU stations with simple conversion factors
    nwu_stations = load_nwu_stations(ec_factor=10.0, tal_factor=1.0, drp_factor=1000.0)
    print("NWU stations: %d (after cleaning)" % len(nwu_stations))

    # Find optimal conversion factors
    approach_drp_sweep(train_df, val_df)    # v31 (also prints optimal factors)

    # Run main approaches
    approach_expanded_hybrid(train_df, val_df, nwu_stations)  # v30
    approach_per_target(train_df, val_df, nwu_stations)       # v32
    approach_blend_sweep(train_df, val_df, nwu_stations)      # v33

    print("\n" + "=" * 80)
    print("ALL SUBMISSIONS GENERATED")
    print("=" * 80)
    print("  v30_expanded_hybrid.csv     — v21 with NWU-enriched IDW pool")
    print("  v31_drp*.csv                — DRP factor sweep")
    print("  v32_per_target_source.csv   — NWU for TAL only, original for EC/DRP")
    print("  v33_blend_*.csv             — Blend ratio sweep")
    print()
    print("SUBMIT ORDER:")
    print("  1. v30 (most likely to beat v21's 0.258)")
    print("  2. v32 (safe: only uses NWU for well-calibrated TAL)")
    print("  3. v33_blend_3070 (lower hybrid weight = less NWU influence)")
