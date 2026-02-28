"""
Training pipeline v5 — Hybrid spatial interpolation + temporal ML.

Key insight from leaderboard:
    - Temporal features only (v12): R² = 0.20 (matches benchmark)
    - Static features hurt: R² = -0.31
    - To reach 0.91, we need to capture BOTH spatial and temporal variance

Strategy:
    Decompose prediction into: y = spatial_baseline + temporal_deviation

    1. Spatial baseline: IDW interpolation from nearby training stations
       (captures between-station variance without overfitting)
    2. Temporal deviation: ML model predicts how each observation deviates
       from the station baseline, using Landsat + climate features
    3. Final = spatial_baseline + temporal_deviation

Also tries:
    - Blending spatial interpolation with ML predictions
    - Target-aware feature engineering
    - Optimized ensemble weights

Usage:
    python src/train_v5.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from submission import create_submission


TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
DATA_DIR = ROOT_DIR / "datasets"


def load_data():
    """Load and merge all temporal features."""
    train = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
    train["Sample Date"] = pd.to_datetime(train["Sample Date"], dayfirst=True)
    val = pd.read_csv(DATA_DIR / "submission_template.csv")
    val["Sample Date"] = pd.to_datetime(val["Sample Date"], dayfirst=True)

    # Landsat
    landsat_tr = pd.read_csv(DATA_DIR / "train_landsat_features.csv")
    landsat_tr["Sample Date"] = pd.to_datetime(landsat_tr["Sample Date"], dayfirst=True)
    landsat_va = pd.read_csv(DATA_DIR / "val_landsat_features.csv")
    landsat_va["Sample Date"] = pd.to_datetime(landsat_va["Sample Date"], dayfirst=True)

    # TerraClimate
    tc_tr = pd.read_csv(DATA_DIR / "train_terraclimate_features.csv")
    tc_tr["Sample Date"] = pd.to_datetime(tc_tr["Sample Date"], dayfirst=True)
    tc_va = pd.read_csv(DATA_DIR / "val_terraclimate_features.csv")
    tc_va["Sample Date"] = pd.to_datetime(tc_va["Sample Date"], dayfirst=True)

    # Extended TerraClimate
    ext_path = DATA_DIR / "processed"
    if (ext_path / "train_terraclimate_extended.csv").exists():
        tc_ext_tr = pd.read_csv(ext_path / "train_terraclimate_extended.csv")
        tc_ext_tr["Sample Date"] = pd.to_datetime(tc_ext_tr["Sample Date"], format="mixed", dayfirst=True)
        tc_ext_va = pd.read_csv(ext_path / "val_terraclimate_extended.csv")
        tc_ext_va["Sample Date"] = pd.to_datetime(tc_ext_va["Sample Date"], format="mixed", dayfirst=True)
    else:
        tc_ext_tr = None
        tc_ext_va = None

    join = ["Latitude", "Longitude", "Sample Date"]
    train = train.merge(landsat_tr, on=join, how="left")
    train = train.merge(tc_tr, on=join, how="left")
    val = val.merge(landsat_va, on=join, how="left")
    val = val.merge(tc_va, on=join, how="left")

    if tc_ext_tr is not None:
        ext_cols = [c for c in tc_ext_tr.columns if c not in ["pet"] or c in join]
        train = train.merge(tc_ext_tr[ext_cols], on=join, how="left")
        val = val.merge(tc_ext_va[ext_cols], on=join, how="left")

    # Add temporal encoding
    for df in [train, val]:
        df["month"] = df["Sample Date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy"] = df["Sample Date"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
        df["is_wet"] = ((df["month"] >= 10) | (df["month"] <= 3)).astype(int)

    # Add Landsat indices
    eps = 1e-10
    for df in [train, val]:
        if "green" in df.columns and "nir" in df.columns:
            df["NDWI"] = (df["green"] - df["nir"]) / (df["green"] + df["nir"] + eps)
        if "nir" in df.columns and "swir16" in df.columns:
            df["MSI"] = df["swir16"] / (df["nir"] + eps)
        if "swir16" in df.columns and "swir22" in df.columns:
            df["SWIR_ratio"] = df["swir16"] / (df["swir22"] + eps)
        for col in ["nir", "green", "swir16", "swir22"]:
            if col in df.columns:
                df[f"{col}_miss"] = df[col].isna().astype(int)

    return train, val


def get_temporal_features(df):
    """Get list of temporal (non-static, non-target) features."""
    exclude = set(TARGETS + ["Latitude", "Longitude", "Sample Date", "month", "doy"])
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]


def compute_idw(train_stations, val_coords, target, k=10, power=2):
    """Inverse Distance Weighting prediction."""
    train_coords = np.radians(train_stations[["Latitude", "Longitude"]].values)
    val_coords_rad = np.radians(val_coords)

    tree = cKDTree(train_coords)
    dists, idxs = tree.query(val_coords_rad, k=k)
    dists_km = dists * 6371

    weights = 1.0 / (dists_km ** power + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    vals = train_stations[target].values
    return np.sum(weights * vals[idxs], axis=1)


def compute_station_baseline(train_df, val_df, target, k=10, power=2):
    """Compute spatial baseline for each validation point using IDW."""
    station_stats = train_df.groupby(["Latitude", "Longitude"])[target].agg(
        ["mean", "median", "std", "count"]
    ).reset_index()

    val_coords = val_df[["Latitude", "Longitude"]].values

    # IDW of station means
    baseline = compute_idw(
        station_stats.rename(columns={"mean": target}),
        val_coords, target, k=k, power=power
    )
    return baseline


def compute_station_baseline_train(train_df, target, k=10, power=2):
    """
    Compute spatial baseline for training points using leave-one-station-out IDW.
    This prevents data leakage — each station's baseline is computed
    from OTHER stations only.
    """
    stations = train_df.groupby(["Latitude", "Longitude"])[target].mean().reset_index()
    station_coords = stations[["Latitude", "Longitude"]].values
    station_vals = stations[target].values

    # Map each training row to its station
    train_df = train_df.copy()
    baselines = np.zeros(len(train_df))

    for i, (lat, lon) in enumerate(station_coords):
        # Other stations (leave this one out)
        mask = np.ones(len(stations), dtype=bool)
        mask[i] = False
        other_coords = np.radians(station_coords[mask])
        other_vals = station_vals[mask]

        this_coord = np.radians([[lat, lon]])
        tree = cKDTree(other_coords)
        actual_k = min(k, len(other_coords))
        dists, idxs = tree.query(this_coord, k=actual_k)
        dists_km = dists * 6371

        weights = 1.0 / (dists_km ** power + 1e-6)
        weights = weights / weights.sum()
        baseline_val = np.sum(weights * other_vals[idxs])

        # Assign to all rows at this station
        row_mask = (train_df["Latitude"] == lat) & (train_df["Longitude"] == lon)
        baselines[row_mask.values] = baseline_val

    return baselines


# ============================================================
# Approach A: Hybrid — spatial baseline + temporal residual
# ============================================================
def approach_hybrid(train_df, val_df):
    """
    Step 1: Spatial baseline from IDW of nearby station means
    Step 2: ML predicts temporal deviation from baseline
    Step 3: Final = baseline + deviation
    """
    from xgboost import XGBRegressor

    print("\n" + "=" * 80)
    print("APPROACH A: HYBRID (spatial IDW + temporal ML residual)")
    print("=" * 80)

    temporal_feats = get_temporal_features(train_df)
    temporal_feats = [f for f in temporal_feats if f in val_df.columns]

    predictions = {}

    for target in TARGETS:
        print(f"\n--- {target} ---")

        for k in [5, 10, 20, 50]:
            # Compute baselines
            train_baseline = compute_station_baseline_train(train_df, target, k=k)
            val_baseline = compute_station_baseline(train_df, val_df, target, k=k)

            # Compute residuals
            residuals = train_df[target] - train_baseline

            # Train ML on residuals
            X_tr = train_df[temporal_feats].fillna(train_df[temporal_feats].median())
            X_va = val_df[temporal_feats].fillna(train_df[temporal_feats].median())

            # CV the residual model
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores_hybrid = []
            scores_idw_only = []

            for tr_idx, va_idx in kf.split(X_tr):
                # IDW-only score
                idw_pred = train_baseline[va_idx]
                scores_idw_only.append(r2_score(train_df[target].iloc[va_idx], idw_pred))

                # Hybrid score
                m = RandomForestRegressor(n_estimators=200, max_depth=10,
                                           min_samples_leaf=5, random_state=42, n_jobs=-1)
                m.fit(X_tr.iloc[tr_idx], residuals.iloc[tr_idx])
                res_pred = m.predict(X_tr.iloc[va_idx])
                hybrid_pred = idw_pred + res_pred
                scores_hybrid.append(r2_score(train_df[target].iloc[va_idx], hybrid_pred))

            print(f"  k={k:3d}: IDW_only = {np.mean(scores_idw_only):.4f}, "
                  f"Hybrid = {np.mean(scores_hybrid):.4f}")

        # Use best k (try k=10 as default)
        best_k = 10
        train_baseline = compute_station_baseline_train(train_df, target, k=best_k)
        val_baseline = compute_station_baseline(train_df, val_df, target, k=best_k)
        residuals = train_df[target] - train_baseline

        X_tr = train_df[temporal_feats].fillna(train_df[temporal_feats].median())
        X_va = val_df[temporal_feats].fillna(train_df[temporal_feats].median())

        m = RandomForestRegressor(n_estimators=500, max_depth=10,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
        m.fit(X_tr, residuals)
        res_pred = m.predict(X_va)

        predictions[target] = np.maximum(val_baseline + res_pred, 0)
        print(f"  Final: baseline_mean={val_baseline.mean():.1f}, "
              f"residual_mean={res_pred.mean():.1f}, "
              f"pred_mean={predictions[target].mean():.1f}")

    create_submission(val_df, predictions, version="v15_hybrid_idw_ml")
    return predictions


# ============================================================
# Approach B: Blended spatial + temporal
# ============================================================
def approach_blended(train_df, val_df):
    """
    Simple blend: alpha * spatial_prediction + (1 - alpha) * ML_prediction
    Find optimal alpha per target.
    """
    from xgboost import XGBRegressor

    print("\n" + "=" * 80)
    print("APPROACH B: BLENDED (alpha * IDW + (1-alpha) * ML)")
    print("=" * 80)

    temporal_feats = get_temporal_features(train_df)
    temporal_feats = [f for f in temporal_feats if f in val_df.columns]

    X_tr = train_df[temporal_feats].fillna(train_df[temporal_feats].median())
    X_va = val_df[temporal_feats].fillna(train_df[temporal_feats].median())

    predictions = {}

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")

        # IDW predictions (leave-one-station-out for training)
        idw_train = compute_station_baseline_train(train_df, target, k=10)
        idw_val = compute_station_baseline(train_df, val_df, target, k=10)

        # ML predictions via CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        best_alpha = 0
        best_score = -999

        for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            scores = []
            for tr_idx, va_idx in kf.split(X_tr):
                # ML part
                m = RandomForestRegressor(n_estimators=200, max_depth=12,
                                           min_samples_leaf=5, random_state=42, n_jobs=-1)
                m.fit(X_tr.iloc[tr_idx], y.iloc[tr_idx])
                ml_pred = m.predict(X_tr.iloc[va_idx])

                # Blend
                blended = alpha * idw_train[va_idx] + (1 - alpha) * ml_pred
                scores.append(r2_score(y.iloc[va_idx], blended))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

        print(f"  Best alpha = {best_alpha:.1f} (IDW weight), CV R² = {best_score:.4f}")

        # Train final ML on all data
        m = RandomForestRegressor(n_estimators=500, max_depth=12,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
        m.fit(X_tr, y)
        ml_val = m.predict(X_va)

        pred = best_alpha * idw_val + (1 - best_alpha) * ml_val
        predictions[target] = np.maximum(pred, 0)
        print(f"  IDW_mean={idw_val.mean():.1f}, ML_mean={ml_val.mean():.1f}, "
              f"Blend_mean={pred.mean():.1f}")

    create_submission(val_df, predictions, version="v16_blended")
    return predictions


# ============================================================
# Approach C: Multi-model ensemble with IDW feature
# ============================================================
def approach_idw_as_feature(train_df, val_df):
    """Use IDW prediction as a FEATURE in the ML model."""
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    print("\n" + "=" * 80)
    print("APPROACH C: IDW AS FEATURE IN ML MODEL")
    print("=" * 80)

    temporal_feats = get_temporal_features(train_df)
    temporal_feats = [f for f in temporal_feats if f in val_df.columns]

    predictions = {}

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")

        # IDW features at multiple scales
        for k in [5, 10, 20]:
            idw_tr = compute_station_baseline_train(train_df, target, k=k)
            idw_va = compute_station_baseline(train_df, val_df, target, k=k)
            train_df[f"_idw_{target[:3]}_{k}"] = idw_tr
            val_df[f"_idw_{target[:3]}_{k}"] = idw_va

        idw_feats = [f"_idw_{target[:3]}_{k}" for k in [5, 10, 20]]
        all_feats = temporal_feats + idw_feats
        all_feats = [f for f in all_feats if f in train_df.columns and f in val_df.columns]

        X_tr = train_df[all_feats].fillna(train_df[all_feats].median())
        X_va = val_df[all_feats].fillna(train_df[all_feats].median())

        # Train ensemble
        rf = RandomForestRegressor(n_estimators=500, max_depth=12,
                                    min_samples_leaf=5, random_state=42, n_jobs=-1)
        xgb = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
                           reg_alpha=0.5, reg_lambda=3.0, tree_method="gpu_hist",
                           device="cuda", random_state=42)
        lgbm = LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
                             reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1)

        # CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in kf.split(X_tr):
            rf.fit(X_tr.iloc[tr_idx], y.iloc[tr_idx])
            xgb.fit(X_tr.iloc[tr_idx], y.iloc[tr_idx])
            lgbm.fit(X_tr.iloc[tr_idx], y.iloc[tr_idx])
            p = (rf.predict(X_tr.iloc[va_idx]) +
                 xgb.predict(X_tr.iloc[va_idx]) +
                 lgbm.predict(X_tr.iloc[va_idx])) / 3
            scores.append(r2_score(y.iloc[va_idx], p))
        print(f"  Ensemble with IDW features: random_CV = {np.mean(scores):.4f}")

        # Train final
        rf.fit(X_tr, y)
        xgb.fit(X_tr, y)
        lgbm.fit(X_tr, y)
        pred = (rf.predict(X_va) + xgb.predict(X_va) + lgbm.predict(X_va)) / 3
        predictions[target] = np.maximum(pred, 0)

    # Clean up temp columns
    for target in TARGETS:
        for k in [5, 10, 20]:
            col = f"_idw_{target[:3]}_{k}"
            train_df.drop(columns=[col], inplace=True, errors="ignore")
            val_df.drop(columns=[col], inplace=True, errors="ignore")

    create_submission(val_df, predictions, version="v17_idw_feature")
    return predictions


# ============================================================
# Approach D: Station-aware ensemble (best of everything)
# ============================================================
def approach_station_aware(train_df, val_df):
    """
    For each validation point:
    1. Find K nearest training stations
    2. Use their temporal patterns as reference
    3. Adjust prediction based on seasonal similarity
    """
    from xgboost import XGBRegressor

    print("\n" + "=" * 80)
    print("APPROACH D: STATION-AWARE WITH TEMPORAL MATCHING")
    print("=" * 80)

    station_stats = train_df.groupby(["Latitude", "Longitude"]).agg({
        t: ["mean", "std", "count"] for t in TARGETS
    }).reset_index()
    station_stats.columns = ["_".join(c).strip("_") for c in station_stats.columns]

    train_coords = np.radians(station_stats[["Latitude", "Longitude"]].values)
    val_coords = np.radians(val_df[["Latitude", "Longitude"]].values)

    tree = cKDTree(train_coords)

    predictions = {}

    for target in TARGETS:
        print(f"\n--- {target} ---")

        # For each val point, find monthly patterns at nearest stations
        val_preds = []

        for i in range(len(val_df)):
            val_row = val_df.iloc[i]
            val_month = val_row["Sample Date"].month
            val_coord = np.radians([[val_row["Latitude"], val_row["Longitude"]]])

            # Find nearest stations
            dists, idxs = tree.query(val_coord, k=min(20, len(station_stats)))
            dists_km = dists[0] * 6371

            # Get monthly values from nearest stations
            weighted_val = 0
            total_weight = 0

            for j, idx in enumerate(idxs[0]):
                st_lat = station_stats.iloc[idx]["Latitude"]
                st_lon = station_stats.iloc[idx]["Longitude"]
                dist = dists_km[j]

                # Get samples at this station for same month
                st_mask = (
                    (train_df["Latitude"] == st_lat) &
                    (train_df["Longitude"] == st_lon) &
                    (train_df["Sample Date"].dt.month == val_month)
                )
                st_data = train_df.loc[st_mask, target]

                if len(st_data) == 0:
                    # Fall back to all months at this station
                    st_mask = (
                        (train_df["Latitude"] == st_lat) &
                        (train_df["Longitude"] == st_lon)
                    )
                    st_data = train_df.loc[st_mask, target]

                if len(st_data) > 0:
                    weight = 1.0 / (dist ** 2 + 1)
                    weighted_val += weight * st_data.mean()
                    total_weight += weight

            if total_weight > 0:
                val_preds.append(weighted_val / total_weight)
            else:
                val_preds.append(train_df[target].mean())

        predictions[target] = np.maximum(np.array(val_preds), 0)
        print(f"  pred_mean = {predictions[target].mean():.1f}, "
              f"pred_std = {predictions[target].std():.1f}")

    create_submission(val_df, predictions, version="v18_station_aware")
    return predictions


if __name__ == "__main__":
    train_df, val_df = load_data()
    print(f"Training: {len(train_df)} rows")
    print(f"Validation: {len(val_df)} rows")

    feats = get_temporal_features(train_df)
    print(f"Temporal features: {len(feats)}")

    approach_hybrid(train_df, val_df)        # v15
    approach_blended(train_df, val_df)       # v16
    approach_idw_as_feature(train_df, val_df)  # v17
    approach_station_aware(train_df, val_df)  # v18

    print("\n" + "=" * 80)
    print("ALL SUBMISSIONS GENERATED")
    print("=" * 80)
    print("  v15_hybrid_idw_ml.csv     — Spatial baseline + ML residual")
    print("  v16_blended.csv           — Optimal alpha blend (IDW + ML)")
    print("  v17_idw_feature.csv       — IDW as feature in ensemble")
    print("  v18_station_aware.csv     — Monthly-aware spatial interpolation")
    print("\nSubmit v17 first (IDW as feature), then v16 (blended).")
