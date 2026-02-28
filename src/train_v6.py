"""
Training pipeline v6 — Optimized hybrid spatial + temporal.

v15 scored 0.249 on leaderboard using k=10 IDW + RF residual.
But CV showed k=50 was better for all targets. This script optimizes:
    1. k value per target
    2. IDW power parameter
    3. Residual model type and hyperparameters
    4. Ensemble of multiple hybrid variants
    5. Blending with pure temporal ML

Usage:
    python src/train_v6.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from submission import create_submission

TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
DATA_DIR = ROOT_DIR / "datasets"


def load_data():
    """Load and merge temporal features."""
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
                df[f"{col}_miss"] = df[col].isna().astype(int)

    return train, val


def get_temporal_features(df):
    exclude = set(TARGETS + ["Latitude", "Longitude", "Sample Date", "month", "doy"])
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]


def idw_leave_one_out(train_df, target, k=10, power=2):
    """Leave-one-station-out IDW for training data."""
    stations = train_df.groupby(["Latitude", "Longitude"])[target].mean().reset_index()
    station_coords = stations[["Latitude", "Longitude"]].values
    station_vals = stations[target].values

    baselines = np.zeros(len(train_df))

    for i in range(len(stations)):
        mask = np.ones(len(stations), dtype=bool)
        mask[i] = False
        other_coords = np.radians(station_coords[mask])
        other_vals = station_vals[mask]

        this_coord = np.radians(station_coords[i:i+1])
        tree = cKDTree(other_coords)
        actual_k = min(k, len(other_coords))
        dists, idxs = tree.query(this_coord, k=actual_k)
        dists_km = dists * 6371

        weights = 1.0 / (dists_km ** power + 1e-6)
        weights = weights / weights.sum()
        baseline_val = np.sum(weights * other_vals[idxs])

        row_mask = ((train_df["Latitude"] == station_coords[i, 0]) &
                    (train_df["Longitude"] == station_coords[i, 1]))
        baselines[row_mask.values] = baseline_val

    return baselines


def idw_predict(train_df, val_coords, target, k=10, power=2):
    """IDW prediction for validation points."""
    stations = train_df.groupby(["Latitude", "Longitude"])[target].mean().reset_index()
    train_coords = np.radians(stations[["Latitude", "Longitude"]].values)
    val_rad = np.radians(val_coords)

    tree = cKDTree(train_coords)
    dists, idxs = tree.query(val_rad, k=k)
    dists_km = dists * 6371

    weights = 1.0 / (dists_km ** power + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)
    vals = stations[target].values
    return np.sum(weights * vals[idxs], axis=1)


def idw_monthly(train_df, val_df, target, k=10, power=2):
    """Month-aware IDW: use monthly station means if available."""
    val_coords = val_df[["Latitude", "Longitude"]].values
    val_months = val_df["Sample Date"].dt.month.values

    # Station-month means
    station_month = train_df.groupby(["Latitude", "Longitude", "month"])[target].mean().reset_index()
    # Station overall means (fallback)
    station_all = train_df.groupby(["Latitude", "Longitude"])[target].mean().reset_index()

    all_coords = np.radians(station_all[["Latitude", "Longitude"]].values)
    tree = cKDTree(all_coords)

    preds = np.zeros(len(val_df))

    for i in range(len(val_df)):
        val_coord = np.radians(val_coords[i:i+1])
        dists, idxs = tree.query(val_coord, k=min(k, len(all_coords)))
        dists_km = dists[0] * 6371

        weighted_val = 0
        total_weight = 0

        for j, idx in enumerate(idxs[0]):
            st_lat = station_all.iloc[idx]["Latitude"]
            st_lon = station_all.iloc[idx]["Longitude"]

            # Try month-specific value
            month_match = station_month[
                (station_month["Latitude"] == st_lat) &
                (station_month["Longitude"] == st_lon) &
                (station_month["month"] == val_months[i])
            ]

            if len(month_match) > 0:
                val = month_match[target].values[0]
            else:
                val = station_all.iloc[idx][target]

            weight = 1.0 / (dists_km[j] ** power + 1e-6)
            weighted_val += weight * val
            total_weight += weight

        preds[i] = weighted_val / total_weight if total_weight > 0 else train_df[target].mean()

    return preds


# ============================================================
# Sweep: Find optimal k, power, and residual model per target
# ============================================================
def optimize_hybrid(train_df):
    """Find best k, power, and residual model for each target."""
    temporal_feats = get_temporal_features(train_df)

    print("=" * 80)
    print("OPTIMIZATION SWEEP")
    print("=" * 80)

    best_config = {}

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")

        best_score = -999
        best_params = {}

        # Sweep k and power
        for k in [5, 10, 20, 30, 50, 80]:
            for power in [1, 1.5, 2, 3]:
                baseline = idw_leave_one_out(train_df, target, k=k, power=power)
                residuals = y - baseline

                X = train_df[temporal_feats].fillna(train_df[temporal_feats].median())
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = []

                for tr_idx, va_idx in kf.split(X):
                    m = RandomForestRegressor(n_estimators=200, max_depth=10,
                                               min_samples_leaf=5, random_state=42, n_jobs=-1)
                    m.fit(X.iloc[tr_idx], residuals.iloc[tr_idx])
                    hybrid = baseline[va_idx] + m.predict(X.iloc[va_idx])
                    scores.append(r2_score(y.iloc[va_idx], hybrid))

                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {"k": k, "power": power}

        print(f"  Best IDW: k={best_params['k']}, power={best_params['power']}, CV={best_score:.4f}")

        # Now sweep residual model with best IDW params
        baseline = idw_leave_one_out(train_df, target, **best_params)
        residuals = y - baseline
        X = train_df[temporal_feats].fillna(train_df[temporal_feats].median())

        residual_models = {
            "rf_d10": lambda: RandomForestRegressor(
                n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
            "rf_d15": lambda: RandomForestRegressor(
                n_estimators=300, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1),
            "rf_d8": lambda: RandomForestRegressor(
                n_estimators=500, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1),
            "et_d10": lambda: ExtraTreesRegressor(
                n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
            "et_d15": lambda: ExtraTreesRegressor(
                n_estimators=300, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1),
            "gbr_d5": lambda: GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                min_samples_leaf=10, random_state=42),
        }

        best_model_name = None
        best_model_score = -999

        for name, factory in residual_models.items():
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for tr_idx, va_idx in kf.split(X):
                m = factory()
                m.fit(X.iloc[tr_idx], residuals.iloc[tr_idx])
                hybrid = baseline[va_idx] + m.predict(X.iloc[va_idx])
                scores.append(r2_score(y.iloc[va_idx], hybrid))
            mean_score = np.mean(scores)
            print(f"    {name:15s} → CV = {mean_score:.4f}")
            if mean_score > best_model_score:
                best_model_score = mean_score
                best_model_name = name

        print(f"  Best residual model: {best_model_name} (CV = {best_model_score:.4f})")
        best_params["model"] = best_model_name
        best_params["model_factory"] = residual_models[best_model_name]
        best_params["cv_score"] = best_model_score
        best_config[target] = best_params

    return best_config


# ============================================================
# Generate optimized submissions
# ============================================================
def generate_optimized(train_df, val_df, config):
    """Generate submission with per-target optimized config."""
    temporal_feats = get_temporal_features(train_df)
    temporal_feats = [f for f in temporal_feats if f in val_df.columns]
    val_coords = val_df[["Latitude", "Longitude"]].values

    print("\n" + "=" * 80)
    print("GENERATING OPTIMIZED SUBMISSIONS")
    print("=" * 80)

    # --- v19: Optimized hybrid ---
    preds_optimized = {}
    for target in TARGETS:
        cfg = config[target]
        k, power = cfg["k"], cfg["power"]

        baseline_val = idw_predict(train_df, val_coords, target, k=k, power=power)
        baseline_train = idw_leave_one_out(train_df, target, k=k, power=power)
        residuals = train_df[target] - baseline_train

        X_tr = train_df[temporal_feats].fillna(train_df[temporal_feats].median())
        X_va = val_df[temporal_feats].fillna(train_df[temporal_feats].median())

        model = cfg["model_factory"]()
        model.fit(X_tr, residuals)
        res_pred = model.predict(X_va)

        preds_optimized[target] = np.maximum(baseline_val + res_pred, 0)
        print(f"  {target}: k={k}, power={power}, model={cfg['model']}, "
              f"pred_mean={preds_optimized[target].mean():.1f}")

    create_submission(val_df, preds_optimized, version="v19_optimized_hybrid")

    # --- v20: Monthly-aware IDW + ML residual ---
    preds_monthly = {}
    for target in TARGETS:
        cfg = config[target]
        k, power = cfg["k"], cfg["power"]

        baseline_val = idw_monthly(train_df, val_df, target, k=k, power=power)

        # For training residuals, use standard IDW (leave-one-out)
        baseline_train = idw_leave_one_out(train_df, target, k=k, power=power)
        residuals = train_df[target] - baseline_train

        X_tr = train_df[temporal_feats].fillna(train_df[temporal_feats].median())
        X_va = val_df[temporal_feats].fillna(train_df[temporal_feats].median())

        model = cfg["model_factory"]()
        model.fit(X_tr, residuals)
        res_pred = model.predict(X_va)

        preds_monthly[target] = np.maximum(baseline_val + res_pred, 0)
        print(f"  {target} (monthly): pred_mean={preds_monthly[target].mean():.1f}")

    create_submission(val_df, preds_monthly, version="v20_monthly_hybrid")

    # --- v21: Ensemble of v19 and v12-style temporal ML ---
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    preds_blend = {}
    for target in TARGETS:
        # Temporal ML prediction (v12 style)
        X_tr = train_df[temporal_feats].fillna(train_df[temporal_feats].median())
        X_va = val_df[temporal_feats].fillna(train_df[temporal_feats].median())
        y = train_df[target]

        rf = RandomForestRegressor(n_estimators=500, max_depth=15,
                                    min_samples_leaf=5, random_state=42, n_jobs=-1)
        xgb = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
                           reg_alpha=0.5, reg_lambda=3.0, tree_method="gpu_hist",
                           device="cuda", random_state=42)
        lgbm = LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
                             reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1)

        rf.fit(X_tr, y)
        xgb.fit(X_tr, y)
        lgbm.fit(X_tr, y)
        ml_pred = (rf.predict(X_va) + xgb.predict(X_va) + lgbm.predict(X_va)) / 3

        # Blend: 50% hybrid + 50% temporal ML
        hybrid_pred = preds_optimized[target]
        blended = 0.5 * hybrid_pred + 0.5 * ml_pred
        preds_blend[target] = np.maximum(blended, 0)
        print(f"  {target} (blend): hybrid_mean={hybrid_pred.mean():.1f}, "
              f"ml_mean={ml_pred.mean():.1f}, blend_mean={blended.mean():.1f}")

    create_submission(val_df, preds_blend, version="v21_hybrid_ml_blend")

    # --- v22: Pure IDW with optimal k/power (no ML) ---
    preds_pure_idw = {}
    for target in TARGETS:
        cfg = config[target]
        preds_pure_idw[target] = np.maximum(
            idw_predict(train_df, val_coords, target, k=cfg["k"], power=cfg["power"]), 0)
        print(f"  {target} (pure IDW): pred_mean={preds_pure_idw[target].mean():.1f}")

    create_submission(val_df, preds_pure_idw, version="v22_pure_idw_optimized")


if __name__ == "__main__":
    train_df, val_df = load_data()
    print(f"Training: {len(train_df)} rows")
    print(f"Validation: {len(val_df)} rows")

    config = optimize_hybrid(train_df)

    print("\n" + "=" * 80)
    print("BEST CONFIG PER TARGET")
    print("=" * 80)
    for target, cfg in config.items():
        print(f"  {target}: k={cfg['k']}, power={cfg['power']}, "
              f"model={cfg['model']}, CV={cfg['cv_score']:.4f}")

    generate_optimized(train_df, val_df, config)

    print("\nSubmissions generated:")
    print("  v19_optimized_hybrid.csv  — Per-target optimized k, power, model")
    print("  v20_monthly_hybrid.csv    — Month-aware IDW + ML residual")
    print("  v21_hybrid_ml_blend.csv   — 50/50 blend of hybrid + pure ML")
    print("  v22_pure_idw_optimized.csv — Pure IDW with optimized k/power")
    print("\nSubmit v19 first, then v21.")
