"""
Training pipeline v4 — Temporal features only.

Key learnings from v1-v3:
    - Static features (soil, terrain, landcover) overfit to station identity
    - The benchmark's 4 temporal features (Landsat + PET) generalize
    - Benchmark gets 0.20 on leaderboard, our static features got -0.31
    - We MUST focus on temporal/observational features

Strategy:
    1. Start from exact benchmark reproduction
    2. Add only TEMPORAL features (climate vars, temporal encoding)
    3. No static features unless they prove useful on leaderboard
    4. Try multiple models and ensembles
    5. Try spatial interpolation as alternative approach

Usage:
    python src/train_v4.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data_loader import build_full_dataset
from feature_builder import build_features, get_feature_columns
from submission import create_submission


TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
DATA_DIR = ROOT_DIR / "datasets"


def load_and_merge():
    """Load data and merge features manually to control exactly what goes in."""
    # Load raw datasets
    train = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
    train["Sample Date"] = pd.to_datetime(train["Sample Date"], dayfirst=True)

    val = pd.read_csv(DATA_DIR / "submission_template.csv")
    val["Sample Date"] = pd.to_datetime(val["Sample Date"], dayfirst=True)

    # Landsat features
    landsat_tr = pd.read_csv(DATA_DIR / "train_landsat_features.csv")
    landsat_tr["Sample Date"] = pd.to_datetime(landsat_tr["Sample Date"], dayfirst=True)
    landsat_va = pd.read_csv(DATA_DIR / "val_landsat_features.csv")
    landsat_va["Sample Date"] = pd.to_datetime(landsat_va["Sample Date"], dayfirst=True)

    # TerraClimate benchmark (PET only)
    tc_tr = pd.read_csv(DATA_DIR / "train_terraclimate_features.csv")
    tc_tr["Sample Date"] = pd.to_datetime(tc_tr["Sample Date"], dayfirst=True)
    tc_va = pd.read_csv(DATA_DIR / "val_terraclimate_features.csv")
    tc_va["Sample Date"] = pd.to_datetime(tc_va["Sample Date"], dayfirst=True)

    # Extended TerraClimate (all 14 vars)
    tc_ext_path = DATA_DIR / "processed"
    tc_ext_tr = None
    tc_ext_va = None
    if (tc_ext_path / "train_terraclimate_extended.csv").exists():
        tc_ext_tr = pd.read_csv(tc_ext_path / "train_terraclimate_extended.csv")
        tc_ext_tr["Sample Date"] = pd.to_datetime(tc_ext_tr["Sample Date"], format="mixed", dayfirst=True)
        tc_ext_va = pd.read_csv(tc_ext_path / "val_terraclimate_extended.csv")
        tc_ext_va["Sample Date"] = pd.to_datetime(tc_ext_va["Sample Date"], format="mixed", dayfirst=True)

    # Merge
    join_cols = ["Latitude", "Longitude", "Sample Date"]
    train_full = train.merge(landsat_tr, on=join_cols, how="left")
    train_full = train_full.merge(tc_tr, on=join_cols, how="left")

    val_full = val.merge(landsat_va, on=join_cols, how="left")
    val_full = val_full.merge(tc_va, on=join_cols, how="left")

    if tc_ext_tr is not None:
        # Drop 'pet' from extended (already have it from benchmark)
        ext_cols = [c for c in tc_ext_tr.columns if c not in ["pet"] or c in join_cols]
        train_full = train_full.merge(tc_ext_tr[ext_cols], on=join_cols, how="left")
        val_full = val_full.merge(tc_ext_va[ext_cols], on=join_cols, how="left")

    return train_full, val_full


def add_temporal_encoding(df):
    """Add temporal features that capture seasonal patterns."""
    df = df.copy()
    df["month"] = df["Sample Date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year"] = df["Sample Date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # South Africa seasons: DJF=summer/wet, JJA=winter/dry
    month = df["month"]
    df["is_wet_season"] = ((month >= 10) | (month <= 3)).astype(int)

    return df


def add_landsat_indices(df):
    """Add spectral water quality indices beyond benchmark."""
    df = df.copy()
    eps = 1e-10

    # NDWI (Green - NIR) / (Green + NIR) — open water detection
    if "green" in df.columns and "nir" in df.columns:
        df["NDWI"] = (df["green"] - df["nir"]) / (df["green"] + df["nir"] + eps)

    # Moisture Stress Index
    if "nir" in df.columns and "swir16" in df.columns:
        df["MSI"] = df["swir16"] / (df["nir"] + eps)

    # SWIR ratio — sediment/turbidity proxy
    if "swir16" in df.columns and "swir22" in df.columns:
        df["SWIR_ratio"] = df["swir16"] / (df["swir22"] + eps)

    # Missing data indicators (cloud cover is informative)
    for col in ["nir", "green", "swir16", "swir22"]:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    return df


def impute_and_prepare(X_train, X_val):
    """Impute NaN with training medians and scale."""
    medians = X_train.median()
    X_train_imp = X_train.fillna(medians)
    X_val_imp = X_val.fillna(medians)
    return X_train_imp, X_val_imp


# ============================================================
# Approach 1: Exact benchmark reproduction
# ============================================================
def approach_benchmark(train_df, val_df):
    """Exact benchmark: RF(100), StandardScaler, 4 features, median imputation."""
    print("\n" + "=" * 80)
    print("APPROACH 1: EXACT BENCHMARK REPRODUCTION")
    print("=" * 80)

    feats = ["swir22", "NDMI", "MNDWI", "pet"]
    X_tr = train_df[feats].fillna(train_df[feats].median())
    X_va = val_df[feats].fillna(train_df[feats].median())

    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=feats, index=X_tr.index)
    X_va_sc = pd.DataFrame(scaler.transform(X_va), columns=feats, index=X_va.index)

    predictions = {}
    for target in TARGETS:
        y = train_df[target]
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_tr_sc, y)
        pred = np.maximum(model.predict(X_va_sc), 0)
        predictions[target] = pred

        # Quick random CV check
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, va in kf.split(X_tr_sc):
            m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            m.fit(X_tr_sc.iloc[tr], y.iloc[tr])
            scores.append(r2_score(y.iloc[va], m.predict(X_tr_sc.iloc[va])))
        print(f"  {target:40s} random_CV_R² = {np.mean(scores):.4f}  pred_mean = {pred.mean():.1f}")

    create_submission(val_df, predictions, version="v8_exact_benchmark")
    return predictions


# ============================================================
# Approach 2: Benchmark + all TerraClimate (temporal only)
# ============================================================
def approach_temporal_climate(train_df, val_df):
    """Benchmark features + all temporal climate variables."""
    print("\n" + "=" * 80)
    print("APPROACH 2: BENCHMARK + ALL TERRACLIMATE (temporal only)")
    print("=" * 80)

    base_feats = ["swir22", "NDMI", "MNDWI", "pet"]
    climate_vars = ["tmax", "tmin", "ppt", "vap", "srad", "ws", "aet",
                    "q", "def", "soil", "pdsi", "vpd"]
    lag_vars = [c for c in train_df.columns if "_lag1" in c or "_roll3" in c]

    feature_sets = {
        "benchmark_4": base_feats,
        "bench+climate": base_feats + [c for c in climate_vars if c in train_df.columns],
        "bench+climate+lags": base_feats + [c for c in climate_vars if c in train_df.columns] + [c for c in lag_vars if c in train_df.columns],
        "bench+climate+temporal": base_feats + [c for c in climate_vars if c in train_df.columns] + ["month_sin", "month_cos", "doy_sin", "doy_cos", "is_wet_season"],
        "all_temporal": base_feats + [c for c in climate_vars if c in train_df.columns] + [c for c in lag_vars if c in train_df.columns] + ["month_sin", "month_cos", "doy_sin", "doy_cos", "is_wet_season"],
    }

    best_set = {}

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")

        best_r2 = -999
        best_name = None

        for name, feats in feature_sets.items():
            avail = [f for f in feats if f in train_df.columns]
            X_tr = train_df[avail].fillna(train_df[avail].median())

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for tr, va in kf.split(X_tr):
                m = RandomForestRegressor(n_estimators=200, max_depth=15,
                                           min_samples_leaf=5, random_state=42, n_jobs=-1)
                m.fit(X_tr.iloc[tr], y.iloc[tr])
                scores.append(r2_score(y.iloc[va], m.predict(X_tr.iloc[va])))

            mean_r2 = np.mean(scores)
            print(f"  {name:30s} ({len(avail):2d} feats) → random_CV = {mean_r2:.4f}")

            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_name = name

        best_set[target] = feature_sets[best_name]
        print(f"  BEST: {best_name}")

    # Train and predict with best feature set per target
    predictions = {}
    for target in TARGETS:
        feats = [f for f in best_set[target] if f in train_df.columns and f in val_df.columns]
        X_tr = train_df[feats].fillna(train_df[feats].median())
        X_va = val_df[feats].fillna(train_df[feats].median())

        y = train_df[target]
        model = RandomForestRegressor(n_estimators=500, max_depth=15,
                                       min_samples_leaf=5, random_state=42, n_jobs=-1)
        model.fit(X_tr, y)
        predictions[target] = np.maximum(model.predict(X_va), 0)

    create_submission(val_df, predictions, version="v9_temporal_climate")
    return predictions


# ============================================================
# Approach 3: Benchmark + Landsat indices + temporal
# ============================================================
def approach_enhanced_landsat(train_df, val_df):
    """Enhanced Landsat features with spectral indices."""
    print("\n" + "=" * 80)
    print("APPROACH 3: ENHANCED LANDSAT + TEMPORAL ENCODING")
    print("=" * 80)

    feats = ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI", "pet",
             "NDWI", "MSI", "SWIR_ratio",
             "nir_missing", "green_missing", "swir16_missing", "swir22_missing",
             "month_sin", "month_cos", "doy_sin", "doy_cos", "is_wet_season"]
    avail = [f for f in feats if f in train_df.columns and f in val_df.columns]

    X_tr = train_df[avail].fillna(train_df[avail].median())
    X_va = val_df[avail].fillna(train_df[avail].median())

    predictions = {}
    for target in TARGETS:
        y = train_df[target]

        # Random CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr, va in kf.split(X_tr):
            m = RandomForestRegressor(n_estimators=300, max_depth=15,
                                       min_samples_leaf=5, random_state=42, n_jobs=-1)
            m.fit(X_tr.iloc[tr], y.iloc[tr])
            scores.append(r2_score(y.iloc[va], m.predict(X_tr.iloc[va])))
        print(f"  {target:40s} random_CV = {np.mean(scores):.4f}  ({len(avail)} feats)")

        model = RandomForestRegressor(n_estimators=500, max_depth=15,
                                       min_samples_leaf=5, random_state=42, n_jobs=-1)
        model.fit(X_tr, y)
        predictions[target] = np.maximum(model.predict(X_va), 0)

    create_submission(val_df, predictions, version="v10_enhanced_landsat")
    return predictions


# ============================================================
# Approach 4: XGBoost with temporal features
# ============================================================
def approach_xgb_temporal(train_df, val_df):
    """XGBoost with only temporal features."""
    from xgboost import XGBRegressor

    print("\n" + "=" * 80)
    print("APPROACH 4: XGBOOST WITH TEMPORAL FEATURES")
    print("=" * 80)

    feats = ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI",
             "NDWI", "MSI", "SWIR_ratio",
             "nir_missing", "green_missing", "swir16_missing", "swir22_missing",
             "month_sin", "month_cos", "doy_sin", "doy_cos", "is_wet_season"]

    # Add climate vars if available
    climate_vars = ["tmax", "tmin", "ppt", "vap", "srad", "ws", "aet", "pet",
                    "q", "def", "soil", "pdsi", "vpd"]
    feats.extend(climate_vars)

    avail = [f for f in feats if f in train_df.columns and f in val_df.columns]

    X_tr = train_df[avail].fillna(train_df[avail].median())
    X_va = val_df[avail].fillna(train_df[avail].median())

    predictions = {}
    for target in TARGETS:
        y = train_df[target]

        configs = {
            "d4_reg": dict(n_estimators=500, max_depth=4, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
                           reg_alpha=0.5, reg_lambda=3.0),
            "d6_mod": dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
                           reg_alpha=0.3, reg_lambda=2.0),
        }

        best_r2 = -999
        best_name = None

        for name, params in configs.items():
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for tr, va in kf.split(X_tr):
                m = XGBRegressor(**params, tree_method="gpu_hist", device="cuda", random_state=42)
                m.fit(X_tr.iloc[tr], y.iloc[tr])
                scores.append(r2_score(y.iloc[va], m.predict(X_tr.iloc[va])))
            mean_r2 = np.mean(scores)
            print(f"  {target:35s} {name:12s} → random_CV = {mean_r2:.4f}")
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_name = name

        # Train final
        model = XGBRegressor(**configs[best_name], tree_method="gpu_hist",
                             device="cuda", random_state=42)
        model.fit(X_tr, y)
        predictions[target] = np.maximum(model.predict(X_va), 0)

    create_submission(val_df, predictions, version="v11_xgb_temporal")
    return predictions


# ============================================================
# Approach 5: Ensemble (RF + XGB + GBR) temporal features
# ============================================================
def approach_ensemble_temporal(train_df, val_df):
    """Average multiple model predictions with temporal features."""
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    print("\n" + "=" * 80)
    print("APPROACH 5: ENSEMBLE (RF + XGB + LGBM) TEMPORAL FEATURES")
    print("=" * 80)

    feats = ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI",
             "NDWI", "MSI", "SWIR_ratio",
             "nir_missing", "green_missing", "swir16_missing", "swir22_missing",
             "month_sin", "month_cos", "doy_sin", "doy_cos", "is_wet_season"]
    climate_vars = ["tmax", "tmin", "ppt", "vap", "srad", "ws", "aet", "pet",
                    "q", "def", "soil", "pdsi", "vpd"]
    feats.extend(climate_vars)
    avail = [f for f in feats if f in train_df.columns and f in val_df.columns]

    X_tr = train_df[avail].fillna(train_df[avail].median())
    X_va = val_df[avail].fillna(train_df[avail].median())

    predictions = {}

    for target in TARGETS:
        y = train_df[target]

        # Train 3 models
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

        p_rf = rf.predict(X_va)
        p_xgb = xgb.predict(X_va)
        p_lgbm = lgbm.predict(X_va)

        # Simple average
        pred = np.maximum((p_rf + p_xgb + p_lgbm) / 3, 0)
        predictions[target] = pred

        # CV of ensemble
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        ens_scores = []
        for tr, va in kf.split(X_tr):
            rf_cv = RandomForestRegressor(n_estimators=300, max_depth=15,
                                           min_samples_leaf=5, random_state=42, n_jobs=-1)
            xgb_cv = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
                                   tree_method="gpu_hist", device="cuda", random_state=42)
            lgbm_cv = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.03,
                                     subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
                                     random_state=42, verbose=-1)
            rf_cv.fit(X_tr.iloc[tr], y.iloc[tr])
            xgb_cv.fit(X_tr.iloc[tr], y.iloc[tr])
            lgbm_cv.fit(X_tr.iloc[tr], y.iloc[tr])
            p = (rf_cv.predict(X_tr.iloc[va]) + xgb_cv.predict(X_tr.iloc[va]) + lgbm_cv.predict(X_tr.iloc[va])) / 3
            ens_scores.append(r2_score(y.iloc[va], p))

        print(f"  {target:40s} ensemble_CV = {np.mean(ens_scores):.4f}  pred_mean = {pred.mean():.1f}")

    create_submission(val_df, predictions, version="v12_ensemble_temporal")
    return predictions


# ============================================================
# Approach 6: Spatial interpolation (no ML — pure proximity)
# ============================================================
def approach_spatial_interpolation(train_df, val_df):
    """Predict using weighted average of nearby training stations."""
    from scipy.spatial import cKDTree

    print("\n" + "=" * 80)
    print("APPROACH 6: SPATIAL INTERPOLATION (IDW)")
    print("=" * 80)

    # Get station-level means from training data
    station_stats = train_df.groupby(["Latitude", "Longitude"]).agg({
        "Total Alkalinity": "mean",
        "Electrical Conductance": "mean",
        "Dissolved Reactive Phosphorus": "mean",
    }).reset_index()

    train_coords = np.radians(station_stats[["Latitude", "Longitude"]].values)
    val_coords = np.radians(val_df[["Latitude", "Longitude"]].values)

    tree = cKDTree(train_coords)

    predictions = {}

    for k in [3, 5, 10, 20, 50]:
        dists, idxs = tree.query(val_coords, k=k)
        # Convert to km
        dists_km = dists * 6371

        for target in TARGETS:
            station_vals = station_stats[target].values

            # Inverse Distance Weighting
            weights = 1.0 / (dists_km + 1e-6)  # avoid div by zero
            weights = weights / weights.sum(axis=1, keepdims=True)

            neighbor_vals = station_vals[idxs]
            pred = np.sum(weights * neighbor_vals, axis=1)
            pred = np.maximum(pred, 0)

            if k not in predictions:
                predictions[k] = {}
            predictions[k][target] = pred

        # Report
        print(f"  k={k:3d}: TA_mean={predictions[k]['Total Alkalinity'].mean():.1f}, "
              f"EC_mean={predictions[k]['Electrical Conductance'].mean():.1f}, "
              f"DRP_mean={predictions[k]['Dissolved Reactive Phosphorus'].mean():.1f}")

    # Generate submissions for k=5 and k=20
    create_submission(val_df, predictions[5], version="v13_idw_k5")
    create_submission(val_df, predictions[20], version="v14_idw_k20")

    return predictions


if __name__ == "__main__":
    train_df, val_df = load_and_merge()

    # Add derived features
    train_df = add_temporal_encoding(train_df)
    val_df = add_temporal_encoding(val_df)
    train_df = add_landsat_indices(train_df)
    val_df = add_landsat_indices(val_df)

    print(f"Training: {len(train_df)} rows")
    print(f"Validation: {len(val_df)} rows")
    temporal_feats = [c for c in train_df.columns if c not in
                      ["Latitude", "Longitude", "Sample Date"] + TARGETS]
    print(f"Available features: {len(temporal_feats)}")

    # Run all approaches
    approach_benchmark(train_df, val_df)         # v8
    approach_temporal_climate(train_df, val_df)   # v9
    approach_enhanced_landsat(train_df, val_df)   # v10
    approach_xgb_temporal(train_df, val_df)       # v11
    approach_ensemble_temporal(train_df, val_df)  # v12
    approach_spatial_interpolation(train_df, val_df)  # v13, v14

    print("\n" + "=" * 80)
    print("ALL SUBMISSIONS GENERATED")
    print("=" * 80)
    print("  v8_exact_benchmark.csv    — RF(100), 4 features, StandardScaler")
    print("  v9_temporal_climate.csv   — RF, best temporal feature set per target")
    print("  v10_enhanced_landsat.csv  — RF, Landsat indices + temporal encoding")
    print("  v11_xgb_temporal.csv      — XGBoost, temporal + climate")
    print("  v12_ensemble_temporal.csv — RF+XGB+LGBM average, temporal + climate")
    print("  v13_idw_k5.csv            — Pure spatial interpolation (k=5)")
    print("  v14_idw_k20.csv           — Pure spatial interpolation (k=20)")
    print("\nSubmit v8 first to calibrate, then v12 (ensemble), then v13 (spatial).")
