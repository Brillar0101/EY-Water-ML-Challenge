"""
Training pipeline v9 — NWU raw values as ML features.

KEY INSIGHT: v21 (R2=0.258) predicts with only 22-43% of true variance.
It regresses toward the global mean because IDW from 162 distant stations
gives a weak spatial signal.

SOLUTION: Use NWU historical measurements as ML FEATURES. The model
learns the NWU-to-competition mapping from 130 overlapping stations,
then applies it to 23 validation stations that have NWU data.

No manual unit conversion needed — XGBoost learns non-linear relationships.

NWU features for each observation:
    - nwu_tal_mean:    station-level historical TAL mean
    - nwu_ec_mean:     station-level historical EC mean (mS/m)
    - nwu_po4_mean:    station-level historical PO4 mean (mg/L)
    - nwu_tal_monthly: monthly TAL mean at this station for this month
    - nwu_ec_monthly:  monthly EC mean at this station for this month
    - nwu_po4_monthly: monthly PO4 mean at this station for this month

Internal validation:
    Station-level GroupKFold on 130 stations with NWU data.
    Compare R2 WITH vs WITHOUT NWU features to confirm improvement.

Usage:
    python src/train_v9.py
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
from sklearn.model_selection import GroupKFold, KFold

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
    exclude = set(TARGETS + ["Latitude", "Longitude", "Sample Date", "month", "doy",
                              "station_id", "has_nwu"])
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = [np.radians(float(x)) for x in [lat1, lon1, lat2, lon2]]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_nwu_features(train_df, val_df):
    """
    Add NWU historical water quality as ML features.

    For training: uses nwu_train_overlap_wq.csv (130 matched stations)
    For validation: uses nwu_nearby_water_quality.csv (23 matched stations)

    Features added (in RAW NWU units — no conversion):
        nwu_tal_mean, nwu_ec_mean, nwu_po4_mean   — station overall means
        nwu_tal_monthly, nwu_ec_monthly, nwu_po4_monthly — station-month means
        nwu_tal_std, nwu_ec_std — station variability
        nwu_sample_count — number of NWU samples (data quality indicator)
    """
    # Load NWU data
    nwu_nearby = pd.read_csv(DATA_DIR / "external" / "nwu_nearby_water_quality.csv")
    nwu_overlap = pd.read_csv(DATA_DIR / "external" / "nwu_train_overlap_wq.csv")
    train_map = pd.read_csv(DATA_DIR / "external" / "train_nwu_station_mapping.csv")
    val_mapping = pd.read_csv(DATA_DIR / "external" / "nearby_stations_mapping.csv")

    # Clean sentinel values in both datasets
    for nwu_data in [nwu_nearby, nwu_overlap]:
        for col in ["ec_msm", "tal_mgl", "po4_mgl"]:
            nwu_data.loc[nwu_data[col] < -9000, col] = np.nan
            nwu_data.loc[nwu_data[col] < 0, col] = np.nan
        nwu_data["date"] = pd.to_datetime(nwu_data["date"])
        nwu_data["month"] = nwu_data["date"].dt.month

    # Compute station-level and monthly stats for OVERLAP (training) stations
    overlap_stats = nwu_overlap.groupby("station_id").agg({
        "tal_mgl": ["mean", "std", "count"],
        "ec_msm": ["mean", "std"],
        "po4_mgl": ["mean", "std"],
    })
    overlap_stats.columns = ["_".join(c) for c in overlap_stats.columns]
    overlap_stats = overlap_stats.reset_index()

    overlap_monthly = nwu_overlap.groupby(["station_id", "month"]).agg({
        "tal_mgl": "mean",
        "ec_msm": "mean",
        "po4_mgl": "mean",
    }).reset_index()

    # Compute station-level and monthly stats for NEARBY (validation) stations
    nearby_stats = nwu_nearby.groupby("station_id").agg({
        "tal_mgl": ["mean", "std", "count"],
        "ec_msm": ["mean", "std"],
        "po4_mgl": ["mean", "std"],
    })
    nearby_stats.columns = ["_".join(c) for c in nearby_stats.columns]
    nearby_stats = nearby_stats.reset_index()

    nearby_monthly = nwu_nearby.groupby(["station_id", "month"]).agg({
        "tal_mgl": "mean",
        "ec_msm": "mean",
        "po4_mgl": "mean",
    }).reset_index()

    # --- Add NWU features to TRAINING data ---
    nwu_feat_cols = [
        "nwu_tal_mean", "nwu_ec_mean", "nwu_po4_mean",
        "nwu_tal_std", "nwu_ec_std",
        "nwu_tal_monthly", "nwu_ec_monthly", "nwu_po4_monthly",
        "nwu_sample_count",
    ]
    for col in nwu_feat_cols:
        train_df[col] = np.nan
        val_df[col] = np.nan

    train_df["has_nwu"] = 0
    val_df["has_nwu"] = 0

    # Map training stations to NWU
    matched_train = 0
    for _, m in train_map.iterrows():
        sid = m["nwu_sid"]
        row = overlap_stats[overlap_stats["station_id"] == sid]
        if len(row) == 0:
            continue

        row = row.iloc[0]
        mask = (
            (abs(train_df["Latitude"] - m["train_lat"]) < 0.001) &
            (abs(train_df["Longitude"] - m["train_lon"]) < 0.001)
        )

        if mask.sum() == 0:
            continue

        matched_train += 1
        train_df.loc[mask, "nwu_tal_mean"] = row["tal_mgl_mean"]
        train_df.loc[mask, "nwu_ec_mean"] = row["ec_msm_mean"]
        train_df.loc[mask, "nwu_po4_mean"] = row["po4_mgl_mean"]
        train_df.loc[mask, "nwu_tal_std"] = row.get("tal_mgl_std", 0)
        train_df.loc[mask, "nwu_ec_std"] = row.get("ec_msm_std", 0)
        train_df.loc[mask, "nwu_sample_count"] = row["tal_mgl_count"]
        train_df.loc[mask, "has_nwu"] = 1

        # Monthly features
        for _, obs in train_df[mask].iterrows():
            obs_month = obs["month"]
            monthly = overlap_monthly[
                (overlap_monthly["station_id"] == sid) &
                (overlap_monthly["month"] == obs_month)
            ]
            idx = obs.name
            if len(monthly) > 0:
                train_df.loc[idx, "nwu_tal_monthly"] = monthly.iloc[0]["tal_mgl"]
                train_df.loc[idx, "nwu_ec_monthly"] = monthly.iloc[0]["ec_msm"]
                train_df.loc[idx, "nwu_po4_monthly"] = monthly.iloc[0]["po4_mgl"]
            else:
                # Use overall mean as fallback for this month
                train_df.loc[idx, "nwu_tal_monthly"] = row["tal_mgl_mean"]
                train_df.loc[idx, "nwu_ec_monthly"] = row["ec_msm_mean"]
                train_df.loc[idx, "nwu_po4_monthly"] = row["po4_mgl_mean"]

    print("Training stations with NWU features: %d" % matched_train)
    print("Training rows with NWU: %d/%d (%.1f%%)" % (
        train_df["has_nwu"].sum(), len(train_df),
        100 * train_df["has_nwu"].mean()))

    # --- Add NWU features to VALIDATION data ---
    # Build val location -> NWU station mapping
    val_locs = val_df[["Latitude", "Longitude"]].drop_duplicates()

    matched_val = 0
    for _, vloc in val_locs.iterrows():
        vlat, vlon = vloc["Latitude"], vloc["Longitude"]

        # Find nearest NWU station
        best_dist = 999
        best_sid = None
        for _, m in val_mapping.iterrows():
            d = haversine_km(vlat, vlon, m["lat"], m["lon"])
            if d < best_dist:
                best_dist = d
                best_sid = m["SAMPLE STATION ID"]

        if best_sid is None or best_dist > 30:
            continue

        row = nearby_stats[nearby_stats["station_id"] == best_sid]
        if len(row) == 0:
            continue

        row = row.iloc[0]
        mask = (
            (abs(val_df["Latitude"] - vlat) < 0.0001) &
            (abs(val_df["Longitude"] - vlon) < 0.0001)
        )

        matched_val += 1
        val_df.loc[mask, "nwu_tal_mean"] = row["tal_mgl_mean"]
        val_df.loc[mask, "nwu_ec_mean"] = row["ec_msm_mean"]
        val_df.loc[mask, "nwu_po4_mean"] = row["po4_mgl_mean"]
        val_df.loc[mask, "nwu_tal_std"] = row.get("tal_mgl_std", 0)
        val_df.loc[mask, "nwu_ec_std"] = row.get("ec_msm_std", 0)
        val_df.loc[mask, "nwu_sample_count"] = row["tal_mgl_count"]
        val_df.loc[mask, "has_nwu"] = 1

        # Monthly features for validation
        for _, obs in val_df[mask].iterrows():
            obs_month = obs["month"]
            monthly = nearby_monthly[
                (nearby_monthly["station_id"] == best_sid) &
                (nearby_monthly["month"] == obs_month)
            ]
            idx = obs.name
            if len(monthly) > 0:
                val_df.loc[idx, "nwu_tal_monthly"] = monthly.iloc[0]["tal_mgl"]
                val_df.loc[idx, "nwu_ec_monthly"] = monthly.iloc[0]["ec_msm"]
                val_df.loc[idx, "nwu_po4_monthly"] = monthly.iloc[0]["po4_mgl"]
            else:
                val_df.loc[idx, "nwu_tal_monthly"] = row["tal_mgl_mean"]
                val_df.loc[idx, "nwu_ec_monthly"] = row["ec_msm_mean"]
                val_df.loc[idx, "nwu_po4_monthly"] = row["po4_mgl_mean"]

    print("Validation locations with NWU features: %d/24" % matched_val)
    print("Validation rows with NWU: %d/%d (%.1f%%)" % (
        val_df["has_nwu"].sum(), len(val_df),
        100 * val_df["has_nwu"].mean()))

    # --- Impute missing NWU features using IDW from available NWU stations ---
    # For training rows without NWU data: interpolate from nearby NWU-matched stations
    train_with_nwu = train_df[train_df["has_nwu"] == 1]
    train_nwu_stations = train_with_nwu.groupby(["Latitude", "Longitude"])[nwu_feat_cols].mean().reset_index()

    coords_nwu = np.radians(train_nwu_stations[["Latitude", "Longitude"]].values)
    tree = cKDTree(coords_nwu)

    missing_mask = train_df["has_nwu"] == 0
    if missing_mask.sum() > 0:
        missing_coords = np.radians(train_df.loc[missing_mask, ["Latitude", "Longitude"]].values)
        k = min(5, len(coords_nwu))
        dists, idxs = tree.query(missing_coords, k=k)
        dists_km = dists * 6371
        weights = 1.0 / (dists_km ** 2 + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)

        for col in nwu_feat_cols:
            vals = train_nwu_stations[col].values
            imputed = np.sum(weights * vals[idxs], axis=1)
            train_df.loc[missing_mask, col] = imputed

        print("Imputed NWU features for %d training rows via IDW" % missing_mask.sum())

    # Impute missing validation NWU features
    val_missing = val_df["has_nwu"] == 0
    if val_missing.sum() > 0:
        # Use NWU nearby stations for validation imputation
        val_nwu_stations = val_df[val_df["has_nwu"] == 1].groupby(
            ["Latitude", "Longitude"])[nwu_feat_cols].mean().reset_index()
        if len(val_nwu_stations) > 0:
            coords_val_nwu = np.radians(val_nwu_stations[["Latitude", "Longitude"]].values)
            tree_val = cKDTree(coords_val_nwu)
            missing_coords = np.radians(val_df.loc[val_missing, ["Latitude", "Longitude"]].values)
            k = min(5, len(coords_val_nwu))
            dists, idxs = tree_val.query(missing_coords, k=k)
            dists_km = dists * 6371
            weights = 1.0 / (dists_km ** 2 + 1e-6)
            weights = weights / weights.sum(axis=1, keepdims=True)
            for col in nwu_feat_cols:
                vals = val_nwu_stations[col].values
                imputed = np.sum(weights * vals[idxs], axis=1)
                val_df.loc[val_missing, col] = imputed
            print("Imputed NWU features for %d validation rows via IDW" % val_missing.sum())

    return train_df, val_df


def internal_validation(train_df):
    """
    Station-level cross-validation comparing WITH vs WITHOUT NWU features.
    Only uses stations that HAVE NWU data (130 stations) for fair comparison.
    """
    print("\n" + "=" * 80)
    print("INTERNAL VALIDATION: NWU features vs baseline")
    print("=" * 80)

    # Filter to stations with NWU data
    nwu_stations = train_df[train_df["has_nwu"] == 1].copy()
    print("Using %d rows from %d stations with NWU data" % (
        len(nwu_stations),
        nwu_stations.groupby(["Latitude", "Longitude"]).ngroups))

    # Create station IDs for GroupKFold
    nwu_stations["station_id"] = (
        nwu_stations["Latitude"].round(3).astype(str) + "_" +
        nwu_stations["Longitude"].round(3).astype(str)
    )

    # Feature sets
    nwu_feat_cols = [
        "nwu_tal_mean", "nwu_ec_mean", "nwu_po4_mean",
        "nwu_tal_std", "nwu_ec_std",
        "nwu_tal_monthly", "nwu_ec_monthly", "nwu_po4_monthly",
        "nwu_sample_count",
    ]

    temporal_feats = get_temporal_features(train_df)
    temporal_feats = [f for f in temporal_feats if f not in nwu_feat_cols]

    feats_baseline = temporal_feats
    feats_with_nwu = temporal_feats + nwu_feat_cols

    groups = nwu_stations["station_id"]
    n_groups = groups.nunique()
    gkf = GroupKFold(n_splits=min(5, n_groups))

    print("\nStation-level GroupKFold (k=%d):" % min(5, n_groups))
    print("-" * 60)

    for target in TARGETS:
        y = nwu_stations[target]

        # Baseline (temporal features only)
        X_base = nwu_stations[feats_baseline].fillna(nwu_stations[feats_baseline].median())
        scores_base = []
        for tr_idx, va_idx in gkf.split(X_base, y, groups):
            rf = RandomForestRegressor(n_estimators=300, max_depth=12,
                                       min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf.fit(X_base.iloc[tr_idx], y.iloc[tr_idx])
            pred = rf.predict(X_base.iloc[va_idx])
            scores_base.append(r2_score(y.iloc[va_idx], pred))

        # With NWU features
        X_nwu = nwu_stations[feats_with_nwu].fillna(nwu_stations[feats_with_nwu].median())
        scores_nwu = []
        for tr_idx, va_idx in gkf.split(X_nwu, y, groups):
            rf = RandomForestRegressor(n_estimators=300, max_depth=12,
                                       min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf.fit(X_nwu.iloc[tr_idx], y.iloc[tr_idx])
            pred = rf.predict(X_nwu.iloc[va_idx])
            scores_nwu.append(r2_score(y.iloc[va_idx], pred))

        base_mean = np.mean(scores_base)
        nwu_mean = np.mean(scores_nwu)
        improvement = nwu_mean - base_mean

        print("  %s:" % target)
        print("    Baseline (temporal only):  R2 = %.4f" % base_mean)
        print("    With NWU features:         R2 = %.4f  (%+.4f)" % (nwu_mean, improvement))

    print()

    # Also try with XGBoost if available
    try:
        from xgboost import XGBRegressor

        print("XGBoost station-level GroupKFold:")
        print("-" * 60)

        for target in TARGETS:
            y = nwu_stations[target]

            X_base = nwu_stations[feats_baseline].fillna(nwu_stations[feats_baseline].median())
            X_nwu = nwu_stations[feats_with_nwu].fillna(nwu_stations[feats_with_nwu].median())

            scores_base = []
            scores_nwu = []

            for tr_idx, va_idx in gkf.split(X_base, y, groups):
                # Baseline
                xgb = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
                                   reg_alpha=0.5, reg_lambda=3.0, random_state=42,
                                   tree_method="hist")
                xgb.fit(X_base.iloc[tr_idx], y.iloc[tr_idx])
                scores_base.append(r2_score(y.iloc[va_idx], xgb.predict(X_base.iloc[va_idx])))

                # With NWU
                xgb2 = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
                                    reg_alpha=0.5, reg_lambda=3.0, random_state=42,
                                    tree_method="hist")
                xgb2.fit(X_nwu.iloc[tr_idx], y.iloc[tr_idx])
                scores_nwu.append(r2_score(y.iloc[va_idx], xgb2.predict(X_nwu.iloc[va_idx])))

            base_mean = np.mean(scores_base)
            nwu_mean = np.mean(scores_nwu)
            print("  %s:" % target)
            print("    Baseline:     R2 = %.4f" % base_mean)
            print("    With NWU:     R2 = %.4f  (%+.4f)" % (nwu_mean, nwu_mean - base_mean))

    except ImportError:
        print("XGBoost not available for CV comparison (will use on cluster)")


def generate_submission(train_df, val_df):
    """Generate final submission with NWU features."""
    print("\n" + "=" * 80)
    print("GENERATING FINAL SUBMISSION")
    print("=" * 80)

    nwu_feat_cols = [
        "nwu_tal_mean", "nwu_ec_mean", "nwu_po4_mean",
        "nwu_tal_std", "nwu_ec_std",
        "nwu_tal_monthly", "nwu_ec_monthly", "nwu_po4_monthly",
        "nwu_sample_count",
    ]

    temporal_feats = get_temporal_features(train_df)
    temporal_feats = [f for f in temporal_feats if f not in nwu_feat_cols]
    all_feats = temporal_feats + nwu_feat_cols
    all_feats = [f for f in all_feats if f in train_df.columns and f in val_df.columns]

    X_tr = train_df[all_feats].fillna(train_df[all_feats].median())
    X_va = val_df[all_feats].fillna(train_df[all_feats].median())

    predictions = {}

    for target in TARGETS:
        y = train_df[target]
        print("\n--- %s ---" % target)

        # Train ensemble
        try:
            from xgboost import XGBRegressor
            from lightgbm import LGBMRegressor

            rf = RandomForestRegressor(n_estimators=500, max_depth=15,
                                       min_samples_leaf=5, random_state=42, n_jobs=-1)
            xgb = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.7, min_child_weight=15,
                               reg_alpha=0.5, reg_lambda=3.0, random_state=42,
                               tree_method="gpu_hist", device="cuda")
            lgbm = LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
                                 reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1)

            rf.fit(X_tr, y)
            xgb.fit(X_tr, y)
            lgbm.fit(X_tr, y)
            pred = (rf.predict(X_va) + xgb.predict(X_va) + lgbm.predict(X_va)) / 3

            # Feature importance from XGBoost
            imp = pd.Series(xgb.feature_importances_, index=all_feats).sort_values(ascending=False)
            print("  Top 10 features:")
            for feat, score in imp.head(10).items():
                marker = " *** NWU ***" if feat.startswith("nwu_") else ""
                print("    %.4f  %s%s" % (score, feat, marker))

        except ImportError:
            print("  (XGBoost/LightGBM not available, using RF only)")
            rf = RandomForestRegressor(n_estimators=500, max_depth=15,
                                       min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y)
            pred = rf.predict(X_va)

            imp = pd.Series(rf.feature_importances_, index=all_feats).sort_values(ascending=False)
            print("  Top 10 features:")
            for feat, score in imp.head(10).items():
                marker = " *** NWU ***" if feat.startswith("nwu_") else ""
                print("    %.4f  %s%s" % (score, feat, marker))

        predictions[target] = np.maximum(pred, 0)
        print("  pred: mean=%.1f, std=%.1f, range=[%.1f, %.1f]" % (
            pred.mean(), pred.std(), pred.min(), pred.max()))

    create_submission(val_df, predictions, version="v36_nwu_features")

    # Also generate a blended version with v21 if available
    v21_path = ROOT_DIR / "outputs" / "submissions" / "submission_v21_hybrid_ml_blend.csv"
    if v21_path.exists():
        v21 = pd.read_csv(v21_path)
        print("\n--- Blending with v21 (0.258 baseline) ---")

        for alpha in [0.3, 0.5, 0.7]:
            blend_preds = {}
            for target in TARGETS:
                blend_preds[target] = np.maximum(
                    alpha * predictions[target] + (1 - alpha) * v21[target].values, 0)

            version = "v37_nwu_blend_%d" % int(alpha * 100)
            create_submission(val_df, blend_preds, version=version)
            print("  %s: TAL=%.1f, EC=%.1f, DRP=%.1f" % (
                version,
                blend_preds["Total Alkalinity"].mean(),
                blend_preds["Electrical Conductance"].mean(),
                blend_preds["Dissolved Reactive Phosphorus"].mean()))

    return predictions


if __name__ == "__main__":
    print("=" * 80)
    print("TRAIN v9 — NWU RAW VALUES AS ML FEATURES")
    print("=" * 80)

    train_df, val_df = load_data()
    print("Training: %d rows" % len(train_df))
    print("Validation: %d rows" % len(val_df))

    # Add NWU features
    train_df, val_df = add_nwu_features(train_df, val_df)

    # Internal validation (CRITICAL: test before submitting)
    internal_validation(train_df)

    # Generate submission
    generate_submission(train_df, val_df)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print()
    print("Check internal validation results above.")
    print("If NWU features IMPROVE station-level CV, submit v36.")
    print("If improvement is marginal, submit v37_nwu_blend_50 (hedged).")
    print("If NWU features HURT, do NOT submit — the approach doesn't work.")
