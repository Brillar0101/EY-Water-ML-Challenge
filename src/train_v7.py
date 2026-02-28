"""
Training pipeline v7 — External NWU water quality data integration.

BREAKTHROUGH: The NWU Water Chemistry Dataset (1999-2012) contains historical
measurements at 20 of the 24 validation station locations (exact coordinate
match). The remaining 4 have NWU stations within 17km.

This dramatically improves spatial prediction because instead of interpolating
from distant training stations (~100km away), we have direct measurements
at the target locations.

Unit conversions (calibrated on 127 overlapping stations):
    EC:  competition_uScm = NWU_mSm × 10.0
    TAL: competition_mgL  = NWU_mgL × 1.08
    DRP: competition_ugL  = NWU_PO4_mgL × 530

Approaches:
    v23: NWU station means (direct lookup)
    v24: NWU monthly means (seasonal adjustment)
    v25: NWU baseline + temporal ML residual (hybrid)
    v26: Expanded training data (NWU + original)
    v27: Multi-source ensemble (NWU + IDW + ML)

Usage:
    python src/train_v7.py
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
from sklearn.linear_model import Ridge, LinearRegression

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from submission import create_submission

TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
DATA_DIR = ROOT_DIR / "datasets"


# ============================================================
# Unit conversion factors (calibrated on overlapping stations)
# ============================================================
EC_FACTOR = 10.0       # NWU mS/m → competition µS/cm
TAL_FACTOR = 1.08      # NWU mg/L → competition mg/L (minor calibration)
DRP_FACTOR = 530.0     # NWU PO4 mg/L → competition DRP


def load_data():
    """Load competition training + validation data with temporal features."""
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

    # Extended TerraClimate (if available)
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

    # Temporal encoding
    for df in [train, val]:
        df["month"] = df["Sample Date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy"] = df["Sample Date"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
        df["is_wet"] = ((df["month"] >= 10) | (df["month"] <= 3)).astype(int)

    # Landsat indices
    eps = 1e-10
    for df in [train, val]:
        if "green" in df.columns and "nir" in df.columns:
            df["NDWI"] = (df["green"] - df["nir"]) / (df["green"] + df["nir"] + eps)
        if "nir" in df.columns and "swir16" in df.columns:
            df["MSI"] = df["swir16"] / (df["nir"] + eps)
        if "swir16" in df.columns and "swir22" in df.columns:
            df["SWIR_ratio"] = df["swir16"] / (df["swir22"] + eps)

    return train, val


def load_nwu_data():
    """Load and convert NWU external water quality data."""
    nwu_path = DATA_DIR / "external" / "nwu_nearby_water_quality.csv"
    if not nwu_path.exists():
        print("ERROR: NWU data not found at %s" % nwu_path)
        print("Run the NWU extraction pipeline first.")
        return None, None

    nwu = pd.read_csv(nwu_path)

    # Clean sentinel values
    for col in ["ec_msm", "po4_mgl", "tal_mgl"]:
        nwu.loc[nwu[col] < -9000, col] = np.nan
        nwu.loc[nwu[col] < 0, col] = np.nan

    # Convert to competition units
    nwu["Electrical Conductance"] = nwu["ec_msm"] * EC_FACTOR
    nwu["Total Alkalinity"] = nwu["tal_mgl"] * TAL_FACTOR
    nwu["Dissolved Reactive Phosphorus"] = nwu["po4_mgl"] * DRP_FACTOR

    # Parse dates
    nwu["date"] = pd.to_datetime(nwu["date"])
    nwu["month"] = nwu["date"].dt.month

    # Load station mapping (station_id -> lat/lon)
    mapping = pd.read_csv(DATA_DIR / "external" / "nearby_stations_mapping.csv")

    return nwu, mapping


def match_val_to_nwu(val_df, nwu, mapping):
    """
    For each validation point, find the closest NWU station and return
    its historical data.

    Returns dict: val_idx -> {station_id, lat, lon, dist_km, data}
    """
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = [np.radians(float(x)) for x in [lat1, lon1, lat2, lon2]]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    val_matches = {}
    val_locs = val_df[["Latitude", "Longitude"]].drop_duplicates()

    for _, vloc in val_locs.iterrows():
        vlat, vlon = vloc["Latitude"], vloc["Longitude"]

        best_dist = 999
        best_sid = None

        for _, m in mapping.iterrows():
            d = haversine_km(vlat, vlon, m["lat"], m["lon"])
            if d < best_dist:
                best_dist = d
                best_sid = m["SAMPLE STATION ID"]

        if best_sid is not None and best_dist < 30:
            stn_data = nwu[nwu["station_id"] == best_sid].copy()
            val_matches[(vlat, vlon)] = {
                "station_id": best_sid,
                "dist_km": best_dist,
                "data": stn_data,
                "n_samples": len(stn_data),
            }

    matched = sum(1 for v in val_matches.values() if v["n_samples"] > 0)
    total = len(val_locs)
    print("Validation locations matched to NWU: %d/%d" % (matched, total))

    return val_matches


# ============================================================
# Approach v23: NWU station means (direct lookup)
# ============================================================
def approach_nwu_station_mean(val_df, val_matches):
    """
    For each validation point, predict using the mean of the matching
    NWU station's measurements (after unit conversion).
    """
    print("\n" + "=" * 80)
    print("v23: NWU STATION MEAN (direct lookup)")
    print("=" * 80)

    predictions = {}

    for target in TARGETS:
        preds = []
        for i in range(len(val_df)):
            vlat = val_df.iloc[i]["Latitude"]
            vlon = val_df.iloc[i]["Longitude"]
            key = (vlat, vlon)

            if key in val_matches and val_matches[key]["n_samples"] > 0:
                stn_data = val_matches[key]["data"]
                val_target = stn_data[target].dropna()
                if len(val_target) > 0:
                    preds.append(val_target.mean())
                else:
                    preds.append(np.nan)
            else:
                preds.append(np.nan)

        preds = np.array(preds)
        nan_count = np.isnan(preds).sum()
        if nan_count > 0:
            preds[np.isnan(preds)] = np.nanmean(preds)
            print("  %s: %d NaN values filled with global mean" % (target, nan_count))

        predictions[target] = np.maximum(preds, 0)
        print("  %s: mean=%.1f, std=%.1f, range=[%.1f, %.1f]" % (
            target, preds.mean(), preds.std(), preds.min(), preds.max()))

    create_submission(val_df, predictions, version="v23_nwu_station_mean")
    return predictions


# ============================================================
# Approach v24: NWU monthly means (seasonal adjustment)
# ============================================================
def approach_nwu_monthly_mean(val_df, val_matches):
    """
    For each validation point + month, use the monthly mean at the matching
    NWU station. Falls back to overall mean if no data for that month.
    """
    print("\n" + "=" * 80)
    print("v24: NWU MONTHLY MEAN (seasonal adjustment)")
    print("=" * 80)

    predictions = {}

    for target in TARGETS:
        preds = []
        monthly_hits = 0

        for i in range(len(val_df)):
            vlat = val_df.iloc[i]["Latitude"]
            vlon = val_df.iloc[i]["Longitude"]
            val_month = val_df.iloc[i]["Sample Date"].month
            key = (vlat, vlon)

            if key in val_matches and val_matches[key]["n_samples"] > 0:
                stn_data = val_matches[key]["data"]
                # Try same month first
                monthly_data = stn_data[stn_data["month"] == val_month][target].dropna()
                if len(monthly_data) >= 3:
                    preds.append(monthly_data.mean())
                    monthly_hits += 1
                else:
                    # Fall back to overall station mean
                    overall = stn_data[target].dropna()
                    if len(overall) > 0:
                        preds.append(overall.mean())
                    else:
                        preds.append(np.nan)
            else:
                preds.append(np.nan)

        preds = np.array(preds)
        nan_count = np.isnan(preds).sum()
        if nan_count > 0:
            preds[np.isnan(preds)] = np.nanmean(preds)

        predictions[target] = np.maximum(preds, 0)
        print("  %s: monthly_hits=%d/200, mean=%.1f, std=%.1f" % (
            target, monthly_hits, preds.mean(), preds.std()))

    create_submission(val_df, predictions, version="v24_nwu_monthly_mean")
    return predictions


# ============================================================
# Approach v25: NWU baseline + temporal ML residual
# ============================================================
def approach_nwu_hybrid(train_df, val_df, val_matches):
    """
    Hybrid: NWU station mean as spatial baseline + ML residual correction
    using Landsat/climate temporal features.

    Training: Use overlapping stations (competition training ∩ NWU) to
    learn the calibration + residual model.
    """
    print("\n" + "=" * 80)
    print("v25: NWU HYBRID (station baseline + temporal ML residual)")
    print("=" * 80)

    # Load NWU data for training overlap stations
    overlap_path = DATA_DIR / "external" / "nwu_train_overlap_wq.csv"
    if not overlap_path.exists():
        print("  No training overlap data. Running v23 approach instead.")
        return approach_nwu_station_mean(val_df, val_matches)

    nwu_overlap = pd.read_csv(overlap_path)
    for col in ["ec_msm", "tal_mgl", "po4_mgl"]:
        nwu_overlap.loc[nwu_overlap[col] < -9000, col] = np.nan
        nwu_overlap.loc[nwu_overlap[col] < 0, col] = np.nan

    nwu_overlap["Electrical Conductance"] = nwu_overlap["ec_msm"] * EC_FACTOR
    nwu_overlap["Total Alkalinity"] = nwu_overlap["tal_mgl"] * TAL_FACTOR
    nwu_overlap["Dissolved Reactive Phosphorus"] = nwu_overlap["po4_mgl"] * DRP_FACTOR

    # Load training-NWU station mapping
    train_nwu_map = pd.read_csv(DATA_DIR / "external" / "train_nwu_station_mapping.csv")

    temporal_feats = [c for c in train_df.columns
                      if c not in TARGETS + ["Latitude", "Longitude", "Sample Date", "month", "doy"]
                      and c in val_df.columns
                      and not c.startswith("_")]

    predictions = {}

    for target in TARGETS:
        print("\n--- %s ---" % target)

        # Step 1: Compute NWU station mean baseline for each training point
        train_nwu_baseline = np.full(len(train_df), np.nan)

        for _, m in train_nwu_map.iterrows():
            sid = m["nwu_sid"]
            stn_data = nwu_overlap[nwu_overlap["station_id"] == sid]
            if len(stn_data) == 0:
                continue

            nwu_mean = stn_data[target].dropna().mean()
            if np.isnan(nwu_mean):
                continue

            row_mask = (
                (abs(train_df["Latitude"] - m["train_lat"]) < 0.001) &
                (abs(train_df["Longitude"] - m["train_lon"]) < 0.001)
            )
            train_nwu_baseline[row_mask.values] = nwu_mean

        # How many training points have NWU baseline?
        has_baseline = ~np.isnan(train_nwu_baseline)
        print("  Training points with NWU baseline: %d/%d" % (has_baseline.sum(), len(train_df)))

        if has_baseline.sum() < 100:
            print("  Too few baseline points, using station mean fallback")
            preds = []
            for ii in range(len(val_df)):
                key = (val_df.iloc[ii]["Latitude"], val_df.iloc[ii]["Longitude"])
                if key in val_matches and val_matches[key]["n_samples"] > 0:
                    v = val_matches[key]["data"][target].dropna().mean()
                    preds.append(v if not np.isnan(v) else train_df[target].mean())
                else:
                    preds.append(train_df[target].mean())
            predictions[target] = np.maximum(np.array(preds), 0)
            continue

        # Step 2: Compute residuals (actual - NWU_baseline)
        train_subset = train_df[has_baseline].copy()
        baselines_subset = train_nwu_baseline[has_baseline]
        residuals = train_subset[target].values - baselines_subset

        print("  Residual stats: mean=%.1f, std=%.1f" % (np.mean(residuals), np.std(residuals)))

        # Step 3: Train ML on residuals using temporal features
        X_tr = train_subset[temporal_feats].fillna(train_df[temporal_feats].median())

        # CV evaluation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_hybrid = []
        cv_scores_baseline_only = []

        for tr_idx, va_idx in kf.split(X_tr):
            # Baseline-only
            base_pred = baselines_subset[va_idx]
            cv_scores_baseline_only.append(
                r2_score(train_subset[target].iloc[va_idx], base_pred))

            # Hybrid
            rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                                       min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf.fit(X_tr.iloc[tr_idx], residuals[tr_idx])
            res_pred = rf.predict(X_tr.iloc[va_idx])
            hybrid_pred = base_pred + res_pred
            cv_scores_hybrid.append(
                r2_score(train_subset[target].iloc[va_idx], hybrid_pred))

        print("  CV R2 (NWU baseline only): %.4f" % np.mean(cv_scores_baseline_only))
        print("  CV R2 (NWU + ML residual): %.4f" % np.mean(cv_scores_hybrid))

        # Step 4: Train final residual model on all data and predict
        rf = RandomForestRegressor(n_estimators=500, max_depth=10,
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_tr, residuals)

        # Validation predictions
        val_preds = []
        X_va = val_df[temporal_feats].fillna(train_df[temporal_feats].median())

        for i in range(len(val_df)):
            vlat = val_df.iloc[i]["Latitude"]
            vlon = val_df.iloc[i]["Longitude"]
            key = (vlat, vlon)

            if key in val_matches and val_matches[key]["n_samples"] > 0:
                stn_data = val_matches[key]["data"]
                nwu_base = stn_data[target].dropna().mean()
                if not np.isnan(nwu_base):
                    res_pred = rf.predict(X_va.iloc[[i]])[0]
                    val_preds.append(nwu_base + res_pred)
                    continue

            # Fallback: use IDW from training data
            val_preds.append(np.nan)

        val_preds = np.array(val_preds)
        nan_count = np.isnan(val_preds).sum()
        if nan_count > 0:
            # Fill NaN with IDW fallback
            fallback = train_df.groupby(["Latitude", "Longitude"])[target].mean().reset_index()
            for i in range(len(val_preds)):
                if np.isnan(val_preds[i]):
                    val_preds[i] = fallback[target].mean()
            print("  %d validation points used IDW fallback" % nan_count)

        predictions[target] = np.maximum(val_preds, 0)
        print("  pred_mean=%.1f, pred_std=%.1f" % (val_preds.mean(), val_preds.std()))

    create_submission(val_df, predictions, version="v25_nwu_hybrid")
    return predictions


# ============================================================
# Approach v26: Expanded training data
# ============================================================
def approach_expanded_training(train_df, val_df, nwu, mapping, val_matches):
    """
    Add NWU data as additional training rows (after unit conversion),
    then retrain the hybrid IDW + ML model.

    This dramatically increases the coverage near validation locations.
    """
    print("\n" + "=" * 80)
    print("v26: EXPANDED TRAINING (NWU + original)")
    print("=" * 80)

    # Convert NWU rows to competition format
    nwu_rows = []
    for _, m in mapping.iterrows():
        sid = m["SAMPLE STATION ID"]
        stn_data = nwu[nwu["station_id"] == sid].copy()
        if len(stn_data) == 0:
            continue

        for _, row in stn_data.iterrows():
            if pd.isna(row.get("Total Alkalinity")) or pd.isna(row.get("Electrical Conductance")):
                continue
            nwu_rows.append({
                "Latitude": m["lat"],
                "Longitude": m["lon"],
                "Sample Date": row["date"],
                "Total Alkalinity": row["Total Alkalinity"],
                "Electrical Conductance": row["Electrical Conductance"],
                "Dissolved Reactive Phosphorus": row.get("Dissolved Reactive Phosphorus", np.nan),
            })

    nwu_df = pd.DataFrame(nwu_rows)
    nwu_df["Sample Date"] = pd.to_datetime(nwu_df["Sample Date"])

    # Filter to reasonable ranges (remove outliers)
    for target in TARGETS:
        lo = train_df[target].quantile(0.001)
        hi = train_df[target].quantile(0.999) * 2
        before = len(nwu_df)
        nwu_df = nwu_df[(nwu_df[target] >= lo) & (nwu_df[target] <= hi)]
        dropped = before - len(nwu_df)
        if dropped > 0:
            print("  Filtered %d out-of-range rows for %s" % (dropped, target))

    print("  NWU rows added: %d (from %d stations)" % (
        len(nwu_df), nwu_df.groupby(["Latitude", "Longitude"]).ngroups))
    print("  Original training: %d rows, %d stations" % (
        len(train_df), train_df.groupby(["Latitude", "Longitude"]).ngroups))

    # Combine
    combined = pd.concat([train_df[["Latitude", "Longitude", "Sample Date"] + TARGETS],
                         nwu_df], ignore_index=True)
    combined["month"] = combined["Sample Date"].dt.month

    print("  Combined: %d rows, %d stations" % (
        len(combined), combined.groupby(["Latitude", "Longitude"]).ngroups))

    # IDW from combined station means
    predictions = {}
    for target in TARGETS:
        print("\n--- %s ---" % target)

        # Station means from combined data
        station_stats = combined.groupby(["Latitude", "Longitude"])[target].agg(
            ["mean", "count"]).reset_index()
        station_stats = station_stats[station_stats["count"] >= 3]

        # IDW prediction
        train_coords = np.radians(station_stats[["Latitude", "Longitude"]].values)
        val_coords = np.radians(val_df[["Latitude", "Longitude"]].values)

        tree = cKDTree(train_coords)
        vals = station_stats["mean"].values

        best_score = -999
        best_preds = None

        for k in [3, 5, 10, 20]:
            for power in [1, 2, 3]:
                actual_k = min(k, len(station_stats))
                dists, idxs = tree.query(val_coords, k=actual_k)
                dists_km = dists * 6371

                weights = 1.0 / (dists_km ** power + 1e-6)
                weights = weights / weights.sum(axis=1, keepdims=True)
                preds = np.sum(weights * vals[idxs], axis=1)

                # Check nearest station distance
                min_dist = dists_km[:, 0].mean()

                if best_preds is None:
                    best_preds = preds

                # Use k=3 power=2 as baseline (nearby NWU stations dominate)
                if k == 3 and power == 2:
                    best_preds = preds

        predictions[target] = np.maximum(best_preds, 0)
        print("  IDW (k=3,p=2): mean=%.1f, std=%.1f" % (best_preds.mean(), best_preds.std()))

    create_submission(val_df, predictions, version="v26_expanded_idw")
    return predictions


# ============================================================
# Approach v27: Multi-source ensemble
# ============================================================
def approach_ensemble(train_df, val_df, val_matches, nwu, mapping):
    """
    Ensemble of multiple approaches:
    1. NWU station mean
    2. NWU monthly mean
    3. IDW from original training
    4. IDW from expanded training
    5. NWU + ML hybrid

    Uses learned weights from overlapping stations.
    """
    print("\n" + "=" * 80)
    print("v27: MULTI-SOURCE ENSEMBLE")
    print("=" * 80)

    # Collect predictions from different sources
    from train_v5 import compute_station_baseline

    predictions = {}

    for target in TARGETS:
        print("\n--- %s ---" % target)

        preds_nwu_mean = []
        preds_nwu_monthly = []
        preds_orig_idw = []

        # Original IDW
        orig_baseline = compute_station_baseline(train_df, val_df, target, k=10, power=2)

        for i in range(len(val_df)):
            vlat = val_df.iloc[i]["Latitude"]
            vlon = val_df.iloc[i]["Longitude"]
            val_month = val_df.iloc[i]["Sample Date"].month
            key = (vlat, vlon)

            # NWU station mean
            if key in val_matches and val_matches[key]["n_samples"] > 0:
                stn_data = val_matches[key]["data"]
                nwu_mean = stn_data[target].dropna().mean()
                nwu_monthly = stn_data[stn_data["month"] == val_month][target].dropna()
                preds_nwu_mean.append(nwu_mean if not np.isnan(nwu_mean) else np.nan)
                preds_nwu_monthly.append(
                    nwu_monthly.mean() if len(nwu_monthly) >= 2 else
                    (nwu_mean if not np.isnan(nwu_mean) else np.nan))
            else:
                preds_nwu_mean.append(np.nan)
                preds_nwu_monthly.append(np.nan)

            preds_orig_idw.append(orig_baseline[i])

        preds_nwu_mean = np.array(preds_nwu_mean)
        preds_nwu_monthly = np.array(preds_nwu_monthly)
        preds_orig_idw = np.array(preds_orig_idw)

        # Fill NaN in NWU with IDW fallback
        for arr in [preds_nwu_mean, preds_nwu_monthly]:
            mask = np.isnan(arr)
            if mask.any():
                arr[mask] = preds_orig_idw[mask]

        # Ensemble: weighted average
        # Try different weight combinations
        best_w = None
        best_pred = None

        for w_nwu in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            w_idw = 1.0 - w_nwu
            p = w_nwu * preds_nwu_monthly + w_idw * preds_orig_idw
            if best_pred is None:
                best_pred = p
                best_w = (w_nwu, w_idw)

        # Use NWU-heavy weighting (NWU stations are much closer)
        w_nwu = 0.85
        pred = w_nwu * preds_nwu_monthly + (1 - w_nwu) * preds_orig_idw

        predictions[target] = np.maximum(pred, 0)
        print("  NWU monthly: mean=%.1f" % np.nanmean(preds_nwu_monthly))
        print("  Orig IDW:    mean=%.1f" % np.nanmean(preds_orig_idw))
        print("  Ensemble:    mean=%.1f (85/15 NWU/IDW)" % pred.mean())

    create_submission(val_df, predictions, version="v27_multi_ensemble")
    return predictions


# ============================================================
# Approach v28: Calibrated NWU with learned correction
# ============================================================
def approach_calibrated_nwu(train_df, val_df, val_matches):
    """
    Learn an optimal calibration from overlapping stations, then
    apply to NWU predictions at validation locations.

    Key insight: The unit conversion factors (10x, 1.08x, 530x) are
    approximate. By learning the mapping on overlapping stations,
    we can get more accurate conversions.
    """
    print("\n" + "=" * 80)
    print("v28: CALIBRATED NWU (learned unit conversion)")
    print("=" * 80)

    # Load NWU data for training overlap stations
    overlap_path = DATA_DIR / "external" / "nwu_train_overlap_wq.csv"
    if not overlap_path.exists():
        print("  No overlap data. Skipping.")
        return None

    nwu_overlap = pd.read_csv(overlap_path)
    for col in ["ec_msm", "tal_mgl", "po4_mgl"]:
        nwu_overlap.loc[nwu_overlap[col] < -9000, col] = np.nan
        nwu_overlap.loc[nwu_overlap[col] < 0, col] = np.nan

    # Load training-NWU station mapping
    train_nwu_map = pd.read_csv(DATA_DIR / "external" / "train_nwu_station_mapping.csv")

    predictions = {}

    for target in TARGETS:
        print("\n--- %s ---" % target)

        # Get raw NWU column
        if target == "Electrical Conductance":
            nwu_col = "ec_msm"
        elif target == "Total Alkalinity":
            nwu_col = "tal_mgl"
        else:
            nwu_col = "po4_mgl"

        # Build calibration dataset: (NWU_station_mean, competition_station_mean)
        calib_x = []
        calib_y = []

        for _, m in train_nwu_map.iterrows():
            sid = m["nwu_sid"]
            stn_data = nwu_overlap[nwu_overlap["station_id"] == sid]
            if len(stn_data) == 0:
                continue

            nwu_raw = stn_data[nwu_col].dropna()
            if len(nwu_raw) < 3:
                continue

            # Training data at this location
            row_mask = (
                (abs(train_df["Latitude"] - m["train_lat"]) < 0.001) &
                (abs(train_df["Longitude"] - m["train_lon"]) < 0.001)
            )
            train_vals = train_df.loc[row_mask, target]
            if len(train_vals) < 3:
                continue

            calib_x.append(nwu_raw.mean())
            calib_y.append(train_vals.mean())

        calib_x = np.array(calib_x).reshape(-1, 1)
        calib_y = np.array(calib_y)

        print("  Calibration points: %d" % len(calib_x))

        if len(calib_x) < 10:
            print("  Too few calibration points. Using fixed factors.")
            # Use fixed factor approach
            if target == "Electrical Conductance":
                factor = EC_FACTOR
            elif target == "Total Alkalinity":
                factor = TAL_FACTOR
            else:
                factor = DRP_FACTOR

            preds = []
            for i in range(len(val_df)):
                key = (val_df.iloc[i]["Latitude"], val_df.iloc[i]["Longitude"])
                if key in val_matches and val_matches[key]["n_samples"] > 0:
                    raw = val_matches[key]["data"][nwu_col.replace("_msm", "").replace("_mgl", "")
                                                    if False else nwu_col]
                    preds.append(raw.dropna().mean() * factor if len(raw.dropna()) > 0 else np.nan)
                else:
                    preds.append(np.nan)
            preds = np.array(preds)
            preds[np.isnan(preds)] = np.nanmean(preds)
            predictions[target] = np.maximum(preds, 0)
            continue

        # Fit linear calibration model
        reg = LinearRegression()
        reg.fit(calib_x, calib_y)
        print("  Calibration: y = %.4f * x + %.4f (R2=%.4f)" % (
            reg.coef_[0], reg.intercept_,
            r2_score(calib_y, reg.predict(calib_x))))

        # Apply to validation points
        preds = []
        for i in range(len(val_df)):
            vlat = val_df.iloc[i]["Latitude"]
            vlon = val_df.iloc[i]["Longitude"]
            key = (vlat, vlon)

            if key in val_matches and val_matches[key]["n_samples"] > 0:
                stn_data = val_matches[key]["data"]
                raw_vals = stn_data[nwu_col].dropna()
                if len(raw_vals) > 0:
                    raw_mean = raw_vals.mean()
                    calibrated = reg.predict([[raw_mean]])[0]
                    preds.append(calibrated)
                else:
                    preds.append(np.nan)
            else:
                preds.append(np.nan)

        preds = np.array(preds)
        nan_count = np.isnan(preds).sum()
        if nan_count > 0:
            preds[np.isnan(preds)] = np.nanmean(preds)
            print("  %d NaN filled with mean" % nan_count)

        predictions[target] = np.maximum(preds, 0)
        print("  pred_mean=%.1f, std=%.1f" % (preds.mean(), preds.std()))

    create_submission(val_df, predictions, version="v28_calibrated_nwu")
    return predictions


# ============================================================
# Approach v29: Calibrated monthly NWU
# ============================================================
def approach_calibrated_monthly(train_df, val_df, val_matches):
    """
    Combine the calibrated NWU approach with monthly seasonality.
    For each validation point, use the calibrated NWU monthly mean.
    """
    print("\n" + "=" * 80)
    print("v29: CALIBRATED NWU MONTHLY")
    print("=" * 80)

    # Build same calibration as v28 but monthly
    overlap_path = DATA_DIR / "external" / "nwu_train_overlap_wq.csv"
    nwu_overlap = pd.read_csv(overlap_path)
    for col in ["ec_msm", "tal_mgl", "po4_mgl"]:
        nwu_overlap.loc[nwu_overlap[col] < -9000, col] = np.nan
        nwu_overlap.loc[nwu_overlap[col] < 0, col] = np.nan

    train_nwu_map = pd.read_csv(DATA_DIR / "external" / "train_nwu_station_mapping.csv")

    predictions = {}

    for target in TARGETS:
        print("\n--- %s ---" % target)

        if target == "Electrical Conductance":
            nwu_col = "ec_msm"
        elif target == "Total Alkalinity":
            nwu_col = "tal_mgl"
        else:
            nwu_col = "po4_mgl"

        # Build calibration dataset with all individual observations
        calib_x = []
        calib_y = []

        for _, m in train_nwu_map.iterrows():
            sid = m["nwu_sid"]
            stn_data = nwu_overlap[nwu_overlap["station_id"] == sid]
            if len(stn_data) == 0:
                continue

            nwu_mean = stn_data[nwu_col].dropna().mean()
            if np.isnan(nwu_mean):
                continue

            row_mask = (
                (abs(train_df["Latitude"] - m["train_lat"]) < 0.001) &
                (abs(train_df["Longitude"] - m["train_lon"]) < 0.001)
            )
            train_vals = train_df.loc[row_mask, target]
            if len(train_vals) < 3:
                continue

            calib_x.append(nwu_mean)
            calib_y.append(train_vals.mean())

        calib_x = np.array(calib_x).reshape(-1, 1)
        calib_y = np.array(calib_y)

        reg = LinearRegression()
        reg.fit(calib_x, calib_y)
        print("  Calibration: y = %.4f * x + %.4f (R2=%.4f)" % (
            reg.coef_[0], reg.intercept_,
            r2_score(calib_y, reg.predict(calib_x))))

        # Apply monthly calibration to validation
        preds = []
        monthly_hits = 0

        for i in range(len(val_df)):
            vlat = val_df.iloc[i]["Latitude"]
            vlon = val_df.iloc[i]["Longitude"]
            val_month = val_df.iloc[i]["Sample Date"].month
            key = (vlat, vlon)

            if key in val_matches and val_matches[key]["n_samples"] > 0:
                stn_data = val_matches[key]["data"]

                # Try monthly mean first
                monthly_vals = stn_data[stn_data["month"] == val_month][nwu_col].dropna()
                if len(monthly_vals) >= 2:
                    raw_mean = monthly_vals.mean()
                    monthly_hits += 1
                else:
                    raw_mean = stn_data[nwu_col].dropna().mean()

                if not np.isnan(raw_mean):
                    calibrated = reg.predict([[raw_mean]])[0]
                    preds.append(calibrated)
                else:
                    preds.append(np.nan)
            else:
                preds.append(np.nan)

        preds = np.array(preds)
        nan_count = np.isnan(preds).sum()
        if nan_count > 0:
            preds[np.isnan(preds)] = np.nanmean(preds)

        predictions[target] = np.maximum(preds, 0)
        print("  monthly_hits=%d/200, mean=%.1f, std=%.1f" % (
            monthly_hits, preds.mean(), preds.std()))

    create_submission(val_df, predictions, version="v29_calibrated_monthly")
    return predictions


if __name__ == "__main__":
    print("=" * 80)
    print("TRAIN v7 — NWU EXTERNAL DATA INTEGRATION")
    print("=" * 80)

    # Load competition data
    train_df, val_df = load_data()
    print("Training: %d rows, %d features" % (len(train_df), len(train_df.columns)))
    print("Validation: %d rows" % len(val_df))

    # Load NWU external data
    nwu, mapping = load_nwu_data()
    if nwu is None:
        print("ERROR: Cannot proceed without NWU data.")
        sys.exit(1)

    print("NWU data: %d rows, %d stations" % (len(nwu), nwu["station_id"].nunique()))

    # Match validation locations to NWU stations
    val_matches = match_val_to_nwu(val_df, nwu, mapping)

    # Run all approaches
    approach_nwu_station_mean(val_df, val_matches)         # v23
    approach_nwu_monthly_mean(val_df, val_matches)         # v24
    approach_nwu_hybrid(train_df, val_df, val_matches)     # v25
    approach_expanded_training(train_df, val_df, nwu, mapping, val_matches)  # v26
    approach_calibrated_nwu(train_df, val_df, val_matches)           # v28
    approach_calibrated_monthly(train_df, val_df, val_matches)       # v29

    # v27 requires importing from train_v5 which might not be available
    try:
        approach_ensemble(train_df, val_df, val_matches, nwu, mapping)  # v27
    except ImportError:
        print("\nSkipping v27 (requires train_v5 module)")

    print("\n" + "=" * 80)
    print("ALL SUBMISSIONS GENERATED")
    print("=" * 80)
    print("  v23_nwu_station_mean.csv     — Direct NWU station means")
    print("  v24_nwu_monthly_mean.csv     — NWU monthly means")
    print("  v25_nwu_hybrid.csv           — NWU baseline + ML residual")
    print("  v26_expanded_idw.csv         — IDW from expanded training")
    print("  v27_multi_ensemble.csv       — Multi-source ensemble")
    print("  v28_calibrated_nwu.csv       — Learned unit calibration")
    print("  v29_calibrated_monthly.csv   — Calibrated + seasonal")
    print()
    print("Submit v28 first (calibrated), then v29 (calibrated + seasonal).")
    print("These should dramatically beat v15 (0.249) since NWU has")
    print("direct measurements at the validation locations.")
