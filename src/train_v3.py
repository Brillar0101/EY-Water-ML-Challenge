"""
Training pipeline v3 — Incremental feature selection.

Lesson learned: 99 features → negative leaderboard score (-0.068).
Benchmark with 4 features → leaderboard 0.20.
More features ≠ better. We must add features ONLY if they help generalization.

This script:
    1. Reproduces the benchmark (4 features, RF)
    2. Tests each feature group's marginal contribution
    3. Selects features that actually improve station-split CV
    4. Generates multiple submissions to test on leaderboard

Usage:
    python src/train_v3.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data_loader import build_full_dataset
from feature_builder import build_features, get_feature_columns
from submission import create_submission


TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]


def load_data():
    print("Loading data...")
    train_df = build_full_dataset("train")
    train_df = build_features(train_df, is_training=True)
    val_df = build_full_dataset("val")
    val_df = build_features(val_df, is_training=False)
    return train_df, val_df


def get_station_ids(df):
    return df.apply(lambda r: f"{r['Latitude']:.6f}_{r['Longitude']:.6f}", axis=1)


def station_cv(model_fn, X, y, station_ids, n_folds=5, use_log=False, y_raw=None):
    """Station-based cross-validation."""
    unique = station_ids.unique()
    np.random.seed(42)
    shuffled = unique.copy()
    np.random.shuffle(shuffled)
    fold_size = len(shuffled) // n_folds

    scores = []
    for i in range(n_folds):
        val_stations = set(shuffled[i * fold_size:(i + 1) * fold_size])
        va_mask = station_ids.isin(val_stations)
        tr_idx = np.where(~va_mask)[0]
        va_idx = np.where(va_mask)[0]

        X_tr = X.iloc[tr_idx].fillna(X.iloc[tr_idx].median())
        X_va = X.iloc[va_idx].fillna(X.iloc[tr_idx].median())

        model = model_fn()
        model.fit(X_tr, y.iloc[tr_idx])
        preds = model.predict(X_va)

        if use_log and y_raw is not None:
            preds = np.maximum(np.expm1(preds), 0)
            score = r2_score(y_raw.iloc[va_idx], preds)
        else:
            score = r2_score(y.iloc[va_idx], preds)
        scores.append(score)

    return np.mean(scores), scores


# ============================================================
# Experiment 0: Reproduce benchmark exactly
# ============================================================
def exp0_benchmark(train_df, val_df, station_ids):
    """Reproduce benchmark: RF + 4 features."""
    print("\n" + "=" * 80)
    print("EXP 0: REPRODUCE BENCHMARK (RF, 4 features)")
    print("=" * 80)

    benchmark_feats = ["swir22", "NDMI", "MNDWI", "pet"]
    available = [f for f in benchmark_feats if f in train_df.columns]

    def make_rf():
        return RandomForestRegressor(n_estimators=200, max_depth=15,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1)

    predictions = {}

    for target in TARGETS:
        y = train_df[target]
        X = train_df[available]
        mean_r2, fold_scores = station_cv(make_rf, X, y, station_ids)
        print(f"  {target:40s} R² = {mean_r2:.4f}  {[f'{s:.3f}' for s in fold_scores]}")

        # Train on all for submission
        X_all = X.fillna(X.median())
        X_val = val_df[available].fillna(X.median())
        model = make_rf()
        model.fit(X_all, y)
        predictions[target] = np.maximum(model.predict(X_val), 0)

    create_submission(val_df, predictions, version="v4_benchmark_rf")
    return predictions


# ============================================================
# Experiment 1: Feature group ablation (add one group at a time)
# ============================================================
def exp1_feature_groups(train_df, station_ids):
    """Test marginal contribution of each feature group."""
    print("\n" + "=" * 80)
    print("EXP 1: FEATURE GROUP MARGINAL CONTRIBUTION")
    print("=" * 80)

    base_feats = ["swir22", "NDMI", "MNDWI", "pet"]

    feature_groups = {
        "landsat_extra": ["nir", "green", "swir16"],
        "temporal": [c for c in train_df.columns if c.startswith(("month_", "season_", "quarter", "day_of_year"))
                     or c in ["month_sin", "month_cos", "doy_sin", "doy_cos"]],
        "climate_full": ["tmax", "tmin", "ppt", "vap", "srad", "ws", "aet",
                         "q", "def", "soil", "pdsi", "vpd"],
        "climate_lags": [c for c in train_df.columns if "_lag1" in c or "_roll3" in c],
        "terrain": ["elevation", "slope", "aspect", "roughness", "twi"],
        "soil": [c for c in train_df.columns if c.startswith("soil_")],
        "landcover_1km": [c for c in train_df.columns if c.startswith("lc_1km_")],
        "landcover_5km": [c for c in train_df.columns if c.startswith("lc_5km_")],
        "missing_flags": [c for c in train_df.columns if c.endswith("_missing")],
        "landsat_indices": [c for c in train_df.columns if c in [
            "NDWI", "MSI", "SWIR_ratio", "green_swir16_ratio"]],
    }

    def make_rf():
        return RandomForestRegressor(n_estimators=200, max_depth=15,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1)

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")

        # Baseline with just benchmark features
        avail_base = [f for f in base_feats if f in train_df.columns]
        base_r2, _ = station_cv(make_rf, train_df[avail_base], y, station_ids)
        print(f"  {'BASE (4 feats)':30s} R² = {base_r2:.4f}")

        # Add each group
        best_groups = []
        for name, feats in feature_groups.items():
            avail = [f for f in feats if f in train_df.columns]
            if not avail:
                continue
            combined = avail_base + avail
            r2, _ = station_cv(make_rf, train_df[combined], y, station_ids)
            delta = r2 - base_r2
            marker = "+" if delta > 0 else "-"
            print(f"  + {name:28s} ({len(avail):2d} feats) → R² = {r2:.4f}  (Δ = {marker}{abs(delta):.4f})")
            if delta > 0.005:
                best_groups.append((name, avail, delta))

        if best_groups:
            best_groups.sort(key=lambda x: -x[2])
            print(f"\n  Helpful groups: {[g[0] for g in best_groups]}")

            # Try combining all helpful groups
            all_helpful = avail_base.copy()
            for _, feats, _ in best_groups:
                all_helpful.extend(feats)
            r2_combined, _ = station_cv(make_rf, train_df[all_helpful], y, station_ids)
            print(f"  All helpful combined ({len(all_helpful)} feats): R² = {r2_combined:.4f}")


# ============================================================
# Experiment 2: Per-target greedy feature selection
# ============================================================
def exp2_greedy_selection(train_df, station_ids):
    """Greedy forward selection per target."""
    print("\n" + "=" * 80)
    print("EXP 2: GREEDY FORWARD FEATURE SELECTION")
    print("=" * 80)

    all_feats = get_feature_columns(train_df)

    def make_rf():
        return RandomForestRegressor(n_estimators=200, max_depth=12,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1)

    best_features_per_target = {}

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")

        selected = []
        remaining = list(all_feats)
        best_score = -999

        for step in range(20):  # Select up to 20 features
            best_feat = None
            best_step_score = -999

            for feat in remaining:
                candidate = selected + [feat]
                r2, _ = station_cv(make_rf, train_df[candidate], y, station_ids)
                if r2 > best_step_score:
                    best_step_score = r2
                    best_feat = feat

            if best_feat is None or best_step_score <= best_score + 0.001:
                break

            selected.append(best_feat)
            remaining.remove(best_feat)
            best_score = best_step_score
            print(f"  Step {step+1:2d}: +{best_feat:35s} → R² = {best_score:.4f} ({len(selected)} feats)")

        best_features_per_target[target] = selected
        print(f"  Final: {len(selected)} features, R² = {best_score:.4f}")

    return best_features_per_target


# ============================================================
# Experiment 3: XGBoost with selected features
# ============================================================
def exp3_xgb_selected(train_df, val_df, station_ids, best_features):
    """Train XGBoost with per-target selected features."""
    from xgboost import XGBRegressor

    print("\n" + "=" * 80)
    print("EXP 3: XGBOOST WITH SELECTED FEATURES")
    print("=" * 80)

    predictions = {}

    for target in TARGETS:
        feats = best_features.get(target)
        if not feats:
            feats = ["swir22", "NDMI", "MNDWI", "pet"]
        feats = [f for f in feats if f in val_df.columns]

        use_log = (target == "Dissolved Reactive Phosphorus")
        y_raw = train_df[target]
        y = np.log1p(y_raw) if use_log else y_raw

        print(f"\n--- {target} ({len(feats)} features) ---")
        print(f"  Features: {feats}")

        # Test multiple XGB configs
        configs = {
            "d6_moderate": dict(
                n_estimators=600, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
                reg_alpha=0.5, reg_lambda=3.0
            ),
            "d8_light": dict(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
                reg_alpha=0.3, reg_lambda=2.0
            ),
            "d4_heavy": dict(
                n_estimators=800, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=15,
                reg_alpha=1.0, reg_lambda=5.0
            ),
        }

        best_r2 = -999
        best_config = None
        best_preds = None

        for name, params in configs.items():
            def make_model(p=params):
                return XGBRegressor(**p, tree_method="gpu_hist", device="cuda", random_state=42)

            X = train_df[feats]
            r2, fold_scores = station_cv(make_model, X, y, station_ids,
                                          use_log=use_log, y_raw=y_raw if use_log else None)
            print(f"  {name:20s} → R² = {r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_config = name

        print(f"  BEST: {best_config} (R² = {best_r2:.4f})")

        # Train on all data with best config
        best_params = configs[best_config]
        model = XGBRegressor(**best_params, tree_method="gpu_hist", device="cuda", random_state=42)
        X_train = train_df[feats].fillna(train_df[feats].median())
        X_val = val_df[feats].fillna(train_df[feats].median())
        model.fit(X_train, y)
        pred = model.predict(X_val)

        if use_log:
            pred = np.maximum(np.expm1(pred), 0)
        pred = np.maximum(pred, 0)
        predictions[target] = pred

    return predictions


# ============================================================
# Experiment 4: RF with selected features (simpler model)
# ============================================================
def exp4_rf_selected(train_df, val_df, station_ids, best_features):
    """Train RF with per-target selected features (simpler, may generalize better)."""
    print("\n" + "=" * 80)
    print("EXP 4: RANDOM FOREST WITH SELECTED FEATURES")
    print("=" * 80)

    predictions = {}

    for target in TARGETS:
        feats = best_features.get(target)
        if not feats:
            feats = ["swir22", "NDMI", "MNDWI", "pet"]
        feats = [f for f in feats if f in val_df.columns]

        y = train_df[target]
        print(f"\n--- {target} ({len(feats)} features) ---")

        configs = {
            "rf_200_d12": lambda: RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1),
            "rf_500_d15": lambda: RandomForestRegressor(
                n_estimators=500, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1),
            "rf_500_d10": lambda: RandomForestRegressor(
                n_estimators=500, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1),
            "rf_1000_d8": lambda: RandomForestRegressor(
                n_estimators=1000, max_depth=8, min_samples_leaf=15, random_state=42, n_jobs=-1),
        }

        best_r2 = -999
        best_config = None

        for name, factory in configs.items():
            X = train_df[feats]
            r2, _ = station_cv(factory, X, y, station_ids)
            print(f"  {name:20s} → R² = {r2:.4f}")
            if r2 > best_r2:
                best_r2 = r2
                best_config = name

        print(f"  BEST: {best_config} (R² = {best_r2:.4f})")

        # Train on all data
        X_train = train_df[feats].fillna(train_df[feats].median())
        X_val = val_df[feats].fillna(train_df[feats].median())
        model = configs[best_config]()
        model.fit(X_train, y)
        predictions[target] = np.maximum(model.predict(X_val), 0)

    return predictions


# ============================================================
# Experiment 5: Ensemble of RF + XGB with selected features
# ============================================================
def exp5_ensemble(train_df, val_df, station_ids, best_features):
    """Blend RF and XGB predictions."""
    from xgboost import XGBRegressor

    print("\n" + "=" * 80)
    print("EXP 5: RF + XGB ENSEMBLE (selected features)")
    print("=" * 80)

    predictions = {}

    for target in TARGETS:
        feats = best_features.get(target)
        if not feats:
            feats = ["swir22", "NDMI", "MNDWI", "pet"]
        feats = [f for f in feats if f in val_df.columns]

        use_log = (target == "Dissolved Reactive Phosphorus")
        y_raw = train_df[target]
        y = np.log1p(y_raw) if use_log else y_raw

        X_train = train_df[feats].fillna(train_df[feats].median())
        X_val = val_df[feats].fillna(train_df[feats].median())

        # RF prediction
        rf = RandomForestRegressor(n_estimators=500, max_depth=12,
                                    min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y)
        rf_pred = rf.predict(X_val)
        if use_log:
            rf_pred = np.maximum(np.expm1(rf_pred), 0)

        # XGB prediction
        xgb = XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
            reg_alpha=0.5, reg_lambda=3.0,
            tree_method="gpu_hist", device="cuda", random_state=42
        )
        xgb.fit(X_train, y)
        xgb_pred = xgb.predict(X_val)
        if use_log:
            xgb_pred = np.maximum(np.expm1(xgb_pred), 0)

        # Simple average blend
        blend = 0.5 * rf_pred + 0.5 * xgb_pred
        predictions[target] = np.maximum(blend, 0)

        print(f"  {target}: RF mean={rf_pred.mean():.1f}, XGB mean={xgb_pred.mean():.1f}, Blend mean={blend.mean():.1f}")

    return predictions


if __name__ == "__main__":
    train_df, val_df = load_data()
    station_ids = get_station_ids(train_df)
    print(f"Dataset: {len(train_df)} rows, {len(get_feature_columns(train_df))} features, "
          f"{len(station_ids.unique())} stations")

    # Exp 0: Reproduce benchmark
    exp0_benchmark(train_df, val_df, station_ids)

    # Exp 1: Which feature groups help?
    exp1_feature_groups(train_df, station_ids)

    # Exp 2: Greedy feature selection
    best_features = exp2_greedy_selection(train_df, station_ids)

    # Exp 3: XGBoost with selected features
    xgb_preds = exp3_xgb_selected(train_df, val_df, station_ids, best_features)

    # Exp 4: RF with selected features
    rf_preds = exp4_rf_selected(train_df, val_df, station_ids, best_features)

    # Exp 5: Ensemble
    ens_preds = exp5_ensemble(train_df, val_df, station_ids, best_features)

    # Generate submissions
    print("\n" + "=" * 80)
    print("GENERATING SUBMISSIONS")
    print("=" * 80)

    print("\n--- v5: XGBoost selected features ---")
    create_submission(val_df, xgb_preds, version="v5_xgb_selected")

    print("\n--- v6: RF selected features ---")
    create_submission(val_df, rf_preds, version="v6_rf_selected")

    print("\n--- v7: Ensemble (RF+XGB) selected features ---")
    create_submission(val_df, ens_preds, version="v7_ensemble")

    print("\nDone! Submit v4 (benchmark), v5, v6, v7 and compare.")
