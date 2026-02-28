"""
Spatial generalization training pipeline.

The core problem: standard tree models memorize station-specific patterns and fail
when predicting at unseen locations. This script implements multiple strategies to
force models to learn transferable environmental relationships.

Strategies:
    1. Extreme regularization (shallow trees, high lambda)
    2. Per-target feature selection via SHAP
    3. Log1p transform for DRP
    4. Environmental KNN baseline
    5. Two-stage model (station baseline + temporal residual)
    6. Stacking ensemble

Usage:
    python src/train_spatial.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data_loader import build_full_dataset
from feature_builder import build_features, get_feature_columns


TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]

# Feature groups for ablation
STATIC_FEATURES = [
    "elevation", "slope", "aspect", "roughness", "twi",
    "soil_ph_0_5", "soil_ph_15_30", "soil_organic_carbon_0_5", "soil_organic_carbon_15_30",
    "soil_clay_pct_0_5", "soil_clay_pct_15_30", "soil_sand_pct_0_5", "soil_sand_pct_15_30",
    "soil_silt_pct_0_5", "soil_silt_pct_15_30", "soil_cec_0_5", "soil_cec_15_30",
    "soil_bulk_density_0_5", "soil_bulk_density_15_30", "soil_nitrogen_0_5", "soil_nitrogen_15_30",
    "soil_sand_clay_ratio", "soil_nutrient_capacity",
]

LANDCOVER_FEATURES_1KM = [c for c in [
    "lc_1km_tree_cover_pct", "lc_1km_shrubland_pct", "lc_1km_grassland_pct",
    "lc_1km_cropland_pct", "lc_1km_built_up_pct", "lc_1km_bare_sparse_pct",
    "lc_1km_water_bodies_pct", "lc_1km_herbaceous_wetland_pct",
    "lc_1km_vegetation_pct", "lc_1km_human_pct", "lc_1km_water_wetland_pct",
    "lc_1km_shannon_diversity",
]]

LANDCOVER_FEATURES_5KM = [c for c in [
    "lc_5km_tree_cover_pct", "lc_5km_shrubland_pct", "lc_5km_grassland_pct",
    "lc_5km_cropland_pct", "lc_5km_built_up_pct", "lc_5km_bare_sparse_pct",
    "lc_5km_water_bodies_pct", "lc_5km_herbaceous_wetland_pct",
    "lc_5km_vegetation_pct", "lc_5km_human_pct", "lc_5km_water_wetland_pct",
    "lc_5km_shannon_diversity",
]]


def load_data():
    """Load and prepare train/val datasets."""
    print("Loading training data...")
    train_df = build_full_dataset("train")
    train_df = build_features(train_df, is_training=True)

    print("Loading validation data...")
    val_df = build_full_dataset("val")
    val_df = build_features(val_df, is_training=False)

    return train_df, val_df


def assign_spatial_groups(df, n_clusters=8):
    """Assign spatial clusters for cross-validation."""
    coords = df[["Latitude", "Longitude"]].values
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(coords)


def spatial_cv(model_fn, X, y, groups, n_splits=5, return_oof=False):
    """Run spatial GroupKFold CV and return mean val R²."""
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    val_scores = []
    oof = np.full(len(y), np.nan)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        model = model_fn()
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        # Handle NaN in X
        X_tr = X_tr.fillna(X_tr.median())
        X_va = X_va.fillna(X_tr.median())

        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = r2_score(y_va, preds)
        val_scores.append(score)
        oof[va_idx] = preds

    mean_r2 = np.mean(val_scores)
    if return_oof:
        return mean_r2, oof
    return mean_r2


# ============================================================
# Strategy 1: Feature group ablation
# ============================================================
def run_feature_ablation(train_df, groups):
    """Test which feature groups help vs hurt spatial generalization."""
    from xgboost import XGBRegressor

    all_features = get_feature_columns(train_df)

    # Feature sets to test
    feature_sets = {
        "all_99": all_features,
        "static_only": [f for f in STATIC_FEATURES + LANDCOVER_FEATURES_1KM + LANDCOVER_FEATURES_5KM if f in all_features],
        "static+climate": [f for f in STATIC_FEATURES + LANDCOVER_FEATURES_1KM + LANDCOVER_FEATURES_5KM if f in all_features] +
                          [f for f in all_features if f.startswith(("tmax", "tmin", "ppt", "vap", "srad", "ws", "aet", "pet", "q_", "def", "soil_m", "pdsi", "vpd"))],
        "no_landsat": [f for f in all_features if f not in ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI"]],
    }

    def make_model():
        return XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.4, min_child_weight=30,
            reg_alpha=2.0, reg_lambda=10.0, tree_method="gpu_hist",
            device="cuda", random_state=42
        )

    print("\n" + "=" * 80)
    print("STRATEGY 1: FEATURE GROUP ABLATION")
    print("=" * 80)

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")
        for name, feats in feature_sets.items():
            available = [f for f in feats if f in train_df.columns]
            X = train_df[available]
            r2 = spatial_cv(make_model, X, y, groups)
            print(f"  {name:25s} ({len(available):3d} feats) → R² = {r2:.4f}")


# ============================================================
# Strategy 2: Extreme regularization
# ============================================================
def run_regularized_xgb(train_df, groups):
    """XGBoost with extreme regularization to prevent station memorization."""
    from xgboost import XGBRegressor

    all_features = get_feature_columns(train_df)

    configs = {
        "depth2_heavy_reg": dict(
            n_estimators=500, max_depth=2, learning_rate=0.02,
            subsample=0.6, colsample_bytree=0.3, min_child_weight=50,
            reg_alpha=5.0, reg_lambda=20.0
        ),
        "depth3_moderate_reg": dict(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.4, min_child_weight=30,
            reg_alpha=2.0, reg_lambda=10.0
        ),
        "depth4_light_reg": dict(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=20,
            reg_alpha=1.0, reg_lambda=5.0
        ),
    }

    print("\n" + "=" * 80)
    print("STRATEGY 2: REGULARIZATION SWEEP")
    print("=" * 80)

    best_configs = {}

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")
        best_r2, best_name = -999, None

        for name, params in configs.items():
            def make_model(p=params):
                return XGBRegressor(**p, tree_method="gpu_hist", device="cuda", random_state=42)
            X = train_df[all_features]
            r2 = spatial_cv(make_model, X, y, groups)
            print(f"  {name:30s} → R² = {r2:.4f}")
            if r2 > best_r2:
                best_r2 = r2
                best_name = name

        best_configs[target] = (best_name, best_r2)
        print(f"  BEST: {best_name} (R² = {best_r2:.4f})")

    return best_configs


# ============================================================
# Strategy 3: Environmental KNN
# ============================================================
def run_env_knn(train_df, groups):
    """
    K-Nearest Neighbors using environmental features.
    Idea: stations with similar soil/terrain/climate have similar water quality.
    """
    env_features = [f for f in STATIC_FEATURES + LANDCOVER_FEATURES_5KM
                    if f in train_df.columns]

    print("\n" + "=" * 80)
    print("STRATEGY 3: ENVIRONMENTAL KNN")
    print("=" * 80)

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")

        for k in [3, 5, 10, 20]:
            def make_model(k=k):
                return KNeighborsRegressor(n_neighbors=k, weights="distance")

            X = train_df[env_features].copy()
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X.fillna(X.median())),
                                     columns=X.columns, index=X.index)
            r2 = spatial_cv(make_model, X_scaled, y, groups)
            print(f"  k={k:3d}  → R² = {r2:.4f}")


# ============================================================
# Strategy 4: Ridge/ElasticNet (linear baseline)
# ============================================================
def run_linear_baseline(train_df, groups):
    """Linear models as a baseline — may generalize better than trees."""
    all_features = get_feature_columns(train_df)

    print("\n" + "=" * 80)
    print("STRATEGY 4: LINEAR MODELS")
    print("=" * 80)

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")

        # Standardize
        X = train_df[all_features].copy()
        X = X.fillna(X.median())
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        for alpha in [0.1, 1.0, 10.0, 100.0]:
            def make_ridge(a=alpha):
                return Ridge(alpha=a)
            r2 = spatial_cv(make_ridge, X_scaled, y, groups)
            print(f"  Ridge(alpha={alpha:6.1f}) → R² = {r2:.4f}")


# ============================================================
# Strategy 5: Two-stage model
# ============================================================
def run_two_stage(train_df, groups):
    """
    Stage 1: Predict station-level mean from static features.
    Stage 2: Predict residual from temporal features.
    Final = Stage1 + Stage2
    """
    from xgboost import XGBRegressor

    static_feats = [f for f in STATIC_FEATURES + LANDCOVER_FEATURES_1KM + LANDCOVER_FEATURES_5KM
                    if f in train_df.columns]
    temporal_feats = [f for f in get_feature_columns(train_df) if f not in static_feats]

    print("\n" + "=" * 80)
    print("STRATEGY 5: TWO-STAGE MODEL")
    print("=" * 80)

    for target in TARGETS:
        print(f"\n--- {target} ---")

        # Compute station means
        station_means = train_df.groupby(["Latitude", "Longitude"])[target].mean()
        train_df["_station_mean"] = train_df.set_index(["Latitude", "Longitude"]).index.map(
            lambda idx: station_means.get(idx, np.nan)
        )
        train_df["_residual"] = train_df[target] - train_df["_station_mean"]

        y_mean = train_df[target]

        # Stage 1: Static features → station mean
        def make_stage1():
            return XGBRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.5, min_child_weight=20,
                reg_alpha=2.0, reg_lambda=10.0, tree_method="gpu_hist",
                device="cuda", random_state=42
            )

        X_static = train_df[static_feats]
        r2_stage1 = spatial_cv(make_stage1, X_static, y_mean, groups)
        print(f"  Stage 1 (static → target):    R² = {r2_stage1:.4f}")

        # Full model for comparison
        all_feats = get_feature_columns(train_df)
        X_all = train_df[all_feats]
        r2_full = spatial_cv(make_stage1, X_all, y_mean, groups)
        print(f"  Full model (all → target):     R² = {r2_full:.4f}")

        # Combined two-stage via CV
        gkf = GroupKFold(n_splits=5)
        combined_scores = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_static, y_mean, groups)):
            # Stage 1
            m1 = make_stage1()
            X_s_tr = X_static.iloc[tr_idx].fillna(X_static.iloc[tr_idx].median())
            X_s_va = X_static.iloc[va_idx].fillna(X_static.iloc[tr_idx].median())
            m1.fit(X_s_tr, y_mean.iloc[tr_idx])
            stage1_pred = m1.predict(X_s_va)

            # Stage 2: temporal features → residual from stage 1
            residual_tr = y_mean.iloc[tr_idx] - m1.predict(X_s_tr)
            m2 = XGBRegressor(
                n_estimators=200, max_depth=2, learning_rate=0.02,
                subsample=0.6, colsample_bytree=0.3, min_child_weight=50,
                reg_alpha=5.0, reg_lambda=20.0, tree_method="gpu_hist",
                device="cuda", random_state=42
            )
            X_t_tr = train_df[temporal_feats].iloc[tr_idx].fillna(0)
            X_t_va = train_df[temporal_feats].iloc[va_idx].fillna(0)
            m2.fit(X_t_tr, residual_tr)
            stage2_pred = m2.predict(X_t_va)

            combined = stage1_pred + stage2_pred
            score = r2_score(y_mean.iloc[va_idx], combined)
            combined_scores.append(score)

        print(f"  Two-stage combined:            R² = {np.mean(combined_scores):.4f}")

        # Clean up
        train_df.drop(columns=["_station_mean", "_residual"], inplace=True, errors="ignore")


# ============================================================
# Strategy 6: DRP with log transform
# ============================================================
def run_drp_log_transform(train_df, groups):
    """Test log1p transform specifically for DRP."""
    from xgboost import XGBRegressor

    all_features = get_feature_columns(train_df)

    print("\n" + "=" * 80)
    print("STRATEGY 6: LOG TRANSFORM FOR DRP")
    print("=" * 80)

    target = "Dissolved Reactive Phosphorus"
    y_raw = train_df[target]
    y_log = np.log1p(y_raw)

    def make_model():
        return XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.4, min_child_weight=30,
            reg_alpha=2.0, reg_lambda=10.0, tree_method="gpu_hist",
            device="cuda", random_state=42
        )

    X = train_df[all_features]

    # Without log
    r2_raw = spatial_cv(make_model, X, y_raw, groups)
    print(f"  DRP (raw):   R² = {r2_raw:.4f}")

    # With log (evaluate in original scale)
    gkf = GroupKFold(n_splits=5)
    log_scores = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_log, groups)):
        model = make_model()
        X_tr = X.iloc[tr_idx].fillna(X.iloc[tr_idx].median())
        X_va = X.iloc[va_idx].fillna(X.iloc[tr_idx].median())
        model.fit(X_tr, y_log.iloc[tr_idx])
        preds_log = model.predict(X_va)
        preds_raw = np.expm1(preds_log)
        preds_raw = np.maximum(preds_raw, 0)
        score = r2_score(y_raw.iloc[va_idx], preds_raw)
        log_scores.append(score)

    print(f"  DRP (log1p): R² = {np.mean(log_scores):.4f}")


# ============================================================
# Strategy 7: LightGBM and CatBoost comparison
# ============================================================
def run_model_comparison(train_df, groups):
    """Compare XGBoost, LightGBM, CatBoost with matched regularization."""
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    all_features = get_feature_columns(train_df)

    print("\n" + "=" * 80)
    print("STRATEGY 7: MODEL COMPARISON (matched regularization)")
    print("=" * 80)

    model_factories = {
        "XGBoost": lambda: XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.4, min_child_weight=30,
            reg_alpha=2.0, reg_lambda=10.0, tree_method="gpu_hist",
            device="cuda", random_state=42
        ),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.4, min_child_samples=30,
            reg_alpha=2.0, reg_lambda=10.0, random_state=42, verbose=-1
        ),
        "CatBoost": lambda: CatBoostRegressor(
            iterations=400, depth=3, learning_rate=0.03,
            subsample=0.7, rsm=0.4, l2_leaf_reg=10.0,
            random_seed=42, verbose=0, task_type="GPU"
        ),
    }

    results = {}

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")
        for model_name, factory in model_factories.items():
            X = train_df[all_features]
            r2 = spatial_cv(factory, X, y, groups)
            print(f"  {model_name:12s} → R² = {r2:.4f}")
            results[(target, model_name)] = r2

    return results


# ============================================================
# Final: Best configuration submission
# ============================================================
def generate_best_submission(train_df, val_df):
    """Train with best config on full training data and predict validation."""
    from xgboost import XGBRegressor
    import sys
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from submission import create_submission

    all_features = get_feature_columns(train_df)
    # Ensure val has the same features
    val_features = [f for f in all_features if f in val_df.columns]

    X_train = train_df[val_features].fillna(train_df[val_features].median())
    X_val = val_df[val_features].fillna(train_df[val_features].median())

    predictions = {}

    for target in TARGETS:
        if target == "Dissolved Reactive Phosphorus":
            # Use log transform for DRP
            y = np.log1p(train_df[target])
            model = XGBRegressor(
                n_estimators=500, max_depth=3, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.4, min_child_weight=30,
                reg_alpha=2.0, reg_lambda=10.0, tree_method="gpu_hist",
                device="cuda", random_state=42
            )
            model.fit(X_train, y)
            preds = np.expm1(model.predict(X_val))
            preds = np.maximum(preds, 0)
        else:
            y = train_df[target]
            model = XGBRegressor(
                n_estimators=500, max_depth=3, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.4, min_child_weight=30,
                reg_alpha=2.0, reg_lambda=10.0, tree_method="gpu_hist",
                device="cuda", random_state=42
            )
            model.fit(X_train, y)
            preds = model.predict(X_val)
            preds = np.maximum(preds, 0)

        predictions[target] = preds
        print(f"  {target}: mean={preds.mean():.2f}, std={preds.std():.2f}")

    create_submission(val_df, predictions, version="v2_spatial")


if __name__ == "__main__":
    train_df, val_df = load_data()
    groups = assign_spatial_groups(train_df)
    feat_cols = get_feature_columns(train_df)

    print(f"\nDataset: {len(train_df)} rows, {len(feat_cols)} features")
    print(f"Spatial groups: {len(np.unique(groups))} clusters")
    print(f"Targets: {TARGETS}")

    # Run all strategies
    run_feature_ablation(train_df, groups)
    run_regularized_xgb(train_df, groups)
    run_env_knn(train_df, groups)
    run_linear_baseline(train_df, groups)
    run_two_stage(train_df, groups)
    run_drp_log_transform(train_df, groups)
    run_model_comparison(train_df, groups)

    # Generate submission with best approach
    print("\n" + "=" * 80)
    print("GENERATING SUBMISSION")
    print("=" * 80)
    generate_best_submission(train_df, val_df)
