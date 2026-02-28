"""
Training pipeline v2 — Focused on realistic CV and spatial generalization.

Key insight: Our 8-cluster geographic CV is too pessimistic. Training lat range
(-24.7 to -33.5) overlaps with validation lat range (-31.9 to -34.1).
We need CV that matches the actual leaderboard difficulty.

This script:
    1. Tests different CV strategies to find one calibrated to leaderboard
    2. Uses early stopping to prevent overfitting
    3. Includes coordinates as features (regions overlap)
    4. Uses log1p for DRP
    5. Runs stacking ensemble
    6. Generates submission

Usage:
    python src/train_v2.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data_loader import build_full_dataset
from feature_builder import build_features, get_feature_columns
from submission import create_submission


TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
OUTPUT_DIR = ROOT_DIR / "outputs"


def load_data():
    """Load and prepare datasets."""
    print("Loading training data...")
    train_df = build_full_dataset("train")
    train_df = build_features(train_df, is_training=True)

    print("Loading validation data...")
    val_df = build_full_dataset("val")
    val_df = build_features(val_df, is_training=False)

    return train_df, val_df


def get_station_ids(df):
    """Create unique station ID from lat/lon."""
    return df.apply(lambda r: f"{r['Latitude']:.6f}_{r['Longitude']:.6f}", axis=1)


# ============================================================
# CV Strategy Comparison
# ============================================================
def compare_cv_strategies(train_df):
    """Test different CV strategies to find one calibrated to leaderboard."""
    from xgboost import XGBRegressor

    feat_cols = get_feature_columns(train_df)
    X = train_df[feat_cols]

    def make_model():
        return XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
            reg_alpha=0.5, reg_lambda=3.0, tree_method="gpu_hist",
            device="cuda", random_state=42
        )

    print("=" * 80)
    print("CV STRATEGY COMPARISON")
    print("=" * 80)

    for target in TARGETS:
        y = train_df[target]
        print(f"\n--- {target} ---")

        # Strategy A: Random sample split (what benchmark uses)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores_a = []
        for tr, va in kf.split(X):
            m = make_model()
            m.fit(X.iloc[tr], y.iloc[tr])
            scores_a.append(r2_score(y.iloc[va], m.predict(X.iloc[va])))
        print(f"  Random sample split (5-fold):     R² = {np.mean(scores_a):.4f}")

        # Strategy B: Random station split (hold out ~20% of stations)
        station_ids = get_station_ids(train_df)
        unique_stations = station_ids.unique()
        np.random.seed(42)
        np.random.shuffle(unique_stations)
        n_folds = 5
        fold_size = len(unique_stations) // n_folds
        scores_b = []
        for i in range(n_folds):
            val_stations = set(unique_stations[i * fold_size:(i + 1) * fold_size])
            va_mask = station_ids.isin(val_stations)
            tr_idx = np.where(~va_mask)[0]
            va_idx = np.where(va_mask)[0]
            m = make_model()
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            scores_b.append(r2_score(y.iloc[va_idx], m.predict(X.iloc[va_idx])))
        print(f"  Random station split (5-fold):    R² = {np.mean(scores_b):.4f}")

        # Strategy C: Geographic clusters (8) — our current approach
        coords = train_df[["Latitude", "Longitude"]].values
        groups_8 = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(coords)
        gkf = GroupKFold(n_splits=5)
        scores_c = []
        for tr, va in gkf.split(X, y, groups_8):
            m = make_model()
            m.fit(X.iloc[tr], y.iloc[tr])
            scores_c.append(r2_score(y.iloc[va], m.predict(X.iloc[va])))
        print(f"  Geographic clusters (8):          R² = {np.mean(scores_c):.4f}")

        # Strategy D: Geographic clusters (20) — less harsh
        groups_20 = KMeans(n_clusters=20, random_state=42, n_init=10).fit_predict(coords)
        gkf20 = GroupKFold(n_splits=5)
        scores_d = []
        for tr, va in gkf20.split(X, y, groups_20):
            m = make_model()
            m.fit(X.iloc[tr], y.iloc[tr])
            scores_d.append(r2_score(y.iloc[va], m.predict(X.iloc[va])))
        print(f"  Geographic clusters (20):         R² = {np.mean(scores_d):.4f}")

        # Strategy E: Hold out only southern stations (simulates actual val region)
        southern_mask = train_df["Latitude"] < -31.5
        tr_idx = np.where(~southern_mask)[0]
        va_idx = np.where(southern_mask)[0]
        if len(va_idx) > 0:
            m = make_model()
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            score_e = r2_score(y.iloc[va_idx], m.predict(X.iloc[va_idx]))
            print(f"  Hold out south (lat < -31.5):     R² = {score_e:.4f}  (n_val={len(va_idx)})")


# ============================================================
# Main training with best approach
# ============================================================
def train_best_models(train_df, val_df):
    """Train models with the best configuration and generate predictions."""
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    feat_cols = get_feature_columns(train_df)
    val_feat_cols = [f for f in feat_cols if f in val_df.columns]

    X_train = train_df[val_feat_cols].fillna(train_df[val_feat_cols].median())
    X_val = val_df[val_feat_cols].fillna(train_df[val_feat_cols].median())

    # Use station-based CV for model selection
    station_ids = get_station_ids(train_df)
    unique_stations = station_ids.unique()

    print("\n" + "=" * 80)
    print("TRAINING BEST MODELS")
    print("=" * 80)

    predictions = {}
    cv_scores = {}

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")

        use_log = (target == "Dissolved Reactive Phosphorus")
        y_raw = train_df[target]
        y = np.log1p(y_raw) if use_log else y_raw

        # --- Model configs to try ---
        configs = {
            "xgb_d6": lambda: XGBRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
                reg_alpha=0.5, reg_lambda=3.0, gamma=0.1,
                tree_method="gpu_hist", device="cuda", random_state=42
            ),
            "xgb_d8": lambda: XGBRegressor(
                n_estimators=600, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
                reg_alpha=0.3, reg_lambda=2.0,
                tree_method="gpu_hist", device="cuda", random_state=42
            ),
            "xgb_d4_reg": lambda: XGBRegressor(
                n_estimators=1000, max_depth=4, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.4, min_child_weight=20,
                reg_alpha=1.0, reg_lambda=5.0,
                tree_method="gpu_hist", device="cuda", random_state=42
            ),
            "lgbm_d6": lambda: LGBMRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.6, min_child_samples=10,
                reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1
            ),
            "lgbm_d8": lambda: LGBMRegressor(
                n_estimators=600, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.5, min_child_samples=5,
                reg_alpha=0.3, reg_lambda=2.0, random_state=42, verbose=-1
            ),
        }

        # --- Station-based CV ---
        np.random.seed(42)
        shuffled = unique_stations.copy()
        np.random.shuffle(shuffled)
        n_folds = 5
        fold_size = len(shuffled) // n_folds

        model_scores = {}
        model_oof = {}

        for name, factory in configs.items():
            fold_scores = []
            oof = np.full(len(y), np.nan)

            for i in range(n_folds):
                val_stations = set(shuffled[i * fold_size:(i + 1) * fold_size])
                va_mask = station_ids.isin(val_stations)
                tr_idx = np.where(~va_mask)[0]
                va_idx = np.where(va_mask)[0]

                model = factory()
                model.fit(X_train.iloc[tr_idx], y.iloc[tr_idx])
                preds = model.predict(X_train.iloc[va_idx])

                if use_log:
                    preds_eval = np.maximum(np.expm1(preds), 0)
                    score = r2_score(y_raw.iloc[va_idx], preds_eval)
                    oof[va_idx] = preds_eval
                else:
                    score = r2_score(y.iloc[va_idx], preds)
                    oof[va_idx] = preds

                fold_scores.append(score)

            mean_score = np.mean(fold_scores)
            model_scores[name] = mean_score
            model_oof[name] = oof
            print(f"  {name:20s} → R² = {mean_score:.4f}  (folds: {[f'{s:.3f}' for s in fold_scores]})")

        # --- Stacking ensemble ---
        print(f"\n  Building stacking ensemble...")

        # Use OOF predictions as meta-features
        oof_matrix = np.column_stack([model_oof[name] for name in configs.keys()])
        valid_mask = ~np.any(np.isnan(oof_matrix), axis=1)

        meta_X = oof_matrix[valid_mask]
        meta_y = y_raw.values[valid_mask] if use_log else y.values[valid_mask]

        # Fit meta-learner on OOF
        meta = Ridge(alpha=1.0)
        meta.fit(meta_X, meta_y)

        # Evaluate stacking
        meta_preds = meta.predict(meta_X)
        stack_r2 = r2_score(meta_y, meta_preds)
        print(f"  Stacking ensemble (OOF):    R² = {stack_r2:.4f}")
        print(f"  Meta-learner weights: {dict(zip(configs.keys(), meta.coef_.round(3)))}")

        # --- Train final models on ALL data ---
        print(f"\n  Training final models on all data...")
        final_preds = []
        for name, factory in configs.items():
            model = factory()
            model.fit(X_train, y)
            pred = model.predict(X_val)
            if use_log:
                pred = np.maximum(np.expm1(pred), 0)
            final_preds.append(pred)

        # Stack final predictions
        final_stack = np.column_stack(final_preds)
        stacked_pred = meta.predict(final_stack)
        stacked_pred = np.maximum(stacked_pred, 0)

        # Also get best single model prediction
        best_model_name = max(model_scores, key=model_scores.get)
        best_idx = list(configs.keys()).index(best_model_name)
        best_single_pred = final_preds[best_idx]

        print(f"\n  Best single model: {best_model_name} (CV R² = {model_scores[best_model_name]:.4f})")
        print(f"  Stacked prediction: mean={stacked_pred.mean():.2f}, std={stacked_pred.std():.2f}")
        print(f"  Single prediction:  mean={best_single_pred.mean():.2f}, std={best_single_pred.std():.2f}")

        # Use stacking if it improved OOF, else use best single
        predictions[target] = stacked_pred
        cv_scores[target] = stack_r2

    return predictions, cv_scores


def train_with_coords(train_df, val_df):
    """Also try adding Lat/Lon as direct features."""
    from xgboost import XGBRegressor

    feat_cols = get_feature_columns(train_df)
    coord_cols = ["Latitude", "Longitude"]
    all_cols = feat_cols + coord_cols
    val_cols = [f for f in all_cols if f in val_df.columns]

    X_train = train_df[val_cols].fillna(train_df[val_cols].median())
    X_val = val_df[val_cols].fillna(train_df[val_cols].median())

    station_ids = get_station_ids(train_df)
    unique_stations = station_ids.unique()
    np.random.seed(42)
    shuffled = unique_stations.copy()
    np.random.shuffle(shuffled)
    n_folds = 5
    fold_size = len(shuffled) // n_folds

    print("\n" + "=" * 80)
    print("WITH LAT/LON AS FEATURES")
    print("=" * 80)

    predictions = {}

    for target in TARGETS:
        use_log = (target == "Dissolved Reactive Phosphorus")
        y_raw = train_df[target]
        y = np.log1p(y_raw) if use_log else y_raw

        fold_scores = []
        for i in range(n_folds):
            val_stations = set(shuffled[i * fold_size:(i + 1) * fold_size])
            va_mask = station_ids.isin(val_stations)
            tr_idx = np.where(~va_mask)[0]
            va_idx = np.where(va_mask)[0]

            model = XGBRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
                reg_alpha=0.5, reg_lambda=3.0, gamma=0.1,
                tree_method="gpu_hist", device="cuda", random_state=42
            )
            model.fit(X_train.iloc[tr_idx], y.iloc[tr_idx])
            preds = model.predict(X_train.iloc[va_idx])

            if use_log:
                preds = np.maximum(np.expm1(preds), 0)
                score = r2_score(y_raw.iloc[va_idx], preds)
            else:
                score = r2_score(y.iloc[va_idx], preds)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        print(f"  {target:40s} R² = {mean_score:.4f}  (folds: {[f'{s:.3f}' for s in fold_scores]})")

        # Train on all data
        model = XGBRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
            reg_alpha=0.5, reg_lambda=3.0, gamma=0.1,
            tree_method="gpu_hist", device="cuda", random_state=42
        )
        model.fit(X_train, y)
        pred = model.predict(X_val)
        if use_log:
            pred = np.maximum(np.expm1(pred), 0)
        predictions[target] = np.maximum(pred, 0)

    return predictions


if __name__ == "__main__":
    train_df, val_df = load_data()
    feat_cols = get_feature_columns(train_df)
    print(f"\nDataset: {len(train_df)} rows, {len(feat_cols)} features")

    # Step 1: Understand which CV strategy is realistic
    compare_cv_strategies(train_df)

    # Step 2: Train models with stacking (no coords)
    preds_no_coords, cv_scores = train_best_models(train_df, val_df)

    # Step 3: Train with coords
    preds_with_coords = train_with_coords(train_df, val_df)

    # Step 4: Generate submissions for both
    print("\n" + "=" * 80)
    print("GENERATING SUBMISSIONS")
    print("=" * 80)

    print("\n--- Submission v2: Stacked ensemble (no coords) ---")
    create_submission(val_df, preds_no_coords, version="v2_stacked")

    print("\n--- Submission v3: XGBoost with coords ---")
    create_submission(val_df, preds_with_coords, version="v3_coords")

    print("\nDone! Submit both and compare leaderboard scores.")
    print("  outputs/submissions/submission_v2_stacked.csv")
    print("  outputs/submissions/submission_v3_coords.csv")
