"""
Model training pipeline with spatial cross-validation.
Supports XGBoost (GPU), LightGBM, CatBoost, and ensemble stacking.

Usage:
    python src/model_trainer.py --model xgboost --target all
    python src/model_trainer.py --model lightgbm --target "Total Alkalinity"
    python src/model_trainer.py --model ensemble --target all
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "datasets"
MODEL_DIR = ROOT_DIR / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = [
    "Total Alkalinity",
    "Electrical Conductance",
    "Dissolved Reactive Phosphorus",
]

ID_COLS = ["Latitude", "Longitude", "Sample Date"]


def assign_spatial_groups(df, n_clusters=8):
    """
    Assign each row to a spatial cluster based on station coordinates.
    Used for spatial cross-validation (GroupKFold).
    """
    coords = df[["Latitude", "Longitude"]].values
    unique_coords = np.unique(coords, axis=0)

    km = KMeans(n_clusters=min(n_clusters, len(unique_coords)), random_state=42, n_init=10)
    km.fit(unique_coords)

    # Map each row's coordinates to its cluster
    from scipy.spatial import cKDTree
    tree = cKDTree(unique_coords)
    _, idx = tree.query(coords)
    groups = km.labels_[idx]

    return groups


def spatial_cv_evaluate(model_fn, X, y, groups, n_splits=5):
    """
    Evaluate a model using spatial GroupKFold cross-validation.
    Returns per-fold R² scores and the overall mean.
    """
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    fold_scores = []
    oof_preds = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn()
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        r2 = r2_score(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_scores.append({"fold": fold, "r2": r2, "rmse": rmse, "n_val": len(val_idx)})
        print(f"  Fold {fold}: R²={r2:.4f}, RMSE={rmse:.2f} (n={len(val_idx)})")

    overall_r2 = r2_score(y[~np.isnan(oof_preds)], oof_preds[~np.isnan(oof_preds)])
    print(f"  Overall OOF R²: {overall_r2:.4f}")

    return fold_scores, oof_preds, overall_r2


def get_xgboost_model(use_gpu=True):
    """Return an XGBoost model factory function."""
    import xgboost as xgb

    def factory():
        params = {
            "n_estimators": 1000,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
        if use_gpu:
            params["tree_method"] = "gpu_hist"
            params["device"] = "cuda"
        return xgb.XGBRegressor(**params)

    return factory


def get_lightgbm_model():
    """Return a LightGBM model factory function."""
    import lightgbm as lgb

    def factory():
        return lgb.LGBMRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    return factory


def get_catboost_model(use_gpu=True):
    """Return a CatBoost model factory function."""
    from catboost import CatBoostRegressor

    def factory():
        params = {
            "iterations": 1000,
            "depth": 8,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 0,
        }
        if use_gpu:
            params["task_type"] = "GPU"
        return CatBoostRegressor(**params)

    return factory


def get_rf_model():
    """Return a Random Forest model factory function (baseline)."""
    from sklearn.ensemble import RandomForestRegressor

    def factory():
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    return factory


def train_single_target(X, y, groups, target_name, model_type="xgboost", use_gpu=True):
    """
    Train and evaluate a model for a single target variable.
    Returns the trained model, CV scores, and out-of-fold predictions.
    """
    print(f"\n{'='*60}")
    print(f"Target: {target_name}")
    print(f"Model: {model_type}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"{'='*60}")

    # Select model
    if model_type == "xgboost":
        model_fn = get_xgboost_model(use_gpu)
    elif model_type == "lightgbm":
        model_fn = get_lightgbm_model()
    elif model_type == "catboost":
        model_fn = get_catboost_model(use_gpu)
    elif model_type == "rf":
        model_fn = get_rf_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Spatial CV evaluation
    fold_scores, oof_preds, overall_r2 = spatial_cv_evaluate(model_fn, X, y, groups)

    # Train final model on ALL data
    print(f"\nTraining final model on all {len(X)} samples...")
    final_model = model_fn()
    final_model.fit(X, y)

    # Save model
    model_path = MODEL_DIR / f"{target_name.replace(' ', '_').lower()}_{model_type}.joblib"
    joblib.dump(final_model, model_path)
    print(f"Model saved: {model_path}")

    return final_model, fold_scores, oof_preds, overall_r2


def train_ensemble(X, y, groups, target_name, use_gpu=True):
    """
    Train a stacking ensemble of XGBoost + LightGBM + CatBoost.
    Uses Ridge regression as the meta-learner.
    """
    print(f"\n{'='*60}")
    print(f"ENSEMBLE: {target_name}")
    print(f"{'='*60}")

    model_types = ["xgboost", "lightgbm", "catboost"]
    model_fns = {
        "xgboost": get_xgboost_model(use_gpu),
        "lightgbm": get_lightgbm_model(),
        "catboost": get_catboost_model(use_gpu),
    }

    gkf = GroupKFold(n_splits=5)
    oof_matrix = np.zeros((len(y), len(model_types)))
    final_models = {}

    for m_idx, m_name in enumerate(model_types):
        print(f"\n--- {m_name} ---")
        model_fn = model_fns[m_name]

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            model = model_fn()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_matrix[val_idx, m_idx] = model.predict(X.iloc[val_idx])

        r2 = r2_score(y, oof_matrix[:, m_idx])
        print(f"  OOF R²: {r2:.4f}")

        # Train final model on all data
        final_model = model_fn()
        final_model.fit(X, y)
        final_models[m_name] = final_model

    # Meta-learner
    print(f"\n--- Meta-learner (Ridge) ---")
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(oof_matrix, y)

    ensemble_preds = meta_model.predict(oof_matrix)
    ensemble_r2 = r2_score(y, ensemble_preds)
    print(f"  Ensemble OOF R²: {ensemble_r2:.4f}")

    # Save everything
    ensemble_path = MODEL_DIR / f"{target_name.replace(' ', '_').lower()}_ensemble.joblib"
    joblib.dump({
        "models": final_models,
        "meta": meta_model,
    }, ensemble_path)
    print(f"Ensemble saved: {ensemble_path}")

    return final_models, meta_model, ensemble_r2


def predict_ensemble(ensemble_dict, X):
    """Generate predictions from a saved ensemble."""
    base_preds = np.column_stack([
        ensemble_dict["models"][name].predict(X)
        for name in ["xgboost", "lightgbm", "catboost"]
    ])
    return ensemble_dict["meta"].predict(base_preds)


def run_full_pipeline(model_type="xgboost", targets=None, use_gpu=True):
    """
    Run the complete training pipeline:
    1. Load and build features
    2. Train models for each target
    3. Generate predictions on validation set
    4. Save submission CSV
    """
    import sys
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from data_loader import build_full_dataset, TARGET_COLS
    from feature_builder import build_features, get_feature_columns

    if targets is None:
        targets = TARGET_COLS

    # Load and build features
    print("Loading training data...")
    train_df = build_full_dataset("train")
    train_df = build_features(train_df, is_training=True)

    print("Loading validation data...")
    val_df = build_full_dataset("val")
    val_df = build_features(val_df, is_training=False)

    # Get feature columns
    feature_cols = get_feature_columns(train_df)
    print(f"\nUsing {len(feature_cols)} features:")
    for c in feature_cols:
        print(f"  {c}")

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    # Assign spatial groups for CV
    groups = assign_spatial_groups(train_df)

    # Train per-target models
    results = {}
    val_predictions = {}

    for target in targets:
        y = train_df[target]

        if model_type == "ensemble":
            models, meta, r2 = train_ensemble(X_train, y, groups, target, use_gpu)
            # Predict validation
            ensemble_dict = {"models": models, "meta": meta}
            val_predictions[target] = predict_ensemble(ensemble_dict, X_val)
        else:
            model, scores, oof, r2 = train_single_target(
                X_train, y, groups, target, model_type, use_gpu
            )
            val_predictions[target] = model.predict(X_val)

        results[target] = r2

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for target, r2 in results.items():
        print(f"  {target}: R² = {r2:.4f}")
    mean_r2 = np.mean(list(results.values()))
    print(f"\n  Mean R²: {mean_r2:.4f}")

    # Generate submission
    from submission import create_submission
    submission_df = create_submission(val_df, val_predictions)

    return results, submission_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train water quality models")
    parser.add_argument("--model", choices=["xgboost", "lightgbm", "catboost", "rf", "ensemble"],
                        default="xgboost")
    parser.add_argument("--target", default="all",
                        help="Target variable or 'all'")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    args = parser.parse_args()

    targets = TARGET_COLS if args.target == "all" else [args.target]
    use_gpu = not args.no_gpu

    run_full_pipeline(model_type=args.model, targets=targets, use_gpu=use_gpu)
