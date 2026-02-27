"""
Evaluation utilities: spatial CV, per-target metrics, SHAP analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold


def compute_metrics(y_true, y_pred):
    """Compute R², RMSE, and MAE."""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def spatial_cv_report(model_fn, X, y, groups, n_splits=5):
    """
    Run spatial CV and return a detailed report DataFrame.
    """
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    records = []
    oof_preds = np.full(len(y), np.nan)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        train_preds = model.predict(X.iloc[train_idx])
        val_preds = model.predict(X.iloc[val_idx])
        oof_preds[val_idx] = val_preds

        records.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "train_r2": r2_score(y.iloc[train_idx], train_preds),
            "val_r2": r2_score(y.iloc[val_idx], val_preds),
            "val_rmse": np.sqrt(mean_squared_error(y.iloc[val_idx], val_preds)),
        })

    report = pd.DataFrame(records)

    valid_mask = ~np.isnan(oof_preds)
    overall = {
        "fold": "OVERALL",
        "n_train": "-",
        "n_val": valid_mask.sum(),
        "train_r2": "-",
        "val_r2": r2_score(y[valid_mask], oof_preds[valid_mask]),
        "val_rmse": np.sqrt(mean_squared_error(y[valid_mask], oof_preds[valid_mask])),
    }
    report = pd.concat([report, pd.DataFrame([overall])], ignore_index=True)

    return report, oof_preds


def feature_importance_shap(model, X, target_name="target", max_display=20):
    """
    Compute SHAP values and display feature importance.
    Returns SHAP values array and feature importance DataFrame.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None, None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\nTop {max_display} features for {target_name}:")
    print(importance.head(max_display).to_string(index=False))

    return shap_values, importance


def residual_analysis(y_true, y_pred, lat, lon):
    """
    Analyze prediction residuals spatially.
    Returns a DataFrame with residuals per location for diagnosis.
    """
    residuals = y_true - y_pred
    df = pd.DataFrame({
        "Latitude": lat,
        "Longitude": lon,
        "actual": y_true,
        "predicted": y_pred,
        "residual": residuals,
        "abs_residual": np.abs(residuals),
        "pct_error": np.abs(residuals) / (np.abs(y_true) + 1e-10) * 100,
    })

    # Summarize by location
    location_summary = df.groupby(["Latitude", "Longitude"]).agg(
        mean_residual=("residual", "mean"),
        mean_abs_residual=("abs_residual", "mean"),
        std_residual=("residual", "std"),
        n_samples=("residual", "count"),
    ).reset_index()

    return df, location_summary
