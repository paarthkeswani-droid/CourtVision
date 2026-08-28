"""Train, evaluate and explain CourtVision prospect models."""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_data.csv"
OUT = ROOT / "outputs"


def models() -> dict:
    return {
        "elastic_net": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.08, l1_ratio=0.35, max_iter=20000)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=4, random_state=42, n_jobs=-1)),
        ]),
    }


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["nba_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    candidates = models()
    cv_mae = {}
    for name, estimator in candidates.items():
        scores = -cross_val_score(estimator, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        cv_mae[name] = float(scores.mean())

    best_name = min(cv_mae, key=cv_mae.get)
    best = candidates[best_name].fit(X_train, y_train)
    pred = best.predict(X_test)
    metrics = {
        "selected_model": best_name,
        "cv_mae": cv_mae,
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "test_r2": float(r2_score(y_test, pred)),
        "n_rows": int(len(df)),
    }

    # Refit on all historical rows for ranking/scouting output after evaluation.
    best.fit(X, y)
    df["predicted_nba_value"] = best.predict(X)
    df["model_grade"] = (df["predicted_nba_value"].rank(pct=True) * 100).round(1)
    df = df.sort_values("predicted_nba_value", ascending=False)

    OUT.mkdir(exist_ok=True)
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    df.to_csv(OUT / "prospect_rankings.csv", index=False)
    joblib.dump(best, OUT / "courtvision_model.joblib")

    fitted = best.named_steps["model"]
    values = fitted.feature_importances_ if hasattr(fitted, "feature_importances_") else np.abs(fitted.coef_)
    pd.DataFrame({"feature": FEATURES, "importance": values}).sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, pred, alpha=0.7)
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual early-career NBA value")
    plt.ylabel("Predicted value")
    plt.title(f"CourtVision holdout evaluation — {best_name}")
    plt.tight_layout(); plt.savefig(OUT / "predicted_vs_actual.png", dpi=180); plt.close()

    top = df.head(15).sort_values("model_grade")
    plt.figure(figsize=(9, 7)); plt.barh(top["player"], top["model_grade"])
    plt.xlabel("CourtVision grade (percentile)"); plt.title("Top statistical prospect profiles")
    plt.tight_layout(); plt.savefig(OUT / "top_prospects.png", dpi=180); plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
