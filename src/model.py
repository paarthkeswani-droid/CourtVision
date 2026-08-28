"""Train, evaluate, explain, and rank CourtVision prospect models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import MODEL_FEATURES


def candidates(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "elastic_net": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", ElasticNet(alpha=0.05, l1_ratio=0.4, max_iter=10000, random_state=random_state))]),
        "random_forest": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.8, random_state=random_state, n_jobs=-1))]),
    }


def percentile_grades(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True, method="average").mul(99).add(1).round(1).to_numpy()


def train_and_evaluate(data: pd.DataFrame, random_state: int = 42):
    missing = sorted(set(MODEL_FEATURES + ["nba_value"]) - set(data.columns))
    if missing:
        raise ValueError(f"Model data is missing columns: {', '.join(missing)}")
    if len(data) < 20:
        raise ValueError("At least 20 rows are required for a stable train/test evaluation")
    X, y = data[MODEL_FEATURES], data["nba_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    folds = min(5, max(2, len(X_train) // 8))
    cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    results = {}
    models = candidates(random_state)
    for name, model in models.items():
        cv_mae = -cross_val_score(model, X_train, y_train, scoring="neg_mean_absolute_error", cv=cv).mean()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        results[name] = {"cv_mae": float(cv_mae), "mae": float(mean_absolute_error(y_test, predicted)), "rmse": float(mean_squared_error(y_test, predicted) ** 0.5), "r2": float(r2_score(y_test, predicted))}
    winner = min(results, key=lambda name: results[name]["cv_mae"])
    best = models[winner].fit(X, y)
    return best, winner, results, (y_test.to_numpy(), models[winner].predict(X_test))


def importance_table(model: Pipeline) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    values = estimator.feature_importances_ if hasattr(estimator, "feature_importances_") else estimator.coef_
    return pd.DataFrame({"feature": MODEL_FEATURES, "importance": values}).assign(abs_importance=lambda x: x.importance.abs()).sort_values("abs_importance", ascending=False).drop(columns="abs_importance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/model_data.csv"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    model, winner, metrics, holdout = train_and_evaluate(data)
    args.outputs.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(data[MODEL_FEATURES])
    rankings = data.copy()
    rankings["predicted_nba_value"] = predictions
    rankings["model_grade"] = percentile_grades(predictions)
    rankings.sort_values("model_grade", ascending=False).to_csv(args.outputs / "prospect_rankings.csv", index=False)
    importance_table(model).to_csv(args.outputs / "feature_importance.csv", index=False)
    (args.outputs / "model_metrics.json").write_text(json.dumps({"selected_model": winner, "models": metrics}, indent=2))
    actual, predicted = holdout
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(actual, predicted, alpha=.7, color="#1f5a50"); lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max()); ax.plot([lo, hi], [lo, hi], "--", color="#ef633c"); ax.set(title=f"CourtVision holdout — {winner}", xlabel="Actual value", ylabel="Predicted value"); fig.tight_layout(); fig.savefig(args.outputs / "predicted_vs_actual.png", dpi=180); plt.close(fig)
    top = rankings.nlargest(15, "model_grade").sort_values("model_grade")
    fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(top["player"], top["model_grade"], color="#1f5a50"); ax.set(xlabel="Prospect grade", xlim=(0, 100), title="Top CourtVision prospects"); fig.tight_layout(); fig.savefig(args.outputs / "top_prospects.png", dpi=180); plt.close(fig)
    print(f"Selected {winner}; wrote rankings and evaluation artifacts to {args.outputs}")


if __name__ == "__main__":
    main()

