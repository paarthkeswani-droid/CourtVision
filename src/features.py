"""Feature engineering for CourtVision prospect data."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLLEGE_COLUMNS = {
    "player", "season", "age", "height_in", "weight_lb", "games", "minutes",
    "pts", "fga", "fta", "three_pa", "three_p", "ft", "ast", "tov",
    "orb", "drb", "stl", "blk",
}

MODEL_FEATURES = [
    "age", "height_in", "weight_lb", "ts_pct", "ast_tov",
    "pts_per40", "ast_per40", "reb_per40", "stl_per40", "blk_per40",
    "three_rate", "ft_rate",
]
FEATURES = MODEL_FEATURES


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while replacing invalid and zero-denominator results with zero."""
    denominator = denominator.replace(0, np.nan)
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def validate_college_data(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLLEGE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"College data is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("College data contains no rows")
    if (pd.to_numeric(frame["minutes"], errors="coerce").fillna(0) < 0).any():
        raise ValueError("minutes cannot be negative")


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized efficiency and per-40 prospect features."""
    validate_college_data(frame)
    data = frame.copy()
    numeric = sorted(REQUIRED_COLLEGE_COLUMNS - {"player", "season"})
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce").fillna(0)

    data["ts_pct"] = safe_divide(data["pts"], 2 * (data["fga"] + 0.44 * data["fta"]))
    data["ast_tov"] = safe_divide(data["ast"], data["tov"])
    data["reb"] = data["orb"] + data["drb"]
    for stat in ("pts", "ast", "reb", "stl", "blk"):
        data[f"{stat}_per40"] = safe_divide(data[stat] * 40, data["minutes"])
    data["three_rate"] = safe_divide(data["three_pa"], data["fga"])
    data["ft_rate"] = safe_divide(data["fta"], data["fga"])
    return data


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible public entry point for feature engineering."""
    return engineer_features(frame)


def early_career_value(outcomes: pd.DataFrame) -> pd.Series:
    """Transparent outcome combining availability and impact on robust scales."""
    required = {"nba_minutes", "nba_ws", "nba_bpm"}
    missing = sorted(required.difference(outcomes.columns))
    if missing:
        raise ValueError(f"NBA outcomes are missing required columns: {', '.join(missing)}")
    minutes = np.log1p(pd.to_numeric(outcomes["nba_minutes"], errors="coerce").clip(lower=0).fillna(0))
    ws = pd.to_numeric(outcomes["nba_ws"], errors="coerce").fillna(0)
    bpm = pd.to_numeric(outcomes["nba_bpm"], errors="coerce").fillna(0)
    return 0.35 * minutes + 0.40 * ws + 0.25 * bpm

