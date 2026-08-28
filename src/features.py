"""Feature engineering for CourtVision."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "age", "height_in", "weight_lb", "ts_pct", "ast_tov",
    "pts_per40", "ast_per40", "reb_per40", "stl_per40", "blk_per40",
    "three_rate", "ft_rate",
]


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Vectorized division that returns NaN for zero denominators."""
    return num.astype(float).div(den.replace(0, np.nan).astype(float))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create pace-resistant and efficiency-oriented prospect features."""
    out = df.copy()
    minutes = out["minutes"]
    out["ts_pct"] = safe_div(out["pts"], 2 * (out["fga"] + 0.44 * out["fta"]))
    out["ast_tov"] = safe_div(out["ast"], out["tov"])
    out["pts_per40"] = safe_div(out["pts"] * 40, minutes)
    out["ast_per40"] = safe_div(out["ast"] * 40, minutes)
    out["reb_per40"] = safe_div((out["orb"] + out["drb"]) * 40, minutes)
    out["stl_per40"] = safe_div(out["stl"] * 40, minutes)
    out["blk_per40"] = safe_div(out["blk"] * 40, minutes)
    out["three_rate"] = safe_div(out["three_pa"], out["fga"])
    out["ft_rate"] = safe_div(out["fta"], out["fga"])
    return out
