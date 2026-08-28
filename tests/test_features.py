import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from features import add_features


def test_feature_math():
    df = pd.DataFrame([{
        "age": 20, "height_in": 78, "weight_lb": 205, "minutes": 100,
        "pts": 60, "fga": 40, "fta": 20, "three_pa": 10, "three_p": 4,
        "ft": 16, "ast": 20, "tov": 10, "orb": 5, "drb": 20,
        "stl": 5, "blk": 2,
    }])
    out = add_features(df)
    assert round(out.loc[0, "pts_per40"], 2) == 24.0
    assert round(out.loc[0, "ast_tov"], 2) == 2.0
    assert round(out.loc[0, "reb_per40"], 2) == 10.0
