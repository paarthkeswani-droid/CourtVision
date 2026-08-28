import pathlib
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from features import engineer_features, early_career_value, safe_divide
from similarities import find_similar_players


def sample():
    return pd.DataFrame([{ "player":"A", "season":2025, "age":19, "height_in":78, "weight_lb":205, "games":30, "minutes":1000, "pts":600, "fga":400, "fta":150, "three_pa":180, "three_p":70, "ft":120, "ast":150, "tov":75, "orb":35, "drb":160, "stl":45, "blk":20 }])


def test_feature_formulas():
    row = engineer_features(sample()).iloc[0]
    assert row.ast_tov == pytest.approx(2)
    assert row.pts_per40 == pytest.approx(24)
    assert row.reb_per40 == pytest.approx(7.8)
    assert row.ts_pct == pytest.approx(600 / (2 * (400 + .44 * 150)))


def test_zero_denominators_are_finite():
    data = sample(); data.loc[0, ["minutes", "fga", "tov"]] = 0
    engineered = engineer_features(data)
    assert np.isfinite(engineered.select_dtypes("number")).all().all()
    assert engineered.loc[0, "pts_per40"] == 0


def test_missing_columns_raise_helpful_error():
    with pytest.raises(ValueError, match="missing required columns"):
        engineer_features(pd.DataFrame({"player": ["A"]}))


def test_similarity_excludes_selected_player():
    rows=[]
    for i in range(4):
        row=sample().iloc[0].to_dict(); row["player"]=chr(65+i); row["age"]+=i; row["pts"]+=i*20; rows.append(row)
    engineered=engineer_features(pd.DataFrame(rows)); result=find_similar_players(engineered,"A",2)
    assert len(result)==2 and "A" not in result.player.tolist()


def test_early_value_rejects_incomplete_outcomes():
    with pytest.raises(ValueError, match="missing required columns"):
        early_career_value(pd.DataFrame({"nba_minutes": [10]}))

