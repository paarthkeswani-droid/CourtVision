"""Find historical statistical comparisons for a selected prospect."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from features import MODEL_FEATURES


def find_similar_players(data: pd.DataFrame, player: str, top_n: int = 10) -> pd.DataFrame:
    matches = data.index[data["player"].str.casefold() == player.casefold()].tolist()
    if not matches:
        raise ValueError(f"Player not found: {player}")
    matrix = SimpleImputer(strategy="median").fit_transform(data[MODEL_FEATURES])
    matrix = StandardScaler().fit_transform(matrix)
    idx = matches[-1]
    scores = cosine_similarity(matrix[[idx]], matrix).ravel()
    result = data[[c for c in ["player", "season", "school", "draft_pick", "nba_value"] if c in data]].copy()
    result["similarity"] = scores
    result = result.drop(index=idx).sort_values("similarity", ascending=False).head(top_n)
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/model_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/similar_players.csv"))
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    result = find_similar_players(pd.read_csv(args.data), args.player, args.top)
    args.output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

