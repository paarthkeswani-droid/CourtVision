"""Find statistical historical comparisons for a CourtVision prospect."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "model_data.csv"
OUT = ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(DATA).reset_index(drop=True)
    matches = df.index[df["player"].str.lower() == args.player.lower()].tolist()
    if not matches:
        raise ValueError(f"Player not found: {args.player}")
    idx = matches[0]
    X = SimpleImputer(strategy="median").fit_transform(df[FEATURES])
    X = StandardScaler().fit_transform(X)
    sims = cosine_similarity(X[idx:idx+1], X)[0]
    order = sims.argsort()[::-1]
    order = [i for i in order if i != idx][:args.top]
    comps = df.loc[order, ["player"] + [c for c in ["season", "school", "nba_value"] if c in df]].copy()
    comps["similarity"] = [round(float(sims[i]), 3) for i in order]
    OUT.mkdir(exist_ok=True)
    comps.to_csv(OUT / "similar_players.csv", index=False)
    print(comps.to_string(index=False))

if __name__ == "__main__":
    main()
