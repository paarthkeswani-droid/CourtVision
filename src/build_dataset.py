"""Build the historical CourtVision modeling table from public-data CSV inputs."""
from pathlib import Path
import pandas as pd
from features import add_features, FEATURES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def normalize_name(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def main() -> None:
    college_path = RAW / "college_players.csv"
    nba_path = RAW / "nba_outcomes.csv"
    if not college_path.exists() or not nba_path.exists():
        raise FileNotFoundError(
            "Add data/raw/college_players.csv and data/raw/nba_outcomes.csv. "
            "See README.md for the required schemas."
        )

    college = pd.read_csv(college_path)
    nba = pd.read_csv(nba_path)
    college["player_key"] = normalize_name(college["player"])
    nba["player_key"] = normalize_name(nba["player"])

    college = add_features(college)
    # Transparent early-career outcome index; weights are configurable.
    nba["nba_value"] = (
        0.002 * nba["nba_minutes"].fillna(0)
        + 1.5 * nba["nba_ws"].fillna(0)
        + 0.75 * nba["nba_bpm"].fillna(0)
    )
    keep_nba = ["player_key", "nba_value", "nba_minutes", "nba_ws", "nba_bpm"]
    model = college.merge(nba[keep_nba], on="player_key", how="inner")
    cols = [c for c in ["player", "season", "school", "conference", "draft_pick"] if c in model] + FEATURES + ["nba_value"]
    model = model[cols].replace([float("inf"), float("-inf")], pd.NA)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    model.to_csv(PROCESSED / "model_data.csv", index=False)
    print(f"Wrote {len(model):,} matched player rows to data/processed/model_data.csv")


if __name__ == "__main__":
    main()
