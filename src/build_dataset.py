"""Build the model-ready historical CourtVision dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import pandas as pd

from features import MODEL_FEATURES, early_career_value, engineer_features


def build_dataset(college_path: Path, outcomes_path: Path) -> pd.DataFrame:
    college = engineer_features(pd.read_csv(college_path))
    outcomes = pd.read_csv(outcomes_path)
    if "player" not in outcomes:
        raise ValueError("NBA outcomes are missing required column: player")
    outcomes = outcomes.copy()
    outcomes["nba_value"] = early_career_value(outcomes)
    merged = college.merge(outcomes, on="player", how="inner", validate="many_to_one")
    if merged.empty:
        raise ValueError("No player names matched between college and NBA outcome files")
    columns = [c for c in ["player", "season", "school", "conference", "draft_pick", "class_year"] if c in merged]
    return merged[columns + MODEL_FEATURES + ["nba_value"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--college", type=Path, default=Path("data/raw/college_players.csv"))
    parser.add_argument("--outcomes", type=Path, default=Path("data/raw/nba_outcomes.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/model_data.csv"))
    parser.add_argument("--database", type=Path, default=Path("data/processed/courtvision.sqlite"))
    args = parser.parse_args()
    dataset = build_dataset(args.college, args.outcomes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    with sqlite3.connect(args.database) as connection:
        dataset.to_sql("prospects", connection, if_exists="replace", index=False)
    print(f"Wrote {len(dataset):,} matched prospect seasons to {args.output}")


if __name__ == "__main__":
    main()

